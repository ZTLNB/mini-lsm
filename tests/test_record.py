"""WAL 记录编解码测试。

这一层是地基 —— 磁盘格式错了,上面所有东西都要返工。
所以测试覆盖得比较细:边界值、损坏检测、截断定位。
"""

import io
import sys
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mini_lsm.errors import (  # noqa: E402
    ChecksumMismatchError,
    TruncatedRecordError,
)
from mini_lsm.record import (  # noqa: E402
    HEADER_SIZE,
    RecordType,
    decode_payload,
    encode_record,
    iter_records,
)


def decode_all(data: bytes, verify_crc: bool = True):
    return list(iter_records(io.BytesIO(data), verify_crc=verify_crc))


class TestEncodeDecode(unittest.TestCase):
    def test_roundtrip_put(self):
        raw = encode_record(RecordType.PUT, b"name", b"alice")
        recs = decode_all(raw)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].rec_type, RecordType.PUT)
        self.assertEqual(recs[0].key, b"name")
        self.assertEqual(recs[0].value, b"alice")

    def test_roundtrip_delete(self):
        raw = encode_record(RecordType.DELETE, b"gone")
        recs = decode_all(raw)
        self.assertEqual(recs[0].rec_type, RecordType.DELETE)
        self.assertEqual(recs[0].key, b"gone")
        self.assertEqual(recs[0].value, b"")
        self.assertTrue(recs[0].is_delete)

    def test_empty_key_allowed(self):
        raw = encode_record(RecordType.PUT, b"", b"v")
        self.assertEqual(decode_all(raw)[0].key, b"")

    def test_empty_value_allowed(self):
        raw = encode_record(RecordType.PUT, b"k", b"")
        self.assertEqual(decode_all(raw)[0].value, b"")

    def test_binary_key_and_value(self):
        key = bytes(range(256))
        value = b"\x00\xff\x00\xff" * 100
        raw = encode_record(RecordType.PUT, key, value)
        rec = decode_all(raw)[0]
        self.assertEqual(rec.key, key)
        self.assertEqual(rec.value, value)

    def test_utf8_content(self):
        raw = encode_record(RecordType.PUT, "姓名".encode(), "张三".encode())
        rec = decode_all(raw)[0]
        self.assertEqual(rec.key.decode(), "姓名")
        self.assertEqual(rec.value.decode(), "张三")

    def test_large_value(self):
        value = b"x" * (1024 * 1024)      # 1 MiB
        raw = encode_record(RecordType.PUT, b"big", value)
        self.assertEqual(decode_all(raw)[0].value, value)

    def test_record_size_field(self):
        raw = encode_record(RecordType.PUT, b"abc", b"12345")
        rec = decode_all(raw)[0]
        self.assertEqual(rec.size, len(raw))
        self.assertEqual(rec.offset, 0)

    def test_multiple_records_offsets(self):
        parts = [
            encode_record(RecordType.PUT, b"a", b"1"),
            encode_record(RecordType.PUT, b"b", b"2"),
            encode_record(RecordType.DELETE, b"a"),
        ]
        recs = decode_all(b"".join(parts))
        self.assertEqual(len(recs), 3)
        # 每条记录的 offset 应等于前面所有记录长度之和
        expected = 0
        for rec, part in zip(recs, parts):
            self.assertEqual(rec.offset, expected)
            expected += len(part)


class TestCorruptionDetection(unittest.TestCase):
    def test_header_size_is_eight(self):
        """格式契约:头部固定 8 字节(4 字节长度 + 4 字节 CRC)。

        这个数字一旦改动,磁盘上所有历史 WAL 文件都会读不出来。
        """
        self.assertEqual(HEADER_SIZE, 8)

    def test_crc_mismatch_detected(self):
        raw = bytearray(encode_record(RecordType.PUT, b"k", b"value"))
        # 篡改**最后一个字节** —— 它在 value 数据区里,不破坏任何长度字段。
        # 这样失败原因就只可能是 CRC,测试意图才清晰。
        raw[-1] ^= 0xFF
        with self.assertRaises(ChecksumMismatchError):
            decode_all(bytes(raw))

    def test_crc_skipped_when_disabled(self):
        raw = bytearray(encode_record(RecordType.PUT, b"k", b"value"))
        raw[-1] ^= 0xFF
        # 关闭校验后结构仍然可解析,应能读出来(用于诊断工具)
        recs = decode_all(bytes(raw), verify_crc=False)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].key, b"k")
        self.assertNotEqual(recs[0].value, b"value")

    def test_structural_corruption_survives_crc_bypass(self):
        """关掉 CRC 也救不了结构性损坏。

        ``HEADER_SIZE + 3`` 这个字节落在 payload 的 key_len 字段里,
        改掉它会让解析器声称"key 有 6 万多字节" —— 即使跳过 CRC,
        也只能报截断。这正是 CRC 存在的意义:它能区分
        "内容变了但结构还在" 和 "结构已经不可信了"。
        """
        raw = bytearray(encode_record(RecordType.PUT, b"k", b"value"))
        raw[HEADER_SIZE + 3] ^= 0xFF
        with self.assertRaises(TruncatedRecordError):
            decode_all(bytes(raw), verify_crc=False)

    def test_truncated_header(self):
        raw = encode_record(RecordType.PUT, b"k", b"v")
        with self.assertRaises(TruncatedRecordError):
            decode_all(raw[:4])           # 头部都读不满

    def test_truncated_payload(self):
        raw = encode_record(RecordType.PUT, b"k", b"value")
        with self.assertRaises(TruncatedRecordError):
            decode_all(raw[:-3])          # payload 少几个字节

    def test_truncation_offset_reported(self):
        """截断位置必须准确 —— 恢复时要靠它决定截到哪。"""
        good = encode_record(RecordType.PUT, b"aaa", b"bbb")
        bad = encode_record(RecordType.PUT, b"ccc", b"ddd")
        with self.assertRaises(TruncatedRecordError) as ctx:
            decode_all(good + bad[:10])
        self.assertEqual(ctx.exception.offset, len(good))

    def test_unknown_record_type(self):
        # 手工构造一条类型字节为 99 的记录
        payload = bytes([99]) + b"\x00\x00\x00\x01k" + b"\x00\x00\x00\x00"
        crc = zlib.crc32(payload)
        import struct

        raw = struct.pack(">II", len(payload), crc) + payload
        with self.assertRaises(TruncatedRecordError):
            decode_all(raw)


class TestEdgeCases(unittest.TestCase):
    def test_empty_file_yields_nothing(self):
        self.assertEqual(decode_all(b""), [])

    def test_payload_decode_missing_key_len(self):
        with self.assertRaises(TruncatedRecordError):
            decode_payload(b"\x01\x00")

    def test_payload_decode_key_data_short(self):
        import struct

        payload = b"\x01" + struct.pack(">I", 10) + b"abc"   # 声称 key 10 字节,实际 3
        with self.assertRaises(TruncatedRecordError):
            decode_payload(payload)

    def test_payload_decode_missing_value_len(self):
        import struct

        payload = b"\x01" + struct.pack(">I", 1) + b"k"      # 没有 value_len
        with self.assertRaises(TruncatedRecordError):
            decode_payload(payload)

    def test_payload_decode_value_data_short(self):
        import struct

        payload = (
            b"\x01" + struct.pack(">I", 1) + b"k"
            + struct.pack(">I", 10) + b"ab"                   # 声称 10 字节,实际 2
        )
        with self.assertRaises(TruncatedRecordError):
            decode_payload(payload)

    def test_record_equality(self):
        from mini_lsm.record import Record

        a = Record(RecordType.PUT, b"k", b"v")
        b = Record(RecordType.PUT, b"k", b"v")
        c = Record(RecordType.PUT, b"k", b"other")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main(verbosity=2)
