"""SSTable 测试。

这一层的两个核心承诺:
    1. **有序** —— 稀疏索引 + 二分查找全靠它,乱序会让查不到数据变成静默失败
    2. **不可变且完整** —— 写完之后文件要么完全可读,要么明确报损坏,
       绝不会"读出一半数据还当成功"

所以除了正常往返,重点覆盖损坏检测的每一条路径。
"""

import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mini_lsm.errors import (  # noqa: E402
    ChecksumMismatchError,
    CorruptionError,
    InvalidArgumentError,
    TruncatedRecordError,
)
from mini_lsm.record import HEADER_SIZE  # noqa: E402
from mini_lsm.sstable import (  # noqa: E402
    DEFAULT_BLOCK_SIZE,
    FOOTER_SIZE,
    SSTableReader,
    SSTableWriter,
    parse_file_id,
    read_all,
    sstable_filename,
)


class SSTableTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.path = self.dir / "sst-000001.sst"

    def write_table(self, entries, block_size=DEFAULT_BLOCK_SIZE, path=None):
        """把 (key, value) 列表写成 SSTable;value 传 None 表示墓碑。"""
        target = path or self.path
        writer = SSTableWriter(target, block_size=block_size)
        for key, value in entries:
            writer.add(key, value)
        return writer.finish()

    def patch(self, path, offset, data: bytes) -> None:
        """就地改写文件里的一段字节,用来伪造损坏。"""
        with open(path, "r+b") as fh:
            fh.seek(offset)
            fh.write(data)


class TestNaming(unittest.TestCase):
    def test_filename_is_zero_padded(self):
        self.assertEqual(sstable_filename(1), "sst-000001.sst")
        self.assertEqual(sstable_filename(123456), "sst-123456.sst")

    def test_filename_sort_order_matches_age(self):
        """补零的意义:按文件名排序 == 按新旧排序。"""
        names = [sstable_filename(i) for i in (1, 2, 10, 100)]
        self.assertEqual(sorted(names), names)

    def test_parse_file_id(self):
        self.assertEqual(parse_file_id("sst-000042.sst"), 42)
        self.assertEqual(parse_file_id("/tmp/sst-7.sst"), 7)

    def test_parse_file_id_rejects_other_names(self):
        for name in ("wal.log", "sst-abc.sst", "sst-1.sst.tmp", "readme.md"):
            self.assertIsNone(parse_file_id(name), name)


class TestWriteRead(SSTableTestCase):
    def test_roundtrip(self):
        entries = [(b"a", b"1"), (b"b", b"2"), (b"c", b"3")]
        self.write_table(entries)

        with SSTableReader(self.path) as reader:
            self.assertEqual(list(reader.iter_entries()), entries)

    def test_entry_count_and_size(self):
        meta = self.write_table([(b"a", b"1"), (b"b", b"2")])
        self.assertEqual(meta.entry_count, 2)
        self.assertGreater(meta.size, 0)
        self.assertEqual(meta.size, self.path.stat().st_size)

    def test_meta_first_and_last_key(self):
        meta = self.write_table([(b"a", b"1"), (b"m", b"2"), (b"z", b"3")])
        self.assertEqual(meta.first_key, b"a")
        self.assertEqual(meta.last_key, b"z")

    def test_empty_table(self):
        meta = self.write_table([])
        self.assertEqual(meta.entry_count, 0)
        self.assertIsNone(meta.first_key)
        self.assertIsNone(meta.last_key)

        with SSTableReader(self.path) as reader:
            self.assertEqual(reader.entry_count, 0)
            self.assertEqual(reader.block_count, 0)
            self.assertIsNone(reader.first_key)
            self.assertIsNone(reader.last_key)
            self.assertEqual(list(reader.iter_entries()), [])
            self.assertEqual(reader.get(b"anything"), (False, None))

    def test_single_entry(self):
        self.write_table([(b"only", b"one")])
        with SSTableReader(self.path) as reader:
            self.assertEqual(reader.get(b"only"), (True, b"one"))
            self.assertEqual(reader.first_key, b"only")
            self.assertEqual(reader.last_key, b"only")

    def test_binary_keys_and_values(self):
        entries = [
            (bytes([i]), bytes(range(256)) * 4)
            for i in range(200)
        ]
        self.write_table(entries)
        with SSTableReader(self.path) as reader:
            self.assertEqual(list(reader.iter_entries()), entries)

    def test_utf8_content(self):
        # 注意顺序:按 UTF-8 字节比较,"城市" < "姓名"
        entries = [("城市".encode(), "深圳".encode()),
                   ("姓名".encode(), "张三".encode())]
        self.write_table(entries)
        with SSTableReader(self.path) as reader:
            self.assertEqual(reader.get("城市".encode()), (True, "深圳".encode()))

    def test_empty_key_and_value(self):
        self.write_table([(b"", b""), (b"k", b"")])
        with SSTableReader(self.path) as reader:
            self.assertEqual(reader.get(b""), (True, b""))
            self.assertEqual(reader.get(b"k"), (True, b""))

    def test_large_value_exceeding_block_size(self):
        """单个 value 比块还大时必须照样能写能读,不能切坏。"""
        big = b"x" * (64 * 1024)
        self.write_table([(b"big", big), (b"small", b"1")], block_size=1024)
        with SSTableReader(self.path) as reader:
            self.assertEqual(reader.get(b"big"), (True, big))
            self.assertEqual(reader.get(b"small"), (True, b"1"))

    def test_reopen_after_close(self):
        self.write_table([(b"a", b"1")])
        reader = SSTableReader(self.path)
        reader.close()
        self.assertTrue(reader.closed)
        with SSTableReader(self.path) as again:
            self.assertEqual(again.get(b"a"), (True, b"1"))


class TestBlocks(SSTableTestCase):
    def test_multiple_blocks_created(self):
        entries = [(f"key{i:04d}".encode(), b"v" * 20) for i in range(50)]
        self.write_table(entries, block_size=128)

        with SSTableReader(self.path) as reader:
            self.assertEqual(reader.entry_count, 50)
            self.assertGreater(reader.block_count, 1)
            # 不管 key 落在哪个块,都要能查到
            for key, value in entries:
                self.assertEqual(reader.get(key), (True, value), key)

    def test_block_count_grows_as_block_size_shrinks(self):
        entries = [(f"k{i:04d}".encode(), b"v" * 20) for i in range(40)]
        counts = []
        for block_size in (256, 512, 2048):
            path = self.dir / f"sst-{block_size:06d}.sst"
            self.write_table(entries, block_size=block_size, path=path)
            with SSTableReader(path) as reader:
                counts.append(reader.block_count)
        self.assertGreater(counts[0], counts[1])
        self.assertGreater(counts[1], counts[2])

    def test_iter_entries_crosses_block_boundaries_in_order(self):
        entries = [(f"k{i:04d}".encode(), f"v{i}".encode()) for i in range(100)]
        self.write_table(entries, block_size=64)
        with SSTableReader(self.path) as reader:
            self.assertGreater(reader.block_count, 10)
            self.assertEqual(list(reader.iter_entries()), entries)


class TestGet(SSTableTestCase):
    def setUp(self):
        super().setUp()
        self.entries = [(f"key{i:03d}".encode(), f"value{i}".encode())
                        for i in range(100)]
        self.write_table(self.entries, block_size=256)

    def test_hit_every_key(self):
        with SSTableReader(self.path) as reader:
            for key, value in self.entries:
                self.assertEqual(reader.get(key), (True, value), key)

    def test_miss_below_range(self):
        with SSTableReader(self.path) as reader:
            self.assertEqual(reader.get(b""), (False, None))
            self.assertEqual(reader.get(b"aaa"), (False, None))

    def test_miss_above_range(self):
        with SSTableReader(self.path) as reader:
            self.assertEqual(reader.get(b"zzzz"), (False, None))

    def test_miss_between_keys(self):
        """落在区间内但确实不存在的 key —— 这是最容易被写错的路径。"""
        with SSTableReader(self.path) as reader:
            self.assertEqual(reader.get(b"key000x"), (False, None))
            self.assertEqual(reader.get(b"key0505"), (False, None))

    def test_first_and_last_key_are_found(self):
        with SSTableReader(self.path) as reader:
            self.assertEqual(reader.get(b"key000"), (True, b"value0"))
            self.assertEqual(reader.get(b"key099"), (True, b"value99"))


class TestTombstones(SSTableTestCase):
    def test_tombstone_returns_found_with_none(self):
        """墓碑必须报 (True, None),而不是 (False, None)。

        这个区分决定上层要不要继续往更旧的文件里找 ——
        报错了就会让被删的键从底层 SSTable 复活。
        """
        self.write_table([(b"alive", b"1"), (b"dead", None)])
        with SSTableReader(self.path) as reader:
            self.assertEqual(reader.get(b"alive"), (True, b"1"))
            self.assertEqual(reader.get(b"dead"), (True, None))

    def test_tombstones_survive_iteration(self):
        self.write_table([(b"a", b"1"), (b"b", None), (b"c", b"3")])
        with SSTableReader(self.path) as reader:
            self.assertEqual(list(reader.iter_entries()),
                             [(b"a", b"1"), (b"b", None), (b"c", b"3")])

    def test_tombstone_counts_as_entry(self):
        meta = self.write_table([(b"a", b"1"), (b"b", None)])
        self.assertEqual(meta.entry_count, 2)

    def test_tombstone_only_table(self):
        self.write_table([(b"gone", None)])
        with SSTableReader(self.path) as reader:
            self.assertEqual(reader.get(b"gone"), (True, None))
            self.assertEqual(reader.entry_count, 1)


class TestWriterValidation(SSTableTestCase):
    def test_rejects_unsorted_keys(self):
        """乱序必须当场报错 —— 静默接受会让二分查找查不到数据。"""
        writer = SSTableWriter(self.path)
        writer.add(b"b", b"1")
        with self.assertRaises(InvalidArgumentError):
            writer.add(b"a", b"2")
        writer.abort()

    def test_rejects_duplicate_keys(self):
        writer = SSTableWriter(self.path)
        writer.add(b"a", b"1")
        with self.assertRaises(InvalidArgumentError):
            writer.add(b"a", b"2")
        writer.abort()

    def test_rejects_add_after_finish(self):
        writer = SSTableWriter(self.path)
        writer.add(b"a", b"1")
        writer.finish()
        with self.assertRaises(InvalidArgumentError):
            writer.add(b"b", b"2")

    def test_rejects_double_finish(self):
        writer = SSTableWriter(self.path)
        writer.add(b"a", b"1")
        writer.finish()
        with self.assertRaises(InvalidArgumentError):
            writer.finish()

    def test_rejects_bad_block_size(self):
        with self.assertRaises(InvalidArgumentError):
            SSTableWriter(self.path, block_size=0)

    def test_creates_parent_dirs(self):
        nested = self.dir / "a" / "b" / "sst-000001.sst"
        writer = SSTableWriter(nested)
        writer.add(b"k", b"v")
        writer.finish()
        self.assertTrue(nested.exists())

    def test_context_manager_aborts_on_error(self):
        with self.assertRaises(RuntimeError):
            with SSTableWriter(self.path) as writer:
                writer.add(b"a", b"1")
                raise RuntimeError("模拟中途出错")

    def test_abort_is_idempotent(self):
        writer = SSTableWriter(self.path)
        writer.add(b"a", b"1")
        writer.abort()
        writer.abort()      # 不应抛异常


class TestCorruptionDetection(SSTableTestCase):
    def test_file_too_small_for_footer(self):
        self.path.write_bytes(b"x" * 10)
        with self.assertRaises(CorruptionError):
            SSTableReader(self.path)

    def test_bad_magic(self):
        self.path.write_bytes(b"\x00" * 64)
        with self.assertRaises(CorruptionError) as ctx:
            SSTableReader(self.path)
        self.assertIn("magic", str(ctx.exception))

    def test_truncated_file(self):
        """把 footer 砍掉一截 —— 应该报损坏,而不是读到一半数据。"""
        self.write_table([(b"a", b"1"), (b"b", b"2")])
        raw = self.path.read_bytes()
        self.path.write_bytes(raw[:-8])
        with self.assertRaises(CorruptionError):
            SSTableReader(self.path)

    def test_index_offset_mismatch(self):
        """footer 里的索引位置和文件实际大小对不上。"""
        self.write_table([(b"a", b"1")])
        size = self.path.stat().st_size
        # 把 index_offset 改成一个明显不对的值(第一个字段,8 字节)
        self.patch(self.path, size - FOOTER_SIZE, struct.pack(">Q", 999))
        with self.assertRaises(CorruptionError) as ctx:
            SSTableReader(self.path)
        self.assertIn("对不上", str(ctx.exception))

    def test_corrupted_data_block_crc(self):
        """篡改数据块里的一个字节 → 读那块时必须报 CRC 错。"""
        self.write_table([(b"a", b"1"), (b"b", b"2")])
        self.patch(self.path, HEADER_SIZE + 5, b"\xff")

        with SSTableReader(self.path) as reader:
            # 打开时只读索引,还发现不了;真正读数据块时才暴露
            with self.assertRaises(ChecksumMismatchError):
                reader.get(b"a")

    def test_corrupted_index_crc(self):
        """篡改索引块 → 打开时就应该报错。"""
        self.write_table([(b"a", b"1"), (b"b", b"2")])
        size = self.path.stat().st_size
        index_offset = struct.unpack(">Q", self.path.read_bytes()[
            size - FOOTER_SIZE:size - FOOTER_SIZE + 8])[0]
        self.patch(self.path, index_offset + HEADER_SIZE + 2, b"\xff")

        with self.assertRaises(ChecksumMismatchError):
            SSTableReader(self.path)

    def test_index_corruption_is_caught_before_use(self):
        """索引被改会在**打开阶段**被 CRC 拦住。

        这比"先信了索引、按偏移 seek 之后才发现读不出来"安全得多 ——
        后者意味着已经拿不可信的数据去定位磁盘了。
        索引块整体有 CRC,所以伪造的块偏移根本没有机会被使用。
        """
        self.write_table([(b"a", b"1"), (b"b", b"2")])
        size = self.path.stat().st_size
        index_offset = struct.unpack(">Q", self.path.read_bytes()[
            size - FOOTER_SIZE:size - FOOTER_SIZE + 8])[0]
        entry_offset = index_offset + HEADER_SIZE
        # 篡改第一个索引项的 block_len(索引项布局:key_len(4)+offset(8)+len(8))
        self.patch(self.path, entry_offset + 4 + 8, struct.pack(">Q", 1 << 40))

        with self.assertRaises(ChecksumMismatchError):
            SSTableReader(self.path)

    def test_read_block_rejects_length_beyond_eof(self):
        """纵深防御:万一真拿到一个越界的块长度,也必须报错而不是读出界。

        正常路径上到不了这里(索引有 CRC 保护),但"读之前先核对长度"
        这个检查本身值得单独钉住 —— 它是最后一道防线。
        """
        self.write_table([(b"a", b"1")])
        with SSTableReader(self.path) as reader:
            with self.assertRaises(CorruptionError):
                reader._read_block(0, 1 << 40)

    def test_corrupted_index_entry_structure(self):
        """索引项声明的 key 长度超过索引块剩余长度。"""
        self.write_table([(b"aaa", b"1")])
        size = self.path.stat().st_size
        index_offset = struct.unpack(">Q", self.path.read_bytes()[
            size - FOOTER_SIZE:size - FOOTER_SIZE + 8])[0]
        entry_offset = index_offset + HEADER_SIZE
        # 把 key_len 改大,但不改 CRC —— 先撞上 CRC 校验
        self.patch(self.path, entry_offset, struct.pack(">I", 1 << 20))
        with self.assertRaises(CorruptionError):
            SSTableReader(self.path)


class TestHelpers(unittest.TestCase):
    def test_read_all(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "sst-000001.sst"
        writer = SSTableWriter(path)
        writer.add(b"a", b"1")
        writer.add(b"b", None)
        writer.finish()

        self.assertEqual(read_all(path), [(b"a", b"1"), (b"b", None)])

    def test_repr_does_not_raise(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "sst-000001.sst"
        writer = SSTableWriter(path)
        writer.add(b"a", b"1")
        writer.finish()
        with SSTableReader(path) as reader:
            self.assertIn("SSTableReader", repr(reader))
            self.assertEqual(len(reader), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
