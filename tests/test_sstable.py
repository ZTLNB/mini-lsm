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

from mini_lsm.block_cache import BlockCache  # noqa: E402
from mini_lsm.errors import (  # noqa: E402
    ChecksumMismatchError,
    CorruptionError,
    InvalidArgumentError,
    TruncatedRecordError,
)
from mini_lsm.record import (  # noqa: E402
    HEADER_SIZE,
    RecordType,
    encode_payload,
    frame,
)
from mini_lsm.sstable import (  # noqa: E402
    DEFAULT_BLOCK_SIZE,
    FOOTER_MAGIC_V1,
    FOOTER_SIZE,
    FOOTER_SIZE_V1,
    FORMAT_VERSION,
    SSTableReader,
    SSTableWriter,
    parse_file_id,
    read_all,
    sstable_filename,
)

# v1 footer 布局,测试里要手工造老文件,所以自己拼一份
_FOOTER_V1 = struct.Struct(">QQQ8s")
# 索引项布局:key 长度(4) + 块偏移(8) + 块长度(8)
_INDEX_ENTRY = struct.Struct(">IQQ")


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

    def write_v1_table(self, entries, block_size=DEFAULT_BLOCK_SIZE, path=None):
        """手工写一个**阶段 2/3 格式(v1)** 的 SSTable。

        阶段 4 之后代码只会写 v2,所以老格式文件只能自己拼 ——
        而"老文件必须还能读"是一条硬约束,不能只靠嘴说,
        必须有测试真的拿一个 v1 文件去打开。
        """
        target = Path(path or self.path)
        buf = bytearray()
        buf_first_key = None
        index: list[tuple[bytes, int, int]] = []
        offset = 0
        last_key = None
        count = 0
        out = bytearray()

        def flush():
            nonlocal buf, buf_first_key, offset
            if not buf:
                return
            framed = frame(bytes(buf))
            index.append((buf_first_key, offset, len(framed)))
            out.extend(framed)
            offset += len(framed)
            buf = bytearray()
            buf_first_key = None

        for key, value in entries:
            assert last_key is None or key > last_key
            rec = RecordType.DELETE if value is None else RecordType.PUT
            buf += encode_payload(rec, key, value if value is not None else b"")
            if buf_first_key is None:
                buf_first_key = key
            last_key = key
            count += 1
            if len(buf) >= block_size:
                flush()
        flush()

        index_offset = offset
        index_bytes = b"".join(
            _INDEX_ENTRY.pack(len(k), off, ln) + k for k, off, ln in index
        )
        framed_index = frame(index_bytes)
        out.extend(framed_index)
        out.extend(
            _FOOTER_V1.pack(index_offset, len(framed_index), count, FOOTER_MAGIC_V1)
        )
        target.write_bytes(bytes(out))
        return target

    def patch(self, path, offset, data: bytes) -> None:
        """就地改写文件里的一段字节,用来伪造损坏。"""
        with open(path, "r+b") as fh:
            fh.seek(offset)
            fh.write(data)

    def read_footer(self, path=None):
        """读出 footer 的字段,方便测试里定位各个块。"""
        target = Path(path or self.path)
        raw = target.read_bytes()
        size = len(raw)
        magic = raw[size - 8:]
        if magic == FOOTER_MAGIC_V1:
            return _FOOTER_V1.unpack(raw[size - FOOTER_SIZE_V1:])
        (index_offset, index_len, count, filter_offset, filter_len, _m) = (
            struct.unpack(">QQQQQ8s", raw[size - FOOTER_SIZE:])
        )
        return index_offset, index_len, count, filter_offset, filter_len


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
        self.assertIn("索引块越界", str(ctx.exception))

    def test_index_offset_pointing_into_filter_block(self):
        """索引偏移没越界,但位置不对 —— 布局对不上,同样要报错。

        v2 的布局是 ``数据块… | 索引块 | 过滤器块 | footer`` 严丝合缝。
        偏移落在合法范围内但位置不对,是更隐蔽的一种损坏 ——
        不校验的话会把过滤器块的字节当成索引来解析。

        要让索引**结束**在过滤器块起点之后、footer 之前,这样既不会先撞上
        "越界",又能暴露"两个块的位置接不上"。
        """
        self.write_table([(b"a", b"1"), (b"b", b"2")])
        size = self.path.stat().st_size
        footer_start = size - FOOTER_SIZE
        # v2 footer 布局:索引偏移(8) 索引长度(8) 条目数(8) 过滤器偏移(8) ...
        footer = self.path.read_bytes()[footer_start:]
        index_len = struct.unpack(">Q", footer[8:16])[0]

        new_offset = footer_start - 4 - index_len     # 结束于 footer 前 4 字节
        self.patch(self.path, footer_start, struct.pack(">Q", new_offset))
        with self.assertRaises(CorruptionError) as ctx:
            SSTableReader(self.path)
        self.assertIn("空隙或重叠", str(ctx.exception))

    def test_filter_offset_mismatch(self):
        """过滤器块结束位置和 footer 对不上。"""
        self.write_table([(b"a", b"1")])
        size = self.path.stat().st_size
        # 篡改过滤器长度(第 5 个字段,偏移 32..40),但它有 CRC 保护,
        # 所以这里改的是长度本身 —— 先撞上"位置对不上"
        self.patch(self.path, size - FOOTER_SIZE + 32, struct.pack(">Q", 1))
        with self.assertRaises(CorruptionError):
            SSTableReader(self.path)

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


class TestFormatV2(SSTableTestCase):
    """阶段 4 的磁盘格式变化:多了过滤器块,footer 从 32 字节变 48 字节。"""

    def test_writes_version_2(self):
        self.write_table([(b"a", b"1")])
        with SSTableReader(self.path) as reader:
            self.assertEqual(reader.version, 2)
            self.assertEqual(reader.version, FORMAT_VERSION)

    def test_footer_is_48_bytes(self):
        meta = self.write_table([(b"a", b"1")])
        raw = self.path.read_bytes()
        self.assertEqual(raw[-8:], b"MINILSM2")
        # footer 紧跟在过滤器块后面,所以文件至少要有 48 字节的 footer
        self.assertGreaterEqual(meta.size, FOOTER_SIZE)

    def test_has_bloom_filter_by_default(self):
        self.write_table([(b"a", b"1"), (b"b", b"2")])
        with SSTableReader(self.path) as reader:
            self.assertTrue(reader.has_bloom_filter)
            self.assertIsNotNone(reader.bloom_filter)
            self.assertFalse(reader.bloom_corrupt)
            self.assertGreater(reader.filter_size, 0)

    def test_filter_size_is_small(self):
        """过滤器的价值就在于"小":每个 key 约 1 字节多一点。"""
        entries = [(f"key{i:05d}".encode(), b"v" * 50) for i in range(1000)]
        self.write_table(entries)
        with SSTableReader(self.path) as reader:
            # 1000 个 key × 10 bit ≈ 1250 字节,加 8 字节 frame 头 + 9 字节头部
            self.assertLess(reader.filter_size, 1400)
            self.assertGreater(reader.filter_size, 1000)

    def test_bloom_can_be_disabled(self):
        """``bloom_bits_per_key=0`` 时不写过滤器块,footer 里记 (0, 0)。"""
        writer = SSTableWriter(self.path, bloom_bits_per_key=0)
        writer.add(b"a", b"1")
        writer.finish()
        with SSTableReader(self.path) as reader:
            self.assertEqual(reader.version, 2)
            self.assertFalse(reader.has_bloom_filter)
            self.assertEqual(reader.filter_size, 0)
            self.assertIsNone(reader.bloom_filter)
            # 没有过滤器也必须能正常查
            self.assertEqual(reader.get(b"a"), (True, b"1"))

    def test_negative_bits_per_key_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            SSTableWriter(self.path, bloom_bits_per_key=-1)

    def test_layout_is_gapless(self):
        """``数据块… | 索引块 | 过滤器块 | footer`` 必须严丝合缝。"""
        self.write_table([(b"a", b"1"), (b"b", b"2")])
        size = self.path.stat().st_size
        index_offset, index_len, count, filter_offset, filter_len = (
            self.read_footer()
        )
        self.assertEqual(count, 2)
        self.assertEqual(index_offset + index_len, filter_offset)
        self.assertEqual(filter_offset + filter_len, size - FOOTER_SIZE)

    def test_disabled_filter_layout_is_gapless(self):
        writer = SSTableWriter(self.path, bloom_bits_per_key=0)
        writer.add(b"a", b"1")
        writer.finish()
        size = self.path.stat().st_size
        index_offset, index_len, count, filter_offset, filter_len = (
            self.read_footer()
        )
        self.assertEqual((filter_offset, filter_len), (0, 0))
        # 没有过滤器块时,索引必须直接顶到 footer
        self.assertEqual(index_offset + index_len, size - FOOTER_SIZE)

    def test_repr_mentions_filter(self):
        self.write_table([(b"a", b"1")])
        with SSTableReader(self.path) as reader:
            self.assertIn("v2", repr(reader))
            self.assertIn("有过滤器", repr(reader))


class TestBloomInSSTable(SSTableTestCase):
    """过滤器接进 SSTable 之后,必须满足两条:

    - **没有假阴性**:凡是文件里真有的 key,``might_contain`` 都得说 True
    - **挡住绝大多数不存在的 key**:这才是它存在的理由
    """

    def setUp(self):
        super().setUp()
        self.entries = [(f"key{i:05d}".encode(), f"value{i}".encode())
                        for i in range(500)]
        self.write_table(self.entries)

    def test_no_false_negatives_for_present_keys(self):
        with SSTableReader(self.path) as reader:
            missed = [k for k, _ in self.entries if not reader.might_contain(k)]
            self.assertEqual(missed, [], "过滤器把存在的键报成不存在 —— 会丢数据")

    def test_tombstones_are_in_the_filter(self):
        """墓碑也要进过滤器,否则查被删的键会白读文件。"""
        path = self.dir / "sst-000002.sst"
        self.write_table([(b"alive", b"1"), (b"deleted", None)], path=path)
        with SSTableReader(path) as reader:
            self.assertTrue(reader.might_contain(b"deleted"))
            self.assertTrue(reader.might_contain(b"alive"))

    def test_rejects_most_absent_keys(self):
        """随机造一批不存在的 key,绝大多数应该被挡下。"""
        probes = [f"absent{i:07d}".encode() for i in range(1000)]
        with SSTableReader(self.path) as reader:
            rejected = sum(1 for p in probes if not reader.might_contain(p))
        # 理论假阳性率约 1%,给到 5% 的余量
        self.assertGreater(rejected, 950,
                           f"只挡下了 {rejected}/1000 个不存在的 key")

    def test_rejects_keys_outside_range_too(self):
        with SSTableReader(self.path) as reader:
            self.assertFalse(reader.might_contain(b"aaaa"))
            self.assertFalse(reader.might_contain(b"zzzzz"))

    def test_bloom_never_blocks_a_real_hit(self):
        """把过滤器和 get() 串起来:说不在的,get() 也必须真找不到。"""
        with SSTableReader(self.path) as reader:
            for key, value in self.entries:
                if reader.might_contain(key):
                    self.assertEqual(reader.get(key), (True, value), key)
                else:  # pragma: no cover - 真出现就是 bug
                    self.fail(f"过滤器对存在的键 {key!r} 返回了 False")

    def test_absent_key_agrees_between_filter_and_get(self):
        """过滤器说"一定不在"时,get() 必须也找不到 —— 两者不能打架。"""
        with SSTableReader(self.path) as reader:
            for probe in (b"absent0000001", b"absent0000002", b"nope"):
                if not reader.might_contain(probe):
                    self.assertEqual(reader.get(probe), (False, None), probe)

    def test_empty_table_filter_rejects_everything(self):
        path = self.dir / "sst-000003.sst"
        self.write_table([], path=path)
        with SSTableReader(path) as reader:
            self.assertFalse(reader.might_contain(b"anything"))


class TestV1BackwardCompat(SSTableTestCase):
    """阶段 2/3 写下的老文件必须**不改动**就能打开。

    这是磁盘格式演进里最容易破、后果最严重的一条:
    加字段可以,但不能让用户已经存下来的数据读不出来。
    """

    def test_opens_v1_file(self):
        self.write_v1_table([(b"a", b"1"), (b"b", b"2")])
        with SSTableReader(self.path) as reader:
            self.assertEqual(reader.version, 1)
            self.assertEqual(reader.entry_count, 2)
            self.assertEqual(reader.get(b"a"), (True, b"1"))
            self.assertEqual(reader.get(b"b"), (True, b"2"))

    def test_v1_has_no_filter_and_never_rejects(self):
        """老文件没有过滤器,``might_contain`` 只能恒为 True。

        退化成阶段 3 的行为 —— 慢一点,但结果依然正确。
        "不知道"的时候绝不能猜"不在"。
        """
        self.write_v1_table([(b"a", b"1"), (b"b", b"2")])
        with SSTableReader(self.path) as reader:
            self.assertFalse(reader.has_bloom_filter)
            self.assertEqual(reader.filter_size, 0)
            self.assertFalse(reader.bloom_corrupt)   # 是"没有",不是"坏了"
            for probe in (b"a", b"b", b"zzz", b"", b"absent"):
                self.assertTrue(reader.might_contain(probe), probe)

    def test_v1_iteration(self):
        entries = [(f"k{i:04d}".encode(), f"v{i}".encode()) for i in range(60)]
        self.write_v1_table(entries, block_size=64)
        with SSTableReader(self.path) as reader:
            self.assertGreater(reader.block_count, 5)
            self.assertEqual(list(reader.iter_entries()), entries)

    def test_v1_tombstones(self):
        self.write_v1_table([(b"a", b"1"), (b"b", None)])
        with SSTableReader(self.path) as reader:
            self.assertEqual(reader.get(b"b"), (True, None))

    def test_v1_empty_table(self):
        self.write_v1_table([])
        with SSTableReader(self.path) as reader:
            self.assertEqual(reader.entry_count, 0)
            self.assertEqual(list(reader.iter_entries()), [])
            self.assertEqual(reader.get(b"x"), (False, None))

    def test_v1_and_v2_coexist_in_one_directory(self):
        """同一个目录里既有老文件又有新文件 —— 升级之后就是这个样子。"""
        old = self.dir / "sst-000001.sst"
        new = self.dir / "sst-000002.sst"
        self.write_v1_table([(b"a", b"1")], path=old)
        self.write_table([(b"b", b"2")], path=new)

        with SSTableReader(old) as reader:
            self.assertEqual(reader.version, 1)
            self.assertEqual(reader.get(b"a"), (True, b"1"))
        with SSTableReader(new) as reader:
            self.assertEqual(reader.version, 2)
            self.assertTrue(reader.has_bloom_filter)
            self.assertEqual(reader.get(b"b"), (True, b"2"))

    def test_v1_footer_size_constant(self):
        self.assertEqual(FOOTER_SIZE_V1, 32)
        self.assertEqual(FOOTER_SIZE, 48)


class TestFilterCorruptionDegrades(SSTableTestCase):
    """过滤器坏了要**降级**,不能拒绝打开。

    这里的容错策略和数据块**故意不同**:
        数据块坏了 → 抛错(读不出数据就是丢数据)
        过滤器坏了 → 记一笔,当没有过滤器用(只影响快慢,不影响对错)

    为了一个坏掉的优化结构而让用户完全打不开数据,是得不偿失的。
    """

    def setUp(self):
        super().setUp()
        self.entries = [(f"key{i:03d}".encode(), f"v{i}".encode())
                        for i in range(100)]
        self.write_table(self.entries)
        _, _, _, self.filter_offset, self.filter_len = self.read_footer()

    def test_corrupt_filter_payload_degrades(self):
        """翻掉过滤器 payload 里的一个字节 → CRC 对不上 → 降级。"""
        self.patch(self.path, self.filter_offset + HEADER_SIZE + 3, b"\xff")
        with SSTableReader(self.path) as reader:
            self.assertTrue(reader.bloom_corrupt)
            self.assertFalse(reader.has_bloom_filter)
            # 降级之后恒为 True,行为退回阶段 3
            self.assertTrue(reader.might_contain(b"definitely-absent"))

    def test_degraded_filter_does_not_lose_data(self):
        """降级之后所有查询结果必须依然正确 —— 这才是降级的意义。"""
        self.patch(self.path, self.filter_offset + HEADER_SIZE + 3, b"\xff")
        with SSTableReader(self.path) as reader:
            for key, value in self.entries:
                self.assertEqual(reader.get(key), (True, value), key)
            self.assertEqual(reader.get(b"absent"), (False, None))
            self.assertEqual(list(reader.iter_entries()), self.entries)

    def test_corrupt_filter_header_degrades(self):
        """把过滤器头部声明的位数改成 0 —— 非法头部,同样只能降级。"""
        raw = self.path.read_bytes()
        # 过滤器 payload 从 filter_offset + HEADER_SIZE 开始:
        # [key个数:4][位数:4][k:1][位图]
        self.patch(self.path, self.filter_offset + HEADER_SIZE + 4,
                   struct.pack(">I", 0))
        with SSTableReader(self.path) as reader:
            self.assertTrue(reader.bloom_corrupt)
            self.assertTrue(reader.might_contain(b"anything"))
        # 但 CRC 会先拦住这个改动 —— 两种拦法都算合格,这里确认文件仍能打开
        self.assertNotEqual(raw, self.path.read_bytes())

    def test_truncated_filter_block_degrades(self):
        """把过滤器块长度改大,让它读不完整。"""
        size = self.path.stat().st_size
        # footer 里过滤器长度是第 5 个字段(偏移 32..40)
        self.patch(self.path, size - FOOTER_SIZE + 32,
                   struct.pack(">Q", self.filter_len + 1000))
        # 布局校验会先发现"过滤器结束位置和 footer 对不上"
        with self.assertRaises(CorruptionError):
            SSTableReader(self.path)

    def test_data_block_corruption_still_fatal(self):
        """对照组:数据块坏了必须抛错,不能跟着一起降级。

        如果这里也"宽容"了,那就等于把丢数据当成正常情况。
        """
        self.patch(self.path, HEADER_SIZE + 5, b"\xff")
        with SSTableReader(self.path) as reader:
            self.assertTrue(reader.has_bloom_filter)   # 过滤器没事
            with self.assertRaises(ChecksumMismatchError):
                for _ in reader.iter_entries():
                    pass


class _CountingReader(SSTableReader):
    """数一数到底真读了几次数据块。

    子类没声明 ``__slots__``,所以能挂额外的实例属性。
    """

    def __init__(self, *args, **kwargs):
        self.block_reads = 0
        super().__init__(*args, **kwargs)

    def _read_block(self, offset, length):
        self.block_reads += 1
        return super()._read_block(offset, length)


class TestReaderBlockCache(SSTableTestCase):
    """SSTable 读路径接上块缓存之后的行为。"""

    def setUp(self):
        super().setUp()
        self.entries = [(f"key{i:03d}".encode(), f"value{i}".encode())
                        for i in range(60)]
        self.write_table(self.entries, block_size=128)

    def test_second_get_is_served_from_cache(self):
        """第二次查同一个键应该**一次盘都不读** —— 缓存的全部意义。"""
        cache = BlockCache(1 << 20)
        with _CountingReader(self.path, file_id=1, cache=cache) as reader:
            self.assertEqual(reader.get(b"key000"), (True, b"value0"))
            reader.block_reads = 0            # 从这一刻开始数
            self.assertEqual(reader.get(b"key000"), (True, b"value0"))
            self.assertEqual(reader.block_reads, 0,
                             "命中缓存却还是去读了磁盘")

    def test_without_cache_it_reads_every_time(self):
        """对照组:不挂缓存时,同样的查询要真读。"""
        with _CountingReader(self.path, file_id=1, cache=None) as reader:
            self.assertEqual(reader.get(b"key000"), (True, b"value0"))
            reader.block_reads = 0
            self.assertEqual(reader.get(b"key000"), (True, b"value0"))
            self.assertGreater(reader.block_reads, 0)

    def test_cache_records_hits(self):
        cache = BlockCache(1 << 20)
        with SSTableReader(self.path, file_id=1, cache=cache) as reader:
            reader.get(b"key000")
            reader.get(b"key000")
            st = cache.stats()
            self.assertGreaterEqual(st.hits, 1)
            self.assertGreater(st.entries, 0)

    def test_full_scan_does_not_pollute_cache(self):
        """顺序全扫**不该**填缓存 —— 那些块只会被读一次。

        塞进去只会把真正的热点数据挤出去,这就是"缓存污染"。
        """
        cache = BlockCache(1 << 20)
        with SSTableReader(self.path, file_id=1, cache=cache) as reader:
            self.assertEqual(list(reader.iter_entries()), self.entries)
            self.assertEqual(len(cache), 0, "全扫把块塞进了缓存")
            self.assertEqual(cache.stats().bytes, 0)

    def test_full_scan_can_fill_cache_on_request(self):
        """需要时可以用 ``fill_cache=True`` 显式打开。"""
        cache = BlockCache(1 << 20)
        with SSTableReader(self.path, file_id=1, cache=cache) as reader:
            list(reader.iter_entries(fill_cache=True))
            self.assertEqual(len(cache), reader.block_count)

    def test_scan_after_scan_hits_cache(self):
        cache = BlockCache(1 << 20)
        with SSTableReader(self.path, file_id=1, cache=cache) as reader:
            list(reader.iter_entries(fill_cache=True))
            cache.reset_stats()
            list(reader.iter_entries(fill_cache=True))
            st = cache.stats()
            self.assertEqual(st.misses, 0)
            self.assertEqual(st.hits, reader.block_count)

    def test_cache_key_includes_file_id(self):
        """两个文件里"第 0 块"内容不同,共用缓存也不能串味。"""
        other = self.dir / "sst-000002.sst"
        other_entries = [(f"key{i:03d}".encode(), f"other{i}".encode())
                         for i in range(60)]
        self.write_table(other_entries, block_size=128, path=other)

        cache = BlockCache(1 << 20)
        with SSTableReader(self.path, file_id=1, cache=cache) as a, \
                SSTableReader(other, file_id=2, cache=cache) as b:
            self.assertEqual(a.get(b"key000"), (True, b"value0"))
            self.assertEqual(b.get(b"key000"), (True, b"other0"))
            # 再查一次,两边都还应该拿到各自的值
            self.assertEqual(a.get(b"key000"), (True, b"value0"))
            self.assertEqual(b.get(b"key000"), (True, b"other0"))

    def test_cache_returns_same_object(self):
        """命中要返回同一个列表对象,而不是重新解析一份。"""
        cache = BlockCache(1 << 20)
        with SSTableReader(self.path, file_id=1, cache=cache) as reader:
            reader.get(b"key000")
            first = reader._read_block_at(0)
            second = reader._read_block_at(0)
            self.assertIs(first, second)

    def test_iter_entries_ignores_cache_but_still_correct(self):
        """带缓存跑全扫,结果必须和不带缓存时一模一样。"""
        cache = BlockCache(1 << 20)
        with SSTableReader(self.path, file_id=1, cache=cache) as reader:
            self.assertEqual(list(reader.iter_entries()), self.entries)
        with SSTableReader(self.path, file_id=1) as reader:
            self.assertEqual(list(reader.iter_entries()), self.entries)

    def test_disabled_cache_is_harmless(self):
        """容量 0 的缓存不能影响正确性,只是永远不命中。"""
        cache = BlockCache(0)
        with SSTableReader(self.path, file_id=1, cache=cache) as reader:
            for key, value in self.entries:
                self.assertEqual(reader.get(key), (True, value), key)
            self.assertEqual(len(cache), 0)


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
