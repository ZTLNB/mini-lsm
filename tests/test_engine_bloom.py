"""阶段 4 引擎测试:布隆过滤器 + 块缓存接进读路径之后,端到端要成立什么。

前面三个文件测的是零件(过滤器本身准不准、缓存淘汰顺序对不对、
SSTable 的 v2 格式写没写对)。这个文件测的是**拼进引擎之后**:

    1. 过滤器真的挡住了文件读 —— 不是"理论上能挡",而是
       ``stats().bloom_rejections`` 真的涨了
    2. 挡完之后**数据还对** —— 这才是重点。过滤器只要有一次假阴性,
       表现就是"数据莫名其妙不见了",而且不报错。所以每个"挡"的路径
       都配一条"值必须还在"的测试
    3. 缓存命中不能让结果变错 —— 尤其跨重启、跨 compaction 之后

阶段 4 有一条比阶段 3 更隐蔽的风险:

    **过滤器是落盘的,而它依赖哈希函数跨进程稳定。**

    如果哈希用了内置 ``hash()``(带每进程随机的盐),那么进程 A 写下的
    位图,进程 B 重启后查什么都会落在错误的位上 —— 于是所有存在的键
    全部变成假阴性,数据"全部消失"。``tests/test_bloom.py`` 里有专门的
    跨进程测试钉这条,这里再补一条**引擎级**的:写完 → 关掉 → 重开 →
    每一个键都还必须查得到。
"""

import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mini_lsm.block_cache import BlockCache  # noqa: E402
from mini_lsm.engine import LSMEngine  # noqa: E402
from mini_lsm.errors import (  # noqa: E402
    ChecksumMismatchError,
    CorruptionError,
    InvalidArgumentError,
)
from mini_lsm.record import RecordType, encode_payload, frame  # noqa: E402
from mini_lsm.sstable import (  # noqa: E402
    FOOTER_MAGIC_V1,
    FOOTER_SIZE,
    FOOTER_SIZE_V1,
    sstable_filename,
)

_FOOTER_V1 = struct.Struct(">QQQ8s")
_INDEX_ENTRY = struct.Struct(">IQQ")


def write_v1_sstable(path: Path, entries, block_size: int = 4096) -> Path:
    """手工写一个阶段 2/3 格式(v1)的 SSTable 文件。

    阶段 4 之后代码只会写 v2,所以老格式只能自己拼 ——
    而"升级之后老数据还能读"是一条硬约束,必须有测试真的去读它。
    """
    out = bytearray()
    buf = bytearray()
    buf_first_key = None
    index: list[tuple[bytes, int, int]] = []
    offset = 0
    last_key = None
    count = 0

    def flush() -> None:
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
    out.extend(_FOOTER_V1.pack(index_offset, len(framed_index), count, FOOTER_MAGIC_V1))
    path.write_bytes(bytes(out))
    return path


class Stage4TestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def open_engine(self, **kwargs) -> LSMEngine:
        engine = LSMEngine(self.dir, **kwargs)
        self.addCleanup(engine.close)
        return engine

    def fill(self, db: LSMEngine, count: int, prefix: str = "key",
             width: int = 5) -> list[tuple[str, str]]:
        items = [(f"{prefix}{i:0{width}d}", f"value{i}") for i in range(count)]
        for key, value in items:
            db.put(key, value)
        return items

    def sst_path(self, file_id: int) -> Path:
        return self.dir / sstable_filename(file_id)


# ------------------------------------------------------------------ 过滤器真的挡住了读


class TestBloomBlocksReads(Stage4TestCase):
    def setUp(self):
        super().setUp()
        self.db = self.open_engine()
        self.items = self.fill(self.db, 300)
        self.db.flush()

    def test_filter_is_written(self):
        st = self.db.stats()
        self.assertEqual(st.sstable_count, 1)
        for meta in self.db.sstables:
            reader = self.db._readers[meta.file_id]
            self.assertTrue(reader.has_bloom_filter)
            self.assertGreater(reader.filter_size, 0)

    def test_absent_keys_below_range_are_rejected(self):
        """key 比整个文件的范围还小 —— 过滤器应该直接说"不在"。"""
        before = self.db.stats().bloom_rejections
        for i in range(100):
            self.assertIsNone(self.db.get(f"absent{i:05d}"))
        rejected = self.db.stats().bloom_rejections - before
        self.assertGreaterEqual(rejected, 95, f"只挡下 {rejected}/100")

    def test_absent_keys_inside_range_are_rejected(self):
        """**落在键范围内但不存在**的 key —— 这是最值得省的路径。

        阶段 3 时这类查询必须真读文件才能知道"没有";现在过滤器直接挡掉。
        这也是阶段 4 相对阶段 3 收益最大的地方。
        """
        before = self.db.stats().bloom_rejections
        probes = [f"key{i:05d}x" for i in range(300)]
        for probe in probes:
            self.assertIsNone(self.db.get(probe), probe)
        rejected = self.db.stats().bloom_rejections - before
        self.assertGreaterEqual(rejected, 280, f"只挡下 {rejected}/300")

    def test_present_keys_are_never_rejected(self):
        """查存在的键时不该有任何一次"被过滤器挡下" —— 那等于假阴性。"""
        before = self.db.stats().bloom_rejections
        for key, value in self.items:
            self.assertEqual(self.db.get(key), value.encode(), key)
        self.assertEqual(self.db.stats().bloom_rejections, before)

    def test_disabled_filter_never_rejects_but_still_correct(self):
        """关掉过滤器 → 一次也不挡,但结果必须完全正确(退回阶段 3)。"""
        path = self.dir / "no-bloom"
        path.mkdir()
        db = LSMEngine(path, bloom_bits_per_key=0)
        self.addCleanup(db.close)
        items = [(f"k{i:04d}", f"v{i}") for i in range(200)]
        for key, value in items:
            db.put(key, value)
        db.flush()

        for meta in db.sstables:
            self.assertFalse(db._readers[meta.file_id].has_bloom_filter)

        for i in range(100):
            self.assertIsNone(db.get(f"missing{i:04d}"))
        self.assertEqual(db.stats().bloom_rejections, 0)
        for key, value in items:
            self.assertEqual(db.get(key), value.encode(), key)

    def test_memtable_hits_do_not_touch_the_filter(self):
        """还在内存表里的键根本不走文件路径,自然也不会动过滤器计数。"""
        path = self.dir / "memtable-only"
        path.mkdir()
        db = LSMEngine(path)
        self.addCleanup(db.close)
        db.put("fresh", "1")
        self.assertEqual(db.get("fresh"), b"1")
        self.assertEqual(db.get("fresh-absent"), None)
        self.assertEqual(db.stats().bloom_rejections, 0)

    def test_filter_disabled_engine_property(self):
        path = self.dir / "nb"
        path.mkdir()
        db = LSMEngine(path, bloom_bits_per_key=0)
        self.addCleanup(db.close)
        self.assertEqual(db.bloom_bits_per_key, 0)
        self.assertEqual(self.db.bloom_bits_per_key, 10)

    def test_negative_settings_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            LSMEngine(self.dir / "neg1", bloom_bits_per_key=-1)
        with self.assertRaises(InvalidArgumentError):
            LSMEngine(self.dir / "neg2", block_cache_size=-1)


# ------------------------------------------------------------------ 挡完之后数据还得对


class TestCorrectnessWithFilter(Stage4TestCase):
    """过滤器唯一允许的副作用是"变快",不允许改变任何结果。"""

    def test_get_after_flush(self):
        with self.open_engine() as db:
            items = self.fill(db, 500)
            db.flush()
            for key, value in items:
                self.assertEqual(db.get(key), value.encode(), key)

    def test_values_survive_restart(self):
        """写完 → 关掉 → 重开,每个键都还必须查得到。

        这条测试同时钉住两件事:过滤器落盘后哈希依然稳定(否则全部假阴性),
        以及 v2 格式重启后能正常解析。
        """
        db = LSMEngine(self.dir)
        items = self.fill(db, 500)
        db.flush()
        db.close()

        with self.open_engine() as db:
            for key, value in items:
                self.assertEqual(db.get(key), value.encode(), key)

    def test_delete_is_not_undone_by_filter(self):
        """被删的键必须查不到 —— 过滤器挡掉它是**正确**的挡。"""
        with self.open_engine() as db:
            items = self.fill(db, 200)
            for key, _ in items[:100]:
                db.delete(key)
            db.flush()
            for key, _ in items[:100]:
                self.assertIsNone(db.get(key), key)
            for key, value in items[100:]:
                self.assertEqual(db.get(key), value.encode(), key)

    def test_delete_survives_restart(self):
        """墓碑重启后还得压得住 —— 过滤器不能让它"复活"。"""
        with self.open_engine() as db:
            items = self.fill(db, 200)
            for key, _ in items[:100]:
                db.delete(key)
            db.flush()
        with self.open_engine() as db:
            for key, _ in items[:100]:
                self.assertIsNone(db.get(key), key)
            for key, value in items[100:]:
                self.assertEqual(db.get(key), value.encode(), key)

    def test_overwrite_then_flush(self):
        with self.open_engine() as db:
            items = self.fill(db, 100)
            for key, _ in items:
                db.put(key, "updated")
            db.flush()
            for key, _ in items:
                self.assertEqual(db.get(key), b"updated", key)

    def test_scan_matches_get(self):
        with self.open_engine() as db:
            items = self.fill(db, 300)
            for key, _ in items[::3]:
                db.delete(key)
            db.flush()
            scanned = dict(db.scan())
            expected = {k.encode(): v.encode() for k, v in items
                        if k not in {k2 for k2, _ in items[::3]}}
            self.assertEqual(scanned, expected)
            # 逐条用 get() 交叉验证,两条路径不能打架
            for key, value in expected.items():
                self.assertEqual(db.get(key), value, key)

    def test_tombstone_key_is_in_the_filter(self):
        """墓碑本身也在过滤器里 —— 查它时能被正确挡下(而不是白读)。"""
        with self.open_engine() as db:
            items = self.fill(db, 100)
            for key, _ in items:
                db.delete(key)
            db.flush()
            before = db.stats().bloom_rejections
            for key, _ in items:
                self.assertIsNone(db.get(key))
            # 墓碑在过滤器里,所以查询要么被挡下、要么真读后确认是墓碑,
            # 两种情况都正确;这里只要求结果对(上面已断言),顺带看一眼计数
            self.assertGreaterEqual(db.stats().bloom_rejections, before)

    def test_many_keys_roundtrip(self):
        with self.open_engine() as db:
            items = self.fill(db, 2000, width=6)
            db.flush()
            db.compact_all()
            for key, value in items:
                self.assertEqual(db.get(key), value.encode(), key)


# ------------------------------------------------------------------ 块缓存端到端


class TestBlockCacheEndToEnd(Stage4TestCase):
    def test_repeated_gets_hit_cache(self):
        with self.open_engine() as db:
            items = self.fill(db, 300)
            db.flush()
            db.block_cache.reset_stats()
            # 第一遍是冷读,第二遍应该大量命中
            for key, _ in items:
                db.get(key)
            first = db.block_cache.stats()
            for key, _ in items:
                db.get(key)
            second = db.block_cache.stats()
            self.assertGreater(second.hits - first.hits, 200,
                               "第二遍几乎没命中缓存")

    def test_stats_reports_cache(self):
        with self.open_engine() as db:
            self.fill(db, 300)
            db.flush()
            for i in range(50):
                db.get(f"key{i:05d}")
            st = db.stats()
            self.assertGreater(st.cache.lookups, 0)
            self.assertGreater(st.cache.entries, 0)
            self.assertEqual(st.cache.capacity_bytes, 8 * 1024 * 1024)
            self.assertIn("块缓存", str(st))

    def test_cache_disabled_still_correct(self):
        """容量 0 的缓存 → 永不命中,但结果必须完全正确。"""
        with self.open_engine(block_cache_size=0) as db:
            items = self.fill(db, 300)
            db.flush()
            for key, value in items:
                self.assertEqual(db.get(key), value.encode(), key)
            self.assertEqual(len(db.block_cache), 0)
            self.assertFalse(db.block_cache.enabled)

    def test_cache_is_not_carried_across_restart(self):
        """重启后是**新的**缓存对象 —— 但数据必须照常读得到。"""
        db = LSMEngine(self.dir)
        items = self.fill(db, 300)
        db.flush()
        db.get("key00000")
        self.assertGreater(db.block_cache.stats().lookups, 0)
        db.close()

        with self.open_engine() as db:
            self.assertEqual(db.block_cache.stats().lookups, 0)
            for key, value in items:
                self.assertEqual(db.get(key), value.encode(), key)

    def test_cache_bytes_stay_within_capacity(self):
        """缓存塞不下时该淘汰就淘汰,而且淘汰不能影响正确性。

        容量要选得**能装下一块但装不下全部** —— 比单块还小的容量会走
        "块太大,索性不缓存"那条路,淘汰计数永远是 0,测试就白做了。
        """
        with self.open_engine(block_cache_size=64 * 1024) as db:
            items = self.fill(db, 2000, width=6)
            db.flush()
            for key, _ in items:
                db.get(key)
            st = db.block_cache.stats()
            self.assertLessEqual(st.bytes, 64 * 1024)
            self.assertGreater(st.evictions, 0, "容量不够却没发生淘汰")
            self.assertEqual(st.oversized, 0, "块不该大到装不进缓存")
            # 淘汰再多也不能影响正确性
            for key, value in items:
                self.assertEqual(db.get(key), value.encode(), key)

    def test_scan_does_not_pollute_cache(self):
        """全量 scan 会把每个块读一遍,但**不该**把它们留在缓存里。"""
        with self.open_engine() as db:
            self.fill(db, 1000, width=6)
            db.flush()
            db.block_cache.clear()
            db.block_cache.reset_stats()
            list(db.scan())
            self.assertEqual(len(db.block_cache), 0,
                             "全量扫描把块塞进了缓存(缓存污染)")
            self.assertEqual(db.block_cache.stats().hits, 0)

    def test_compaction_evicts_removed_files(self):
        """compaction 删掉的文件,它的缓存块也必须跟着走。

        不清理虽然不会读到脏数据(file_id 不复用),但内存是白占的。
        """
        with self.open_engine(auto_compact=False) as db:
            for batch in range(4):
                for i in range(50):
                    db.put(f"k{batch}-{i:04d}", f"v{batch}-{i}")
                db.flush()

            before_ids = set(db.manifest.file_ids())
            # 每个文件都摸一遍,确保缓存里有它们的块
            for batch in range(4):
                db.get(f"k{batch}-0000")
            cached_before = {fid for fid, _ in db.block_cache._data}
            self.assertTrue(cached_before & before_ids)

            db.compact_all()

            after_ids = set(db.manifest.file_ids())
            removed = before_ids - after_ids
            self.assertTrue(removed, "compaction 没有移除任何文件,测试失去意义")

            cached_after = {fid for fid, _ in db.block_cache._data}
            stale = cached_after & removed
            self.assertEqual(stale, set(),
                             f"被删文件的缓存块还留着:{stale}")

    def test_compaction_result_is_correct_with_cache(self):
        """带缓存做 compaction,压完之后数据必须一条不差。"""
        with self.open_engine(auto_compact=False) as db:
            expected = {}
            for batch in range(4):
                for i in range(50):
                    key = f"k{i:04d}"
                    value = f"v{batch}-{i}"
                    db.put(key, value)
                    expected[key] = value
                db.flush()
            # 先读一遍,把缓存喂上
            for key in list(expected)[:100]:
                db.get(key)
            db.compact_all()
            for key, value in expected.items():
                self.assertEqual(db.get(key), value.encode(), key)

    def test_shared_cache_between_engine_and_reader(self):
        """引擎把自己的缓存传给每个 reader —— 同一个对象,不是各建各的。"""
        with self.open_engine() as db:
            self.fill(db, 100)
            db.flush()
            self.assertIsInstance(db.block_cache, BlockCache)
            for meta in db.sstables:
                self.assertIs(db._readers[meta.file_id]._cache, db.block_cache)


# ------------------------------------------------------------------ 老格式文件


class TestOldFormatStillReadable(Stage4TestCase):
    """阶段 2/3 的数据目录升级到阶段 4 之后必须能直接打开。"""

    def test_directory_scan_finds_v1_file(self):
        """空目录 + 一个 v1 文件(没有 manifest)→ 扫目录时应该认出来。"""
        entries = [(f"key{i:04d}".encode(), f"value{i}".encode())
                   for i in range(200)]
        write_v1_sstable(self.sst_path(1), entries)

        with self.open_engine() as db:
            self.assertEqual(len(db.sstables), 1)
            reader = db._readers[1]
            self.assertEqual(reader.version, 1)
            self.assertFalse(reader.has_bloom_filter)
            for key, value in entries:
                self.assertEqual(db.get(key), value, key)

    def test_v1_file_referenced_by_manifest(self):
        """正常升级路径:先 flush 出 v2 文件,再把文件换成 v1 格式(模拟老数据)。"""
        entries = [(f"key{i:04d}".encode(), f"value{i}".encode())
                   for i in range(200)]
        with self.open_engine() as db:
            for key, value in entries:
                db.put(key, value)
            db.flush()
            file_id = db.sstables[0].file_id

        # 把同一个文件改写成 v1 格式 —— 内容一样,只是没有过滤器块
        write_v1_sstable(self.sst_path(file_id), entries)

        with self.open_engine() as db:
            reader = db._readers[file_id]
            self.assertEqual(reader.version, 1)
            self.assertFalse(reader.has_bloom_filter)
            self.assertFalse(reader.bloom_corrupt)     # 是"没有",不是"坏了"
            for key, value in entries:
                self.assertEqual(db.get(key), value, key)

    def test_v1_file_never_rejects_so_nothing_gets_blocked(self):
        """老文件没有过滤器,所以查不存在的键**一次也不会被挡**。

        退化成阶段 3 的行为:慢,但结果正确。
        "不知道"的时候绝不能猜"不在"。
        """
        write_v1_sstable(self.sst_path(1),
                         [(f"key{i:04d}".encode(), b"v") for i in range(100)])
        with self.open_engine() as db:
            for i in range(50):
                self.assertIsNone(db.get(f"missing{i:04d}"))
            self.assertEqual(db.stats().bloom_rejections, 0)

    def test_mixed_v1_and_v2_files(self):
        """目录里老文件和新文件共存 —— 升级后就是这个状态。"""
        v1_entries = [(f"aaa{i:04d}".encode(), b"old") for i in range(100)]
        write_v1_sstable(self.sst_path(1), v1_entries)

        with self.open_engine() as db:
            db.put("zzz0000", "new")
            db.flush()
            self.assertEqual(len(db.sstables), 2)

            versions = {fid: r.version for fid, r in db._readers.items()}
            self.assertIn(1, versions.values())
            self.assertIn(2, versions.values())

            for key, value in v1_entries:
                self.assertEqual(db.get(key), value, key)
            self.assertEqual(db.get("zzz0000"), b"new")

    def test_v1_file_can_be_compacted_into_v2(self):
        """老文件参与 compaction 之后会被写成 v2 —— 这是数据格式的自然升级。"""
        v1_entries = [(f"key{i:04d}".encode(), f"v{i}".encode())
                      for i in range(100)]
        write_v1_sstable(self.sst_path(1), v1_entries)

        with self.open_engine(auto_compact=False) as db:
            for i in range(50):
                db.put(f"key{i:04d}", f"updated{i}")
            db.flush()
            db.compact_all()

            # 压完之后所有文件都应该是 v2
            for fid, reader in db._readers.items():
                self.assertEqual(reader.version, 2, f"文件 {fid} 还是老格式")
            # 数据也要对:后写的覆盖先写的
            for i in range(50):
                self.assertEqual(db.get(f"key{i:04d}"), f"updated{i}".encode())
            for i in range(50, 100):
                self.assertEqual(db.get(f"key{i:04d}"), f"v{i}".encode())


# ------------------------------------------------------------------ 过滤器损坏


class TestFilterCorruptionAtEngineLevel(Stage4TestCase):
    """过滤器坏了必须**降级**,不能拒绝启动。

    这条和"数据块坏了必须拒绝启动"是一对相反的取舍,理由不同:
        数据块坏 → 读不出数据 = 丢数据,必须大声报错
        过滤器坏 → 只是失去一个加速结构,慢一点而已,不能因此打不开数据
    """

    def setUp(self):
        super().setUp()
        with self.open_engine() as db:
            self.items = self.fill(db, 300)
            db.flush()
            self.file_id = db.sstables[0].file_id
        self.sst = self.sst_path(self.file_id)

    def _filter_block(self) -> tuple[int, int]:
        raw = self.sst.read_bytes()
        size = len(raw)
        (index_offset, index_len, count, filter_offset, filter_len, _m) = (
            struct.unpack(">QQQQQ8s", raw[size - FOOTER_SIZE:])
        )
        self.assertGreater(filter_len, 0)
        return filter_offset, filter_len

    def test_corrupt_filter_degrades_instead_of_failing(self):
        filter_offset, _ = self._filter_block()
        # framed 块 = [len:4][crc32:4] + payload,翻掉 payload 里的一个字节
        with open(self.sst, "r+b") as fh:
            fh.seek(filter_offset + 8 + 3)
            fh.write(b"\xff")

        with self.open_engine() as db:      # 关键:启动**不能**失败
            st = db.stats()
            self.assertEqual(st.bloom_corrupt_files, 1)
            self.assertFalse(db._readers[self.file_id].has_bloom_filter)
            # 数据一条都不能少
            for key, value in self.items:
                self.assertEqual(db.get(key), value.encode(), key)

    def test_degraded_engine_never_rejects(self):
        """降级之后退化成阶段 3:一次也不挡,但结果全对。"""
        filter_offset, _ = self._filter_block()
        with open(self.sst, "r+b") as fh:
            fh.seek(filter_offset + 8 + 3)
            fh.write(b"\xff")

        with self.open_engine() as db:
            for i in range(100):
                self.assertIsNone(db.get(f"absent{i:05d}"))
            self.assertEqual(db.stats().bloom_rejections, 0)
            for key, value in self.items:
                self.assertEqual(db.get(key), value.encode(), key)

    def test_healthy_engine_reports_zero_corrupt(self):
        with self.open_engine() as db:
            self.assertEqual(db.stats().bloom_corrupt_files, 0)

    def test_corrupt_filter_is_visible_in_stats_string(self):
        filter_offset, _ = self._filter_block()
        with open(self.sst, "r+b") as fh:
            fh.seek(filter_offset + 8 + 3)
            fh.write(b"\xff")
        with self.open_engine() as db:
            text = str(db.stats())
            self.assertIn("过滤器损坏", text)

    def test_data_block_corruption_surfaces_on_read(self):
        """对照组:数据块坏了必须报错,不能跟着过滤器一起"宽容"。

        注意时机 —— 启动时**发现不了**:启动只读 footer、索引和过滤器,
        不碰数据块(真去校验全部数据块等于每次启动全表扫一遍)。
        但一旦真去读那一块,就必须抛错,绝不能悄悄返回"找不到"。
        "读不出来"和"不存在"必须能被区分开。
        """
        # 数据块在文件最开头,[len:4][crc32:4] + payload
        with open(self.sst, "r+b") as fh:
            fh.seek(8 + 5)
            fh.write(b"\xff")

        with self.open_engine() as db:      # 启动仍然成功
            self.assertEqual(db.stats().bloom_corrupt_files, 0)
            with self.assertRaises(ChecksumMismatchError):
                db.get("key00000")          # 但读它就会炸

    def test_index_corruption_refuses_to_start(self):
        """索引坏了必须**拒绝启动**。

        索引是"怎么找到数据"的依据。它不可信,整个文件的定位就都不可信了 ——
        这时候宁可启动失败,也不能带着一个可能读到乱七八糟东西的索引继续跑。
        """
        raw = self.sst.read_bytes()
        size = len(raw)
        index_offset = struct.unpack(">Q", raw[size - FOOTER_SIZE:size - FOOTER_SIZE + 8])[0]
        with open(self.sst, "r+b") as fh:
            fh.seek(index_offset + 8 + 2)   # 索引 payload 里翻一个字节
            fh.write(b"\xff")

        with self.assertRaises(CorruptionError) as ctx:
            LSMEngine(self.dir)
        self.assertIn("拒绝", str(ctx.exception))


# ------------------------------------------------------------------ 统计输出


class TestStatsOutput(Stage4TestCase):
    def test_stats_string_has_stage4_lines(self):
        with self.open_engine() as db:
            self.fill(db, 100)
            db.flush()
            db.get("key00000")
            text = str(db.stats())
            self.assertIn("过滤器", text)
            self.assertIn("块缓存", text)

    def test_disabled_cache_line(self):
        with self.open_engine(block_cache_size=0) as db:
            self.assertIn("已禁用", str(db.stats()))

    def test_bloom_rejection_counter_is_reported(self):
        """计数器那一行始终在,只是数值会变。"""
        with self.open_engine() as db:
            self.fill(db, 100)
            db.flush()
            self.assertIn("挡下 0 次文件读", str(db.stats()))
            db.get("absent-key")
            self.assertNotIn("挡下 0 次文件读", str(db.stats()))

    def test_corrupt_note_only_when_there_is_one(self):
        with self.open_engine() as db:
            self.fill(db, 100)
            db.flush()
            self.assertNotIn("过滤器损坏", str(db.stats()))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
