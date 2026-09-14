# mini-lsm

从零实现的 **LSM-Tree 存储引擎**，纯 Python、零第三方依赖。

这个项目不是把某个现成库包装一下 —— 而是把 LSM-Tree 的每一层
（WAL、内存表、SSTable、Compaction）都手写出来，磁盘格式、CRC 校验、
崩溃恢复语义全部自己定义。目标是把"数据库到底怎么保证数据不丢"
这件事讲清楚。

> **当前进度：阶段 2 / 5 已完成**
> 已有能力：预写日志（WAL）+ 内存有序表（MemTable）+ **SSTable 落盘** + 崩溃恢复
> 还没有：Compaction、Bloom Filter、MVCC

---

## 为什么值得写这个项目

大多数人理解的"存储"是 `open()` + `write()`。但真实的存储引擎要回答一堆
更难的问题：

- 进程在 `write()` 写到一半时被 `kill -9`，重启后怎么知道哪条数据是完整的？
- 删除一个键，为什么不能真的把它抹掉？
- 为什么"先把数据写进内存、再写日志"是错的？
- 数据比内存还大时怎么办？重启要把几百 GB 重放一遍吗？
- 磁盘上一个字节被翻转了，怎么发现？

这个项目用约 2000 行 Python 把这些问题逐一解决掉。

---

## 快速开始

不需要安装任何东西，Python 3.10+ 即可。

```bash
# 跑演示：读写 → 刷盘 → 重启 → 数据超过内存 → 墓碑遮蔽 → 崩溃恢复
python demo.py

# 跑测试
python -m unittest discover -s tests
```

作为库使用：

```python
import sys
sys.path.insert(0, "src")

from mini_lsm import LSMEngine

with LSMEngine("./mydata") as db:
    db.put("name", "alice")
    db.put("city", "深圳")
    print(db.get_str("name"))        # alice

    db.delete("city")
    print(db.get("city"))            # None

    for key, value in db.scan(b"a", b"z"):
        print(key, value)
```

数据目录里是 `wal.log` 和若干 `sst-000001.sst`。内存表写满后自动刷成
SSTable，所以数据量可以远超内存。

---

## 架构

```
                    写入路径（顺序不能反）
    put(k, v) ──► ① 追加到 WAL（落盘，持久性来源）
                    │
                    └─► ② 写入 MemTable（内存，快速生效）
                              │
                              │  满了（默认 4 MiB）
                              ▼
                        ③ flush：整体写成 SSTable
                              │  原子改名 + fsync
                              ▼
                        ④ 截断 WAL（数据已经在新地方了）

                    读取路径（从新到旧，命中即止）
    get(k) ──► MemTable ──miss──► L0 最新的 SSTable
                                    │ miss
                                    ▼
                                  L0 更旧的 SSTable → ...
                                    │
                              墓碑也会终止查找

                    范围扫描
    scan() ──► MergingIterator(内存表, sst_N, ..., sst_1)
                    → 一条有序流，同 key 取最新
```

### 代码结构

| 文件 | 职责 | 行数 |
|------|------|------|
| `src/mini_lsm/errors.py` | 异常体系，区分「数据损坏」与「用法错误」 | 56 |
| `src/mini_lsm/record.py` | 记录编解码 + CRC32（WAL 和 SSTable 共用） | 297 |
| `src/mini_lsm/wal.py` | 预写日志：追加、重放、崩溃截断恢复 | 258 |
| `src/mini_lsm/memtable.py` | 内存有序表，含墓碑语义 | 179 |
| `src/mini_lsm/sstable.py` | SSTable：有序块 + 稀疏索引 + footer | 508 |
| `src/mini_lsm/iterator.py` | 多来源归并迭代器 | 97 |
| `src/mini_lsm/engine.py` | 引擎主入口，串联以上全部 | 480 |

---

## 磁盘格式

### WAL（`wal.log`）

纯顺序追加，没有索引、没有页结构 —— 因为顺序写是磁盘上最快的操作，
而 WAL 的性能决定了整个写入路径的上限。

每条记录（大端序）：

```
偏移      长度        字段
------    --------    --------------------------------------
0         4           payload_len   —— payload 的字节数
4         4           crc32         —— payload 的 CRC32 校验和
8         1           rec_type      —— 1=PUT, 2=DELETE
9         4           key_len
13        key_len     key
...       4           value_len
...       value_len   value
------    --------    --------------------------------------
合计      8 + payload_len
```

### SSTable（`sst-NNNNNN.sst`）

不可变的有序磁盘表。文件布局：

```
+---------------------------+
| 数据块 0                  |   多条 entry，按 key 升序
+---------------------------+   block := [len:4][crc32:4] + entries
| 数据块 1                  |
+---------------------------+
| ...                       |
+---------------------------+
| 索引块                    |   稀疏索引：每块的**起始 key** → (偏移, 长度)
+---------------------------+   同样带 [len:4][crc32:4]
| footer（固定 32 字节）    |   索引偏移 + 索引长度 + 条目数 + magic
+---------------------------+
```

entry 的字节布局和 WAL 的 payload **完全一致**：

```
rec_type(1) + key_len(4) + key + value_len(4) + value
```

两者共用同一套编解码 —— 少一套格式就少一处会写错的地方。

---

## 四个关键设计

### 1. 为什么 `payload_len` 放在最前面？

读取时先拿到长度，才知道后面要读多少字节。这样即使文件被截断，
也能**立刻**判断出"这条记录不完整"，而不是读到一半才发现。
崩溃恢复的正确性直接依赖这一点。

### 2. 为什么 CRC 按块算而不是按条算？

SSTable 的读取粒度就是块 —— 要么整块读进来用，要么根本不读。
按块校验和按条校验能发现的问题一样多，但校验开销只有 1/N。

另外，CRC 能区分两种损坏，这个区分很有用：

- **内容变了但结构还在** → 跳过 CRC 还能读出数据（诊断工具有用）
- **结构已经不可信**（比如长度字段被改）→ 即使跳过 CRC 也只能报截断

后者正是 CRC 存在的意义。

### 3. 为什么是"稀疏"索引？

每 4 KiB 一个索引项，而不是每个 key 一个。索引常驻内存，必须小 ——
1 亿条记录如果逐条建索引，光索引就要好几 GB。稀疏索引让索引大小
正比于"**块的个数**"而不是"记录的个数"，代价是查一个 key 需要在
块内顺序扫一遍（块只有 4 KiB，可以忽略）。

### 4. 为什么用"临时文件 + 原子改名"写 SSTable？

因为"文件名对得上"就等于"内容完整"。崩溃时只会留下一个 `.tmp`，
启动时直接删掉即可 —— 数据还在 WAL 里，不会丢。

这让启动逻辑简单到几乎没有出错空间：**扫目录，按编号排序，打开**。
不需要 manifest。（阶段 3 有了 compaction，需要成批原子地增删文件，
那时才会引入 manifest。）

---

## 崩溃恢复

进程可能在**写记录的任意时刻**被杀死，日志末尾就会留下半条记录。

恢复策略只有一条规则：

> 从头顺序读，读到损坏就**截断到上一条完好记录**，保留之前的一切。

绝不能因为末尾几个坏字节就把整个日志判死刑。

`WAL.recover()` 会重放日志并把文件**真的**截断到有效边界，
所以函数返回后，日志里每个字节都是可解析的有效数据。

实际跑出来的效果（`python demo.py`）：

```
--- 7. 崩溃恢复:WAL 尾部残缺 ---
  WAL 从 85 字节被砍到 78 字节
  user:01 = alice
  user:02 = bob
  user:03 = None  (半截记录,被丢弃)
  启动恢复:重放 2 条记录(含截断)
  截断原因: payload 不完整:期望 21 字节,实际 14 字节(偏移 56)
```

### 刷盘时的步骤顺序

`flush()` 里**顺序是唯一重要的东西**：

```
1. 把内存表快照写成 SSTable 并 fsync   ← 数据先落到新地方
2. 原子改名成正式文件
3. 把新表插到 L0 最前面（内存里）
4. 换一个空的内存表
5. 截断 WAL                            ← 最后才丢弃旧地方
```

任何一步之后崩溃都是安全的：只要 WAL 还在，重放一遍就能重建内存表。
重放是**幂等**的 —— 同样的 put 应用两次，结果一样。

---

## 三条必须遵守的规则

**1. 先写日志，再改内存**

```python
def put(self, key, value):
    self._wal.append(RecordType.PUT, kb, vb)   # ① 先落盘
    self._memtable.put(kb, vb)                 # ② 再改内存
```

顺序反了的话，"日志还没写完就崩溃"的场景下，内存里的改动会丢失，
而调用方已经收到"写成功"了 —— 这直接违反持久性。

**2. 删除必须留墓碑，而且墓碑要能跨层压制**

删除不能真的把键从表里抹掉。因为**更旧的版本可能躺在某个 SSTable 里**，
抹掉墓碑之后，那个旧值就会"复活"。

所以 `get()` 遇到墓碑时返回 `(True, None)`，而不是 `(False, None)` ——
`True` 表示"我确定这个键被删了"，于是**停止**往更旧的文件里找。

这是最容易写错、后果最严重的地方。测试里专门有一组：

```
test_tombstone_shadows_value_in_older_sstable
test_tombstone_in_memtable_shadows_sstable
test_delete_only_in_oldest_layer
```

**3. 删除不存在的键也要写墓碑**

听起来反直觉，但同样是因为"旧版本可能躺在磁盘上"。
如果跳过墓碑，那次删除就丢了。

---

## 内存表（MemTable）

LSM 里唯一**可写**的数据结构。所有写入先落到这里，攒够了再整体刷成
不可变的 SSTable。

**为什么强调"有序"？** 刷盘时希望写出的 SSTable 内部有序 ——
这样查询才能二分查找，范围扫描才能顺序读。实现上用 `dict` 存储 +
遍历时 `sorted()`，兼顾 O(1) 查找和有序输出。

**墓碑用 `None` 表示。** 于是 `get()` 返回 `None` 有两种含义：
键不存在，或者键已被删除。对调用方而言这是同一件事。
需要区分时用 `get_entry()`：

```python
mt.put(b"alive", b"v")
mt.put(b"dead", b"v")
mt.delete(b"dead")

mt.get_entry(b"alive")     # (True, b'v')
mt.get_entry(b"dead")      # (True, None)   ← 墓碑
mt.get_entry(b"never")     # (False, None)  ← 从未出现
```

顺带一个容易忽略的推论：**同一代内存表里 put 完又 delete 同一个键，
只会留下一条墓碑** —— 内存表是个 map，旧值被直接覆盖，
刷盘时不可能把已经作废的值写出去。

---

## 归并迭代器

一个 key 可能同时存在于内存表和好几个 SSTable 里。`MergingIterator`
让"多个来源"对上层看起来像"一个有序表" —— 扫描和将来的 compaction
都靠它，不用各自写一遍合并逻辑。

契约有三条：

1. 每个来源产出 `(key, value)`，按 key 升序
2. `value is None` 表示墓碑
3. **来源的先后顺序就是新旧顺序**：下标 0 最新

实现是标准 k 路归并，堆元素是 `(key, source_index, seq, value)`。
那个 `seq` 序号不是装饰：`bytes` 和 `None` 没法比大小，一旦堆比较
走到 value 上就会抛 `TypeError`。同 key 且同来源下标时靠 `seq` 决胜，
彻底避免比较 value。

墓碑会被**照常产出** —— 过滤是调用方的事。面向用户的 `scan()` 跳过它们，
但 compaction 必须看到，否则"删除"这个信息就丢了。

---

## 测试

**230 个测试，全部通过。**

```bash
$ python -m unittest discover -s tests
Ran 230 tests in 0.794s
OK
```

| 测试文件 | 覆盖内容 | 用例数 |
|----------|----------|--------|
| `test_record.py` | 编解码往返、边界值、CRC、截断定位 | 23 |
| `test_wal.py` | 追加、重放、崩溃截断、生命周期 | 21 |
| `test_memtable.py` | 读写、墓碑语义、有序性、容量 | 31 |
| `test_iterator.py` | 归并、同 key 取最新、墓碑、惰性 | 23 |
| `test_sstable.py` | 块/索引/footer、二分查找、损坏检测 | 45 |
| `test_engine.py` | CRUD、扫描、重启恢复、崩溃恢复 | 48 |
| `test_engine_sstable.py` | 刷盘、跨层遮蔽、多来源读、启动健壮性 | 39 |
| **合计** | | **230** |

几个刻意写得比较刁钻的用例：

- **`test_tombstone_shadows_value_in_older_sstable`** —— 保证删除不会跨层失效
- **`test_restart_does_not_replay_flushed_data`** —— 钉住阶段 2 的核心收益
- **`test_data_exceeds_memtable_capacity`** —— 2000 条数据 + 4 KiB 内存表
- **`test_read_block_rejects_length_beyond_eof`** —— 见下方"踩过的坑"
- **`test_corrupted_sstable_prevents_startup`** —— 宁可拒绝启动也不带着坏文件跑
- **`test_put_then_delete_same_key_collapses`** —— 固化内存表 map 语义
- **`test_writes_are_durable_without_close`** —— 持久性不依赖优雅关闭
- **`test_scan_snapshot_is_stable_during_iteration`** —— 迭代期间写入不影响结果

---

## 踩过的坑

**读块之前必须先校验长度。** 最早的 `_read_block` 是先 `read(length)`
再检查"读到的够不够"。测试用一个 1 TiB 的长度去打它，结果不是抛出
`CorruptionError`，而是 `MemoryError` —— 因为 `read(1 << 40)` 会先尝试
分配 1 TiB 内存，进程当场就没了。

这不是理论问题：块长度最终来自磁盘上的索引，损坏的数据可以让它变成
任何数字。**先核对边界，再申请内存。**

顺带也发现了一个好性质：索引块整体有 CRC，所以被篡改的索引会在
**打开阶段**就被拦住，伪造的块偏移根本没有机会被使用。
`_read_block` 里的长度校验于是成了纵深防御的第二道。

---

## 路线图

| 阶段 | 内容 | 状态 |
|------|------|------|
| **1** | **WAL + MemTable + 崩溃恢复** | **已完成** |
| **2** | **SSTable 刷盘（有序块 + 稀疏索引）+ 多来源读路径** | **已完成** |
| 3 | Compaction：L0 → L1 → L2 分层归并，清理墓碑 | 待开始 |
| 4 | Bloom Filter + Block Index，减少无谓磁盘读 | 待开始 |
| 5 | 范围扫描流式迭代器 + MVCC 快照读 | 待开始 |

### 现在的能力边界

阶段 2 解除了两个限制，但还有两个明显的问题 —— 它们正是阶段 3、4 要做的：

- **没有 compaction**：删掉的键只留下墓碑，SSTable 只增不减。
  `demo.py` 里 8 KiB 的内存表写 3000 条记录就刷出了 **49 个 SSTable**，
  查一个 key 最坏要翻 49 个文件。
- **没有 Bloom Filter**：查一个**不存在**的 key 必须把每个 SSTable 都查一遍。

其他已知限制：

- 每个 SSTable 常驻一个文件句柄，文件多了会吃掉句柄数
- 没有并发控制，`put`/`get` 用一把粗锁串行化
- 没有事务，`put_many` 中途失败会留下部分写入
- `scan()` 会把结果先物化成列表，扫大库占内存（阶段 5 改成流式）

---

## 设计取舍

**为什么 WAL 默认不 fsync？**

`sync_on_write=False` 时能扛住**进程崩溃**（数据在 OS 页缓存里，
进程死了但内核还在），但断电可能丢最近几条。`True` 才扛得住断电，
但慢一个数量级。默认选 False 是因为大多数场景下"进程崩溃不丢数据"
已经够了，断电保护可以按需开启：

```python
db = LSMEngine("./data", wal_sync_on_write=True)
```

**为什么关引擎时不自动刷盘？**

那样每次关闭都会留下一个很小的 SSTable。内存表的数据由 WAL 保证，
下次启动重放即可 —— 让"关闭"保持廉价。

**为什么写句柄延迟打开？**

`WAL.__init__` 不创建文件、不占句柄，只有真正 `append` 时才打开。
因为"只想重放看看日志里有什么"是很常见的用法（`read_records`、
诊断工具），构造时就打开写句柄会让这种用法泄漏文件句柄。

**为什么异常要分层？**

```
LSMError
├── InvalidArgumentError    用法错误 —— 直接报给用户，不该重试
├── ClosedError             对已关闭的引擎操作
└── CorruptionError         数据损坏 —— 恢复路径上应被捕获并截断
    ├── TruncatedRecordError      写入中途崩溃留下的残骸（预期内）
    └── ChecksumMismatchError     磁盘数据被改写或位翻转
```

让调用方能精确区分"数据坏了"（截断恢复）和"你用法错了"（直接报错），
而不是笼统地 `except Exception`。

---

## 环境

- Python 3.10+（用到了 `X | Y` 类型语法和 `dataclass`）
- 零第三方依赖
- 在 Python 3.13 / Windows 上开发和测试

## 许可

MIT
