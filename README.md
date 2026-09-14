# mini-lsm

从零实现的 **LSM-Tree 存储引擎**，纯 Python、零第三方依赖。

这个项目不是把某个现成库包装一下 —— 而是把 LSM-Tree 的每一层
（WAL、内存表、SSTable、Compaction）都手写出来，磁盘格式、CRC 校验、
崩溃恢复语义全部自己定义。目标是把"数据库到底怎么保证数据不丢"
这件事讲清楚。

> **当前进度：阶段 1 / 5 已完成**
> 已有能力：预写日志（WAL）+ 内存有序表（MemTable）+ 崩溃恢复
> 还没有：SSTable 落盘、Compaction、MVCC

---

## 为什么值得写这个项目

大多数人理解的"存储"是 `open()` + `write()`。但真实的存储引擎要回答一堆
更难的问题：

- 进程在 `write()` 写到一半时被 `kill -9`，重启后怎么知道哪条数据是完整的？
- 删除一个键，为什么不能真的把它抹掉？
- 为什么"先把数据写进内存、再写日志"是错的？
- 磁盘上一个字节被翻转了，怎么发现？

这个项目用大约 1000 行 Python（不含测试）把这些问题逐一解决掉。

---

## 快速开始

不需要安装任何东西，Python 3.10+ 即可。

```bash
# 跑演示：基本读写 → 重启恢复 → 模拟崩溃
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

数据目录里只有一个 `wal.log`。关掉进程再打开，数据还在。

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
                       【阶段 2】刷成 SSTable

                    读取路径
    get(k) ──► MemTable ──miss──► 【阶段 2】L0 → L1 → L2 ...
                  │
              命中即返回（墓碑返回 None）
```

### 代码结构

| 文件 | 职责 | 行数 |
|------|------|------|
| `src/mini_lsm/errors.py` | 异常体系，区分「数据损坏」与「用法错误」 | 56 |
| `src/mini_lsm/record.py` | WAL 记录的二进制编解码 + CRC32 | 210 |
| `src/mini_lsm/wal.py` | 预写日志：追加、重放、崩溃截断恢复 | 244 |
| `src/mini_lsm/memtable.py` | 内存有序表，含墓碑语义 | 179 |
| `src/mini_lsm/engine.py` | 引擎主入口，串联 WAL 与 MemTable | 297 |

---

## 磁盘格式

阶段 1 只有一种文件：`wal.log`。它是**纯顺序追加**的，没有索引，
没有页结构 —— 因为顺序写是磁盘上最快的操作，而 WAL 的性能决定了
整个写入路径的上限。

每条记录的格式（大端序）：

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

### 两个关键设计

**为什么把 `payload_len` 放在最前面？**

读取时先拿到长度，才知道后面要读多少字节。这样即使文件被截断，
也能**立刻**判断出"这条记录不完整"，而不是读到一半才发现。
崩溃恢复的正确性直接依赖这一点。

**为什么 CRC 覆盖整个 payload 而不是分字段校验？**

简单且够用。分字段校验能定位到具体出错字段，但 WAL 场景下
"这条记录坏了"已经足够 —— 处理方式都是截断到上一条好记录。

顺便说一个测试里暴露出来的细节：CRC 能区分两种损坏。

- **内容变了但结构还在** → 跳过 CRC 还能读出数据（诊断工具有用）
- **结构已经不可信**（比如长度字段被改）→ 即使跳过 CRC 也只能报截断

后者正是 CRC 存在的意义。

---

## 崩溃恢复：这个项目最关键的部分

进程可能在**写记录的任意时刻**被杀死，日志末尾就会留下半条记录。

恢复策略只有一条规则：

> 从头顺序读，读到损坏就**截断到上一条完好记录**，保留之前的一切。

绝不能因为末尾几个坏字节就把整个日志判死刑。

`WAL.recover()` 会重放日志并把文件**真的**截断到有效边界，
所以函数返回后，日志里每个字节都是可解析的有效数据。

实际跑出来的效果（`python demo.py`）：

```
--- 4. 模拟写入中途崩溃(WAL 尾部残缺) ---
  WAL 从 85 字节被砍到 78 字节
  user:01 = alice
  user:02 = bob
  user:03 = None  (半截记录,被丢弃)
  启动恢复:重放 2 条记录(含截断)
  截断原因: payload 不完整:期望 21 字节,实际 14 字节(偏移 56)
  修复后 WAL 大小: 56 字节

--- 5. 恢复之后继续写入 ---
  现在 WAL 里有 4 条记录,损坏标记 = False
  键: ['user:01', 'user:02', 'user:04', 'user:05']
```

注意最后一行：恢复后引擎**必须还能继续写**。否则这个库就废了。

### 三条必须遵守的规则

**1. 先写日志，再改内存**

```python
def put(self, key, value):
    self._wal.append(RecordType.PUT, kb, vb)   # ① 先落盘
    self._memtable.put(kb, vb)                 # ② 再改内存
```

顺序反了的话，"日志还没写完就崩溃"的场景下，内存里的改动会丢失，
而调用方已经收到"写成功"了 —— 这直接违反持久性。

**2. 删除必须留墓碑**

删除不能真的把键从表里抹掉。因为**更旧的版本可能躺在某个 SSTable 里**，
抹掉墓碑之后，那个旧值就会"复活"。

```python
def delete(self, key):
    self._wal.append(RecordType.DELETE, kb)    # 写墓碑
    self._memtable.delete(kb)                  # 表里 value 置 None
```

**3. 删除不存在的键也要写墓碑**

听起来反直觉，但同样是因为"旧版本可能躺在磁盘上"。
如果跳过墓碑，那次删除就丢了。

---

## 内存表（MemTable）

LSM 里唯一**可写**的数据结构。所有写入先落到这里，攒够了再整体刷成
不可变的 SSTable。

**为什么强调"有序"？** 阶段 2 刷盘时，希望写出的 SSTable 内部有序 ——
这样查询才能二分查找，范围扫描才能顺序读。如果内存表本身无序，
刷盘时还得额外排序。实现上用 `dict` 存储 + 遍历时 `sorted()`，
兼顾 O(1) 查找和有序输出。

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

内存占用是**估算**的（`len(key) + len(value) + 64`）。不精确，
但足够用来判断"该刷盘了"。

---

## 测试

**122 个测试，全部通过。**

```bash
$ python -m unittest discover -s tests
Ran 122 tests in 0.127s
OK
```

| 测试文件 | 覆盖内容 | 用例数 |
|----------|----------|--------|
| `test_record.py` | 编解码往返、边界值、CRC、截断定位 | 23 |
| `test_wal.py` | 追加、重放、崩溃截断、生命周期 | 21 |
| `test_memtable.py` | 读写、墓碑语义、有序性、容量 | 31 |
| `test_engine.py` | CRUD、扫描、重启恢复、崩溃恢复、生命周期 | 47 |
| **合计** | | **122** |

几个刻意写得比较刁钻的用例：

- **`test_delete_leaves_tombstone_in_table`** —— 保证墓碑没被优化掉
- **`test_delete_missing_key_still_creates_tombstone`** —— 上面说的反直觉规则
- **`test_writes_are_durable_without_close`** —— 不调 `close()` 直接重开，验证持久性不依赖优雅关闭
- **`test_engine_can_write_after_recovery`** —— 恢复后还能继续写
- **`test_structural_corruption_survives_crc_bypass`** —— 固化"CRC 关掉也救不了结构性损坏"这个行为
- **`test_scan_snapshot_is_stable_during_iteration`** —— 迭代期间写入不影响已取到的结果

---

## 路线图

| 阶段 | 内容 | 状态 |
|------|------|------|
| **1** | **WAL + MemTable + 崩溃恢复** | **已完成** |
| 2 | MemTable 刷成 SSTable（有序块 + 稀疏索引），读路径合并多来源 | 待开始 |
| 3 | Compaction：L0 → L1 → L2 分层归并，清理墓碑 | 待开始 |
| 4 | Bloom Filter + Block Index，减少无谓磁盘读 | 待开始 |
| 5 | 范围扫描迭代器 + MVCC 快照读 | 待开始 |

### 阶段 1 的能力边界

说清楚免得误解 —— **这还不是一个能用的数据库**：

- 所有数据同时存在于内存表和 WAL 中，**数据量不能超过内存**
- 重启时要把**整个 WAL 重放一遍**，启动时间随数据量线性增长
- 没有并发控制，`put`/`get` 用一把粗锁串行化
- 没有事务，`put_many` 中途失败会留下部分写入

这两个限制（内存上限 + 启动变慢）会在阶段 2 引入 SSTable 之后解除。
这正是一个真实的 LSM 引擎必须要有 SSTable 的原因。

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
