"""布隆过滤器(Bloom Filter)—— 用极小的内存回答"这个 key 一定不在吗"。

要解决的问题:
    阶段 3 之后,查一个**不存在**的 key 仍然要逐层真的去读文件 ——
    因为只有读到文件内容,才知道它里面没有这个 key。实测 3000 条记录时
    每次点查平均读 1.5 个文件,其中"查不存在的 key"贡献了绝大部分。
    文件越多、层越深,这个开销越大。

布隆过滤器的能力边界(必须说清楚,不然会用错):
    - ``might_contain(key) is False`` → **一定不在**,可以放心跳过这个文件
    - ``might_contain(key) is True``  → **可能在**,还得真去读

    也就是说它**只有假阳性,没有假阴性**。这个不对称是有意的:
    假阳性只是"白读一次文件",慢一点;假阴性会直接**丢数据**
    (键明明在,却被告知不在)。所以任何实现上的取舍都必须倒向
    "宁可说在"。

⚠️ 一个必须踩对的坑:哈希函数不能用内置的 ``hash()``
    CPython 对 ``bytes`` 的 ``hash()`` 带**每进程随机化**的盐
    (``PYTHONHASHSEED``,出于防哈希碰撞攻击的考虑)。而布隆过滤器是
    **要落盘**的 —— 进程 A 写下的过滤器,进程 B 重启后要用同一套
    位位置去查。如果哈希随进程变化,那么重启之后所有查询都会落在
    错误的位上,于是**已经存在的键被报告为不存在** —— 这正是
    最不能出现的那种 bug。

    所以这里用 ``hashlib.blake2b``:它是标准库、跨平台跨版本稳定、
    分布均匀,而且一次调用就能拿到 128 位。
"""

from __future__ import annotations

import hashlib
import math
import struct
from array import array
from typing import Iterable

from .errors import CorruptionError, InvalidArgumentError

__all__ = [
    "DEFAULT_BITS_PER_KEY",
    "MIN_BITS",
    "MAX_K",
    "BloomFilter",
    "BloomBuilder",
    "estimate_false_positive_rate",
]

#: 每个 key 分到多少 bit。10 bit/key 对应约 1% 的假阳性率 —— 这是
#: 布隆过滤器的经典默认值:再往上加 bit,收益递减而内存线性增长。
DEFAULT_BITS_PER_KEY = 10

#: 过滤器最少这么多 bit。太小的话一个 key 就把位图占满了,
#: 假阳性率会退化成"永远返回在"。
MIN_BITS = 64

#: 哈希函数个数的上限。k 超过这个值之后收益不再增长
#: (最优 k = bits_per_key * ln2,10 bit/key 时约等于 7)。
MAX_K = 30

#: 序列化头部:key 个数(4) + 位数(4) + 哈希个数(1)
_HEADER = struct.Struct(">IIB")
_HEADER_SIZE = _HEADER.size

#: 用 blake2b 取 128 位,前 64 位做 h1,后 64 位做 h2
_DIGEST_SIZE = 16


def _hash_key(key: bytes) -> tuple[int, int]:
    """把 key 映射成两个 64 位无符号整数 ``(h1, h2)``。

    为什么是 blake2b 而不是内置 ``hash()``:见模块开头的说明 ——
    ``hash()`` 每个进程的盐都不同,落盘之后重启就失效了。
    """
    digest = hashlib.blake2b(key, digest_size=_DIGEST_SIZE).digest()
    h1 = int.from_bytes(digest[:8], "big")
    h2 = int.from_bytes(digest[8:], "big")
    # 让 h2 是奇数:这样它和任何 2 的幂的位数都互质,
    # "h1 + i*h2" 生成的序列能均匀铺满整个位图,不会在小范围内打转。
    # (这是 Kirsch-Mitzenmacher 双哈希法的标准做法)
    return h1, h2 | 1


def optimal_k(bits_per_key: int) -> int:
    """由 bits/key 推出最优哈希个数 k = bits_per_key * ln2。"""
    return max(1, min(MAX_K, round(bits_per_key * math.log(2))))


def bits_for_keys(num_keys: int, bits_per_key: int) -> int:
    """按 key 个数算需要多少 bit(向上取整到 8 的倍数,方便按字节存)。"""
    wanted = max(MIN_BITS, num_keys * bits_per_key)
    return ((wanted + 7) // 8) * 8


def estimate_false_positive_rate(num_keys: int, num_bits: int, k: int) -> float:
    """理论假阳性率 ``(1 - e^(-kn/m))^k``。

    这个公式假设各位之间独立,是个近似值;但和实测值足够接近,
    测试里直接拿它当预期值用。
    """
    if num_bits <= 0 or num_keys <= 0 or k <= 0:
        return 0.0
    return (1.0 - math.exp(-k * num_keys / num_bits)) ** k


class BloomFilter:
    """不可变的位图 + 一组哈希位置。

    构造请用 :meth:`build` 或 :class:`BloomBuilder`;从磁盘恢复用
    :meth:`from_bytes`。
    """

    __slots__ = ("_bits", "_num_bits", "_k", "_num_keys")

    def __init__(self, bits: bytes, num_bits: int, k: int, num_keys: int) -> None:
        self._bits = bits
        self._num_bits = num_bits
        self._k = k
        self._num_keys = num_keys

    # ------------------------------------------------------------ 查询

    def might_contain(self, key: bytes) -> bool:
        """``False`` 表示**一定不在**;``True`` 表示"可能在,得真去读"。

        绝不会有假阴性 —— 这是整个设计的基石。加进去的每一个 key
        在这里都必然返回 ``True``。
        """
        if self._num_bits <= 0:
            return False
        h1, h2 = _hash_key(key)
        num_bits = self._num_bits
        bits = self._bits
        for i in range(self._k):
            pos = (h1 + i * h2) % num_bits
            if not (bits[pos >> 3] & (1 << (pos & 7))):
                return False        # 只要有一位是 0,就一定没加过
        return True

    def __contains__(self, key: bytes) -> bool:
        return self.might_contain(key)

    # ------------------------------------------------------------ 属性

    @property
    def num_keys(self) -> int:
        """构建时加进去的 key 个数。"""
        return self._num_keys

    @property
    def num_bits(self) -> int:
        return self._num_bits

    @property
    def k(self) -> int:
        """哈希函数个数。"""
        return self._k

    @property
    def size(self) -> int:
        """序列化之后的字节数。"""
        return _HEADER_SIZE + len(self._bits)

    @property
    def bit_density(self) -> float:
        """被置为 1 的位的比例。约 50% 时过滤器处于最佳工作点。"""
        if self._num_bits <= 0:
            return 0.0
        return sum(bin(byte).count("1") for byte in self._bits) / self._num_bits

    @property
    def false_positive_rate(self) -> float:
        """按当前参数估算的假阳性率。"""
        return estimate_false_positive_rate(self._num_keys, self._num_bits, self._k)

    # ------------------------------------------------------------ 序列化

    def to_bytes(self) -> bytes:
        """序列化成 ``[key个数:4][位数:4][k:1][位图]``。"""
        return _HEADER.pack(self._num_keys, self._num_bits, self._k) + self._bits

    @classmethod
    def from_bytes(cls, raw: bytes) -> "BloomFilter":
        """从磁盘读回来。

        ⚠️ 校验顺序:先核对长度够不够,再按声明的位数算需要多少字节 ——
        绝不能拿一个损坏的 ``num_bits`` 直接去切片或分配。
        (和 ``sstable._read_block`` 里"先核对边界再申请内存"是同一条教训)
        """
        if len(raw) < _HEADER_SIZE:
            raise CorruptionError(
                f"布隆过滤器头部不完整:期望 {_HEADER_SIZE} 字节,实际 {len(raw)} 字节"
            )

        num_keys, num_bits, k = _HEADER.unpack_from(raw, 0)
        if num_bits <= 0:
            raise CorruptionError(f"布隆过滤器的位数非法:{num_bits}")
        if not 1 <= k <= MAX_K:
            raise CorruptionError(f"布隆过滤器的哈希个数非法:{k}")

        expected = _HEADER_SIZE + (num_bits + 7) // 8
        if len(raw) != expected:
            raise CorruptionError(
                f"布隆过滤器长度对不上:声明的位数 {num_bits} 需要 {expected} 字节,"
                f"实际 {len(raw)} 字节"
            )

        return cls(raw[_HEADER_SIZE:], num_bits, k, num_keys)

    # ------------------------------------------------------------ 构建

    @classmethod
    def build(
        cls, keys: Iterable[bytes], bits_per_key: int = DEFAULT_BITS_PER_KEY
    ) -> "BloomFilter":
        """一次拿到全部 key 时用它;流式场景用 :class:`BloomBuilder`。"""
        builder = BloomBuilder(bits_per_key)
        for key in keys:
            builder.add(key)
        return builder.build()

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"<BloomFilter keys={self._num_keys} bits={self._num_bits} k={self._k} "
            f"fp≈{self.false_positive_rate:.4%}>"
        )


class BloomBuilder:
    """流式构建布隆过滤器。

    为什么需要它:SSTable 是**流式**写出来的 —— 边读内存表边写数据块,
    写到末尾才知道总共有多少 key。而位图的位数 ``num_bits`` 依赖 key 总数,
    所以不能一上来就分配。

    做法是:每来一个 key 只留下它的两个 8 字节哈希(共 16 字节),
    等全部 add 完再按实际 key 数分配位图、统一置位。

    为什么不直接缓存 key 本身:
        缓存 key 要存下整个 key(通常十几到几十字节)外加一个 Python
        对象的开销(约 50 字节);缓存哈希对固定 16 字节且能用
        ``array("Q")`` 紧凑存储 —— 内存省一个数量级。
    """

    __slots__ = ("_bits_per_key", "_h1", "_h2", "_count")

    def __init__(self, bits_per_key: int = DEFAULT_BITS_PER_KEY) -> None:
        if bits_per_key < 1:
            raise InvalidArgumentError("bits_per_key 至少为 1")
        self._bits_per_key = bits_per_key
        # 用 array("Q") 而不是 list:每个 64 位整数固定占 8 字节。
        # 换成 list 的话,每个元素是一个指向 Python int 对象的指针,
        # 而每个 int 对象自己还要约 28 字节 —— 内存差 4 倍以上。
        self._h1: array = array("Q")
        self._h2: array = array("Q")
        self._count = 0

    def add(self, key: bytes) -> None:
        """加一个 key。重复添加同一个 key 不影响正确性(只是浪费位)。"""
        h1, h2 = _hash_key(key)
        self._h1.append(h1)
        self._h2.append(h2)
        self._count += 1

    @property
    def num_keys(self) -> int:
        return self._count

    def build(self) -> BloomFilter:
        """分配位图并把所有 key 置位。可以重复调用(结果一致)。"""
        num_bits = bits_for_keys(self._count, self._bits_per_key)
        k = optimal_k(self._bits_per_key)
        bits = bytearray(num_bits >> 3)

        for h1, h2 in zip(self._h1, self._h2):
            for i in range(k):
                pos = (h1 + i * h2) % num_bits
                bits[pos >> 3] |= 1 << (pos & 7)

        return BloomFilter(bytes(bits), num_bits, k, self._count)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<BloomBuilder keys={self._count} bits_per_key={self._bits_per_key}>"
