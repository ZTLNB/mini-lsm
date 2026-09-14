"""跨模块共用的小工具。

单独放一个模块是为了**打破循环导入**:``snapshot`` 和 ``engine`` 都需要
"把用户输入统一转成 bytes",而 ``engine`` 又要导入 ``snapshot``。
把这个小函数放在两边都能依赖的中立位置,比在函数体里做局部导入干净。
"""

from __future__ import annotations

from .errors import InvalidArgumentError

__all__ = ["to_bytes"]


def to_bytes(value: object, name: str) -> bytes:
    """把用户输入统一转成 bytes。

    允许传 str(按 UTF-8 编码),这样交互式使用时不必到处写 ``b""``。
    允许 bytearray / memoryview,但**一定复制一份** —— 否则调用方之后
    修改那个缓冲区,会把已经写进内存表或 WAL 的数据一起改掉。
    """
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise InvalidArgumentError(
        f"{name} 必须是 str 或 bytes,实际是 {type(value).__name__}"
    )
