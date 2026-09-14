"""异常体系。

分层设计的目的:让调用方能够精确区分"数据损坏"(需要截断恢复)
和"用法错误"(应该直接报错),而不是笼统地 catch Exception。
"""

from __future__ import annotations


class LSMError(Exception):
    """所有本引擎异常的基类。"""


class InvalidArgumentError(LSMError):
    """调用方传入了非法参数,属于用法错误,不应重试。"""


class ClosedError(LSMError):
    """对已关闭的引擎执行操作。"""

    def __init__(self, message: str = "存储引擎已关闭") -> None:
        super().__init__(message)


class CorruptionError(LSMError):
    """磁盘数据损坏 —— 字节内容不符合预期格式。

    这类异常在**恢复路径**上不应向上抛出,而应被捕获后触发截断,
    因为 LSM 的设计前提就是"崩溃可能留下半截数据"。
    """

    def __init__(self, message: str, offset: int | None = None) -> None:
        self.offset = offset
        location = f"(偏移 {offset})" if offset is not None else ""
        super().__init__(f"{message}{location}")


class TruncatedRecordError(CorruptionError):
    """记录被截断 —— 通常是进程在写入过程中崩溃留下的残骸。

    这是**预期内**的情况,不是 bug。
    """

    def __init__(self, message: str = "记录不完整,疑似写入中途崩溃", offset: int | None = None) -> None:
        super().__init__(message, offset)


class ChecksumMismatchError(CorruptionError):
    """CRC 校验失败 —— 数据在磁盘上被改写或位翻转。"""

    def __init__(self, expected: int, actual: int, offset: int | None = None) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"CRC 校验失败:期望 {expected:#010x},实际 {actual:#010x}", offset
        )
