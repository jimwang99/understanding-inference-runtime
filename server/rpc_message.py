from abc import ABC
from dataclasses import dataclass, field
from itertools import count

from .error_code import ErrCode


@dataclass(kw_only=True)
class RPCMessage(ABC):
    src_device_uid: str
    dst_device_uid: str
    uid: int = field(default_factory=count().__next__)


@dataclass(kw_only=True)
class AllocRequest(RPCMessage):
    size: int


@dataclass(kw_only=True)
class AllocResponse(RPCMessage):
    addr: int
    err: ErrCode


@dataclass(kw_only=True)
class FreeRequest(RPCMessage):
    addr: int


@dataclass(kw_only=True)
class FreeResponse(RPCMessage):
    err: ErrCode


@dataclass(kw_only=True)
class ReadRequest(RPCMessage):
    addr: int
    size: int


@dataclass(kw_only=True)
class ReadResponse(RPCMessage):
    data: bytes
    err: ErrCode


@dataclass(kw_only=True)
class WriteRequest(RPCMessage):
    addr: int
    data: bytes


@dataclass(kw_only=True)
class WriteResponse(RPCMessage):
    err: ErrCode


@dataclass(kw_only=True)
class ExecuteRequest(RPCMessage):
    executable_addr: int
    args_addr: int


@dataclass(kw_only=True)
class ExecuteResponse(RPCMessage):
    err: ErrCode


@dataclass(kw_only=True)
class ExitRequest(RPCMessage):
    reason: str


@dataclass(kw_only=True)
class ExitResponse(RPCMessage):
    err: ErrCode
