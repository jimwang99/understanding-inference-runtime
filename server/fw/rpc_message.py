from abc import ABC
from dataclasses import dataclass, field
from itertools import count

from server.error_code import ErrCode
from ..data_types import Tensor, Event, Pointers, Executable


@dataclass(kw_only=True)
class RPCMessage(ABC):
    src_device_uid: str
    dst_device_uid: str
    uid: int = field(default_factory=count().__next__)


# ==============================================================================
# memory
# ==============================================================================


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


# ==============================================================================
# tensor
# ==============================================================================


@dataclass(kw_only=True)
class ReadTensorRequest(RPCMessage):
    addr: int


@dataclass(kw_only=True)
class ReadTensorResponse(RPCMessage):
    data: Tensor
    err: ErrCode


@dataclass(kw_only=True)
class WriteTensorRequest(RPCMessage):
    addr: int
    data: Tensor


@dataclass(kw_only=True)
class WriteTensorResponse(RPCMessage):
    err: ErrCode


# ==============================================================================
# event
# ==============================================================================


@dataclass(kw_only=True)
class SetEventRequest(RPCMessage):
    addr: int


@dataclass(kw_only=True)
class SetEventResponse(RPCMessage):
    err: ErrCode


# ==============================================================================
# pointers
# ==============================================================================


@dataclass(kw_only=True)
class WritePointersRequest(RPCMessage):
    addr: int
    data: Pointers


@dataclass(kw_only=True)
class WritePointersResponse(RPCMessage):
    err: ErrCode


# ==============================================================================
# executable
# ==============================================================================


@dataclass(kw_only=True)
class WriteExecutableRequest(RPCMessage):
    addr: int
    data: Executable


@dataclass(kw_only=True)
class WriteExecutableResponse(RPCMessage):
    err: ErrCode


@dataclass(kw_only=True)
class ExecuteRequest(RPCMessage):
    executable_addr: int
    args_addr: int


@dataclass(kw_only=True)
class ExecuteResponse(RPCMessage):
    err: ErrCode


# ==============================================================================
# exit
# ==============================================================================


@dataclass(kw_only=True)
class ExitRequest(RPCMessage):
    reason: str


@dataclass(kw_only=True)
class ExitResponse(RPCMessage):
    err: ErrCode
