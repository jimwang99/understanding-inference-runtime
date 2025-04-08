from __future__ import annotations

import threading
import time
from abc import ABC

from server.mock_hardware import MockDevice
from .thread_support import ThreadSupport
from ..data_types import Tensor
from .rpc_message import RPCMessage, AllocResponse, ReadTensorResponse


class MessageFuture:
    def __init__(
        self, req: RPCMessage, saved: dict[int, MessageFuture] | None = None
    ) -> None:
        self.lock = threading.Lock()
        self.req_mesg = req
        self.rsp_mesg: RPCMessage | None = None
        self.saved = saved
        if self.saved:
            self.saved[self.req_mesg.uid] = self

    def __repr__(self) -> str:
        return f"<MessageFuture: {self.req_mesg} {self.rsp_mesg}>"

    def __del__(self) -> None:
        if not self.rsp_mesg:
            self.wait()
        if self.saved:
            del self.saved[self.req_mesg.uid]

    def set(self, rsp: RPCMessage) -> None:
        self.lock.acquire()
        self.rsp_mesg = rsp
        self.lock.release()

    def wait(self, timeout: float = 1.0) -> RPCMessage | None:
        start_time = time.time()
        self.lock.acquire()
        while not self.rsp_mesg:
            if timeout != 0.0 and time.time() - start_time > timeout:
                raise TimeoutError(
                    f"MessageFuture {self.req_mesg.uid} timed out after {timeout} seconds"
                )
                return None
            self.lock.release()
            time.sleep(0.01)
            self.lock.acquire()
        self.lock.release()
        return self.rsp_mesg


class MessageFutureHost(ThreadSupport, ABC):
    def __init__(self, device: MockDevice) -> None:
        ThreadSupport.__init__(self, device)
        self.mesg_futures: dict[int, MessageFuture] = {}

    def create_mesg_future(self, req: RPCMessage) -> MessageFuture:
        mf = MessageFuture(req, self.mesg_futures)
        self.mesg_futures[req.uid] = mf
        return mf

    def set_mesg_future(self, rsp: RPCMessage) -> None:
        if rsp.uid in self.mesg_futures:
            self.mesg_futures[rsp.uid].set(rsp)
        else:
            self.raise_exception(
                ValueError(
                    f"No message future found for {rsp.uid}, existing futures are {self.mesg_futures.keys()}"
                )
            )

    def wait_mesg_future(self, arg: int | MessageFuture, timeout: float = 1.0) -> RPCMessage | None:
        if isinstance(arg, int):
            uid = arg
            if uid in self.mesg_futures:
                future = self.mesg_futures[uid]
            else:
                self.raise_exception(
                    ValueError(
                        f"No message future found for {uid}, existing futures are {self.mesg_futures.keys()}"
                    )
                )
                return None
        else:
            future = arg

        try:
            return future.wait(timeout=timeout)
        except TimeoutError as e:
            self.raise_exception(e)
            return None
    
    def extract_mesg_future(self, arg: int | MessageFuture, timeout: float = 1.0) -> int | Tensor | None:
        mesg = self.wait_mesg_future(arg, timeout)
        self.assert_cond(mesg is not None, "return mesg is None")

        if isinstance(mesg, AllocResponse):
            return mesg.addr
        elif isinstance(mesg, ReadTensorResponse):
            return mesg.data
        return None