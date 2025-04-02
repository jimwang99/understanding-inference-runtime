from __future__ import annotations

import multiprocessing as mp
import numpy as np
import pickle
import queue
import threading
import time

from abc import ABC, abstractmethod
from loguru import logger
from typing import Any

from .error_code import ErrCode
from .mock_hardware import MockDevice
from .rpc_message import (
    RPCMessage,
    AllocRequest,
    AllocResponse,
    FreeRequest,
    FreeResponse,
    ReadRequest,
    ReadResponse,
    WriteRequest,
    WriteResponse,
    ExecuteRequest,
    ExecuteResponse,
    ExitRequest,
    ExitResponse,
)

WAIT_TIME = 0.01


class Tensor:
    mtype = "Tensor"

    def __init__(self, name: str, data: np.ndarray):
        self.name = name
        self.data = data

    def __repr__(self) -> str:
        return f"<Tensor: {self.name} {self.data.shape}>"

    def size(self) -> int:
        return self.data.nbytes


class Event:
    mtype = "Event"

    def __init__(self, name: str, value: bool = False):
        self.name = name
        self.value = value
        self.lock = threading.Lock()

    def __repr__(self) -> str:
        return f"<Event: {self.name} {self.value}>"

    def size(self) -> int:
        return 8

    def set(self) -> None:
        self.lock.acquire()
        self.value = True
        self.lock.release()

    def clear(self) -> None:
        self.lock.acquire()
        self.value = False
        self.lock.release()

    def wait(self, timeout: float = 1.0) -> None:
        start_time = time.time()
        while True:
            self.lock.acquire()
            if self.value:
                self.lock.release()
                break
            self.lock.release()
            time.sleep(WAIT_TIME)
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Event {self.name} timed out after {timeout} seconds"
                )

    def wait_and_clear(self, timeout: float = 1.0) -> None:
        start_time = time.time()
        while True:
            self.lock.acquire()
            if self.value:
                self.value = False
                self.lock.release()
                break
            self.lock.release()
            time.sleep(WAIT_TIME)
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Event {self.name} timed out after {timeout} seconds"
                )


class Pointers:
    mtype = "Pointers"

    def __init__(self, name: str, ptrs: list[str]):
        self.name = name
        self.ptrs = ptrs

    def __repr__(self) -> str:
        s = f"<Pointers: {self.name} {' '.join(self.ptrs)}>"
        return s

    def size(self) -> int:
        return len(self.ptrs) * 8


class Executable(ABC):
    mtype = "Executable"

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        s = f"<Executable: {self.name}>"
        return s

    def size(self) -> int:
        return len(pickle.dumps(self))

    @abstractmethod
    def __call__(self, args: Pointers, host: MockDevice) -> ErrCode:
        raise NotImplementedError


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

    def wait(self, timeout: float = 1.0) -> RPCMessage:
        start_time = time.time()
        self.lock.acquire()
        while not self.rsp_mesg:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"MessageFuture {self.req_mesg.uid} timed out after {timeout} seconds"
                )
            self.lock.release()
            time.sleep(WAIT_TIME)
            self.lock.acquire()
        self.lock.release()
        return self.rsp_mesg


class MessageFutureHost:
    def __init__(self) -> None:
        self.mesg_futures: dict[int, MessageFuture] = {}

    def create_mesg_future(self, req: RPCMessage) -> MessageFuture:
        mf = MessageFuture(req, self.mesg_futures)
        self.mesg_futures[req.uid] = mf
        return mf

    def set_mesg_future(self, rsp: RPCMessage) -> None:
        if rsp.uid in self.mesg_futures:
            self.mesg_futures[rsp.uid].set(rsp)
        else:
            raise ValueError(
                f"No message future found for {rsp.uid}, existing futures are {self.mesg_futures.keys()}"
            )

    def wait_mesg_future(self, uid: int) -> RPCMessage:
        if uid in self.mesg_futures:
            return self.mesg_futures[uid].wait()
        else:
            raise ValueError(
                f"No message future found for {uid}, existing futures are {self.mesg_futures.keys()}"
            )


class MockFirmware(MessageFutureHost):
    def __init__(self, device: MockDevice) -> None:
        super().__init__()
        self.device = device
        self.proc: mp.Process | None = None
        self.threads: dict[str, threading.Thread] = {}

        self.quit_event = mp.Event()

    def start(self) -> None:
        logger.info(f"{self.device.uid}: Starting process")
        logger.info(f"{self.device.uid}: {self=}")
        self.proc = mp.Process(target=self._start_threads)
        self.proc.start()

    def _start_threads(self) -> None:
        logger.info(f"{self.device.uid}: Starting threads")
        logger.info(f"{self.device.uid}: {self=}")
        self.threads["input_handler"] = threading.Thread(target=self.input_handler)
        self.threads["execution_handler"] = threading.Thread(
            target=self.execution_handler
        )
        for name, thread in self.threads.items():
            logger.info(f"{self.device.uid}: Starting {name} thread")
            thread.start()

    def quit(self) -> None:
        logger.info(f"{self.device.uid}: Setting quit event")
        self.quit_event.set()

    def _join_threads(self) -> None:
        logger.info(f"{self.device.uid}: Joining threads")
        for name, thread in self.threads.items():
            logger.info(f"{self.device.uid}: Joining {name} thread")
            thread.join()

    def join(self) -> None:
        if self.proc is not None:
            logger.info(f"{self.device.uid}: Joining process")
            self.proc.join()

    # ==========================================================================
    # create / delete a new object locally
    # ==========================================================================
    def create_any(self, data: Tensor | Event | Pointers | Executable) -> int:
        addr, err = self.device.alloc(data.size())
        if err != ErrCode.ESUCC:
            raise RuntimeError(
                f"Failed to allocate memory for {data.mtype} {data.name}"
            )
        self.device.write(addr, pickle.dumps(data))
        return addr

    def create_tensor(self, name: str, shape: tuple[int, ...]) -> int:
        data = Tensor(name, np.empty(shape, dtype=np.float32))
        return self.create_any(data)

    def create_event(self, name: str) -> int:
        data = Event(name)
        return self.create_any(data)

    def create_pointers(self, name: str, ptrs: list[str]) -> int:
        data = Pointers(name, ptrs)
        return self.create_any(data)

    def delete_any(self, addr: int) -> None:
        err = self.device.free(addr)
        if err != ErrCode.ESUCC:
            raise RuntimeError(f"Failed to free memory at {addr}")

    # ==========================================================================
    # RPC interface to a connected device
    # ==========================================================================

    def _alloc_remote(
        self, size: int, device_alias: str
    ) -> tuple[MockDevice, AllocRequest]:
        device = self.device.get_device_by_alias(device_alias)
        mesg = AllocRequest(
            src_device_uid=self.device.uid,
            dst_device_uid=device.uid,
            size=size,
        )
        self.device.get_output_queue_by_alias(device_alias).put(mesg)
        return device, mesg

    def _free_remote(
        self, addr: int, device_alias: str
    ) -> tuple[MockDevice, FreeRequest]:
        device = self.device.get_device_by_alias(device_alias)
        mesg = FreeRequest(
            src_device_uid=self.device.uid,
            dst_device_uid=device.uid,
            addr=addr,
        )
        self.device.get_output_queue_by_alias(device_alias).put(mesg)
        return device, mesg

    def _exit_remote(
        self, reason: str, device_alias: str
    ) -> tuple[MockDevice, ExitRequest]:
        device = self.device.get_device_by_alias(device_alias)
        mesg = ExitRequest(
            src_device_uid=self.device.uid,
            dst_device_uid=device.uid,
            reason=reason,
        )
        self.device.get_output_queue_by_alias(device_alias).put(mesg)
        return device, mesg

    # --------------------------------------------------------------------------
    # blocking interface
    # --------------------------------------------------------------------------
    def alloc_remote(self, size: int, device_alias: str) -> int:
        _, req = self._alloc_remote(size, device_alias)

        rsp = self.device.input_queue.get()
        assert rsp.uid == req.uid
        assert isinstance(rsp, AllocResponse)
        if rsp.err != ErrCode.ESUCC:
            raise RuntimeError(f"Failed to allocate memory on {device_alias}")
        return rsp.addr

    def free_remote(self, addr: int, device_alias: str) -> None:
        _, req = self._free_remote(addr, device_alias)

        rsp = self.device.input_queue.get()
        assert rsp.uid == req.uid
        assert isinstance(rsp, FreeResponse)
        if rsp.err != ErrCode.ESUCC:
            raise RuntimeError(f"Failed to free memory on {device_alias}")

    def exit_remote(self, reason: str, device_alias: str) -> None:
        _, req = self._exit_remote(reason, device_alias)

        rsp = self.device.input_queue.get()
        assert rsp.uid == req.uid
        assert isinstance(rsp, ExitResponse)
        if rsp.err != ErrCode.ESUCC:
            raise RuntimeError(f"Failed to exit on {device_alias}")

    # --------------------------------------------------------------------------
    # non-blocking interface
    # --------------------------------------------------------------------------

    def alloc_remote_nb(self, size: int, device_alias: str) -> MessageFuture:
        _, req = self._alloc_remote(size, device_alias)
        return self.create_mesg_future(req)

    def free_remote_nb(self, addr: int, device_alias: str) -> MessageFuture:
        _, req = self._free_remote(addr, device_alias)
        return self.create_mesg_future(req)

    def exit_remote_nb(self, reason: str, device_alias: str) -> MessageFuture:
        _, req = self._exit_remote(reason, device_alias)
        return self.create_mesg_future(req)

    # def read_remote_nb(self, addr: int, size: int, device_alias: str) -> None:
    #     device = self.device.get_device_by_alias(device_alias)
    #     mesg = ReadRequest(
    #         src_device_uid=self.device.uid,
    #         dst_device_uid=device.uid,
    #         addr=addr,
    #         size=size,
    #     )
    #     self.device.get_output_queue_by_alias(device_alias).put(mesg)

    # def write_remote_nb(self, addr: int, data: bytes, device_alias: str) -> None:
    #     device = self.device.get_device_by_alias(device_alias)
    #     mesg = WriteRequest(
    #         src_device_uid=self.device.uid,
    #         dst_device_uid=device.uid,
    #         addr=addr,
    #         data=data,
    #     )
    #     self.device.get_output_queue_by_alias(device_alias).put(mesg)

    # def execute_remote_nb(
    #     self, executable_addr: int, args_addr: int, device_alias: str
    # ) -> None:
    #     device = self.device.get_device_by_alias(device_alias)
    #     mesg = ExecuteRequest(
    #         src_device_uid=self.device.uid,
    #         dst_device_uid=device.uid,
    #         executable_addr=executable_addr,
    #         args_addr=args_addr,
    #     )
    #     self.device.get_output_queue_by_alias(device_alias).put(mesg)

    # ==========================================================================
    # create / delete a new object on connected devices
    # ==========================================================================
    # def create_remote_any(
    #     self,
    #     data: Tensor | Event | Pointers | Executable,
    #     device_alias: str,
    # ) -> int:
    #     self.alloc_remote_nb(data.size(), device_alias)

    # def create_remote_tensor(
    #     self, name: str, shape: tuple[int, ...], device_alias: str
    # ) -> int:
    #     data = Tensor(name, np.empty(shape, dtype=np.float32))
    #     return self.create_remote_any(data, device_alias)

    # def create_remote_event(self, name: str, device_alias: str) -> int:
    #     data = Event(name)
    #     return self.create_remote_any(data, device_alias)

    # def create_pointers_on_device(
    #     self, name: str, ptrs: list[str], device_alias: str
    # ) -> int:
    #     data = Pointers(name, ptrs)
    #     return self.create_any_on_device(data, device_alias)

    # def delete_any_on_device(self, addr: int, device_alias: str) -> None:
    #     device = self.device.get_device_by_alias(device_alias)
    #     err = device.free(addr)
    #     if err != ErrCode.ESUCC:
    #         raise RuntimeError(f"Failed to free memory at {addr}")

    # ==========================================================================
    # input message handler
    # ==========================================================================
    def input_handler(self, is_test: bool = False) -> None:
        logger.info(f"{self.device.uid}: {self=}")
        while not self.quit_event.is_set() and not is_test:
            try:
                mesg = self.device.input_queue.get_nowait()
            except queue.Empty:
                time.sleep(WAIT_TIME)
                continue
            logger.trace(f"{self.device.uid}: {mesg=}")
            assert mesg.dst_device_uid == self.device.uid, (
                f"{mesg.dst_device_uid=} != {self.device.uid=}"
            )
            # process the message
            if isinstance(mesg, AllocRequest):
                addr, err = self.device.alloc(mesg.size)
                self.device.output_queues[mesg.src_device_uid].put(
                    AllocResponse(
                        uid=mesg.uid,
                        src_device_uid=mesg.dst_device_uid,
                        dst_device_uid=mesg.src_device_uid,
                        addr=addr,
                        err=err,
                    )
                )
            elif isinstance(mesg, FreeRequest):
                err = self.device.free(mesg.addr)
                self.device.output_queues[mesg.src_device_uid].put(
                    FreeResponse(
                        uid=mesg.uid,
                        src_device_uid=mesg.dst_device_uid,
                        dst_device_uid=mesg.src_device_uid,
                        err=err,
                    )
                )
            elif isinstance(mesg, ReadRequest):
                raw, err = self.device.read(mesg.addr)
                self.device.output_queues[mesg.src_device_uid].put(
                    ReadResponse(
                        uid=mesg.uid,
                        src_device_uid=mesg.dst_device_uid,
                        dst_device_uid=mesg.src_device_uid,
                        data=raw,
                        err=err,
                    )
                )
            elif isinstance(mesg, WriteRequest):
                err = self.device.write(mesg.addr, mesg.data)
                self.device.output_queues[mesg.src_device_uid].put(
                    WriteResponse(
                        uid=mesg.uid,
                        src_device_uid=mesg.dst_device_uid,
                        dst_device_uid=mesg.src_device_uid,
                        err=err,
                    )
                )
            elif isinstance(mesg, ExecuteRequest):
                # defer to execution queue
                self.device.execution_queue.put(mesg)
            elif isinstance(mesg, ExitRequest):
                logger.info(f'{self.device.uid}: Exiting because of "{mesg.reason}"')
                self.device.output_queues[mesg.src_device_uid].put(
                    ExitResponse(
                        uid=mesg.uid,
                        src_device_uid=mesg.dst_device_uid,
                        dst_device_uid=mesg.src_device_uid,
                        err=ErrCode.ESUCC,
                    )
                )
                self.quit()
            elif isinstance(mesg, AllocResponse):
                self.set_mesg_future(mesg)
            elif isinstance(mesg, FreeResponse):
                self.set_mesg_future(mesg)
            elif isinstance(mesg, ReadResponse):
                self.set_mesg_future(mesg)
            elif isinstance(mesg, WriteResponse):
                self.set_mesg_future(mesg)
            elif isinstance(mesg, ExecuteResponse):
                self.set_mesg_future(mesg)
            elif isinstance(mesg, ExitResponse):
                self.set_mesg_future(mesg)
            else:
                raise ValueError(f"Invalid message type: {type(mesg)=}")

    def execution_handler(self, is_test: bool = False) -> None:
        logger.info(f"{self.device.uid}: {self=}")
        while not self.quit_event.is_set() and not is_test:
            try:
                mesg = self.device.execution_queue.get_nowait()
            except queue.Empty:
                time.sleep(WAIT_TIME)
                continue
            assert mesg.dst_device_uid == self.device.uid
            # process the message
            if isinstance(mesg, ExecuteRequest):
                pass
            else:
                raise ValueError(f"Invalid message type: {type(mesg)=}")
