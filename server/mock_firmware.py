from __future__ import annotations

import multiprocessing as mp
import numpy as np
import pickle
import queue
import threading
import time

from abc import ABC, abstractmethod
from loguru import logger

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

    def wait(self, timeout: float = 1.0, clear: bool = False) -> ErrCode:
        start_time = time.time()
        while True:
            self.lock.acquire()
            if self.value:
                if clear:
                    self.value = False
                self.lock.release()
                break
            self.lock.release()
            time.sleep(WAIT_TIME)
            if time.time() - start_time > timeout:
                return ErrCode.ETIME
        return ErrCode.ESUCC


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
        self.firmware: MockFirmware | None = None
        self.args: Pointers | None = None

    def __repr__(self) -> str:
        s = f"<Executable: {self.name}>"
        return s

    def size(self) -> int:
        return len(pickle.dumps(self))

    def __call__(self, firmware: MockFirmware, args: Pointers) -> ErrCode:
        self.firmware = firmware
        self.args = args
        return self._execute()

    @abstractmethod
    def _execute(self) -> ErrCode:
        return ErrCode.ENOENT

    # ==========================================================================
    # Kernel interface
    # ==========================================================================

    def quit(self) -> None:
        assert self.firmware is not None
        self.firmware.quit()

    # --------------------------------------------------------------------------
    # create / delete a new object locally
    # --------------------------------------------------------------------------
    def create_any(self, data: Tensor | Event | Pointers | Executable, device_uid: str = "", blocking: bool = False) -> int:
        assert self.firmware is not None
        if device_uid:
            # remote allocation
            if blocking:
                addr, err = self.firmware.alloc_remote_b(data.size(), device_uid)
            else:
                addr, err = self.firmware.alloc_remote_nb(data.size(), device_uid)
        else:
            # local allocation
            addr, err = self.firmware.alloc_local(data.size())
            if err != ErrCode.ESUCC:
                raise RuntimeError(
                    f"Failed to allocate memory for {data.mtype} {data.name}"
                )
            self.firmware.write_local(addr, pickle.dumps(data))
        return addr

    def delete_any(self, addr: int) -> None:
        assert self.firmware is not None
        err = self.firmware.free_local(addr)
        if err != ErrCode.ESUCC:
            raise RuntimeError(f"Failed to free memory at {addr}")

    def create_tensor(self, name: str, shape: tuple[int, ...]) -> int:
        data = Tensor(name, np.empty(shape, dtype=np.float32))
        return self.create_any(data)

    def create_event(self, name: str) -> int:
        data = Event(name)
        return self.create_any(data)

    def create_pointers(self, name: str, ptrs: list[str]) -> int:
        data = Pointers(name, ptrs)
        return self.create_any(data)

    def create_executable(self, name: str, executable: Executable) -> int:
        return self.create_any(executable)


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


EMPTY_POINTERS = Pointers("empty", [])


class MockFirmware(MessageFutureHost):
    def __init__(self, device: MockDevice) -> None:
        super().__init__()
        self.device = device

        self.alloc_local = self.device.alloc
        self.free_local = self.device.free
        self.read_local = self.device.read
        self.write_local = self.device.write
        self.get_remote_device = self.device.get_device_by_alias
        self.get_output_queue = self.device.get_output_queue_by_alias
        self.input_queue = self.device.input_queue
        self.execution_queue = self.device.execution_queue

        self.proc: mp.Process | None = None
        self.threads: dict[str, threading.Thread] = {}

        self.quit_event = mp.Event()

        self.exceptions: mp.Queue[Exception] = mp.Queue()

    def start(self) -> None:
        logger.info(f"{self.device.uid}: Starting process {self=}")
        self.proc = mp.Process(target=self._start_threads)
        self.proc.start()

    def _start_threads(self) -> None:
        logger.info(f"{self.device.uid}: Starting threads {self=}")
        self.threads["input_handler"] = threading.Thread(target=self.input_handler)
        self.threads["execution_handler"] = threading.Thread(
            target=self.execution_handler
        )
        for name, thread in self.threads.items():
            logger.info(f"{self.device.uid}: Starting {name} thread")
            thread.start()

    def quit(self) -> None:
        logger.debug(f"{self.device.uid}: Setting quit event")
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

            if not self.exceptions.empty():
                while not self.exceptions.empty():
                    e = self.exceptions.get()
                    logger.error(f"{self.device.uid}: Exception in process: {e}")
                raise RuntimeError(f"{self.device.uid}: Exception in process")

            logger.info(f"{self.device.uid}: Process joined")

    def _raise(self, e: Exception) -> None:
        self.exceptions.put(e)
        raise e

    def _assert(self, cond: bool, msg: str) -> None:
        if not cond:
            self._raise(AssertionError(msg))
        assert cond, msg

    # ==========================================================================
    # RPC interface to a connected device
    # ==========================================================================

    def _alloc_remote(self, size: int, remote_device_alias: str) -> AllocRequest:
        remote_device = self.get_remote_device(remote_device_alias)
        mesg = AllocRequest(
            src_device_uid=self.device.uid,
            dst_device_uid=remote_device.uid,
            size=size,
        )
        self.get_output_queue(remote_device_alias).put(mesg)
        return mesg

    def _free_remote(self, addr: int, remote_device_alias: str) -> FreeRequest:
        remote_device = self.get_remote_device(remote_device_alias)
        mesg = FreeRequest(
            src_device_uid=self.device.uid,
            dst_device_uid=remote_device.uid,
            addr=addr,
        )
        self.get_output_queue(remote_device_alias).put(mesg)
        return mesg

    def _exit_remote(self, reason: str, remote_device_alias: str) -> ExitRequest:
        remote_device = self.get_remote_device(remote_device_alias)
        mesg = ExitRequest(
            src_device_uid=self.device.uid,
            dst_device_uid=remote_device.uid,
            reason=reason,
        )
        self.get_output_queue(remote_device_alias).put(mesg)
        return mesg

    def _read_remote(self, addr: int, remote_device_alias: str) -> ReadRequest:
        remote_device = self.get_remote_device(remote_device_alias)
        mesg = ReadRequest(
            src_device_uid=self.device.uid,
            dst_device_uid=remote_device.uid,
            addr=addr,
        )
        self.get_output_queue(remote_device_alias).put(mesg)
        return mesg

    def _write_remote(
        self, addr: int, data: bytes, remote_device_alias: str
    ) -> WriteRequest:
        remote_device = self.get_remote_device(remote_device_alias)
        mesg = WriteRequest(
            src_device_uid=self.device.uid,
            dst_device_uid=remote_device.uid,
            addr=addr,
            data=data,
        )
        self.get_output_queue(remote_device_alias).put(mesg)
        return mesg

    # --------------------------------------------------------------------------
    # blocking interface
    # --------------------------------------------------------------------------
    def alloc_remote_b(self, size: int, remote_device_alias: str) -> int:
        req = self._alloc_remote(size, remote_device_alias)

        rsp = self.input_queue.get()
        self._assert(rsp.uid == req.uid, f"{rsp.uid=} != {req.uid=}")
        self._assert(isinstance(rsp, AllocResponse), f"{type(rsp)=} is not AllocResponse")
        assert isinstance(rsp, AllocResponse) # to satisfy type checker
        if rsp.err != ErrCode.ESUCC:
            self._raise(
                RuntimeError(
                    f"Failed to allocate memory on {remote_device_alias}. {rsp.err=}"
                )
            )
        return rsp.addr

    def free_remote_b(self, addr: int, remote_device_alias: str) -> None:
        req = self._free_remote(addr, remote_device_alias)

        rsp = self.input_queue.get()
        self._assert(rsp.uid == req.uid, f"{rsp.uid=} != {req.uid=}")
        self._assert(isinstance(rsp, FreeResponse), f"{type(rsp)=} is not FreeResponse")
        assert isinstance(rsp, FreeResponse) # to satisfy type checker
        if rsp.err != ErrCode.ESUCC:
            self._raise(
                RuntimeError(
                    f"Failed to free memory on {remote_device_alias}. {rsp.err=}"
                )
            )

    def exit_remote_b(self, reason: str, remote_device_alias: str) -> None:
        req = self._exit_remote(reason, remote_device_alias)

        rsp = self.input_queue.get()
        self._assert(rsp.uid == req.uid, f"{rsp.uid=} != {req.uid=}")
        self._assert(isinstance(rsp, ExitResponse), f"{type(rsp)=} is not ExitResponse")
        assert isinstance(rsp, ExitResponse) # to satisfy type checker
        if rsp.err != ErrCode.ESUCC:
            self._raise(
                RuntimeError(f"Failed to exit on {remote_device_alias}. {rsp.err=}")
            )

    def read_remote_b(self, addr: int, remote_device_alias: str) -> bytes:
        req = self._read_remote(addr, remote_device_alias)

        rsp = self.input_queue.get()
        self._assert(rsp.uid == req.uid, f"{rsp.uid=} != {req.uid=}")
        self._assert(isinstance(rsp, ReadResponse), f"{type(rsp)=} is not ReadResponse")
        assert isinstance(rsp, ReadResponse) # to satisfy type checker
        if rsp.err != ErrCode.ESUCC:
            self._raise(
                RuntimeError(f"Failed to read from {remote_device_alias}. {rsp.err=}")
            )
        return rsp.data

    def write_remote_b(self, addr: int, data: bytes, remote_device_alias: str) -> None:
        req = self._write_remote(addr, data, remote_device_alias)

        rsp = self.input_queue.get()
        self._assert(rsp.uid == req.uid, f"{rsp.uid=} != {req.uid=}")
        self._assert(isinstance(rsp, WriteResponse), f"{type(rsp)=} is not WriteResponse")
        assert isinstance(rsp, WriteResponse) # to satisfy type checker
        if rsp.err != ErrCode.ESUCC:
            self._raise(
                RuntimeError(f"Failed to write to {remote_device_alias}. {rsp.err=}")
            )

    # --------------------------------------------------------------------------
    # non-blocking interface
    # --------------------------------------------------------------------------

    def alloc_remote_nb(self, size: int, remote_device_alias: str) -> MessageFuture:
        req = self._alloc_remote(size, remote_device_alias)
        return self.create_mesg_future(req)

    def free_remote_nb(self, addr: int, remote_device_alias: str) -> MessageFuture:
        req = self._free_remote(addr, remote_device_alias)
        return self.create_mesg_future(req)

    def exit_remote_nb(self, reason: str, remote_device_alias: str) -> MessageFuture:
        req = self._exit_remote(reason, remote_device_alias)
        return self.create_mesg_future(req)

    def read_remote_nb(self, addr: int, remote_device_alias: str) -> MessageFuture:
        req = self._read_remote(addr, remote_device_alias)
        return self.create_mesg_future(req)

    def write_remote_nb(
        self, addr: int, data: bytes, remote_device_alias: str
    ) -> MessageFuture:
        req = self._write_remote(addr, data, remote_device_alias)
        return self.create_mesg_future(req)

    # ==========================================================================
    # input message handler
    # ==========================================================================
    def input_handler(self, is_test: bool = False) -> None:
        logger.info(f"{self.device.uid}: Running input handler of {self=}")
        while not self.quit_event.is_set() and not is_test:
            # get a message from the input queue
            # use non-blocking get so that we can check `self.quit_event`
            try:
                mesg = self.input_queue.get_nowait()
            except queue.Empty:
                time.sleep(WAIT_TIME)
                continue

            logger.trace(f"{self.device.uid}: {mesg=}")
            assert mesg.dst_device_uid == self.device.uid, (
                f"{mesg.dst_device_uid=} != {self.device.uid=}"
            )

            # ------------------------------------------------------------------
            # process the request messages
            # ------------------------------------------------------------------
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
                # because execution usually runs for a while
                # and we don't want to block the input handler thread
                self.execution_queue.put(mesg)
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
            # ------------------------------------------------------------------
            # process the response messages
            # ------------------------------------------------------------------
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
                self._raise(ValueError(f"Invalid message type: {type(mesg)=}"))

    # ==========================================================================
    # execution message handler
    # ==========================================================================
    # we choose to handel execution messages in a separate thread because
    # there will be "waiting for events" operations in the executable, and we
    # don't want to block the input handler thread

    def execution_handler(self, is_test: bool = False) -> None:
        logger.info(f"{self.device.uid}: Running execution handler of {self=}")
        while not self.quit_event.is_set() and not is_test:
            # get a message from the execution queue
            # use non-blocking get so that we can check `self.quit_event`
            try:
                mesg = self.execution_queue.get_nowait()
            except queue.Empty:
                time.sleep(WAIT_TIME)
                continue

            is_local = mesg.src_device_uid == self.device.uid
            logger.trace(f"{self.device.uid}: {mesg=} is_local={is_local}")
            assert mesg.dst_device_uid == self.device.uid
            assert isinstance(mesg, ExecuteRequest), f"{type(mesg)=}"

            # ------------------------------------------------------------------
            # process the execution request
            # ------------------------------------------------------------------
            # read executable from local memory
            raw, err = self.device.read(mesg.executable_addr)
            if err != ErrCode.ESUCC:
                if is_local:
                    self._raise(
                        RuntimeError(
                            f"Failed to read executable from local memory. {err=}"
                        )
                    )
                else:
                    self.device.output_queues[mesg.src_device_uid].put(
                        ExecuteResponse(
                            uid=mesg.uid,
                            src_device_uid=mesg.dst_device_uid,
                            dst_device_uid=mesg.src_device_uid,
                            err=err,
                        )
                    )
                continue

            # convert memory raw bytes to Executable object
            executable = pickle.loads(raw)
            assert isinstance(executable, Executable), f"{type(executable)=}"

            # read args from local memory
            raw, err = self.device.read(mesg.args_addr)
            if err != ErrCode.ESUCC:
                if is_local:
                    self._raise(
                        RuntimeError(f"Failed to read args from local memory. {err=}")
                    )
                else:
                    self.device.output_queues[mesg.src_device_uid].put(
                        ExecuteResponse(
                            uid=mesg.uid,
                            src_device_uid=mesg.dst_device_uid,
                            dst_device_uid=mesg.src_device_uid,
                            err=err,
                        )
                    )
                continue

            # convert memory raw bytes to Pointers object
            args = pickle.loads(raw)
            assert isinstance(args, Pointers), f"{type(args)=}"

            # execute the executable
            logger.debug(f"{self.device.uid}: Executing {executable} with {args}")
            err = executable(self, args)

            # check result
            if err != ErrCode.ESUCC:
                if is_local:
                    self._raise(
                        RuntimeError(f"Failed to execute executable. {err=}")
                    )
                else:
                    # send response to the source device
                    self.device.output_queues[mesg.src_device_uid].put(
                        ExecuteResponse(
                            uid=mesg.uid,
                            src_device_uid=mesg.dst_device_uid,
                            dst_device_uid=mesg.src_device_uid,
                            err=err,
                        )
                    )
            logger.debug(f"{self.device.uid}: Executed {executable} with {args}")

    def load_boot_program(
        self, executable: Executable, args: Pointers = EMPTY_POINTERS
    ) -> None:
        logger.debug(f"{self.device.uid}: Loading boot program {executable} with {args}")

        executable_addr, err = self.alloc_local(executable.size())
        assert err == ErrCode.ESUCC

        args_addr, err = self.alloc_local(args.size())
        assert err == ErrCode.ESUCC

        self.write_local(executable_addr, pickle.dumps(executable))
        self.write_local(args_addr, pickle.dumps(args))

        self.execution_queue.put(
            ExecuteRequest(
                src_device_uid=self.device.uid,
                dst_device_uid=self.device.uid,
                uid=-1,
                executable_addr=executable_addr,
                args_addr=args_addr,
            )
        )
