import numpy as np
import queue
import time

from abc import ABC
from loguru import logger

from server.error_code import ErrCode
from server.mock_hardware import MockDevice

from ..data_types import Tensor, Event, Pointers, Executable
from .future_host import MessageFutureHost
from .rpc_message import (
    RPCMessage,
    AllocRequest,
    AllocResponse,
    FreeRequest,
    FreeResponse,
    ReadTensorRequest,
    ReadTensorResponse,
    WriteTensorRequest,
    WriteTensorResponse,
    SetEventRequest,
    SetEventResponse,
    WritePointersRequest,
    WritePointersResponse,
    WriteExecutableRequest,
    WriteExecutableResponse,
    ExecuteRequest,
    ExecuteResponse,
    ExitRequest,
    ExitResponse,
)
from .thread_support import ThreadSupport


class RPCHost(MessageFutureHost, ThreadSupport, ABC):
    def __init__(self, device: MockDevice) -> None:
        MessageFutureHost.__init__(self, device)
        ThreadSupport.__init__(self, device)
        ABC.__init__(self)

        self.device = device
        self.execution_queue: queue.Queue[ExecuteRequest] = queue.Queue()

        self.register_thread("message_handling", self.message_handling_thread_entry)
        self.register_thread("execution_handling", self.execution_handling_thread_entry)

    def handle_message(self, mesg: RPCMessage) -> None:
        # ------------------------------------------------------------------
        # process the request messages
        # ------------------------------------------------------------------
        # memory
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
            return
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
            return

        # tensor
        elif isinstance(mesg, ReadTensorRequest):
            tensor, err = self.device.read(mesg.addr)
            self.assert_cond(isinstance(tensor, Tensor), f"{type(tensor)=}")
            self.device.output_queues[mesg.src_device_uid].put(
                ReadTensorResponse(
                    uid=mesg.uid,
                    src_device_uid=mesg.dst_device_uid,
                    dst_device_uid=mesg.src_device_uid,
                    data=tensor,
                    err=err,
                )
            )
            return
        elif isinstance(mesg, WriteTensorRequest):
            tensor, err = self.device.read(mesg.addr)
            self.assert_cond(isinstance(tensor, Tensor), f"{type(tensor)=}")
            if err != ErrCode.ESUCC:
                self.device.output_queues[mesg.src_device_uid].put(
                    WriteTensorResponse(
                        uid=mesg.uid,
                        src_device_uid=mesg.dst_device_uid,
                        dst_device_uid=mesg.src_device_uid,
                        err=err,
                    )
                )
                return
            # check size
            if tensor.size() != mesg.data.size():
                self.device.output_queues[mesg.src_device_uid].put(
                    WriteTensorResponse(
                        uid=mesg.uid,
                        src_device_uid=mesg.dst_device_uid,
                        dst_device_uid=mesg.src_device_uid,
                        err=ErrCode.EINVAL,
                    )
                )
                return
            # write data to tensor
            np.copyto(mesg.data.data, tensor.data)
            self.device.output_queues[mesg.src_device_uid].put(
                WriteTensorResponse(
                    uid=mesg.uid,
                    src_device_uid=mesg.dst_device_uid,
                    dst_device_uid=mesg.src_device_uid,
                    err=ErrCode.ESUCC,
                )
            )
            return

        # event
        elif isinstance(mesg, SetEventRequest):
            event, err = self.device.read(mesg.addr)
            self.assert_cond(isinstance(event, Event), f"{type(event)=}")
            if err != ErrCode.ESUCC:
                self.device.output_queues[mesg.src_device_uid].put(
                    SetEventResponse(
                        uid=mesg.uid,
                        src_device_uid=mesg.dst_device_uid,
                        dst_device_uid=mesg.src_device_uid,
                        err=err,
                    )
                )
                return
            event.set()
            self.device.output_queues[mesg.src_device_uid].put(
                SetEventResponse(
                    uid=mesg.uid,
                    src_device_uid=mesg.dst_device_uid,
                    dst_device_uid=mesg.src_device_uid,
                    err=ErrCode.ESUCC,
                )
            )
            return

        # pointers
        elif isinstance(mesg, WritePointersRequest):
            err = self.device.write(mesg.addr, mesg.data)
            self.device.output_queues[mesg.src_device_uid].put(
                WritePointersResponse(
                    uid=mesg.uid,
                    src_device_uid=mesg.dst_device_uid,
                    dst_device_uid=mesg.src_device_uid,
                    err=err,
                )
            )
            return

        # executable
        elif isinstance(mesg, WriteExecutableRequest):
            err = self.device.write(mesg.addr, mesg.data)
            self.device.output_queues[mesg.src_device_uid].put(
                WriteExecutableResponse(
                    uid=mesg.uid,
                    src_device_uid=mesg.dst_device_uid,
                    dst_device_uid=mesg.src_device_uid,
                    err=err,
                )
            )
            return
        elif isinstance(mesg, ExecuteRequest):
            # defer to execution queue
            # because execution usually runs for a while
            # and we don't want to block the input handler thread
            self.execution_queue.put(mesg)
            return

        # exit
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
            return

        # ------------------------------------------------------------------
        # process the response messages
        # ------------------------------------------------------------------
        elif isinstance(
            mesg,
            (
                AllocResponse,
                FreeResponse,
                ReadTensorResponse,
                WriteTensorResponse,
                SetEventResponse,
                WritePointersResponse,
                WriteExecutableResponse,
                ExecuteResponse,
                ExitResponse,
            ),
        ):
            self.set_mesg_future(mesg)
            return
        else:
            self.raise_exception(ValueError(f"Invalid message type: {type(mesg)=}"))

    def handle_execute(self, mesg: ExecuteRequest) -> None:
        is_local = mesg.src_device_uid == self.device.uid

        # ------------------------------------------------------------------
        # process the execution request
        # ------------------------------------------------------------------
        # read executable from local memory
        executable, err = self.device.read(mesg.executable_addr)
        if err != ErrCode.ESUCC:
            if is_local:
                self.raise_exception(
                    RuntimeError(f"Failed to read executable from local memory. {err=}")
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
            return
        self.assert_cond(isinstance(executable, Executable), f"{type(executable)=}")

        # read args from local memory
        args, err = self.device.read(mesg.args_addr)
        if err != ErrCode.ESUCC:
            if is_local:
                self.raise_exception(
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
            return
        self.assert_cond(isinstance(args, Pointers), f"{type(args)=}")

        # execute the executable
        logger.debug(f"{self.device.uid}: Executing {executable} with {args}")
        err = executable(self, args)

        # check result
        if err != ErrCode.ESUCC:
            if is_local:
                self.raise_exception(RuntimeError(f"Failed to execute executable. {err=}"))
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
        else:
            # If this is a remote request and execution succeeded, send a success response
            if not is_local:
                self.device.output_queues[mesg.src_device_uid].put(
                    ExecuteResponse(
                        uid=mesg.uid,
                        src_device_uid=mesg.dst_device_uid,
                        dst_device_uid=mesg.src_device_uid,
                        err=ErrCode.ESUCC,
                    )
                )
        logger.debug(f"{self.device.uid}: Executed {executable} with {args}")

    # ==========================================================================
    # input message handler
    # ==========================================================================
    def message_handling_thread_entry(self, is_test: bool = False) -> None:
        logger.info(f"{self.device.uid}: Running input handler of {self=}")
        while not self.quit_event.is_set():
            # get a message from the input queue
            # use non-blocking get so that we can check `self.quit_event`
            try:
                mesg = self.device.input_queue.get_nowait()
            except queue.Empty:
                if is_test:
                    logger.debug(f"{self.device.uid}: exit testing because input queue is empty")
                    break
                time.sleep(0.01)
                continue

            logger.trace(f"{self.device.uid}: handling {mesg=}")
            assert mesg.dst_device_uid == self.device.uid, (
                f"{mesg.dst_device_uid=} != {self.device.uid=}"
            )

            self.handle_message(mesg)

            if is_test:
                logger.debug(f"{self.device.uid}: exit testing after handling {mesg=}")
                break

    # ==========================================================================
    # execution message handler
    # ==========================================================================
    # we choose to handel execution messages in a separate thread because
    # there will be "waiting for events" operations in the executable, and we
    # don't want to block the input handler thread
    def execution_handling_thread_entry(self, is_test: bool = False) -> None:
        logger.info(f"{self.device.uid}: Running execution handler of {self=}")
        while not self.quit_event.is_set():
            # get a message from the execution queue
            # use non-blocking get so that we can check `self.quit_event`
            try:
                mesg = self.execution_queue.get_nowait()
            except queue.Empty:
                if is_test:
                    logger.debug(f"{self.device.uid}: exit testing because execution queue is empty")
                    break
                time.sleep(0.01)
                continue

            logger.trace(f"{self.device.uid}: handling {mesg=}")
            assert mesg.dst_device_uid == self.device.uid, (
                f"{mesg.dst_device_uid=} != {self.device.uid=}"
            )

            self.handle_execute(mesg)

            if is_test:
                logger.debug(f"{self.device.uid}: exit testing after handling {mesg=}")
                break
