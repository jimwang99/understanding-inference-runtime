from __future__ import annotations

from abc import ABC

from server.mock_hardware import MockDevice
from ..data_types import Tensor, Pointers, Executable

from .thread_support import ThreadSupport
from .future_host import MessageFutureHost, MessageFuture
from .rpc_message import (
    AllocRequest,
    FreeRequest,
    ExitRequest,
    ReadTensorRequest,
    WriteTensorRequest,
    SetEventRequest,
    WritePointersRequest,
    WriteExecutableRequest,
    ExecuteRequest,
)


class RPCClient(MessageFutureHost, ThreadSupport, ABC):
    def __init__(self, device: MockDevice) -> None:
        MessageFutureHost.__init__(self, device)
        ThreadSupport.__init__(self, device)
        ABC.__init__(self)

        self.device = device

    def _get_remote_device(self, remote_device_name: str) -> MockDevice:
        if remote_device_name in self.device.connected_devices.keys():
            return self.device.connected_devices[remote_device_name]
        else:
            return self.device.get_device_by_alias(remote_device_name)

    def alloc_remote_nb(self, size: int, remote_device_name: str) -> MessageFuture:
        remote_device = self._get_remote_device(remote_device_name)
        mesg = AllocRequest(
            src_device_uid=self.device.uid,
            dst_device_uid=remote_device.uid,
            size=size,
        )
        self.device.output_queues[remote_device.uid].put(mesg)
        return self.create_mesg_future(mesg)

    def free_remote_nb(self, addr: int, remote_device_name: str) -> MessageFuture:
        remote_device = self._get_remote_device(remote_device_name)
        mesg = FreeRequest(
            src_device_uid=self.device.uid,
            dst_device_uid=remote_device.uid,
            addr=addr,
        )
        self.device.output_queues[remote_device.uid].put(mesg)
        return self.create_mesg_future(mesg)

    def exit_remote_nb(self, reason: str, remote_device_name: str) -> MessageFuture:
        remote_device = self._get_remote_device(remote_device_name)
        mesg = ExitRequest(
            src_device_uid=self.device.uid,
            dst_device_uid=remote_device.uid,
            reason=reason,
        )
        self.device.output_queues[remote_device.uid].put(mesg)
        return self.create_mesg_future(mesg)

    def read_tensor_remote_nb(
        self, addr: int, remote_device_name: str
    ) -> MessageFuture:
        remote_device = self._get_remote_device(remote_device_name)
        mesg = ReadTensorRequest(
            src_device_uid=self.device.uid,
            dst_device_uid=remote_device.uid,
            addr=addr,
        )
        self.device.output_queues[remote_device.uid].put(mesg)
        return self.create_mesg_future(mesg)

    def write_tensor_remote_nb(
        self, addr: int, tensor: Tensor, remote_device_name: str
    ) -> MessageFuture:
        remote_device = self._get_remote_device(remote_device_name)
        mesg = WriteTensorRequest(
            src_device_uid=self.device.uid,
            dst_device_uid=remote_device.uid,
            addr=addr,
            data=tensor,
        )
        self.device.output_queues[remote_device.uid].put(mesg)
        return self.create_mesg_future(mesg)

    def set_event_remote_nb(self, addr: int, remote_device_name: str) -> MessageFuture:
        remote_device = self._get_remote_device(remote_device_name)
        mesg = SetEventRequest(
            src_device_uid=self.device.uid,
            dst_device_uid=remote_device.uid,
            addr=addr,
        )
        self.device.output_queues[remote_device.uid].put(mesg)
        return self.create_mesg_future(mesg)

    def write_pointers_remote_nb(
        self, addr: int, pointers: Pointers, remote_device_name: str
    ) -> MessageFuture:
        remote_device = self._get_remote_device(remote_device_name)
        mesg = WritePointersRequest(
            src_device_uid=self.device.uid,
            dst_device_uid=remote_device.uid,
            addr=addr,
            data=pointers,
        )
        self.device.output_queues[remote_device.uid].put(mesg)
        return self.create_mesg_future(mesg)

    def write_executable_remote_nb(
        self, addr: int, executable: Executable, remote_device_name: str
    ) -> MessageFuture:
        remote_device = self._get_remote_device(remote_device_name)
        mesg = WriteExecutableRequest(
            src_device_uid=self.device.uid,
            dst_device_uid=remote_device.uid,
            addr=addr,
            data=executable,
        )
        self.device.output_queues[remote_device.uid].put(mesg)
        return self.create_mesg_future(mesg)

    def execute_remote_nb(
        self, executable_addr: int, args_addr: int, remote_device_name: str
    ) -> MessageFuture:
        remote_device = self._get_remote_device(remote_device_name)
        mesg = ExecuteRequest(
            src_device_uid=self.device.uid,
            dst_device_uid=remote_device.uid,
            executable_addr=executable_addr,
            args_addr=args_addr,
        )
        self.device.output_queues[remote_device.uid].put(mesg)
        return self.create_mesg_future(mesg)
