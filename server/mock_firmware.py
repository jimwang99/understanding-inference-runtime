from __future__ import annotations

from loguru import logger

from .data_types import Executable, Pointers
from .error_code import ErrCode
from .mock_hardware import MockDevice

from .fw.rpc_message import ExecuteRequest
from .fw.rpc_host import RPCHost
from .fw.rpc_client import RPCClient


class MockFirmware(RPCClient, RPCHost):
    def __init__(self, device: MockDevice) -> None:
        RPCClient.__init__(self, device=device)
        RPCHost.__init__(self, device=device)

        self.device = device

    def load_boot_program(
        self, executable: Executable, args: Pointers | None = None
    ) -> None:
        logger.debug(
            f"{self.device.uid}: Loading boot program {executable} with {args}"
        )

        executable_addr, err = self.device.alloc(executable.size())
        assert err == ErrCode.ESUCC

        if args is None:
            args = Pointers(name="", ptrs=[])

        args_addr, err = self.device.alloc(args.size())
        assert err == ErrCode.ESUCC

        self.device.write(executable_addr, executable)
        self.device.write(args_addr, args)

        self.execution_queue.put(
            ExecuteRequest(
                src_device_uid=self.device.uid,
                dst_device_uid=self.device.uid,
                uid=-1,
                executable_addr=executable_addr,
                args_addr=args_addr,
            )
        )
