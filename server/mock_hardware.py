from __future__ import annotations
import multiprocessing as mp

from dataclasses import dataclass
from loguru import logger

from .error_code import ErrCode
from .rpc_message import RPCMessage

KB = 1024
MB = 1024 * 1024
GB = 1024 * 1024 * 1024
TB = 1024 * 1024 * 1024 * 1024


@dataclass
class MemBlock:
    addr: int
    size: int
    raw: bytes = b""

    def __repr__(self) -> str:
        return f"<Memory: {self.addr:08x} {self.size / MB:.2f}MB>"


class MockDevice:
    def __init__(
        self,
        uid: str,
        dtype: str,
        max_mem_size: int | str,
    ):
        self.uid = uid
        self.dtype = dtype
        if isinstance(max_mem_size, str):
            if max_mem_size.endswith("TB"):
                self.max_mem_size = int(max_mem_size[:-2]) * TB
            elif max_mem_size.endswith("GB"):
                self.max_mem_size = int(max_mem_size[:-2]) * GB
            elif max_mem_size.endswith("MB"):
                self.max_mem_size = int(max_mem_size[:-2]) * MB
            elif max_mem_size.endswith("KB"):
                self.max_mem_size = int(max_mem_size[:-2]) * KB
            else:
                raise ValueError(f"Invalid max_mem_size: {max_mem_size=}")
        else:
            self.max_mem_size = max_mem_size

        self.mem_blocks: dict[int, MemBlock] = {}
        self.remaining_mem_size = self.max_mem_size
        self.cur_mem_block_uid = (
            0  # use a unique ID to represent the address in this mock device
        )

        self.connected_devices: dict[str, MockDevice] = {}
        self.device_alias_to_uid: dict[str, str] = {}

        self.input_queue: mp.Queue[RPCMessage] = mp.Queue()
        self.output_queues: dict[str, mp.Queue[RPCMessage]] = {}
        self.execution_queue: mp.Queue[RPCMessage] = mp.Queue()

    def __str__(self) -> str:
        return self.uid

    def __repr__(self) -> str:
        s = f"<{self.dtype}: {self.uid} {self.max_mem_size / GB:.2f}GB>"
        for device_alias in self.device_alias_to_uid.keys():
            s += f"\n  - {device_alias}: {self.get_device_by_alias(device_alias).uid}"
        s += ">"
        return s

    def connect(self, device: MockDevice, alias: str) -> None:
        assert device.uid not in self.connected_devices
        assert alias not in self.device_alias_to_uid
        self.connected_devices[device.uid] = device
        self.device_alias_to_uid[alias] = device.uid
        device.output_queues[self.uid] = self.input_queue

    def get_device_by_alias(self, alias: str) -> MockDevice:
        return self.connected_devices[self.device_alias_to_uid[alias]]

    def get_output_queue_by_alias(self, alias: str) -> mp.Queue[RPCMessage]:
        return self.output_queues[self.device_alias_to_uid[alias]]

    def alloc(self, size: int) -> tuple[int, ErrCode]:
        logger.debug(f"{self.uid}: Allocating memory of {size=} bytes")
        if size > self.remaining_mem_size:
            logger.error(
                f"{self.uid}: Not enough memory {size=}>{self.remaining_mem_size=}"
            )
            return 0, ErrCode.ENOMEM

        mb = MemBlock(
            addr=self.cur_mem_block_uid,
            size=size,
        )
        self.remaining_mem_size -= size
        self.cur_mem_block_uid += 1

        self.mem_blocks[mb.addr] = mb
        return mb.addr, ErrCode.ESUCC

    def free(self, addr: int) -> ErrCode:
        logger.debug(f"{self.uid}: Freeing memory {addr=}")
        try:
            mb = self.mem_blocks[addr]
            self.remaining_mem_size += mb.size
            del self.mem_blocks[mb.addr]
            logger.debug(
                f"{self.uid}: Remaining memory {self.remaining_mem_size} bytes"
            )
            return ErrCode.ESUCC
        except KeyError:
            logger.error(f"{self.uid}: Memory {addr=} not found")
            return ErrCode.ENODATA

    def read(
        self,
        addr: int,
    ) -> tuple[bytes, ErrCode]:
        logger.debug(f"{self.uid}: Reading from {addr=}")
        try:
            mb = self.mem_blocks[addr]
        except KeyError:
            logger.error(f"{self.uid}: Memory {addr=} not found")
            return b"", ErrCode.ENODATA

        if mb.raw is None:
            logger.error(f"{self.uid}: Memory {addr=} is not initialized")
            return b"", ErrCode.ENODATA

        return mb.raw, ErrCode.ESUCC

    def write(
        self,
        addr: int,
        raw: bytes,
    ) -> ErrCode:
        logger.debug(f"{self.uid}: Writing bytes to {addr=}")
        try:
            mb = self.mem_blocks[addr]
        except KeyError:
            logger.error(f"{self.uid}: Memory {addr=} not found")
            return ErrCode.ENODATA

        mb.raw = raw
        return ErrCode.ESUCC


class MockSystem:
    def __init__(self, uid: str) -> None:
        self.uid = uid
        self.nodes: dict[str, MockNode] = {}

    def __repr__(self) -> str:
        s = f"<system {self.uid}>"
        for node in self.nodes.values():
            s += f"\n{node}"
        s += "\n</system>"
        return s

    def add_node(self, node: MockNode) -> None:
        assert node.system is self
        assert node.uid not in self.nodes
        self.nodes[node.uid] = node


class MockNode:
    def __init__(self, uid: str, system: MockSystem) -> None:
        self.system = system
        self.uid = uid
        self.devices: dict[str, MockDevice] = {}

    def __repr__(self) -> str:
        s = f"<node {self.uid}>\n"
        for device in self.devices.values():
            s += f"{device}\n"
        s += "</node>"
        return s

    def add_device(self, device: MockDevice) -> None:
        assert device.uid not in self.devices
        self.devices[device.uid] = device


class MockNodeRing(MockNode):
    def __init__(self, uid: str, num_gpus: int, system: MockSystem) -> None:
        super().__init__(uid, system)
        self.num_gpus = num_gpus

        self.cpu = MockDevice(uid + ".cpu", "CPU", 1 * TB)
        self.gpus = [
            MockDevice(uid + f".gpu{i}", "GPU", 64 * GB) for i in range(num_gpus)
        ]

        self.add_device(self.cpu)
        for i, gpu in enumerate(self.gpus):
            self.add_device(gpu)
            self.cpu.connect(gpu, f"local_gpu{i}")
            gpu.connect(self.cpu, "local_cpu")

        for i, gpu in enumerate(self.gpus):
            gpu.connect(self.gpus[(i - 1) % num_gpus], "prev_local_gpu")
            gpu.connect(self.gpus[(i + 1) % num_gpus], "next_local_gpu")


class MockSystemSingleRing(MockSystem):
    def __init__(self, uid: str, num_gpus_per_node: int) -> None:
        super().__init__(uid)
        n = MockNodeRing(uid + ".node", num_gpus_per_node, self)
        self.add_node(n)


def make_mock_system(topology: str, name: str, **kwargs) -> MockSystemSingleRing:
    if topology == "single_ring":
        return MockSystemSingleRing(name, kwargs["num_gpus_per_node"])
    else:
        raise ValueError(f"Invalid topology: {topology=}")


if __name__ == "__main__":
    system = make_mock_system("single_ring", "test", num_gpus_per_node=3)
    print(system)
