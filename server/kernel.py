import numpy as np

from abc import ABC
from loguru import logger

from server.error_code import ErrCode
from server.data_types import Executable, Tensor, Event, Pointers
from server.fw.future_host import MessageFuture


class Kernel(Executable, ABC):
    def __init__(self, name: str):
        super().__init__(name)

    def quit(self) -> None:
        assert self.fw is not None
        self.fw.quit()
    
    # ==========================================================================
    # local
    # ==========================================================================

    def _create_any_local(self, data: Tensor | Event | Pointers | Executable) -> int:
        assert self.fw is not None
        addr, err = self.fw.device.alloc(data.size())
        if err != ErrCode.ESUCC:
            self.fw.raise_exception(
                RuntimeError(f"Failed to allocate memory for {data.mtype} {data.name}")
            )
        self.fw.device.write(addr, data)
        return addr

    def _delete_any_local(self, addr: int, dtype: type[Tensor] | type[Event] | type[Pointers] | type[Executable]) -> None:
        assert self.fw is not None
        data, err = self.fw.device.read(addr)
        self.fw.assert_cond(err == ErrCode.ESUCC, f"Failed to find {dtype.__name__} at {addr}")
        self.fw.assert_cond(isinstance(data, dtype), f"{type(data)=} is not {dtype=}")
        err = self.fw.device.free(addr)
        if err != ErrCode.ESUCC:
            self.fw.raise_exception(RuntimeError(f"Failed to free memory at {addr}"))

    def create_tensor(
        self,
        name: str,
        array: np.ndarray,
    ) -> tuple[int, Tensor]:
        assert self.fw is not None # to satisfy type checker
        logger.trace(f"{self.fw.device.uid}: Creating local tensor {name}")
        tensor = Tensor(name, array)
        addr = self._create_any_local(tensor)
        return addr, tensor

    def create_event(self, name: str) -> tuple[int, Event]:
        assert self.fw is not None # to satisfy type checker
        logger.trace(f"{self.fw.device.uid}: Creating local event {name}")
        event = Event(name)
        addr = self._create_any_local(event)
        return addr, event

    def create_pointers(self, name: str, ptrs: list[str]) -> tuple[int, Pointers]:
        assert self.fw is not None # to satisfy type checker
        logger.trace(f"{self.fw.device.uid}: Creating local pointers {name}")
        pointers = Pointers(name, ptrs)
        addr = self._create_any_local(pointers)
        return addr, pointers

    def create_executable(self, name: str, executable: Executable) -> int:
        assert self.fw is not None # to satisfy type checker
        logger.trace(f"{self.fw.device.uid}: Creating local executable {name}")
        return self._create_any_local(executable)

    def delete_tensor(self, addr: int) -> None:
        assert self.fw is not None # to satisfy type checker
        logger.trace(f"{self.fw.device.uid}: Deleting local tensor {addr}")
        self._delete_any_local(addr, Tensor)

    def delete_event(self, addr: int) -> None:
        assert self.fw is not None # to satisfy type checker
        logger.trace(f"{self.fw.device.uid}: Deleting local event {addr}")
        self._delete_any_local(addr, Event)

    def delete_pointers(self, addr: int) -> None:
        assert self.fw is not None # to satisfy type checker
        logger.trace(f"{self.fw.device.uid}: Deleting local pointers {addr}")
        self._delete_any_local(addr, Pointers)

    def delete_executable(self, addr: int) -> None:
        assert self.fw is not None # to satisfy type checker
        logger.trace(f"{self.fw.device.uid}: Deleting executable {addr}")
        self._delete_any_local(addr, Executable)

    # ==========================================================================
    # remote
    # ==========================================================================
    
    def _create_any_remote(self, data: Tensor | Event | Pointers | Executable, remote_device_name: str) -> MessageFuture:
        assert self.fw is not None # to satisfy type checker
        return self.fw.alloc_remote_nb(data.size(), remote_device_name)
    
    def _delete_any_remote(self, addr: int, remote_device_name: str) -> MessageFuture:
        assert self.fw is not None # to satisfy type checker
        return self.fw.free_remote_nb(addr, remote_device_name)
    
    def create_remote_tensor(self, name: str, array: np.ndarray, remote_device: str) -> MessageFuture:
        assert self.fw is not None # to satisfy type checker
        logger.trace(f"{self.fw.device.uid}: Creating remote tensor {name} on {remote_device}")
        return self._create_any_remote(Tensor(name, array), remote_device)

    def create_remote_event(self, name: str, remote_device: str) -> MessageFuture:
        assert self.fw is not None # to satisfy type checker
        logger.trace(f"{self.fw.device.uid}: Creating remote event {name} on {remote_device}")
        return self._create_any_remote(Event(name), remote_device)
    
    def create_remote_pointers(self, name: str, ptrs: list[str], remote_device: str) -> MessageFuture:
        assert self.fw is not None # to satisfy type checker
        logger.trace(f"{self.fw.device.uid}: Creating remote pointers {name} on {remote_device}")
        return self._create_any_remote(Pointers(name, ptrs), remote_device)
    
    def create_remote_executable(self, name: str, executable: Executable, remote_device: str) -> MessageFuture:
        assert self.fw is not None # to satisfy type checker
        logger.trace(f"{self.fw.device.uid}: Creating remote executable {name} on {remote_device}")
        return self._create_any_remote(executable, remote_device)
    
    def delete_remote_tensor(self, addr: int, remote_device: str) -> MessageFuture:
        assert self.fw is not None # to satisfy type checker
        logger.trace(f"{self.fw.device.uid}: Deleting remote tensor {addr} on {remote_device}")
        return self._delete_any_remote(addr, remote_device)
    
    def delete_remote_event(self, addr: int, remote_device: str) -> MessageFuture:
        assert self.fw is not None # to satisfy type checker
        logger.trace(f"{self.fw.device.uid}: Deleting remote event {addr} on {remote_device}")
        return self._delete_any_remote(addr, remote_device)
    
    def delete_remote_pointers(self, addr: int, remote_device: str) -> MessageFuture:
        assert self.fw is not None # to satisfy type checker
        logger.trace(f"{self.fw.device.uid}: Deleting remote pointers {addr} on {remote_device}")
        return self._delete_any_remote(addr, remote_device)
    
    def delete_remote_executable(self, addr: int, remote_device: str) -> MessageFuture:
        assert self.fw is not None # to satisfy type checker
        logger.trace(f"{self.fw.device.uid}: Deleting remote executable {addr} on {remote_device}")
        return self._delete_any_remote(addr, remote_device)
    
    def extract_future(self, future: MessageFuture) -> int | Tensor | None:
        assert self.fw is not None # to satisfy type checker
        return self.fw.extract_mesg_future(future)
