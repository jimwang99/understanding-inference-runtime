from __future__ import annotations

import numpy as np
import threading
import time
import pickle
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from server.error_code import ErrCode

if TYPE_CHECKING:
    from server.mock_firmware import MockFirmware


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
        return 8 + 8

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
            time.sleep(0.01)
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
        return len(self.ptrs) * 8 + 8


class Executable(ABC):
    mtype = "Executable"

    def __init__(self, name: str):
        self.name = name
        self.fw: MockFirmware | None = None
        self.args: Pointers | None = None

    def __repr__(self) -> str:
        s = f"<Executable: {self.name}>"
        return s

    def size(self) -> int:
        return len(pickle.dumps(self))

    def __call__(self, fw: MockFirmware, args: Pointers) -> ErrCode:
        self.fw = fw 
        self.args = args
        return self._execute()

    @abstractmethod
    def _execute(self) -> ErrCode:
        return ErrCode.ENOENT
