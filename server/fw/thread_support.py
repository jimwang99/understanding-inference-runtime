import queue
import threading

from abc import ABC
from loguru import logger
from typing import Callable

from server.mock_hardware import MockDevice

class _PropagateException(ABC):
    def __init__(self) -> None:
        self.exceptions: queue.Queue[Exception] = queue.Queue()

    def check_exceptions(self) -> None:
        if not self.exceptions.empty():
            while not self.exceptions.empty():
                e = self.exceptions.get()
                logger.error(f"Exception in thread: {e}")
            raise RuntimeError(f"Exception in thread: {e}")

    def raise_exception(self, e: Exception) -> None:
        self.exceptions.put(e)
        raise e

    def assert_cond(self, cond: bool, msg: str) -> None:
        if not cond:
            self.raise_exception(AssertionError(msg))
        assert cond, msg


class _QuitEvent(ABC):
    def __init__(self) -> None:
        self.quit_event = threading.Event()
    
    def quit(self) -> None:
        self.quit_event.set()

    def is_quit(self) -> bool:
        return self.quit_event.is_set()


class ThreadSupport(_QuitEvent, _PropagateException, ABC):
    def __init__(self, device: MockDevice) -> None:
        _QuitEvent.__init__(self)
        _PropagateException.__init__(self)
        ABC.__init__(self)

        self.device = device
        self.threads: dict[str, threading.Thread] = {}
    
    def register_thread(self, name: str, target: Callable) -> None:
        self.threads[name] = threading.Thread(target=target)

    def start(self) -> None:
        for name, thread in self.threads.items():
            logger.info(f"{self.device.uid}: Starting {name} thread")
            thread.start()

    def join(self) -> None:
        for name, thread in self.threads.items():
            logger.info(f"{self.device.uid}: Joining {name} thread")
            thread.join()

        self.check_exceptions()
