import numpy as np
import unittest

from loguru import logger

from server.error_code import ErrCode
from server.mock_hardware import MockDevice
from server.mock_firmware import MockFirmware
from server.kernel import Kernel

class MockBootProgramSimple(Kernel):
    def __init__(self) -> None:
        super().__init__("MockBootProgramSimple")

    def _execute(self) -> ErrCode:
        t0_addr, t0 = self.create_tensor("test_tensor_0", np.array([[1, 2, 3], [4, 5, 6]]))
        e0_addr, e0 = self.create_event("test_event_0")
        p0_addr, p0 = self.create_pointers("test_pointers_0", [str(t0_addr), str(e0_addr)])

        self.quit()
        return ErrCode.ESUCC


class TestMockModelSimple(unittest.TestCase):
    def test_boot_program_simple(self):
        self.device_host = MockDevice(uid="host", dtype="CPU", max_mem_size="1GB")
        self.host = MockFirmware(self.device_host)
        self.host.load_boot_program(MockBootProgramSimple())
        self.host.start()
        self.host.join()
    
    # def test_load_program_to_gpu(self):
    #     self.device_host = MockDevice(uid="host", dtype="CPU", max_mem_size="1GB")
    #     self.device_gpu = MockDevice(uid="gpu", dtype="GPU", max_mem_size="1GB")
    #     self.device_host.connect(self.device_gpu, "gpu")
    #     self.device_gpu.connect(self.device_host, "host")

    #     self.host = MockFirmware(self.device_host)
    #     self.gpu = MockFirmware(self.device_gpu)

    #     self.host.load_boot_program(MockBootProgramSimple())

    #     self.host.start()
    #     self.gpu.start()

    #     self.host.join()
    #     self.gpu.join()
