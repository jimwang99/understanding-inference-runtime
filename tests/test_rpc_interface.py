import unittest

from loguru import logger

from server.mock_hardware import MockDevice
from server.mock_firmware import MockFirmware


class TestRPCInterface(unittest.TestCase):
    def setUp(self):
        self.device_prod = MockDevice(uid="prod", dtype="PRODUCER", max_mem_size="1TB")
        self.device_cons = MockDevice(uid="cons", dtype="CONSUMER", max_mem_size="1TB")
        self.device_prod.connect(self.device_cons, "cons")
        self.device_cons.connect(self.device_prod, "prod")

        self.prod = MockFirmware(self.device_prod)
        self.cons = MockFirmware(self.device_cons)

    def test_blocking_alloc_free(self):
        self.prod._start_threads()
        self.cons.start()

        addr = self.prod.alloc_remote(8, "cons")
        self.prod.free_remote(addr, "cons")

        self.prod.exit_remote("test ends", "cons")
        self.prod.quit()
        self.prod._join_threads()
        self.cons.join()

    def test_non_blocking_alloc_free(self):
        self.prod._start_threads()
        self.cons.start()

        f0 = self.prod.alloc_remote_nb(8, "cons")
        f1 = self.prod.alloc_remote_nb(16, "cons")
        f2 = self.prod.alloc_remote_nb(32, "cons")
        f3 = self.prod.alloc_remote_nb(64, "cons")

        f0.wait()
        f4 = self.prod.free_remote_nb(f0.rsp_mesg.addr, "cons")
        f1.wait()
        f5 = self.prod.free_remote_nb(f1.rsp_mesg.addr, "cons")
        f2.wait()
        f6 = self.prod.free_remote_nb(f2.rsp_mesg.addr, "cons")
        f3.wait()
        f7 = self.prod.free_remote_nb(f3.rsp_mesg.addr, "cons")

        f4.wait()
        f5.wait()
        f6.wait()
        f7.wait()

        f8 = self.prod.exit_remote_nb("test ends", "cons")
        f8.wait()

        self.prod.quit()
        self.prod._join_threads()
        self.cons.join()

    def test_blocking_read_write(self):
        self.prod._start_threads()
        self.cons.start()

        data = b"hello"

        addr = self.prod.alloc_remote(len(data), "cons")
        self.prod.write_remote(addr, data, "cons")
        data_rsp = self.prod.read_remote(addr, "cons")
        self.assertEqual(data_rsp, data)

        self.prod.exit_remote("test ends", "cons")
        self.prod.quit()
        self.prod._join_threads()
        self.cons.join()

    def test_non_blocking_read_write(self):
        self.prod._start_threads()
        self.cons.start()

        data = b"hello"

        f0 = self.prod.alloc_remote_nb(len(data), "cons")
        f0.wait()
        addr = f0.rsp_mesg.addr

        f1 = self.prod.write_remote_nb(addr, data, "cons")
        f2 = self.prod.read_remote_nb(addr, "cons")

        f1.wait()
        f2.wait()

        self.assertEqual(f2.rsp_mesg.data, data)

        f3 = self.prod.exit_remote_nb("test ends", "cons")
        f3.wait()

        self.prod.quit()
        self.prod._join_threads()
        self.cons.join()


if __name__ == "__main__":
    unittest.main()
