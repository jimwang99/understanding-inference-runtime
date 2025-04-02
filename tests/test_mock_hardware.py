import unittest

from server.mock_hardware import make_mock_system

from loguru import logger


class TestMockHardware(unittest.TestCase):
    def test_single_ring_8x(self):
        system = make_mock_system("single_ring", "test_system", num_gpus_per_node=8)
        logger.info(system)


if __name__ == "__main__":
    unittest.main()
