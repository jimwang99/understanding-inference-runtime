import unittest
import numpy as np
from unittest.mock import MagicMock

from server.kernel import Kernel
from server.data_types import Tensor, Event, Pointers, Executable
from server.mock_hardware import MockDevice
from server.mock_firmware import MockFirmware
from server.fw.future_host import MessageFuture
from server.error_code import ErrCode


# Create a concrete implementation of Kernel for testing
class ConcreteKernel(Kernel):
    def __init__(self, name="test_kernel"):
        super().__init__(name)
        
    def _execute(self):
        return ErrCode.ESUCC


# Create a concrete implementation of Executable for testing
class MockExecutable(Executable):
    def __init__(self, name="test_executable", return_code=ErrCode.ESUCC):
        super().__init__(name)
        self.return_code = return_code
        
    def _execute(self):
        return self.return_code


class TestKernel(unittest.TestCase):
    def setUp(self):
        # Setup mock device and firmware
        self.mock_device = MagicMock(spec=MockDevice)
        self.mock_device.uid = "test_device"
        
        # Create a concrete kernel for testing
        self.kernel = ConcreteKernel()
        
        # Setup mock firmware
        self.mock_fw = MagicMock(spec=MockFirmware)
        self.mock_fw.device = self.mock_device
        
        # Attach firmware to kernel
        self.kernel.fw = self.mock_fw
        
    def test_init(self):
        """Test kernel initialization."""
        kernel = ConcreteKernel("test_name")
        self.assertEqual(kernel.name, "test_name")
        
    def test_quit(self):
        """Test quit method."""
        self.kernel.quit()
        self.mock_fw.quit.assert_called_once()
        
    # ===========================================================================
    # Local Operations Tests
    # ===========================================================================
    
    def test_create_tensor(self):
        """Test creating a local tensor."""
        # Setup
        test_array = np.array([[1, 2, 3], [4, 5, 6]])
        self.mock_device.alloc.return_value = (0x1000, ErrCode.ESUCC)
        
        # Execute
        addr, tensor = self.kernel.create_tensor("test_tensor", test_array)
        
        # Verify
        self.mock_device.alloc.assert_called_once()
        self.mock_device.write.assert_called_once()
        self.assertEqual(addr, 0x1000)
        self.assertEqual(tensor.name, "test_tensor")
        np.testing.assert_array_equal(tensor.data, test_array)
        
    def test_create_event(self):
        """Test creating a local event."""
        # Setup
        self.mock_device.alloc.return_value = (0x2000, ErrCode.ESUCC)
        
        # Execute
        addr, event = self.kernel.create_event("test_event")
        
        # Verify
        self.mock_device.alloc.assert_called_once()
        self.mock_device.write.assert_called_once()
        self.assertEqual(addr, 0x2000)
        self.assertEqual(event.name, "test_event")
        self.assertFalse(event.value)  # Default value should be False

    def test_create_pointers(self):
        """Test creating local pointers."""
        # Setup
        test_ptrs = ["0x1000", "0x2000"]
        self.mock_device.alloc.return_value = (0x3000, ErrCode.ESUCC)
        
        # Execute
        addr, pointers = self.kernel.create_pointers("test_pointers", test_ptrs)
        
        # Verify
        self.mock_device.alloc.assert_called_once()
        self.mock_device.write.assert_called_once()
        self.assertEqual(addr, 0x3000)
        self.assertEqual(pointers.name, "test_pointers")
        self.assertEqual(pointers.ptrs, test_ptrs)
        
    def test_create_executable(self):
        """Test creating a local executable."""
        # Setup
        executable = MockExecutable()
        self.mock_device.alloc.return_value = (0x4000, ErrCode.ESUCC)
        
        # Execute
        addr = self.kernel.create_executable("test_executable", executable)
        
        # Verify
        self.mock_device.alloc.assert_called_once()
        self.mock_device.write.assert_called_once()
        self.assertEqual(addr, 0x4000)
        
    def test_delete_tensor(self):
        """Test deleting a local tensor."""
        # Setup
        tensor = Tensor("test_tensor", np.array([1, 2, 3]))
        self.mock_device.read.return_value = (tensor, ErrCode.ESUCC)
        self.mock_device.free.return_value = ErrCode.ESUCC
        
        # Execute
        self.kernel.delete_tensor(0x1000)
        
        # Verify
        self.mock_device.read.assert_called_once_with(0x1000)
        self.mock_device.free.assert_called_once_with(0x1000)

    def test_delete_event(self):
        """Test deleting a local event."""
        # Setup
        event = Event("test_event")
        self.mock_device.read.return_value = (event, ErrCode.ESUCC)
        self.mock_device.free.return_value = ErrCode.ESUCC
        
        # Execute
        self.kernel.delete_event(0x2000)
        
        # Verify
        self.mock_device.read.assert_called_once_with(0x2000)
        self.mock_device.free.assert_called_once_with(0x2000)

    def test_delete_pointers(self):
        """Test deleting local pointers."""
        # Setup
        pointers = Pointers("test_pointers", ["0x1000", "0x2000"])
        self.mock_device.read.return_value = (pointers, ErrCode.ESUCC)
        self.mock_device.free.return_value = ErrCode.ESUCC
        
        # Execute
        self.kernel.delete_pointers(0x3000)
        
        # Verify
        self.mock_device.read.assert_called_once_with(0x3000)
        self.mock_device.free.assert_called_once_with(0x3000)
        
    def test_delete_executable(self):
        """Test deleting a local executable."""
        # Setup
        executable = MockExecutable()
        self.mock_device.read.return_value = (executable, ErrCode.ESUCC)
        self.mock_device.free.return_value = ErrCode.ESUCC
        
        # Execute
        self.kernel.delete_executable(0x4000)
        
        # Verify
        self.mock_device.read.assert_called_once_with(0x4000)
        self.mock_device.free.assert_called_once_with(0x4000)
        
    def test_delete_any_local_error(self):
        """Test error handling in _delete_any_local."""
        # Setup - bad read
        self.mock_device.read.return_value = (None, ErrCode.ENODATA)
        self.mock_fw.assert_cond.side_effect = AssertionError
        
        # Execute and verify assertion
        with self.assertRaises(AssertionError):
            self.kernel._delete_any_local(0x5000, Tensor)
        
        # Setup - wrong type
        event = Event("test_event")
        self.mock_device.read.return_value = (event, ErrCode.ESUCC)
        
        # Execute and verify assertion
        with self.assertRaises(AssertionError):
            self.kernel._delete_any_local(0x5000, Tensor)
            
        # Setup - free error
        tensor = Tensor("test_tensor", np.array([1, 2, 3]))
        self.mock_device.read.return_value = (tensor, ErrCode.ESUCC) 
        self.mock_device.free.return_value = ErrCode.ENODATA
        
        # Execute and verify exception
        with self.assertRaises(RuntimeError):
            self.kernel._delete_any_local(0x5000, Tensor)
            
    # ===========================================================================
    # Remote Operations Tests
    # ===========================================================================
    
    def test_create_remote_tensor(self):
        """Test creating a remote tensor."""
        # Setup
        test_array = np.array([[1, 2, 3], [4, 5, 6]])
        mock_future = MagicMock(spec=MessageFuture)
        self.mock_fw.alloc_remote_nb.return_value = mock_future
        
        # Execute
        future = self.kernel.create_remote_tensor("test_tensor", test_array, "remote_device")
        
        # Verify
        self.assertEqual(future, mock_future)
        self.mock_fw.alloc_remote_nb.assert_called_once()
        
    def test_create_remote_event(self):
        """Test creating a remote event."""
        # Setup
        mock_future = MagicMock(spec=MessageFuture)
        self.mock_fw.alloc_remote_nb.return_value = mock_future
        
        # Execute
        future = self.kernel.create_remote_event("test_event", "remote_device")
        
        # Verify
        self.assertEqual(future, mock_future)
        self.mock_fw.alloc_remote_nb.assert_called_once()
        
    def test_create_remote_pointers(self):
        """Test creating remote pointers."""
        # Setup
        test_ptrs = ["0x1000", "0x2000"]
        mock_future = MagicMock(spec=MessageFuture)
        self.mock_fw.alloc_remote_nb.return_value = mock_future
        
        # Execute
        future = self.kernel.create_remote_pointers("test_pointers", test_ptrs, "remote_device")
        
        # Verify
        self.assertEqual(future, mock_future)
        self.mock_fw.alloc_remote_nb.assert_called_once()
        
    def test_create_remote_executable(self):
        """Test creating a remote executable."""
        # Setup
        executable = MockExecutable()
        mock_future = MagicMock(spec=MessageFuture)
        self.mock_fw.alloc_remote_nb.return_value = mock_future
        
        # Execute
        future = self.kernel.create_remote_executable("test_executable", executable, "remote_device")
        
        # Verify
        self.assertEqual(future, mock_future)
        self.mock_fw.alloc_remote_nb.assert_called_once()
        
    def test_delete_remote_tensor(self):
        """Test deleting a remote tensor."""
        # Setup
        mock_future = MagicMock(spec=MessageFuture)
        self.mock_fw.free_remote_nb.return_value = mock_future
        
        # Execute
        future = self.kernel.delete_remote_tensor(0x1000, "remote_device")
        
        # Verify
        self.assertEqual(future, mock_future)
        self.mock_fw.free_remote_nb.assert_called_once_with(0x1000, "remote_device")
        
    def test_delete_remote_event(self):
        """Test deleting a remote event."""
        # Setup
        mock_future = MagicMock(spec=MessageFuture)
        self.mock_fw.free_remote_nb.return_value = mock_future
        
        # Execute
        future = self.kernel.delete_remote_event(0x2000, "remote_device")
        
        # Verify
        self.assertEqual(future, mock_future)
        self.mock_fw.free_remote_nb.assert_called_once_with(0x2000, "remote_device")
        
    def test_delete_remote_pointers(self):
        """Test deleting remote pointers."""
        # Setup
        mock_future = MagicMock(spec=MessageFuture)
        self.mock_fw.free_remote_nb.return_value = mock_future
        
        # Execute
        future = self.kernel.delete_remote_pointers(0x3000, "remote_device")
        
        # Verify
        self.assertEqual(future, mock_future)
        self.mock_fw.free_remote_nb.assert_called_once_with(0x3000, "remote_device")
        
    def test_delete_remote_executable(self):
        """Test deleting a remote executable."""
        # Setup
        mock_future = MagicMock(spec=MessageFuture)
        self.mock_fw.free_remote_nb.return_value = mock_future
        
        # Execute
        future = self.kernel.delete_remote_executable(0x4000, "remote_device")
        
        # Verify
        self.assertEqual(future, mock_future)
        self.mock_fw.free_remote_nb.assert_called_once_with(0x4000, "remote_device")
        
    def test_extract_future(self):
        """Test extracting value from a future."""
        # Setup
        mock_future = MagicMock(spec=MessageFuture)
        expected_result = 0x5000
        self.mock_fw.extract_mesg_future.return_value = expected_result
        
        # Execute
        result = self.kernel.extract_future(mock_future)
        
        # Verify
        self.assertEqual(result, expected_result)
        self.mock_fw.extract_mesg_future.assert_called_once_with(mock_future)


if __name__ == "__main__":
    unittest.main()
