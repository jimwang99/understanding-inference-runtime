import unittest
import queue
import numpy as np
from unittest.mock import MagicMock

from server.fw.rpc_client import RPCClient
from server.data_types import Tensor, Executable, Pointers
from server.fw.rpc_message import (
    AllocRequest,
    FreeRequest,
    ReadTensorRequest,
    WriteTensorRequest,
    SetEventRequest,
    WritePointersRequest,
    WriteExecutableRequest,
    ExecuteRequest,
    ExitRequest,
)
from server.fw.future_host import MessageFuture
from server.error_code import ErrCode


# Create a concrete implementation of Executable for testing
class MockExecutable(Executable):
    def __init__(self, name="test_executable", return_code=ErrCode.ESUCC):
        super().__init__(name)
        self.return_code = return_code
        
    def _execute(self):
        return self.return_code


class TestRPCClient(unittest.TestCase):
    """Test suite for the RPCClient class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock device
        self.mock_device = MagicMock()
        self.mock_device.uid = "test_device"
        
        # Set up the remote device
        self.remote_device = MagicMock()
        self.remote_device.uid = "remote_device"
        
        # Configure the mock_device to return the remote_device when asked
        self.mock_device.get_device_by_alias.return_value = self.remote_device
        
        # Set up connected devices
        self.mock_device.connected_devices = {}
        
        # Set up output queue
        self.output_queue = queue.Queue()
        self.mock_device.output_queues = {self.remote_device.uid: self.output_queue}
        
        # Create a concrete implementation of the abstract RPCClient class
        class ConcreteRPCClient(RPCClient):
            pass
        
        self.rpc_client = ConcreteRPCClient(device=self.mock_device)
        
        # Mock create_mesg_future to return a mock MessageFuture
        self.mock_future = MagicMock(spec=MessageFuture)
        self.rpc_client.create_mesg_future = MagicMock(return_value=self.mock_future)
    
    def test_initialization(self):
        """Test that the RPCClient initializes correctly."""
        self.assertEqual(self.rpc_client.device, self.mock_device)
    
    def test_alloc_remote_nb(self):
        """Test allocation of memory on a remote device."""
        # Setup
        size = 1024
        remote_device_alias = "remote_device"
        
        # Execute
        future = self.rpc_client.alloc_remote_nb(size, remote_device_alias)
        
        # Verify
        self.mock_device.get_device_by_alias.assert_called_once_with(remote_device_alias)
        
        # Check the message in the queue
        request = self.output_queue.get_nowait()
        self.assertIsInstance(request, AllocRequest)
        self.assertEqual(request.src_device_uid, self.mock_device.uid)
        self.assertEqual(request.dst_device_uid, self.remote_device.uid)
        self.assertEqual(request.size, size)
        
        # Check that create_mesg_future was called with the request
        self.rpc_client.create_mesg_future.assert_called_once_with(request)
        
        # Check that the future was returned
        self.assertEqual(future, self.mock_future)
    
    def test_free_remote_nb(self):
        """Test freeing memory on a remote device."""
        # Setup
        addr = 0x1000
        remote_device_alias = "remote_device"
        
        # Execute
        future = self.rpc_client.free_remote_nb(addr, remote_device_alias)
        
        # Verify
        self.mock_device.get_device_by_alias.assert_called_once_with(remote_device_alias)
        
        # Check the message in the queue
        request = self.output_queue.get_nowait()
        self.assertIsInstance(request, FreeRequest)
        self.assertEqual(request.src_device_uid, self.mock_device.uid)
        self.assertEqual(request.dst_device_uid, self.remote_device.uid)
        self.assertEqual(request.addr, addr)
        
        # Check that create_mesg_future was called with the request
        self.rpc_client.create_mesg_future.assert_called_once_with(request)
        
        # Check that the future was returned
        self.assertEqual(future, self.mock_future)
    
    def test_exit_remote_nb(self):
        """Test requesting a remote device to exit."""
        # Setup
        reason = "Test exit"
        remote_device_alias = "remote_device"
        
        # Execute
        future = self.rpc_client.exit_remote_nb(reason, remote_device_alias)
        
        # Verify
        self.mock_device.get_device_by_alias.assert_called_once_with(remote_device_alias)
        
        # Check the message in the queue
        request = self.output_queue.get_nowait()
        self.assertIsInstance(request, ExitRequest)
        self.assertEqual(request.src_device_uid, self.mock_device.uid)
        self.assertEqual(request.dst_device_uid, self.remote_device.uid)
        self.assertEqual(request.reason, reason)
        
        # Check that create_mesg_future was called with the request
        self.rpc_client.create_mesg_future.assert_called_once_with(request)
        
        # Check that the future was returned
        self.assertEqual(future, self.mock_future)
    
    def test_read_tensor_remote_nb(self):
        """Test reading a tensor from a remote device."""
        # Setup
        addr = 0x2000
        remote_device_alias = "remote_device"
        
        # Execute
        future = self.rpc_client.read_tensor_remote_nb(addr, remote_device_alias)
        
        # Verify
        self.mock_device.get_device_by_alias.assert_called_once_with(remote_device_alias)
        
        # Check the message in the queue
        request = self.output_queue.get_nowait()
        self.assertIsInstance(request, ReadTensorRequest)
        self.assertEqual(request.src_device_uid, self.mock_device.uid)
        self.assertEqual(request.dst_device_uid, self.remote_device.uid)
        self.assertEqual(request.addr, addr)
        
        # Check that create_mesg_future was called with the request
        self.rpc_client.create_mesg_future.assert_called_once_with(request)
        
        # Check that the future was returned
        self.assertEqual(future, self.mock_future)
    
    def test_write_tensor_remote_nb(self):
        """Test writing a tensor to a remote device."""
        # Setup
        addr = 0x3000
        tensor = Tensor(name="test_tensor", data=np.array([1, 2, 3]))
        remote_device_alias = "remote_device"
        
        # Execute
        future = self.rpc_client.write_tensor_remote_nb(addr, tensor, remote_device_alias)
        
        # Verify
        self.mock_device.get_device_by_alias.assert_called_once_with(remote_device_alias)
        
        # Check the message in the queue
        request = self.output_queue.get_nowait()
        self.assertIsInstance(request, WriteTensorRequest)
        self.assertEqual(request.src_device_uid, self.mock_device.uid)
        self.assertEqual(request.dst_device_uid, self.remote_device.uid)
        self.assertEqual(request.addr, addr)
        self.assertEqual(request.data, tensor)
        
        # Check that create_mesg_future was called with the request
        self.rpc_client.create_mesg_future.assert_called_once_with(request)
        
        # Check that the future was returned
        self.assertEqual(future, self.mock_future)
    
    def test_set_event_remote_nb(self):
        """Test setting an event on a remote device."""
        # Setup
        addr = 0x4000
        remote_device_alias = "remote_device"
        
        # Execute
        future = self.rpc_client.set_event_remote_nb(addr, remote_device_alias)
        
        # Verify
        self.mock_device.get_device_by_alias.assert_called_once_with(remote_device_alias)
        
        # Check the message in the queue
        request = self.output_queue.get_nowait()
        self.assertIsInstance(request, SetEventRequest)
        self.assertEqual(request.src_device_uid, self.mock_device.uid)
        self.assertEqual(request.dst_device_uid, self.remote_device.uid)
        self.assertEqual(request.addr, addr)
        
        # Check that create_mesg_future was called with the request
        self.rpc_client.create_mesg_future.assert_called_once_with(request)
        
        # Check that the future was returned
        self.assertEqual(future, self.mock_future)
    
    def test_write_pointers_remote_nb(self):
        """Test writing pointers to a remote device."""
        # Setup
        addr = 0x5000
        pointers = Pointers(name="test_pointers", ptrs=["0x6000", "0x7000"])
        remote_device_alias = "remote_device"
        
        # Execute
        future = self.rpc_client.write_pointers_remote_nb(addr, pointers, remote_device_alias)
        
        # Verify
        self.mock_device.get_device_by_alias.assert_called_once_with(remote_device_alias)
        
        # Check the message in the queue
        request = self.output_queue.get_nowait()
        self.assertIsInstance(request, WritePointersRequest)
        self.assertEqual(request.src_device_uid, self.mock_device.uid)
        self.assertEqual(request.dst_device_uid, self.remote_device.uid)
        self.assertEqual(request.addr, addr)
        self.assertEqual(request.data, pointers)
        
        # Check that create_mesg_future was called with the request
        self.rpc_client.create_mesg_future.assert_called_once_with(request)
        
        # Check that the future was returned
        self.assertEqual(future, self.mock_future)
    
    def test_write_executable_remote_nb(self):
        """Test writing an executable to a remote device."""
        # Setup
        addr = 0x8000
        executable = MockExecutable()
        remote_device_alias = "remote_device"
        
        # Execute
        future = self.rpc_client.write_executable_remote_nb(addr, executable, remote_device_alias)
        
        # Verify
        self.mock_device.get_device_by_alias.assert_called_once_with(remote_device_alias)
        
        # Check the message in the queue
        request = self.output_queue.get_nowait()
        self.assertIsInstance(request, WriteExecutableRequest)
        self.assertEqual(request.src_device_uid, self.mock_device.uid)
        self.assertEqual(request.dst_device_uid, self.remote_device.uid)
        self.assertEqual(request.addr, addr)
        self.assertEqual(request.data, executable)
        
        # Check that create_mesg_future was called with the request
        self.rpc_client.create_mesg_future.assert_called_once_with(request)
        
        # Check that the future was returned
        self.assertEqual(future, self.mock_future)
    
    def test_execute_remote_nb(self):
        """Test executing a program on a remote device."""
        # Setup
        executable_addr = 0x9000
        args_addr = 0xA000
        remote_device_alias = "remote_device"
        
        # Execute
        future = self.rpc_client.execute_remote_nb(executable_addr, args_addr, remote_device_alias)
        
        # Verify
        self.mock_device.get_device_by_alias.assert_called_once_with(remote_device_alias)
        
        # Check the message in the queue
        request = self.output_queue.get_nowait()
        self.assertIsInstance(request, ExecuteRequest)
        self.assertEqual(request.src_device_uid, self.mock_device.uid)
        self.assertEqual(request.dst_device_uid, self.remote_device.uid)
        self.assertEqual(request.executable_addr, executable_addr)
        self.assertEqual(request.args_addr, args_addr)
        
        # Check that create_mesg_future was called with the request
        self.rpc_client.create_mesg_future.assert_called_once_with(request)
        
        # Check that the future was returned
        self.assertEqual(future, self.mock_future)
        
    def test_get_remote_device_from_connected_devices(self):
        """Test that _get_remote_device finds device in connected_devices."""
        # Setup
        connected_device = MagicMock()
        connected_device.uid = "connected_device"
        self.mock_device.connected_devices = {"connected": connected_device}
        
        # Execute
        remote_device = self.rpc_client._get_remote_device("connected")
        
        # Verify
        self.assertEqual(remote_device, connected_device)
        self.mock_device.get_device_by_alias.assert_not_called()


if __name__ == "__main__":
    unittest.main()
