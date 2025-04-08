import unittest
import queue
import numpy as np
from unittest.mock import MagicMock

from server.fw.rpc_host import RPCHost
from server.data_types import Tensor, Event, Pointers, Executable
from server.fw.rpc_message import (
    AllocRequest,
    AllocResponse,
    FreeRequest,
    FreeResponse,
    ReadTensorRequest,
    ReadTensorResponse,
    WriteTensorRequest,
    WriteTensorResponse,
    SetEventRequest,
    SetEventResponse,
    WritePointersRequest,
    WritePointersResponse,
    WriteExecutableRequest,
    WriteExecutableResponse,
    ExecuteRequest,
    ExecuteResponse,
    ExitRequest,
    ExitResponse,
)
from server.error_code import ErrCode


# Create a concrete implementation of Executable for testing
class ExecutableTest(Executable):
    def __init__(self, name="test_executable", return_code=ErrCode.ESUCC):
        super().__init__(name)
        self.return_code = return_code
        
    def _execute(self):
        return self.return_code


class TestRPCHost(unittest.TestCase):
    """Test suite for the RPCHost class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock device
        self.mock_device = MagicMock()
        self.mock_device.uid = "test_device"
        self.mock_device.input_queue = queue.Queue()
        self.mock_device.output_queues = {
            "test_device": queue.Queue(),
            "client": queue.Queue(),
        }
        
        # Create a concrete implementation of the abstract RPCHost class
        class ConcreteRPCHost(RPCHost):
            pass
        
        self.rpc_host = ConcreteRPCHost(device=self.mock_device)
        
        # Mock the thread registration to avoid actual thread creation
        self.rpc_host.register_thread = MagicMock()
        
    def tearDown(self):
        """Clean up after each test method."""
        # Ensure the host is stopped
        if hasattr(self, "rpc_host"):
            self.rpc_host.quit_event.set()
    
    def test_initialization(self):
        """Test that the RPCHost initializes correctly."""
        self.assertEqual(self.rpc_host.device, self.mock_device)
        self.assertIsInstance(self.rpc_host.execution_queue, queue.Queue)
        self.mock_device.assert_not_called()
    
    def test_handle_message_alloc_request(self):
        """Test handling of AllocRequest messages."""
        # Setup
        request = AllocRequest(
            uid=1, src_device_uid="client", dst_device_uid="test_device", size=1024
        )
        self.mock_device.alloc.return_value = (0x1000, ErrCode.ESUCC)
        
        # Execute
        self.rpc_host.handle_message(request)
        
        # Verify
        self.mock_device.alloc.assert_called_once_with(1024)
        response = self.mock_device.output_queues["client"].get_nowait()
        self.assertIsInstance(response, AllocResponse)
        self.assertEqual(response.uid, 1)
        self.assertEqual(response.src_device_uid, "test_device")
        self.assertEqual(response.dst_device_uid, "client")
        self.assertEqual(response.addr, 0x1000)
        self.assertEqual(response.err, ErrCode.ESUCC)
    
    def test_handle_message_free_request(self):
        """Test handling of FreeRequest messages."""
        # Setup
        request = FreeRequest(
            uid=2, src_device_uid="client", dst_device_uid="test_device", addr=0x1000
        )
        self.mock_device.free.return_value = ErrCode.ESUCC
        
        # Execute
        self.rpc_host.handle_message(request)
        
        # Verify
        self.mock_device.free.assert_called_once_with(0x1000)
        response = self.mock_device.output_queues["client"].get_nowait()
        self.assertIsInstance(response, FreeResponse)
        self.assertEqual(response.uid, 2)
        self.assertEqual(response.src_device_uid, "test_device")
        self.assertEqual(response.dst_device_uid, "client")
        self.assertEqual(response.err, ErrCode.ESUCC)
    
    def test_handle_message_read_tensor_request(self):
        """Test handling of ReadTensorRequest messages."""
        # Setup
        tensor = Tensor(name="test_tensor", data=np.array([1, 2, 3]))
        request = ReadTensorRequest(
            uid=3, src_device_uid="client", dst_device_uid="test_device", addr=0x1000
        )
        self.mock_device.read.return_value = (tensor, ErrCode.ESUCC)
        
        # Execute
        self.rpc_host.handle_message(request)
        
        # Verify
        self.mock_device.read.assert_called_once_with(0x1000)
        response = self.mock_device.output_queues["client"].get_nowait()
        self.assertIsInstance(response, ReadTensorResponse)
        self.assertEqual(response.uid, 3)
        self.assertEqual(response.src_device_uid, "test_device")
        self.assertEqual(response.dst_device_uid, "client")
        self.assertEqual(response.data, tensor)
        self.assertEqual(response.err, ErrCode.ESUCC)
    
    def test_handle_message_write_tensor_request(self):
        """Test handling of WriteTensorRequest messages."""
        # Setup
        tensor = Tensor(name="test_tensor", data=np.array([1, 2, 3]))
        request = WriteTensorRequest(
            uid=4,
            src_device_uid="client",
            dst_device_uid="test_device",
            addr=0x1000,
            data=tensor,
        )
        self.mock_device.read.return_value = (tensor, ErrCode.ESUCC)
        
        # Execute
        self.rpc_host.handle_message(request)
        
        # Verify
        self.mock_device.read.assert_called_once_with(0x1000)
        response = self.mock_device.output_queues["client"].get_nowait()
        self.assertIsInstance(response, WriteTensorResponse)
        self.assertEqual(response.uid, 4)
        self.assertEqual(response.src_device_uid, "test_device")
        self.assertEqual(response.dst_device_uid, "client")
        self.assertEqual(response.err, ErrCode.ESUCC)
    
    def test_handle_message_set_event_request(self):
        """Test handling of SetEventRequest messages."""
        # Setup
        event = Event(name="test_event")
        request = SetEventRequest(
            uid=5, src_device_uid="client", dst_device_uid="test_device", addr=0x1000
        )
        self.mock_device.read.return_value = (event, ErrCode.ESUCC)
        
        # Execute
        self.rpc_host.handle_message(request)
        
        # Verify
        self.mock_device.read.assert_called_once_with(0x1000)
        self.assertTrue(event.value)  # Check event.value instead of event.is_set()
        response = self.mock_device.output_queues["client"].get_nowait()
        self.assertIsInstance(response, SetEventResponse)
        self.assertEqual(response.uid, 5)
        self.assertEqual(response.src_device_uid, "test_device")
        self.assertEqual(response.dst_device_uid, "client")
        self.assertEqual(response.err, ErrCode.ESUCC)
    
    def test_handle_message_write_pointers_request(self):
        """Test handling of WritePointersRequest messages."""
        # Setup
        pointers = Pointers(name="test_pointers", ptrs=["0x1000", "0x2000"])
        request = WritePointersRequest(
            uid=6,
            src_device_uid="client",
            dst_device_uid="test_device",
            addr=0x3000,
            data=pointers,
        )
        self.mock_device.write.return_value = ErrCode.ESUCC
        
        # Execute
        self.rpc_host.handle_message(request)
        
        # Verify
        self.mock_device.write.assert_called_once_with(0x3000, pointers)
        response = self.mock_device.output_queues["client"].get_nowait()
        self.assertIsInstance(response, WritePointersResponse)
        self.assertEqual(response.uid, 6)
        self.assertEqual(response.src_device_uid, "test_device")
        self.assertEqual(response.dst_device_uid, "client")
        self.assertEqual(response.err, ErrCode.ESUCC)
    
    def test_handle_message_write_executable_request(self):
        """Test handling of WriteExecutableRequest messages."""
        # Setup
        executable = ExecutableTest()
        request = WriteExecutableRequest(
            uid=7,
            src_device_uid="client",
            dst_device_uid="test_device",
            addr=0x4000,
            data=executable,
        )
        self.mock_device.write.return_value = ErrCode.ESUCC
        
        # Execute
        self.rpc_host.handle_message(request)
        
        # Verify
        self.mock_device.write.assert_called_once_with(0x4000, executable)
        response = self.mock_device.output_queues["client"].get_nowait()
        self.assertIsInstance(response, WriteExecutableResponse)
        self.assertEqual(response.uid, 7)
        self.assertEqual(response.src_device_uid, "test_device")
        self.assertEqual(response.dst_device_uid, "client")
        self.assertEqual(response.err, ErrCode.ESUCC)
    
    def test_handle_message_execute_request(self):
        """Test handling of ExecuteRequest messages."""
        # Setup
        request = ExecuteRequest(
            uid=8,
            src_device_uid="client",
            dst_device_uid="test_device",
            executable_addr=0x5000,
            args_addr=0x6000,
        )
        
        # Execute
        self.rpc_host.handle_message(request)
        
        # Verify
        # The request should be added to the execution queue
        self.assertEqual(self.rpc_host.execution_queue.qsize(), 1)
        queued_request = self.rpc_host.execution_queue.get_nowait()
        self.assertEqual(queued_request, request)
    
    def test_handle_message_exit_request(self):
        """Test handling of ExitRequest messages."""
        # Setup
        request = ExitRequest(
            uid=9,
            src_device_uid="client",
            dst_device_uid="test_device",
            reason="Test exit",
        )
        
        # Execute
        self.rpc_host.handle_message(request)
        
        # Verify
        response = self.mock_device.output_queues["client"].get_nowait()
        self.assertIsInstance(response, ExitResponse)
        self.assertEqual(response.uid, 9)
        self.assertEqual(response.src_device_uid, "test_device")
        self.assertEqual(response.dst_device_uid, "client")
        self.assertEqual(response.err, ErrCode.ESUCC)
        self.assertTrue(self.rpc_host.quit_event.is_set())
    
    def test_handle_message_response(self):
        """Test handling of response messages."""
        # Setup
        response = AllocResponse(
            uid=10,
            src_device_uid="test_device",
            dst_device_uid="client",
            addr=0x7000,
            err=ErrCode.ESUCC,
        )
        
        # Mock the set_mesg_future method
        self.rpc_host.set_mesg_future = MagicMock()
        
        # Execute
        self.rpc_host.handle_message(response)
        
        # Verify
        self.rpc_host.set_mesg_future.assert_called_once_with(response)
    
    def test_handle_execute_local(self):
        """Test handling of execute request from local device."""
        # Setup
        executable = ExecutableTest()
        args = Pointers(name="test_args", ptrs=["0x8000", "0x9000"])
        request = ExecuteRequest(
            uid=11,
            src_device_uid="test_device",
            dst_device_uid="test_device",
            executable_addr=0xA000,
            args_addr=0xB000,
        )
        
        self.mock_device.read.side_effect = [
            (executable, ErrCode.ESUCC),
            (args, ErrCode.ESUCC),
        ]
        
        # Execute
        self.rpc_host.handle_execute(request)
        
        # Verify
        self.assertEqual(self.mock_device.read.call_count, 2)
        self.mock_device.read.assert_any_call(0xA000)
        self.mock_device.read.assert_any_call(0xB000)
    
    def test_handle_execute_remote(self):
        """Test handling of execute request from remote device."""
        # Setup
        executable = ExecutableTest()
        args = Pointers(name="test_args", ptrs=["0x8000", "0x9000"])
        request = ExecuteRequest(
            uid=12,
            src_device_uid="client",
            dst_device_uid="test_device",
            executable_addr=0xA000,
            args_addr=0xB000,
        )
        
        self.mock_device.read.side_effect = [
            (executable, ErrCode.ESUCC),
            (args, ErrCode.ESUCC),
        ]
        
        # Execute
        self.rpc_host.handle_execute(request)
        
        # Verify
        self.assertEqual(self.mock_device.read.call_count, 2)
        self.mock_device.read.assert_any_call(0xA000)
        self.mock_device.read.assert_any_call(0xB000)
        
        # A response should be sent back to the client
        response = self.mock_device.output_queues["client"].get_nowait()
        self.assertIsInstance(response, ExecuteResponse)
        self.assertEqual(response.uid, 12)
        self.assertEqual(response.src_device_uid, "test_device")
        self.assertEqual(response.dst_device_uid, "client")
        self.assertEqual(response.err, ErrCode.ESUCC)
    
    def test_handle_execute_error_reading_executable(self):
        """Test handling of execute request with error reading executable."""
        # Setup
        request = ExecuteRequest(
            uid=13,
            src_device_uid="client",
            dst_device_uid="test_device",
            executable_addr=0xA000,
            args_addr=0xB000,
        )
        
        self.mock_device.read.return_value = (None, ErrCode.EINVAL)
        
        # Execute
        self.rpc_host.handle_execute(request)
        
        # Verify
        self.mock_device.read.assert_called_once_with(0xA000)
        # A response should be sent back to the client with the error
        response = self.mock_device.output_queues["client"].get_nowait()
        self.assertIsInstance(response, ExecuteResponse)
        self.assertEqual(response.uid, 13)
        self.assertEqual(response.src_device_uid, "test_device")
        self.assertEqual(response.dst_device_uid, "client")
        self.assertEqual(response.err, ErrCode.EINVAL)
    
    def test_handle_execute_error_reading_args(self):
        """Test handling of execute request with error reading args."""
        # Setup
        executable = ExecutableTest()
        request = ExecuteRequest(
            uid=14,
            src_device_uid="client",
            dst_device_uid="test_device",
            executable_addr=0xA000,
            args_addr=0xB000,
        )
        
        self.mock_device.read.side_effect = [
            (executable, ErrCode.ESUCC),
            (None, ErrCode.EINVAL),
        ]
        
        # Execute
        self.rpc_host.handle_execute(request)
        
        # Verify
        self.assertEqual(self.mock_device.read.call_count, 2)
        self.mock_device.read.assert_any_call(0xA000)
        self.mock_device.read.assert_any_call(0xB000)
        # A response should be sent back to the client with the error
        response = self.mock_device.output_queues["client"].get_nowait()
        self.assertIsInstance(response, ExecuteResponse)
        self.assertEqual(response.uid, 14)
        self.assertEqual(response.src_device_uid, "test_device")
        self.assertEqual(response.dst_device_uid, "client")
        self.assertEqual(response.err, ErrCode.EINVAL)
    
    def test_handle_execute_error_executing(self):
        """Test handling of execute request with error during execution."""
        # Setup
        executable = ExecutableTest(return_code=ErrCode.EINVAL)
        args = Pointers(name="test_args", ptrs=["0x8000", "0x9000"])
        request = ExecuteRequest(
            uid=15,
            src_device_uid="client",
            dst_device_uid="test_device",
            executable_addr=0xA000,
            args_addr=0xB000,
        )
        
        self.mock_device.read.side_effect = [
            (executable, ErrCode.ESUCC),
            (args, ErrCode.ESUCC),
        ]
        
        # Execute
        self.rpc_host.handle_execute(request)
        
        # Verify
        self.assertEqual(self.mock_device.read.call_count, 2)
        self.mock_device.read.assert_any_call(0xA000)
        self.mock_device.read.assert_any_call(0xB000)
        # A response should be sent back to the client with the error
        response = self.mock_device.output_queues["client"].get_nowait()
        self.assertIsInstance(response, ExecuteResponse)
        self.assertEqual(response.uid, 15)
        self.assertEqual(response.src_device_uid, "test_device")
        self.assertEqual(response.dst_device_uid, "client")
        self.assertEqual(response.err, ErrCode.EINVAL)
    
    def test_message_handling_thread_entry(self):
        """Test the message handling thread entry point."""
        # Setup
        request = AllocRequest(
            uid=16, src_device_uid="client", dst_device_uid="test_device", size=1024
        )
        self.mock_device.alloc.return_value = (0x1000, ErrCode.ESUCC)
        
        # Execute
        # Run the thread entry with is_test=True to avoid infinite loop
        self.rpc_host.message_handling_thread_entry(is_test=True)
        
        # Verify
        # No messages should be processed since we didn't add any to the queue
        self.mock_device.alloc.assert_not_called()
        
        # Now add a message to the queue and run again
        self.mock_device.input_queue.put(request)
        self.rpc_host.message_handling_thread_entry(is_test=True)
        
        # Verify the message was processed
        self.mock_device.alloc.assert_called_once_with(1024)
        response = self.mock_device.output_queues["client"].get_nowait()
        self.assertIsInstance(response, AllocResponse)
        self.assertEqual(response.uid, 16)
        self.assertEqual(response.src_device_uid, "test_device")
        self.assertEqual(response.dst_device_uid, "client")
        self.assertEqual(response.addr, 0x1000)
        self.assertEqual(response.err, ErrCode.ESUCC)
    
    def test_execution_handling_thread_entry(self):
        """Test the execution handling thread entry point."""
        # Setup
        executable = ExecutableTest()
        args = Pointers(name="test_args", ptrs=["0x8000", "0x9000"])
        request = ExecuteRequest(
            uid=17,
            src_device_uid="client",
            dst_device_uid="test_device",
            executable_addr=0xA000,
            args_addr=0xB000,
        )
        
        self.mock_device.read.side_effect = [
            (executable, ErrCode.ESUCC),
            (args, ErrCode.ESUCC),
        ]
        
        # Execute
        # Run the thread entry with is_test=True to avoid infinite loop
        self.rpc_host.execution_handling_thread_entry(is_test=True)
        
        # Verify
        # No messages should be processed since we didn't add any to the queue
        self.mock_device.read.assert_not_called()
        
        # Now add a message to the queue and run again
        self.rpc_host.execution_queue.put(request)
        self.rpc_host.execution_handling_thread_entry(is_test=True)
        
        # Verify the message was processed
        self.assertEqual(self.mock_device.read.call_count, 2)
        self.mock_device.read.assert_any_call(0xA000)
        self.mock_device.read.assert_any_call(0xB000)
        # A response should be sent back to the client
        response = self.mock_device.output_queues["client"].get_nowait()
        self.assertIsInstance(response, ExecuteResponse)
        self.assertEqual(response.uid, 17)
        self.assertEqual(response.src_device_uid, "test_device")
        self.assertEqual(response.dst_device_uid, "client")
        self.assertEqual(response.err, ErrCode.ESUCC)


if __name__ == "__main__":
    unittest.main()
