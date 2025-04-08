SHELL = /bin/bash -f

test:
	LOGURU_LEVEL=TRACE pytest -s -vv \
		tests/test_mock_hardware.py \
		tests/test_rpc_host.py \
		tests/test_rpc_client.py
