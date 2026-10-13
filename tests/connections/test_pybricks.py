"""Tests for the pybricks connection module."""

import asyncio
import contextlib
import os
import tempfile
from unittest.mock import AsyncMock, patch

import pytest
from reactivex.subject import Subject

from pybricksdev.ble.pybricks import PYBRICKS_COMMAND_EVENT_UUID, Command, Event
from pybricksdev.connections.pybricks import (
    ConnectionState,
    HubCapabilityFlag,
    HubKind,
    PybricksHubBLE,
    StatusFlag,
)


class TestPybricksHub:
    """Tests for the PybricksHub base class functionality."""

    def test_app_data_event_routing(self):
        hub = PybricksHubBLE("mock_device")
        app_data, stdout, status = [], [], []
        hub.app_data_observable.subscribe(app_data.append)
        hub.stdout_observable.subscribe(stdout.append)
        hub.status_observable.subscribe(status.append)

        hub._pybricks_service_handler(0, bytes([Event.WRITE_APP_DATA, 0, 255]))
        hub._pybricks_service_handler(0, bytes([Event.WRITE_STDOUT]) + b"text")
        hub._pybricks_service_handler(
            0,
            bytes([Event.STATUS_REPORT])
            + int(StatusFlag.USER_PROGRAM_RUNNING).to_bytes(4, "little"),
        )

        assert app_data == [b"\x00\xff"]
        assert stdout == [b"text"]
        assert status[-1] == StatusFlag.USER_PROGRAM_RUNNING

    @pytest.mark.asyncio
    async def test_write_app_data_packet_and_bounds(self):
        hub = PybricksHubBLE("mock_device")
        hub.write_gatt_char = AsyncMock()

        await hub.write_app_data(b"abc")
        hub.write_gatt_char.assert_awaited_with(
            PYBRICKS_COMMAND_EVENT_UUID, b"\x07\x00\x00abc", response=True
        )
        await hub.write_app_data(bytes(range(17)), offset=65535)
        hub.write_gatt_char.assert_awaited_with(
            PYBRICKS_COMMAND_EVENT_UUID,
            b"\x07\xff\xff" + bytes(range(17)),
            response=True,
        )
        assert hub.write_gatt_char.await_count == 2

        for offset in (-1, 65536):
            with pytest.raises(ValueError, match="offset"):
                await hub.write_app_data(b"x", offset=offset)
        with pytest.raises(ValueError, match="too big"):
            await hub.write_app_data(bytes(range(18)))
        assert hub.write_gatt_char.await_count == 2

    @pytest.mark.asyncio
    async def test_start_user_program_slot_selection(self):
        hub = PybricksHubBLE("mock_device")
        hub.write_gatt_char = AsyncMock()
        hub._num_of_slots = 5
        hub._selected_slot = 4  # May be stale; the hub owns default selection.

        await hub.start_user_program(None)
        hub.write_gatt_char.assert_awaited_with(
            PYBRICKS_COMMAND_EVENT_UUID,
            bytes([Command.START_USER_PROGRAM]),
            response=True,
        )

        await hub.start_user_program(2)
        hub.write_gatt_char.assert_awaited_with(
            PYBRICKS_COMMAND_EVENT_UUID,
            bytes([Command.START_USER_PROGRAM, 2]),
            response=True,
        )

        hub._num_of_slots = 0  # Legacy firmware has no slot argument.
        await hub.start_user_program(2)
        hub.write_gatt_char.assert_awaited_with(
            PYBRICKS_COMMAND_EVENT_UUID,
            bytes([Command.START_USER_PROGRAM]),
            response=True,
        )
        assert hub.write_gatt_char.await_count == 3

    @pytest.mark.asyncio
    async def test_download_modern_protocol(self):
        """Test downloading with modern protocol and capability flags."""
        hub = PybricksHubBLE("mock_device")
        hub._mpy_abi_version = 6
        hub._client = AsyncMock()
        hub.get_capabilities = AsyncMock(return_value={"pybricks": {"mpy": True}})
        hub.download_user_program = AsyncMock()
        hub.connection_state_observable.on_next(ConnectionState.CONNECTED)
        hub._capability_flags = HubCapabilityFlag.USER_PROG_MULTI_FILE_MPY6

        with contextlib.ExitStack() as stack:
            # Create and manage temporary file
            temp = stack.enter_context(
                tempfile.NamedTemporaryFile(
                    suffix=".py", mode="w+", delete=False, encoding="utf-8"
                )
            )
            temp.write("print('test')")
            temp_path = temp.name
            stack.callback(os.unlink, temp_path)

            await hub.download(temp_path)
            hub.download_user_program.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_legacy_firmware(self):
        """Test downloading with legacy firmware."""
        hub = PybricksHubBLE("mock_device")
        hub._mpy_abi_version = None  # Legacy firmware
        hub._client = AsyncMock()
        hub.download_user_program = AsyncMock()
        hub.hub_kind = HubKind.BOOST
        hub.connection_state_observable.on_next(ConnectionState.CONNECTED)
        hub._capability_flags = HubCapabilityFlag.USER_PROG_MULTI_FILE_MPY6

        with contextlib.ExitStack() as stack:
            # Create and manage temporary file
            temp = stack.enter_context(
                tempfile.NamedTemporaryFile(
                    suffix=".py", mode="w+", delete=False, encoding="utf-8"
                )
            )
            temp.write("print('test')")
            temp_path = temp.name
            stack.callback(os.unlink, temp_path)

            await hub.download(temp_path)
            hub.download_user_program.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_unsupported_capabilities(self):
        """Test downloading when hub doesn't support required capabilities."""
        hub = PybricksHubBLE("mock_device")
        hub._mpy_abi_version = 6
        hub._client = AsyncMock()
        hub.get_capabilities = AsyncMock(return_value={"pybricks": {"mpy": False}})
        hub.connection_state_observable.on_next(ConnectionState.CONNECTED)
        hub._capability_flags = 0

        with contextlib.ExitStack() as stack:
            # Create and manage temporary file
            temp = stack.enter_context(
                tempfile.NamedTemporaryFile(
                    suffix=".py", mode="w+", delete=False, encoding="utf-8"
                )
            )
            temp.write("print('test')")
            temp_path = temp.name
            stack.callback(os.unlink, temp_path)

            with pytest.raises(
                RuntimeError,
                match="Hub is not compatible with any of the supported file formats",
            ):
                await hub.download(temp_path)

    @pytest.mark.asyncio
    async def test_download_compile_error(self):
        """Test handling compilation errors."""
        hub = PybricksHubBLE("mock_device")
        hub._mpy_abi_version = 6
        hub._client = AsyncMock()
        hub.get_capabilities = AsyncMock(return_value={"pybricks": {"mpy": True}})
        hub.connection_state_observable.on_next(ConnectionState.CONNECTED)
        hub._capability_flags = HubCapabilityFlag.USER_PROG_MULTI_FILE_MPY6
        hub._max_user_program_size = 1000  # Set a reasonable size limit

        with contextlib.ExitStack() as stack:
            # Create and manage temporary file
            temp = stack.enter_context(
                tempfile.NamedTemporaryFile(
                    suffix=".py", mode="w+", delete=False, encoding="utf-8"
                )
            )
            temp.write("print('test'  # Missing closing parenthesis")
            temp_path = temp.name
            stack.callback(os.unlink, temp_path)

            # Mock compile_multi_file to raise SyntaxError
            stack.enter_context(
                patch(
                    "pybricksdev.connections.pybricks.compile_multi_file",
                    side_effect=SyntaxError("invalid syntax"),
                )
            )

            with pytest.raises(SyntaxError, match="invalid syntax"):
                await hub.download(temp_path)

    @pytest.mark.asyncio
    async def test_run_modern_protocol(self):
        """Test running a program with modern protocol."""
        hub = PybricksHubBLE("mock_device")
        hub._mpy_abi_version = None  # Use modern protocol
        hub._client = AsyncMock()
        hub.client = AsyncMock()
        hub.get_capabilities = AsyncMock(return_value={"pybricks": {"mpy": True}})
        hub.download_user_program = AsyncMock()
        hub.start_user_program = AsyncMock()
        hub.write_gatt_char = AsyncMock()
        hub.connection_state_observable.on_next(ConnectionState.CONNECTED)
        hub._capability_flags = HubCapabilityFlag.USER_PROG_MULTI_FILE_MPY6
        hub.hub_kind = HubKind.BOOST

        # Mock the status observable to simulate program start and stop
        status_subject = Subject()
        hub.status_observable = status_subject
        hub._stdout_line_queue = asyncio.Queue()
        hub._enable_line_handler = True

        with contextlib.ExitStack() as stack:
            # Create and manage temporary file
            temp = stack.enter_context(
                tempfile.NamedTemporaryFile(
                    suffix=".py", mode="w+", delete=False, encoding="utf-8"
                )
            )
            temp.write("print('test')")
            temp_path = temp.name
            stack.callback(os.unlink, temp_path)

            # Start the run task
            run_task = asyncio.create_task(hub.run(temp_path))

            # Simulate program start
            await asyncio.sleep(0.1)
            status_subject.on_next(StatusFlag.USER_PROGRAM_RUNNING)

            # Simulate program stop after a short delay
            await asyncio.sleep(0.1)
            status_subject.on_next(0)  # Clear all flags

            # Wait for run task to complete
            await run_task

            # Verify the expected calls were made
            hub.download_user_program.assert_called_once()
            hub.start_user_program.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "legacy", [0, HubCapabilityFlag.USER_PROG_MULTI_FILE_MPY6_1_NATIVE]
)
async def test_download_native_v63(tmp_path, legacy):
    source = tmp_path / "program.py"
    source.write_text(
        "import micropython\n@micropython.viper\n"
        "def add(x: int, y: int) -> int: return x + y\nprint(add(20, 22))\n"
    )
    hub = PybricksHubBLE("mock_device")
    hub.hub_kind = HubKind.TECHNIC_LARGE
    hub.connection_state_observable.on_next(ConnectionState.CONNECTED)
    hub._capability_flags = (
        HubCapabilityFlag.USER_PROG_FORMAT_MULTI_MPY_V6_3_NATIVE | legacy
    )
    hub.download_user_program = AsyncMock()
    await hub.download(str(source))
    (package,) = hub.download_user_program.call_args.args
    name, _, mpy = package[4:].partition(b"\0")
    assert name == b"__main__"
    assert len(mpy) == int.from_bytes(package[:4], "little")
    assert mpy[:4] == bytes([ord("M"), 6, (5 << 2) | 3, 31])
