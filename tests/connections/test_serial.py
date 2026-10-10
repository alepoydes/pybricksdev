# SPDX-License-Identifier: MIT
"""Exercise actual asynchronous serial I/O using a synthetic peer on a PTY."""

import asyncio
import contextlib
import os
import struct

import pytest

from pybricksdev.ble.pybricks import PYBRICKS_COMMAND_EVENT_UUID, SW_REV_UUID
from pybricksdev.connections import ConnectionState
from pybricksdev.connections.serial import PybricksHubSerial
from pybricksdev.usb.cobs import pack, unpack

pytestmark = pytest.mark.skipif(os.name != "posix", reason="PTY needs POSIX")


@contextlib.asynccontextmanager
async def peer(*, silent=False, bad_version=False, reject=False, silent_commands=False):
    import pty

    master, slave = pty.openpty()
    os.set_blocking(master, False)
    loop = asyncio.get_running_loop()
    received = asyncio.Queue()
    buffer = bytearray()

    def read():
        buffer.extend(os.read(master, 4096))
        while b"\x02" in buffer:
            frame, _, rest = buffer.partition(b"\x02")
            buffer[:] = rest
            if frame:
                received.put_nowait(unpack(frame))

    loop.add_reader(master, read)
    commands = []

    async def serve():
        while True:
            request = await received.get()
            if silent:
                continue
            if request[0] == 3:
                value = {
                    b"\x01\x28\x2a": b"2.0.0" if bad_version else b"1.7.0",
                    b"\x01\x00\x2a": b"Synthetic hub",
                    b"\x01\x26\x2a": b"4.1.0b6",
                    b"\x01\x50\x2a": bytes.fromhex("01 97 03 81 00 01 00"),
                    b"\x02\x03\x00": struct.pack("<HIIB", 64, 0, 0, 0),
                }.get(request[1:], b"")
                response = request + value
            elif request[0] == 2:
                commands.append(request[2:])
                if silent_commands:
                    continue
                # A stale tag must not satisfy the current command.
                stale = bytes([1, (request[1] + 1) % 256, 0, 0, 0, 0])
                os.write(master, pack(stale))
                response = bytes([1, request[1], 0x80 if reject else 0, 0, 0, 0])
            elif request == b"\x01\x01":
                response = bytes.fromhex("02 00 00 10 00 00 00 00")
            else:
                continue
            # Deliver an invalid frame and then a valid frame split across reads.
            framed = pack(response)
            os.write(master, b"\x01\x03\x02" + framed[:2])
            await asyncio.sleep(0)
            os.write(master, framed[2:])

    task = asyncio.create_task(serve())
    hub = PybricksHubSerial(os.ttyname(slave), timeout=0.2)
    try:
        yield hub, commands
    finally:
        await hub.disconnect()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        loop.remove_reader(master)
        os.close(master)
        os.close(slave)


@pytest.mark.asyncio
async def test_identity_status_commands_and_reconnect():
    async with peer() as (hub, commands):
        for _ in range(2):
            await hub.connect()
            assert hub.name == "Synthetic hub"
            assert str(hub.fw_version) == "4.1.0b6"
            assert hub.hub_kind == 0x81
            event = asyncio.Event()
            with hub.status_observable.subscribe(
                lambda flags: event.set() if flags else None
            ):
                await asyncio.wait_for(event.wait(), 0.5)
            assert int(hub.status_observable.value) == 1 << 12
            await asyncio.gather(
                hub.stop_user_program(),
                hub.read_gatt_char(SW_REV_UUID),
            )
            assert hub.rejected_frames > 0
            await hub.disconnect()
            assert hub.connection_state_observable.value == ConnectionState.DISCONNECTED
        assert commands == [b"\x00", b"\x00"]


@pytest.mark.asyncio
async def test_command_error_does_not_disconnect():
    async with peer(reject=True) as (hub, _):
        await hub.connect()
        with pytest.raises(RuntimeError, match="Write failed: 128"):
            await hub.stop_user_program()
        assert hub.connection_state_observable.value == ConnectionState.CONNECTED
        assert await hub.read_gatt_char(SW_REV_UUID) == b"1.7.0"
        with pytest.raises(ValueError, match="advertised size"):
            await hub.write_gatt_char(PYBRICKS_COMMAND_EVENT_UUID, bytes(65), True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options, error",
    [({"silent": True}, TimeoutError), ({"bad_version": True}, RuntimeError)],
)
async def test_failed_connection_releases_port(options, error):
    async with peer(**options) as (hub, _):
        for _ in range(2):
            with pytest.raises(error):
                await hub.connect()
            assert hub.connection_state_observable.value == ConnectionState.DISCONNECTED
        # Opening an exclusive descriptor after failure must succeed.
        import serial

        with serial.Serial(hub.port, exclusive=True):
            pass


@pytest.mark.asyncio
async def test_cancelled_connection_releases_port():
    async with peer(silent=True) as (hub, _):
        task = asyncio.create_task(hub.connect())
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert hub.connection_state_observable.value == ConnectionState.DISCONNECTED
        import serial

        with serial.Serial(hub.port, exclusive=True):
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_interrupted_command_closes_session(cancel):
    async with peer(silent_commands=True) as (hub, _):
        await hub.connect()
        command = asyncio.create_task(hub.stop_user_program())
        if cancel:
            await asyncio.sleep(0.02)
            command.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await command
        assert hub.connection_state_observable.value == ConnectionState.DISCONNECTED
        from pybricksdev.connections.pybricks import HubDisconnectError

        with pytest.raises(HubDisconnectError):
            await hub.read_gatt_char(SW_REV_UUID)
        import serial

        with serial.Serial(hub.port, exclusive=True):
            pass
