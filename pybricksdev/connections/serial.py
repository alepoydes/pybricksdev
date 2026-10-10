# SPDX-License-Identifier: MIT
"""Pybricks Profile 1.6+ over USB CDC ACM (SPIKE COBS framing)."""

import asyncio
import contextlib
import os
import struct
from typing import Callable
from uuid import UUID

import semver
import serial_asyncio
from packaging.version import Version

from pybricksdev.ble.nus import NUS_TX_UUID
from pybricksdev.ble.pybricks import (
    DEVICE_NAME_UUID,
    FW_REV_UUID,
    PNP_ID_UUID,
    PYBRICKS_COMMAND_EVENT_UUID,
    PYBRICKS_HUB_CAPABILITIES_UUID,
    SW_REV_UUID,
    StatusFlag,
    unpack_hub_capabilities,
    unpack_pnp_id,
)
from pybricksdev.connections import ConnectionState
from pybricksdev.connections.pybricks import HubDisconnectError, PybricksHub
from pybricksdev.usb.cobs import DELIMITER, MAX_BLOCK_SIZE, pack, unpack
from pybricksdev.usb.pybricks import PybricksUsbInEpMessageType as Incoming
from pybricksdev.usb.pybricks import PybricksUsbOutEpMessageType as Outgoing

# Characteristic sizes are uint16. Include the read envelope and COBS overhead.
MAX_FRAME = 65539 + 65539 // MAX_BLOCK_SIZE + 3


class PybricksHubSerial(PybricksHub):
    """Connect to an explicitly selected CDC port without detaching its driver.

    ``timeout`` bounds each exchange, including connection identification. A lost
    reply closes the session: read replies have no transaction tag, and accepting
    a late reply after another request would be ambiguous. USB permissions and
    selection of a compatible interface remain the caller's responsibility.
    """

    def __init__(self, port: str, *, timeout: float = 2.0):
        super().__init__()
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.port = port
        self.timeout = timeout
        self.name: str | None = None
        self.protocol_version: semver.VersionInfo | None = None
        self.rejected_frames = 0
        self._writer = None
        self._monitor_task = None
        self._pending = None
        self._expected = None
        self._request_lock = asyncio.Lock()
        self._tag = 0
        self._notify_callback = None

    async def connect(self):
        if self.connection_state_observable.value != ConnectionState.DISCONNECTED:
            raise RuntimeError("already connecting or connected")
        await self._close()
        try:
            await super().connect()
        except BaseException:
            # Identification can fail before the base installs its cleanup callback.
            # Preserve the identification/subscription error through cleanup.
            with contextlib.suppress(Exception):
                await self._close()
            raise

    async def _client_connect(self):
        options = {"exclusive": True} if os.name == "posix" else {}
        reader, self._writer = await serial_asyncio.open_serial_connection(
            url=self.port, baudrate=115200, **options
        )
        self.rejected_frames = 0
        self.status_observable.on_next(StatusFlag(0))
        self._notify_callback = None
        self._monitor_task = asyncio.create_task(self._monitor(reader))
        # Flush an incomplete preceding frame without guessing the protocol.
        self._writer.write(bytes([DELIMITER]))
        self.protocol_version = semver.VersionInfo.parse(
            (await self.read_gatt_char(SW_REV_UUID)).decode()
        )
        if (
            not semver.VersionInfo(1, 6, 0)
            <= self.protocol_version
            < semver.VersionInfo(2)
        ):
            raise RuntimeError(f"Unsupported CDC protocol: {self.protocol_version}")
        self.name = (await self.read_gatt_char(DEVICE_NAME_UUID)).decode()
        self.fw_version = Version((await self.read_gatt_char(FW_REV_UUID)).decode())
        _, _, self.hub_kind, self.hub_variant = unpack_pnp_id(
            await self.read_gatt_char(PNP_ID_UUID)
        )
        (
            self._max_write_size,
            self._capability_flags,
            self._max_user_program_size,
            self._num_of_slots,
        ) = unpack_hub_capabilities(
            await self.read_gatt_char(PYBRICKS_HUB_CAPABILITIES_UUID)
        )
        if self._max_write_size < 1:
            raise ValueError("Hub advertised a zero command size")

    async def _close(self):
        task, self._monitor_task = self._monitor_task, None
        writer, self._writer = self._writer, None
        if self._pending is not None and not self._pending.done():
            self._pending.set_exception(HubDisconnectError("serial session closed"))
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        try:
            if writer is not None:
                writer.close()
                try:
                    await asyncio.wait_for(writer.wait_closed(), self.timeout)
                except BaseException:
                    writer.transport.abort()
                    raise
        finally:
            self._handle_disconnect()

    async def disconnect(self):
        await self._close()

    async def _exchange(self, message, expected):
        async with self._request_lock:
            if self._writer is None or self._writer.is_closing():
                raise HubDisconnectError("serial session is closed")
            self._pending = asyncio.get_running_loop().create_future()
            self._expected = expected

            async def exchange():
                self._writer.write(pack(message))
                await self._writer.drain()
                return await self._pending

            try:
                return await asyncio.wait_for(exchange(), self.timeout)
            except BaseException:
                with contextlib.suppress(Exception):
                    await self._close()
                raise
            finally:
                self._pending = self._expected = None

    async def read_gatt_char(self, uuid: str) -> bytearray:
        uuid = str(UUID(uuid))
        if uuid.endswith("-0000-1000-8000-00805f9b34fb") and uuid.startswith("0000"):
            service = 1
        elif uuid.startswith("c5f5") and uuid.endswith("-8280-46da-89f4-6d8051e4aeef"):
            service = 2
        else:
            raise ValueError(f"Unsupported characteristic namespace: {uuid}")
        selector = struct.pack("<BH", service, int(uuid[4:8], 16))
        value = await self._exchange(
            bytes([Outgoing.READ]) + selector, bytes([Incoming.READ_REPLY]) + selector
        )
        return bytearray(value)

    async def write_gatt_char(self, uuid: str, data, response: bool) -> None:
        if str(UUID(uuid)) != PYBRICKS_COMMAND_EVENT_UUID or not response:
            raise ValueError("CDC requires a Pybricks command with response")
        if not 1 <= len(data) <= self._max_write_size:
            raise ValueError("Command exceeds the hub's advertised size")
        tag = self._tag
        self._tag = (tag + 1) % 256
        reply = await self._exchange(
            bytes([Outgoing.COMMAND, tag]) + data, bytes([Incoming.RESPONSE, tag])
        )
        status = int.from_bytes(reply, "little")
        if status:
            raise RuntimeError(f"Write failed: {status}")

    async def start_notify(self, uuid: str, callback: Callable) -> None:
        if uuid == NUS_TX_UUID:
            return  # Profile 1.6+ carries stdout in the Pybricks event stream.
        if str(UUID(uuid)) != PYBRICKS_COMMAND_EVENT_UUID:
            raise ValueError("Only Pybricks events are supported over CDC")
        if self._writer is None or self._writer.is_closing():
            raise HubDisconnectError("serial session is closed")
        self._notify_callback = callback
        self._writer.write(pack(bytes([Outgoing.SUBSCRIBE, 1])))
        await asyncio.wait_for(self._writer.drain(), self.timeout)

    async def _monitor(self, reader):
        frame = bytearray()
        overflow = False
        try:
            while chunk := await reader.read(1024):
                for byte in chunk:
                    if byte != DELIMITER:
                        if len(frame) < MAX_FRAME:
                            frame.append(byte)
                        else:
                            overflow = True
                        continue
                    try:
                        if overflow:
                            raise ValueError("oversized frame")
                        message = unpack(frame) if frame else None
                    except ValueError:
                        self.rejected_frames += 1
                    else:
                        if message is not None and not self._receive(message):
                            self.rejected_frames += 1
                    finally:
                        frame.clear()
                        overflow = False
            raise HubDisconnectError("serial device disconnected")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self._pending is not None and not self._pending.done():
                self._pending.set_exception(error)
            if self._writer is not None:
                self._writer.transport.abort()
            self._handle_disconnect()

    def _receive(self, message):
        if not message:
            return False
        if message[0] == Incoming.EVENT:
            if len(message) < 2 or (message[1] == 0 and len(message) < 6):
                return False
            if self._notify_callback is not None:
                self._notify_callback(None, message[1:])
        elif message[0] in (Incoming.RESPONSE, Incoming.READ_REPLY):
            if message[0] == Incoming.RESPONSE and len(message) != 6:
                return False
            if message[0] == Incoming.READ_REPLY and len(message) < 4:
                return False
            if (
                self._pending is not None
                and not self._pending.done()
                and message.startswith(self._expected)
            ):
                self._pending.set_result(message[len(self._expected) :])
        else:
            return False
        return True
