#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Pybricks Authors

"""
Consistent Overhead Byte Stuffing (COBS) framing, SPIKE Prime variant.

This is the same scheme implemented in the firmware (see lib/pbio/src/cobs.c
and pbio/cobs.h). Both the SPIKE Prime protocol (spike.py) and the Pybricks USB
protocol (pybricks.py) frame their messages with this encoding.
"""

DELIMITER = 0x02
NO_DELIMITER = 0xFF
COBS_CODE_OFFSET = DELIMITER
MAX_BLOCK_SIZE = 84
XOR = 0x03


def cobs_encode(data: bytes) -> bytearray:
    """Encode data with SPIKE Prime's COBS variant (no 0x00/0x01/0x02 in output)."""
    buffer = bytearray()
    code_index = block = 0

    def begin_block():
        nonlocal code_index, block
        code_index = len(buffer)
        buffer.append(NO_DELIMITER)
        block = 1

    begin_block()
    for byte in data:
        if byte > DELIMITER:
            buffer.append(byte)
            block += 1
        if byte <= DELIMITER or block > MAX_BLOCK_SIZE:
            if byte <= DELIMITER:
                delimiter_base = byte * MAX_BLOCK_SIZE
                block_offset = block + COBS_CODE_OFFSET
                buffer[code_index] = delimiter_base + block_offset
            begin_block()
    buffer[code_index] = block + COBS_CODE_OFFSET
    return buffer


def cobs_decode(data: bytes) -> bytes:
    """Decode a complete SPIKE COBS body; reject truncated or reserved bytes."""
    buffer = bytearray()
    offset = 0
    while offset < len(data):
        code = data[offset]
        if code <= DELIMITER:
            raise ValueError("Reserved COBS code")
        if code == NO_DELIMITER:
            value, length = None, MAX_BLOCK_SIZE
        else:
            value, length = divmod(code - COBS_CODE_OFFSET - 1, MAX_BLOCK_SIZE)
        end = offset + 1 + length
        if end > len(data) or any(b <= DELIMITER for b in data[offset + 1 : end]):
            raise ValueError("Truncated or reserved COBS block")
        buffer.extend(data[offset + 1 : end])
        if end < len(data) and value is not None:
            buffer.append(value)
        offset = end
    return bytes(buffer)


def pack(data: bytes) -> bytes:
    """COBS-encode, XOR with 0x03, and frame with a trailing 0x02 delimiter."""
    buffer = cobs_encode(data)
    for i in range(len(buffer)):
        buffer[i] ^= XOR
    buffer.append(DELIMITER)
    return bytes(buffer)


def unpack(frame: bytes) -> bytes:
    """Unframe (strip optional 0x01 prefix), un-XOR, COBS-decode."""
    start = 1 if frame and frame[0] == 0x01 else 0
    unframed = bytes(b ^ XOR for b in frame[start:])
    return cobs_decode(unframed)
