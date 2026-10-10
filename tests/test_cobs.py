# SPDX-License-Identifier: MIT
import pytest

from pybricksdev.usb.cobs import pack, unpack


@pytest.mark.parametrize(
    "message, wire",
    [
        # Pybricks lib/pbio/test/src/test_cobs.c known vectors.
        (b"\x00", bytes.fromhex("00 00 02")),
        (b"ABC", bytes.fromhex("05 42 41 40 02")),
    ],
)
def test_known_vectors(message, wire):
    assert pack(message) == wire
    assert unpack(wire[:-1]) == message
    assert unpack(b"\x01" + wire[:-1]) == message


@pytest.mark.parametrize(
    "body", [b"\x03", b"\x01\x03", b"\x00\x01", bytes.fromhex("5b 00 00 06 29")]
)
def test_invalid_frames(body):
    with pytest.raises(ValueError):
        unpack(body)


def test_block_boundaries_and_all_values():
    for length in (0, 1, 82, 83, 84, 85, 168, 512):
        for pattern in (b"\x00", b"\x01", b"\x02", b"\xff", bytes(range(256))):
            message = (pattern * (length + 1))[:length]
            wire = pack(message)
            assert not {1, 2, 3}.intersection(wire[:-1])
            assert unpack(wire[:-1]) == message
