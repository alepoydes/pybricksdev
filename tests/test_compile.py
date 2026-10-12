# SPDX-License-Identifier: MIT
# Copyright (c) 2022-2026 The Pybricks Authors


import contextlib
import os
import struct
import sys
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from pybricksdev.compile import compile_file, compile_multi_file

# TODO: Remove this when we drop support for Python 3.10.
if sys.version_info < (3, 11):
    from contextlib import AbstractContextManager

    class chdir(AbstractContextManager[None]):
        """Non thread-safe context manager to change the current working directory."""

        def __init__(self, path: str) -> None:
            self.path = path
            self._old_cwd: list[str] = []

        def __enter__(self) -> None:
            self._old_cwd.append(os.getcwd())
            os.chdir(self.path)

        def __exit__(self, *excinfo: Any) -> None:
            os.chdir(self._old_cwd.pop())

    setattr(contextlib, "chdir", chdir)


@pytest.mark.parametrize("abi", [5, 6])
@pytest.mark.asyncio
async def test_compile_file(abi: int):
    with TemporaryDirectory() as temp_dir:
        with open(os.path.join(temp_dir, "test.py"), "w", encoding="utf-8") as f:
            f.write("print('test')")

        mpy = await compile_file(
            os.path.dirname(f.name), os.path.basename(f.name), abi=abi
        )

        magic, abi_ver, flags, int_bits = struct.unpack_from("<BBBB", mpy)

        assert chr(magic) == "M"
        assert abi_ver == abi
        assert flags == 0
        assert int_bits == 31


@pytest.mark.parametrize("abi", [4, (6, 9), (5, 1)])
@pytest.mark.asyncio
async def test_compile_file_invalid_abi(abi):
    with TemporaryDirectory() as temp_dir:
        with open(os.path.join(temp_dir, "test.py"), "w", encoding="utf-8") as f:
            f.write("print('test')")

        with pytest.raises(ValueError, match="mpy_version must be 5 or 6"):
            await compile_file(
                os.path.dirname(f.name), os.path.basename(f.name), abi=abi
            )


@pytest.mark.parametrize("abi", [5, 6])
@pytest.mark.asyncio
async def test_compile_multi_file(abi: int):
    with TemporaryDirectory() as temp_dir, contextlib.chdir(temp_dir):
        with open(os.path.join(temp_dir, "test.py"), "w", encoding="utf-8") as f:
            f.writelines(
                [
                    "from pybricks import version\n",
                    "import test1\n",
                    "from test2 import thing2\n",
                    "from nested.test3 import thing3\n",
                    "from test4 import thing4\n",
                    "from nested.test5 import thing5\n",
                ]
            )

        with open(os.path.join(temp_dir, "test1.py"), "w", encoding="utf-8") as f1:
            f1.write("thing1 = 'thing1'\n")

        with open(os.path.join(temp_dir, "test2.py"), "w", encoding="utf-8") as f2:
            f2.write("thing2 = 'thing2'\n")

        os.mkdir("nested")

        # We didn't implement a better way for older ABI yet.
        uses_module_finder = abi == 5

        # Work around bug where ModuleFinder can't handle implicit namespace
        # packages by adding an __init__.py file.
        if uses_module_finder:
            with open(
                os.path.join(temp_dir, "nested", "__init__.py"), "w", encoding="utf-8"
            ) as init:
                init.write("")

        with open(
            os.path.join(temp_dir, "nested", "test3.py"), "w", encoding="utf-8"
        ) as f3:
            f3.write("thing3 = 'thing3'\n")

        # test4 and test5 are to test package modules with non-empty __init__.py

        os.mkdir("test4")

        with open(
            os.path.join(temp_dir, "test4", "__init__.py"), "w", encoding="utf-8"
        ) as f4:
            f4.write("thing4 = 'thing4'\n")

        os.mkdir(os.path.join("nested", "test5"))

        with open(
            os.path.join(temp_dir, "nested", "test5", "__init__.py"),
            "w",
            encoding="utf-8",
        ) as f5:
            f5.write("thing5 = 'thing5'\n")

        multi_mpy = await compile_multi_file("test.py", abi)
        pos = 0

        def unpack_mpy(data: bytes) -> tuple[bytes, bytes]:
            nonlocal pos

            size = struct.unpack_from("<I", multi_mpy, pos)[0]
            pos += 4

            name = bytearray()

            # read zero-terminated string
            while multi_mpy[pos] != 0:
                name.append(multi_mpy[pos])
                pos += 1

            pos += 1  # skip 0 byte

            mpy = multi_mpy[pos : pos + size]
            pos += size

            return name, mpy

        names = set[str]()

        name1, mpy1 = unpack_mpy(multi_mpy)
        name2, mpy2 = unpack_mpy(multi_mpy)
        names.add(name2.decode())
        name3, mpy3 = unpack_mpy(multi_mpy)
        names.add(name3.decode())

        if uses_module_finder:
            # ModuleFinder requires __init__.py.
            name4, mpy4 = unpack_mpy(multi_mpy)
            names.add(name4.decode())

        name5, mpy5 = unpack_mpy(multi_mpy)
        names.add(name5.decode())

        name6, mpy6 = unpack_mpy(multi_mpy)
        names.add(name6.decode())

        name7, mpy7 = unpack_mpy(multi_mpy)
        names.add(name7.decode())

        assert pos == len(multi_mpy)

        # It is important that the main module is first.
        assert name1.decode() == "__main__"

        # The other modules can be in any order.
        assert "test1" in names
        assert "test2" in names
        if uses_module_finder:
            assert "nested" in names
        assert "nested.test3" in names

        if uses_module_finder:
            assert len(names) == 6
        else:
            assert len(names) == 5

        def check_mpy(mpy: bytes) -> None:
            magic, abi_ver, flags, int_bits = struct.unpack_from("<BBBB", mpy)

            assert chr(magic) == "M"
            assert abi_ver == abi
            assert flags == 0
            assert int_bits == 31

        check_mpy(mpy1)
        check_mpy(mpy2)
        check_mpy(mpy3)
        if uses_module_finder:
            check_mpy(mpy4)  # pyright: ignore[reportPossiblyUnboundVariable]
        check_mpy(mpy5)
        check_mpy(mpy6)
        check_mpy(mpy7)


def unpack_modules(package):
    modules = {}
    while package:
        size = int.from_bytes(package[:4], "little")
        name, _, remainder = package[4:].partition(b"\0")
        modules[name.decode()] = remainder[:size]
        package = remainder[size:]
    return modules


@pytest.mark.asyncio
async def test_native_v63_modules_and_imports(tmp_path):
    program = tmp_path / "program.py"
    program.write_text("from fast import calculate\nprint(calculate(21))\n")
    fast = tmp_path / "fast.py"
    fast.write_text(
        "import micropython\nfrom constants import factor\n"
        "@micropython.native\ndef calculate(x): return x * factor\n"
    )
    (tmp_path / "constants.py").write_text("factor = 2\n")
    native = await compile_file(str(tmp_path), fast.name, (6, 3), ["-march=armv7m"])
    assert native[:4] == bytes([ord("M"), 6, (5 << 2) | 3, 31])
    fast.unlink()
    (tmp_path / "fast.mpy").write_bytes(native)
    modules = unpack_modules(await compile_multi_file(str(program), (6, 3)))
    assert set(modules) == {"__main__", "fast", "constants"}
    assert modules["fast"] == native


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header, abi, error",
    [
        (b"M", (6, 3), "invalid .mpy header"),
        (bytes([ord("M"), 5, 0, 31]), (6, 3), "major version"),
        (bytes([ord("M"), 6, 21, 31]), (6, 3), "minor version"),
        (bytes([ord("M"), 6, 23, 31]), 6, "no native ABI"),
    ],
)
async def test_reject_incompatible_precompiled_module(tmp_path, header, abi, error):
    program = tmp_path / "program.py"
    program.write_text("import incompatible\n")
    (tmp_path / "incompatible.mpy").write_bytes(header)
    with pytest.raises(ValueError, match=error):
        await compile_multi_file(str(program), abi)


@pytest.mark.asyncio
async def test_precompiled_bytecode_package(tmp_path):
    program = tmp_path / "program.py"
    program.write_text("import package\nprint(package.value)\n")
    (tmp_path / "package").mkdir()
    init = tmp_path / "package/__init__.py"
    init.write_text("value = 42\n")
    bytecode = await compile_file(str(tmp_path), "package/__init__.py", 6)
    init.unlink()
    init.with_suffix(".mpy").write_bytes(bytecode)
    modules = unpack_modules(await compile_multi_file(str(program), 6))
    assert set(modules) == {"__main__", "package"}
    assert modules["package"] == bytecode
