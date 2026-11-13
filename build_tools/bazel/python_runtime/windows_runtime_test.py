# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import _ctypes
import _hashlib
import _sqlite3
import _ssl
import _tkinter
import hashlib
import importlib.resources
import os
import pathlib
import sqlite3
import ssl
import subprocess
import sys
import tkinter


def main() -> None:
    assert sys.version_info[:3] == (3, 12, 13)
    assert sys.dont_write_bytecode
    assert pathlib.Path(sys.executable).suffix.lower() == ".exe"

    pathlib.Path(ssl.__file__).read_bytes()
    email_resources = importlib.resources.files("email")
    assert email_resources.joinpath("__init__.py").read_text(encoding="utf-8")

    assert hashlib.sha256(b"hrx").hexdigest()
    with sqlite3.connect(":memory:") as database:
        assert database.execute("SELECT 6 * 7").fetchone() == (42,)
    tcl_library = pathlib.Path(os.environ["TCL_LIBRARY"])
    runtime_root = tcl_library.parents[1]
    assert tcl_library.joinpath("init.tcl").is_file()
    assert pathlib.Path(os.environ["TK_LIBRARY"], "tk.tcl").is_file()
    for extension_module in (_ctypes, _hashlib, _sqlite3, _ssl, _tkinter):
        assert pathlib.Path(extension_module.__file__).samefile(
            runtime_root / "DLLs" / f"{extension_module.__name__}.pyd"
        )
    interpreter = tkinter.Tcl()
    assert interpreter.eval("info patchlevel") == "8.6.12"
    assert interpreter.eval("expr 6 * 7") == "42"

    child_output = subprocess.check_output(
        [
            sys.executable,
            "-I",
            "-c",
            "import hashlib, os, pathlib, sqlite3, ssl, tkinter; "
            "runtime_root = pathlib.Path(os.environ['TCL_LIBRARY']).parents[1]; "
            "assert pathlib.Path(tkinter._tkinter.__file__).samefile("
            "runtime_root / 'DLLs' / '_tkinter.pyd'); "
            "assert tkinter.Tcl().eval('info patchlevel') == '8.6.12'; "
            "print(hashlib.sha256(b'child').hexdigest())",
        ],
        text=True,
    )
    assert child_output.strip() == hashlib.sha256(b"child").hexdigest()


if __name__ == "__main__":
    main()
