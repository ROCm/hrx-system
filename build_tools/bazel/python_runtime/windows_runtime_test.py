# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib
import importlib.resources
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
    assert tkinter.Tcl().eval("expr 6 * 7") == "42"

    child_output = subprocess.check_output(
        [
            sys.executable,
            "-I",
            "-c",
            "import hashlib, sqlite3, ssl, tkinter; "
            "print(hashlib.sha256(b'child').hexdigest())",
        ],
        text=True,
    )
    assert child_output.strip() == hashlib.sha256(b"child").hexdigest()


if __name__ == "__main__":
    main()
