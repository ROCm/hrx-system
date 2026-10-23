# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Records native access and resource users for files named by failed builds.

The collector runs this process with a timeout. Each observation is emitted as
one JSON line before querying Restart Manager, so a slow resource query cannot
erase completed access observations. Opening with DELETE access checks sharing
and permissions without deleting the file or changing its disposition.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import os
from collections.abc import Iterator
from ctypes import wintypes
from pathlib import Path

GENERIC_READ = 0x80000000
DELETE = 0x00010000
FILE_SHARE_ALL = 0x00000007
OPEN_EXISTING = 3
FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
MAX_RESOURCE_USERS = 256
MAX_TREE_PATHS = 1024


class UniqueProcess(ctypes.Structure):
    _fields_ = [
        # Process ID, paired with creation time to distinguish PID reuse.
        ("process_id", wintypes.DWORD),
        # Windows FILETIME counts 100-nanosecond ticks since 1601-01-01 UTC.
        ("start_time", wintypes.FILETIME),
    ]


class ProcessInfo(ctypes.Structure):
    _fields_ = [
        # Identity of an application using a registered resource.
        ("process", UniqueProcess),
        # Restart Manager's application or service display name.
        ("application_name", wintypes.WCHAR * 256),
        # Service name, empty for ordinary applications.
        ("service_name", wintypes.WCHAR * 64),
        # RM_APP_TYPE classifies services, windows, and critical processes.
        ("application_type", wintypes.UINT),
        # RM_APP_STATUS flags reported by Restart Manager.
        ("status", wintypes.ULONG),
        # Terminal Services session ID, or RM_INVALID_SESSION.
        ("session_id", wintypes.DWORD),
        # Required ABI field; collection never requests a restart.
        ("restartable", wintypes.BOOL),
    ]


def win32_result(code: int) -> dict:
    return {"winerror": code, "message": ctypes.FormatError(code).strip()}


class FileAccess:
    def __init__(self):
        # Bind only when invoked on Windows, keeping imports portable.
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        self.kernel.CreateFileW.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL

    def probe(self, path: str, access: int, *, directory: bool = False) -> dict:
        result = {"observed_at": dt.datetime.now(dt.UTC).isoformat()}
        handle = self.kernel.CreateFileW(
            path,
            access,
            FILE_SHARE_ALL,
            None,
            OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS if directory else 0,
            None,
        )
        if handle == ctypes.c_void_p(-1).value:
            result.update(win32_result(ctypes.get_last_error()))
        else:
            result.update(win32_result(0))
            if not self.kernel.CloseHandle(handle):
                result["close_error"] = win32_result(ctypes.get_last_error())
        return result


def resource_users(paths: list[str]) -> dict:
    """Queries one related group without stopping or restarting its users."""
    manager = ctypes.WinDLL("Rstrtmgr", use_last_error=True)
    manager.RmStartSession.argtypes = [
        ctypes.POINTER(wintypes.DWORD),
        wintypes.DWORD,
        wintypes.LPWSTR,
    ]
    manager.RmStartSession.restype = wintypes.DWORD
    manager.RmRegisterResources.argtypes = [
        wintypes.DWORD,
        wintypes.UINT,
        ctypes.POINTER(wintypes.LPCWSTR),
        wintypes.UINT,
        ctypes.POINTER(UniqueProcess),
        wintypes.UINT,
        ctypes.POINTER(wintypes.LPCWSTR),
    ]
    manager.RmRegisterResources.restype = wintypes.DWORD
    manager.RmGetList.argtypes = [
        wintypes.DWORD,
        ctypes.POINTER(wintypes.UINT),
        ctypes.POINTER(wintypes.UINT),
        ctypes.POINTER(ProcessInfo),
        ctypes.POINTER(wintypes.DWORD),
    ]
    manager.RmGetList.restype = wintypes.DWORD
    manager.RmEndSession.argtypes = [wintypes.DWORD]
    manager.RmEndSession.restype = wintypes.DWORD

    result = {"paths": paths, "observed_at": dt.datetime.now(dt.UTC).isoformat()}
    session = wintypes.DWORD()
    session_key = ctypes.create_unicode_buffer(33)
    code = manager.RmStartSession(ctypes.byref(session), 0, session_key)
    result["start_session"] = win32_result(code)
    if code:
        return result
    try:
        names = (wintypes.LPCWSTR * len(paths))(*paths)
        code = manager.RmRegisterResources(session, len(paths), names, 0, None, 0, None)
        result["register_resources"] = win32_result(code)
        if code:
            return result
        # One bounded snapshot avoids racing a changing process count in a
        # retry loop. ERROR_MORE_DATA retains the required count explicitly.
        processes = (ProcessInfo * MAX_RESOURCE_USERS)()
        needed = wintypes.UINT()
        count = wintypes.UINT(MAX_RESOURCE_USERS)
        reboot_reasons = wintypes.DWORD()
        code = manager.RmGetList(
            session,
            ctypes.byref(needed),
            ctypes.byref(count),
            processes,
            ctypes.byref(reboot_reasons),
        )
        result["get_list"] = win32_result(code)
        result["required_count"] = needed.value
        if not code:
            result["processes"] = [
                {
                    "pid": entry.process.process_id,
                    "start_time_100ns": (entry.process.start_time.dwHighDateTime << 32)
                    | entry.process.start_time.dwLowDateTime,
                    "application_name": entry.application_name,
                    "service_name": entry.service_name,
                    "application_type": entry.application_type,
                    "session_id": entry.session_id,
                }
                for entry in processes[: count.value]
            ]
        return result
    finally:
        result["end_session"] = win32_result(manager.RmEndSession(session))


def tree_paths(root: str) -> Iterator[str]:
    """Observes a bounded remainder of a failed removal without following links."""
    yield root
    count = 1

    def record_error(error: OSError) -> None:
        print(
            json.dumps(
                {
                    "tree_error": {
                        "path": error.filename,
                        "error": str(error),
                        "winerror": getattr(error, "winerror", None),
                    }
                }
            ),
            flush=True,
        )

    try:
        if Path(root).lstat().st_file_attributes & FILE_ATTRIBUTE_REPARSE_POINT:
            return
    except OSError as error:
        record_error(error)
        return
    for parent, directories, files in os.walk(root, onerror=record_error):
        for name in directories + files:
            if count == MAX_TREE_PATHS:
                print(
                    json.dumps({"tree_limit": {"root": root, "paths": count}}),
                    flush=True,
                )
                return
            yield str(Path(parent) / name)
            count += 1
        # Windows directory junctions are reparse points, but are not always
        # classified as symbolic links. The failed tree does not own their
        # targets, so neither kind may be traversed by the diagnostic.
        descend = []
        for name in directories:
            try:
                attributes = (Path(parent) / name).lstat().st_file_attributes
                if not attributes & FILE_ATTRIBUTE_REPARSE_POINT:
                    descend.append(name)
            except OSError as error:
                record_error(error)
        directories[:] = descend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read", action="append", default=[])
    parser.add_argument("--delete", action="append", default=[])
    parser.add_argument("--delete-tree", action="append", default=[])
    args = parser.parse_args()
    delete_paths = {str(Path(path).absolute()) for path in args.delete}
    paths = dict.fromkeys(
        str(Path(path).absolute()) for path in args.read + args.delete
    )
    access = FileAccess()
    resources = []
    for path in paths:
        result = {"path": path, "read": access.probe(path, GENERIC_READ)}
        if path in delete_paths:
            result["delete"] = access.probe(path, DELETE)
        print(json.dumps({"file_access": result}), flush=True)
        # Absent search-path candidates cannot identify a resource owner.
        # Keep sharing and permissions failures eligible for the query.
        if result["read"]["winerror"] not in (2, 3):
            resources.append(path)
    if resources:
        print(json.dumps({"resource_users": resource_users(resources)}), flush=True)

    for root in dict.fromkeys(str(Path(path).absolute()) for path in args.delete_tree):
        resources = []
        for path in tree_paths(root):
            result = {
                "path": path,
                "delete": access.probe(path, DELETE, directory=Path(path).is_dir()),
            }
            print(json.dumps({"file_access": result}), flush=True)
            # Restart Manager accepts files, not directories. Query only files
            # whose removal is currently denied; successful ancestor probes
            # do not prove that their children can be removed.
            if result["delete"]["winerror"] not in (0, 2, 3) and Path(path).is_file():
                resources.append(path)
        if resources:
            print(json.dumps({"resource_users": resource_users(resources)}), flush=True)


if __name__ == "__main__":
    main()
