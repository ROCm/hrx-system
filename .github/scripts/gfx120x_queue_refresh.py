# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""One debugger-owned queue-update experiment; not a runtime recovery policy."""

import json
import os
import re
import struct
import time
from pathlib import Path

import gdb


def report(event, **fields):
    print("[queue-refresh] " + json.dumps({"event": event, **fields}), flush=True)


def read_identity(name):
    try:
        return Path("/sys/module/amdgpu", name).read_text().strip()
    except FileNotFoundError:
        return None
    except OSError as error:
        return "unreadable: " + str(error)


def candidate_driver(release, version, taint):
    if version is not None:
        # AMD's fourth component identifies the packaged release independently
        # of its kernel base. Release 31.60 is the first with MES mode 2.
        match = re.fullmatch(r"\d+\.\d+\.\d+\.(\d+)", version)
        return match is not None and 0 < int(match[1]) < 31600000
    # Unidentified out-of-tree modules cannot inherit the host's fix history.
    if taint is None or "O" in taint:
        return False
    if taint.startswith("unreadable:"):
        return False
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", release)
    if match is None:
        return False
    return tuple(map(int, match.groups())) < (7, 3, 0)


identity = {
    "release": os.uname().release,
    "version": read_identity("version"),
    "srcversion": read_identity("srcversion"),
    "taint": read_identity("taint"),
}
eligible = candidate_driver(identity["release"], identity["version"], identity["taint"])
report("identity", **identity, candidate=eligible)

# All of this state belongs to the debugger. The inferior retains its normal
# command, allocation, completion, and teardown paths.
active = None
pending = False
attempted = False
exit_code = None
publication_count = 0


def snapshot(record):
    queue = record["queue"].dereference()
    native = record["native"].dereference()
    control = int(native["control"]["host_pointer"])
    plan = native["plan"]["control"]
    inferior = gdb.selected_inferior()

    def read_u64(address):
        return struct.unpack("<Q", inferior.read_memory(address, 8))[0]

    signal = queue["control"]["contents"]["data"].cast(
        gdb.lookup_type("iree_hal_amd_gpu_aql_signal_t").pointer()
    )
    return {
        "queue_id": int(native["queue_identifier"]),
        "producer": read_u64(control + int(plan["write_index_byte_offset"])),
        "consumed_native": read_u64(control + int(plan["read_index_byte_offset"])),
        "completion": int(signal.dereference()["value"]) & 0xFFFFFFFF,
    }


class PublicationReturn(gdb.FinishBreakpoint):
    def __init__(self):
        super().__init__(gdb.newest_frame(), internal=True)

    def stop(self):
        global active
        if active is not None and attempted:
            report("publication-return", **snapshot(active))
        active = None
        native_wait.enabled = False
        publication_entry.enabled = not attempted
        return False


class PublicationEntry(gdb.Breakpoint):
    def stop(self):
        global active, publication_count
        if not eligible or attempted:
            self.enabled = False
            return False
        queue = gdb.newest_frame().read_var("queue")
        value = queue.dereference()
        gfx = value["context"].dereference()["target_info"]["gfx_ip"]
        if int(gfx["major"]) != 12 or int(gfx["minor"]) != 0:
            return False
        if bool(value["aql"]) or bool(value["kernel"]):
            return False
        gpu_queue = value["user"]["queue"].cast(
            gdb.lookup_type("amdf_gpu_user_queue_t").pointer()
        )
        active = {
            "queue": queue,
            "native": gpu_queue.dereference()["umd"],
            "thread": gdb.selected_thread().global_num,
            "start": time.monotonic(),
        }
        publication_count += 1
        if publication_count == 1:
            report("first-publication", **snapshot(active))
        self.enabled = False
        native_wait.enabled = True
        PublicationReturn()
        return False


class NativeWait(gdb.Breakpoint):
    def stop(self):
        global pending
        if active is None or attempted:
            return False
        if gdb.selected_thread().global_num != active["thread"]:
            return False
        # This is an observation threshold for one intervention, not a new
        # completion deadline. The original test harness still owns timeout.
        if time.monotonic() - active["start"] < 5:
            return False
        if snapshot(active)["completion"] == 0:
            return False
        self.enabled = False
        pending = True
        return True


def refresh():
    global attempted, pending
    attempted = True
    pending = False
    report("before", **snapshot(active))
    gdb.execute("thread apply all bt 8")
    # Another host thread may own the provider's native-memory lock. It must
    # remain runnable while the existing callback acquires that lock.
    gdb.execute("set scheduler-locking off")
    native_pointer = active["native"]
    native = native_pointer.dereference()
    # Use the native callback directly: the scratch helper which also invokes
    # this callback requires joined execution and is not a live-queue API.
    payload = struct.pack(
        "<QIIII",
        int(native["ring"]["device_address"])
        + int(native["plan"]["ring"]["primary_byte_offset"]),
        int(native["queue_identifier"]),
        int(native["plan"]["ring"]["primary_byte_length"]),
        100,
        int(native["native_priority"]),
    )
    allocation = int(gdb.parse_and_eval(f"(void*)malloc({len(payload)})"))
    if allocation == 0:
        raise RuntimeError("debugger argument allocation failed")
    gdb.selected_inferior().write_memory(allocation, payload)
    gdb.set_convenience_variable("refresh_native", native_pointer)
    status = int(
        gdb.parse_and_eval(
            "amdf_gpu_kfd_user_queue_update_native(0, "
            "$refresh_native->device, "
            f"(const struct kfd_ioctl_update_queue_args*){allocation})"
        )
    )
    gdb.parse_and_eval(f"(void)free((void*){allocation})")
    report("after", status=status, **snapshot(active))
    if status != 0:
        raise RuntimeError("native queue update failed; no second attempt")


def exited(event):
    global exit_code
    exit_code = getattr(event, "exit_code", 1)


gdb.events.exited.connect(exited)
gdb.execute("set pagination off")
gdb.execute("set confirm off")
gdb.execute("set breakpoint pending on")
gdb.execute("set startup-with-shell off")
gdb.execute("set disable-randomization off")
gdb.execute("set print thread-events off")
gdb.execute("handle SIGPIPE nostop noprint pass")
native_wait = NativeWait("amdf_gpu_umd_event_wait", internal=True)
native_wait.enabled = False
publication_entry = PublicationEntry("iree_hal_amd_gpu_code_queue_wait", internal=True)
try:
    gdb.execute("run")
    while exit_code is None:
        if not pending:
            raise RuntimeError("unexpected inferior stop")
        refresh()
        gdb.execute("continue")
except Exception as error:
    report("experiment-error", error=str(error))
    gdb.execute("thread apply all bt 12")
    gdb.execute("quit 2")
report("exit", code=exit_code, publications=publication_count, attempted=attempted)
gdb.execute(f"quit {exit_code}")
