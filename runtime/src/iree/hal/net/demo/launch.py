# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Launches CPU collective workers and joins their final output before shutdown."""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import threading


def forward_output(process, output_lock):
    for line in process.stdout:
        with output_lock:
            print(line, end="", flush=True)
        if " ready_to_close=1" in line:
            return True
    return False


def forward_errors(process, output_lock):
    for line in process.stderr:
        with output_lock:
            print(line, end="", file=sys.stderr, flush=True)


def launch(binary, count, iterations):
    processes = []
    contacts = []
    output_lock = threading.Lock()
    readers = concurrent.futures.ThreadPoolExecutor(max_workers=2 * count)
    try:
        outputs = {}
        errors = []
        for rank in range(count):
            process = subprocess.Popen(
                [
                    binary,
                    f"--rank={rank}",
                    f"--count={count}",
                    "--address=127.0.0.1:0",
                    f"--iterations={iterations}",
                    *(f"--peer_address={address}" for address in contacts),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes.append(process)
            if rank + 1 < count:
                for line in process.stderr:
                    with output_lock:
                        print(line, end="", file=sys.stderr, flush=True)
                    if line.startswith("Listening on "):
                        contacts.append(line.removeprefix("Listening on ").strip())
                        break
                else:
                    raise RuntimeError(f"rank {rank} did not publish its address")
            outputs[readers.submit(forward_output, process, output_lock)] = process
            errors.append(readers.submit(forward_errors, process, output_lock))
        ready = []
        failed = False
        for future in concurrent.futures.as_completed(outputs):
            process = outputs[future]
            if future.result():
                if failed:
                    process.stdin.write("abort\n")
                    process.stdin.flush()
                else:
                    ready.append(process)
            else:
                failed = True
                for waiting in ready:
                    waiting.stdin.write("abort\n")
                    waiting.stdin.flush()
                ready.clear()
        # Every successful worker has joined its actual local operations.
        # Closing membership can now neither cancel another rank's final
        # request nor authorize premature buffer reuse.
        for process in ready:
            process.stdin.write("close\n")
            process.stdin.flush()
        for process in processes:
            process.wait()
            failed |= process.returncode != 0
        for future in errors:
            future.result()
        return 1 if failed else 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
        # Kill processes before joining readers: an exception during startup or
        # shutdown must not leave a reader waiting on a still-live worker pipe.
        readers.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--count", type=int, default=3, choices=range(2, 9))
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()
    return launch(os.path.abspath(args.binary), args.count, args.iterations)


if __name__ == "__main__":
    sys.exit(main())
