# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercises the complete CPU projection in separate worker processes."""

import argparse
import concurrent.futures
import os
import subprocess
import sys


def run_group(binary, count, iterations, interrupt_rank=None):
    processes = []
    try:
        contacts = []
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
                startup = []
                for line in process.stderr:
                    startup.append(line)
                    if line.startswith("Listening on "):
                        contacts.append(line.removeprefix("Listening on ").strip())
                        break
                else:
                    raise AssertionError("Listener did not start:\n" + "".join(startup))
        if interrupt_rank is not None:
            # One checked request witnesses real allocation, compute and IO.
            # The long run cannot finish into its bounded, unread stdout pipe.
            first = processes[interrupt_rank].stdout.readline()
            assert "request=1 " in first, first
            processes[interrupt_rank].kill()

        def read_to_close(process):
            lines = []
            for line in process.stdout:
                lines.append(line)
                if " ready_to_close=1" in line:
                    return True, "".join(lines)
            return False, "".join(lines)

        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as readers:
            futures = {
                readers.submit(read_to_close, process): rank
                for rank, process in enumerate(processes)
            }
            outputs = [None] * count
            ready = [False] * count
            for future in concurrent.futures.as_completed(futures):
                rank = futures[future]
                ready[rank], outputs[rank] = future.result()
            if interrupt_rank is None:
                assert all(ready), outputs
            for rank, process in enumerate(processes):
                if ready[rank]:
                    process.stdin.write(
                        "close\n" if interrupt_rank is None else "abort\n"
                    )
                    process.stdin.flush()
            futures = {
                readers.submit(process.communicate): rank
                for rank, process in enumerate(processes)
            }
            results = [None] * count
            for future in concurrent.futures.as_completed(futures):
                rank = futures[future]
                output, errors = future.result()
                results[rank] = (outputs[rank] + output, errors)
        rows = []
        for rank, (process, (output, errors)) in enumerate(zip(processes, results)):
            print(f"rank {rank}:\n{output}{errors}", flush=True)
            if interrupt_rank is not None:
                if rank == interrupt_rank:
                    assert process.returncode != 0
                else:
                    assert process.returncode == 1, (process.returncode, errors)
                    assert "Worker verified" not in errors
                    for diagnostic in (
                        "ERROR: AddressSanitizer",
                        "ERROR: LeakSanitizer",
                        "runtime error:",
                    ):
                        assert diagnostic not in errors, errors
                continue
            assert process.returncode == 0, (process.returncode, errors)
            assert "Worker verified X*W + residual. All work retired." in errors
            records = [
                dict(field.split("=", 1) for field in line.split())
                for line in output.splitlines()
            ]
            requests = [record for record in records if "request" in record]
            assert [int(row["request"]) for row in requests] == [
                *range(1, iterations + 1),
                0,
            ]
            for row in requests:
                assert int(row["rank"]) == rank
                odd = int(row["request"]) % 2
                assert row["shape"] == ("13x48x24" if odd else "8x32x16")
                assert int(row["checked"]) == (312 if odd else 128)
            assert records[-4] == {
                "rank": str(rank),
                "independent_requests": str(iterations),
                "held_consumer": "pending",
            }
            assert records[-2] == {
                "rank": str(rank),
                "live_tensor_reservations": "0",
                "reusable_plans": str(4 * (count - 1)),
            }
            assert records[-1] == {"rank": str(rank), "ready_to_close": "1"}
            rows.append([row["checksum"] for row in requests])
        if interrupt_rank is None:
            assert all(row == rows[0] for row in rows), rows
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--launcher", required=True)
    args = parser.parse_args()
    binary = os.path.abspath(args.binary)
    for count in (2, 3, 4):
        launched = subprocess.run(
            [
                sys.executable,
                args.launcher,
                f"--binary={binary}",
                f"--count={count}",
                "--iterations=3",
            ],
            capture_output=True,
            text=True,
        )
        assert launched.returncode == 0, (launched.stdout, launched.stderr)
        assert launched.stderr.count("All work retired.") == count, launched.stderr
        for rank in range(count):
            assert f"rank={rank} ready_to_close=1" in launched.stdout
        for iterations in (1, 9):
            run_group(binary, count, iterations)
        for rank in range(count):
            run_group(binary, count, 1000000, interrupt_rank=rank)
    rejected = subprocess.run(
        [sys.executable, args.launcher, f"--binary={binary}", "--iterations=0"],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0, rejected
    assert "rank 0 did not publish its address" in rejected.stderr, rejected.stderr


if __name__ == "__main__":
    main()
