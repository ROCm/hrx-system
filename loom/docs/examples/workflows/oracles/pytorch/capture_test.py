# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualifies the public example with real PyTorch and installed-tool interfaces."""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from capture import capture_example, capture_module, save_tensor

TOOL = Path(sys.argv.pop(1)).resolve()
BENCHMARK = Path(sys.argv.pop(1)).resolve()
SOURCE = Path(sys.argv.pop(1)).resolve()
DEVICE = sys.argv.pop(1)


class CaptureTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_example_records_original_input_and_reused_module_calls(self):
        capture_example(self.root / "captures")
        for call, start in enumerate((-4, -2)):
            original = np.arange(start, start + 8, dtype=np.float32).reshape(4, 2).T
            directory = self.root / "captures/relu" / f"call-{call}"
            self.assertTrue(np.load(directory / "input.npy").flags.c_contiguous)
            np.testing.assert_array_equal(
                np.load(directory / "input.npy", allow_pickle=False), original
            )
            np.testing.assert_array_equal(
                np.load(directory / "output.npy", allow_pickle=False),
                np.maximum(original, 0),
            )
            metadata = json.loads((directory / "input.json").read_text())
            self.assertEqual(metadata["stride_elements"], [1, 2])
            self.assertEqual(metadata["shape"], [2, 4])
            with (directory / "input.npy").open("rb") as stream:
                self.assertEqual(
                    metadata["sha256"],
                    hashlib.file_digest(stream, "sha256").hexdigest(),
                )
        self.assertTrue((self.root / "captures/run.json").is_file())

    def test_bf16_and_float8_preserve_bits_without_casting(self):
        bfloat_bits = torch.tensor([-32768, 0, 32704, 16256], dtype=torch.int16)
        tensors = {"bf16": bfloat_bits.view(torch.bfloat16)}
        tensors.update(
            (name, torch.arange(8, dtype=torch.int8).view(getattr(torch, name)))
            for name in dir(torch)
            if name.startswith("float8_")
            and isinstance(getattr(torch, name), torch.dtype)
        )
        for name, tensor in tensors.items():
            with self.subTest(dtype=name):
                path = self.root / f"{name}.npy"
                save_tensor(path, tensor)
                carrier = torch.int16 if name == "bf16" else torch.int8
                np.testing.assert_array_equal(
                    np.load(path, allow_pickle=False), tensor.view(carrier).numpy()
                )
                metadata = json.loads(path.with_suffix(".json").read_text())
                self.assertEqual(metadata["dtype"], str(tensor.dtype))

    def test_scalar_empty_and_native_values_are_not_retained(self):
        tensors = [torch.empty((0, 3)), torch.tensor(3.25, dtype=torch.float64)]
        tensors.extend(
            torch.tensor([0, 1, 2], dtype=dtype)
            for dtype in (
                torch.bool,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.float16,
                torch.float32,
                torch.complex64,
            )
        )
        for index, tensor in enumerate(tensors):
            path = self.root / f"value-{index}.npy"
            save_tensor(path, tensor)
            np.testing.assert_array_equal(
                np.load(path, allow_pickle=False), tensor.numpy()
            )
        value = torch.arange(8, dtype=torch.float32)
        reference = weakref.ref(value)
        save_tensor(self.root / "temporary.npy", value)
        del value
        self.assertIsNone(reference())

    def test_selection_and_hook_cleanup_on_failure(self):
        class Selected(torch.nn.Module):
            def forward(self, *, value):
                return {"hidden": value + 1}

        module = Selected()
        with capture_module(
            module,
            self.root / "selected",
            select_input=lambda args, kwargs: kwargs["value"],
            select_output=lambda result: result["hidden"],
        ):
            module(value=torch.tensor([4.0]))
        self.assertFalse(module._forward_hooks)
        self.assertFalse(module._forward_pre_hooks)
        np.testing.assert_array_equal(
            np.load(self.root / "selected/call-0/output.npy", allow_pickle=False), [5.0]
        )
        with self.assertRaisesRegex(RuntimeError, "stop capture"):
            with capture_module(module, self.root / "failed"):
                raise RuntimeError("stop capture")
        self.assertFalse(module._forward_hooks)
        self.assertFalse(module._forward_pre_hooks)
        with self.assertRaises(FileExistsError):
            save_tensor(self.root / "selected/call-0/output.npy", torch.tensor([7.0]))

    def test_recursive_calls_keep_their_own_input_and_output(self):
        class Recursive(torch.nn.Module):
            def forward(self, value):
                return self(value - 1) + 2 if value.item() else value

        module = Recursive()
        with capture_module(module, self.root / "recursive"):
            module(torch.tensor([2.0]))
        for call, value in enumerate((2, 1, 0)):
            directory = self.root / "recursive" / f"call-{call}"
            np.testing.assert_array_equal(np.load(directory / "input.npy"), [value])
            np.testing.assert_array_equal(
                np.load(directory / "output.npy"), [value * 2]
            )

    def test_overlapping_threads_keep_their_own_input_and_output(self):
        first_entered = threading.Event()
        second_entered = threading.Event()
        first_finished = threading.Event()

        class Overlapping(torch.nn.Module):
            def forward(self, value):
                if value.item() == 1:
                    first_entered.set()
                    second_entered.wait()
                else:
                    second_entered.set()
                    first_finished.wait()
                return value + 10

        module = Overlapping()

        def invoke(value):
            try:
                return module(torch.tensor([float(value)]))
            finally:
                if value == 1:
                    first_finished.set()

        with capture_module(module, self.root / "threads"):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(invoke, 1)
                first_entered.wait()
                second = executor.submit(invoke, 2)
                first.result()
                second.result()
        for call, value in enumerate((1, 2)):
            directory = self.root / "threads" / f"call-{call}"
            np.testing.assert_array_equal(np.load(directory / "input.npy"), [value])
            np.testing.assert_array_equal(
                np.load(directory / "output.npy"), [value + 10]
            )

    def test_caught_failures_do_not_publish_a_complete_capture(self):
        class Failing(torch.nn.Module):
            def forward(self, value):
                raise ValueError("forward failed")

        for name, module, select_output, error in (
            ("forward", Failing(), lambda value: value, ValueError),
            ("output", torch.nn.Identity(), lambda value: None, TypeError),
        ):
            with self.subTest(phase=name):
                with self.assertRaisesRegex(
                    RuntimeError, "incomplete module invocation"
                ):
                    with capture_module(
                        module, self.root / name, select_output=select_output
                    ):
                        with self.assertRaises(error):
                            module(torch.tensor([1.0]))
                self.assertFalse(module._forward_hooks)
                self.assertFalse(module._forward_pre_hooks)

    def test_unsupported_storage_is_not_silently_converted(self):
        value = torch.tensor([1.0, 2.0])
        for tensor in (
            value.to_sparse(),
            torch.quantize_per_tensor(value, 0.5, 0, torch.qint8),
        ):
            with self.assertRaisesRegex(ValueError, "explicit packed representation"):
                save_tensor(self.root / "unsupported.npy", tensor)
        self.assertFalse((self.root / "unsupported.npy").exists())

    def run_tool(self, tool, *arguments):
        result = subprocess.run(
            [str(tool), str(self.source), *arguments], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return result

    def prepare_replay(self):
        capture_example(self.root / "captures")
        self.source = self.root / "replay.loom"
        shutil.copyfile(SOURCE, self.source)

    def test_real_reference_capture_replays_through_loom(self):
        self.prepare_replay()
        result = self.run_tool(
            TOOL,
            "--case=@relu_capture",
            f"--device={DEVICE}",
            f"--file-output-root={self.root / 'outputs'}",
            "--execution-progress=true",
        )
        progress = [line for line in result.stderr.splitlines() if " count=" in line]
        self.assertTrue(progress, result.stderr)
        self.assertTrue(all(" count=1 " in line for line in progress))
        self.run_tool(TOOL, "--case=@replay_outputs")
        self.run_tool(TOOL, "--case=@bf16_carrier")

    def test_benchmark_capture_is_only_in_correctness_phase(self):
        self.prepare_replay()
        for correctness in (False, True):
            self.run_tool(
                BENCHMARK,
                "--benchmark=@captured_relu",
                f"--device={DEVICE}",
                f"--scenario-correctness={str(correctness).lower()}",
                "--iterations=1",
                "--warmup-iterations=0",
                "--min-time-ms=0",
                "--warmup-time-ms=0",
                "--max-batches=1",
                f"--file-output-root={self.root / 'outputs'}",
                f"--output={self.root / 'measurements.jsonl'}",
            )
            self.assertEqual((self.root / "outputs").exists(), correctness)
        self.run_tool(TOOL, "--case=@replay_outputs")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
