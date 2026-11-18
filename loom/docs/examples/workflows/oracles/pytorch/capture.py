# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Save selected eager-model observations without keeping a tensor history."""

import argparse
import hashlib
import itertools
import json
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


# --8<-- [start:save_tensor]
def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    """Save one dense logical value and its original physical description."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("select a tensor from the module's inputs or outputs")
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise ValueError("capture the explicit packed representation for this layout")
    host = tensor.detach().resolve_conj().resolve_neg().to(device="cpu").contiguous()
    # NPY has no standard BF16 or float8 descriptor. Signed integer carriers
    # preserve the bits and match Loom's i16/i8 fixture types.
    if host.dtype == torch.bfloat16:
        host = host.view(torch.int16)
    elif str(host.dtype).startswith("torch.float8_"):
        host = host.view(torch.int8)
    array = host.numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride_elements": list(tensor.stride()),
        "storage_offset_elements": tensor.storage_offset(),
        "device": str(tensor.device),
        "npy_dtype": array.dtype.str,
        "sha256": digest,
    }
    with path.with_suffix(".json").open("x") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")


# --8<-- [end:save_tensor]


# --8<-- [start:hooks]
@contextmanager
def capture_module(
    module,
    directory,
    *,
    select_input=lambda args, kwargs: args[0],
    select_output=lambda output: output,
):
    """Capture a selected input/output tensor for each successful eager call."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    calls = itertools.count()
    pending = {}  # Per-thread call identities; no retained tensor payloads.
    lock = threading.Lock()

    def before(module, args, kwargs):
        with lock:
            call = directory / f"call-{next(calls)}"
            pending.setdefault(threading.get_ident(), []).append(call)
        save_tensor(call / "input.npy", select_input(args, kwargs))

    def after(module, args, kwargs, output):
        with lock:
            call = pending[threading.get_ident()][-1]
        save_tensor(call / "output.npy", select_output(output))
        with lock:
            pending[threading.get_ident()].pop()

    handles = []
    try:
        handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(module.register_forward_hook(after, with_kwargs=True))
        yield
        if any(pending.values()):
            raise RuntimeError("capture contains an incomplete module invocation")
    finally:
        for handle in handles:
            handle.remove()


# --8<-- [end:hooks]


def capture_example(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    # An in-place module makes input timing observable: the pre-hook must finish
    # writing before forward mutates the input. Transpose also tests layout.
    module = torch.nn.ReLU(inplace=True).eval()
    with torch.inference_mode():
        for name, dtype in (("relu", torch.float32), ("bf16", torch.bfloat16)):
            with capture_module(module, root / name):
                for start in (-4, -2):
                    value = (
                        torch.arange(start, start + 8, dtype=dtype).reshape(4, 2).t()
                    )
                    module(value)
    # Publication of this small record means the complete example succeeded.
    # A real port also records checkpoint, source, input, and tokenizer identity.
    (root / "run.json").write_text(
        json.dumps(
            {"torch": torch.__version__, "numpy": np.__version__, "example": "ReLU"},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for this capture run")
    capture_example(parser.parse_args().output)
