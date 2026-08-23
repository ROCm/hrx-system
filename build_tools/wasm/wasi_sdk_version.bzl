# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Pinned wasi-sdk release and host archive digests."""

# wasi-sdk 30 includes LLVM 21.1.4, lld, wasi-libc, libc++, and compiler-rt.
WASI_SDK_VERSION = "30.0"
WASI_SDK_TAG = "wasi-sdk-30"
WASI_SDK_URL_TEMPLATE = "https://github.com/WebAssembly/wasi-sdk/releases/download/{tag}/wasi-sdk-{version}-{arch}-{os}.tar.gz"

WASI_SDK_SHA256 = {
    "arm64-linux": "6f2977942308d91b0123978da3c6a0d6fce780994b3b020008c617e26764ea40",
    "arm64-macos": "2c2ed99296857e60fd14c3f40fe226231f296409502491094704089c31a16740",
    "arm64-windows": "b9552f207ea4287616dbf7c40bc0fbd5a9271ba6f8333fa606b63636f75060c2",
    "x86_64-linux": "0507679dff16814b74516cd969a9b16d2ced1347388024bc7966264648c78bfb",
    "x86_64-macos": "1594a0791309781bf0d0224431c3556ec4a2326b205687b659f6550d08d8b13e",
    "x86_64-windows": "e87d6bf9f9ca3482a75f1cbc630f095b4ae8c98d586708bac7adf08c03b327bc",
}
