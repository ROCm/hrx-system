// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// One reusable source library spanning packed quantization, semantic matrix
// fragments, and configurable asynchronous staging. Embedders can import this
// translation unit once and specialize individual products from its loombc.

#include "iq4xs_gate_up.cxx"
#include "nvfp4_matrix.cxx"
#include "nvfp4_matrix_providers.cxx"
#include "q4k_q8_swiglu.cxx"
