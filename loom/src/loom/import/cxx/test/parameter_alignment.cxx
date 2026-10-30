// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

using Words = unsigned __attribute__((vector_size(64)));

[[loom::kernel, loom::workgroup_size(1, 1, 1), loom::workgroup_count(1, 1, 1)]]
void copy_block([[loom::assume_aligned(64)]] const unsigned* input,
                [[loom::assume_aligned(64)]] unsigned* output) {
  *reinterpret_cast<Words*>(output) =
      *reinterpret_cast<const Words*>(input + 16);
}
