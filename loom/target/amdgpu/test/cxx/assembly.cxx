// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/kernel.h>
#include <loomcxx/low.h>

struct [[loom::representation("amdgpu.gfx11.generic.core")]] Contract {};
using Word = unsigned __attribute__((ext_vector_type(1)));
using Float = float __attribute__((ext_vector_type(1)));

template <class T>
static T pack(T even, T odd) {
  return loom::low::assembly<Contract, T>(R"loom(
      (%even: reg<amdgpu.vgpr>, %odd: reg<amdgpu.vgpr>) -> (reg<amdgpu.vgpr>) {
        %selector = s_mov_b32 0x05040100
        %packed = v_perm_b32 %odd, %even, %selector
        return %packed
      }
  )loom",
                                          even, odd);
}

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void pack_words(const unsigned* input, unsigned* output) {
  unsigned lane = loom::kernel::workitem::id.x;
  Word even = {input[2 * lane + 1]};
  Word odd = {input[2 * lane + 2]};
  output[lane + 1] = pack(even, odd)[0];
}

[[loom::kernel, loom::workgroup_size(32, 1, 1), loom::workgroup_count(1, 1, 1)]]
void pack_floats(const unsigned* input, unsigned* output) {
  unsigned lane = loom::kernel::workitem::id.x;
  Word even = {input[2 * lane + 1]};
  Word odd = {input[2 * lane + 2]};
  Float result =
      pack(__builtin_bit_cast(Float, even), __builtin_bit_cast(Float, odd));
  Word bits = __builtin_bit_cast(Word, result);
  output[lane + 1] = bits[0];
}
