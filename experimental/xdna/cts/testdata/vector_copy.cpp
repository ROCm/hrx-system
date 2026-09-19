// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/kernel.h>

typedef unsigned U16 __attribute__((vector_size(64)));

[[loom::force_inline]] void copy_blocks(const unsigned* input,
                                        unsigned* output) {
  unsigned count = input[0];
  unsigned source_stride = input[1];
  unsigned destination_stride = input[2];
  unsigned destination_origin = input[3];
  loom::assume(count < 21u);
  loom::assume(source_stride < 4u);
  loom::assume(destination_stride < 3u);
  loom::assume(destination_origin < 2u);
  const auto* source = reinterpret_cast<const U16*>(input + 16u);
  auto* destination = reinterpret_cast<U16*>(output);
  const U16 guard = {0x6badcafeu, 0x6badcafeu, 0x6badcafeu, 0x6badcafeu,
                     0x6badcafeu, 0x6badcafeu, 0x6badcafeu, 0x6badcafeu,
                     0x6badcafeu, 0x6badcafeu, 0x6badcafeu, 0x6badcafeu,
                     0x6badcafeu, 0x6badcafeu, 0x6badcafeu, 0x6badcafeu};
  for (unsigned block = 0; block < 64u; ++block) {
    destination[block] = guard;
  }
  for (unsigned block = 0; block < count; ++block) {
    loom::assume(block < 20u);
#if SCALAR_COPY
    for (unsigned lane = 0; lane < 16u; ++lane) {
      output[(block * destination_stride + destination_origin) * 16u + lane] =
          input[16u + block * source_stride * 16u + lane];
    }
#else
    destination[block * destination_stride + destination_origin] =
        source[block * source_stride];
#endif
  }
}
