// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/low.h>

struct [[loom::representation("amd.xdna.aie2p.core")]] Aie2p {};
using SignedBytes = signed char __attribute__((ext_vector_type(128)));
using Packed = unsigned char __attribute__((ext_vector_type(64)));

[[loom::force_inline]] Packed saturate_s4(SignedBytes values) {
  return loom::low::assembly<Aie2p, Packed>(R"loom(
      (%values: reg<aie2p.vec256 x4>) -> (reg<aie2p.vec256 x2>) {
        set.pack-size 0
        set.saturation 1
        %packed = vpack.x.signed %values
        return %packed
      }
  )loom",
                                            values);
}

[[loom::force_inline]] Packed truncate_s4(SignedBytes values) {
  return loom::low::assembly<Aie2p, Packed>(R"loom(
      (%values: reg<aie2p.vec256 x4>) -> (reg<aie2p.vec256 x2>) {
        set.pack-size 0
        set.saturation 0
        %packed = vpack.x.signed %values
        return %packed
      }
  )loom",
                                            values);
}

[[loom::force_inline]] void pack_blocks(const signed char* input,
                                        unsigned char* output) {
  const auto* source = reinterpret_cast<const SignedBytes*>(input);
  auto* destination = reinterpret_cast<Packed*>(output);
  const Packed guard = {
      0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5,
      0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5,
      0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5,
      0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5,
      0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5,
      0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5,
  };
  destination[0] = guard;
  destination[5] = guard;
  for (unsigned block = 0; block < 2; ++block) {
    SignedBytes values = source[block];
    destination[1 + block] = saturate_s4(values);
    destination[3 + block] = truncate_s4(values);
  }
}
