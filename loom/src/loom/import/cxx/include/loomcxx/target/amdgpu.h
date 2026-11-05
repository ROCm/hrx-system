// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_TARGET_AMDGPU_H_
#define LOOMCXX_TARGET_AMDGPU_H_

#include <loomcxx/target.h>

namespace loom::amdgpu {

// Explicit AMDHSA target-ID feature assertions. Unspecified features inherit
// the selected processor row's requirements.
struct amdhsa_features {
  // SRAM ECC feature polarity.
  loom::target::feature sramecc = loom::target::feature::unspecified;
  // XNACK feature polarity.
  loom::target::feature xnack = loom::target::feature::unspecified;
};

// Source-owned amdgpu.target definition. The kind names an exact or generic
// processor row; optional fields retain explicit author constraints.
struct [[loom::target("amdgpu.target")]] target {
  // Registered AMDGPU processor-row keyword, such as "gfx942".
  const char* kind;
  // Required execution subgroup width when explicitly set.
  loom::target::optional<unsigned long long> subgroup_size;
  // Explicit AMDHSA target-ID feature assertions.
  amdhsa_features features{};
};

}  // namespace loom::amdgpu

#endif  // LOOMCXX_TARGET_AMDGPU_H_
