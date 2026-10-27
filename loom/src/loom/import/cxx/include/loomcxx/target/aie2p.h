// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_TARGET_AIE2P_H_
#define LOOMCXX_TARGET_AIE2P_H_

#include <loomcxx/target.h>

namespace loom::aie2p {

// Source-owned aie2p.target definition. The kind selects a core, logical
// array, or physical configuration program. A configuration target may name
// the exact physical XDNA deployment profile it initializes.
struct [[loom::target("aie2p.target")]] target {
  // Registered AIE2P target-row keyword: "core", "array", or
  // "configuration".
  const char* kind;
  // Exact physical XDNA deployment profile key when explicitly selected.
  loom::target::optional<const char*> device_profile;
};

}  // namespace loom::aie2p

#endif  // LOOMCXX_TARGET_AIE2P_H_
