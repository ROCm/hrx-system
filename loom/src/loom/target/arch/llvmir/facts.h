// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable LLVMIR target facts.

#ifndef LOOM_TARGET_ARCH_LLVMIR_FACTS_H_
#define LOOM_TARGET_ARCH_LLVMIR_FACTS_H_

#include "loom/target/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_llvmir_target_facts_t {
  // Target-neutral facts shared by all target families.
  loom_target_facts_t base;

  // LLVM target triple emitted into modules and passed to LLVM tools.
  iree_string_view_t target_triple;

  // LLVM data layout emitted into modules when authored.
  iree_string_view_t data_layout;

  // LLVM target CPU passed to LLVM tools when authored.
  iree_string_view_t target_cpu;

  // LLVM target feature string passed to LLVM tools when authored.
  iree_string_view_t target_features;

  // Authorship of LLVMIR-family facts projected from the target witness.
  struct {
    // True when the target triple was explicitly authored.
    bool target_triple;
    // True when the data layout was explicitly authored.
    bool data_layout;
    // True when the target CPU was explicitly authored.
    bool target_cpu;
    // True when the target feature string was explicitly authored.
    bool target_features;
  } authored;
} loom_llvmir_target_facts_t;

// Static fact type used by LLVMIR target projection and structured profiles.
extern const loom_target_fact_type_t loom_llvmir_target_fact_type;

// Returns |facts| as LLVMIR facts, or NULL for another target family.
static inline const loom_llvmir_target_facts_t* loom_llvmir_target_facts_cast(
    const loom_target_facts_t* facts) {
  return facts != NULL && facts->fact_type == &loom_llvmir_target_fact_type
             ? (const loom_llvmir_target_facts_t*)facts
             : NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_LLVMIR_FACTS_H_
