// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable x86 target facts.

#ifndef LOOM_TARGET_ARCH_X86_FACTS_H_
#define LOOM_TARGET_ARCH_X86_FACTS_H_

#include "iree/base/cpu_data.h"
#include "loom/target/arch/x86/feature_bits.h"
#include "loom/target/arch/x86/records/target_records.h"
#include "loom/target/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

// Set of executable x86 target kinds. Target kind zero is reserved, so each
// generated nonzero kind maps directly to one bit.
typedef uint32_t loom_x86_target_kind_set_t;

#define LOOM_X86_TARGET_KIND_BIT(target_kind) \
  ((loom_x86_target_kind_set_t)(UINT32_C(1) << (target_kind)))

static inline bool loom_x86_target_kind_set_contains(
    loom_x86_target_kind_set_t set, uint8_t target_kind) {
  return iree_any_bit_set(set, LOOM_X86_TARGET_KIND_BIT(target_kind));
}

typedef struct loom_x86_target_facts_t {
  // Target-neutral facts shared by all target families.
  loom_target_facts_t base;

  // Complete execution-device CPU facts, or UNKNOWN for authored profiles.
  iree_cpu_data_t cpu_data;

  // Optional instruction features explicitly enabled by the source facts.
  loom_x86_feature_bits_t positive_feature_assertions;

  // Optional instruction features explicitly disabled by the source facts.
  loom_x86_feature_bits_t negative_feature_assertions;

  // Processor scheduling model selected by authored IR or exact CPU identity.
  loom_x86_target_tune_t tune;

  // Whether |tune| is an authored specialization requirement.
  bool tune_is_explicit;

  // Executable target rows supported by the retained CPU facts. The set always
  // contains |base.selector| and is a singleton once execution is selected.
  loom_x86_target_kind_set_t executable_target_kinds;
} loom_x86_target_facts_t;

// Static fact type used by x86 target projection and structured profiles.
extern const loom_target_fact_type_t loom_x86_target_fact_type;

// Returns |facts| as x86 facts, or NULL for another target family.
static inline const loom_x86_target_facts_t* loom_x86_target_facts_cast(
    const loom_target_facts_t* facts) {
  return facts != NULL && facts->fact_type == &loom_x86_target_fact_type
             ? (const loom_x86_target_facts_t*)facts
             : NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_FACTS_H_
