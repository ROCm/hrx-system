// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor-set readiness checks for target-low consumers.
//
// Requirement verification proves that a descriptor set carries the semantic
// payloads a specific compiler phase family consumes. The descriptor tables
// themselves are target-owned static data; requirement checks are explicit
// target-package audits, not production preflight scans.

#ifndef LOOM_CODEGEN_LOW_REQUIREMENTS_H_
#define LOOM_CODEGEN_LOW_REQUIREMENTS_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

// Bitset of descriptor-set payload requirements.
typedef uint32_t loom_low_descriptor_requirement_flags_t;

// Descriptor set must expose at least one descriptor and the core tables used
// by target-low consumers.
#define LOOM_LOW_DESCRIPTOR_REQUIREMENT_CORE_TABLES ((uint32_t)1u << 0)

// Every descriptor must carry a target mnemonic or packet spelling.
#define LOOM_LOW_DESCRIPTOR_REQUIREMENT_MNEMONICS ((uint32_t)1u << 1)

// Every descriptor must carry a primary semantic tag.
#define LOOM_LOW_DESCRIPTOR_REQUIREMENT_SEMANTIC_TAGS ((uint32_t)1u << 2)

// Every operand/result row must name at least one register-class alternative.
#define LOOM_LOW_DESCRIPTOR_REQUIREMENT_OPERAND_REG_CLASSES ((uint32_t)1u << 3)

// Referenced schedule classes with non-zero cost or effects must consume at
// least one target resource.
#define LOOM_LOW_DESCRIPTOR_REQUIREMENT_ISSUE_RESOURCES ((uint32_t)1u << 4)

// Common foundation required by target-low lowering, scheduling, allocation
// diagnostics, and emission.
#define LOOM_LOW_DESCRIPTOR_REQUIREMENT_TARGET_LOW_FOUNDATION \
  (LOOM_LOW_DESCRIPTOR_REQUIREMENT_CORE_TABLES |              \
   LOOM_LOW_DESCRIPTOR_REQUIREMENT_MNEMONICS |                \
   LOOM_LOW_DESCRIPTOR_REQUIREMENT_SEMANTIC_TAGS |            \
   LOOM_LOW_DESCRIPTOR_REQUIREMENT_OPERAND_REG_CLASSES |      \
   LOOM_LOW_DESCRIPTOR_REQUIREMENT_ISSUE_RESOURCES)

// Verifies that |descriptor_set| satisfies the requested target-low payload
// requirements. The structural verifier is run first so failures report the
// malformed base table before consumer-specific readiness checks run.
iree_status_t loom_low_descriptor_set_verify_requirements(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_low_descriptor_requirement_flags_t requirements);

// Verifies that every descriptor set in |registry| satisfies |requirements|.
iree_status_t loom_low_descriptor_registry_verify_requirements(
    const loom_low_descriptor_registry_t* registry,
    loom_low_descriptor_requirement_flags_t requirements);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_CODEGEN_LOW_REQUIREMENTS_H_
