// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_SANITIZER_H_
#define LOOMC_SANITIZER_H_

#include "loomc/base.h"

/// @file
/// Sanitizer compile options.
///
/// Loom sanitizer options select semantic assertion classes. ASAN-like,
/// UBSAN-like, and TSAN-like are public preset names over one shared bitset;
/// compiler and target internals traffic in the concrete assertion classes.

#ifdef __cplusplus
extern "C" {
#endif

enum loomc_sanitizer_check_bit_e {
  /// Enables memory access assertions.
  LOOMC_SANITIZER_CHECK_ACCESS = 1u << 0,

  /// Enables SSA value assertions.
  LOOMC_SANITIZER_CHECK_VALUE = 1u << 1,

  /// Enables operation-level assertions.
  LOOMC_SANITIZER_CHECK_OPERATION = 1u << 2,

  /// Enables data-race observations.
  LOOMC_SANITIZER_CHECK_RACE = 1u << 3,
};
/// Bitset of `loomc_sanitizer_check_bit_e` values selecting assertion classes.
typedef uint64_t loomc_sanitizer_checks_t;

/// ASAN-like preset over Loom's unified sanitizer check bitset.
#define LOOMC_SANITIZER_CHECKS_ASAN_LIKE \
  ((loomc_sanitizer_checks_t)LOOMC_SANITIZER_CHECK_ACCESS)

/// UBSAN-like preset over Loom's unified sanitizer check bitset.
#define LOOMC_SANITIZER_CHECKS_UBSAN_LIKE                   \
  ((loomc_sanitizer_checks_t)(LOOMC_SANITIZER_CHECK_VALUE | \
                              LOOMC_SANITIZER_CHECK_OPERATION))

/// TSAN-like preset over Loom's unified sanitizer check bitset.
#define LOOMC_SANITIZER_CHECKS_TSAN_LIKE \
  ((loomc_sanitizer_checks_t)LOOMC_SANITIZER_CHECK_RACE)

enum loomc_sanitizer_flag_bit_e {
  /// No sanitizer flags are currently defined.
  LOOMC_SANITIZER_FLAG_NONE = 0u,
};
/// Bitset of `loomc_sanitizer_flag_bit_e` values controlling sanitizer
/// behavior.
typedef uint32_t loomc_sanitizer_flags_t;

/// Target-specific behavior when a lowered sanitizer assertion fails.
typedef enum loomc_sanitizer_reporting_mode_e {
  /// Use the target's default structured sanitizer report path.
  LOOMC_SANITIZER_REPORTING_MODE_DEFAULT = 0,

  /// Trap directly on sanitizer failures without emitting structured reports.
  LOOMC_SANITIZER_REPORTING_MODE_TRAP = 1,

  /// Emit structured sanitizer reports without using a direct trap.
  LOOMC_SANITIZER_REPORTING_MODE_REPORT_ONLY = 2,
} loomc_sanitizer_reporting_mode_t;

/// Invocation option extension that enables sanitizer assertions.
///
/// Attach this descriptor to `loomc_target_pipeline_options_t::next` when the
/// target pipeline should contain sanitizer pass slots. The target pipeline
/// treats `checks == 0` as disabled and emits no sanitizer pass IR. The
/// reporting mode is carried to the target assertion lowering path.
typedef struct loomc_sanitizer_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS`.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Next invocation option extension.
  const void* next;

  /// Assertion classes enabled by the caller.
  loomc_sanitizer_checks_t checks;

  /// Additional behavior flags enabled by the caller.
  loomc_sanitizer_flags_t flags;

  /// Target failure reporting behavior for lowered sanitizer assertions.
  loomc_sanitizer_reporting_mode_t reporting_mode;
} loomc_sanitizer_options_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_SANITIZER_H_
