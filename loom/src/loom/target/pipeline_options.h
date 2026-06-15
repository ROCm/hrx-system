// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target lowering pass-pipeline options shared by compile front doors and
// target-linked tooling adapters.

#ifndef LOOM_TARGET_PIPELINE_OPTIONS_H_
#define LOOM_TARGET_PIPELINE_OPTIONS_H_

#include <stdint.h>

#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_target_control_flow_lowering_e {
  // Lower source SCF to explicit CFG before source-to-low.
  LOOM_TARGET_CONTROL_FLOW_LOWERING_CFG = 0,
  // Preserve supported source SCF through source-to-low as executable low SCF.
  LOOM_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW = 1,
} loom_target_control_flow_lowering_t;

typedef struct loom_target_pipeline_options_t {
  // Maximum source-to-low diagnostics. Zero uses source-to-low's default.
  uint32_t source_to_low_max_errors;
  // Source-to-low legality diagnostics to emit while selecting target-low.
  loom_target_low_legality_diagnostic_flags_t
      source_to_low_legality_diagnostic_flags;
  // Control-flow shape selected for the source-to-low boundary.
  loom_target_control_flow_lowering_t control_flow_lowering;
} loom_target_pipeline_options_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_PIPELINE_OPTIONS_H_
