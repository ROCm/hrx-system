// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_PARAMETER_ACTIVITY_H_
#define LOOM_RUNTIME_PARAMETER_ACTIVITY_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cumulative owner-side submissions, independent of telemetry retention. These
// are accepted producer operations, not measured storage traffic or device
// time.
typedef struct loom_runtime_parameter_activity_t {
  // Fully submitted loads, excluding readiness imports by later consumers.
  uint64_t activations;
  // Active-to-inactive releases of physical backing.
  uint64_t deactivations;
  // Payload bytes accepted by cold read tasks, including later I/O failures.
  uint64_t read_bytes;
  // Final tensor bytes whose preparation dispatch was accepted.
  uint64_t prepared_bytes;
  // Gather groups accepted by cold read tasks across all activations.
  uint64_t groups;
} loom_runtime_parameter_activity_t;

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_RUNTIME_PARAMETER_ACTIVITY_H_
