// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVICE_TEXT_SCHEDULE_H_
#define LOOM_SERVICE_TEXT_SCHEDULE_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_service_text_schedule_mode_e {
  // One shared weight traversal for all admitted prompt and decode spans.
  LOOM_SERVICE_TEXT_SCHEDULE_PACKED = 0,
  // Same ready-span partition, executed with ordinary isolated model stages.
  LOOM_SERVICE_TEXT_SCHEDULE_ISOLATED,
  // Isolated prefill math even for decode, for a same-math packed control.
  LOOM_SERVICE_TEXT_SCHEDULE_MATCHED,
} loom_service_text_schedule_mode_t;

typedef enum loom_service_text_packing_mode_e {
  // Fill one epoch with all ready prompt and decode spans.
  LOOM_SERVICE_TEXT_PACKING_MIXED = 0,
  // Pack only the first ready row's phase, using the same rotating priority.
  LOOM_SERVICE_TEXT_PACKING_SEPARATE,
} loom_service_text_packing_mode_t;

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVICE_TEXT_SCHEDULE_H_
