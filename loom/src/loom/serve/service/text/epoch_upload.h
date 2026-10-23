// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_EPOCH_UPLOAD_H_
#define LOOM_SERVE_SERVICE_TEXT_EPOCH_UPLOAD_H_

#include "loom/serve/runtime/execution/execution.h"

#ifdef __cplusplus
extern "C" {
#endif

// Source-encoded epoch packets. The source owns their layout and live ranges;
// the serialized invocation retains host storage through actual queue
// retirement.
enum {
  LOOM_SERVE_TEXT_UPLOAD_METADATA = 0,
  LOOM_SERVE_TEXT_UPLOAD_INPUT = 1,
  LOOM_SERVE_TEXT_UPLOAD_PACKET_COUNT = 2,
};

// Binds a source-authored transfer plan to the epoch's host source packets and
// device targets, then lowers its independent byte placements to HAL uploads.
// Both binding tables use the packet ordinals above; source and target offsets
// are independent. See runtime/program/transfer_plan.h for the encoded schema.
// The entire plan is validated before any upload. Bytes outside the ranges
// remain untouched; source initializes every byte its device consumers read.
//
// Descriptors are consumed before return without allocating or copying
// payloads. Host packets remain immutable until the execution frontier retires,
// including partial submission failure. The plan itself may be reused after
// return.
iree_status_t loom_serve_text_epoch_upload(
    loom_serve_execution_t* execution, iree_const_byte_span_t plan,
    const iree_const_byte_span_t packets[LOOM_SERVE_TEXT_UPLOAD_PACKET_COUNT],
    iree_hal_buffer_t* const targets[LOOM_SERVE_TEXT_UPLOAD_PACKET_COUNT]);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVICE_TEXT_EPOCH_UPLOAD_H_
