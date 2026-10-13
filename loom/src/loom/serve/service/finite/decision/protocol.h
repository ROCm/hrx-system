// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_FINITE_DECISION_PROTOCOL_H_
#define LOOM_SERVE_SERVICE_FINITE_DECISION_PROTOCOL_H_

#include "loom/serve/service/finite/protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

// Static /v1/systemone modality contract: choice, score and noul questions.
// Each source result row is a normalized LE F32 probability vector in public
// option order. Scores use ascending levels; noul uses [false, true]. The
// encoder computes only protocol answer/confidence statistics, never softmax.
// Equal maxima select the first public ordinal. Output token usage is zero.
//
// Source input contains LE i64 records followed by original validated JSON:
// header [state_offset, state_length, state_type, question_count];
// question [id_offset, id_length, kind, instructions_offset,
//           instructions_length, instructions_type, first_option,
//           option_count];
// option [id_offset, id_length, criterion_offset, criterion_length, type].
// All ranges are absolute; strings exclude quotes and retain JSON escapes.
// Types match runtime/program/json.h. Question kinds are choice=0, score=1,
// noul=2. Implicit score/noul option IDs have zero length. Absent noul criteria
// have zero length and null type. Source owns canonical formatting and
// truncation.
const loom_serve_finite_protocol_t* loom_serve_decision_protocol(void);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVICE_FINITE_DECISION_PROTOCOL_H_
