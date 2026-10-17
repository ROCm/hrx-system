// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_RECOGNITION_REQUEST_H_
#define LOOM_SERVE_SERVICE_RECOGNITION_REQUEST_H_

#include "loom/serve/service/recognition/events.h"

// Immutable source options, borrowed through asynchronous request preparation.
// Audio arrives separately through bounded sample-frame snapshots.
typedef struct loom_serve_recognition_request_t {
  // Optional UTF-8 language selector; empty requests source language policy.
  iree_string_view_t language;
  // Optional UTF-8 recognition context, interpreted by the source model.
  iree_string_view_t prompt;
  // Source decoding temperature; zero requests deterministic decoding.
  float temperature;
  // Nonempty subset of the admitted source's supported output types.
  loom_serve_recognition_output_flags_t outputs;
} loom_serve_recognition_request_t;

#endif  // LOOM_SERVE_SERVICE_RECOGNITION_REQUEST_H_
