// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_FINITE_PROTOCOL_H_
#define LOOM_SERVE_SERVICE_FINITE_PROTOCOL_H_

#include "iree/base/api.h"
#include "iree/vm/buffer.h"

// Initial member of each modality's request. The source packet and protocol
// metadata survive until CPU encoding releases the cohort, including reset
// peers.
typedef struct loom_serve_finite_request_t {
  // Owned read-only source packet, with modality-defined indexed records.
  iree_vm_buffer_t* input;
  // Expected result rows in this HTTP request's original order.
  iree_host_size_t count;
} loom_serve_finite_request_t;

// Native modality wiring for the shared finite cohort service, not a model
// plugin interface. Source programs still own preparation and learned math.
typedef struct loom_serve_finite_protocol_t {
  // Static modality name used in diagnostics and event names.
  const char* name;
  // Static POST endpoint handled by this modality.
  const char* endpoint;
  // Client-visible output modality with static storage.
  iree_string_view_t output_modality;
  // Static HTTP status for malformed schema or source input rejection.
  const char* invalid_status;
  // Complete concrete request size, including its initial finite request
  // member.
  iree_host_size_t request_size;
  // Initializes zeroed concrete storage; failure leaves it deinitializable.
  iree_status_t (*initialize)(iree_string_view_t body,
                              loom_serve_finite_request_t* request,
                              iree_allocator_t allocator);
  // Releases all concrete request storage, including its source input buffer.
  void (*deinitialize)(loom_serve_finite_request_t* request);
  // Encodes trusted row spans from the checked source plan in client order.
  // Semantic result failures propagate; no native neural postprocessing occurs.
  iree_status_t (*response)(const loom_serve_finite_request_t* request,
                            iree_string_view_t model,
                            const iree_const_byte_span_t* rows,
                            uint64_t input_tokens,
                            iree_string_builder_t* output);
} loom_serve_finite_protocol_t;

#endif  // LOOM_SERVE_SERVICE_FINITE_PROTOCOL_H_
