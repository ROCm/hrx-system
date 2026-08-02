// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_QWEN_TOOLING_ROUTE_TRACE_FILE_H_
#define EXPERIMENTAL_QWEN_TOOLING_ROUTE_TRACE_FILE_H_

#include "experimental/qwen/runtime/model_shape.h"
#include "iree/base/api.h"
#include "iree/io/parameter_index.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Current version of the little-endian Qwen route-trace file format.
#define QWEN_ROUTE_TRACE_FILE_VERSION 1u

// Fixed model schema represented by a route trace.
typedef enum qwen_route_trace_model_e {
  // Qwen3-30B-A3B with the exact geometry accepted by the Qwen model loader.
  QWEN_ROUTE_TRACE_MODEL_QWEN3_30B_A3B = 1u,
} qwen_route_trace_model_t;

// Metadata serialized before the route planes.
typedef struct qwen_route_trace_file_metadata_t {
  // Model schema that produced the trace.
  qwen_route_trace_model_t model;
  // Number of physical token rows allocated in each route plane.
  iree_host_size_t context_capacity;
  // Number of logical rows written by completed model issues.
  iree_host_size_t captured_token_count;
  // Number of token rows in the initial prefill.
  iree_host_size_t prompt_token_count;
  // Number of tokens selected by the model, including a terminal EOS.
  iree_host_size_t generated_token_count;
  // Number of entries in the source parameter index.
  iree_host_size_t parameter_count;
  // Stable hash of source parameter keys, storage kinds, ranges, and metadata.
  uint64_t parameter_layout_fingerprint;
  // Total encoded parameter bytes resident in the model allocation.
  uint64_t encoded_parameter_bytes;
} qwen_route_trace_file_metadata_t;

// Borrowed view into a validated serialized route-trace file.
typedef struct qwen_route_trace_file_view_t {
  // Metadata decoded from the file header.
  qwen_route_trace_file_metadata_t metadata;
  // Context-major [token][layer][route] little-endian I32 expert IDs.
  iree_const_byte_span_t route_ids;
  // Context-major [token][layer][route] little-endian F32 routing weights.
  iree_const_byte_span_t route_weights;
} qwen_route_trace_file_view_t;

// Computes a stable layout fingerprint without reading parameter payloads.
//
// The fingerprint identifies the indexed tensor names, lengths, source ranges,
// storage kinds, and metadata. It intentionally does not hash multi-gigabyte
// weight contents and must not be presented as a cryptographic checkpoint hash.
IREE_API_EXPORT iree_status_t qwen_route_trace_parameter_layout_fingerprint(
    iree_io_parameter_index_t* parameter_index, uint64_t* out_fingerprint);

// Builds one complete serialized route-trace file in host memory.
//
// |route_payload| is the exact payload returned by
// qwen_request_read_route_trace: the full ID plane followed by the full weight
// plane. On success, the caller owns |out_file_data->data| and must free it
// with |host_allocator|.
IREE_API_EXPORT iree_status_t qwen_route_trace_file_build(
    const qwen_route_trace_file_metadata_t* metadata,
    iree_const_byte_span_t route_payload, iree_allocator_t host_allocator,
    iree_byte_span_t* out_file_data);

// Parses and validates one complete route-trace file.
//
// The returned spans borrow |file_data| and remain valid for the same lifetime.
IREE_API_EXPORT iree_status_t qwen_route_trace_file_parse(
    iree_const_byte_span_t file_data, qwen_route_trace_file_view_t* out_view);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_TOOLING_ROUTE_TRACE_FILE_H_
