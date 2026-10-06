// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_IMAGE_MODEL_H_
#define EXPERIMENTAL_LOOM_SERVE_IMAGE_MODEL_H_

#include "experimental/loom_serve/runtime/device.h"

#ifdef __cplusplus
extern "C" {
#endif

// One source-defined diffusion image residency with serialized request storage.
// The application owns serialization; simultaneous generate calls are invalid.
typedef struct loom_serve_image_model_t loom_serve_image_model_t;

typedef struct loom_serve_image_model_options_t {
  // Source catalog directory, borrowed only during creation.
  iree_string_view_t source_directory;
  // Checkpoint asset directory, passed to the source bootstrap.
  iree_string_view_t checkpoint_directory;
  // Optional adapter asset, interpreted by the source bootstrap.
  iree_string_view_t adapter_path;
  // Fixed output height in pixels.
  uint32_t height;
  // Fixed output width in pixels.
  uint32_t width;
  // Text capacity requested from the source; its meaning is model-defined.
  uint32_t text_tokens;
} loom_serve_image_model_options_t;

// Source ABI:
// prepare.loom: prepare(i64 height, i64 width, i64 text_capacity,
//                      buffer checkpoint, buffer adapter)
//   -> (i64 request_capacity, buffer tokenizer_asset, buffer name, buffer
//   state)
// The source declares commands/configuration/parameter domains through prepare
// imports. tokenizer_asset is relative to checkpoint. Returned state is opaque
// and retained independently of the cold program; its VM environment stays
// live. control.loom:
//   prepare_request(buffer state, buffer i64_stage_tags, i32 stage_count,
//                   buffer prompt, i64 seed, f32 strength) -> (i32 stage,
//                                                             buffer payload)
//   generate(i32 stage, i64 output_bytes, hal.buffer payload, hal.buffer rgb,
//            hal.buffer workspace)
// The warm program imports input capabilities and runner.execute_N/feedback.
// Each command has shared reflected fixed parameter domains and three dynamic
// bindings: opaque request bytes, output F32 CHW RGB [-1,1], and workspace.
// All retained stages have identical parameter placement. Input capacity comes
// from source; workspace extent/alignment comes from compiler reflection.
// Creation indexes parameters and records commands without reading weight
// payloads. The device owner outlives the model and serializes all model calls.
// Its physical pool backs parameter roots independently of cached commands.
// First generation or explicit activation streams/prepares weights. Already
// active generation does no JIT or parameter preparation. Failure releases
// partial ownership and leaves *out_model NULL.
iree_status_t loom_serve_image_model_create(
    loom_serve_device_t* device,
    const loom_serve_image_model_options_t* options,
    loom_serve_image_model_t** out_model, iree_allocator_t host_allocator);

// Drains accepted work before releasing buffers, programs and device services.
// No generate call or borrowed RGB may remain active. NULL is accepted.
iree_status_t loom_serve_image_model_destroy(loom_serve_image_model_t* model);

// Reloads or releases every checkpoint domain while preserving compiled
// commands. Deactivation requires unpinned elastic backing. Next generation
// pins every domain together and activates on demand, reclaiming eligible idle
// weights of other models if needed. Execution/I/O failure is terminal.
iree_status_t loom_serve_image_model_activate(loom_serve_image_model_t* model);
iree_status_t loom_serve_image_model_deactivate(
    loom_serve_image_model_t* model);
// Borrowed group admission/retention handle; explicit pins may span requests.
loom_serve_residency_t* loom_serve_image_model_residency(
    const loom_serve_image_model_t* model);
loom_serve_memory_statistics_t loom_serve_image_model_weight_statistics(
    const loom_serve_image_model_t* model);

// Borrows the source-declared public identifier for this residency.
iree_string_view_t loom_serve_image_model_name(
    const loom_serve_image_model_t* model);

// Source prepares one opaque payload and selects a retained command. Its
// payload is kept alive through both accepted execution frontiers, including
// failures after partial submission. Request errors before submission leave
// the model reusable; execution failure is terminal. Prompt is borrowed only
// for this call. Success returns little-endian F32 CHW RGB, borrowed until the
// next generate call or destruction. Failure leaves *out_rgb empty.
iree_status_t loom_serve_image_model_generate(loom_serve_image_model_t* model,
                                              iree_string_view_t prompt,
                                              uint64_t seed, float strength,
                                              iree_const_byte_span_t* out_rgb);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // EXPERIMENTAL_LOOM_SERVE_IMAGE_MODEL_H_
