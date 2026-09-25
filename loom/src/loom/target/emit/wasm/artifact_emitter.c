// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/artifact_emitter.h"

#include "loom/target/emit/wasm/module_binary.h"

static iree_status_t loom_wasm_artifact_emit(
    const loom_target_emit_request_t* request, bool* out_emitted,
    loom_target_emit_artifact_t* out_artifact) {
  *out_emitted = false;
  *out_artifact = (loom_target_emit_artifact_t){0};
  if (request->artifact_manifest.mode !=
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Wasm module artifacts do not produce artifact manifests");
  }

  loom_wasm_module_binary_t module = {0};
  bool module_emitted = false;
  IREE_RETURN_IF_ERROR(loom_wasm_emit_low_module(
      request->module, request->low_descriptor_registry,
      request->diagnostic_emitter, request->scratch_arena, request->allocator,
      &module_emitted, &module));
  if (!module_emitted) {
    return iree_ok_status();
  }
  iree_byte_span_t contents =
      iree_make_byte_span(module.data, module.data_length);
  iree_status_t status = iree_byte_sequence_create_from_span_move(
      &contents, request->allocator, &out_artifact->contents);
  if (iree_status_is_ok(status)) {
    out_artifact->target_artifact_format =
        LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY;
    *out_emitted = true;
  }
  iree_allocator_free(request->allocator, contents.data);
  return status;
}

static const loom_target_emitter_t loom_wasm_artifact_emitter = {
    .name = IREE_SVL("wasm-binary"),
    .public_artifact_format = IREE_SVL("wasm-binary"),
    .default_identifier = IREE_SVL("module.wasm"),
    .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY,
    .default_pipeline_options =
        {
            .control_flow_lowering =
                LOOM_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW,
        },
    .emit = loom_wasm_artifact_emit,
};

static const loom_target_emitter_t* const kLoomWasmArtifactEmitters[] = {
    &loom_wasm_artifact_emitter,
};

const loom_target_provider_t loom_wasm_artifact_emitter_provider = {
    .emitter_list =
        {
            .values = kLoomWasmArtifactEmitters,
            .count = IREE_ARRAYSIZE(kLoomWasmArtifactEmitters),
        },
};
