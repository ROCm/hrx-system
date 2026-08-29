// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/vm/artifact_emitter.h"

#include "loom/target/emit/vm/module_emitter.h"

static iree_status_t loom_vm_bytecode_artifact_emit(
    const loom_target_emit_request_t* request,
    loom_target_emit_artifact_t* out_artifact) {
  *out_artifact = (loom_target_emit_artifact_t){0};
  if (request->artifact_manifest.mode !=
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "VM bytecode emission does not produce artifact manifests");
  }

  loom_target_emit_export_projection_t* export_projections = NULL;
  const iree_host_size_t export_projection_capacity =
      request->function_versions != NULL ? request->function_versions->count
                                         : 0;
  if (export_projection_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        request->scratch_arena, export_projection_capacity,
        sizeof(*export_projections), (void**)&export_projections));
  }
  loom_target_emit_export_projection_buffer_t export_projection = {
      .values = export_projections,
      .capacity = export_projection_capacity,
  };
  const loom_vm_module_emitter_options_t emission_options = {
      .descriptor_registry = request->low_descriptor_registry,
      .diagnostic_emitter = request->diagnostic_emitter,
      .function_versions = request->function_versions,
      .export_projection = &export_projection,
  };
  iree_byte_sequence_t* contents = NULL;
  iree_status_t status = loom_vm_emit_module(request->module, &emission_options,
                                             request->scratch_arena,
                                             request->allocator, &contents);
  if (iree_status_is_ok(status) && contents != NULL) {
    out_artifact->target_artifact_format =
        LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE;
    out_artifact->contents = contents;
    out_artifact->export_projections = export_projection.values;
    out_artifact->export_projection_count = export_projection.count;
    contents = NULL;
  }
  iree_byte_sequence_release(contents);
  return status;
}

static const loom_target_emitter_t loom_vm_bytecode_artifact_emitter = {
    .name = IREE_SVL("vm"),
    .public_artifact_format = IREE_SVL("vm"),
    .default_identifier = IREE_SVL("module.vm"),
    .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_VM_BYTECODE,
    .emit = loom_vm_bytecode_artifact_emit,
};

static const loom_target_emitter_t* const kLoomVmArtifactEmitters[] = {
    &loom_vm_bytecode_artifact_emitter,
};

const loom_target_provider_t loom_vm_artifact_emitter_provider = {
    .emitter_list =
        {
            .values = kLoomVmArtifactEmitters,
            .count = IREE_ARRAYSIZE(kLoomVmArtifactEmitters),
        },
};
