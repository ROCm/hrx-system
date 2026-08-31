// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/vm/launch_config_compiler.h"

#include "loom/pass/builder.h"
#include "loom/target/arch/vm/launch_config_program.h"
#include "loom/target/emit/vm/module_emitter.h"

static iree_status_t loom_vm_launch_config_contribute_pipeline(
    const loom_target_pipeline_contribution_t* contribution) {
  iree_string_view_t pass_name = iree_string_view_empty();
  if (contribution->phase ==
      LOOM_TARGET_PIPELINE_PHASE_SOURCE_ROOT_MATERIALIZATION) {
    pass_name = IREE_SV("vm-materialize-kernel-launch-configs");
  } else if (contribution->phase ==
             LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_MODULE_FINALIZATION) {
    pass_name = IREE_SV("vm-finalize-kernel-launch-configs");
  } else {
    return iree_ok_status();
  }
  loom_op_t* run_op = NULL;
  return loom_pass_ir_build_run(contribution->builder, 0, pass_name,
                                loom_named_attr_slice_empty(), &run_op);
}

static iree_status_t loom_vm_launch_config_prepare(
    iree_arena_allocator_t* arena,
    const loom_pass_environment_capability_t** out_capability) {
  *out_capability = NULL;
  loom_vm_launch_config_program_t* program = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*program), (void**)&program));
  loom_vm_launch_config_program_initialize(arena, program);
  *out_capability = loom_vm_launch_config_program_capability(program);
  return iree_ok_status();
}

static iree_status_t loom_vm_launch_config_emit_program(
    const loom_vm_launch_config_program_t* program, loom_module_t* module,
    const loom_vm_module_emitter_options_t* options,
    iree_arena_allocator_t* scratch_arena, iree_allocator_t host_allocator,
    iree_byte_sequence_t** out_contents) {
  loom_vm_launch_config_program_closure_t closure = {0};
  IREE_RETURN_IF_ERROR(loom_vm_launch_config_program_build_closure(
      program, module, scratch_arena, &closure));
  const loom_vm_module_emission_selection_t selection = {
      .symbol_liveness = &closure.symbol_liveness,
      .export_symbols = closure.root_symbols,
      .export_function_versions = closure.root_function_versions,
      .flags = LOOM_VM_MODULE_EMISSION_SELECTION_FLAG_STATELESS,
  };
  loom_vm_module_emitter_options_t launch_options = *options;
  launch_options.selection = &selection;
  return loom_vm_emit_module(module, &launch_options, scratch_arena,
                             host_allocator, out_contents);
}

static iree_status_t loom_vm_launch_config_emit(
    const loom_pass_environment_capability_t* capability,
    const loom_target_emit_request_t* request,
    loom_target_emit_artifact_t* out_artifact) {
  *out_artifact = (loom_target_emit_artifact_t){0};
  if (request->artifact_manifest.mode !=
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "VM launch-config emission does not produce artifact manifests");
  }

  const loom_vm_launch_config_program_t* program =
      (const loom_vm_launch_config_program_t*)capability;
  IREE_RETURN_IF_ERROR(
      loom_vm_launch_config_program_require_finalized(program));

  loom_target_emit_export_projection_t* export_projections = NULL;
  if (program->entries.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        request->scratch_arena, program->entries.count,
        sizeof(*export_projections), (void**)&export_projections));
  }
  loom_target_emit_export_projection_buffer_t export_projection = {
      .values = export_projections,
      .capacity = program->entries.count,
  };
  const loom_vm_module_emitter_options_t options = {
      .descriptor_registry = request->low_descriptor_registry,
      .diagnostic_emitter = request->diagnostic_emitter,
      .function_versions = request->function_versions,
      .export_projection = &export_projection,
  };
  iree_byte_sequence_t* contents = NULL;
  iree_status_t status = loom_vm_launch_config_emit_program(
      program, request->module, &options, request->scratch_arena,
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

static const loom_target_launch_config_compiler_t
    loom_vm_launch_config_compiler = {
        .public_artifact_format = IREE_SVL("vm"),
        .file_extension = IREE_SVL(".launch-config.vm"),
        .default_identifier = IREE_SVL("launch-config.vm"),
        .prepare = loom_vm_launch_config_prepare,
        .emit = loom_vm_launch_config_emit,
};

const loom_target_provider_t loom_vm_launch_config_compiler_provider = {
    .pass_registry = &loom_vm_launch_config_pass_registry,
    .contribute_pipeline = loom_vm_launch_config_contribute_pipeline,
    .launch_config_compiler = &loom_vm_launch_config_compiler,
};
