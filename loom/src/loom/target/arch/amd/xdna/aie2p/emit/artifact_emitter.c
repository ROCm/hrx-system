// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/artifact_emitter.h"

#include "iree/base/byte_sequence.h"
#include "iree/io/vec_stream.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_plan.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_object.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/tile_image.h"
#include "loom/target/function_version.h"
#include "loom/target/reporting/low.h"

static iree_status_t loom_aie2p_tile_elf_select_function(
    const loom_target_emit_request_t* request, loom_op_t** out_function_op,
    const loom_target_facts_t** out_function_target_facts) {
  *out_function_op = NULL;
  *out_function_target_facts = NULL;

  const loom_function_version_list_t* function_versions =
      request->function_versions;
  if (function_versions != NULL && function_versions->count != 0) {
    if (function_versions->count != 1 || function_versions->values == NULL ||
        function_versions->values[0] == NULL) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AIE2P tile ELF emission requires exactly one specialized function "
          "version per artifact");
    }
    loom_function_version_t* function_version = function_versions->values[0];
    *out_function_op = function_version->function.op;
    *out_function_target_facts =
        loom_target_function_version_target_facts(function_version);
  } else {
    loom_symbol_t* symbol = NULL;
    loom_module_for_each_symbol(request->module, symbol) {
      if (!loom_low_function_def_isa(symbol->defining_op)) continue;
      if (*out_function_op != NULL) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AIE2P tile ELF emission requires exactly one prepared target-low "
            "function per artifact");
      }
      *out_function_op = symbol->defining_op;
    }
  }

  if (!loom_low_function_def_isa(*out_function_op)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P tile ELF emission requires one prepared target-low function");
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_tile_elf_require_core_representation(
    const loom_target_emit_request_t* request, loom_op_t* function_op) {
  const loom_func_like_t function =
      loom_func_like_cast(request->module, function_op);
  const loom_string_id_t contract_id = loom_func_like_repr_contract(function);
  const loom_low_descriptor_set_t* descriptor_set = NULL;
  if (contract_id < request->module->strings.count) {
    descriptor_set = loom_low_descriptor_registry_lookup(
        request->low_descriptor_registry,
        request->module->strings.entries[contract_id]);
  }
  if (descriptor_set != NULL &&
      descriptor_set->stable_id == AIE2P_CORE_DESCRIPTOR_SET_ID) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "AIE2P tile ELF emission requires an amd.xdna.aie2p.core Low "
      "function");
}

static iree_status_t loom_aie2p_tile_elf_emit(
    const loom_target_emit_request_t* request,
    loom_target_emit_artifact_t* out_artifact) {
  *out_artifact = (loom_target_emit_artifact_t){0};
  if (request->artifact_manifest.mode !=
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AIE2P tile ELF emission does not produce artifact manifests");
  }
  loom_op_t* function_op = NULL;
  const loom_target_facts_t* function_target_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_tile_elf_select_function(
      request, &function_op, &function_target_facts));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_tile_elf_require_core_representation(request, function_op));

  loom_target_compile_report_t* report = request->compile_report;
  if (report != NULL) {
    loom_target_compile_report_initialize_if_empty(report, request->allocator);
    report->artifact_kind = LOOM_TARGET_COMPILE_ARTIFACT_KIND_TARGET_ARTIFACT;
    loom_target_compile_report_record_low_kernel_workload(report, function_op);
  }

  loom_low_planning_statistics_t planning_statistics = {0};
  const loom_low_emission_frame_options_t frame_options = {
      .descriptor_registry = request->low_descriptor_registry,
      .function_target_facts = function_target_facts,
      .memory_access_table = loom_low_memory_access_table_empty(),
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
      .emitter = request->diagnostic_emitter,
      .statistics = report != NULL ? &planning_statistics : NULL,
  };
  const loom_low_emission_frame_spill_free_options_t spill_free_options = {
      .materialization_options =
          {
              .has_supported_storage_spaces = true,
              .supported_storage_spaces = LOOM_LOW_STORAGE_SPACE_SET_NONE,
              .record_materialized_spills = true,
              .emitter = request->diagnostic_emitter,
          },
  };
  loom_low_emission_frame_t frame = {0};
  iree_status_t status = loom_low_emission_frame_build_spill_free(
      request->module, function_op, &frame_options, &spill_free_options,
      request->scratch_arena, &frame);
  if (report != NULL) {
    loom_target_compile_report_record_low_planning(report,
                                                   &planning_statistics);
  }
  if (iree_status_is_ok(status) && report != NULL) {
    const loom_target_bundle_t* bundle =
        loom_low_resolved_target_bundle(&frame.target);
    if (bundle != NULL) {
      loom_target_compile_report_record_target_bundle(report, bundle);
    }
    status =
        loom_target_compile_report_record_low_emission_frame(report, &frame);
  }

  loom_aie2p_bundle_plan_t bundle_plan = {0};
  if (iree_status_is_ok(status)) {
    status = loom_aie2p_bundle_plan_build(&frame, request->scratch_arena,
                                          &bundle_plan);
  }
  loom_aie2p_leaf_contribution_t contribution = {0};
  if (iree_status_is_ok(status)) {
    status = loom_aie2p_leaf_object_emit(&bundle_plan, request->scratch_arena,
                                         &contribution);
  }

  iree_io_stream_t* stream = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_vec_stream_create(IREE_IO_STREAM_MODE_READABLE |
                                           IREE_IO_STREAM_MODE_WRITABLE |
                                           IREE_IO_STREAM_MODE_SEEKABLE,
                                       4096, request->allocator, &stream);
  }
  if (iree_status_is_ok(status)) {
    status = loom_aie2p_tile_image_write(&contribution, stream,
                                         request->scratch_arena);
  }
  const iree_io_stream_pos_t stream_length =
      stream != NULL ? iree_io_stream_length(stream) : 0;
  if (iree_status_is_ok(status) && stream_length <= 0) {
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "AIE2P tile ELF output is empty");
  }

  iree_byte_sequence_t* contents = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_vec_stream_move_contents(stream, &contents);
  }
  if (iree_status_is_ok(status) && report != NULL) {
    loom_target_compile_report_record_emission(report, bundle_plan.slot_count,
                                               bundle_plan.encoded_byte_length,
                                               bundle_plan.encoded_byte_length);
    loom_target_compile_report_record_artifact_size(report,
                                                    (uint64_t)stream_length);
  }
  if (iree_status_is_ok(status)) {
    *out_artifact = (loom_target_emit_artifact_t){
        .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
        .contents = contents,
    };
    contents = NULL;
  }

  iree_byte_sequence_release(contents);
  iree_io_stream_release(stream);
  return status;
}

const loom_target_emitter_t loom_aie2p_tile_elf_emitter = {
    .name = IREE_SVL("amd-xdna-aie2p-tile-elf"),
    .public_artifact_format = IREE_SVL("elf-aie2p"),
    .default_identifier = IREE_SVL("tile.elf"),
    .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .emit = loom_aie2p_tile_elf_emit,
};
