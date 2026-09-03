// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/artifact_emitter.h"

#include "iree/base/byte_sequence.h"
#include "iree/io/vec_stream.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amd/xdna/aie2p/array/plan.h"
#include "loom/target/arch/amd/xdna/aie2p/array/program.h"
#include "loom/target/arch/amd/xdna/aie2p/array/resident.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_compile.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/tile_link.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/xdna_product.h"
#include "loom/target/arch/amd/xdna/device/profile.h"
#include "loom/target/function_version.h"
#include "loom/target/reporting/low.h"

enum {
  LOOM_AIE2P_STRIX_HALO_PCI_VENDOR_ID = 0x1022,
  LOOM_AIE2P_STRIX_HALO_PCI_DEVICE_ID = 0x17F0,
  LOOM_AIE2P_STRIX_HALO_PCI_REVISION = 0x11,
};

static bool loom_aie2p_xdna_has_contract(const loom_module_t* module,
                                         loom_op_t* function_op,
                                         iree_string_view_t contract) {
  const loom_func_like_t function = loom_func_like_cast(module, function_op);
  const loom_string_id_t contract_id = loom_func_like_repr_contract(function);
  return contract_id < module->strings.count &&
         iree_string_view_equal(module->strings.entries[contract_id], contract);
}

static iree_status_t loom_aie2p_xdna_find_array_entry(
    const loom_target_emit_request_t* request, loom_op_t** out_function_op,
    iree_string_view_t* out_entry_name) {
  *out_function_op = NULL;
  *out_entry_name = iree_string_view_empty();
  loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(request->module, symbol) {
    if (symbol->defining_op == NULL ||
        !loom_low_func_def_isa(symbol->defining_op) ||
        !loom_aie2p_xdna_has_contract(request->module, symbol->defining_op,
                                      IREE_SV("amd.xdna.aie2p.array"))) {
      continue;
    }
    const loom_func_like_t function =
        loom_func_like_cast(request->module, symbol->defining_op);
    if (loom_func_like_visibility(function) == 0) continue;
    if (loom_func_like_abi(function) != LOOM_TARGET_ABI_ARRAY_PROGRAM ||
        !iree_all_bits_set(symbol->flags,
                           LOOM_SYMBOL_FLAG_PUBLIC | LOOM_SYMBOL_FLAG_RETAIN)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "public AIE2P array entries must be retained array programs");
    }
    if (*out_function_op != NULL) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AIE2P XDNA emission requires exactly one public array entry");
    }
    *out_function_op = symbol->defining_op;
    *out_entry_name = request->module->strings.entries[symbol->name_id];
  }
  if (*out_function_op == NULL) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "AIE2P XDNA emission requires one public retained array program");
  }
  return iree_ok_status();
}

static const loom_target_facts_t* loom_aie2p_xdna_function_target_facts(
    const loom_target_emit_request_t* request, loom_op_t* function_op) {
  const loom_target_function_version_t* version =
      loom_target_function_version_list_find(
          request->function_versions,
          loom_func_like_cast(request->module, function_op));
  return version != NULL ? version->function_target_facts : NULL;
}

static iree_status_t loom_aie2p_xdna_compile_source_leaves(
    const loom_target_emit_request_t* request,
    loom_aie2p_array_leaf_t** out_leaves, iree_host_size_t* out_leaf_count) {
  *out_leaves = NULL;
  *out_leaf_count = 0;
  iree_host_size_t leaf_count = 0;
  loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(request->module, symbol) {
    if (symbol->defining_op != NULL &&
        loom_low_func_def_isa(symbol->defining_op) &&
        loom_aie2p_xdna_has_contract(request->module, symbol->defining_op,
                                     IREE_SV("amd.xdna.aie2p.core"))) {
      ++leaf_count;
    }
  }
  if (leaf_count == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "AIE2P array program has no core workers");
  }

  loom_aie2p_array_leaf_t* leaves = NULL;
  loom_aie2p_leaf_contribution_t* contributions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      request->scratch_arena, leaf_count, sizeof(*leaves), (void**)&leaves));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      request->scratch_arena, leaf_count, sizeof(*contributions),
      (void**)&contributions));
  iree_host_size_t leaf_index = 0;
  loom_module_for_each_symbol(request->module, symbol) {
    loom_op_t* function_op = symbol->defining_op;
    if (function_op == NULL || !loom_low_func_def_isa(function_op) ||
        !loom_aie2p_xdna_has_contract(request->module, function_op,
                                      IREE_SV("amd.xdna.aie2p.core"))) {
      continue;
    }
    const loom_aie2p_leaf_compile_options_t options = {
        .descriptor_registry = request->low_descriptor_registry,
        .function_target_facts =
            loom_aie2p_xdna_function_target_facts(request, function_op),
        .diagnostic_emitter = request->diagnostic_emitter,
    };
    IREE_RETURN_IF_ERROR(loom_aie2p_leaf_compile(
        request->module, function_op, &options, request->scratch_arena,
        &contributions[leaf_index]));
    leaves[leaf_index] = (loom_aie2p_array_leaf_t){
        .entry =
            {
                .module_id = 0,
                .symbol_id =
                    (loom_symbol_id_t)(symbol -
                                       request->module->symbols.entries),
            },
        .contribution = &contributions[leaf_index],
    };
    ++leaf_index;
  }
  *out_leaves = leaves;
  *out_leaf_count = leaf_count;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_xdna_plan_tile_link(
    const loom_aie2p_array_plan_t* plan, uint32_t worker_index,
    const loom_aie2p_leaf_realization_t* realization,
    loom_aie2p_tile_storage_placement_t* storage_placements,
    loom_aie2p_tile_link_layout_t* out_layout) {
  if (realization->storage_domain_count > LOOM_STORAGE_SPACE_COUNT_) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AIE2P resident worker has too many storage domains");
  }
  for (iree_host_size_t i = 0; i < realization->storage_domain_count; ++i) {
    const loom_aie2p_leaf_storage_domain_t* domain =
        &realization->storage_domains[i];
    const loom_aie2p_array_worker_storage_plan_t* placement = NULL;
    for (iree_host_size_t j = 0; j < plan->worker_storage_count; ++j) {
      if (plan->worker_storage[j].worker_index == worker_index &&
          plan->worker_storage[j].storage_space == domain->storage_space) {
        placement = &plan->worker_storage[j];
        break;
      }
    }
    const loom_aie2p_leaf_storage_requirement_t* requirement =
        loom_aie2p_leaf_storage_requirement(realization, domain->storage_space);
    if (placement == NULL ||
        placement->byte_length != requirement->byte_length) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AIE2P resident storage changed after physical array planning");
    }
    storage_placements[i] = (loom_aie2p_tile_storage_placement_t){
        .storage_space = domain->storage_space,
        .load_address = placement->load_address,
    };
  }

  const loom_xdna_tile_coordinate_t coordinate =
      plan->worker_plans[worker_index].coordinate;
  const loom_xdna_tile_facts_t* tile = NULL;
  IREE_RETURN_IF_ERROR(
      loom_xdna_array_tile_facts(plan->family, coordinate, &tile));
  if (tile->kind != LOOM_XDNA_TILE_KIND_COMPUTE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P resident worker is not on a compute tile");
  }
  *out_layout = (loom_aie2p_tile_link_layout_t){
      .program_address = tile->memory.program_base,
      .program_byte_capacity = tile->memory.program_capacity,
      .storage_placements = storage_placements,
      .storage_placement_count = realization->storage_domain_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_aie2p_xdna_compile_resident_tiles(
    const loom_target_emit_request_t* request,
    const loom_aie2p_array_plan_t* plan,
    const loom_aie2p_array_resident_program_t* resident_program,
    loom_aie2p_xdna_tile_t** out_tiles) {
  *out_tiles = NULL;
  loom_aie2p_xdna_tile_t* tiles = NULL;
  loom_aie2p_leaf_contribution_t* contributions = NULL;
  loom_aie2p_linked_tile_t* linked_tiles = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      request->scratch_arena, resident_program->worker_count, sizeof(*tiles),
      (void**)&tiles));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      request->scratch_arena, resident_program->worker_count,
      sizeof(*contributions), (void**)&contributions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      request->scratch_arena, resident_program->worker_count,
      sizeof(*linked_tiles), (void**)&linked_tiles));

  for (iree_host_size_t i = 0; i < resident_program->worker_count; ++i) {
    const loom_aie2p_array_resident_worker_t* resident =
        &resident_program->workers[i];
    loom_aie2p_leaf_contribution_t* contribution = &contributions[i];
    loom_target_compile_report_t worker_report;
    loom_target_compile_report_t* worker_report_ptr = NULL;
    if (request->compile_report != NULL) {
      loom_target_compile_report_initialize(&worker_report,
                                            request->compile_report->allocator);
      worker_report.requested_detail_flags =
          request->compile_report->requested_detail_flags;
      worker_report_ptr = &worker_report;
    }
    const loom_aie2p_leaf_compile_options_t worker_compile_options = {
        .descriptor_registry = request->low_descriptor_registry,
        .diagnostic_emitter = request->diagnostic_emitter,
        .compile_report = worker_report_ptr,
    };
    iree_status_t status = loom_aie2p_leaf_compile(
        request->module, resident->function_op, &worker_compile_options,
        request->scratch_arena, contribution);
    if (iree_status_is_ok(status) && worker_report_ptr != NULL) {
      status = loom_target_compile_report_record_entry_report(
          request->compile_report, worker_report_ptr);
    }
    if (worker_report_ptr != NULL) {
      loom_target_compile_report_deinitialize(worker_report_ptr);
    }
    IREE_RETURN_IF_ERROR(status);
    if (contribution->realization.resource_import_count != 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "materialized AIE2P resident worker retains resource imports");
    }
    loom_aie2p_tile_storage_placement_t
        storage_placements[LOOM_STORAGE_SPACE_COUNT_];
    loom_aie2p_tile_link_layout_t link_layout = {0};
    IREE_RETURN_IF_ERROR(loom_aie2p_xdna_plan_tile_link(
        plan, resident->worker_index, &contribution->realization,
        storage_placements, &link_layout));
    IREE_RETURN_IF_ERROR(loom_aie2p_tile_link(
        contribution, &link_layout, request->scratch_arena, &linked_tiles[i]));
    tiles[i] = (loom_aie2p_xdna_tile_t){
        .coordinate = plan->worker_plans[resident->worker_index].coordinate,
        .contribution = contribution,
        .linked_tile = &linked_tiles[i],
    };
  }
  *out_tiles = tiles;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_xdna_emit(
    const loom_target_emit_request_t* request,
    loom_target_emit_artifact_t* out_artifact) {
  *out_artifact = (loom_target_emit_artifact_t){0};
  if (request->artifact_manifest.mode !=
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AIE2P XDNA emission does not produce artifact manifests");
  }

  loom_op_t* array_function = NULL;
  iree_string_view_t entry_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_aie2p_xdna_find_array_entry(request, &array_function, &entry_name));
  if (request->compile_report != NULL) {
    loom_target_compile_report_initialize_if_empty(request->compile_report,
                                                   request->allocator);
    request->compile_report->artifact_kind =
        LOOM_TARGET_COMPILE_ARTIFACT_KIND_TARGET_ARTIFACT;
    loom_target_compile_report_record_low_kernel_workload(
        request->compile_report, array_function);
  }

  loom_aie2p_array_leaf_t* source_leaves = NULL;
  iree_host_size_t source_leaf_count = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_xdna_compile_source_leaves(
      request, &source_leaves, &source_leaf_count));
  loom_aie2p_array_plan_t array_plan = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_build(
      request->module, array_function, source_leaves, source_leaf_count,
      request->scratch_arena, &array_plan));
  loom_aie2p_array_program_t array_program = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_array_program_build(
      &array_plan, request->scratch_arena, &array_program));
  loom_aie2p_array_resident_program_t resident_program = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_array_materialize_resident_program(
      request->module, &array_plan, request->scratch_arena, &resident_program));
  loom_aie2p_xdna_tile_t* tiles = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_xdna_compile_resident_tiles(
      request, &array_plan, &resident_program, &tiles));

  const loom_xdna_device_profile_t* device_profile = NULL;
  IREE_RETURN_IF_ERROR(loom_xdna_device_profile_resolve_pci(
      LOOM_AIE2P_STRIX_HALO_PCI_VENDOR_ID, LOOM_AIE2P_STRIX_HALO_PCI_DEVICE_ID,
      LOOM_AIE2P_STRIX_HALO_PCI_REVISION, &device_profile));
  const loom_aie2p_xdna_product_t product = {
      .device_profile = device_profile,
      .entry_name = entry_name,
      .array_plan = &array_plan,
      .array_program = &array_program,
      .tiles = tiles,
      .tile_count = resident_program.worker_count,
  };

  iree_io_stream_t* stream = NULL;
  iree_status_t status = iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE,
      4096, request->allocator, &stream);
  if (iree_status_is_ok(status)) {
    status =
        loom_aie2p_xdna_product_write(&product, stream, request->scratch_arena);
  }
  const iree_io_stream_pos_t stream_length =
      stream != NULL ? iree_io_stream_length(stream) : 0;
  if (iree_status_is_ok(status) && stream_length <= 0) {
    status =
        iree_make_status(IREE_STATUS_INTERNAL, "AIE2P XDNA output is empty");
  }

  iree_byte_sequence_t* contents = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_vec_stream_move_contents(stream, &contents);
  }
  if (iree_status_is_ok(status) && request->compile_report != NULL) {
    loom_target_compile_report_record_artifact_size(request->compile_report,
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

const loom_target_emitter_t loom_aie2p_xdna_emitter = {
    .name = IREE_SVL("amd-xdna-strix-halo"),
    .public_artifact_format = IREE_SVL("xdna-strix-halo"),
    .default_identifier = IREE_SVL("module.xdna"),
    .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .emit = loom_aie2p_xdna_emit,
};
