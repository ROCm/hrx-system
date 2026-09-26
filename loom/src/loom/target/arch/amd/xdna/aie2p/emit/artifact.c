// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/artifact.h"

#include "iree/io/vec_stream.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function_requirements.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/amd/xdna/aie2p/array/plan.h"
#include "loom/target/arch/amd/xdna/aie2p/array/program.h"
#include "loom/target/arch/amd/xdna/aie2p/array/resident.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/array_report.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_compile.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/tile_link.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/xdna_product.h"
#include "loom/target/arch/amd/xdna/aie2p/facts.h"
#include "loom/target/arch/amd/xdna/device/profile.h"
#include "loom/target/function_version.h"
#include "loom/target/reporting/low.h"

static bool loom_aie2p_xdna_has_contract(const loom_module_t* module,
                                         const loom_op_t* function_op,
                                         iree_string_view_t contract) {
  const loom_func_like_t function =
      loom_func_like_const_cast(module, function_op);
  const loom_string_id_t contract_id = loom_func_like_repr_contract(function);
  return contract_id < module->strings.count &&
         iree_string_view_equal(
             loom_string_table_get(&module->strings, contract_id), contract);
}

typedef struct loom_aie2p_xdna_source_entry_t {
  // Public array function defining the entry.
  const loom_op_t* function_op;
  // Stable exported symbol name.
  iree_string_view_t name;
} loom_aie2p_xdna_source_entry_t;

static iree_status_t loom_aie2p_xdna_collect_array_entries(
    const loom_aie2p_xdna_artifact_request_t* request,
    loom_aie2p_xdna_source_entry_t** out_entries,
    iree_host_size_t* out_entry_count) {
  *out_entries = NULL;
  *out_entry_count = 0;
  iree_host_size_t entry_count = 0;
  const loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(request->module, symbol) {
    if (symbol->defining_op == NULL ||
        !loom_low_func_def_isa(symbol->defining_op) ||
        !loom_aie2p_xdna_has_contract(request->module, symbol->defining_op,
                                      IREE_SV("amd.xdna.aie2p.array"))) {
      continue;
    }
    const loom_func_like_t function =
        loom_func_like_const_cast(request->module, symbol->defining_op);
    if (loom_func_like_visibility(function) == 0) {
      continue;
    }
    if (loom_func_like_abi(function) != LOOM_TARGET_ABI_ARRAY_PROGRAM ||
        !iree_all_bits_set(symbol->flags,
                           LOOM_SYMBOL_FLAG_PUBLIC | LOOM_SYMBOL_FLAG_RETAIN)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "public AIE2P array entries must be retained array programs");
    }
    ++entry_count;
  }
  if (entry_count == 0) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "AIE2P XDNA emission requires a public retained array program");
  }

  loom_aie2p_xdna_source_entry_t* entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      request->scratch_arena, entry_count, sizeof(*entries), (void**)&entries));
  iree_host_size_t entry_index = 0;
  loom_module_for_each_symbol(request->module, symbol) {
    if (symbol->defining_op == NULL ||
        !loom_low_func_def_isa(symbol->defining_op) ||
        !loom_aie2p_xdna_has_contract(request->module, symbol->defining_op,
                                      IREE_SV("amd.xdna.aie2p.array"))) {
      continue;
    }
    const loom_func_like_t function =
        loom_func_like_const_cast(request->module, symbol->defining_op);
    if (loom_func_like_visibility(function) == 0) {
      continue;
    }
    entries[entry_index++] = (loom_aie2p_xdna_source_entry_t){
        .function_op = symbol->defining_op,
        .name =
            loom_string_table_get(&request->module->strings, symbol->name_id),
    };
  }
  IREE_ASSERT_EQ(entry_index, entry_count);
  *out_entries = entries;
  *out_entry_count = entry_count;
  return iree_ok_status();
}

static const loom_target_facts_t* loom_aie2p_xdna_function_target_facts(
    const loom_aie2p_xdna_artifact_request_t* request,
    const loom_op_t* function_op) {
  const loom_target_function_version_t* version =
      loom_target_function_version_list_find(
          request->function_versions,
          loom_func_like_const_cast(request->module, function_op));
  return version != NULL ? version->function_target_facts : NULL;
}

static iree_status_t loom_aie2p_xdna_resolve_device_profile(
    const loom_aie2p_xdna_artifact_request_t* request,
    const loom_aie2p_xdna_source_entry_t* entries, iree_host_size_t entry_count,
    const loom_xdna_device_profile_t** out_device_profile) {
  const loom_xdna_device_profile_t* device_profile = request->device_profile;
  for (iree_host_size_t i = 0; i < entry_count; ++i) {
    const loom_aie2p_target_facts_t* target_facts =
        loom_aie2p_target_facts_cast(loom_aie2p_xdna_function_target_facts(
            request, entries[i].function_op));
    const loom_xdna_device_profile_t* function_profile =
        target_facts != NULL ? target_facts->device_profile : NULL;
    if (function_profile == NULL) {
      continue;
    }
    if (device_profile != NULL && device_profile != function_profile) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "public AIE2P array entries select incompatible device profiles");
    }
    device_profile = function_profile;
  }
  if (device_profile == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P XDNA emission requires an exact device profile");
  }
  *out_device_profile = device_profile;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_xdna_collect_source_leaves(
    const loom_aie2p_xdna_artifact_request_t* request,
    loom_aie2p_array_leaf_t** out_leaves, iree_host_size_t* out_leaf_count,
    bool* out_valid) {
  *out_leaves = NULL;
  *out_leaf_count = 0;
  *out_valid = false;
  iree_host_size_t leaf_count = 0;
  const loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(request->module, symbol) {
    if (symbol->defining_op != NULL &&
        loom_low_func_def_isa(symbol->defining_op) &&
        loom_aie2p_xdna_has_contract(request->module, symbol->defining_op,
                                     IREE_SV("amd.xdna.aie2p.core"))) {
      ++leaf_count;
    }
  }
  loom_aie2p_array_leaf_t* leaves = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      request->scratch_arena, leaf_count, sizeof(*leaves), (void**)&leaves));
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, request->scratch_arena);
  loom_target_function_version_snapshot_t versions = {0};
  IREE_RETURN_IF_ERROR(loom_target_function_version_snapshot_build(
      request->module, request->function_versions, request->scratch_arena,
      &versions));
  iree_host_size_t leaf_index = 0;
  loom_module_for_each_symbol(request->module, symbol) {
    const loom_op_t* function_op = symbol->defining_op;
    if (function_op == NULL || !loom_low_func_def_isa(function_op) ||
        !loom_aie2p_xdna_has_contract(request->module, function_op,
                                      IREE_SV("amd.xdna.aie2p.core"))) {
      continue;
    }
    const loom_symbol_id_t symbol_id =
        (loom_symbol_id_t)(symbol - request->module->symbols.entries);
    const loom_target_function_version_t* version =
        loom_target_function_version_snapshot_at(&versions, symbol_id);
    loom_low_resolved_target_t target = {0};
    IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
        request->module, &symbol_facts, function_op,
        version != NULL ? version->function_target_facts : NULL,
        request->low_descriptor_registry, request->diagnostic_emitter,
        &target));
    if (target.descriptor_set == NULL) {
      return iree_ok_status();
    }
    // Array planning consumes declared interfaces and storage. Physical
    // lifetimes belong to the resident program after imports are bound and
    // protocol operations are inserted, not to this shared source body.
    loom_low_function_requirements_t requirements = {0};
    IREE_RETURN_IF_ERROR(loom_low_function_requirements_build(
        request->module, loom_low_func_def_body(function_op),
        request->scratch_arena, &requirements));
    leaves[leaf_index] = (loom_aie2p_array_leaf_t){
        .entry =
            {
                .module_id = 0,
                .symbol_id = symbol_id,
            },
        .function_op = function_op,
        .function_target_facts = target.target_facts,
        .memory_accesses = version != NULL ? version->memory_accesses : NULL,
        .requirements = requirements,
    };
    ++leaf_index;
  }
  *out_leaves = leaves;
  *out_leaf_count = leaf_count;
  *out_valid = true;
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
  const loom_xdna_tile_facts_t* tile =
      loom_xdna_array_tile_facts(plan->family, coordinate);
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
    const loom_aie2p_xdna_artifact_request_t* request,
    loom_module_t* resident_module, const loom_aie2p_array_plan_t* plan,
    const loom_aie2p_array_resident_program_t* resident_program,
    loom_aie2p_xdna_tile_t** out_tiles, bool* out_compiled) {
  *out_compiled = false;
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
        .function_target_facts = resident->function_target_facts,
        .memory_accesses = resident->memory_accesses,
        .descriptor_registry = request->low_descriptor_registry,
        .diagnostic_emitter = request->diagnostic_emitter,
        .compile_report = worker_report_ptr,
    };
    bool leaf_compiled = false;
    iree_status_t status = loom_aie2p_leaf_compile(
        resident_module, resident->function_op, &worker_compile_options,
        request->scratch_arena, &leaf_compiled, contribution);
    if (worker_report_ptr != NULL) {
      status = iree_status_join(
          status, loom_target_compile_report_record_entry_report(
                      request->compile_report, worker_report_ptr));
      loom_target_compile_report_deinitialize(worker_report_ptr);
    }
    IREE_RETURN_IF_ERROR(status);
    if (!leaf_compiled) {
      return iree_ok_status();
    }
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
  *out_compiled = true;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_xdna_compile_resident_entries(
    const loom_aie2p_xdna_artifact_request_t* request,
    loom_module_t* resident_module, const loom_aie2p_array_plan_t* array_plans,
    iree_host_size_t entry_count, loom_aie2p_xdna_entry_t* product_entries,
    bool* out_compiled) {
  *out_compiled = false;
  for (iree_host_size_t i = 0; i < entry_count; ++i) {
    loom_aie2p_array_resident_program_t resident_program = {0};
    IREE_RETURN_IF_ERROR(loom_aie2p_array_materialize_resident_program(
        request->module, resident_module, &array_plans[i],
        request->scratch_arena, &resident_program));
    loom_aie2p_xdna_tile_t* tiles = NULL;
    bool tiles_compiled = false;
    IREE_RETURN_IF_ERROR(loom_aie2p_xdna_compile_resident_tiles(
        request, resident_module, &array_plans[i], &resident_program, &tiles,
        &tiles_compiled));
    if (!tiles_compiled) {
      return iree_ok_status();
    }
    product_entries[i].tiles = tiles;
    product_entries[i].tile_count = resident_program.worker_count;
  }
  *out_compiled = true;
  return iree_ok_status();
}

typedef struct loom_aie2p_xdna_resident_storage_t {
  // Host allocator owning this storage record and the resident module.
  iree_allocator_t allocator;
  // Private block pool backing every resident-module arena allocation.
  iree_arena_block_pool_t block_pool;
  // Private module containing materialized resident worker functions.
  loom_module_t* module;
} loom_aie2p_xdna_resident_storage_t;

static void loom_aie2p_xdna_release_resident_storage(void* storage_ptr) {
  loom_aie2p_xdna_resident_storage_t* storage =
      (loom_aie2p_xdna_resident_storage_t*)storage_ptr;
  loom_module_free(storage->module);
  iree_arena_block_pool_deinitialize(&storage->block_pool);
  iree_allocator_free(storage->allocator, storage);
}

static iree_status_t loom_aie2p_xdna_allocate_resident_storage(
    const loom_aie2p_xdna_artifact_request_t* request,
    loom_aie2p_xdna_resident_storage_t** out_storage) {
  *out_storage = NULL;
  const iree_allocator_t allocator = request->compile_report->allocator;
  loom_aie2p_xdna_resident_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*storage), (void**)&storage));
  *storage = (loom_aie2p_xdna_resident_storage_t){
      .allocator = allocator,
  };
  iree_arena_block_pool_initialize(32 * 1024, allocator, &storage->block_pool);
  iree_status_t status = loom_module_allocate(
      request->module->context, IREE_SV("aie2p.xdna.resident"),
      &storage->block_pool, /*hints=*/NULL, allocator, &storage->module);
  if (iree_status_is_ok(status)) {
    *out_storage = storage;
  } else {
    loom_aie2p_xdna_release_resident_storage(storage);
  }
  return status;
}

iree_status_t loom_aie2p_xdna_artifact_emit(
    const loom_aie2p_xdna_artifact_request_t* request, bool* out_emitted,
    iree_byte_sequence_t** out_contents) {
  IREE_ASSERT_ARGUMENT(request);
  IREE_ASSERT_ARGUMENT(request->module);
  IREE_ASSERT_ARGUMENT(request->low_descriptor_registry);
  IREE_ASSERT_ARGUMENT(request->scratch_arena);
  IREE_ASSERT_ARGUMENT(out_emitted);
  IREE_ASSERT_ARGUMENT(out_contents);
  *out_emitted = false;
  *out_contents = NULL;

  loom_aie2p_xdna_source_entry_t* source_entries = NULL;
  iree_host_size_t entry_count = 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_xdna_collect_array_entries(
      request, &source_entries, &entry_count));
  const loom_xdna_device_profile_t* device_profile = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_xdna_resolve_device_profile(
      request, source_entries, entry_count, &device_profile));
  if (request->compile_report != NULL) {
    loom_target_compile_report_initialize_if_empty(request->compile_report,
                                                   request->allocator);
  }

  loom_aie2p_array_leaf_t* source_leaves = NULL;
  iree_host_size_t source_leaf_count = 0;
  bool source_leaves_valid = false;
  IREE_RETURN_IF_ERROR(loom_aie2p_xdna_collect_source_leaves(
      request, &source_leaves, &source_leaf_count, &source_leaves_valid));
  if (!source_leaves_valid) {
    return iree_ok_status();
  }

  loom_aie2p_array_plan_t* array_plans = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(request->scratch_arena, entry_count,
                                sizeof(*array_plans), (void**)&array_plans));
  loom_aie2p_array_program_t* array_programs = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      request->scratch_arena, entry_count, sizeof(*array_programs),
      (void**)&array_programs));
  loom_aie2p_xdna_entry_t* product_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      request->scratch_arena, entry_count, sizeof(*product_entries),
      (void**)&product_entries));
  for (iree_host_size_t i = 0; i < entry_count; ++i) {
    const loom_aie2p_xdna_source_entry_t* source_entry = &source_entries[i];
    if (request->compile_report != NULL) {
      loom_target_compile_report_record_low_kernel_workload(
          request->compile_report, source_entry->function_op);
    }
    bool valid = false;
    IREE_RETURN_IF_ERROR(loom_aie2p_array_plan_build(
        request->module, source_entry->function_op, source_leaves,
        source_leaf_count, request->diagnostic_emitter, request->scratch_arena,
        &array_plans[i], &valid));
    if (!valid) {
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_array_program_build(
        &array_plans[i], request->scratch_arena, &array_programs[i]));
    product_entries[i] = (loom_aie2p_xdna_entry_t){
        .name = source_entry->name,
        .array_plan = &array_plans[i],
        .array_program = &array_programs[i],
    };
  }

  loom_module_t* resident_module = NULL;
  iree_status_t status = iree_ok_status();
  if (request->compile_report != NULL) {
    loom_aie2p_xdna_resident_storage_t* resident_storage = NULL;
    status =
        loom_aie2p_xdna_allocate_resident_storage(request, &resident_storage);
    if (iree_status_is_ok(status)) {
      resident_module = resident_storage->module;
      loom_target_compile_report_take_storage(
          request->compile_report, resident_storage,
          loom_aie2p_xdna_release_resident_storage);
    }
  } else {
    status = loom_module_allocate(
        request->module->context, IREE_SV("aie2p.xdna.resident"),
        request->scratch_arena->block_pool, /*hints=*/NULL, request->allocator,
        &resident_module);
  }
  bool resident_entries_compiled = false;
  if (iree_status_is_ok(status)) {
    status = loom_aie2p_xdna_compile_resident_entries(
        request, resident_module, array_plans, entry_count, product_entries,
        &resident_entries_compiled);
  }
  if (request->compile_report == NULL) {
    loom_module_free(resident_module);
  }
  IREE_RETURN_IF_ERROR(status);
  if (!resident_entries_compiled) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < entry_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_array_report_record(
        request->module, product_entries[i].name, product_entries[i].array_plan,
        product_entries[i].tiles, request->compile_report,
        request->scratch_arena));
  }

  const loom_aie2p_xdna_product_t product = {
      .device_profile = device_profile,
      .entries = product_entries,
      .entry_count = entry_count,
  };

  iree_io_stream_t* stream = NULL;
  status = iree_io_vec_stream_create(IREE_IO_STREAM_MODE_READABLE |
                                         IREE_IO_STREAM_MODE_WRITABLE |
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
  if (iree_status_is_ok(status)) {
    *out_contents = contents;
    contents = NULL;
    *out_emitted = true;
  }
  iree_byte_sequence_release(contents);
  iree_io_stream_release(stream);
  return status;
}

static iree_status_t loom_aie2p_xdna_emit_target_artifact(
    const loom_target_emit_request_t* request, bool* out_emitted,
    loom_target_emit_artifact_t* out_artifact) {
  *out_emitted = false;
  *out_artifact = (loom_target_emit_artifact_t){0};
  if (request->artifact_manifest.mode !=
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "XDNA entry metadata is embedded in the canonical ELF; sidecar "
        "artifact manifests are not supported");
  }
  const loom_aie2p_xdna_artifact_request_t artifact_request = {
      .module = request->module,
      .function_versions = request->function_versions,
      .low_descriptor_registry = request->low_descriptor_registry,
      .compile_report = request->compile_report,
      .diagnostic_emitter = request->diagnostic_emitter,
      .scratch_arena = request->scratch_arena,
      .allocator = request->allocator,
  };
  IREE_RETURN_IF_ERROR(loom_aie2p_xdna_artifact_emit(
      &artifact_request, out_emitted, &out_artifact->contents));
  if (!*out_emitted) {
    return iree_ok_status();
  }
  out_artifact->target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;
  return iree_ok_status();
}

static const loom_target_emitter_t loom_aie2p_xdna_artifact_emitter = {
    .name = IREE_SVL("xdna"),
    .public_artifact_format = IREE_SVL("xdna"),
    .default_identifier = IREE_SVL("module.xdna"),
    .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .emit = loom_aie2p_xdna_emit_target_artifact,
};

static const loom_target_emitter_t* const kXdnaArtifactEmitters[] = {
    &loom_aie2p_xdna_artifact_emitter,
};

const loom_target_provider_t loom_aie2p_xdna_artifact_provider = {
    .emitter_list =
        {
            .values = kXdnaArtifactEmitters,
            .count = IREE_ARRAYSIZE(kXdnaArtifactEmitters),
        },
};
