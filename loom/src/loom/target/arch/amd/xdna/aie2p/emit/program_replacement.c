// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_replacement.h"

#include <string.h>

#include "loom/analysis/pipeline/entries.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/program_capacity.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/program_loading.h"

iree_status_t loom_aie2p_program_replacement_prepare(
    loom_aie2p_program_source_table_t* sources,
    const loom_aie2p_program_source_function_t* source,
    const loom_aie2p_program_object_t* native, iree_arena_allocator_t* arena,
    loom_aie2p_program_layout_t* out_layout) {
  iree_arena_allocator_t scratch;
  iree_arena_initialize(arena->block_pool, &scratch);
  const iree_host_size_t count = source->helpers.count + 2;
  iree_host_size_t* owners = NULL;
  iree_host_size_t* shared_owners = NULL;
  iree_status_t status = iree_arena_allocate_array(
      &scratch, count, sizeof(*owners), (void**)&owners);
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&scratch, native->function_count,
                                       sizeof(*shared_owners),
                                       (void**)&shared_owners);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < native->function_count; ++i) {
      shared_owners[i] = IREE_HOST_SIZE_MAX;
    }
    // Native equivalence includes imported callee/data identities and all
    // relocations. Equivalent ordinary helpers share their common caller
    // lifetime even when source specialization produced distinct symbols.
    // Bound phase entries keep their separate execution lifetimes.
    const loom_pipeline_lifetimes_t* lifetimes =
        &source->pipeline_entry->scope->lifetimes;
    for (iree_host_size_t i = 0; i < source->helpers.count; ++i) {
      if (loom_aie2p_program_source_table_lookup(sources,
                                                 source->helpers.values[i])
              ->pipeline_entry) {
        continue;
      }
      const iree_host_size_t canonical =
          native->functions[i + 1].canonical_index;
      const iree_host_size_t owner = source->entry_owners[i];
      shared_owners[canonical] =
          shared_owners[canonical] == IREE_HOST_SIZE_MAX
              ? owner
              : loom_pipeline_lifetimes_common_owner(
                    lifetimes, shared_owners[canonical], owner);
    }
    owners[0] = owners[1] = source->entry_owners[source->helpers.count];
    for (iree_host_size_t i = 0; i < source->helpers.count; ++i) {
      owners[i + 2] =
          loom_aie2p_program_source_table_lookup(sources,
                                                 source->helpers.values[i])
                  ->pipeline_entry
              ? source->entry_owners[i]
              : shared_owners[native->functions[i + 1].canonical_index];
    }
    status = loom_aie2p_program_layout_initialize(owners, count, arena,
                                                  &scratch, out_layout);
  }
  iree_arena_deinitialize(&scratch);
  return status;
}

static iree_host_size_t loom_aie2p_program_replacement_group(
    const loom_aie2p_program_layout_t* layout, iree_host_size_t owner) {
  if (layout->groups[0].owner == owner) {
    return 0;
  }
  iree_host_size_t first = 1;
  iree_host_size_t last = layout->group_count;
  while (first < last) {
    const iree_host_size_t middle = first + (last - first) / 2;
    if (layout->groups[middle].owner < owner) {
      first = middle + 1;
    } else {
      last = middle;
    }
  }
  return first;
}

static loom_aie2p_program_helper_t loom_aie2p_program_replacement_helper(
    const loom_aie2p_program_clone_function_t* function) {
  return (loom_aie2p_program_helper_t){
      .function = function->entry,
      .direct_callees = function->direct_callees,
      .may_return = function->may_return,
      .function_target_facts = function->function_target_facts,
      .memory_accesses = function->memory_accesses,
      .loop_execution = function->loop_execution,
  };
}

static iree_status_t loom_aie2p_program_replacement_compile_impl(
    const loom_aie2p_xdna_artifact_request_t* request,
    loom_aie2p_program_source_table_t* sources,
    const loom_aie2p_program_source_function_t* source,
    const loom_aie2p_program_clone_t* source_clone,
    const loom_aie2p_program_compiler_t* source_compiler, loom_module_t* module,
    uint32_t queue_header, uint32_t queue_command, const uint32_t* descriptors,
    loom_target_compile_report_t* report, loom_aie2p_program_layout_t* layout,
    bool* out_compiled) {
  iree_arena_allocator_t* arena = request->scratch_arena;
  loom_aie2p_program_clone_t clone;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_clone_initialize(
      request->module, module, arena, &clone));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_clone_reserve_function(&clone, source));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_clone_reserve_data(&clone, source->program_data));
  for (iree_host_size_t i = 0; i < source->helpers.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_program_clone_reserve_function(
        &clone, loom_aie2p_program_source_table_lookup(
                    sources, source->helpers.values[i])));
  }
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_clone_data(&clone, source->program_data));
  for (iree_host_size_t i = 0; i < source->helpers.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_program_clone_function(
        &clone, loom_aie2p_program_source_table_lookup(
                    sources, source->helpers.values[i])));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_program_clone_function(&clone, source));
  loom_symbol_ref_t loading_helper;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_loading_build(
      module, queue_header, queue_command, &loading_helper));
  loom_aie2p_program_compiler_t* compiler = NULL;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_compiler_allocate(module, arena, report, &compiler));
  bool* changed = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*changed), (void**)&changed));
  memset(changed, 0, module->symbols.count * sizeof(*changed));
  changed[loading_helper.symbol_id] = true;
  loom_aie2p_program_helper_t* helpers = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, source->helpers.count + 1, sizeof(*helpers), (void**)&helpers));
  helpers[0] = (loom_aie2p_program_helper_t){
      .function = loading_helper,
      .may_return = true,
      .function_target_facts = source->target.target_facts};
  for (iree_host_size_t i = 0; i <= source->helpers.count; ++i) {
    const loom_aie2p_program_source_function_t* input =
        i == source->helpers.count ? source
                                   : loom_aie2p_program_source_table_lookup(
                                         sources, source->helpers.values[i]);
    loom_aie2p_program_clone_function_t* function =
        clone.functions_by_source[input->symbol.symbol_id];
    const loom_low_call_boundary_index_t calls =
        input->requirements.call_boundaries;
    uint32_t* call_descriptors = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, calls.count,
                                                   sizeof(*call_descriptors),
                                                   (void**)&call_descriptors));
    for (iree_host_size_t j = 0; j < calls.count; ++j) {
      const loom_symbol_ref_t callee =
          loom_low_func_call_callee(calls.calls[j]);
      const loom_pipeline_entry_binding_t* binding =
          loom_aie2p_program_source_table_lookup(sources, callee)
              ->pipeline_entry;
      call_descriptors[j] = UINT32_MAX;
      if (binding == NULL || binding->owner == source->entry_owners[i] ||
          binding->role == LOOM_PIPELINE_ENTRY_RETIRE) {
        continue;
      }
      const iree_host_size_t group =
          loom_aie2p_program_replacement_group(layout, binding->owner);
      if (group != 0) {
        call_descriptors[j] = descriptors[group - 1];
      }
    }
    const iree_host_size_t original_count = function->call_boundaries.count;
    IREE_RETURN_IF_ERROR(loom_aie2p_program_loading_insert(
        module, loading_helper, call_descriptors, function));
    bool modified = function->call_boundaries.count != original_count;
    for (iree_host_size_t j = 0; j < function->direct_callees.count; ++j) {
      modified |= changed[function->direct_callees.values[j].symbol_id];
    }
    changed[function->entry.symbol_id] = modified;
    if (i == source->helpers.count) {
      continue;
    }
    helpers[i + 1] = loom_aie2p_program_replacement_helper(function);
    if (!modified) {
      const loom_aie2p_program_helper_t original =
          loom_aie2p_program_replacement_helper(
              source_clone->functions_by_source[input->symbol.symbol_id]);
      IREE_RETURN_IF_ERROR(loom_aie2p_program_compiler_reuse_helper(
          compiler, &helpers[i + 1], source_compiler, &original));
    }
  }
  loom_low_read_only_data_set_t data;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_clone_project_data(
      &clone, source->program_data, &data));
  const loom_aie2p_program_clone_function_t* root =
      clone.functions_by_source[source->symbol.symbol_id];
  const loom_aie2p_leaf_compile_options_t options = {
      .descriptor_registry = request->low_descriptor_registry,
      .function_target_facts = root->function_target_facts,
      .memory_accesses = root->memory_accesses,
      .loop_execution = root->loop_execution,
      .diagnostic_emitter = request->diagnostic_emitter,
      .compile_report = report,
  };
  loom_aie2p_program_code_group_t* groups = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, layout->group_count, sizeof(*groups), (void**)&groups));
  const loom_aie2p_program_partition_t partition = {
      .function_groups = layout->function_groups,
      .group_count = layout->group_count,
      .groups = groups,
  };
  IREE_RETURN_IF_ERROR(loom_aie2p_program_compile(
      compiler, module->symbols.entries[root->entry.symbol_id].defining_op,
      root->may_return,
      (loom_aie2p_program_helper_list_t){helpers, source->helpers.count + 1},
      &data, &options, &partition, out_compiled, &layout->program));
  if (!*out_compiled) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_program_layout_finish(
      &source->pipeline_entry->scope->lifetimes, groups, arena, layout));
  if (report) {
    loom_target_compile_report_record_emission(
        report, report->emitted_instruction_count,
        report->emitted_code_byte_count, layout->byte_length);
  }
  return loom_aie2p_program_capacity_admit(
      request->module, source->function_op, layout->byte_length,
      request->diagnostic_emitter, out_compiled);
}

iree_status_t loom_aie2p_program_replacement_compile(
    const loom_aie2p_xdna_artifact_request_t* request,
    loom_aie2p_program_source_table_t* sources,
    const loom_aie2p_program_source_function_t* source,
    const loom_aie2p_program_clone_t* source_clone,
    const loom_aie2p_program_compiler_t* source_compiler, uint32_t queue_header,
    uint32_t queue_command, const uint32_t* descriptors,
    loom_aie2p_program_layout_t* layout, bool* out_compiled) {
  *out_compiled = false;
  loom_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_xdna_allocate_private_module(request, &module));
  loom_target_compile_report_t report;
  loom_target_compile_report_t* report_ptr = NULL;
  if (request->compile_report) {
    loom_target_compile_report_initialize(request->compile_report->allocator,
                                          &report);
    report.requested_detail_flags =
        request->compile_report->requested_detail_flags;
    report_ptr = &report;
  }
  iree_status_t status = loom_aie2p_program_replacement_compile_impl(
      request, sources, source, source_clone, source_compiler, module,
      queue_header, queue_command, descriptors, report_ptr, layout,
      out_compiled);
  if (report_ptr) {
    status = iree_status_join(
        status, *out_compiled ? loom_target_compile_report_record_entry_report(
                                    request->compile_report, report_ptr)
                              : loom_target_compile_report_merge_evidence(
                                    request->compile_report, report_ptr));
    loom_target_compile_report_deinitialize(report_ptr);
  } else {
    loom_module_free(module);
  }
  return status;
}
