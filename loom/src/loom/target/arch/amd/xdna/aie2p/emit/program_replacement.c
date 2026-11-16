// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_replacement.h"

#include <string.h>

#include "loom/analysis/pipeline/entries.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_program.h"
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
          native->functions[i + 1].equivalent_index;
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
              : shared_owners[native->functions[i + 1].equivalent_index];
    }
    status = loom_aie2p_program_layout_initialize(lifetimes, owners, count,
                                                  arena, &scratch, out_layout);
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

typedef struct loom_aie2p_program_replacement_candidate_t {
  // Private IR, retained by the artifact report when evidence is requested.
  loom_module_t* module;
  // Original source identities projected into this candidate's private IR.
  loom_aie2p_program_clone_t clone;
  // Prepared helpers and their exact native call contracts.
  loom_aie2p_program_compiler_t* compiler;
  // Native closure before physical composition.
  loom_aie2p_program_compilation_t compilation;
  // Descriptor at each original call site, indexed in callee-first source
  // order.
  uint32_t** call_descriptors;
  // Candidate-local evidence; only the selected program enters the final
  // report.
  loom_target_compile_report_t report;
} loom_aie2p_program_replacement_candidate_t;

static iree_status_t loom_aie2p_program_replacement_initialize(
    const loom_aie2p_xdna_artifact_request_t* request,
    loom_aie2p_program_replacement_candidate_t* candidate) {
  IREE_RETURN_IF_ERROR(
      loom_aie2p_xdna_allocate_private_module(request, &candidate->module));
  if (request->compile_report) {
    loom_target_compile_report_initialize(request->compile_report->allocator,
                                          &candidate->report);
    candidate->report.requested_detail_flags =
        request->compile_report->requested_detail_flags;
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_replacement_prepare_native(
    const loom_aie2p_xdna_artifact_request_t* request,
    loom_aie2p_program_source_table_t* sources,
    const loom_aie2p_program_source_function_t* source,
    const loom_aie2p_program_clone_t* source_clone,
    const loom_aie2p_program_compiler_t* source_compiler, uint32_t queue_header,
    uint32_t queue_command, const uint32_t* descriptors,
    const loom_aie2p_program_layout_t* layout, bool needs_loading,
    const loom_aie2p_program_replacement_candidate_t* previous,
    loom_aie2p_program_replacement_candidate_t* candidate, bool* out_compiled) {
  iree_arena_allocator_t* arena = request->scratch_arena;
  loom_module_t* module = candidate->module;
  loom_target_compile_report_t* report =
      request->compile_report ? &candidate->report : NULL;
  loom_aie2p_program_clone_t* clone = &candidate->clone;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_clone_initialize(
      request->module, module, arena, clone));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_clone_reserve_function(clone, source));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_clone_reserve_data(clone, source->program_data));
  for (iree_host_size_t i = 0; i < source->helpers.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_program_clone_reserve_function(
        clone, loom_aie2p_program_source_table_lookup(
                   sources, source->helpers.values[i])));
  }
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_clone_data(clone, source->program_data));
  for (iree_host_size_t i = 0; i < source->helpers.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_program_clone_function(
        clone, loom_aie2p_program_source_table_lookup(
                   sources, source->helpers.values[i])));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_program_clone_function(clone, source));
  loom_symbol_ref_t loading_helper = {0};
  if (needs_loading) {
    IREE_RETURN_IF_ERROR(loom_aie2p_program_loading_build(
        module, queue_header, queue_command, &loading_helper));
  }
  loom_aie2p_program_compiler_t* compiler = NULL;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_compiler_allocate(module, arena, report, &compiler));
  candidate->compiler = compiler;
  bool* changed = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*changed), (void**)&changed));
  memset(changed, 0, module->symbols.count * sizeof(*changed));
  if (needs_loading) {
    changed[loading_helper.symbol_id] = previous == NULL;
  }
  const iree_host_size_t loading_count = needs_loading ? 1 : 0;
  loom_aie2p_program_helper_t* helpers = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, source->helpers.count + loading_count,
                                sizeof(*helpers), (void**)&helpers));
  if (needs_loading) {
    helpers[0] = (loom_aie2p_program_helper_t){
        .function = loading_helper,
        .may_return = true,
        .function_target_facts = source->target.target_facts};
    if (previous) {
      IREE_RETURN_IF_ERROR(loom_aie2p_program_compiler_reuse_helper(
          compiler, &helpers[0], previous->compiler,
          &previous->compilation.helpers.values[0]));
    }
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, source->helpers.count + 1, sizeof(*candidate->call_descriptors),
      (void**)&candidate->call_descriptors));
  for (iree_host_size_t i = 0; i <= source->helpers.count; ++i) {
    const loom_aie2p_program_source_function_t* input =
        i == source->helpers.count ? source
                                   : loom_aie2p_program_source_table_lookup(
                                         sources, source->helpers.values[i]);
    loom_aie2p_program_clone_function_t* function =
        clone->functions_by_source[input->symbol.symbol_id];
    const loom_low_call_boundary_index_t calls =
        input->requirements.call_boundaries;
    uint32_t* call_descriptors = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, calls.count,
                                                   sizeof(*call_descriptors),
                                                   (void**)&call_descriptors));
    candidate->call_descriptors[i] = call_descriptors;
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
    if (previous) {
      modified = calls.count != 0 &&
                 memcmp(call_descriptors, previous->call_descriptors[i],
                        calls.count * sizeof(*call_descriptors)) != 0;
    }
    for (iree_host_size_t j = 0; j < function->direct_callees.count; ++j) {
      modified |= changed[function->direct_callees.values[j].symbol_id];
    }
    changed[function->entry.symbol_id] = modified;
    if (i == source->helpers.count) {
      continue;
    }
    helpers[i + loading_count] =
        loom_aie2p_program_replacement_helper(function);
    if (!modified) {
      const loom_aie2p_program_helper_t original =
          loom_aie2p_program_replacement_helper(
              (previous ? previous->clone.functions_by_source
                        : source_clone
                              ->functions_by_source)[input->symbol.symbol_id]);
      IREE_RETURN_IF_ERROR(loom_aie2p_program_compiler_reuse_helper(
          compiler, &helpers[i + loading_count],
          previous ? previous->compiler : source_compiler, &original));
    }
  }
  loom_low_read_only_data_set_t data;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_clone_project_data(
      clone, source->program_data, &data));
  const loom_aie2p_program_clone_function_t* root =
      clone->functions_by_source[source->symbol.symbol_id];
  const loom_aie2p_leaf_compile_options_t options = {
      .descriptor_registry = request->low_descriptor_registry,
      .function_target_facts = root->function_target_facts,
      .memory_accesses = root->memory_accesses,
      .loop_execution = root->loop_execution,
      .diagnostic_emitter = request->diagnostic_emitter,
      .compile_report = report,
  };
  return loom_aie2p_program_prepare(
      compiler, module->symbols.entries[root->entry.symbol_id].defining_op,
      root->may_return,
      (loom_aie2p_program_helper_list_t){helpers,
                                         source->helpers.count + loading_count},
      &data, &options, out_compiled, &candidate->compilation);
}

static iree_status_t loom_aie2p_program_replacement_compose(
    loom_aie2p_program_replacement_candidate_t* candidate,
    iree_arena_allocator_t* arena, loom_aie2p_program_layout_t* layout) {
  loom_aie2p_program_code_group_t* groups = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, layout->group_count, sizeof(*groups), (void**)&groups));
  const loom_aie2p_program_partition_t partition = {
      .function_groups = layout->function_groups,
      .group_count = layout->group_count,
      .groups = groups,
  };
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_compose(candidate->compiler, &candidate->compilation,
                                 &partition, &layout->program));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_layout_finish(groups, layout));
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_replacement_compile_impl(
    const loom_aie2p_xdna_artifact_request_t* request,
    loom_aie2p_program_source_table_t* sources,
    const loom_aie2p_program_source_function_t* source,
    const loom_aie2p_program_clone_t* source_clone,
    const loom_aie2p_program_compiler_t* source_compiler, uint32_t queue_header,
    uint32_t queue_command, const uint32_t* descriptors,
    loom_aie2p_program_replacement_candidate_t candidates[2],
    loom_aie2p_program_replacement_candidate_t** out_selected,
    loom_aie2p_program_layout_t* layout, bool* out_compiled) {
  iree_arena_allocator_t* arena = request->scratch_arena;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_replacement_initialize(request, &candidates[0]));
  *out_selected = &candidates[0];
  IREE_RETURN_IF_ERROR(loom_aie2p_program_replacement_prepare_native(
      request, sources, source, source_clone, source_compiler, queue_header,
      queue_command, descriptors, layout, true, NULL, &candidates[0],
      out_compiled));
  if (!*out_compiled) {
    return iree_ok_status();
  }
  const loom_aie2p_program_native_t* native = &candidates[0].compilation.native;
  loom_aie2p_program_layout_t selected_layout;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_layout_copy(
      layout, native->function_count, arena, &selected_layout));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_layout_retain(native, 0, arena, &selected_layout));
  iree_host_size_t* group_indices = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_layout_compact(
      native->function_count, arena, &selected_layout, &group_indices));
  if (selected_layout.group_count != layout->group_count) {
    uint32_t* selected_descriptors = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, layout->group_count - 1, sizeof(*selected_descriptors),
        (void**)&selected_descriptors));
    for (iree_host_size_t i = 1; i < layout->group_count; ++i) {
      selected_descriptors[i - 1] =
          group_indices[i] ? descriptors[group_indices[i] - 1] : UINT32_MAX;
    }
    IREE_RETURN_IF_ERROR(
        loom_aie2p_program_replacement_initialize(request, &candidates[1]));
    *out_selected = &candidates[1];
    IREE_RETURN_IF_ERROR(loom_aie2p_program_replacement_prepare_native(
        request, sources, source, source_clone, source_compiler, queue_header,
        queue_command, selected_descriptors, layout,
        selected_layout.group_count > 1, &candidates[0], &candidates[1],
        out_compiled));
    if (!*out_compiled) {
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_program_replacement_compose(
      *out_selected, arena, &selected_layout));
  if (selected_layout.byte_length > LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE &&
      *out_selected == &candidates[1]) {
    // Removing calls can change allocation and packet layout. Exact final
    // sizes, not the screening estimate, determine whether to keep the change.
    // The original native closure remains available with all loading calls.
    *out_selected = &candidates[0];
    IREE_RETURN_IF_ERROR(
        loom_aie2p_program_layout_retain(native, 1, arena, layout));
    IREE_RETURN_IF_ERROR(
        loom_aie2p_program_replacement_compose(*out_selected, arena, layout));
  } else {
    *layout = selected_layout;
  }
  loom_target_compile_report_t* report =
      request->compile_report ? &(*out_selected)->report : NULL;
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
  loom_aie2p_program_replacement_candidate_t candidates[2] = {0};
  loom_aie2p_program_replacement_candidate_t* selected = NULL;
  iree_status_t status = loom_aie2p_program_replacement_compile_impl(
      request, sources, source, source_clone, source_compiler, queue_header,
      queue_command, descriptors, candidates, &selected, layout, out_compiled);
  if (selected && request->compile_report) {
    status = iree_status_join(
        status, *out_compiled
                    ? loom_target_compile_report_record_entry_report(
                          request->compile_report, &selected->report)
                    : loom_target_compile_report_merge_evidence(
                          request->compile_report, &selected->report));
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(candidates); ++i) {
    if (!candidates[i].module) {
      continue;
    }
    if (request->compile_report) {
      loom_target_compile_report_deinitialize(&candidates[i].report);
    } else {
      loom_module_free(candidates[i].module);
    }
  }
  return status;
}
