// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/source_snapshots.h"

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/module.h"

iree_status_t loom_link_source_snapshots_initialize(
    const loom_link_module_index_t* index,
    const loom_link_plan_module_projection_t* projection,
    const loom_link_plan_materialization_environment_t* environment,
    iree_arena_allocator_t* scratch_arena,
    iree_arena_allocator_t* product_arena,
    loom_link_source_snapshots_t* out_snapshots) {
  *out_snapshots = (loom_link_source_snapshots_t){.environment = environment};
  if (!environment->source_entry) {
    return iree_ok_status();
  }

  iree_host_size_t total_count = 0;
  iree_host_size_t maximum_count = 0;
  for (iree_host_size_t i = 0; i < projection->modules.count; ++i) {
    const loom_link_module_index_module_t* module =
        projection->modules.values[i].source_module;
    const loom_link_module_index_provider_t* provider =
        loom_link_module_index_provider_at(index, module->provider_ordinal);
    const iree_host_size_t count =
        module->materialized_module
            ? module->materialized_module->sources.count
            : provider->bytecode.metadata
                  .modules[module->provider_module_ordinal]
                  .sources.count;
    if (!iree_host_size_checked_add(total_count, count, &total_count)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "linked source snapshot row allocation overflow");
    }
    maximum_count = iree_max(maximum_count, count);
  }
  loom_source_entry_t* current_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, maximum_count,
                                                 sizeof(*current_entries),
                                                 (void**)&current_entries));
  out_snapshots->current.entries = current_entries;
  loom_source_entry_t* target_entries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(product_arena, total_count,
                                                 sizeof(*target_entries),
                                                 (void**)&target_entries));
  for (iree_host_size_t i = 0; i < total_count; ++i) {
    target_entries[i] =
        (loom_source_entry_t){.source_id = LOOM_SOURCE_ID_INVALID};
  }
  out_snapshots->target.entries = target_entries;
  return iree_ok_status();
}

static bool loom_link_source_snapshots_resolve(void* user_data,
                                               const loom_module_t* module,
                                               loom_location_id_t location,
                                               loom_source_range_t* out_range) {
  loom_link_source_snapshots_t* snapshots = user_data;
  return loom_source_table_resolve(&snapshots->current, module, location,
                                   out_range) ||
         loom_source_table_resolve(&snapshots->target, module, location,
                                   out_range);
}

iree_status_t loom_link_source_snapshots_emit(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loom_link_source_snapshots_t* snapshots = user_data;
  const loom_source_resolver_t resolver = {
      .fn = loom_link_source_snapshots_resolve, .user_data = snapshots};
  loom_diagnostic_emission_t contextual = *emission;
  if (!contextual.source_resolver && snapshots->environment->source_entry) {
    contextual.source_resolver = &resolver;
  }
  return iree_diagnostic_emit(snapshots->environment->diagnostic_emitter,
                              &contextual);
}

iree_status_t loom_link_source_snapshots_capture(
    void* user_data, const loom_module_t* source_module,
    const loom_module_t* target_module, const loom_source_id_t* target_sources,
    bool* out_accepted) {
  loom_link_source_snapshots_t* snapshots = user_data;
  *out_accepted = true;
  snapshots->current.module = source_module;
  snapshots->current.count = source_module->sources.count;
  snapshots->target.module = target_module;
  snapshots->target.count = target_module->sources.count;
  loom_source_entry_t* current_entries =
      (loom_source_entry_t*)snapshots->current.entries;
  loom_source_entry_t* target_entries =
      (loom_source_entry_t*)snapshots->target.entries;
  for (iree_host_size_t i = 0; i < snapshots->current.count; ++i) {
    current_entries[i] =
        (loom_source_entry_t){.source_id = LOOM_SOURCE_ID_INVALID};
  }
  const loom_module_t* original = snapshots->input->materialized_module;
  if (!original) {
    return iree_ok_status();
  }

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < original->sources.count &&
                               iree_status_is_ok(status) && *out_accepted;
       ++i) {
    loom_source_entry_t entry = snapshots->environment->source_entry(
        snapshots->environment->user_data, snapshots->input,
        (loom_source_id_t)i);
    if (entry.source_id == LOOM_SOURCE_ID_INVALID) {
      continue;
    }
    const loom_source_id_t current_id = snapshots->projected_sources
                                            ? snapshots->projected_sources[i]
                                            : (loom_source_id_t)i;
    entry.source_id = current_id;
    current_entries[current_id] = entry;
    const loom_source_id_t target_id = target_sources[current_id];
    loom_source_entry_t* target_entry = &target_entries[target_id];
    if (target_entry->source_id != LOOM_SOURCE_ID_INVALID &&
        !iree_string_view_equal(target_entry->source, entry.source)) {
      const loom_diagnostic_param_t parameter =
          loom_param_string(entry.filename);
      const loom_diagnostic_emission_t emission = {
          .module = source_module,
          .error = LOOM_ERR_SYMBOL_019,
          .params = &parameter,
          .param_count = 1,
      };
      *out_accepted = false;
      status = loom_link_source_snapshots_emit(snapshots, &emission);
    } else {
      entry.source_id = target_id;
      *target_entry = entry;
    }
  }
  return status;
}
