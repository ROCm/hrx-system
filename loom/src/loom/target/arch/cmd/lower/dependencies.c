// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/dependencies.h"

#include <string.h>

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/module.h"
#include "loom/target/arch/cmd/lower/lower.h"
#include "loom/target/arch/cmd/lower/transients.h"
#include "loom/target/arch/cmd/program.h"
#include "loom/util/adaptive_sort.h"

typedef struct loom_cmd_memory_access_t {
  // Completed physical byte range; resource_index UINT32_MAX means ambient.
  loom_cmd_buffer_binding_t range;
  // Comparable source alias scope; NONE cannot separate external bindings.
  loom_value_fact_alias_scope_id_t alias_scope;
  // Scheduled wave owning this access.
  uint32_t wave;
  // Consumer/producer access bits from the portable command contract.
  uint32_t scopes;
  // Physical overlap component, assigned before the wave sweep.
  uint32_t component;
} loom_cmd_memory_access_t;

typedef struct loom_cmd_pending_dependency_t {
  // Complete descriptor to deduplicate after the wave sweep.
  loom_cmd_memory_dependency_t dependency;
  // Wave receiving this dependency.
  uint32_t wave;
} loom_cmd_pending_dependency_t;

typedef struct loom_cmd_memory_component_t {
  // Shared physical envelope, or resource_index UINT32_MAX for global memory.
  loom_cmd_buffer_binding_t range;
  // Outstanding dispatch/transfer writes, independent of their age.
  uint32_t writers;
  // Consumer scopes already reached by the outstanding writes in each domain.
  uint32_t visible_to[2];
  // Union of current-wave accesses, gathered before updating any writer.
  uint32_t wave_scopes;
} loom_cmd_memory_component_t;

typedef struct loom_cmd_dependency_build_t {
  // Source facts and the completed placement plan, immutable during planning.
  const loom_value_fact_table_t* facts;
  // Completed placement and execution schedule.
  const loom_cmd_lower_plan_t* lower;
  // Module lending the source-value placement index.
  loom_module_t* module;
  // Per-root temporary storage; no graph or interval state escapes this arena.
  iree_arena_allocator_t* arena;
  // Appended actual accesses in schedule order.
  loom_cmd_memory_access_t* accesses;
  // Number of initialized accesses.
  iree_host_size_t access_count;
  // Allocated access capacity.
  iree_host_size_t access_capacity;
  // An ambient access connects all physical ranges into one global component.
  bool has_unbounded_access;
} loom_cmd_dependency_build_t;

static loom_cmd_buffer_binding_t loom_cmd_dependency_placement(
    const loom_cmd_dependency_build_t* build, loom_value_id_t value) {
  const loom_value_ordinal_t ordinal =
      value == LOOM_VALUE_ID_INVALID
          ? LOOM_VALUE_ORDINAL_INVALID
          : loom_module_value_ordinal_scratch_lookup(build->module, value);
  if (ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    return (loom_cmd_buffer_binding_t){.resource_index = UINT32_MAX};
  }
  if (ordinal < build->lower->binding_count) {
    return build->lower->bindings[ordinal];
  }
  const loom_cmd_buffer_range_t* range =
      &build->lower->buffer_ranges[ordinal - build->lower->binding_count];
  return (loom_cmd_buffer_binding_t){
      .role = range->role,
      .resource_index = range->resource_index,
      .byte_offset = range->byte_offset,
      .byte_length = range->byte_length,
  };
}

static iree_status_t loom_cmd_dependency_append(
    loom_cmd_dependency_build_t* build, loom_cmd_memory_access_t access) {
  if (access.range.byte_length == 0 &&
      access.range.resource_index != UINT32_MAX) {
    return iree_ok_status();
  }
  if (build->access_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "command memory access table is too large");
  }
  if (build->access_count == build->access_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        build->arena, build->access_count, build->access_count + 1,
        sizeof(*build->accesses), &build->access_capacity,
        (void**)&build->accesses));
  }
  build->has_unbounded_access |= access.range.resource_index == UINT32_MAX;
  build->accesses[build->access_count++] = access;
  return iree_ok_status();
}

static iree_status_t loom_cmd_dependency_reference(
    loom_cmd_dependency_build_t* build, uint32_t wave, loom_value_id_t value,
    uint32_t scopes, loom_storage_access_footprint_t footprint,
    uint64_t exact_byte_length) {
  loom_value_id_t root = LOOM_VALUE_ID_INVALID;
  loom_value_fact_alias_scope_id_t alias_scope =
      LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE;
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(build->facts, value);
  loom_value_fact_buffer_reference_t buffer;
  loom_value_fact_view_reference_t view;
  if (loom_value_facts_query_buffer_reference(&build->facts->context, facts,
                                              &buffer)) {
    root = loom_value_fact_buffer_reference_resolve_root_value(buffer, value);
    alias_scope = buffer.alias_scope_id;
  } else if (loom_value_facts_query_view_reference(&build->facts->context,
                                                   facts, &view)) {
    root = loom_value_fact_view_reference_resolve_root_value(view, value);
    alias_scope = view.alias_scope_id;
  }
  loom_cmd_buffer_binding_t range = loom_cmd_dependency_placement(
      build, footprint == LOOM_STORAGE_ACCESS_FOOTPRINT_STORAGE ? root : value);
  if (exact_byte_length && range.resource_index != UINT32_MAX) {
    range.byte_length = exact_byte_length;
  }
  return loom_cmd_dependency_append(
      build, (loom_cmd_memory_access_t){.range = range,
                                        .alias_scope = alias_scope,
                                        .wave = wave,
                                        .scopes = scopes});
}

static iree_status_t loom_cmd_dependency_timestamp(
    loom_cmd_dependency_build_t* build,
    const loom_cmd_schedule_command_t* command,
    loom_cmd_transient_requirement_t* transient,
    iree_diagnostic_emitter_t emitter, bool* out_valid) {
  const loom_cmd_buffer_binding_t target =
      loom_cmd_dependency_placement(build, command->arguments.values[0]);
  iree_string_view_t requirement = iree_string_view_empty();
  if (target.resource_index == UINT32_MAX) {
    requirement = IREE_SV("an exactly placed buffer range");
  } else if (target.role == LOOM_CMD_BUFFER_ROLE_FIXED) {
    requirement = IREE_SV("writable invocation or transient storage");
  } else if (target.byte_offset % LOOM_CMD_PROGRAM_TIMESTAMP_ALIGNMENT != 0 ||
             target.byte_length < LOOM_CMD_PROGRAM_TIMESTAMP_BYTE_LENGTH) {
    requirement = IREE_SV("an aligned eight-byte destination");
  } else if (target.resource_index == transient->binding_index) {
    transient->minimum_alignment = iree_max(
        transient->minimum_alignment, LOOM_CMD_PROGRAM_TIMESTAMP_ALIGNMENT);
  }
  if (iree_string_view_is_empty(requirement)) {
    return iree_ok_status();
  }
  const loom_diagnostic_param_t params[] = {loom_param_string(requirement)};
  const loom_diagnostic_emission_t emission = {
      .module = build->module,
      .op = command->source_op,
      .error = LOOM_ERR_LOWERING_071,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  *out_valid = false;
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_cmd_dependency_collect(
    loom_cmd_dependency_build_t* build,
    const loom_storage_access_summary_t* const* kernel_accesses,
    const uint32_t* requirements, loom_cmd_transient_requirement_t* transient,
    iree_diagnostic_emitter_t emitter, bool* out_valid) {
  const loom_cmd_schedule_plan_t* schedule = build->lower->schedule;
  for (uint32_t wave_index = 0; wave_index < schedule->wave_count;
       ++wave_index) {
    const loom_cmd_schedule_wave_t wave = schedule->waves[wave_index];
    for (iree_host_size_t i = 0; i < wave.command_count; ++i) {
      const iree_host_size_t index = wave.command_offset + i;
      const loom_cmd_schedule_command_t* command = &schedule->commands[index];
      if (command->kind == LOOM_CMD_SCHEDULE_COMMAND_KIND_TIMESTAMP) {
        IREE_RETURN_IF_ERROR(loom_cmd_dependency_timestamp(
            build, command, transient, emitter, out_valid));
        if (!*out_valid) {
          return iree_ok_status();
        }
        if (schedule->wave_count > 1) {
          IREE_RETURN_IF_ERROR(loom_cmd_dependency_reference(
              build, wave_index, command->arguments.values[0],
              LOOM_CMD_PROGRAM_ACCESS_TRANSFER_WRITE,
              LOOM_STORAGE_ACCESS_FOOTPRINT_REFERENCE,
              LOOM_CMD_PROGRAM_TIMESTAMP_BYTE_LENGTH));
        }
        continue;
      }
      if (schedule->wave_count <= 1) {
        continue;
      }
      const loom_cmd_dispatch_count_t* count =
          &build->lower->dispatch_counts[index];
      if (count->kind != LOOM_CMD_DISPATCH_COUNT_KIND_DIRECT) {
        IREE_RETURN_IF_ERROR(loom_cmd_dependency_reference(
            build, wave_index, count->payload.indirect_source_value,
            LOOM_CMD_PROGRAM_ACCESS_INDIRECT_COMMAND_READ,
            LOOM_STORAGE_ACCESS_FOOTPRINT_REFERENCE, 3 * sizeof(uint32_t)));
      }
      const loom_storage_access_summary_t* summary = kernel_accesses
          [requirements[build->lower->dispatches[index].entry_index]];
      if (!summary || summary->has_unbounded_memory_access) {
        IREE_RETURN_IF_ERROR(loom_cmd_dependency_append(
            build, (loom_cmd_memory_access_t){
                       .range = {.resource_index = UINT32_MAX},
                       .wave = wave_index,
                       .scopes = LOOM_CMD_PROGRAM_ACCESS_DISPATCH_READ |
                                 LOOM_CMD_PROGRAM_ACCESS_DISPATCH_WRITE}));
        continue;
      }
      for (uint16_t j = 0; j < summary->argument_count; ++j) {
        const loom_storage_access_argument_t argument = summary->arguments[j];
        uint32_t scopes = 0;
        if (argument.effects & LOOM_STORAGE_ACCESS_READ) {
          scopes |= LOOM_CMD_PROGRAM_ACCESS_DISPATCH_READ;
        }
        if (argument.effects & LOOM_STORAGE_ACCESS_WRITE) {
          scopes |= LOOM_CMD_PROGRAM_ACCESS_DISPATCH_WRITE;
        }
        if (argument.effects &
            (LOOM_STORAGE_ACCESS_ESCAPE | LOOM_STORAGE_ACCESS_ASYNC)) {
          scopes |= LOOM_CMD_PROGRAM_ACCESS_DISPATCH_READ |
                    LOOM_CMD_PROGRAM_ACCESS_DISPATCH_WRITE;
        }
        if (!scopes) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_cmd_dependency_reference(
            build, wave_index, command->source_arguments.values[j], scopes,
            argument.footprint, 0));
      }
    }
  }
  return iree_ok_status();
}

static uint64_t loom_cmd_dependency_root(loom_cmd_buffer_binding_t range) {
  return range.resource_index == UINT32_MAX
             ? UINT64_MAX
             : ((uint64_t)range.role << 32) | range.resource_index;
}

static uint64_t loom_cmd_dependency_end(loom_cmd_buffer_binding_t range) {
  return range.byte_length == UINT64_MAX
             ? UINT64_MAX
             : range.byte_offset + range.byte_length;
}

static bool loom_cmd_dependency_range_less(
    loom_cmd_memory_access_t* const* lhs,
    loom_cmd_memory_access_t* const* rhs) {
  const uint64_t left = loom_cmd_dependency_root((*lhs)->range);
  const uint64_t right = loom_cmd_dependency_root((*rhs)->range);
  return left != right ? left < right
                       : (*lhs)->range.byte_offset < (*rhs)->range.byte_offset;
}
LOOM_DEFINE_ADAPTIVE_SORT(loom_cmd_dependency_sort_ranges,
                          loom_cmd_memory_access_t*,
                          loom_cmd_dependency_range_less)

static bool loom_cmd_dependency_descriptor_less(
    const loom_cmd_pending_dependency_t* lhs,
    const loom_cmd_pending_dependency_t* rhs) {
  if (lhs->dependency.range_index != rhs->dependency.range_index) {
    return lhs->dependency.range_index < rhs->dependency.range_index;
  }
  if (lhs->dependency.source != rhs->dependency.source) {
    return lhs->dependency.source < rhs->dependency.source;
  }
  return lhs->dependency.target < rhs->dependency.target;
}
LOOM_DEFINE_ADAPTIVE_SORT(loom_cmd_dependency_sort_descriptors,
                          loom_cmd_pending_dependency_t,
                          loom_cmd_dependency_descriptor_less)

static bool loom_cmd_dependency_scope_less(
    const loom_value_fact_alias_scope_id_t* lhs,
    const loom_value_fact_alias_scope_id_t* rhs) {
  return *lhs < *rhs;
}
LOOM_DEFINE_ADAPTIVE_SORT(loom_cmd_dependency_sort_scopes,
                          loom_value_fact_alias_scope_id_t,
                          loom_cmd_dependency_scope_less)

static iree_status_t loom_cmd_dependency_partition(
    loom_cmd_dependency_build_t* build, uint32_t transient_binding,
    loom_cmd_memory_component_t** out_components, uint32_t* out_count) {
  const iree_host_size_t count = build->access_count;
  loom_cmd_memory_access_t** sorted = NULL;
  loom_cmd_memory_component_t* components = NULL;
  loom_value_fact_alias_scope_id_t* external_scopes = NULL;
  if (build->has_unbounded_access) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(build->arena, sizeof(*components),
                                             (void**)&components));
    *components =
        (loom_cmd_memory_component_t){.range = {.resource_index = UINT32_MAX}};
    for (iree_host_size_t i = 0; i < count; ++i) {
      build->accesses[i].component = 0;
    }
    *out_components = components;
    *out_count = 1;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      build->arena, count, sizeof(*external_scopes), (void**)&external_scopes));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      build->arena, count, sizeof(*sorted), (void**)&sorted));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      build->arena, count, sizeof(*components), (void**)&components));
  for (iree_host_size_t i = 0; i < count; ++i) {
    sorted[i] = &build->accesses[i];
  }
  loom_cmd_dependency_sort_ranges(sorted, count);

  // All fresh command allocations share one owned slab. Source noalias does
  // not separate reused slab bytes. External roots use only canonical alias
  // scopes, and an unscoped reference connects every potentially aliased root.
  bool external_aliases = false;
  iree_host_size_t external_roots = 0;
  for (iree_host_size_t begin = 0; begin < count;) {
    const uint64_t root = loom_cmd_dependency_root(sorted[begin]->range);
    const bool transient =
        sorted[begin]->range.role == LOOM_CMD_BUFFER_ROLE_REBINDABLE &&
        sorted[begin]->range.resource_index == transient_binding;
    iree_host_size_t end = begin + 1;
    const loom_value_fact_alias_scope_id_t alias = sorted[begin]->alias_scope;
    while (end < count &&
           loom_cmd_dependency_root(sorted[end]->range) == root) {
      if (!transient && alias != sorted[end]->alias_scope) {
        external_aliases = true;
      }
      ++end;
    }
    if (!transient) {
      external_scopes[external_roots++] = alias;
      external_aliases |= alias == LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE;
    }
    begin = end;
  }
  loom_cmd_dependency_sort_scopes(external_scopes, external_roots);
  for (iree_host_size_t i = 1; i < external_roots; ++i) {
    external_aliases |= external_scopes[i - 1] == external_scopes[i];
  }
  *out_count = 0;
  uint32_t external_component = UINT32_MAX;
  if (external_aliases && external_roots > 1) {
    external_component = (*out_count)++;
    components[external_component] =
        (loom_cmd_memory_component_t){.range = {.resource_index = UINT32_MAX}};
  }
  uint64_t prior_root = UINT64_MAX;
  uint32_t component = UINT32_MAX;
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_cmd_memory_access_t* access = sorted[i];
    const uint64_t root = loom_cmd_dependency_root(access->range);
    const bool transient =
        access->range.role == LOOM_CMD_BUFFER_ROLE_REBINDABLE &&
        access->range.resource_index == transient_binding;
    if (!transient && external_component != UINT32_MAX) {
      access->component = external_component;
      continue;
    }
    if (root != prior_root || component == UINT32_MAX ||
        access->range.byte_offset >=
            loom_cmd_dependency_end(components[component].range)) {
      component = (*out_count)++;
      components[component] =
          (loom_cmd_memory_component_t){.range = access->range};
    } else {
      const uint64_t end =
          iree_max(loom_cmd_dependency_end(access->range),
                   loom_cmd_dependency_end(components[component].range));
      components[component].range.byte_length =
          end == UINT64_MAX ? UINT64_MAX
                            : end - components[component].range.byte_offset;
    }
    access->component = component;
    prior_root = root;
  }
  *out_components = components;
  return iree_ok_status();
}

static iree_status_t loom_cmd_dependency_sweep(
    loom_cmd_dependency_build_t* build, uint32_t transient_binding,
    iree_arena_allocator_t* arena, loom_cmd_dependency_plan_t* out_plan) {
  if (!build->access_count) {
    return iree_ok_status();
  }
  loom_cmd_memory_component_t* components = NULL;
  uint32_t component_count = 0;
  IREE_RETURN_IF_ERROR(loom_cmd_dependency_partition(
      build, transient_binding, &components, &component_count));
  uint32_t* touched = NULL;
  loom_cmd_pending_dependency_t* pending = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      build->arena, component_count, sizeof(*touched), (void**)&touched));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      build->arena, build->access_count, sizeof(*pending), (void**)&pending));
  uint32_t pending_count = 0;
  const uint32_t writers[] = {LOOM_CMD_PROGRAM_ACCESS_DISPATCH_WRITE,
                              LOOM_CMD_PROGRAM_ACCESS_TRANSFER_WRITE};
  for (iree_host_size_t begin = 0; begin < build->access_count;) {
    const uint32_t wave = build->accesses[begin].wave;
    uint32_t touched_count = 0;
    iree_host_size_t end = begin;
    for (; end < build->access_count && build->accesses[end].wave == wave;
         ++end) {
      const loom_cmd_memory_access_t access = build->accesses[end];
      loom_cmd_memory_component_t* component = &components[access.component];
      if (!component->wave_scopes) {
        touched[touched_count++] = access.component;
      }
      component->wave_scopes |= access.scopes;
    }
    for (uint32_t i = 0; i < touched_count; ++i) {
      const uint32_t index = touched[i];
      loom_cmd_memory_component_t* component = &components[index];
      const uint32_t target = component->wave_scopes;
      uint32_t source = 0;
      for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(writers); ++j) {
        if ((component->writers & writers[j]) &&
            (component->visible_to[j] & target) != target) {
          source |= writers[j];
          component->visible_to[j] |= target;
        }
      }
      if (source) {
        pending[pending_count++] = (loom_cmd_pending_dependency_t){
            .dependency = {.source = source,
                           .target = target,
                           .range_index =
                               component->range.resource_index == UINT32_MAX
                                   ? UINT32_MAX
                                   : index},
            .wave = wave};
      }
      for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(writers); ++j) {
        if (target & writers[j]) {
          component->writers |= writers[j];
          component->visible_to[j] = 0;
        }
      }
      component->wave_scopes = 0;
    }
    begin = end;
  }
  if (!pending_count) {
    return iree_ok_status();
  }

  loom_cmd_dependency_sort_descriptors(pending, pending_count);
  uint32_t unique_count = 0;
  uint32_t range_count = 0;
  for (uint32_t i = 0; i < pending_count; ++i) {
    unique_count +=
        !i || loom_cmd_dependency_descriptor_less(&pending[i - 1], &pending[i]);
    range_count += pending[i].dependency.range_index != UINT32_MAX &&
                   (!i || pending[i - 1].dependency.range_index !=
                              pending[i].dependency.range_index);
  }
  loom_cmd_memory_dependency_t* dependencies = NULL;
  loom_cmd_buffer_binding_t* ranges = NULL;
  loom_cmd_wave_dependencies_t* waves = NULL;
  uint32_t* indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, unique_count, sizeof(*dependencies), (void**)&dependencies));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, range_count, sizeof(*ranges), (void**)&ranges));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, build->lower->schedule->wave_count,
                                sizeof(*waves), (void**)&waves));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, pending_count, sizeof(*indices), (void**)&indices));
  memset(waves, 0, build->lower->schedule->wave_count * sizeof(*waves));
  for (uint32_t i = 0; i < pending_count; ++i) {
    ++waves[pending[i].wave].count;
  }
  uint32_t offset = 0;
  for (iree_host_size_t i = 0; i < build->lower->schedule->wave_count; ++i) {
    const uint32_t count = waves[i].count;
    waves[i] = (loom_cmd_wave_dependencies_t){.offset = offset};
    offset += count;
  }
  uint32_t dependency_count = 0;
  uint32_t range_index = 0;
  for (uint32_t i = 0; i < pending_count; ++i) {
    const uint32_t component_index = pending[i].dependency.range_index;
    if (component_index != UINT32_MAX &&
        (!i || component_index != pending[i - 1].dependency.range_index)) {
      ranges[range_index++] = components[component_index].range;
    }
    if (!i ||
        loom_cmd_dependency_descriptor_less(&pending[i - 1], &pending[i])) {
      dependencies[dependency_count++] = (loom_cmd_memory_dependency_t){
          .source = pending[i].dependency.source,
          .target = pending[i].dependency.target,
          .range_index =
              component_index == UINT32_MAX ? UINT32_MAX : range_index - 1,
      };
    }
    loom_cmd_wave_dependencies_t* wave = &waves[pending[i].wave];
    indices[wave->offset + wave->count++] = dependency_count - 1;
  }
  *out_plan = (loom_cmd_dependency_plan_t){.ranges = ranges,
                                           .dependencies = dependencies,
                                           .dependency_count = dependency_count,
                                           .dependency_indices = indices,
                                           .waves = waves};
  return iree_ok_status();
}

iree_status_t loom_cmd_dependency_plan_build(
    loom_module_t* module, const loom_value_fact_table_t* facts,
    const loom_value_id_t* binding_values, const loom_cmd_lower_plan_t* lower,
    const loom_storage_access_summary_t* const* kernel_accesses,
    const uint32_t* requirements, loom_cmd_transient_requirement_t* transient,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    bool* out_valid, loom_cmd_dependency_plan_t* out_plan) {
  *out_valid = true;
  *out_plan = (loom_cmd_dependency_plan_t){0};
  loom_module_value_ordinal_scratch_acquire(module);
  for (iree_host_size_t i = 0; i < lower->binding_count; ++i) {
    loom_module_value_ordinal_scratch_set(module, binding_values[i],
                                          (loom_value_ordinal_t)i);
  }
  for (iree_host_size_t i = 0; i < lower->buffer_range_count; ++i) {
    loom_module_value_ordinal_scratch_set(
        module, lower->buffer_ranges[i].source_value,
        (loom_value_ordinal_t)(lower->binding_count + i));
  }
  iree_arena_allocator_t temporary_arena;
  iree_arena_initialize(arena->block_pool, &temporary_arena);
  loom_cmd_dependency_build_t build = {.facts = facts,
                                       .lower = lower,
                                       .module = module,
                                       .arena = &temporary_arena};
  iree_status_t status = loom_cmd_dependency_collect(
      &build, kernel_accesses, requirements, transient, emitter, out_valid);
  if (*out_valid && iree_status_is_ok(status)) {
    status = loom_cmd_dependency_sweep(&build, transient->binding_index, arena,
                                       out_plan);
  }
  iree_arena_deinitialize(&temporary_arena);
  for (iree_host_size_t i = 0; i < lower->buffer_range_count; ++i) {
    loom_module_value_ordinal_scratch_clear(
        module, lower->buffer_ranges[i].source_value);
  }
  for (iree_host_size_t i = 0; i < lower->binding_count; ++i) {
    loom_module_value_ordinal_scratch_clear(module, binding_values[i]);
  }
  loom_module_value_ordinal_scratch_release(module);
  return status;
}
