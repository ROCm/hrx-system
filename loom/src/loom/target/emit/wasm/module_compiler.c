// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/module_compiler.h"

#include <string.h>

#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function_model.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/arch/wasm/descriptors/descriptors.h"
#include "loom/target/arch/wasm/ops/ops.h"
#include "loom/target/arch/wasm/provider.h"
#include "loom/target/artifact_diagnostics.h"
#include "loom/target/function_version.h"
#include "loom/target/registers.h"

typedef struct loom_wasm_program_build_t {
  // Module being planned.
  loom_module_t* module;
  // Compiler function versions observed against this module symbol snapshot.
  loom_target_function_version_snapshot_t function_versions;
  // Mutable function plan storage.
  loom_wasm_function_plan_t* functions;
  // Number of initialized function plans.
  iree_host_size_t function_count;
  // Mutable interned signature storage.
  loom_wasm_function_type_t* types;
  // Number of initialized signatures.
  iree_host_size_t type_count;
  // Mutable symbol-to-function-index table.
  uint32_t* function_indices_by_symbol;
  // Number of exported functions.
  iree_host_size_t export_count;
} loom_wasm_program_build_t;

static iree_string_view_t loom_wasm_program_string_or_empty(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return loom_string_table_get(&module->strings, string_id);
}

static iree_string_view_t loom_wasm_program_symbol_name(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  if (symbol == NULL) {
    return IREE_SV("<unnamed>");
  }
  const iree_string_view_t name =
      loom_wasm_program_string_or_empty(module, symbol->name_id);
  return iree_string_view_is_empty(name) ? IREE_SV("<unnamed>") : name;
}

static iree_string_view_t loom_wasm_program_function_export_name(
    const loom_module_t* module, const loom_symbol_t* symbol,
    const loom_op_t* function_op) {
  const loom_func_like_t function =
      loom_func_like_const_cast(module, function_op);
  if (!loom_func_like_is_exported(function) &&
      (symbol->flags & LOOM_SYMBOL_FLAG_PUBLIC) == 0) {
    return iree_string_view_empty();
  }
  const iree_string_view_t export_name = loom_wasm_program_string_or_empty(
      module, loom_func_like_export_symbol(function));
  return iree_string_view_is_empty(export_name)
             ? loom_wasm_program_symbol_name(module, symbol)
             : export_name;
}

static iree_status_t loom_wasm_program_build_value_type_list(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_value_id_t* value_ids, uint32_t value_count,
    iree_string_view_t value_kind, iree_diagnostic_emitter_t emitter,
    iree_arena_allocator_t* arena, bool* out_accepted,
    const loom_wasm_value_type_t** out_types) {
  *out_accepted = false;
  *out_types = NULL;
  if (value_count == 0) {
    *out_accepted = true;
    return iree_ok_status();
  }
  loom_wasm_value_type_t* types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, value_count, sizeof(*types), (void**)&types));
  for (uint32_t i = 0; i < value_count; ++i) {
    const loom_type_t type = loom_module_value_type(module, value_ids[i]);
    if (loom_low_register_type_unit_count(type) != 1) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(
              loom_low_diagnostic_function_name(module, function_op)),
          loom_param_string(value_kind),
          loom_param_string(
              loom_low_diagnostic_value_name(module, value_ids[i])),
          loom_param_type(type),
          loom_param_string(IREE_SV("wasm-binary")),
          loom_param_string(IREE_SV("one register unit")),
      };
      return iree_diagnostic_emit(
          emitter,
          &(loom_diagnostic_emission_t){.op = function_op,
                                        .error = LOOM_ERR_TARGET_056,
                                        .params = params,
                                        .param_count = IREE_ARRAYSIZE(params)});
    }
    types[i] = loom_wasm_value_type_from_descriptor_register_class(
        loom_low_register_type_class_id(type));
  }
  *out_types = types;
  *out_accepted = true;
  return iree_ok_status();
}

static iree_status_t loom_wasm_program_build_function_type(
    const loom_low_allocation_table_t* allocation,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    bool* out_accepted, loom_wasm_function_type_t* out_type) {
  *out_accepted = false;
  *out_type = (loom_wasm_function_type_t){0};
  const loom_func_like_t function =
      loom_func_like_const_cast(allocation->module, allocation->function_op);
  uint16_t parameter_count = 0;
  const loom_value_id_t* parameters =
      loom_func_like_arg_ids(function, &parameter_count);
  const loom_value_slice_t results =
      loom_low_func_def_results(allocation->function_op);

  IREE_RETURN_IF_ERROR(loom_wasm_program_build_value_type_list(
      allocation->module, allocation->function_op, parameters, parameter_count,
      IREE_SV("parameter"), emitter, arena, out_accepted,
      &out_type->parameters));
  if (!*out_accepted) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_wasm_program_build_value_type_list(
      allocation->module, allocation->function_op, results.values,
      results.count, IREE_SV("result"), emitter, arena, out_accepted,
      &out_type->results));
  if (*out_accepted) {
    out_type->parameter_count = parameter_count;
    out_type->result_count = results.count;
  }
  return iree_ok_status();
}

static bool loom_wasm_program_function_types_equal(
    const loom_wasm_function_type_t* lhs,
    const loom_wasm_function_type_t* rhs) {
  if (lhs->parameter_count != rhs->parameter_count ||
      lhs->result_count != rhs->result_count) {
    return false;
  }
  if (lhs->parameter_count != 0 &&
      memcmp(lhs->parameters, rhs->parameters,
             lhs->parameter_count * sizeof(*lhs->parameters)) != 0) {
    return false;
  }
  return lhs->result_count == 0 ||
         memcmp(lhs->results, rhs->results,
                lhs->result_count * sizeof(*lhs->results)) == 0;
}

static uint32_t loom_wasm_program_intern_function_type(
    loom_wasm_program_build_t* build, const loom_wasm_function_type_t* type) {
  for (iree_host_size_t i = 0; i < build->type_count; ++i) {
    if (loom_wasm_program_function_types_equal(&build->types[i], type)) {
      return (uint32_t)i;
    }
  }
  const uint32_t type_index = (uint32_t)build->type_count;
  build->types[build->type_count++] = *type;
  return type_index;
}

static iree_status_t loom_wasm_program_build_function_allocation(
    loom_module_t* module, loom_op_t* function_op,
    const loom_target_facts_t* function_target_facts,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_low_allocation_table_t* out_allocation, bool* out_accepted) {
  *out_allocation = (loom_low_allocation_table_t){0};
  *out_accepted = false;
  bool synthesis_admitted = false;
  IREE_RETURN_IF_ERROR(loom_low_diagnostic_admit_allocation_synthesis(
      module, function_op, diagnostic_emitter, &synthesis_admitted));
  if (!synthesis_admitted) {
    return iree_ok_status();
  }
  loom_low_function_model_t model = {0};
  iree_status_t status = loom_low_function_model_initialize(
      module, function_op, function_target_facts, descriptor_registry,
      diagnostic_emitter,
      LOOM_LOW_FUNCTION_MODEL_FLAG_REGION_TREE |
          LOOM_LOW_FUNCTION_MODEL_FLAG_REQUIRE_SCHEDULE_MODEL,
      arena, &model);
  if (iree_status_is_ok(status) && model.context.error_count == 0 &&
      model.context.target.descriptor_set !=
          loom_wasm_core_simd128_descriptor_set()) {
    status = loom_target_artifact_emit_requirement(
        diagnostic_emitter, function_op, IREE_SV("Wasm"),
        IREE_SV("the wasm.core.simd128 representation"));
    loom_low_function_model_deinitialize(&model);
    return status;
  }
  if (iree_status_is_ok(status) && model.context.error_count == 0) {
    const loom_low_allocation_options_t allocation_options = {
        .emitter = diagnostic_emitter,
    };
    status = loom_low_allocate_function(&model, &allocation_options, arena,
                                        out_allocation);
  }
  if (iree_status_is_ok(status) && model.context.error_count == 0) {
    status = loom_low_allocation_diagnostics_emit(out_allocation, /*flags=*/0,
                                                  diagnostic_emitter);
  }
  const uint32_t context_error_count = model.context.error_count;
  loom_low_function_model_deinitialize(&model);
  if (iree_status_is_ok(status) && context_error_count != 0) {
    return iree_ok_status();
  }
  if (iree_status_is_ok(status) && out_allocation->error_count != 0) {
    return iree_ok_status();
  }
  if (iree_status_is_ok(status) &&
      (out_allocation->spill_count != 0 ||
       out_allocation->spill_plan_count != 0 ||
       out_allocation->edge_copy_count != 0 ||
       out_allocation->edge_copy_group_count != 0)) {
    return loom_target_artifact_emit_requirement(
        diagnostic_emitter, function_op, IREE_SV("Wasm"),
        IREE_SV("structured allocation without spills or CFG edge moves"));
  }
  if (iree_status_is_ok(status)) {
    *out_accepted = true;
  }
  return status;
}

static iree_status_t loom_wasm_program_project_function_allocation(
    const loom_low_allocation_table_t* allocation, uint32_t location_count,
    iree_arena_allocator_t* arena, loom_wasm_function_plan_t* function) {
  const loom_func_like_t function_like =
      loom_func_like_const_cast(allocation->module, allocation->function_op);
  uint16_t parameter_count = 0;
  const loom_value_id_t* parameter_ids =
      loom_func_like_arg_ids(function_like, &parameter_count);
  const iree_host_size_t class_count = allocation->physical_extents.count;

  loom_wasm_value_type_t* local_types = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, parameter_count + location_count,
                                sizeof(*local_types), (void**)&local_types));
  loom_wasm_value_locals_t* locals_by_value_ordinal = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, allocation->liveness.value_count, sizeof(*locals_by_value_ordinal),
      (void**)&locals_by_value_ordinal));
  memset(locals_by_value_ordinal, 0,
         allocation->liveness.value_count * sizeof(*locals_by_value_ordinal));
  uint32_t* local_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, allocation->unit_point_count, sizeof(*local_indices),
      (void**)&local_indices));
  uint32_t packet_move_count = 0;
  for (iree_host_size_t i = 0; i < allocation->packet_move_group_count; ++i) {
    packet_move_count +=
        allocation->packet_move_groups[i].move_group.moves.count != 0;
  }
  loom_wasm_packet_moves_t* packet_moves = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, packet_move_count, sizeof(*packet_moves), (void**)&packet_moves));
  loom_wasm_local_move_t* local_moves = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, allocation->packet_move_count,
                                sizeof(*local_moves), (void**)&local_moves));

  // Allocation owns dense class extents and the per-value unit domain. Project
  // them directly instead of searching previously encountered assignments.
  // Parameter indices are fixed by the ABI, so a tuple's units may map to
  // nonconsecutive locals when parameter classes are interleaved.
  const iree_arena_checkpoint_t checkpoint = iree_arena_checkpoint_save(arena);
  uint32_t* class_starts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, class_count, sizeof(*class_starts), (void**)&class_starts));
  uint32_t* locals_by_location = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, location_count,
                                                 sizeof(*locals_by_location),
                                                 (void**)&locals_by_location));
  uint32_t class_start = 0;
  for (iree_host_size_t i = 0; i < class_count; ++i) {
    class_starts[i] = class_start;
    class_start += allocation->physical_extents.ends_by_reg_class[i];
  }
  for (iree_host_size_t i = 0; i < location_count; ++i) {
    locals_by_location[i] = LOOM_WASM_PROGRAM_INDEX_NONE;
  }

  loom_low_allocation_value_scratch_t value_scratch = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_acquire_value_scratch(allocation, &value_scratch));
  for (uint32_t i = 0; i < parameter_count; ++i) {
    local_types[i] = function->type.parameters[i];
    const loom_low_allocation_assignment_t* assignment =
        loom_low_allocation_try_map_active_value_assignment(
            allocation, parameter_ids[i], NULL);
    if (assignment != NULL) {
      locals_by_location[class_starts[assignment->descriptor_reg_class_id] +
                         assignment->location_base] = i;
    }
  }

  uint32_t local_count = parameter_count;
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    const loom_value_ordinal_t value_ordinal =
        loom_module_value_ordinal_scratch_lookup(allocation->module,
                                                 assignment->value_id);
    locals_by_value_ordinal[value_ordinal] = (loom_wasm_value_locals_t){
        .start = assignment->unit_point_start,
        .count = assignment->unit_count,
    };
    const uint32_t location_start =
        class_starts[assignment->descriptor_reg_class_id] +
        assignment->location_base;
    const loom_wasm_value_type_t value_type =
        loom_wasm_value_type_from_descriptor_register_class(
            assignment->descriptor_reg_class_id);
    for (uint32_t unit = 0; unit < assignment->unit_count; ++unit) {
      const uint32_t start_point =
          loom_low_allocation_live_range_assignment_unit_start_point(
              allocation->unit_start_points, allocation->unit_point_count,
              assignment, unit);
      const uint32_t end_point =
          loom_low_allocation_live_range_assignment_unit_end_point(
              allocation->unit_end_points, allocation->unit_point_count,
              assignment, unit);
      uint32_t local_index = LOOM_WASM_PROGRAM_INDEX_NONE;
      if (start_point < end_point) {
        local_index = locals_by_location[location_start + unit];
        if (local_index == LOOM_WASM_PROGRAM_INDEX_NONE) {
          local_index = local_count++;
          local_types[local_index] = value_type;
          locals_by_location[location_start + unit] = local_index;
        }
      }
      local_indices[assignment->unit_point_start + unit] = local_index;
    }
  }
  // Decomposed concats carry references to their components instead of owning
  // a contiguous live range. Placement retains both the dependency order and
  // the captured source of each component; project those physical reads after
  // all assignment homes have been mapped.
  const loom_low_placement_table_t* placement = &allocation->placement;
  for (iree_host_size_t i = placement->storage_value_order_count; i > 0; --i) {
    const loom_value_ordinal_t value_ordinal =
        placement->storage_value_order[i - 1];
    const loom_wasm_value_locals_t result =
        locals_by_value_ordinal[value_ordinal];
    if (result.count == 0) {
      continue;
    }
    const loom_low_placement_relation_range_t range =
        loom_low_placement_relation_range_for_value_ordinal(placement,
                                                            value_ordinal);
    for (uint32_t j = 0; j < range.count; ++j) {
      const uint32_t relation_index = range.start + j;
      const loom_low_placement_relation_t* relation =
          &placement->relations[relation_index];
      if (relation->cause != LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT) {
        continue;
      }
      const loom_low_placement_concat_source_t source =
          loom_low_placement_concat_source(placement, relation_index);
      const loom_wasm_value_locals_t source_locals =
          locals_by_value_ordinal[source.value_ordinal];
      if (source_locals.count == 0) {
        continue;
      }
      for (uint32_t unit = 0; unit < relation->unit_count; ++unit) {
        uint32_t* result_local =
            &local_indices[result.start + relation->result_unit_offset + unit];
        if (*result_local == LOOM_WASM_PROGRAM_INDEX_NONE) {
          *result_local =
              local_indices[source_locals.start + source.unit_offset + unit];
        }
      }
    }
  }
  // The shared allocator has already resolved packet-copy cycles and retained
  // their scratch target IDs. Emission receives only the final local sequence.
  uint32_t group_index = 0;
  iree_host_size_t move_index = 0;
  for (iree_host_size_t i = 0; i < allocation->packet_move_group_count; ++i) {
    const loom_low_allocation_packet_move_group_t* group =
        &allocation->packet_move_groups[i];
    if (group->move_group.moves.count == 0) {
      continue;
    }
    packet_moves[group_index++] = (loom_wasm_packet_moves_t){
        .op = loom_liveness_operation_at(&allocation->liveness,
                                         group->source_ordinal)
                  ->op,
        .start = move_index,
        .count = group->move_group.moves.count,
    };
    for (iree_host_size_t j = 0; j < group->move_group.moves.count; ++j) {
      const loom_low_move_t* move =
          &allocation->moves[group->move_group.moves.start + j];
      // Wasm's unbounded numeric target-ID classes require only ordinary
      // copies, including cycle saves into fresh locals.
      IREE_ASSERT_EQ(move->kind, LOOM_LOW_MOVE_KIND_COPY);
      const uint32_t source_location =
          class_starts[move->source.descriptor_reg_class_id] +
          move->source.location;
      const uint32_t target_location =
          class_starts[move->destination.descriptor_reg_class_id] +
          move->destination.location;
      if (locals_by_location[target_location] == LOOM_WASM_PROGRAM_INDEX_NONE) {
        locals_by_location[target_location] = local_count;
        local_types[local_count++] =
            loom_wasm_value_type_from_descriptor_register_class(
                move->destination.descriptor_reg_class_id);
      }
      local_moves[move_index++] = (loom_wasm_local_move_t){
          .source = locals_by_location[source_location],
          .target = locals_by_location[target_location],
      };
    }
  }
  loom_low_allocation_release_value_scratch(&value_scratch);
  iree_arena_checkpoint_restore(&checkpoint);

  function->schedule_model = allocation->target.schedule_model;
  function->value_ids = allocation->liveness.value_ids;
  function->locals_by_value_ordinal = locals_by_value_ordinal;
  function->local_indices = local_indices;
  function->packet_moves = packet_moves;
  function->packet_move_count = packet_move_count;
  function->local_moves = local_moves;
  function->value_count =
      (loom_value_ordinal_t)allocation->liveness.value_count;
  function->local_types = local_types;
  function->parameter_count = parameter_count;
  function->local_count = local_count;
  return iree_ok_status();
}

static iree_status_t loom_wasm_program_build_function(
    loom_wasm_program_build_t* build, loom_op_t* function_op,
    const loom_target_facts_t* function_target_facts,
    loom_wasm_function_plan_t* function,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    bool* out_accepted) {
  *out_accepted = false;
  loom_low_allocation_table_t allocation = {0};
  bool allocation_accepted = false;
  IREE_RETURN_IF_ERROR(loom_wasm_program_build_function_allocation(
      build->module, function_op, function_target_facts, descriptor_registry,
      diagnostic_emitter, arena, &allocation, &allocation_accepted));
  if (!allocation_accepted) {
    return iree_ok_status();
  }
  bool signature_accepted = false;
  IREE_RETURN_IF_ERROR(loom_wasm_program_build_function_type(
      &allocation, diagnostic_emitter, arena, &signature_accepted,
      &function->type));
  if (!signature_accepted) {
    return iree_ok_status();
  }
  uint64_t location_count = 0;
  for (iree_host_size_t i = 0; i < allocation.physical_extents.count; ++i) {
    location_count += allocation.physical_extents.ends_by_reg_class[i];
  }
  const uint64_t local_capacity =
      function->type.parameter_count + location_count;
  if (local_capacity > UINT32_MAX) {
    return loom_target_artifact_emit_limit(
        diagnostic_emitter, function_op, IREE_SV("Wasm"),
        IREE_SV("local table capacity"), local_capacity, UINT32_MAX);
  }
  IREE_RETURN_IF_ERROR(loom_wasm_program_project_function_allocation(
      &allocation, (uint32_t)location_count, arena, function));
  function->type_index =
      loom_wasm_program_intern_function_type(build, &function->type);
  *out_accepted = true;
  return iree_ok_status();
}

iree_status_t loom_wasm_program_plan_build(
    loom_module_t* module,
    const loom_function_version_list_t* function_versions,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    bool* out_accepted, loom_wasm_program_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(descriptor_registry);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_accepted);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_accepted = false;
  *out_plan = (loom_wasm_program_plan_t){0};
  if (module->symbols.count == 0) {
    return loom_target_artifact_emit_requirement(
        diagnostic_emitter, NULL, IREE_SV("Wasm"),
        IREE_SV("at least one low.func.def"));
  }

  loom_wasm_program_build_t build = {
      .module = module,
  };
  IREE_RETURN_IF_ERROR(loom_target_function_version_snapshot_build(
      module, function_versions, arena, &build.function_versions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module->symbols.count,
                                                 sizeof(*build.functions),
                                                 (void**)&build.functions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, module->symbols.count,
                                                 sizeof(*build.types),
                                                 (void**)&build.types));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*build.function_indices_by_symbol),
      (void**)&build.function_indices_by_symbol));
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    build.function_indices_by_symbol[i] = LOOM_WASM_PROGRAM_INDEX_NONE;
  }

  for (iree_host_size_t symbol_index = 0; symbol_index < module->symbols.count;
       ++symbol_index) {
    const loom_symbol_t* symbol = &module->symbols.entries[symbol_index];
    loom_op_t* defining_op = symbol->defining_op;
    if (defining_op == NULL) {
      continue;
    }
    if (!loom_low_func_def_isa(defining_op) &&
        !loom_low_func_decl_isa(defining_op) &&
        !loom_low_kernel_def_isa(defining_op)) {
      continue;
    }
    const iree_string_view_t export_name =
        loom_wasm_program_function_export_name(module, symbol, defining_op);
    const loom_target_function_version_t* function_version =
        loom_target_function_version_snapshot_at(
            &build.function_versions, (loom_symbol_id_t)symbol_index);
    loom_diagnostic_export_scope_t diagnostic_scope = {
        .export_name = function_version
                           ? loom_target_facts_export_name(
                                 function_version->function_target_facts)
                           : export_name,
        .target = diagnostic_emitter,
    };
    const iree_diagnostic_emitter_t function_emitter =
        loom_diagnostic_export_scope_emitter(&diagnostic_scope);
    if (loom_low_func_decl_isa(defining_op)) {
      return loom_target_artifact_emit_requirement(
          function_emitter, defining_op, IREE_SV("Wasm"),
          IREE_SV("function definitions; module imports are not supported"));
    }
    if (loom_low_kernel_def_isa(defining_op)) {
      return loom_target_artifact_emit_requirement(
          function_emitter, defining_op, IREE_SV("Wasm"),
          IREE_SV("host functions instead of kernel entry points"));
    }
    loom_wasm_function_plan_t* function =
        &build.functions[build.function_count];
    *function = (loom_wasm_function_plan_t){
        .body = loom_low_func_def_body(defining_op),
        .name = loom_wasm_program_symbol_name(module, symbol),
        .source_export_name = diagnostic_scope.export_name,
        .export_name = export_name,
        .function_index = (uint32_t)build.function_count,
    };
    if (function->name.size > UINT32_MAX ||
        function->export_name.size > UINT32_MAX) {
      return loom_target_artifact_emit_limit(
          function_emitter, defining_op, IREE_SV("Wasm"),
          IREE_SV("function name byte length"),
          iree_max(function->name.size, function->export_name.size),
          UINT32_MAX);
    }
    build.function_indices_by_symbol[symbol_index] = function->function_index;
    if (!iree_string_view_is_empty(function->export_name)) {
      ++build.export_count;
    }
    bool function_accepted = false;
    IREE_RETURN_IF_ERROR(loom_wasm_program_build_function(
        &build, defining_op,
        function_version != NULL ? function_version->function_target_facts
                                 : NULL,
        function, descriptor_registry, function_emitter, arena,
        &function_accepted));
    if (!function_accepted) {
      return iree_ok_status();
    }
    ++build.function_count;
  }

  if (build.function_count == 0) {
    return loom_target_artifact_emit_requirement(
        diagnostic_emitter, NULL, IREE_SV("Wasm"),
        IREE_SV("at least one low.func.def"));
  }
  *out_plan = (loom_wasm_program_plan_t){
      .module = module,
      .functions = build.functions,
      .function_count = build.function_count,
      .types = build.types,
      .type_count = build.type_count,
      .function_indices_by_symbol = build.function_indices_by_symbol,
      .symbol_count = module->symbols.count,
      .export_count = build.export_count,
  };
  *out_accepted = true;
  return iree_ok_status();
}

iree_status_t loom_wasm_compile_module_binary(
    loom_module_t* module,
    const loom_function_version_list_t* function_versions,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    iree_allocator_t allocator, bool* out_emitted,
    loom_wasm_module_binary_t* out_module) {
  IREE_ASSERT_ARGUMENT(out_emitted);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_emitted = false;
  *out_module = (loom_wasm_module_binary_t){0};

  loom_wasm_program_plan_t plan = {0};
  bool accepted = false;
  IREE_RETURN_IF_ERROR(loom_wasm_program_plan_build(
      module, function_versions, descriptor_registry, diagnostic_emitter, arena,
      &accepted, &plan));
  if (!accepted) {
    return iree_ok_status();
  }
  return loom_wasm_program_emit_binary(&plan, diagnostic_emitter, allocator,
                                       out_emitted, out_module);
}

static iree_status_t loom_wasm_module_emit(
    const loom_target_emit_request_t* request, bool* out_emitted,
    loom_target_emit_artifact_t* out_artifact) {
  *out_emitted = false;
  *out_artifact = (loom_target_emit_artifact_t){0};
  if (request->artifact_manifest.mode !=
      LOOM_TARGET_ARTIFACT_MANIFEST_MODE_NONE) {
    return loom_target_artifact_emit_requirement(
        request->diagnostic_emitter, NULL, IREE_SV("Wasm"),
        IREE_SV("artifact manifest mode 'none'"));
  }

  loom_wasm_module_binary_t module = {0};
  bool module_emitted = false;
  IREE_RETURN_IF_ERROR(loom_wasm_compile_module_binary(
      request->module, request->function_versions,
      request->low_descriptor_registry, request->diagnostic_emitter,
      request->scratch_arena, request->allocator, &module_emitted, &module));
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

const loom_target_emitter_t loom_wasm_module_emitter = {
    .name = IREE_SVL("wasm-binary"),
    .public_artifact_format = IREE_SVL("wasm-binary"),
    .default_identifier = IREE_SVL("module.wasm"),
    .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_WASM_BINARY,
    .default_pipeline_options =
        {
            .control_flow_lowering =
                LOOM_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW,
        },
    .emit = loom_wasm_module_emit,
};

const loom_target_provider_t loom_wasm_module_provider = {
    .emitter_list =
        {
            .values =
                (const loom_target_emitter_t* const[]){
                    &loom_wasm_module_emitter},
            .count = 1,
        },
    .canonical_module_emitter = &loom_wasm_module_emitter,
    .canonical_module_fact_type = &loom_wasm_target_fact_type,
};

static const loom_target_provider_t* const kWasmCompilerProviders[] = {
    &loom_wasm_target_provider,
    &loom_wasm_module_provider,
};

const loom_target_provider_set_t loom_wasm_compiler_provider_set = {
    .providers = kWasmCompilerProviders,
    .provider_count = IREE_ARRAYSIZE(kWasmCompilerProviders),
};
