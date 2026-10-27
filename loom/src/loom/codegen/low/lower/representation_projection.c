// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/representation_projection.h"

#include <inttypes.h>
#include <string.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/descriptor_traits.h"
#include "loom/codegen/low/representation_binding.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/registers.h"

typedef struct loom_low_representation_projection_t {
  // Representation contract used to interpret the authored function.
  const loom_low_descriptor_set_t* source_descriptor_set;
  // Exact representation contract selected for the function version.
  const loom_low_descriptor_set_t* target_descriptor_set;
} loom_low_representation_projection_t;

typedef struct loom_low_representation_value_update_t {
  // Function-local SSA value whose register type is projected.
  loom_value_id_t value;
  // Target representation type replacing the authored type.
  loom_type_t type;
} loom_low_representation_value_update_t;

typedef struct loom_low_representation_descriptor_update_t {
  // Descriptor-backed packet whose ordinal is projected.
  loom_op_t* op;
  // Source representation descriptor ordinal retained during planning.
  uint32_t source_ordinal;
  // Target representation descriptor ordinal.
  uint32_t ordinal;
  // Generic IR traits implied by the target representation descriptor.
  loom_trait_flags_t effective_traits;
  // Intrinsic write-space bound in the target representation.
  loom_memory_space_mask_t intrinsic_write_spaces;
} loom_low_representation_descriptor_update_t;

struct loom_low_representation_projection_plan_t {
  // Authored function whose representation is published after planning.
  loom_func_like_t function;
  // Canonical exact representation attribute retained during planning.
  loom_string_id_t target_descriptor_set_key;
  // First argument carrier in value_updates.
  uint16_t argument_offset;
  // First result carrier in value_updates.
  uint16_t result_offset;
  // Exact representation descriptors consumed by callers and publication.
  const loom_low_descriptor_set_t* descriptor_set;
  // Borrowed immutable function target facts selected by module planning.
  const loom_target_facts_t* target_facts;
  // Planned SSA type updates.
  loom_low_representation_value_update_t* value_updates;
  // Capacity of |value_updates|.
  iree_host_size_t value_capacity;
  // Number of populated value updates.
  iree_host_size_t value_count;
  // Planned descriptor ordinal updates.
  loom_low_representation_descriptor_update_t* descriptor_updates;
  // Capacity of |descriptor_updates|.
  iree_host_size_t descriptor_capacity;
  // Number of populated descriptor updates.
  iree_host_size_t descriptor_count;
};

iree_status_t loom_low_representation_projection_index_build(
    const loom_module_t* module,
    const loom_low_representation_projection_plan_t* const* plans,
    iree_host_size_t plan_count, iree_arena_allocator_t* arena,
    loom_low_representation_projection_index_t* out_index) {
  *out_index = (loom_low_representation_projection_index_t){0};
  if (plan_count == 0) {
    return iree_ok_status();
  }
  uint16_t* ordinals = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, module->symbols.count, sizeof(*ordinals), (void**)&ordinals));
  memset(ordinals, 0, module->symbols.count * sizeof(*ordinals));
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    const loom_symbol_ref_t callee = loom_func_like_callee(plans[i]->function);
    ordinals[callee.symbol_id] = (uint16_t)(i + 1);
  }
  *out_index = (loom_low_representation_projection_index_t){
      .plans = plans,
      .ordinals = ordinals,
      .symbol_count = module->symbols.count,
  };
  return iree_ok_status();
}

const loom_low_representation_projection_plan_t*
loom_low_representation_projection_index_find(
    const loom_low_representation_projection_index_t* index,
    loom_symbol_id_t symbol_id) {
  const uint16_t ordinal = index != NULL && symbol_id < index->symbol_count
                               ? index->ordinals[symbol_id]
                               : 0;
  return ordinal != 0 ? index->plans[ordinal - 1] : NULL;
}

const loom_low_descriptor_set_t* loom_low_representation_projection_descriptors(
    const loom_low_representation_projection_plan_t* plan) {
  return plan->descriptor_set;
}

const loom_target_facts_t* loom_low_representation_projection_target_facts(
    const loom_low_representation_projection_plan_t* plan) {
  return plan->target_facts;
}

loom_type_t loom_low_representation_projection_argument_type(
    const loom_low_representation_projection_plan_t* plan, uint16_t index) {
  return plan->value_updates[plan->argument_offset + index].type;
}

loom_type_t loom_low_representation_projection_result_type(
    const loom_low_representation_projection_plan_t* plan, uint16_t index) {
  return plan->value_updates[plan->result_offset + index].type;
}

static bool loom_low_representation_count_add(iree_host_size_t amount,
                                              iree_host_size_t* total) {
  return iree_host_size_checked_add(*total, amount, total);
}

static bool loom_low_representation_count_op(
    const loom_op_t* op, iree_host_size_t* value_count,
    iree_host_size_t* descriptor_count) {
  if (!loom_low_representation_count_add(op->result_count, value_count)) {
    return false;
  }
  if (loom_low_func_decl_isa(op) &&
      !loom_low_representation_count_add(op->operand_count, value_count)) {
    return false;
  }
  if (loom_low_descriptor_packet_kind(op) != LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    if (!loom_low_representation_count_add(1, descriptor_count)) {
      return false;
    }
  }
  loom_region_t* const* regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    const loom_region_t* region = regions[region_index];
    if (!region) {
      continue;
    }
    for (uint16_t block_index = 0; block_index < region->block_count;
         ++block_index) {
      const loom_block_t* block = loom_region_const_block(region, block_index);
      if (!loom_low_representation_count_add(block->arg_count, value_count)) {
        return false;
      }
      const loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        if (!loom_low_representation_count_op(child_op, value_count,
                                              descriptor_count)) {
          return false;
        }
      }
    }
  }
  return true;
}

static iree_status_t loom_low_representation_project_register_type(
    loom_module_t* module,
    const loom_low_representation_projection_t* projection,
    loom_type_t source_type, loom_type_t* out_target_type) {
  *out_target_type = source_type;
  if (!loom_type_is_register(source_type) ||
      projection->source_descriptor_set == projection->target_descriptor_set) {
    return iree_ok_status();
  }

  const uint64_t source_descriptor_set_id =
      loom_low_register_type_descriptor_set_stable_id(source_type);
  if (source_descriptor_set_id !=
      projection->source_descriptor_set->stable_id) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low function register type belongs to descriptor set 0x%016" PRIx64
        ", expected 0x%016" PRIx64,
        source_descriptor_set_id, projection->source_descriptor_set->stable_id);
  }

  const uint16_t source_class_id = loom_low_register_type_class_id(source_type);
  if (source_class_id >= projection->source_descriptor_set->reg_class_count) {
    const iree_string_view_t source_set_name = loom_low_descriptor_set_string(
        projection->source_descriptor_set,
        projection->source_descriptor_set->key_string_ref);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low function register class %u is outside descriptor set '%.*s'",
        (unsigned)source_class_id, (int)source_set_name.size,
        source_set_name.data);
  }

  const loom_low_reg_class_t* source_class =
      &projection->source_descriptor_set->reg_classes[source_class_id];
  const iree_string_view_t source_class_name = loom_low_descriptor_set_string(
      projection->source_descriptor_set, source_class->name_string_ref);
  uint16_t target_class_id = LOOM_LOW_REG_CLASS_NONE;
  if (!loom_low_descriptor_set_lookup_register_class(
          projection->target_descriptor_set, source_class_name,
          &target_class_id, /*out_descriptor_register_class=*/NULL)) {
    const iree_string_view_t source_set_name = loom_low_descriptor_set_string(
        projection->source_descriptor_set,
        projection->source_descriptor_set->key_string_ref);
    const iree_string_view_t target_set_name = loom_low_descriptor_set_string(
        projection->target_descriptor_set,
        projection->target_descriptor_set->key_string_ref);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low representation '%.*s' claims compatibility with '%.*s' but "
        "register class '%.*s' is absent",
        (int)source_set_name.size, source_set_name.data,
        (int)target_set_name.size, target_set_name.data,
        (int)source_class_name.size, source_class_name.data);
  }

  const uint32_t unit_count = loom_low_register_type_unit_count(source_type);
  const loom_type_t* value_type = loom_type_register_value_type(source_type);
  if (value_type != NULL) {
    return loom_low_build_typed_register_type(
        module, projection->target_descriptor_set, target_class_id, unit_count,
        *value_type, out_target_type);
  }
  return loom_low_build_register_type(projection->target_descriptor_set,
                                      target_class_id, unit_count,
                                      out_target_type);
}

static iree_status_t loom_low_representation_plan_value(
    loom_module_t* module,
    const loom_low_representation_projection_t* projection,
    loom_value_id_t value, loom_low_representation_projection_plan_t* plan) {
  if (plan->value_count >= plan->value_capacity) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "low representation value plan overflow");
  }
  loom_type_t target_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_representation_project_register_type(
      module, projection, loom_module_value_type(module, value), &target_type));
  plan->value_updates[plan->value_count++] =
      (loom_low_representation_value_update_t){
          .value = value,
          .type = target_type,
      };
  return iree_ok_status();
}

static iree_status_t loom_low_representation_plan_descriptor(
    loom_op_t* op, uint32_t source_ordinal,
    const loom_low_representation_projection_t* projection,
    loom_low_representation_projection_plan_t* plan) {
  if (plan->descriptor_count >= plan->descriptor_capacity) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "low representation descriptor plan overflow");
  }
  const loom_low_descriptor_t* source_descriptor =
      loom_low_descriptor_set_descriptor_at(projection->source_descriptor_set,
                                            source_ordinal);
  if (source_descriptor == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low function descriptor ordinal %u is outside its representation "
        "contract",
        source_ordinal);
  }
  const uint32_t target_ordinal =
      loom_low_descriptor_set_lookup_descriptor_by_id(
          projection->target_descriptor_set, source_descriptor->stable_id);
  if (target_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    const iree_string_view_t descriptor_key = loom_low_descriptor_set_string(
        projection->source_descriptor_set, source_descriptor->key_string_ref);
    const iree_string_view_t source_set_name = loom_low_descriptor_set_string(
        projection->source_descriptor_set,
        projection->source_descriptor_set->key_string_ref);
    const iree_string_view_t target_set_name = loom_low_descriptor_set_string(
        projection->target_descriptor_set,
        projection->target_descriptor_set->key_string_ref);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low representation '%.*s' claims compatibility with '%.*s' but "
        "descriptor '%.*s' is absent",
        (int)source_set_name.size, source_set_name.data,
        (int)target_set_name.size, target_set_name.data,
        (int)descriptor_key.size, descriptor_key.data);
  }
  const loom_low_descriptor_t* target_descriptor =
      loom_low_descriptor_set_descriptor_at(projection->target_descriptor_set,
                                            target_ordinal);
  if (target_descriptor == NULL) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "target descriptor lookup returned an invalid ordinal %u",
        target_ordinal);
  }
  loom_memory_space_mask_t write_spaces = 0;
  const loom_trait_flags_t traits = loom_low_descriptor_effective_traits(
      projection->target_descriptor_set, target_descriptor, &write_spaces);
  plan->descriptor_updates[plan->descriptor_count++] =
      (loom_low_representation_descriptor_update_t){
          .op = op,
          .source_ordinal = source_ordinal,
          .ordinal = target_ordinal,
          .effective_traits = traits,
          .intrinsic_write_spaces = write_spaces,
      };
  return iree_ok_status();
}

static iree_status_t loom_low_representation_plan_op(
    loom_module_t* module, loom_op_t* op,
    const loom_low_representation_projection_t* projection,
    loom_low_representation_projection_plan_t* plan) {
  // Bodyless argument definitions occupy the declaration's operand tuple.
  // Definition arguments are reached through their entry block below.
  if (loom_low_func_decl_isa(op)) {
    const loom_value_slice_t arguments = loom_low_func_decl_args(op);
    for (uint16_t i = 0; i < arguments.count; ++i) {
      IREE_RETURN_IF_ERROR(loom_low_representation_plan_value(
          module, projection, arguments.values[i], plan));
    }
  }
  const loom_low_descriptor_packet_kind_t packet_kind =
      loom_low_descriptor_packet_kind(op);
  if (packet_kind != LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    IREE_RETURN_IF_ERROR(loom_low_representation_plan_descriptor(
        op, loom_low_descriptor_packet_ordinal(op, packet_kind), projection,
        plan));
  }

  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_representation_plan_value(module, projection,
                                                            results[i], plan));
  }

  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    loom_region_t* region = regions[region_index];
    if (!region) {
      continue;
    }
    for (uint16_t block_index = 0; block_index < region->block_count;
         ++block_index) {
      loom_block_t* block = loom_region_block(region, block_index);
      for (uint16_t i = 0; i < block->arg_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_low_representation_plan_value(
            module, projection, loom_block_arg_id(block, i), plan));
      }
      loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        IREE_RETURN_IF_ERROR(loom_low_representation_plan_op(module, child_op,
                                                             projection, plan));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_representation_allocate_plan(
    const loom_op_t* function_op, iree_arena_allocator_t* scratch_arena,
    loom_low_representation_projection_plan_t* out_plan) {
  *out_plan = (loom_low_representation_projection_plan_t){0};
  if (!loom_low_representation_count_op(function_op, &out_plan->value_capacity,
                                        &out_plan->descriptor_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "low function representation projection size "
                            "overflow");
  }
  if (out_plan->value_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, out_plan->value_capacity,
        sizeof(*out_plan->value_updates), (void**)&out_plan->value_updates));
  }
  if (out_plan->descriptor_capacity != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, out_plan->descriptor_capacity,
                                  sizeof(*out_plan->descriptor_updates),
                                  (void**)&out_plan->descriptor_updates));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_representation_plan_identity(
    const loom_module_t* module, loom_func_like_t function,
    iree_arena_allocator_t* arena,
    loom_low_representation_projection_plan_t* plan) {
  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  const loom_value_id_t* results = loom_op_const_results(function.op);
  *plan = (loom_low_representation_projection_plan_t){
      .result_offset = argument_count,
      .value_capacity =
          (iree_host_size_t)argument_count + function.op->result_count,
  };
  if (plan->value_capacity == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, plan->value_capacity,
                                                 sizeof(*plan->value_updates),
                                                 (void**)&plan->value_updates));
  for (iree_host_size_t i = 0; i < plan->value_capacity; ++i) {
    const loom_value_id_t value =
        i < argument_count ? arguments[i] : results[i - argument_count];
    plan->value_updates[plan->value_count++] =
        (loom_low_representation_value_update_t){
            .value = value,
            .type = loom_module_value_type(module, value),
        };
  }
  return iree_ok_status();
}

iree_status_t loom_low_apply_function_representation(
    loom_module_t* module,
    const loom_low_representation_projection_plan_t* plan, bool* out_changed) {
  *out_changed = false;
  for (iree_host_size_t i = 0; i < plan->value_count; ++i) {
    const loom_low_representation_value_update_t* update =
        &plan->value_updates[i];
    if (loom_type_equal(loom_module_value_type(module, update->value),
                        update->type)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_module_set_value_type(module, update->value, update->type));
    *out_changed = true;
  }
  for (iree_host_size_t i = 0; i < plan->descriptor_count; ++i) {
    const loom_low_representation_descriptor_update_t* update =
        &plan->descriptor_updates[i];
    if (update->source_ordinal != update->ordinal) {
      const loom_attribute_t descriptor =
          loom_attr_scoped_enum(update->ordinal);
      switch (loom_low_descriptor_packet_kind(update->op)) {
        case LOOM_LOW_DESCRIPTOR_PACKET_OP: {
          IREE_RETURN_IF_ERROR(
              loom_low_op_set_descriptor(module, update->op, descriptor));
          break;
        }
        case LOOM_LOW_DESCRIPTOR_PACKET_CONST: {
          IREE_RETURN_IF_ERROR(
              loom_low_const_set_descriptor(module, update->op, descriptor));
          break;
        }
        case LOOM_LOW_DESCRIPTOR_PACKET_COND_BR: {
          IREE_RETURN_IF_ERROR(loom_low_op_cond_br_set_descriptor(
              module, update->op, descriptor));
          break;
        }
        case LOOM_LOW_DESCRIPTOR_PACKET_NONE:
        default:
          IREE_ASSERT_UNREACHABLE("descriptor update is not a Low packet");
          IREE_BUILTIN_UNREACHABLE();
      }
      *out_changed = true;
    }
    if (update->op->traits != update->effective_traits ||
        update->op->intrinsic_write_spaces != update->intrinsic_write_spaces) {
      loom_module_drop_op_summaries(module, update->op);
      update->op->traits = update->effective_traits;
      update->op->intrinsic_write_spaces = update->intrinsic_write_spaces;
      loom_module_record_op_summaries(module, update->op);
      *out_changed = true;
    }
  }
  loom_attribute_t* representation_attr = &loom_op_attrs(
      plan->function.op)[plan->function.vtable->repr_contract_attr_index];
  if (loom_attr_as_string_id(*representation_attr) !=
      plan->target_descriptor_set_key) {
    *representation_attr = loom_attr_string(plan->target_descriptor_set_key);
    *out_changed = true;
  }
  return iree_ok_status();
}

iree_status_t loom_low_plan_function_representation(
    loom_module_t* module, loom_func_like_t function,
    const loom_target_facts_t* target_facts,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    const loom_low_representation_projection_plan_t** out_plan) {
  *out_plan = NULL;
  if (!module || !loom_func_like_isa(function) || !target_facts ||
      !descriptor_registry || !arena) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low function representation projection requires a function, target "
        "facts, descriptor registry, and scratch arena");
  }
  if (!loom_low_func_def_isa(function.op) &&
      !loom_low_func_decl_isa(function.op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "representation projection requires low.func.def "
                            "or low.func.decl");
  }

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(module->arena.block_pool, &scratch_arena);
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, &scratch_arena);
  loom_low_resolved_target_t source_target = {0};
  iree_status_t status = loom_low_resolve_function_target(
      module, &symbol_facts, function.op, target_facts, descriptor_registry,
      emitter, &source_target);
  iree_arena_deinitialize(&scratch_arena);
  IREE_RETURN_IF_ERROR(status);
  if (source_target.descriptor_set == NULL) {
    return iree_ok_status();
  }

  const loom_target_bundle_t* bundle = loom_target_facts_bundle(target_facts);
  if (!bundle || !bundle->config) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low function target facts have no target configuration");
  }
  loom_target_low_source_binding_t target_binding = {0};
  IREE_RETURN_IF_ERROR(loom_target_low_source_binding_select(
      descriptor_registry, bundle, &target_binding));
  const loom_low_descriptor_set_t* target_descriptor_set =
      target_binding.descriptor_set;

  const loom_low_representation_projection_t projection = {
      .source_descriptor_set = source_target.descriptor_set,
      .target_descriptor_set = target_descriptor_set,
  };
  loom_low_representation_projection_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*plan), (void**)&plan));
  if (projection.source_descriptor_set == projection.target_descriptor_set) {
    // Call planning still needs the signature and selected target facts. The
    // verified body already has exactly these register and descriptor meanings;
    // retaining identity updates for every packet would only duplicate them.
    IREE_RETURN_IF_ERROR(
        loom_low_representation_plan_identity(module, function, arena, plan));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_low_representation_allocate_plan(function.op, arena, plan));
    if (loom_low_func_decl_isa(function.op)) {
      plan->result_offset = function.op->operand_count;
    } else {
      plan->argument_offset = function.op->result_count;
    }
    IREE_RETURN_IF_ERROR(loom_low_representation_plan_op(module, function.op,
                                                         &projection, plan));
  }
  plan->function = function;
  plan->descriptor_set = target_descriptor_set;
  plan->target_facts = target_facts;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, bundle->config->contract_set.spelling,
                                &plan->target_descriptor_set_key));
  *out_plan = plan;
  return iree_ok_status();
}
