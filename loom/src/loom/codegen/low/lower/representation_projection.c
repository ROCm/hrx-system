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
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
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
  // Attribute containing the descriptor ordinal.
  uint8_t attr_index;
  // Target representation descriptor ordinal.
  uint32_t ordinal;
} loom_low_representation_descriptor_update_t;

typedef struct loom_low_representation_projection_plan_t {
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
} loom_low_representation_projection_plan_t;

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
  if (loom_low_op_isa(op) || loom_low_const_isa(op)) {
    if (!loom_low_representation_count_add(1, descriptor_count)) return false;
  }
  loom_region_t* const* regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    const loom_region_t* region = regions[region_index];
    if (!region) continue;
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
        projection->source_descriptor_set->key_string_offset);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low function register class %u is outside descriptor set '%.*s'",
        (unsigned)source_class_id, (int)source_set_name.size,
        source_set_name.data);
  }

  const loom_low_reg_class_t* source_class =
      &projection->source_descriptor_set->reg_classes[source_class_id];
  const iree_string_view_t source_class_name = loom_low_descriptor_set_string(
      projection->source_descriptor_set, source_class->name_string_offset);
  uint16_t target_class_id = LOOM_LOW_REG_CLASS_NONE;
  if (!loom_low_descriptor_set_lookup_register_class(
          projection->target_descriptor_set, source_class_name,
          &target_class_id, /*out_descriptor_register_class=*/NULL)) {
    const iree_string_view_t source_set_name = loom_low_descriptor_set_string(
        projection->source_descriptor_set,
        projection->source_descriptor_set->key_string_offset);
    const iree_string_view_t target_set_name = loom_low_descriptor_set_string(
        projection->target_descriptor_set,
        projection->target_descriptor_set->key_string_offset);
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
    loom_op_t* op, uint8_t attr_index,
    const loom_low_representation_projection_t* projection,
    loom_low_representation_projection_plan_t* plan) {
  if (plan->descriptor_count >= plan->descriptor_capacity) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "low representation descriptor plan overflow");
  }
  const uint32_t source_ordinal =
      loom_attr_as_scoped_enum(loom_op_const_attrs(op)[attr_index]);
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
  const iree_string_view_t descriptor_key = loom_low_descriptor_set_string(
      projection->source_descriptor_set, source_descriptor->key_string_offset);
  const uint32_t target_ordinal = loom_low_descriptor_set_lookup_descriptor(
      projection->target_descriptor_set, descriptor_key);
  if (target_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    const iree_string_view_t source_set_name = loom_low_descriptor_set_string(
        projection->source_descriptor_set,
        projection->source_descriptor_set->key_string_offset);
    const iree_string_view_t target_set_name = loom_low_descriptor_set_string(
        projection->target_descriptor_set,
        projection->target_descriptor_set->key_string_offset);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low representation '%.*s' claims compatibility with '%.*s' but "
        "descriptor '%.*s' is absent",
        (int)source_set_name.size, source_set_name.data,
        (int)target_set_name.size, target_set_name.data,
        (int)descriptor_key.size, descriptor_key.data);
  }
  plan->descriptor_updates[plan->descriptor_count++] =
      (loom_low_representation_descriptor_update_t){
          .op = op,
          .attr_index = attr_index,
          .ordinal = target_ordinal,
      };
  return iree_ok_status();
}

static iree_status_t loom_low_representation_plan_op(
    loom_module_t* module, loom_op_t* op,
    const loom_low_representation_projection_t* projection,
    loom_low_representation_projection_plan_t* plan) {
  if (loom_low_op_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_low_representation_plan_descriptor(
        op, loom_low_op_descriptor_ATTR_INDEX, projection, plan));
  } else if (loom_low_const_isa(op)) {
    IREE_RETURN_IF_ERROR(loom_low_representation_plan_descriptor(
        op, loom_low_const_descriptor_ATTR_INDEX, projection, plan));
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
    if (!region) continue;
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

static iree_status_t loom_low_representation_apply_plan(
    loom_module_t* module,
    const loom_low_representation_projection_plan_t* plan, bool* out_changed) {
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
    loom_attribute_t* attr = &loom_op_attrs(update->op)[update->attr_index];
    if (loom_attr_as_scoped_enum(*attr) == update->ordinal) continue;
    *attr = loom_attr_scoped_enum(update->ordinal);
    *out_changed = true;
  }
  return iree_ok_status();
}

iree_status_t loom_low_project_function_representation(
    loom_module_t* module, loom_func_like_t function,
    const loom_target_facts_t* target_facts,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* scratch_arena,
    bool* out_valid, bool* out_changed) {
  *out_valid = false;
  *out_changed = false;
  if (!module || !loom_func_like_isa(function) || !target_facts ||
      !descriptor_registry || !scratch_arena) {
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

  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, scratch_arena);
  loom_low_resolved_target_t source_target = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      module, &symbol_facts, function.op, target_facts, descriptor_registry,
      emitter, &source_target));
  if (source_target.descriptor_set == NULL) {
    return iree_ok_status();
  }

  const loom_target_bundle_t* bundle = loom_target_facts_bundle(target_facts);
  if (!bundle || !bundle->config) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low function target facts have no target configuration");
  }
  const loom_low_descriptor_set_t* target_descriptor_set =
      loom_low_descriptor_registry_lookup(descriptor_registry,
                                          bundle->config->contract_set_key);
  if (target_descriptor_set == NULL) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "target bundle '%.*s' selected low descriptor set '%.*s' that is not "
        "linked",
        (int)bundle->name.size, bundle->name.data,
        (int)bundle->config->contract_set_key.size,
        bundle->config->contract_set_key.data);
  }

  const loom_low_representation_projection_t projection = {
      .source_descriptor_set = source_target.descriptor_set,
      .target_descriptor_set = target_descriptor_set,
  };
  loom_low_representation_projection_plan_t plan = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_representation_allocate_plan(function.op, scratch_arena, &plan));
  IREE_RETURN_IF_ERROR(
      loom_low_representation_plan_op(module, function.op, &projection, &plan));

  loom_string_id_t target_descriptor_set_key = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, bundle->config->contract_set_key, &target_descriptor_set_key));
  IREE_RETURN_IF_ERROR(
      loom_low_representation_apply_plan(module, &plan, out_changed));
  loom_attribute_t* representation_attr =
      &loom_op_attrs(function.op)[function.vtable->repr_contract_attr_index];
  if (loom_attr_as_string_id(*representation_attr) !=
      target_descriptor_set_key) {
    *representation_attr = loom_attr_string(target_descriptor_set_key);
    *out_changed = true;
  }
  *out_valid = true;
  return iree_ok_status();
}
