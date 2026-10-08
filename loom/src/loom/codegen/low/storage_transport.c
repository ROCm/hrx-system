// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/storage_transport.h"

#include <string.h>

#include "loom/codegen/low/function_model.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"

typedef struct loom_low_storage_transport_builder_t {
  // Immutable source, type, and storage facts for this planning generation.
  const loom_low_function_model_t* model;
  // Synchronous storage spaces supported by the final transport consumer.
  loom_low_storage_space_set_t spaces;
  // Arena owning the completed plan.
  iree_arena_allocator_t* arena;
  // Mutable binding rows until the plan is published.
  loom_low_storage_transport_binding_t* bindings;
  // Number of initialized rows.
  iree_host_size_t count;
  // Allocated row capacity.
  iree_host_size_t capacity;
  // Direct index populated with each binding.
  uint32_t* indices;
  // Retained physical effects at each boundary.
  loom_low_storage_transport_effect_t* effects;
  // Number of affected boundaries.
  iree_host_size_t effect_count;
  // Allocated boundary capacity.
  iree_host_size_t effect_capacity;
} loom_low_storage_transport_builder_t;

// A physical register value has no embedded dynamic type operands. Attribute
// users still forbid moving its representation out of registers.
static bool loom_low_storage_transport_has_only_user(const loom_value_t* value,
                                                     const loom_op_t* user) {
  if (!value->use_count || loom_value_has_attribute_uses(value)) {
    return false;
  }
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    if (loom_use_user_op(uses[i]) != user) {
      return false;
    }
  }
  return true;
}

// Entry/result stores may move earlier only for a private root with exactly
// this writer. Views, addresses, and other consumers require the ordinary
// scheduled store; their alias/effect contract is deliberately unchanged.
static bool loom_low_storage_transport_is_unique_store(
    const loom_module_t* module, const loom_op_t* store) {
  const loom_value_t* storage =
      loom_module_value(module, loom_low_spill_storage(store));
  if (loom_value_is_block_arg(storage) ||
      !loom_low_storage_reserve_isa(loom_value_def_op(storage)) ||
      loom_value_has_attribute_uses(storage)) {
    return false;
  }
  const loom_use_t* uses = loom_value_uses(storage);
  for (uint32_t i = 0; i < storage->use_count; ++i) {
    const loom_op_t* user = loom_use_user_op(uses[i]);
    if (user != store && !loom_low_reload_isa(user)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_low_storage_transport_append(
    loom_low_storage_transport_builder_t* builder, const loom_op_t* traffic,
    const loom_op_t* boundary, uint32_t boundary_ordinal) {
  const bool is_store = loom_low_spill_isa(traffic);
  const loom_value_id_t value_id = is_store ? loom_low_spill_value(traffic)
                                            : loom_low_reload_result(traffic);
  const loom_module_t* module = builder->model->context.module;
  const loom_value_t* value = loom_module_value(module, value_id);
  // The callable conventions consuming this plan use scalar register cells.
  if (loom_low_register_type_unit_count(value->type) != 1 ||
      !loom_low_storage_transport_has_only_user(
          value, is_store ? traffic : boundary) ||
      (is_store &&
       !loom_low_storage_transport_is_unique_store(module, traffic))) {
    return iree_ok_status();
  }
  const loom_low_storage_layout_t* layout =
      &builder->model->context.requirements.storage_layout;
  loom_low_storage_layout_reference_t reference;
  loom_low_storage_layout_lookup_reference(
      &layout->index, layout->records,
      is_store ? loom_low_spill_storage(traffic)
               : loom_low_reload_storage(traffic),
      &reference);
  if (!loom_low_storage_space_set_contains(builder->spaces,
                                           reference.reservation.space)) {
    return iree_ok_status();
  }
  const uint16_t register_class = loom_low_register_type_class_id(value->type);
  const uint64_t relative_offset = is_store ? loom_low_spill_offset(traffic)
                                            : loom_low_reload_offset(traffic);
  // Generic Low verification checks the starting offset. Only complete
  // physical cells may join a boundary; ordinary emission retains the target's
  // authored-span diagnostic for an access that does not fit.
  const uint32_t cell_bytes =
      builder->model->context.target.descriptor_set->reg_classes[register_class]
          .alloc_unit_bits /
      8;
  if (cell_bytes > reference.byte_length - relative_offset) {
    return iree_ok_status();
  }
  if (!builder->indices) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->model->context.value_domain.value_count,
        sizeof(*builder->indices), (void**)&builder->indices));
    memset(builder->indices, 0xFF,
           builder->model->context.value_domain.value_count *
               sizeof(*builder->indices));
  }
  if (builder->count == builder->capacity) {
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(builder->arena, builder->count,
                              builder->count + 1, sizeof(*builder->bindings),
                              &builder->capacity, (void**)&builder->bindings));
  }
  const loom_value_ordinal_t ordinal = loom_local_value_domain_ordinal(
      &builder->model->context.value_domain, value_id);
  builder->indices[ordinal] = (uint32_t)builder->count;
  builder->bindings[builder->count++] = (loom_low_storage_transport_binding_t){
      .space = reference.reservation.space,
      .register_class = register_class,
      .byte_offset = reference.reservation.byte_offset + reference.byte_offset +
                     relative_offset,
  };
  if (boundary_ordinal != UINT32_MAX) {
    // A boundary's pre-call reads and post-call stores are consecutive in
    // this traversal. Merge them once without a function-sized node array.
    loom_low_storage_transport_effect_t* effect =
        builder->effect_count ? &builder->effects[builder->effect_count - 1]
                              : NULL;
    if (!effect || effect->source_ordinal != boundary_ordinal) {
      if (builder->effect_count == builder->effect_capacity) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            builder->arena, builder->effect_count, builder->effect_count + 1,
            sizeof(*builder->effects), &builder->effect_capacity,
            (void**)&builder->effects));
      }
      effect = &builder->effects[builder->effect_count++];
      *effect = (loom_low_storage_transport_effect_t){.source_ordinal =
                                                          boundary_ordinal};
    }
    effect->traits |=
        is_store ? LOOM_TRAIT_WRITES_MEMORY : LOOM_TRAIT_READS_MEMORY;
  }
  return iree_ok_status();
}

iree_status_t loom_low_storage_transport_build(
    const loom_low_function_model_t* model,
    loom_low_storage_space_set_t synchronous_spaces,
    iree_arena_allocator_t* arena,
    const loom_low_storage_transport_t** out_plan) {
  *out_plan = NULL;
  if (!synchronous_spaces ||
      !model->context.requirements.storage_layout.record_count) {
    return iree_ok_status();
  }
  loom_low_storage_transport_builder_t builder = {
      .model = model, .spaces = synchronous_spaces, .arena = arena};
  const loom_block_t* entry =
      loom_region_const_entry_block(model->context.body);
  uint32_t ordinal = 0;
  iree_status_t status = iree_ok_status();
  for (uint16_t b = 0;
       b < model->context.body->block_count && iree_status_is_ok(status); ++b) {
    loom_block_t* block = model->context.body->blocks[b];
    const loom_op_t* read_begin = NULL;
    const loom_op_t* store_boundary =
        block == entry ? model->context.function_op : NULL;
    uint32_t store_boundary_ordinal = UINT32_MAX;
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!iree_status_is_ok(status)) {
        break;
      }
      if (loom_low_reload_isa(op)) {
        if (!read_begin) {
          read_begin = op;
        }
        store_boundary = NULL;
      } else {
        if (loom_low_func_call_isa(op) || loom_low_return_isa(op)) {
          for (const loom_op_t* read = read_begin;
               read && read != op && iree_status_is_ok(status);
               read = read->next_op) {
            status =
                loom_low_storage_transport_append(&builder, read, op, ordinal);
          }
        }
        read_begin = NULL;
        if (store_boundary && loom_low_spill_isa(op)) {
          const loom_value_t* value = loom_module_value(
              model->context.module, loom_low_spill_value(op));
          const bool matches =
              loom_value_is_block_arg(value)
                  ? store_boundary == model->context.function_op &&
                        loom_value_def_block(value) == entry
                  : loom_value_def_op(value) == store_boundary;
          if (matches) {
            status = loom_low_storage_transport_append(
                &builder, op, store_boundary, store_boundary_ordinal);
          } else {
            store_boundary = NULL;
          }
        } else if (loom_low_func_call_isa(op)) {
          store_boundary = op;
          store_boundary_ordinal = ordinal;
        } else if (!loom_low_storage_reserve_isa(op)) {
          store_boundary = NULL;
        }
      }
      ++ordinal;
    }
  }
  if (iree_status_is_ok(status) && builder.count) {
    loom_low_storage_transport_t* plan = NULL;
    status = iree_arena_allocate(arena, sizeof(*plan), (void**)&plan);
    if (iree_status_is_ok(status)) {
      *plan = (loom_low_storage_transport_t){
          .bindings = builder.bindings,
          .binding_count = builder.count,
          .bindings_by_value_ordinal = builder.indices,
          .effects = builder.effects,
          .effect_count = builder.effect_count,
      };
      *out_plan = plan;
    }
  }
  return status;
}
