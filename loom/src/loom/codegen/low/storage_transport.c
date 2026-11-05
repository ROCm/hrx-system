// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/storage_transport.h"

#include <string.h>

#include "loom/analysis/read_motion.h"
#include "loom/codegen/low/function_model.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/low/storage.h"
#include "loom/target/registers.h"
#include "loom/util/adaptive_sort.h"

typedef struct loom_low_storage_transport_read_t {
  // Load whose source extent participates in read/write ordering.
  const loom_op_t* op;
  // Resolved cell and ownership transfer, shared with binding publication.
  loom_low_storage_transport_binding_t binding;
  // Exclusive end of the accessed extent.
  uint64_t byte_end;
  // Order within the current candidate read segment.
  uint32_t ordinal;
  // Whether this occurrence can join its sole boundary consumer.
  bool foldable;
} loom_low_storage_transport_read_t;

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
  // Reusable sorting workspace, released when the plan is published.
  iree_arena_allocator_t scratch_arena;
  // Extents of reads in a segment containing a destructive load.
  loom_low_storage_transport_read_t* reads;
  // Number of read records available in reusable scratch.
  iree_host_size_t read_capacity;
  // Whether any published binding consumes a backing cell.
  bool has_moving_reads;
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
      loom_module_value(module, loom_low_storage_store_storage(store));
  if (loom_value_is_block_arg(storage) ||
      !loom_low_storage_reserve_isa(loom_value_def_op(storage)) ||
      loom_value_has_attribute_uses(storage)) {
    return false;
  }
  const loom_use_t* uses = loom_value_uses(storage);
  for (uint32_t i = 0; i < storage->use_count; ++i) {
    const loom_op_t* user = loom_use_user_op(uses[i]);
    if (user != store && !loom_low_storage_load_isa(user)) {
      return false;
    }
  }
  return true;
}

static bool loom_low_storage_transport_resolve(
    loom_low_storage_transport_builder_t* builder, const loom_op_t* traffic,
    const loom_op_t* boundary,
    loom_low_storage_transport_binding_t* out_binding,
    uint64_t* out_byte_length) {
  const bool is_store = loom_low_storage_store_isa(traffic);
  const loom_value_id_t value_id = is_store
                                       ? loom_low_storage_store_value(traffic)
                                       : loom_low_storage_load_result(traffic);
  const loom_module_t* module = builder->model->context.module;
  const loom_value_t* value = loom_module_value(module, value_id);
  const loom_low_storage_layout_t* layout =
      &builder->model->context.requirements.storage_layout;
  loom_low_storage_layout_reference_t reference;
  loom_low_storage_layout_lookup_reference(
      &layout->index, layout->records,
      is_store ? loom_low_storage_store_storage(traffic)
               : loom_low_storage_load_storage(traffic),
      &reference);
  const uint16_t register_class = loom_low_register_type_class_id(value->type);
  const uint32_t cell_bytes =
      builder->model->context.target.descriptor_set->reg_classes[register_class]
          .alloc_unit_bits /
      8;
  const uint64_t relative_offset = loom_low_storage_transfer_offset(traffic);
  *out_binding = (loom_low_storage_transport_binding_t){
      .space = reference.reservation.space,
      .moves_source = !is_store && loom_low_storage_transfer_moves(traffic),
      .register_class = register_class,
      .byte_offset = reference.reservation.byte_offset + reference.byte_offset +
                     relative_offset,
  };
  *out_byte_length =
      (uint64_t)loom_low_register_type_unit_count(value->type) * cell_bytes;
  // Callable transport uses complete scalar cells. An access that does not
  // fit retains ordinary emission and its target-owned span diagnostic.
  return loom_low_register_type_unit_count(value->type) == 1 &&
         cell_bytes <= reference.byte_length - relative_offset &&
         loom_low_storage_space_set_contains(builder->spaces,
                                             reference.reservation.space) &&
         loom_low_storage_transport_has_only_user(
             value, is_store ? traffic : boundary) &&
         (!is_store ||
          loom_low_storage_transport_is_unique_store(module, traffic));
}

static iree_status_t loom_low_storage_transport_append_binding(
    loom_low_storage_transport_builder_t* builder, loom_value_id_t value_id,
    bool is_store, const loom_low_storage_transport_binding_t* binding,
    uint32_t boundary_ordinal) {
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
  builder->has_moving_reads |= binding->moves_source;
  builder->bindings[builder->count] = *binding;
  builder->bindings[builder->count].cell_index = (uint32_t)builder->count;
  ++builder->count;
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
    if (binding->moves_source) {
      effect->traits |= LOOM_TRAIT_WRITES_MEMORY;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_storage_transport_append(
    loom_low_storage_transport_builder_t* builder, const loom_op_t* traffic,
    const loom_op_t* boundary, uint32_t boundary_ordinal) {
  loom_low_storage_transport_binding_t binding;
  uint64_t byte_length = 0;
  if (!loom_low_storage_transport_resolve(builder, traffic, boundary, &binding,
                                          &byte_length)) {
    return iree_ok_status();
  }
  const bool is_store = loom_low_storage_store_isa(traffic);
  return loom_low_storage_transport_append_binding(
      builder,
      is_store ? loom_low_storage_store_value(traffic)
               : loom_low_storage_load_result(traffic),
      is_store, &binding, boundary_ordinal);
}

static bool loom_low_storage_transport_read_less(
    const loom_low_storage_transport_read_t* lhs,
    const loom_low_storage_transport_read_t* rhs) {
  if (lhs->binding.space != rhs->binding.space) {
    return lhs->binding.space < rhs->binding.space;
  }
  if (lhs->binding.byte_offset != rhs->binding.byte_offset) {
    return lhs->binding.byte_offset < rhs->binding.byte_offset;
  }
  return lhs->binding.register_class != rhs->binding.register_class
             ? lhs->binding.register_class < rhs->binding.register_class
             : lhs->ordinal < rhs->ordinal;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_low_storage_transport_sort_reads,
                          loom_low_storage_transport_read_t,
                          loom_low_storage_transport_read_less)

// A moving read can join a parallel boundary only when no later read observes
// its cleared cell. Flattened extents include aliases through storage views.
// The last such conflict ends the candidate segment; its prefix remains in
// source order. All surviving reads observe the same pre-boundary cell state.
static iree_status_t loom_low_storage_transport_append_reads(
    loom_low_storage_transport_builder_t* builder, const loom_op_t* begin,
    const loom_op_t* boundary, uint32_t boundary_ordinal) {
  iree_host_size_t count = 0;
  uint32_t prefix_count = 0;
  for (const loom_op_t* op = begin; op != boundary; op = op->next_op) {
    if (!loom_low_storage_load_isa(op)) {
      continue;
    }
    if (count == builder->read_capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          &builder->scratch_arena, count, count + 1, sizeof(*builder->reads),
          &builder->read_capacity, (void**)&builder->reads));
    }
    loom_low_storage_transport_read_t* read = &builder->reads[count];
    uint64_t byte_length = 0;
    read->foldable = loom_low_storage_transport_resolve(
        builder, op, boundary, &read->binding, &byte_length);
    read->op = op;
    read->ordinal = (uint32_t)count;
    read->byte_end = read->binding.byte_offset + byte_length;
    if (read->binding.moves_source && !read->foldable) {
      // An explicit destructive load is a write barrier for earlier copies.
      prefix_count = (uint32_t)count + 1;
    }
    ++count;
  }
  loom_low_storage_transport_sort_reads(builder->reads, count);
  for (iree_host_size_t start = 0; start < count;) {
    iree_host_size_t end = start + 1;
    uint64_t byte_end = builder->reads[start].byte_end;
    uint32_t last_ordinal = builder->reads[start].ordinal;
    while (end < count &&
           builder->reads[end].binding.space ==
               builder->reads[start].binding.space &&
           builder->reads[end].binding.byte_offset < byte_end) {
      byte_end = iree_max(byte_end, builder->reads[end].byte_end);
      last_ordinal = iree_max(last_ordinal, builder->reads[end].ordinal);
      ++end;
    }
    for (iree_host_size_t i = start; i < end; ++i) {
      const loom_low_storage_transport_read_t* read = &builder->reads[i];
      if (loom_low_storage_transfer_moves(read->op) &&
          read->ordinal < last_ordinal && read->ordinal >= prefix_count) {
        prefix_count = read->ordinal + 1;
      }
    }
    start = end;
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t start = 0;
       start < count && iree_status_is_ok(status);) {
    iree_host_size_t end = start;
    bool consumes_cell = false;
    while (end < count &&
           builder->reads[end].binding.space ==
               builder->reads[start].binding.space &&
           builder->reads[end].binding.register_class ==
               builder->reads[start].binding.register_class &&
           builder->reads[end].binding.byte_offset ==
               builder->reads[start].binding.byte_offset) {
      const loom_low_storage_transport_read_t* read = &builder->reads[end++];
      consumes_cell |= read->ordinal >= prefix_count && read->foldable &&
                       read->binding.moves_source;
    }
    // Consumption belongs to the boundary's cell, independently of which
    // aliased read the parallel-move solver schedules last.
    for (iree_host_size_t i = start; i < end && iree_status_is_ok(status);
         ++i) {
      loom_low_storage_transport_read_t* read = &builder->reads[i];
      if (read->ordinal >= prefix_count && read->foldable) {
        read->binding.moves_source = consumes_cell;
        status = loom_low_storage_transport_append_binding(
            builder, loom_low_storage_load_result(read->op), false,
            &read->binding, boundary_ordinal);
      }
    }
    start = end;
  }
  return status;
}

static bool loom_low_storage_transport_cell_less(
    const loom_low_storage_transport_binding_t* bindings, const uint32_t* lhs,
    const uint32_t* rhs) {
  const loom_low_storage_transport_binding_t* a = &bindings[*lhs];
  const loom_low_storage_transport_binding_t* b = &bindings[*rhs];
  if (a->space != b->space) {
    return a->space < b->space;
  }
  if (a->register_class != b->register_class) {
    return a->register_class < b->register_class;
  }
  return a->byte_offset != b->byte_offset ? a->byte_offset < b->byte_offset
                                          : *lhs < *rhs;
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT(
    loom_low_storage_transport_sort_cells, uint32_t,
    const loom_low_storage_transport_binding_t*,
    loom_low_storage_transport_cell_less)

static iree_status_t loom_low_storage_transport_identify_cells(
    loom_low_storage_transport_builder_t* builder) {
  if (!builder->has_moving_reads) {
    return iree_ok_status();
  }
  uint32_t* indices = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&builder->scratch_arena, builder->count,
                                sizeof(*indices), (void**)&indices));
  for (uint32_t i = 0; i < builder->count; ++i) {
    indices[i] = i;
  }
  loom_low_storage_transport_sort_cells(builder->bindings, indices,
                                        builder->count);
  for (iree_host_size_t i = 1; i < builder->count; ++i) {
    const loom_low_storage_transport_binding_t* previous =
        &builder->bindings[indices[i - 1]];
    loom_low_storage_transport_binding_t* current =
        &builder->bindings[indices[i]];
    if (current->space == previous->space &&
        current->register_class == previous->register_class &&
        current->byte_offset == previous->byte_offset) {
      current->cell_index = previous->cell_index;
    }
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
  iree_arena_initialize(arena->block_pool, &builder.scratch_arena);
  const loom_block_t* entry =
      loom_region_const_entry_block(model->context.body);
  uint32_t ordinal = 0;
  iree_status_t status = iree_ok_status();
  for (uint16_t b = 0;
       b < model->context.body->block_count && iree_status_is_ok(status); ++b) {
    loom_block_t* block = model->context.body->blocks[b];
    // First reload in the current read-barrier segment. Segments end at their
    // first boundary, so the boundary scans below remain linear in block size.
    const loom_op_t* read_begin = NULL;
    bool has_moving_read = false;
    const loom_op_t* store_boundary =
        block == entry ? model->context.function_op : NULL;
    uint32_t store_boundary_ordinal = UINT32_MAX;
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!iree_status_is_ok(status)) {
        break;
      }
      if (loom_low_storage_load_isa(op)) {
        if (!read_begin) {
          read_begin = op;
        }
        has_moving_read |= loom_low_storage_transfer_moves(op);
        store_boundary = NULL;
      } else {
        const bool is_boundary =
            loom_low_func_call_isa(op) || loom_low_return_isa(op);
        if (is_boundary) {
          if (has_moving_read) {
            status = loom_low_storage_transport_append_reads(
                &builder, read_begin, op, ordinal);
          }
          for (const loom_op_t* read = has_moving_read ? NULL : read_begin;
               read && read != op && iree_status_is_ok(status);
               read = read->next_op) {
            if (loom_low_storage_load_isa(read)) {
              status = loom_low_storage_transport_append(&builder, read, op,
                                                         ordinal);
            }
          }
        }
        if (is_boundary ||
            (read_begin &&
             (!loom_motion_read_can_cross_op(model->context.module, op) ||
              (has_moving_read && loom_traits_may_read(loom_op_effective_traits(
                                      model->context.module, op)))))) {
          read_begin = NULL;
          has_moving_read = false;
        }
        if (store_boundary && loom_low_storage_store_isa(op)) {
          const loom_value_t* value = loom_module_value(
              model->context.module, loom_low_storage_store_value(op));
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
  if (iree_status_is_ok(status)) {
    status = loom_low_storage_transport_identify_cells(&builder);
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
  iree_arena_deinitialize(&builder.scratch_arena);
  return status;
}
