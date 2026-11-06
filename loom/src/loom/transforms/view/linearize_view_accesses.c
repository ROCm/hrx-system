// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/view/linearize_view_accesses.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/intern_table.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ir/structural_hash.h"
#include "loom/ir/types.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cache.h"
#include "loom/ops/encoding/roles.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/index/carrier.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/function_version.h"
#include "loom/util/fact_table.h"
#include "loom/util/walk.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

#define LOOM_LINEARIZE_VIEW_ACCESSES_STATISTICS(V, statistics_type)        \
  V(statistics_type, views_created, "views-created",                       \
    "Number of rank-1 buffer views created.")                              \
  V(statistics_type, loads_linearized, "loads-linearized",                 \
    "Number of scalar view.load operations linearized.")                   \
  V(statistics_type, stores_linearized, "stores-linearized",               \
    "Number of scalar view.store operations linearized.")                  \
  V(statistics_type, vector_loads_linearized, "vector-loads-linearized",   \
    "Number of vector.load operations linearized.")                        \
  V(statistics_type, vector_stores_linearized, "vector-stores-linearized", \
    "Number of vector.store operations linearized.")

LOOM_PASS_STATISTICS_DEFINE(loom_linearize_view_accesses_statistics,
                            loom_linearize_view_accesses_statistics_t,
                            LOOM_LINEARIZE_VIEW_ACCESSES_STATISTICS)

static const loom_pass_info_t loom_linearize_view_accesses_pass_info_storage = {
    .name = IREE_SVL("linearize-view-accesses"),
    .description =
        IREE_SVL("Linearize row-contiguous multi-dimensional view accesses."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_linearize_view_accesses_statistics_layout,
};

const loom_pass_info_t* loom_linearize_view_accesses_pass_info(void) {
  return &loom_linearize_view_accesses_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Worklists
//===----------------------------------------------------------------------===//

typedef struct loom_linearize_view_accesses_op_list_t {
  // Collected memory operations in structural order, ordered within each block.
  loom_op_t** ops;
  // Number of collected operations.
  iree_host_size_t count;
  // Allocated operation pointer capacity.
  iree_host_size_t capacity;
} loom_linearize_view_accesses_op_list_t;

typedef struct loom_linearize_view_accesses_view_map_entry_t {
  // Original multi-dimensional buffer.view result.
  loom_value_id_t original_view;
  // Rank-1 buffer.view result built for the same buffer and byte offset.
  loom_value_id_t linear_view;
} loom_linearize_view_accesses_view_map_entry_t;

typedef struct loom_linearize_view_accesses_view_map_t {
  // Original-view index into stable mapping rows, allocated lazily.
  loom_intern_table_t index;
  // Pass-owned mapping rows in fixed-size pooled chunks.
  loom_segmented_storage_t rows;
} loom_linearize_view_accesses_view_map_t;

typedef struct loom_linearize_view_accesses_collect_context_t {
  // Pass scratch arena used for the collected operation pointer list.
  iree_arena_allocator_t* arena;
  // Collected memory operations.
  loom_linearize_view_accesses_op_list_t* accesses;
} loom_linearize_view_accesses_collect_context_t;

typedef uint8_t loom_linearize_view_accesses_access_kind_t;
enum loom_linearize_view_accesses_access_kind_e {
  LOOM_LINEARIZE_VIEW_ACCESSES_ACCESS_NONE = 0,
  LOOM_LINEARIZE_VIEW_ACCESSES_ACCESS_VIEW_LOAD = 1,
  LOOM_LINEARIZE_VIEW_ACCESSES_ACCESS_VIEW_STORE = 2,
  LOOM_LINEARIZE_VIEW_ACCESSES_ACCESS_VECTOR_LOAD = 3,
  LOOM_LINEARIZE_VIEW_ACCESSES_ACCESS_VECTOR_STORE = 4,
};

// Classifies the exact generated operation schemas this pass can rebuild.
// Collection and rewriting share this one allowlist so admitting a new schema
// requires defining its rewrite behavior at the same ownership boundary.
static loom_linearize_view_accesses_access_kind_t
loom_linearize_view_accesses_access_kind(const loom_op_t* op) {
  switch (op->kind) {
    case LOOM_OP_VIEW_LOAD:
      return LOOM_LINEARIZE_VIEW_ACCESSES_ACCESS_VIEW_LOAD;
    case LOOM_OP_VIEW_STORE:
      return LOOM_LINEARIZE_VIEW_ACCESSES_ACCESS_VIEW_STORE;
    case LOOM_OP_VECTOR_LOAD:
      return LOOM_LINEARIZE_VIEW_ACCESSES_ACCESS_VECTOR_LOAD;
    case LOOM_OP_VECTOR_STORE:
      return LOOM_LINEARIZE_VIEW_ACCESSES_ACCESS_VECTOR_STORE;
    default:
      return LOOM_LINEARIZE_VIEW_ACCESSES_ACCESS_NONE;
  }
}

typedef struct loom_linearize_view_accesses_context_t {
  // Owning pass instance.
  loom_pass_t* pass;
  // Typed statistics storage for the current pass invocation.
  loom_linearize_view_accesses_statistics_t* statistics;
  // Module being transformed.
  loom_module_t* module;
  // Rewriter used for IR mutations.
  loom_rewriter_t* rewriter;
  // Function-scoped facts proving coordinates and resolving SSA layouts.
  const loom_value_fact_table_t* fact_table;
  // Lazily materialized rank-1 views.
  loom_linearize_view_accesses_view_map_t* view_map;
  // Block-local arithmetic emitted by this pass, indexed by result value ID.
  loom_intern_table_t index_ops;
} loom_linearize_view_accesses_context_t;

typedef struct loom_linearize_view_accesses_vector_tile_t {
  // First view axis covered by the vector payload.
  uint8_t first_vector_axis;
  // Number of outer vector axes selecting distinct contiguous memory runs.
  uint8_t row_axis_count;
  // Number of scalar elements in each contiguous memory run.
  int64_t inner_lane_count;
  // Number of row-vector accesses in the tile.
  int64_t row_count;
  // Total scalar element count in the rank-N vector payload.
  int64_t flat_lane_count;
  // Rank-N vector dimensions in logical lane order.
  int64_t vector_dims[LOOM_TYPE_MAX_RANK];
  // Rank-1 vector type used for each memory row.
  loom_type_t row_vector_type;
  // Rank-1 vector type used for flat register assembly/disassembly.
  loom_type_t flat_vector_type;
} loom_linearize_view_accesses_vector_tile_t;

typedef struct loom_linearize_view_accesses_index_term_t {
  // Original coordinate with a proved in-bounds range.
  loom_value_id_t value;
  // Dense element stride for this coordinate.
  int64_t stride;
} loom_linearize_view_accesses_index_term_t;

typedef struct loom_linearize_view_accesses_index_t {
  // Contribution from static and exact-valued coordinates, in elements.
  int64_t static_offset;
  // Number of nonconstant coordinate terms.
  uint8_t term_count;
  // Proved coordinates in innermost-to-outermost axis order.
  loom_linearize_view_accesses_index_term_t terms[LOOM_TYPE_MAX_RANK];
} loom_linearize_view_accesses_index_t;

typedef struct loom_linearize_view_accesses_layout_t {
  // Physical element stride of each logical view axis.
  int64_t strides[LOOM_TYPE_MAX_RANK];
  // Number of elements spanning every address reachable through the view.
  int64_t linear_length;
  // Semantic role of an explicit strided encoding.
  loom_encoding_role_t encoding_role;
  // Exact storage-schema encoding retained after replacing a strided layout.
  uint16_t storage_schema_encoding_id;
  // Number of populated physical strides.
  uint8_t rank;
} loom_linearize_view_accesses_layout_t;

static iree_status_t loom_linearize_view_accesses_op_list_push(
    iree_arena_allocator_t* arena, loom_linearize_view_accesses_op_list_t* list,
    loom_op_t* op) {
  if (list->count >= list->capacity) {
    if (list->capacity == 0) {
      list->capacity = 32;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, list->capacity, sizeof(loom_op_t*), (void**)&list->ops));
    } else {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          arena, list->count, list->count + 1, sizeof(loom_op_t*),
          &list->capacity, (void**)&list->ops));
    }
  }
  list->ops[list->count++] = op;
  return iree_ok_status();
}

#define LOOM_LINEARIZE_VIEW_ROWS_PER_SEGMENT 128u

static loom_linearize_view_accesses_view_map_entry_t*
loom_linearize_view_accesses_view_map_entry(
    loom_linearize_view_accesses_view_map_t* map, uint32_t index) {
  loom_linearize_view_accesses_view_map_entry_t* rows =
      loom_segmented_storage_segment(
          &map->rows, index / LOOM_LINEARIZE_VIEW_ROWS_PER_SEGMENT);
  return &rows[index % LOOM_LINEARIZE_VIEW_ROWS_PER_SEGMENT];
}

typedef struct loom_linearize_view_accesses_view_probe_t {
  // Borrowed mapping whose rows are compared.
  loom_linearize_view_accesses_view_map_t* map;
  // Original view being flattened.
  loom_value_id_t original_view;
} loom_linearize_view_accesses_view_probe_t;

static bool loom_linearize_view_accesses_view_equal(const void* user_data,
                                                    uint32_t index) {
  const loom_linearize_view_accesses_view_probe_t* probe = user_data;
  return loom_linearize_view_accesses_view_map_entry(probe->map, index)
             ->original_view == probe->original_view;
}

static iree_status_t loom_linearize_view_accesses_collect_access(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  loom_linearize_view_accesses_collect_context_t* collect_context =
      (loom_linearize_view_accesses_collect_context_t*)user_data;
  *out_result = LOOM_WALK_CONTINUE;
  if (loom_linearize_view_accesses_access_kind(op) ==
      LOOM_LINEARIZE_VIEW_ACCESSES_ACCESS_NONE) {
    return iree_ok_status();
  }
  return loom_linearize_view_accesses_op_list_push(
      collect_context->arena, collect_context->accesses, op);
}

//===----------------------------------------------------------------------===//
// Linearization
//===----------------------------------------------------------------------===//

static bool loom_linearize_view_accesses_static_view_layout(
    const loom_fact_context_t* fact_context, const loom_module_t* module,
    loom_type_t view_type, loom_linearize_view_accesses_layout_t* out_layout) {
  if (!loom_type_is_view(view_type) || loom_type_rank(view_type) <= 1 ||
      !loom_type_is_all_static(view_type)) {
    return false;
  }

  loom_value_facts_t stride_storage[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {
      0};
  loom_value_fact_address_layout_t layout = {0};
  if (!loom_encoding_query_type_address_layout(
          fact_context, module, view_type, stride_storage,
          IREE_ARRAYSIZE(stride_storage), &layout)) {
    return false;
  }
  if (layout.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE &&
      layout.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED) {
    return false;
  }

  const uint8_t rank = loom_type_rank(view_type);
  out_layout->rank = rank;
  out_layout->encoding_role = LOOM_ENCODING_ROLE_UNKNOWN;
  out_layout->storage_schema_encoding_id = 0;
  if (layout.kind == LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE) {
    int64_t stride = 1;
    for (uint8_t reverse_axis = 0; reverse_axis < rank; ++reverse_axis) {
      const uint8_t axis = rank - reverse_axis - 1;
      const int64_t dimension = loom_type_dim_static_size_at(view_type, axis);
      if (dimension <= 0) {
        return false;
      }
      out_layout->strides[axis] = stride;
      if (!iree_checked_mul_i64(stride, dimension, &stride)) {
        return false;
      }
    }
    out_layout->linear_length = stride;
    return true;
  }

  if (layout.rank != rank || layout.strides == NULL) {
    return false;
  }
  int64_t maximum_offset = 0;
  for (uint8_t axis = 0; axis < rank; ++axis) {
    const int64_t dimension = loom_type_dim_static_size_at(view_type, axis);
    int64_t stride = 0;
    if (dimension <= 0 ||
        !loom_value_facts_as_exact_i64(layout.strides[axis], &stride) ||
        stride < 0) {
      return false;
    }
    int64_t axis_maximum_offset = 0;
    if (!iree_checked_mul_i64(dimension - 1, stride, &axis_maximum_offset) ||
        !iree_checked_add_i64(maximum_offset, axis_maximum_offset,
                              &maximum_offset)) {
      return false;
    }
    out_layout->strides[axis] = stride;
  }
  // A dense rank-1 replacement is profitable when the logical trailing row is
  // physically contiguous. Transposed and lane-strided layouts retain their
  // ranked form for direct source-memory lowering.
  if (out_layout->strides[rank - 1] != 1) {
    return false;
  }
  if (!iree_checked_add_i64(maximum_offset, 1, &out_layout->linear_length)) {
    return false;
  }

  out_layout->encoding_role =
      loom_type_has_static_encoding(view_type)
          ? loom_encoding_static_role(
                module, loom_module_encoding(module, view_type.encoding_id))
          : loom_encoding_value_role(module,
                                     loom_type_encoding_value_id(view_type));
  if (out_layout->encoding_role == LOOM_ENCODING_ROLE_ADDRESS_LAYOUT) {
    return true;
  }
  if (out_layout->encoding_role != LOOM_ENCODING_ROLE_PHYSICAL_STORAGE) {
    return false;
  }

  loom_value_fact_storage_schema_t storage_schema = {0};
  if (!loom_encoding_query_type_storage_schema(fact_context, module, view_type,
                                               &storage_schema) ||
      storage_schema.static_spec_encoding_id == 0) {
    return false;
  }
  out_layout->storage_schema_encoding_id =
      storage_schema.static_spec_encoding_id;
  return true;
}

// Flattening discards the original per-axis footprint. Only discard it when
// existing facts prove the entire access, including every row of a vector tile.
// Otherwise the ranked access remains available to the production proof gate.
static bool loom_linearize_view_accesses_prepare_index(
    const loom_value_fact_table_t* fact_table,
    const loom_linearize_view_accesses_layout_t* layout, loom_type_t view_type,
    loom_type_t value_type, loom_attribute_t static_indices,
    loom_value_slice_t dynamic_indices,
    loom_linearize_view_accesses_index_t* out_index) {
  *out_index = (loom_linearize_view_accesses_index_t){0};
  const uint8_t rank = loom_type_rank(view_type);
  const uint8_t vector_rank =
      loom_type_is_vector(value_type) ? loom_type_rank(value_type) : 0;
  const uint8_t first_vector_axis = rank - vector_rank;
  uint16_t dynamic_index_position = dynamic_indices.count;
  for (uint8_t reverse_axis = 0; reverse_axis < rank; ++reverse_axis) {
    const uint8_t axis = rank - reverse_axis - 1;
    const int64_t stride = layout->strides[axis];
    const int64_t dimension = loom_type_dim_static_size_at(view_type, axis);
    const int64_t footprint = axis < first_vector_axis
                                  ? 1
                                  : loom_type_dim_static_size_at(
                                        value_type, axis - first_vector_axis);
    if (footprint > dimension) {
      return false;
    }
    const int64_t upper_bound = dimension - footprint;
    int64_t coordinate = static_indices.i64_array[axis];
    if (coordinate == INT64_MIN) {
      const loom_value_id_t value =
          dynamic_indices.values[--dynamic_index_position];
      const loom_value_facts_t facts =
          loom_value_fact_table_lookup(fact_table, value);
      if (loom_value_facts_is_float(facts) || facts.range_lo < 0 ||
          facts.range_hi > upper_bound) {
        return false;
      }
      if (!loom_value_facts_as_exact_i64(facts, &coordinate)) {
        out_index->terms[out_index->term_count++] =
            (loom_linearize_view_accesses_index_term_t){
                .value = value,
                .stride = stride,
            };
        continue;
      }
    } else if (coordinate < 0 || coordinate > upper_bound) {
      return false;
    }
    int64_t coordinate_offset = 0;
    if (!iree_checked_mul_i64(coordinate, stride, &coordinate_offset) ||
        !iree_checked_add_i64(out_index->static_offset, coordinate_offset,
                              &out_index->static_offset)) {
      return false;
    }
  }
  return true;
}

// Probe recipes before allocating IR. Only operations emitted by this pass
// enter the index, and all remain live. Structural access order gives
// same-block dominance without a separate availability walk; cross-block
// sharing remains the ordinary CSE pass's responsibility. Equality checks the
// current operands: replacing a load result can change an indexed operation's
// key, in which case the old hash can miss reuse but cannot return a different
// expression.
typedef struct loom_linearize_view_accesses_index_op_t {
  // Index constant, add, multiply, multiply-add, or left shift.
  loom_op_kind_t kind;
  // Constant payload, meaningful only for index.constant.
  int64_t constant;
  // Two operands for binary arithmetic or three for multiply-add.
  loom_value_id_t operands[3];
} loom_linearize_view_accesses_index_op_t;

typedef struct loom_linearize_view_accesses_index_probe_t {
  // Builder supplying the module and current insertion block.
  const loom_builder_t* builder;
  // Recipe to compare with an already-emitted operation.
  const loom_linearize_view_accesses_index_op_t* recipe;
} loom_linearize_view_accesses_index_probe_t;

static bool loom_linearize_view_accesses_index_equal(const void* user_data,
                                                     uint32_t value_id) {
  const loom_linearize_view_accesses_index_probe_t* probe = user_data;
  const loom_op_t* op =
      loom_value_def_op(loom_module_value(probe->builder->module, value_id));
  const loom_linearize_view_accesses_index_op_t* recipe = probe->recipe;
  if (op->parent_block != probe->builder->ip.block ||
      op->kind != recipe->kind) {
    return false;
  }
  if (recipe->kind == LOOM_OP_INDEX_CONSTANT) {
    return loom_attr_as_i64(loom_index_constant_value(op)) == recipe->constant;
  }
  return memcmp(loom_op_const_operands(op), recipe->operands,
                op->operand_count * sizeof(recipe->operands[0])) == 0;
}

static iree_status_t loom_linearize_view_accesses_build_index_op(
    loom_linearize_view_accesses_context_t* context,
    loom_linearize_view_accesses_index_op_t recipe, loom_location_id_t location,
    loom_value_id_t* out_value) {
  loom_builder_t* builder = &context->rewriter->builder;
  uint32_t hash = loom_structural_hash_initialize();
  hash = loom_structural_hash_mix_u64(hash, (uintptr_t)builder->ip.block);
  hash = loom_structural_hash_mix_u16(hash, recipe.kind);
  if (recipe.kind == LOOM_OP_INDEX_CONSTANT) {
    hash = loom_structural_hash_mix_u64(hash, (uint64_t)recipe.constant);
  } else {
    hash = loom_structural_hash_mix_u32(hash, recipe.operands[0]);
    hash = loom_structural_hash_mix_u32(hash, recipe.operands[1]);
    if (recipe.kind == LOOM_OP_INDEX_MADD) {
      hash = loom_structural_hash_mix_u32(hash, recipe.operands[2]);
    }
  }
  hash = loom_structural_hash_finalize(hash);
  const loom_linearize_view_accesses_index_probe_t key = {builder, &recipe};
  loom_intern_probe_t probe =
      loom_intern_table_probe(&context->index_ops, hash,
                              loom_linearize_view_accesses_index_equal, &key);
  if (probe.index != UINT32_MAX) {
    *out_value = probe.index;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(
      context->pass->arena, &context->index_ops, hash, 1, &probe.slot));
  loom_op_t* op = NULL;
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  switch (recipe.kind) {
    case LOOM_OP_INDEX_CONSTANT: {
      IREE_RETURN_IF_ERROR(loom_index_constant_build(
          builder, loom_attr_i64(recipe.constant), type, location, &op));
      break;
    }
    case LOOM_OP_INDEX_ADD: {
      IREE_RETURN_IF_ERROR(loom_index_add_build(builder, recipe.operands[0],
                                                recipe.operands[1], type,
                                                location, &op));
      break;
    }
    case LOOM_OP_INDEX_MUL: {
      IREE_RETURN_IF_ERROR(loom_index_mul_build(
          builder, recipe.operands[0], recipe.operands[1], location, &op));
      break;
    }
    case LOOM_OP_INDEX_MADD: {
      IREE_RETURN_IF_ERROR(
          loom_index_madd_build(builder, recipe.operands[0], recipe.operands[1],
                                recipe.operands[2], location, &op));
      break;
    }
    case LOOM_OP_INDEX_SHLI: {
      IREE_RETURN_IF_ERROR(loom_index_shli_build(
          builder, recipe.operands[0], recipe.operands[1], location, &op));
      break;
    }
    default: {
      IREE_ASSERT_UNREACHABLE("invalid linear index recipe");
      IREE_BUILTIN_UNREACHABLE();
    }
  }
  *out_value = loom_op_results(op)[0];
  loom_intern_table_insert(&context->index_ops, probe.slot, hash, *out_value);
  return iree_ok_status();
}

static iree_status_t loom_linearize_view_accesses_build_index_constant(
    loom_linearize_view_accesses_context_t* context, int64_t value,
    loom_location_id_t location, loom_value_id_t* out_index) {
  return loom_linearize_view_accesses_build_index_op(
      context,
      (loom_linearize_view_accesses_index_op_t){.kind = LOOM_OP_INDEX_CONSTANT,
                                                .constant = value},
      location, out_index);
}

static iree_status_t loom_linearize_view_accesses_build_linear_index(
    loom_linearize_view_accesses_context_t* context,
    const loom_linearize_view_accesses_index_t* index, int64_t row_offset,
    loom_location_id_t location, loom_value_id_t* out_linear_index,
    int64_t* out_static_linear_index) {
  *out_linear_index = LOOM_VALUE_ID_INVALID;
  const int64_t static_offset = index->static_offset + row_offset;
  *out_static_linear_index = static_offset;
  loom_value_id_t accumulator = LOOM_VALUE_ID_INVALID;
  for (uint8_t i = 0; i < index->term_count; ++i) {
    const loom_value_id_t axis_index = index->terms[i].value;
    const int64_t stride = index->terms[i].stride;
    // The coordinate footprint was proved even when its physical stride is
    // zero. Its contribution can now be omitted without discarding that proof.
    if (stride == 0) {
      continue;
    }
    if (stride == 1) {
      if (accumulator == LOOM_VALUE_ID_INVALID) {
        accumulator = axis_index;
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_build_index_op(
          context,
          (loom_linearize_view_accesses_index_op_t){
              .kind = LOOM_OP_INDEX_ADD, .operands = {accumulator, axis_index}},
          location, &accumulator));
    } else {
      const bool first_term = accumulator == LOOM_VALUE_ID_INVALID;
      loom_op_kind_t kind = first_term ? LOOM_OP_INDEX_MUL : LOOM_OP_INDEX_MADD;
      int64_t factor = stride;
      if (first_term && iree_math_is_power_of_two_i64(stride)) {
        const int64_t amount = iree_math_floor_log2_u64(stride);
        const loom_fact_context_t fact_context = {
            .target_facts = loom_value_fact_table_block_target_facts(
                context->fact_table, context->rewriter->builder.ip.block),
        };
        const int32_t bitwidth = loom_index_target_carrier_bitwidth(
            &fact_context, LOOM_SCALAR_TYPE_INDEX);
        if (amount < bitwidth) {
          kind = LOOM_OP_INDEX_SHLI;
          factor = amount;
        }
      }
      loom_value_id_t factor_index = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_build_index_constant(
          context, factor, location, &factor_index));
      IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_build_index_op(
          context,
          (loom_linearize_view_accesses_index_op_t){
              .kind = kind,
              .operands = {axis_index, factor_index, accumulator}},
          location, &accumulator));
    }
  }

  if (static_offset != 0 && accumulator != LOOM_VALUE_ID_INVALID) {
    loom_value_id_t offset_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_build_index_constant(
        context, static_offset, location, &offset_value));
    IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_build_index_op(
        context,
        (loom_linearize_view_accesses_index_op_t){
            .kind = LOOM_OP_INDEX_ADD, .operands = {accumulator, offset_value}},
        location, &accumulator));
  }

  *out_linear_index = accumulator;
  return iree_ok_status();
}

static iree_status_t loom_linearize_view_accesses_get_linear_view(
    loom_linearize_view_accesses_context_t* context, loom_op_t* view_op,
    loom_type_t view_type, const loom_linearize_view_accesses_layout_t* layout,
    loom_value_id_t* out_linear_view) {
  loom_linearize_view_accesses_view_map_t* map = context->view_map;
  const loom_value_id_t original_view = loom_buffer_view_result(view_op);
  const uint32_t hash = loom_structural_hash_finalize(original_view);
  const loom_linearize_view_accesses_view_probe_t key = {map, original_view};
  loom_intern_probe_t probe = loom_intern_table_probe(
      &map->index, hash, loom_linearize_view_accesses_view_equal, &key);
  if (probe.index != UINT32_MAX) {
    *out_linear_view =
        loom_linearize_view_accesses_view_map_entry(map, probe.index)
            ->linear_view;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(
      context->pass->arena, &map->index, hash, 1, &probe.slot));
  const uint32_t row = (uint32_t)map->index.count;
  if (row / LOOM_LINEARIZE_VIEW_ROWS_PER_SEGMENT == map->rows.segment_count) {
    void* segment = NULL;
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
        &map->rows, context->pass->arena, &segment));
  }

  uint16_t linear_encoding_id = view_type.encoding_id;
  loom_encoding_flags_t linear_encoding_flags = view_type.encoding_flags;
  if (layout->encoding_role == LOOM_ENCODING_ROLE_ADDRESS_LAYOUT) {
    linear_encoding_id = 0;
    linear_encoding_flags = 0;
  } else if (layout->encoding_role == LOOM_ENCODING_ROLE_PHYSICAL_STORAGE) {
    const loom_value_fact_encoding_summary_t linear_summary = {
        .role = LOOM_ENCODING_ROLE_PHYSICAL_STORAGE,
        .address_layout =
            {
                .kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE,
            },
        .storage_schema =
            {
                .static_spec_encoding_id = layout->storage_schema_encoding_id,
            },
    };
    IREE_RETURN_IF_ERROR(loom_encoding_intern_exact_summary(
        context->module, &linear_summary, &linear_encoding_id));
    IREE_ASSERT(linear_encoding_id != 0);
    linear_encoding_flags = 0;
  }

  loom_type_t linear_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, loom_type_element_type(view_type),
      loom_dim_pack_static(layout->linear_length), linear_encoding_id);
  linear_type.encoding_flags = linear_encoding_flags;
  linear_type = loom_type_view_with_alignment(
      linear_type, loom_type_view_alignment(view_type));

  loom_builder_t* builder = &context->rewriter->builder;
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_after(builder, view_op);
  loom_op_t* linear_view_op = NULL;
  iree_status_t status =
      loom_buffer_view_build(builder, loom_buffer_view_buffer(view_op),
                             loom_buffer_view_byte_offset(view_op), linear_type,
                             view_op->location, &linear_view_op);
  loom_builder_restore(builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  loom_value_id_t linear_view = loom_buffer_view_result(linear_view_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_copy_value_name(
      context->rewriter, loom_buffer_view_result(view_op), linear_view));
  *loom_linearize_view_accesses_view_map_entry(map, row) =
      (loom_linearize_view_accesses_view_map_entry_t){original_view,
                                                      linear_view};
  loom_intern_table_insert(&map->index, probe.slot, hash, row);
  ++context->statistics->views_created;
  *out_linear_view = linear_view;
  return iree_ok_status();
}

static bool loom_linearize_view_accesses_get_source_view(
    const loom_fact_context_t* fact_context, loom_module_t* module,
    loom_value_id_t view, loom_op_t** out_view_op, loom_type_t* out_view_type,
    loom_linearize_view_accesses_layout_t* out_layout) {
  const loom_value_t* view_value = loom_module_value(module, view);
  if (loom_value_is_block_arg(view_value)) {
    return false;
  }
  loom_op_t* view_op = loom_value_def_op(view_value);
  if (!loom_buffer_view_isa(view_op)) {
    return false;
  }
  loom_type_t view_type = loom_module_value_type(module, view);
  if (!loom_linearize_view_accesses_static_view_layout(fact_context, module,
                                                       view_type, out_layout)) {
    return false;
  }

  *out_view_op = view_op;
  *out_view_type = view_type;
  return true;
}

static bool loom_linearize_view_accesses_static_contiguous_vector_access(
    const loom_fact_context_t* fact_context, loom_module_t* module,
    loom_type_t view_type, loom_type_t vector_type,
    const loom_linearize_view_accesses_layout_t* layout) {
  loom_vector_memory_access_t access;
  if (!loom_vector_memory_access_describe(fact_context, module, view_type,
                                          vector_type, &access)) {
    return false;
  }
  if (access.vector_rank != 1) {
    return false;
  }
  if (access.first_vector_axis + 1 != access.view_rank) {
    return false;
  }
  if (loom_type_dim_is_dynamic_at(vector_type, 0)) {
    return false;
  }

  int64_t lane_count = loom_type_dim_static_size_at(vector_type, 0);
  if (lane_count <= 0 || lane_count > layout->linear_length) {
    return false;
  }
  const int64_t trailing_axis_dim =
      loom_type_dim_static_size_at(view_type, access.view_rank - 1);
  if (lane_count > trailing_axis_dim) {
    return false;
  }
  return layout->strides[access.view_rank - 1] == 1;
}

static bool loom_linearize_view_accesses_static_contiguous_vector_tile(
    const loom_fact_context_t* fact_context, loom_module_t* module,
    loom_type_t view_type, loom_type_t vector_type,
    loom_linearize_view_accesses_vector_tile_t* out_tile) {
  *out_tile = (loom_linearize_view_accesses_vector_tile_t){0};

  loom_vector_memory_access_t access;
  if (!loom_vector_memory_access_describe(fact_context, module, view_type,
                                          vector_type, &access)) {
    return false;
  }
  if (access.vector_rank <= 1 || !loom_type_is_all_static(vector_type)) {
    return false;
  }
  int64_t inner_axis_stride = 0;
  if (!loom_vector_memory_access_static_axis_stride(
          &access, access.first_vector_axis + access.vector_rank - 1,
          &inner_axis_stride) ||
      inner_axis_stride != 1) {
    return false;
  }

  uint64_t flat_lane_count = 0;
  if (!loom_type_static_element_count(vector_type, &flat_lane_count) ||
      flat_lane_count > INT64_MAX || flat_lane_count > UINT16_MAX) {
    return false;
  }

  const uint8_t vector_rank = access.vector_rank;
  int64_t inner_lane_count =
      loom_type_dim_static_size_at(vector_type, vector_rank - 1);
  if (inner_lane_count <= 0 ||
      flat_lane_count % (uint64_t)inner_lane_count != 0) {
    return false;
  }
  int64_t vector_dims[LOOM_TYPE_MAX_RANK] = {0};
  for (uint8_t axis = 0; axis < vector_rank; ++axis) {
    const int64_t dim = loom_type_dim_static_size_at(vector_type, axis);
    if (dim <= 0) {
      return false;
    }
    vector_dims[axis] = dim;
  }

  // Logical row boundaries do not split physically adjacent elements. Keep the
  // largest contiguous suffix intact for target-specific access-width
  // selection.
  uint8_t row_axis_count = vector_rank - 1;
  while (row_axis_count) {
    const uint8_t axis = row_axis_count - 1;
    int64_t stride = 0;
    if (vector_dims[axis] != 1 &&
        (!loom_vector_memory_access_static_axis_stride(
             &access, access.first_vector_axis + axis, &stride) ||
         stride != inner_lane_count)) {
      break;
    }
    inner_lane_count *= vector_dims[axis];
    --row_axis_count;
  }
  const uint64_t row_count = flat_lane_count / (uint64_t)inner_lane_count;

  const loom_scalar_type_t element_type = loom_type_element_type(vector_type);
  *out_tile = (loom_linearize_view_accesses_vector_tile_t){
      .first_vector_axis = access.first_vector_axis,
      .row_axis_count = row_axis_count,
      .inner_lane_count = inner_lane_count,
      .row_count = (int64_t)row_count,
      .flat_lane_count = (int64_t)flat_lane_count,
      .row_vector_type = loom_type_shaped_1d(
          LOOM_TYPE_VECTOR, element_type,
          loom_dim_pack_static((uint64_t)inner_lane_count), /*encoding_id=*/0),
      .flat_vector_type = loom_type_shaped_1d(
          LOOM_TYPE_VECTOR, element_type, loom_dim_pack_static(flat_lane_count),
          /*encoding_id=*/0),
  };
  memcpy(out_tile->vector_dims, vector_dims,
         vector_rank * sizeof(out_tile->vector_dims[0]));
  return true;
}

static void loom_linearize_view_accesses_tile_row_offsets(
    const loom_linearize_view_accesses_vector_tile_t* tile, int64_t row_ordinal,
    int64_t* out_axis_static_offsets) {
  int64_t remaining = row_ordinal;
  for (uint8_t reverse_axis = 0; reverse_axis < tile->row_axis_count;
       ++reverse_axis) {
    const uint8_t vector_axis =
        (uint8_t)(tile->row_axis_count - reverse_axis - 1);
    const int64_t dim = tile->vector_dims[vector_axis];
    const int64_t coord = remaining % dim;
    remaining /= dim;
    out_axis_static_offsets[tile->first_vector_axis + vector_axis] = coord;
  }
}

static iree_status_t loom_linearize_view_accesses_build_tile_row_index(
    loom_linearize_view_accesses_context_t* context,
    const loom_linearize_view_accesses_layout_t* layout,
    const loom_linearize_view_accesses_index_t* index,
    const loom_linearize_view_accesses_vector_tile_t* tile, int64_t row_ordinal,
    loom_location_id_t location, loom_value_id_t* out_linear_index,
    int64_t* out_static_linear_index) {
  int64_t axis_static_offsets[LOOM_TYPE_MAX_RANK] = {0};
  loom_linearize_view_accesses_tile_row_offsets(tile, row_ordinal,
                                                axis_static_offsets);

  // The original footprint proof bounds this row within the checked physical
  // span, so every product and their sum fit the span's signed 64-bit domain.
  int64_t row_offset = 0;
  for (uint8_t axis = 0; axis < layout->rank; ++axis) {
    row_offset += axis_static_offsets[axis] * layout->strides[axis];
  }
  return loom_linearize_view_accesses_build_linear_index(
      context, index, row_offset, location, out_linear_index,
      out_static_linear_index);
}

static iree_status_t loom_linearize_view_accesses_rewrite_load(
    loom_linearize_view_accesses_context_t* context, loom_op_t* load_op,
    bool* out_changed) {
  loom_op_t* view_op = NULL;
  loom_type_t view_type = loom_type_none();
  loom_linearize_view_accesses_layout_t layout;
  const loom_fact_context_t* fact_context = &context->fact_table->context;
  if (!loom_linearize_view_accesses_get_source_view(
          fact_context, context->module, loom_view_load_view(load_op), &view_op,
          &view_type, &layout)) {
    return iree_ok_status();
  }

  loom_linearize_view_accesses_index_t index;
  if (!loom_linearize_view_accesses_prepare_index(
          context->fact_table, &layout, view_type,
          loom_module_value_type(context->module,
                                 loom_view_load_result(load_op)),
          loom_view_load_static_indices(load_op),
          loom_view_load_indices(load_op), &index)) {
    return iree_ok_status();
  }

  loom_builder_t* builder = &context->rewriter->builder;
  loom_builder_set_before(builder, load_op);
  loom_value_id_t linear_index = LOOM_VALUE_ID_INVALID;
  int64_t static_linear_index = INT64_MIN;
  IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_build_linear_index(
      context, &index, /*row_offset=*/0, load_op->location, &linear_index,
      &static_linear_index));

  loom_value_id_t linear_view = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_get_linear_view(
      context, view_op, view_type, &layout, &linear_view));

  int64_t static_indices[] = {
      linear_index == LOOM_VALUE_ID_INVALID ? static_linear_index : INT64_MIN};
  loom_value_id_t dynamic_indices[] = {linear_index};
  iree_host_size_t dynamic_index_count = linear_index == LOOM_VALUE_ID_INVALID
                                             ? 0
                                             : IREE_ARRAYSIZE(dynamic_indices);
  loom_view_load_build_flags_t build_flags = 0;
  const loom_cache_policy_t policy =
      loom_cache_policy_cast(context->module, load_op);
  const loom_attribute_t cache_scope = loom_cache_policy_scope(policy);
  const loom_attribute_t cache_temporal = loom_cache_policy_temporal(policy);
  if (cache_scope.kind != 0) {
    build_flags |= LOOM_VIEW_LOAD_BUILD_FLAG_HAS_CACHE_SCOPE;
  }
  if (cache_temporal.kind != 0) {
    build_flags |= LOOM_VIEW_LOAD_BUILD_FLAG_HAS_CACHE_TEMPORAL;
  }

  loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(context->rewriter);
  loom_op_t* linear_load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_view_load_build(
      builder, build_flags, loom_view_load_memory_flags(load_op), linear_view,
      dynamic_indices, dynamic_index_count, static_indices,
      IREE_ARRAYSIZE(static_indices), loom_view_load_cache_scope(load_op),
      loom_view_load_cache_temporal(load_op),
      loom_module_value_type(context->module, loom_view_load_result(load_op)),
      load_op->location, &linear_load_op));
  loom_value_id_t replacement = loom_view_load_result(linear_load_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      context->rewriter, load_op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      context->rewriter, load_op, &replacement, 1));
  ++context->statistics->loads_linearized;
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_linearize_view_accesses_rewrite_store(
    loom_linearize_view_accesses_context_t* context, loom_op_t* store_op,
    bool* out_changed) {
  loom_op_t* view_op = NULL;
  loom_type_t view_type = loom_type_none();
  loom_linearize_view_accesses_layout_t layout;
  const loom_fact_context_t* fact_context = &context->fact_table->context;
  if (!loom_linearize_view_accesses_get_source_view(
          fact_context, context->module, loom_view_store_view(store_op),
          &view_op, &view_type, &layout)) {
    return iree_ok_status();
  }

  loom_linearize_view_accesses_index_t index;
  if (!loom_linearize_view_accesses_prepare_index(
          context->fact_table, &layout, view_type,
          loom_module_value_type(context->module,
                                 loom_view_store_value(store_op)),
          loom_view_store_static_indices(store_op),
          loom_view_store_indices(store_op), &index)) {
    return iree_ok_status();
  }

  loom_builder_t* builder = &context->rewriter->builder;
  loom_builder_set_before(builder, store_op);
  loom_value_id_t linear_index = LOOM_VALUE_ID_INVALID;
  int64_t static_linear_index = INT64_MIN;
  IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_build_linear_index(
      context, &index, /*row_offset=*/0, store_op->location, &linear_index,
      &static_linear_index));

  loom_value_id_t linear_view = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_get_linear_view(
      context, view_op, view_type, &layout, &linear_view));

  int64_t static_indices[] = {
      linear_index == LOOM_VALUE_ID_INVALID ? static_linear_index : INT64_MIN};
  loom_value_id_t dynamic_indices[] = {linear_index};
  iree_host_size_t dynamic_index_count = linear_index == LOOM_VALUE_ID_INVALID
                                             ? 0
                                             : IREE_ARRAYSIZE(dynamic_indices);
  loom_view_store_build_flags_t build_flags = 0;
  const loom_cache_policy_t policy =
      loom_cache_policy_cast(context->module, store_op);
  const loom_attribute_t cache_scope = loom_cache_policy_scope(policy);
  const loom_attribute_t cache_temporal = loom_cache_policy_temporal(policy);
  if (cache_scope.kind != 0) {
    build_flags |= LOOM_VIEW_STORE_BUILD_FLAG_HAS_CACHE_SCOPE;
  }
  if (cache_temporal.kind != 0) {
    build_flags |= LOOM_VIEW_STORE_BUILD_FLAG_HAS_CACHE_TEMPORAL;
  }

  loom_op_t* linear_store_op = NULL;
  IREE_RETURN_IF_ERROR(loom_view_store_build(
      builder, build_flags, loom_view_store_memory_flags(store_op),
      loom_view_store_value(store_op), linear_view, dynamic_indices,
      dynamic_index_count, static_indices, IREE_ARRAYSIZE(static_indices),
      loom_view_store_cache_scope(store_op),
      loom_view_store_cache_temporal(store_op), store_op->location,
      &linear_store_op));
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(context->rewriter, store_op));
  ++context->statistics->stores_linearized;
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_linearize_view_accesses_rewrite_vector_load(
    loom_linearize_view_accesses_context_t* context, loom_op_t* load_op,
    bool* out_changed) {
  loom_op_t* view_op = NULL;
  loom_type_t view_type = loom_type_none();
  loom_linearize_view_accesses_layout_t layout;
  const loom_fact_context_t* fact_context = &context->fact_table->context;
  if (!loom_linearize_view_accesses_get_source_view(
          fact_context, context->module, loom_vector_load_view(load_op),
          &view_op, &view_type, &layout)) {
    return iree_ok_status();
  }

  loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_load_result(load_op));
  loom_linearize_view_accesses_index_t index;
  if (!loom_type_is_all_static(result_type) ||
      !loom_linearize_view_accesses_prepare_index(
          context->fact_table, &layout, view_type, result_type,
          loom_vector_load_static_indices(load_op),
          loom_vector_load_indices(load_op), &index)) {
    return iree_ok_status();
  }
  if (!loom_linearize_view_accesses_static_contiguous_vector_access(
          fact_context, context->module, view_type, result_type, &layout)) {
    loom_linearize_view_accesses_vector_tile_t tile = {0};
    if (!loom_linearize_view_accesses_static_contiguous_vector_tile(
            fact_context, context->module, view_type, result_type, &tile)) {
      return iree_ok_status();
    }

    loom_vector_memory_cache_policy_t cache_policy = {0};
    if (!loom_vector_memory_cache_policy_from_op(context->module, load_op,
                                                 &cache_policy)) {
      return iree_ok_status();
    }

    loom_builder_t* builder = &context->rewriter->builder;
    loom_builder_set_before(builder, load_op);
    loom_value_id_t linear_view = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_get_linear_view(
        context, view_op, view_type, &layout, &linear_view));

    loom_value_id_t* row_values = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->pass->arena, (iree_host_size_t)tile.row_count,
        sizeof(*row_values), (void**)&row_values));
    for (int64_t row = 0; row < tile.row_count; ++row) {
      loom_value_id_t linear_index = LOOM_VALUE_ID_INVALID;
      int64_t static_linear_index = INT64_MIN;
      IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_build_tile_row_index(
          context, &layout, &index, &tile, row, load_op->location,
          &linear_index, &static_linear_index));

      int64_t static_indices[] = {linear_index == LOOM_VALUE_ID_INVALID
                                      ? static_linear_index
                                      : INT64_MIN};
      loom_value_id_t dynamic_indices[] = {linear_index};
      iree_host_size_t dynamic_index_count =
          linear_index == LOOM_VALUE_ID_INVALID
              ? 0
              : IREE_ARRAYSIZE(dynamic_indices);
      loom_op_t* row_load_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_load_build(
          builder, cache_policy.build_flags,
          loom_vector_load_memory_flags(load_op), linear_view, dynamic_indices,
          dynamic_index_count, static_indices, IREE_ARRAYSIZE(static_indices),
          cache_policy.cache_scope, cache_policy.cache_temporal,
          tile.row_vector_type, load_op->location, &row_load_op));
      row_values[row] = loom_vector_load_result(row_load_op);
    }

    loom_value_id_t flat_value = row_values[0];
    if (tile.row_count > 1) {
      loom_op_t* concat_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_concat_build(
          builder, /*axis=*/0, row_values, (iree_host_size_t)tile.row_count,
          tile.flat_vector_type, load_op->location, &concat_op));
      flat_value = loom_vector_concat_result(concat_op);
    }

    loom_value_id_t value_checkpoint =
        loom_rewriter_value_checkpoint(context->rewriter);
    loom_op_t* bitcast_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_bitcast_build(builder, flat_value, tile.flat_vector_type,
                                  result_type, load_op->location, &bitcast_op));
    loom_value_id_t replacement = loom_vector_bitcast_result(bitcast_op);
    IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
        context->rewriter, load_op, &replacement, 1, value_checkpoint));
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
        context->rewriter, load_op, &replacement, 1));
    ++context->statistics->vector_loads_linearized;
    *out_changed = true;
    return iree_ok_status();
  }

  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_vector_memory_cache_policy_from_op(context->module, load_op,
                                               &cache_policy)) {
    return iree_ok_status();
  }

  loom_builder_t* builder = &context->rewriter->builder;
  loom_builder_set_before(builder, load_op);
  loom_value_id_t linear_index = LOOM_VALUE_ID_INVALID;
  int64_t static_linear_index = INT64_MIN;
  IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_build_linear_index(
      context, &index, /*row_offset=*/0, load_op->location, &linear_index,
      &static_linear_index));

  loom_value_id_t linear_view = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_get_linear_view(
      context, view_op, view_type, &layout, &linear_view));

  int64_t static_indices[] = {
      linear_index == LOOM_VALUE_ID_INVALID ? static_linear_index : INT64_MIN};
  loom_value_id_t dynamic_indices[] = {linear_index};
  iree_host_size_t dynamic_index_count = linear_index == LOOM_VALUE_ID_INVALID
                                             ? 0
                                             : IREE_ARRAYSIZE(dynamic_indices);
  loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(context->rewriter);
  loom_op_t* linear_load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_load_build(
      builder, cache_policy.build_flags, loom_vector_load_memory_flags(load_op),
      linear_view, dynamic_indices, dynamic_index_count, static_indices,
      IREE_ARRAYSIZE(static_indices), cache_policy.cache_scope,
      cache_policy.cache_temporal, result_type, load_op->location,
      &linear_load_op));
  loom_value_id_t replacement = loom_vector_load_result(linear_load_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      context->rewriter, load_op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      context->rewriter, load_op, &replacement, 1));
  ++context->statistics->vector_loads_linearized;
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_linearize_view_accesses_rewrite_vector_store(
    loom_linearize_view_accesses_context_t* context, loom_op_t* store_op,
    bool* out_changed) {
  loom_op_t* view_op = NULL;
  loom_type_t view_type = loom_type_none();
  loom_linearize_view_accesses_layout_t layout;
  const loom_fact_context_t* fact_context = &context->fact_table->context;
  if (!loom_linearize_view_accesses_get_source_view(
          fact_context, context->module, loom_vector_store_view(store_op),
          &view_op, &view_type, &layout)) {
    return iree_ok_status();
  }

  loom_type_t value_type = loom_module_value_type(
      context->module, loom_vector_store_value(store_op));
  loom_linearize_view_accesses_index_t index;
  if (!loom_type_is_all_static(value_type) ||
      !loom_linearize_view_accesses_prepare_index(
          context->fact_table, &layout, view_type, value_type,
          loom_vector_store_static_indices(store_op),
          loom_vector_store_indices(store_op), &index)) {
    return iree_ok_status();
  }
  if (!loom_linearize_view_accesses_static_contiguous_vector_access(
          fact_context, context->module, view_type, value_type, &layout)) {
    loom_linearize_view_accesses_vector_tile_t tile = {0};
    if (!loom_linearize_view_accesses_static_contiguous_vector_tile(
            fact_context, context->module, view_type, value_type, &tile)) {
      return iree_ok_status();
    }

    loom_vector_memory_cache_policy_t cache_policy = {0};
    if (!loom_vector_memory_cache_policy_from_op(context->module, store_op,
                                                 &cache_policy)) {
      return iree_ok_status();
    }

    loom_builder_t* builder = &context->rewriter->builder;
    loom_builder_set_before(builder, store_op);
    loom_value_id_t linear_view = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_get_linear_view(
        context, view_op, view_type, &layout, &linear_view));

    loom_op_t* bitcast_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
        builder, loom_vector_store_value(store_op), value_type,
        tile.flat_vector_type, store_op->location, &bitcast_op));
    const loom_value_id_t flat_value = loom_vector_bitcast_result(bitcast_op);

    for (int64_t row = 0; row < tile.row_count; ++row) {
      loom_value_id_t row_value = flat_value;
      if (tile.row_count > 1) {
        int64_t static_offsets[] = {row * tile.inner_lane_count};
        loom_op_t* slice_op = NULL;
        IREE_RETURN_IF_ERROR(loom_vector_slice_build(
            builder, flat_value, /*offsets=*/NULL, /*offsets_count=*/0,
            static_offsets, IREE_ARRAYSIZE(static_offsets),
            tile.row_vector_type, store_op->location, &slice_op));
        row_value = loom_vector_slice_result(slice_op);
      }

      loom_value_id_t linear_index = LOOM_VALUE_ID_INVALID;
      int64_t static_linear_index = INT64_MIN;
      IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_build_tile_row_index(
          context, &layout, &index, &tile, row, store_op->location,
          &linear_index, &static_linear_index));

      int64_t static_indices[] = {linear_index == LOOM_VALUE_ID_INVALID
                                      ? static_linear_index
                                      : INT64_MIN};
      loom_value_id_t dynamic_indices[] = {linear_index};
      iree_host_size_t dynamic_index_count =
          linear_index == LOOM_VALUE_ID_INVALID
              ? 0
              : IREE_ARRAYSIZE(dynamic_indices);
      loom_op_t* row_store_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_store_build(
          builder, cache_policy.build_flags,
          loom_vector_store_memory_flags(store_op), row_value, linear_view,
          dynamic_indices, dynamic_index_count, static_indices,
          IREE_ARRAYSIZE(static_indices), cache_policy.cache_scope,
          cache_policy.cache_temporal, store_op->location, &row_store_op));
    }

    IREE_RETURN_IF_ERROR(loom_rewriter_erase(context->rewriter, store_op));
    ++context->statistics->vector_stores_linearized;
    *out_changed = true;
    return iree_ok_status();
  }

  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_vector_memory_cache_policy_from_op(context->module, store_op,
                                               &cache_policy)) {
    return iree_ok_status();
  }

  loom_builder_t* builder = &context->rewriter->builder;
  loom_builder_set_before(builder, store_op);
  loom_value_id_t linear_index = LOOM_VALUE_ID_INVALID;
  int64_t static_linear_index = INT64_MIN;
  IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_build_linear_index(
      context, &index, /*row_offset=*/0, store_op->location, &linear_index,
      &static_linear_index));

  loom_value_id_t linear_view = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_linearize_view_accesses_get_linear_view(
      context, view_op, view_type, &layout, &linear_view));

  int64_t static_indices[] = {
      linear_index == LOOM_VALUE_ID_INVALID ? static_linear_index : INT64_MIN};
  loom_value_id_t dynamic_indices[] = {linear_index};
  iree_host_size_t dynamic_index_count = linear_index == LOOM_VALUE_ID_INVALID
                                             ? 0
                                             : IREE_ARRAYSIZE(dynamic_indices);
  loom_op_t* linear_store_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_store_build(
      builder, cache_policy.build_flags,
      loom_vector_store_memory_flags(store_op),
      loom_vector_store_value(store_op), linear_view, dynamic_indices,
      dynamic_index_count, static_indices, IREE_ARRAYSIZE(static_indices),
      cache_policy.cache_scope, cache_policy.cache_temporal, store_op->location,
      &linear_store_op));
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(context->rewriter, store_op));
  ++context->statistics->vector_stores_linearized;
  *out_changed = true;
  return iree_ok_status();
}

static iree_status_t loom_linearize_view_accesses_rewrite_access(
    loom_linearize_view_accesses_context_t* context, loom_op_t* op,
    bool* out_changed) {
  if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_ok_status();
  }
  switch (loom_linearize_view_accesses_access_kind(op)) {
    case LOOM_LINEARIZE_VIEW_ACCESSES_ACCESS_VIEW_LOAD:
      return loom_linearize_view_accesses_rewrite_load(context, op,
                                                       out_changed);
    case LOOM_LINEARIZE_VIEW_ACCESSES_ACCESS_VIEW_STORE:
      return loom_linearize_view_accesses_rewrite_store(context, op,
                                                        out_changed);
    case LOOM_LINEARIZE_VIEW_ACCESSES_ACCESS_VECTOR_LOAD:
      return loom_linearize_view_accesses_rewrite_vector_load(context, op,
                                                              out_changed);
    case LOOM_LINEARIZE_VIEW_ACCESSES_ACCESS_VECTOR_STORE:
      return loom_linearize_view_accesses_rewrite_vector_store(context, op,
                                                               out_changed);
    case LOOM_LINEARIZE_VIEW_ACCESSES_ACCESS_NONE:
      return iree_ok_status();
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Pass entry point
//===----------------------------------------------------------------------===//

iree_status_t loom_linearize_view_accesses_run(loom_pass_t* pass,
                                               loom_module_t* module,
                                               loom_func_like_t function) {
  if (!loom_func_like_body(function)) {
    return iree_ok_status();
  }

  loom_linearize_view_accesses_op_list_t accesses = {0};
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  loom_linearize_view_accesses_collect_context_t collect_context = {
      .arena = pass->arena,
      .accesses = &accesses,
  };
  IREE_RETURN_IF_ERROR(
      loom_walk_function(module, function, LOOM_WALK_PRE_ORDER,
                         (loom_walk_callback_t){
                             .fn = loom_linearize_view_accesses_collect_access,
                             .user_data = &collect_context,
                         },
                         &walk_result));
  if (accesses.count == 0) {
    return iree_ok_status();
  }

  loom_value_fact_table_t* fact_table = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_facts_acquire(
      pass, module,
      loom_pass_value_fact_scope_function_for_target(
          function,
          loom_target_function_version_target_facts(pass->function_version)),
      &fact_table));

  loom_linearize_view_accesses_view_map_t view_map = {0};
  IREE_RETURN_IF_ERROR(
      loom_intern_table_initialize(pass->arena, 0, &view_map.index));
  loom_segmented_storage_initialize(
      LOOM_LINEARIZE_VIEW_ROWS_PER_SEGMENT *
          sizeof(loom_linearize_view_accesses_view_map_entry_t),
      iree_alignof(loom_linearize_view_accesses_view_map_entry_t),
      &view_map.rows);

  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, pass->arena);
  iree_status_t status = iree_ok_status();
  bool changed = false;
  loom_linearize_view_accesses_context_t context = {
      .pass = pass,
      .statistics = loom_linearize_view_accesses_statistics(pass),
      .module = module,
      .rewriter = &rewriter,
      .fact_table = fact_table,
      .view_map = &view_map,
  };
  IREE_RETURN_IF_ERROR(
      loom_intern_table_initialize(pass->arena, 0, &context.index_ops));
  for (iree_host_size_t i = 0; i < accesses.count && iree_status_is_ok(status);
       ++i) {
    bool access_changed = false;
    status = loom_linearize_view_accesses_rewrite_access(
        &context, accesses.ops[i], &access_changed);
    changed = changed || access_changed;
  }
  if (changed) {
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
    if (iree_status_is_ok(status)) {
      loom_pass_mark_changed(pass);
    }
  }

  loom_rewriter_deinitialize(&rewriter);
  return status;
}
