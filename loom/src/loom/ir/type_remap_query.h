// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scoped equality queries over one immutable SSA value remap.
//
// The ordinary loom_type_equal_after_value_remap API is allocation-free and
// appropriate for independent comparisons. A boundary that owns many roots
// under the same remap can use this query to retain completed compound work for
// the duration of that boundary. Inline types remain allocation-free. Compound
// state is allocated lazily from the module block pool and released at
// deinitialization; no query state is retained in the module or its types.

#ifndef LOOM_IR_TYPE_REMAP_QUERY_H_
#define LOOM_IR_TYPE_REMAP_QUERY_H_

#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_type_remap_proof_state_t loom_type_remap_proof_state_t;
typedef struct loom_type_remap_exact_state_t loom_type_remap_exact_state_t;

// One scoped batch of comparisons sharing an immutable source-to-target map.
//
// The module, remap spans, and every source/target type must remain immutable
// until deinitialization. Types must be canonical values owned by |module| or
// inline types with no backing payload. This is a trusted compiler-internal
// query used after parsing and verification, not an external validation
// boundary.
typedef struct loom_type_remap_query_t {
  // Module owning all canonical types and the scratch block pool.
  const loom_module_t* module;
  // Borrowed immutable source-to-target value map.
  const loom_type_value_remap_t* remap;
  // Scoped arena initialized on the first compound comparison.
  iree_arena_allocator_t scratch_arena;
  // Lazily allocated completed-proof state for function type DAGs.
  loom_type_remap_proof_state_t* proof_state;
  // Lazily allocated exact normalization state used after inequality.
  loom_type_remap_exact_state_t* exact_state;
  // Whether |scratch_arena| has been initialized.
  bool scratch_initialized;
  // Whether subsequent compound comparisons require exact normalization.
  bool exact_mode;
} loom_type_remap_query_t;

// Applies |remap| to one source-side SSA value ID. Definition-slice promises
// are consumed directly; their producer must have established the recorded
// ownership and contiguous index domain.
loom_value_id_t loom_type_value_remap_apply(
    const loom_module_t* module, const loom_type_value_remap_t* remap,
    loom_value_id_t value_id);

// Initializes an empty allocation-free query. |module| and |remap| are
// borrowed until loom_type_remap_query_deinitialize.
void loom_type_remap_query_initialize(const loom_module_t* module,
                                      const loom_type_value_remap_t* remap,
                                      loom_type_remap_query_t* out_query);

// Releases all temporary blocks acquired by |query|. NULL is tolerated.
void loom_type_remap_query_deinitialize(loom_type_remap_query_t* query);

// Compares a pair requiring compound proof or exact normalization. Callers use
// loom_type_remap_query_equal so inline roots remain in their owning loop.
iree_status_t loom_type_remap_query_equal_compound(
    loom_type_remap_query_t* query, loom_type_t source_type,
    loom_type_t target_type, bool* out_equal);

// Tries an allocation-free inline comparison. Returns false when the pair
// requires compound proof or normalization and leaves |out_equal| false.
IREE_ATTRIBUTE_ALWAYS_INLINE static inline bool
loom_type_remap_query_try_inline(const loom_type_remap_query_t* query,
                                 loom_type_t source_type,
                                 loom_type_t target_type, bool* out_equal) {
  *out_equal = false;
  const loom_type_kind_t source_kind = loom_type_kind(source_type);
  if (source_kind != loom_type_kind(target_type)) {
    return true;
  }

  if (source_kind == LOOM_TYPE_FUNCTION || source_kind == LOOM_TYPE_DIALECT ||
      source_kind == LOOM_TYPE_PARAMETERIZED ||
      (source_kind == LOOM_TYPE_REGISTER &&
       (loom_type_register_has_value_type(source_type) ||
        loom_type_register_has_value_type(target_type))) ||
      ((loom_type_is_shaped(source_type) || loom_type_is_pool(source_type)) &&
       (!loom_type_has_inline_dims(source_type) ||
        !loom_type_has_inline_dims(target_type)))) {
    return false;
  }

  bool equal = true;
  if (loom_type_is_shaped(source_type) || loom_type_is_pool(source_type)) {
    equal = source_type.header == target_type.header &&
            source_type.encoding_flags == target_type.encoding_flags;
    for (uint8_t i = 0; equal && i < loom_type_rank(source_type); ++i) {
      uint64_t dimension = loom_type_dim(source_type, i);
      if (loom_dim_is_dynamic(dimension)) {
        dimension = loom_dim_pack_dynamic(loom_type_value_remap_apply(
            query->module, query->remap, loom_dim_value_id(dimension)));
      }
      equal = dimension == loom_type_dim(target_type, i);
    }
    uint32_t encoding = source_type.encoding_id;
    if (equal && loom_type_has_ssa_encoding(source_type)) {
      encoding =
          loom_type_value_remap_apply(query->module, query->remap,
                                      loom_type_encoding_value_id(source_type));
    }
    equal &= encoding == target_type.encoding_id;
  } else {
    equal = source_type.header == target_type.header &&
            source_type.encoding_id == target_type.encoding_id &&
            source_type.encoding_flags == target_type.encoding_flags &&
            source_type.dims[0] == target_type.dims[0] &&
            source_type.dims[1] == target_type.dims[1];
  }
  *out_equal = equal;
  return true;
}

// Compares |source_type| after applying the query remap with |target_type|.
//
// The result is exact: structural hashes are never accepted as equality.
// Inline types remain in the caller and cannot allocate. Compound comparison
// is outlined and can acquire temporary blocks. A non-OK status is terminal
// for the query and leaves |out_equal| false.
IREE_ATTRIBUTE_ALWAYS_INLINE static inline iree_status_t
loom_type_remap_query_equal(loom_type_remap_query_t* query,
                            loom_type_t source_type, loom_type_t target_type,
                            bool* out_equal) {
  IREE_ASSERT_ARGUMENT(query);
  IREE_ASSERT_ARGUMENT(out_equal);
  if (!loom_type_remap_query_try_inline(query, source_type, target_type,
                                        out_equal)) {
    return loom_type_remap_query_equal_compound(query, source_type, target_type,
                                                out_equal);
  }
  if (!*out_equal) {
    query->exact_mode = true;
  }
  return iree_ok_status();
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_IR_TYPE_REMAP_QUERY_H_
