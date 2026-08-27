// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/view/target_legalization.h"

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/atomic.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/rewrite/rewriter.h"

// Compare-exchange failure cannot carry release semantics. This table maps a
// verified atomic RMW ordering to the strongest valid failure ordering that
// preserves its acquire semantics.
static const loom_atomic_ordering_t kAtomicRmwFailureOrderings[] = {
    [LOOM_ATOMIC_ORDERING_RELAXED] = LOOM_ATOMIC_ORDERING_RELAXED,
    [LOOM_ATOMIC_ORDERING_ACQUIRE] = LOOM_ATOMIC_ORDERING_ACQUIRE,
    [LOOM_ATOMIC_ORDERING_RELEASE] = LOOM_ATOMIC_ORDERING_RELAXED,
    [LOOM_ATOMIC_ORDERING_ACQ_REL] = LOOM_ATOMIC_ORDERING_ACQUIRE,
    [LOOM_ATOMIC_ORDERING_SEQ_CST] = LOOM_ATOMIC_ORDERING_SEQ_CST,
};
static_assert(IREE_ARRAYSIZE(kAtomicRmwFailureOrderings) ==
                  LOOM_ATOMIC_ORDERING_COUNT_,
              "all atomic RMW orderings must map to a failure ordering");

static loom_view_atomic_cmpxchg_build_flags_t
loom_view_legalize_atomic_cmpxchg_cache_policy(loom_memory_access_t access,
                                               uint8_t* out_cache_scope,
                                               uint8_t* out_cache_temporal) {
  loom_view_atomic_cmpxchg_build_flags_t build_flags = 0;
  const loom_attribute_t cache_scope = loom_memory_access_cache_scope(access);
  if (!loom_attr_is_absent(cache_scope)) {
    build_flags |= LOOM_VIEW_ATOMIC_CMPXCHG_BUILD_FLAG_HAS_CACHE_SCOPE;
    *out_cache_scope = loom_attr_as_enum(cache_scope);
  }
  const loom_attribute_t cache_temporal =
      loom_memory_access_cache_temporal(access);
  if (!loom_attr_is_absent(cache_temporal)) {
    build_flags |= LOOM_VIEW_ATOMIC_CMPXCHG_BUILD_FLAG_HAS_CACHE_TEMPORAL;
    *out_cache_temporal = loom_attr_as_enum(cache_temporal);
  }
  return build_flags;
}

static iree_status_t loom_view_legalize_build_atomic_addf_before_region(
    loom_builder_t* builder, loom_op_t* loop, loom_memory_access_t access,
    loom_type_t float_type, loom_type_t integer_type,
    loom_location_id_t location) {
  const loom_value_id_t expected =
      loom_region_entry_arg_id(loom_scf_while_before(loop), 0);
  loom_op_t* add_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_addf_build(
      builder, /*instance_flags=*/0, expected, loom_memory_access_value(access),
      float_type, location, &add_op));

  uint8_t cache_scope = 0;
  uint8_t cache_temporal = 0;
  const loom_view_atomic_cmpxchg_build_flags_t build_flags =
      loom_view_legalize_atomic_cmpxchg_cache_policy(access, &cache_scope,
                                                     &cache_temporal);
  const loom_value_slice_t indices = loom_memory_access_dynamic_indices(access);
  const loom_attribute_t static_indices =
      loom_memory_access_static_indices(access);
  const loom_atomic_ordering_t success_ordering =
      (loom_atomic_ordering_t)loom_attr_as_enum(
          loom_memory_access_atomic_ordering(access));
  loom_op_t* cmpxchg_op = NULL;
  IREE_RETURN_IF_ERROR(loom_view_atomic_cmpxchg_build(
      builder, build_flags, expected, loom_scalar_addf_result(add_op),
      loom_memory_access_view(access), indices.values, indices.count,
      static_indices.i64_array, static_indices.count, success_ordering,
      kAtomicRmwFailureOrderings[success_ordering],
      (loom_atomic_scope_t)loom_attr_as_enum(
          loom_memory_access_atomic_scope(access)),
      cache_scope, cache_temporal, float_type, location, &cmpxchg_op));
  const loom_value_id_t observed = loom_view_atomic_cmpxchg_old(cmpxchg_op);

  loom_op_t* expected_bits_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(builder, expected, float_type,
                                                 integer_type, location,
                                                 &expected_bits_op));
  loom_op_t* observed_bits_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(builder, observed, float_type,
                                                 integer_type, location,
                                                 &observed_bits_op));
  loom_op_t* retry_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
      builder, LOOM_SCALAR_CMPI_PREDICATE_NE,
      loom_scalar_bitcast_result(observed_bits_op),
      loom_scalar_bitcast_result(expected_bits_op), location, &retry_op));

  loom_op_t* condition_op = NULL;
  return loom_scf_condition_build(builder, loom_scalar_cmpi_result(retry_op),
                                  &observed, 1, location, &condition_op);
}

static iree_status_t loom_view_legalize_build_atomic_addf_after_region(
    loom_builder_t* builder, loom_op_t* loop, loom_location_id_t location) {
  const loom_value_id_t observed =
      loom_region_entry_arg_id(loom_scf_while_after(loop), 0);
  loom_op_t* yield_op = NULL;
  return loom_scf_yield_build(builder, &observed, 1, location, &yield_op);
}

iree_status_t loom_view_target_legalize_atomic_addf_reference(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  const loom_memory_access_t access =
      loom_memory_access_cast(context->module, op);
  if (loom_attr_as_enum(loom_memory_access_atomic_kind(access)) !=
      LOOM_ATOMIC_KIND_ADDF) {
    return iree_ok_status();
  }

  const loom_type_t float_type =
      loom_module_value_type(context->module, loom_memory_access_value(access));
  if (!loom_type_equal(float_type, loom_type_scalar(LOOM_SCALAR_TYPE_F32))) {
    return iree_ok_status();
  }
  const loom_type_t integer_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* zero_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_constant_build(&rewriter->builder, loom_attr_f64(0.0),
                                 float_type, op->location, &zero_op));
  const loom_value_id_t initial_expected = loom_scalar_constant_result(zero_op);
  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_while_build(
      &rewriter->builder, &initial_expected, 1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, op->location, &loop));

  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &rewriter->builder, loop, loom_scf_while_before(loop));
  iree_status_t status = loom_view_legalize_build_atomic_addf_before_region(
      &rewriter->builder, loop, access, float_type, integer_type, op->location);
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  saved_ip = loom_builder_enter_region(&rewriter->builder, loop,
                                       loom_scf_while_after(loop));
  status = loom_view_legalize_build_atomic_addf_after_region(
      &rewriter->builder, loop, op->location);
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  if (loom_view_atomic_rmw_isa(op)) {
    const loom_value_id_t old_value = loom_scf_while_results(loop).values[0];
    IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
        rewriter, op, &old_value, 1, value_checkpoint));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_replace_all_uses_and_erase(rewriter, op, &old_value, 1));
  } else {
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
  }
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static iree_status_t loom_view_legalize_atomic_addf(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->contract_query_result->outcome !=
      LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED) {
    return iree_ok_status();
  }
  return loom_view_target_legalize_atomic_addf_reference(context, op,
                                                         out_result);
}

static const loom_target_legalizer_entry_t kViewLegalizerEntries[] = {
    {
        .root_kind = LOOM_OP_VIEW_ATOMIC_REDUCE,
        .legalize = loom_view_legalize_atomic_addf,
    },
    {
        .root_kind = LOOM_OP_VIEW_ATOMIC_RMW,
        .legalize = loom_view_legalize_atomic_addf,
    },
};

static const loom_target_legalizer_provider_t kViewLegalizerProvider = {
    .name = IREE_SVL("view"),
    .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE,
    .entries = kViewLegalizerEntries,
    .entry_count = IREE_ARRAYSIZE(kViewLegalizerEntries),
};

const loom_target_legalizer_provider_t* loom_view_target_legalizer_provider(
    void) {
  return &kViewLegalizerProvider;
}
