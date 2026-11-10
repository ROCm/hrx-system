// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/view/atomic.h"

#include "loom/ir/module.h"
#include "loom/ops/cache.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/view/ops.h"

// All arithmetic entries are attribute-free binary operations with one result.
// Zero entries are exchanges, which require no arithmetic operation.
static const loom_op_kind_t kAtomicCombines[LOOM_ATOMIC_KIND_COUNT_] = {
    [LOOM_ATOMIC_KIND_ADDI] = LOOM_OP_SCALAR_ADDI,
    [LOOM_ATOMIC_KIND_ADDF] = LOOM_OP_SCALAR_ADDF,
    [LOOM_ATOMIC_KIND_SUBI] = LOOM_OP_SCALAR_SUBI,
    [LOOM_ATOMIC_KIND_ANDI] = LOOM_OP_SCALAR_ANDI,
    [LOOM_ATOMIC_KIND_ORI] = LOOM_OP_SCALAR_ORI,
    [LOOM_ATOMIC_KIND_XORI] = LOOM_OP_SCALAR_XORI,
    [LOOM_ATOMIC_KIND_MINSI] = LOOM_OP_SCALAR_MINSI,
    [LOOM_ATOMIC_KIND_MAXSI] = LOOM_OP_SCALAR_MAXSI,
    [LOOM_ATOMIC_KIND_MINUI] = LOOM_OP_SCALAR_MINUI,
    [LOOM_ATOMIC_KIND_MAXUI] = LOOM_OP_SCALAR_MAXUI,
    [LOOM_ATOMIC_KIND_MINIMUMF] = LOOM_OP_SCALAR_MINIMUMF,
    [LOOM_ATOMIC_KIND_MAXIMUMF] = LOOM_OP_SCALAR_MAXIMUMF,
    [LOOM_ATOMIC_KIND_MINNUMF] = LOOM_OP_SCALAR_MINNUMF,
    [LOOM_ATOMIC_KIND_MAXNUMF] = LOOM_OP_SCALAR_MAXNUMF,
};

iree_status_t loom_view_atomic_build_combine(
    loom_builder_t* builder, loom_atomic_kind_t kind, loom_value_id_t old,
    loom_value_id_t value, loom_type_t type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  const loom_op_kind_t scalar_kind = kAtomicCombines[kind];
  if (!scalar_kind) {
    *out_value = value;
    return iree_ok_status();
  }
  loom_op_t* combine = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(builder, scalar_kind, 2, 1, 0,
                                                0, 0, location, &combine));
  loom_op_operands(combine)[0] = old;
  loom_op_operands(combine)[1] = value;
  IREE_RETURN_IF_ERROR(
      loom_builder_define_result(builder, type, &loom_op_results(combine)[0]));
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, combine));
  *out_value = loom_op_results(combine)[0];
  return iree_ok_status();
}

static iree_status_t loom_view_atomic_private_load(loom_builder_t* builder,
                                                   loom_memory_access_t access,
                                                   loom_type_t type,
                                                   loom_value_id_t* out_value) {
  const loom_value_slice_t indices = loom_memory_access_dynamic_indices(access);
  const loom_attribute_t static_indices =
      loom_memory_access_static_indices(access);
  const loom_cache_policy_t policy =
      loom_cache_policy_cast(builder->module, access.op);
  const loom_attribute_t scope = loom_cache_policy_scope(policy);
  const loom_attribute_t temporal = loom_cache_policy_temporal(policy);
  const loom_view_load_build_flags_t flags =
      loom_attr_is_absent(scope)
          ? 0
          : LOOM_VIEW_LOAD_BUILD_FLAG_HAS_CACHE_SCOPE |
                LOOM_VIEW_LOAD_BUILD_FLAG_HAS_CACHE_TEMPORAL;
  loom_op_t* load = NULL;
  IREE_RETURN_IF_ERROR(loom_view_load_build(
      builder, flags,
      loom_memory_access_flags(access) & LOOM_MEMORY_ACCESS_FLAG_ROOT_IN_BOUNDS,
      loom_memory_access_view(access), indices.values, indices.count,
      static_indices.i64_array, static_indices.count, loom_attr_as_enum(scope),
      loom_attr_as_enum(temporal), type, access.op->location, &load));
  *out_value = loom_view_load_result(load);
  return iree_ok_status();
}

static iree_status_t loom_view_atomic_private_store(loom_builder_t* builder,
                                                    loom_memory_access_t access,
                                                    loom_value_id_t value) {
  const loom_value_slice_t indices = loom_memory_access_dynamic_indices(access);
  const loom_attribute_t static_indices =
      loom_memory_access_static_indices(access);
  const loom_cache_policy_t policy =
      loom_cache_policy_cast(builder->module, access.op);
  const loom_attribute_t scope = loom_cache_policy_scope(policy);
  const loom_attribute_t temporal = loom_cache_policy_temporal(policy);
  const loom_view_store_build_flags_t flags =
      loom_attr_is_absent(scope)
          ? 0
          : LOOM_VIEW_STORE_BUILD_FLAG_HAS_CACHE_SCOPE |
                LOOM_VIEW_STORE_BUILD_FLAG_HAS_CACHE_TEMPORAL;
  loom_op_t* store = NULL;
  return loom_view_store_build(
      builder, flags,
      loom_memory_access_flags(access) & LOOM_MEMORY_ACCESS_FLAG_ROOT_IN_BOUNDS,
      value, loom_memory_access_view(access), indices.values, indices.count,
      static_indices.i64_array, static_indices.count, loom_attr_as_enum(scope),
      loom_attr_as_enum(temporal), access.op->location, &store);
}

static iree_status_t loom_view_atomic_payload_bits(loom_builder_t* builder,
                                                   loom_value_id_t value,
                                                   loom_type_t type,
                                                   loom_location_id_t location,
                                                   loom_value_id_t* out_bits) {
  const loom_scalar_type_t element = loom_type_element_type(type);
  if (!loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_FLOAT, element)) {
    *out_bits = value;
    return iree_ok_status();
  }
  static const loom_scalar_type_t kIntegerTypes[LOOM_SCALAR_TYPE_COUNT_] = {
      [LOOM_SCALAR_TYPE_F8E4M3] = LOOM_SCALAR_TYPE_I8,
      [LOOM_SCALAR_TYPE_F8E5M2] = LOOM_SCALAR_TYPE_I8,
      [LOOM_SCALAR_TYPE_F16] = LOOM_SCALAR_TYPE_I16,
      [LOOM_SCALAR_TYPE_BF16] = LOOM_SCALAR_TYPE_I16,
      [LOOM_SCALAR_TYPE_F32] = LOOM_SCALAR_TYPE_I32,
      [LOOM_SCALAR_TYPE_F64] = LOOM_SCALAR_TYPE_I64,
  };
  loom_op_t* cast = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_bitcast_build(
      builder, value, type, loom_type_scalar(kIntegerTypes[element]), location,
      &cast));
  *out_bits = loom_scalar_bitcast_result(cast);
  return iree_ok_status();
}

iree_status_t loom_view_atomic_rewrite_private(loom_rewriter_t* rewriter,
                                               loom_op_t* op) {
  const loom_memory_access_t access =
      loom_memory_access_cast(rewriter->module, op);
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t value = loom_memory_access_value(access);
  loom_value_id_t old = LOOM_VALUE_ID_INVALID;
  if (!loom_view_atomic_store_isa(op)) {
    const loom_type_t type = loom_module_value_type(
        rewriter->module, op->result_count ? loom_op_results(op)[0] : value);
    IREE_RETURN_IF_ERROR(
        loom_view_atomic_private_load(builder, access, type, &old));
    if (loom_view_atomic_cmpxchg_isa(op)) {
      loom_value_id_t old_bits = LOOM_VALUE_ID_INVALID;
      loom_value_id_t expected_bits = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_view_atomic_payload_bits(
          builder, old, type, op->location, &old_bits));
      IREE_RETURN_IF_ERROR(loom_view_atomic_payload_bits(
          builder, loom_memory_access_expected(access), type, op->location,
          &expected_bits));
      loom_op_t* compare = NULL;
      IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
          builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, old_bits, expected_bits,
          op->location, &compare));
      loom_op_t* select = NULL;
      IREE_RETURN_IF_ERROR(
          loom_scf_select_build(builder, loom_scalar_cmpi_result(compare),
                                loom_memory_access_replacement(access), old,
                                type, op->location, &select));
      value = loom_scf_select_result(select);
    } else if (!loom_view_atomic_load_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_view_atomic_build_combine(
          builder,
          (loom_atomic_kind_t)loom_attr_as_enum(
              loom_memory_access_atomic_kind(access)),
          old, value, type, op->location, &value));
    }
  }
  if (!loom_view_atomic_load_isa(op)) {
    IREE_RETURN_IF_ERROR(
        loom_view_atomic_private_store(builder, access, value));
  }
  if (op->result_count) {
    IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
        rewriter, op, &old, 1, checkpoint));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_replace_all_uses_and_erase(rewriter, op, &old, 1));
  } else {
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
  }
  return iree_ok_status();
}

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
loom_view_atomic_cmpxchg_cache_policy(loom_cache_policy_t policy,
                                      uint8_t* out_cache_scope,
                                      uint8_t* out_cache_temporal) {
  loom_view_atomic_cmpxchg_build_flags_t build_flags = 0;
  const loom_attribute_t cache_scope = loom_cache_policy_scope(policy);
  if (!loom_attr_is_absent(cache_scope)) {
    build_flags |= LOOM_VIEW_ATOMIC_CMPXCHG_BUILD_FLAG_HAS_CACHE_SCOPE;
    *out_cache_scope = loom_attr_as_enum(cache_scope);
  }
  const loom_attribute_t cache_temporal = loom_cache_policy_temporal(policy);
  if (!loom_attr_is_absent(cache_temporal)) {
    build_flags |= LOOM_VIEW_ATOMIC_CMPXCHG_BUILD_FLAG_HAS_CACHE_TEMPORAL;
    *out_cache_temporal = loom_attr_as_enum(cache_temporal);
  }
  return build_flags;
}

static iree_status_t loom_view_atomic_build_cmpxchg_before_region(
    loom_builder_t* builder, loom_op_t* loop, loom_memory_access_t access,
    loom_type_t value_type, loom_atomic_kind_t kind,
    loom_location_id_t location) {
  const loom_value_id_t expected =
      loom_region_entry_arg_id(loom_scf_while_before(loop), 0);
  loom_value_id_t combined = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_view_atomic_build_combine(
      builder, kind, expected, loom_memory_access_value(access), value_type,
      location, &combined));

  uint8_t cache_scope = 0;
  uint8_t cache_temporal = 0;
  const loom_view_atomic_cmpxchg_build_flags_t build_flags =
      loom_view_atomic_cmpxchg_cache_policy(
          loom_cache_policy_cast(builder->module, access.op), &cache_scope,
          &cache_temporal);
  const loom_value_slice_t indices = loom_memory_access_dynamic_indices(access);
  const loom_attribute_t static_indices =
      loom_memory_access_static_indices(access);
  const loom_atomic_ordering_t success_ordering =
      (loom_atomic_ordering_t)loom_attr_as_enum(
          loom_memory_access_atomic_ordering(access));
  loom_op_t* cmpxchg_op = NULL;
  IREE_RETURN_IF_ERROR(loom_view_atomic_cmpxchg_build(
      builder, build_flags,
      loom_memory_access_flags(access) & LOOM_MEMORY_ACCESS_FLAG_ROOT_IN_BOUNDS,
      expected, combined, loom_memory_access_view(access), indices.values,
      indices.count, static_indices.i64_array, static_indices.count,
      success_ordering, kAtomicRmwFailureOrderings[success_ordering],
      (loom_atomic_scope_t)loom_attr_as_enum(
          loom_memory_access_atomic_scope(access)),
      cache_scope, cache_temporal, value_type, location, &cmpxchg_op));
  const loom_value_id_t observed = loom_view_atomic_cmpxchg_old(cmpxchg_op);

  loom_value_id_t expected_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_view_atomic_payload_bits(
      builder, expected, value_type, location, &expected_bits));
  loom_value_id_t observed_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_view_atomic_payload_bits(
      builder, observed, value_type, location, &observed_bits));
  loom_op_t* retry_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
      builder, LOOM_SCALAR_CMPI_PREDICATE_NE, observed_bits, expected_bits,
      location, &retry_op));

  loom_op_t* condition_op = NULL;
  return loom_scf_condition_build(builder, loom_scalar_cmpi_result(retry_op),
                                  &observed, 1, location, &condition_op);
}

static iree_status_t loom_view_atomic_build_cmpxchg_after_region(
    loom_builder_t* builder, loom_op_t* loop, loom_location_id_t location) {
  const loom_value_id_t observed =
      loom_region_entry_arg_id(loom_scf_while_after(loop), 0);
  loom_op_t* yield_op = NULL;
  return loom_scf_yield_build(builder, &observed, 1, location, &yield_op);
}

iree_status_t loom_view_atomic_rewrite_cmpxchg(loom_rewriter_t* rewriter,
                                               loom_op_t* op) {
  const loom_memory_access_t access =
      loom_memory_access_cast(rewriter->module, op);
  const loom_atomic_kind_t kind = (loom_atomic_kind_t)loom_attr_as_enum(
      loom_memory_access_atomic_kind(access));
  const loom_type_t value_type = loom_module_value_type(
      rewriter->module, loom_memory_access_value(access));
  const loom_attribute_t zero =
      loom_scalar_type_is_float(loom_type_element_type(value_type))
          ? loom_attr_f64(0.0)
          : loom_attr_i64(0);
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* zero_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      &rewriter->builder, zero, value_type, op->location, &zero_op));
  const loom_value_id_t initial_expected = loom_scalar_constant_result(zero_op);
  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_while_build(&rewriter->builder, &initial_expected, 1,
                           /*iter_args_types=*/NULL,
                           /*result_types=*/NULL, /*result_count=*/1,
                           /*tied_results=*/NULL,
                           /*tied_result_count=*/0, op->location, &loop));

  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      &rewriter->builder, loop, loom_scf_while_before(loop));
  iree_status_t status = loom_view_atomic_build_cmpxchg_before_region(
      &rewriter->builder, loop, access, value_type, kind, op->location);
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  saved_ip = loom_builder_enter_region(&rewriter->builder, loop,
                                       loom_scf_while_after(loop));
  status = loom_view_atomic_build_cmpxchg_after_region(&rewriter->builder, loop,
                                                       op->location);
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
  return iree_ok_status();
}
