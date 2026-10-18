// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ir/module.h"
#include "loom/ops/atomic.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/fragment_memory/plan.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/memory_packet.h"
#include "loom/target/arch/amdgpu/lower/sanitizer.h"
#include "loom/target/arch/amdgpu/lower/sanitizer_race.h"
#include "loom/target/arch/amdgpu/lower/source_alloca_layout.h"
#include "loom/target/arch/amdgpu/lower/sync.h"
#include "loom/target/arch/amdgpu/lower/system_memory.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

typedef enum loom_amdgpu_sanitizer_race_workgroup_support_e {
  LOOM_AMDGPU_SANITIZER_RACE_WORKGROUP_SUPPORTED = 0,
  LOOM_AMDGPU_SANITIZER_RACE_WORKGROUP_FIXED_SIZE_REQUIRED,
  LOOM_AMDGPU_SANITIZER_RACE_WORKGROUP_WORKITEM_CAPACITY_EXCEEDED,
} loom_amdgpu_sanitizer_race_workgroup_support_t;

static loom_amdgpu_sanitizer_race_workgroup_support_t
loom_amdgpu_sanitizer_race_workgroup_support(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle) {
  loom_target_workgroup_size_t workgroup_size = {0};
  if (!loom_amdgpu_required_workgroup_size(module, function, bundle,
                                           &workgroup_size) ||
      workgroup_size.x == 0 || workgroup_size.y == 0 || workgroup_size.z == 0) {
    return LOOM_AMDGPU_SANITIZER_RACE_WORKGROUP_FIXED_SIZE_REQUIRED;
  }

  const uint64_t flat_workgroup_size =
      (uint64_t)workgroup_size.x * workgroup_size.y * workgroup_size.z;
  if (flat_workgroup_size > LOOM_AMDGPU_TSAN_SHADOW_ENTRY_WORKITEM_CAPACITY) {
    return LOOM_AMDGPU_SANITIZER_RACE_WORKGROUP_WORKITEM_CAPACITY_EXCEEDED;
  }
  return LOOM_AMDGPU_SANITIZER_RACE_WORKGROUP_SUPPORTED;
}

static iree_status_t loom_amdgpu_sanitizer_race_verify_workgroup(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  switch (loom_amdgpu_sanitizer_race_workgroup_support(
      loom_target_low_legality_module(context),
      loom_target_low_legality_function(context),
      loom_target_low_legality_bundle(context))) {
    case LOOM_AMDGPU_SANITIZER_RACE_WORKGROUP_SUPPORTED:
      return iree_ok_status();
    case LOOM_AMDGPU_SANITIZER_RACE_WORKGROUP_FIXED_SIZE_REQUIRED:
      return loom_amdgpu_low_legality_reject(
          context, op,
          IREE_SV("target_contract.sanitizer_race.fixed_workgroup_required"));
    case LOOM_AMDGPU_SANITIZER_RACE_WORKGROUP_WORKITEM_CAPACITY_EXCEEDED:
      return loom_amdgpu_low_legality_reject(
          context, op,
          IREE_SV("target_contract.sanitizer_race.workitem_capacity"));
    default:
      IREE_ASSERT_UNREACHABLE("valid AMDGPU race sanitizer workgroup support");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static bool loom_amdgpu_sanitizer_race_required_descriptors_present(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_descriptor_ref_t required_refs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B32_SADDR,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B128_SADDR,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_CMPSWAP_B64_RTN_SADDR,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_OR_B32_RTN_SADDR,
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_OR_B32_SADDR,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_CO_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_SUB_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32_RHS_SYMBOL_REL32_LO,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32_RHS_SYMBOL_REL32_HI,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_I32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_U64,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LT_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LE_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_LG_U64,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_CSELECT_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_GETPC_B64,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LOAD_DWORD_OFFSET_ONLY,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHL_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHR_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHR_B64_RHS_INLINE,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_HI_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_I32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32_M0_IMM,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_CI_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_HI_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32_LIT,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_READFIRSTLANE_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_CO_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_CO_CI_U32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32,
  };
  if (!loom_amdgpu_descriptor_set_has_all_refs(descriptor_set, required_refs,
                                               IREE_ARRAYSIZE(required_refs))) {
    return false;
  }
  if (!loom_amdgpu_system_memory_release_ordering_available(descriptor_set) ||
      !loom_amdgpu_system_memory_acquire_ordering_available(descriptor_set)) {
    return false;
  }
  return loom_amdgpu_descriptor_set_has_ref(
             descriptor_set,
             LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U32_RTN_SADDR) ||
         loom_amdgpu_descriptor_set_has_ref(
             descriptor_set,
             LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_ADD_U64_RTN_SADDR);
}

static iree_string_view_t
loom_amdgpu_sanitizer_race_source_memory_rejection_key(
    const loom_low_source_memory_access_diagnostic_t* diagnostic) {
  if (diagnostic->rejection_bits != 0) {
    return loom_low_source_memory_access_rejection_key(
        diagnostic->rejection_bits);
  }
  return IREE_SV("target_contract.sanitizer_race.source_memory_access");
}

static bool loom_amdgpu_sanitizer_race_access_source_kind(
    loom_sanitizer_race_access_kind_t kind,
    loom_low_source_memory_operation_kind_t* out_source_kind) {
  switch (kind) {
    case LOOM_SANITIZER_RACE_ACCESS_KIND_READ:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
      return true;
    case LOOM_SANITIZER_RACE_ACCESS_KIND_WRITE:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE;
      return true;
    case LOOM_SANITIZER_RACE_ACCESS_KIND_READ_WRITE:
      *out_source_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_ATOMIC_RMW;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_sanitizer_race_access_report_kind(
    loom_sanitizer_race_access_kind_t kind, bool atomic,
    loom_amdgpu_tsan_access_kind_t* out_report_kind,
    loom_amdgpu_tsan_shadow_access_kind_t* out_shadow_kind) {
  if (atomic) {
    *out_report_kind = LOOM_AMDGPU_TSAN_ACCESS_KIND_ATOMIC;
    *out_shadow_kind = LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_ATOMIC;
    return true;
  }
  switch (kind) {
    case LOOM_SANITIZER_RACE_ACCESS_KIND_READ:
      *out_report_kind = LOOM_AMDGPU_TSAN_ACCESS_KIND_READ;
      *out_shadow_kind = LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ;
      return true;
    case LOOM_SANITIZER_RACE_ACCESS_KIND_WRITE:
      *out_report_kind = LOOM_AMDGPU_TSAN_ACCESS_KIND_WRITE;
      *out_shadow_kind = LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_WRITE;
      return true;
    case LOOM_SANITIZER_RACE_ACCESS_KIND_READ_WRITE:
      *out_report_kind = LOOM_AMDGPU_TSAN_ACCESS_KIND_READ_WRITE;
      *out_shadow_kind = LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ_WRITE;
      return true;
    default:
      *out_report_kind = LOOM_AMDGPU_TSAN_ACCESS_KIND_UNKNOWN;
      *out_shadow_kind = LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_EMPTY;
      return false;
  }
}

static void loom_amdgpu_sanitizer_race_fragment_access_report_kind(
    loom_sanitizer_race_fragment_access_kind_t kind,
    loom_amdgpu_tsan_access_kind_t* out_report_kind,
    loom_amdgpu_tsan_shadow_access_kind_t* out_shadow_kind) {
  switch (kind) {
    case LOOM_SANITIZER_RACE_FRAGMENT_ACCESS_KIND_READ:
      *out_report_kind = LOOM_AMDGPU_TSAN_ACCESS_KIND_READ;
      *out_shadow_kind = LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_READ;
      return;
    case LOOM_SANITIZER_RACE_FRAGMENT_ACCESS_KIND_WRITE:
      *out_report_kind = LOOM_AMDGPU_TSAN_ACCESS_KIND_WRITE;
      *out_shadow_kind = LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_WRITE;
      return;
    default:
      IREE_ASSERT_UNREACHABLE("verified sanitizer fragment access kind");
      *out_report_kind = LOOM_AMDGPU_TSAN_ACCESS_KIND_UNKNOWN;
      *out_shadow_kind = LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_EMPTY;
      return;
  }
}

static bool loom_amdgpu_sanitizer_race_access_payload_type(
    const loom_module_t* module, loom_value_id_t view_value_id,
    loom_type_t* out_vector_type) {
  *out_vector_type = loom_type_none();
  if (view_value_id >= module->values.count) {
    return false;
  }
  const loom_type_t view_type = loom_module_value_type(module, view_value_id);
  if (!loom_type_is_view(view_type)) {
    return false;
  }
  *out_vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, loom_type_element_type(view_type),
                          loom_dim_pack_static(1), /*encoding_id=*/0);
  return true;
}

static iree_status_t loom_amdgpu_sanitizer_race_reject_memory_selection(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_low_source_memory_access_plan_t* source,
    const loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  bool handled = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_memory_access_rejection_diagnostic(
      context, op, source, diagnostic, &handled));
  if (handled) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_reject(
      context, op,
      loom_amdgpu_memory_access_rejection_key(diagnostic->rejection_bits));
}

static bool loom_amdgpu_sanitizer_race_access_plan_build(
    const loom_module_t* module, const loom_view_region_table_t* view_regions,
    const loom_amdgpu_source_alloca_layout_t* alloca_layout,
    const loom_op_t* op, loom_low_source_memory_access_plan_t* out_source,
    loom_amdgpu_sanitizer_race_access_plan_t* out_plan,
    loom_low_source_memory_access_diagnostic_t* out_source_diagnostic,
    loom_amdgpu_memory_access_diagnostic_t* out_memory_diagnostic) {
  *out_plan = (loom_amdgpu_sanitizer_race_access_plan_t){0};
  *out_source_diagnostic = (loom_low_source_memory_access_diagnostic_t){0};
  *out_memory_diagnostic = (loom_amdgpu_memory_access_diagnostic_t){0};

  loom_low_source_memory_operation_kind_t source_kind =
      LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  if (!loom_amdgpu_sanitizer_race_access_source_kind(
          loom_sanitizer_race_access_kind(op), &source_kind)) {
    return false;
  }

  const loom_value_id_t view_value_id = loom_sanitizer_race_access_view(op);
  loom_type_t vector_type = loom_type_none();
  if (!loom_amdgpu_sanitizer_race_access_payload_type(module, view_value_id,
                                                      &vector_type)) {
    out_source_diagnostic->rejection_bits =
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_LAYOUT;
    return false;
  }

  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_low_source_memory_access_plan_build_indexed(
          view_regions, source_kind, view_value_id,
          loom_sanitizer_race_access_indices(op),
          loom_sanitizer_race_access_static_indices(op), vector_type,
          cache_policy, /*axis_byte_strides=*/NULL, out_source,
          out_source_diagnostic)) {
    return false;
  }
  out_plan->address = loom_amdgpu_memory_access_from_source(out_source);
  if (out_source->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    out_memory_diagnostic->rejection_bits =
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_MEMORY_SPACE;
    return false;
  }

  loom_amdgpu_memory_access_t access = {0};
  if (!loom_amdgpu_memory_access_select_u32_vaddr_byte_offset(
          module, alloca_layout, out_source, &access, out_memory_diagnostic)) {
    return false;
  }

  loom_amdgpu_tsan_access_kind_t report_kind =
      LOOM_AMDGPU_TSAN_ACCESS_KIND_UNKNOWN;
  loom_amdgpu_tsan_shadow_access_kind_t shadow_kind =
      LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_EMPTY;
  if (!loom_amdgpu_sanitizer_race_access_report_kind(
          loom_sanitizer_race_access_kind(op),
          loom_sanitizer_race_access_atomic(op), &report_kind, &shadow_kind)) {
    out_source_diagnostic->rejection_bits =
        LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_UNSUPPORTED_OP;
    return false;
  }

  *out_plan = (loom_amdgpu_sanitizer_race_access_plan_t){
      .observation =
          {
              .report_access_kind = report_kind,
              .shadow_access_kind = shadow_kind,
              .atomic = loom_sanitizer_race_access_atomic(op),
          },
      .address = access,
      .access_size =
          out_source->element_byte_count * out_source->vector_lane_count,
  };
  return true;
}

static iree_status_t loom_amdgpu_sanitizer_race_verify_access_address(
    loom_target_low_legality_context_t* context, const loom_op_t* op) {
  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_view_region_table_t* view_regions =
      loom_target_low_legality_view_regions(context);
  const loom_amdgpu_source_alloca_layout_t* alloca_layout = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_for_low_legality(
      context, &alloca_layout));
  loom_amdgpu_sanitizer_race_access_plan_t plan = {0};
  loom_low_source_memory_access_plan_t source = {0};
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_amdgpu_memory_access_diagnostic_t memory_diagnostic = {0};
  if (!loom_amdgpu_sanitizer_race_access_plan_build(
          module, view_regions, alloca_layout, op, &source, &plan,
          &source_diagnostic, &memory_diagnostic)) {
    if (source_diagnostic.rejection_bits != 0) {
      if (iree_any_bit_set(source_diagnostic.rejection_bits,
                           LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_LAYOUT)) {
        return loom_amdgpu_low_legality_reject(
            context, op,
            IREE_SV("target_contract.sanitizer_race.payload_type"));
      }
      if (iree_any_bit_set(
              source_diagnostic.rejection_bits,
              LOOM_LOW_SOURCE_MEMORY_ACCESS_REJECTION_UNSUPPORTED_OP)) {
        return loom_amdgpu_low_legality_reject(
            context, op, IREE_SV("target_contract.sanitizer_race.access_kind"));
      }
    }
    if (memory_diagnostic.rejection_bits ==
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_MEMORY_SPACE) {
      return loom_amdgpu_low_legality_reject(
          context, op,
          IREE_SV("target_contract.sanitizer_race.workgroup_memory_required"));
    }
    if (memory_diagnostic.rejection_bits != 0) {
      return loom_amdgpu_sanitizer_race_reject_memory_selection(
          context, op, plan.address.source, &memory_diagnostic);
    }
    return loom_amdgpu_low_legality_reject(
        context, op,
        loom_amdgpu_sanitizer_race_source_memory_rejection_key(
            &source_diagnostic));
  }
  return loom_amdgpu_sanitizer_race_verify_workgroup(context, op);
}

iree_status_t loom_amdgpu_select_sanitizer_race_access_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_sanitizer_race_access_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_sanitizer_race_access_plan_t){0};
  *out_selected = false;
  if (!loom_sanitizer_race_access_isa(source_op)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_sanitizer_race_required_descriptors_present(
          loom_low_lower_context_descriptor_set(context))) {
    return iree_ok_status();
  }
  if (loom_amdgpu_sanitizer_race_workgroup_support(
          loom_low_lower_context_module(context),
          loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context)) !=
      LOOM_AMDGPU_SANITIZER_RACE_WORKGROUP_SUPPORTED) {
    return iree_ok_status();
  }

  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  const loom_amdgpu_source_alloca_layout_t* alloca_layout = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_for_lower_context(
      context, &alloca_layout));
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  loom_amdgpu_memory_access_diagnostic_t memory_diagnostic = {0};
  loom_low_source_memory_access_plan_t source = {0};
  if (!loom_amdgpu_sanitizer_race_access_plan_build(
          loom_low_lower_context_module(context), view_regions, alloca_layout,
          source_op, &source, out_plan, &source_diagnostic,
          &memory_diagnostic)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_site_id_for_op(
      context, source_op, &out_plan->observation.site_id));
  if (out_plan->observation.site_id == LOOM_SANITIZER_SITE_ID_INVALID) {
    return iree_ok_status();
  }
  loom_low_source_memory_access_plan_t* retained_source = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*retained_source), (void**)&retained_source));
  *retained_source = source;
  out_plan->address.source = retained_source;
  IREE_RETURN_IF_ERROR(loom_amdgpu_plan_memory_dynamic_terms(
      context, retained_source, &out_plan->address.dynamic_term_plans));
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_sanitizer_race_fragment_access_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_sanitizer_race_fragment_access_plan_t* out_plan,
    bool* out_selected) {
  *out_plan = (loom_amdgpu_sanitizer_race_fragment_access_plan_t){0};
  *out_selected = false;
  if (!loom_sanitizer_race_fragment_access_isa(source_op)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_sanitizer_race_required_descriptors_present(
          loom_low_lower_context_descriptor_set(context)) ||
      loom_amdgpu_sanitizer_race_workgroup_support(
          loom_low_lower_context_module(context),
          loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context)) !=
          LOOM_AMDGPU_SANITIZER_RACE_WORKGROUP_SUPPORTED) {
    return iree_ok_status();
  }

  bool fragment_memory_selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_sanitizer_race_fragment_memory_plan(
      context, source_op, &out_plan->fragment_memory,
      &fragment_memory_selected));
  if (!fragment_memory_selected ||
      out_plan->fragment_memory.source.memory_space !=
          LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return iree_ok_status();
  }

  loom_amdgpu_tsan_access_kind_t report_kind =
      LOOM_AMDGPU_TSAN_ACCESS_KIND_UNKNOWN;
  loom_amdgpu_tsan_shadow_access_kind_t shadow_kind =
      LOOM_AMDGPU_TSAN_SHADOW_ACCESS_KIND_EMPTY;
  loom_amdgpu_sanitizer_race_fragment_access_report_kind(
      loom_sanitizer_race_fragment_access_kind(source_op), &report_kind,
      &shadow_kind);
  out_plan->observation = (loom_amdgpu_sanitizer_race_observation_plan_t){
      .report_access_kind = report_kind,
      .shadow_access_kind = shadow_kind,
      .atomic = false,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_sanitizer_site_id_for_op(
      context, source_op, &out_plan->observation.site_id));
  if (out_plan->observation.site_id == LOOM_SANITIZER_SITE_ID_INVALID) {
    return iree_ok_status();
  }
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_sanitizer_race_sync_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_sanitizer_race_sync_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_sanitizer_race_sync_plan_t){0};
  *out_selected = false;
  if (!loom_sanitizer_race_sync_isa(source_op)) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_sanitizer_race_required_descriptors_present(
          loom_low_lower_context_descriptor_set(context))) {
    return iree_ok_status();
  }
  if (loom_sanitizer_race_sync_memory_space(source_op) !=
          LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
      loom_sanitizer_race_sync_scope(source_op) !=
          LOOM_ATOMIC_SCOPE_WORKGROUP) {
    return iree_ok_status();
  }
  const loom_atomic_ordering_t ordering =
      loom_sanitizer_race_sync_ordering(source_op);
  if (ordering != LOOM_ATOMIC_ORDERING_ACQ_REL &&
      ordering != LOOM_ATOMIC_ORDERING_SEQ_CST) {
    return iree_ok_status();
  }
  if (loom_amdgpu_sanitizer_race_workgroup_support(
          loom_low_lower_context_module(context),
          loom_low_lower_context_source_function(context),
          loom_low_lower_context_bundle(context)) !=
      LOOM_AMDGPU_SANITIZER_RACE_WORKGROUP_SUPPORTED) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_select_workgroup_barrier_plan(context, &out_plan->barrier));
  if (out_plan->barrier.kind == LOOM_AMDGPU_KERNEL_BARRIER_LOWERING_KIND_NONE) {
    return iree_ok_status();
  }
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_sanitizer_race_access(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  *out_handled = false;
  if (!loom_sanitizer_race_access_isa(op)) {
    return iree_ok_status();
  }
  *out_handled = true;
  return loom_amdgpu_sanitizer_race_verify_access_address(context, op);
}

iree_status_t loom_amdgpu_low_legality_verify_sanitizer_race_fragment_access(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  *out_handled = false;
  if (!loom_sanitizer_race_fragment_access_isa(op)) {
    return iree_ok_status();
  }

  bool fragment_memory_handled = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_legality_verify_fragment_memory(
      provider, context, op, &fragment_memory_handled));
  IREE_ASSERT_TRUE(fragment_memory_handled);
  *out_handled = true;
  return loom_amdgpu_sanitizer_race_verify_workgroup(context, op);
}

iree_status_t loom_amdgpu_low_legality_verify_sanitizer_race_sync(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  *out_handled = false;
  if (!loom_sanitizer_race_sync_isa(op)) {
    return iree_ok_status();
  }
  *out_handled = true;

  if (loom_sanitizer_race_sync_memory_space(op) !=
      LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        IREE_SV("target_contract.sanitizer_race.workgroup_memory_required"));
  }
  if (loom_sanitizer_race_sync_scope(op) != LOOM_ATOMIC_SCOPE_WORKGROUP) {
    return loom_amdgpu_low_legality_reject(
        context, op,
        IREE_SV("target_contract.sanitizer_race.workgroup_scope_required"));
  }

  switch (loom_sanitizer_race_sync_ordering(op)) {
    case LOOM_ATOMIC_ORDERING_ACQ_REL:
    case LOOM_ATOMIC_ORDERING_SEQ_CST:
      break;
    case LOOM_ATOMIC_ORDERING_ACQUIRE:
    case LOOM_ATOMIC_ORDERING_RELEASE:
    case LOOM_ATOMIC_ORDERING_RELAXED:
    case LOOM_ATOMIC_ORDERING_COUNT_:
    default:
      return loom_amdgpu_low_legality_reject(
          context, op,
          IREE_SV("target_contract.sanitizer_race.acq_rel_sync_required"));
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_sanitizer_race_verify_workgroup(context, op));
  if (!loom_amdgpu_workgroup_barrier_lowering_available(
          loom_target_low_legality_descriptor_set(context))) {
    return loom_amdgpu_low_legality_reject(
        context, op, IREE_SV("descriptor.workgroup_barrier"));
  }
  return iree_ok_status();
}
