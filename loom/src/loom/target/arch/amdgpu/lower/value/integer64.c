// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/value/integer64.h"

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/types.h"

typedef struct loom_amdgpu_descriptor_requirement_span_t {
  // Descriptor requirements required by this span.
  const loom_amdgpu_descriptor_requirement_t* requirements;
  // Number of entries in requirements.
  iree_host_size_t requirement_count;
} loom_amdgpu_descriptor_requirement_span_t;

typedef struct loom_amdgpu_i64_alu_descriptor_requirement_row_t {
  // First descriptor requirement span for the selected operation kind.
  loom_amdgpu_descriptor_requirement_span_t first;
  // Optional second descriptor requirement span for fused operation kinds.
  loom_amdgpu_descriptor_requirement_span_t second;
  // Table-specific operation properties interpreted by the owning row set.
  uint8_t flags;
} loom_amdgpu_i64_alu_descriptor_requirement_row_t;

enum {
  LOOM_AMDGPU_ADDRESS_I64_ALU_ROW_FLAG_USES_VGPR = 1u << 0,
};

#define LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(requirements_) \
  {                                                            \
      .requirements = requirements_,                           \
      .requirement_count = IREE_ARRAYSIZE(requirements_),      \
  }

static bool loom_amdgpu_descriptor_requirement_span_present(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_requirement_span_t span,
    iree_string_view_t* out_constraint_key) {
  if (span.requirement_count == 0) {
    return true;
  }
  IREE_ASSERT(span.requirements != NULL);
  return loom_amdgpu_descriptor_requirements_present(
      descriptor_set, span.requirements, span.requirement_count,
      out_constraint_key);
}

static bool loom_amdgpu_i64_alu_descriptor_requirements_present(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_i64_alu_descriptor_requirement_row_t* rows,
    iree_host_size_t row_count, iree_host_size_t row_index,
    iree_string_view_t fallback_constraint_key,
    iree_string_view_t* out_constraint_key) {
  if (row_index >= row_count || rows[row_index].first.requirement_count == 0) {
    *out_constraint_key = fallback_constraint_key;
    return false;
  }
  const loom_amdgpu_i64_alu_descriptor_requirement_row_t* row =
      &rows[row_index];
  return loom_amdgpu_descriptor_requirement_span_present(
             descriptor_set, row->first, out_constraint_key) &&
         loom_amdgpu_descriptor_requirement_span_present(
             descriptor_set, row->second, out_constraint_key);
}

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuOffsetAddVgprDescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32_copy"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_add_co_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_U32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_add_co_ci_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_CO_CI_U32,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuOffsetAddSgprDescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.s_mov_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.s_add_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.s_addc_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuScalarI64SubVgprDescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32_copy"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_sub_co_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_CO_U32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_sub_co_ci_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_SUB_CO_CI_U32,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuScalarI64ShlVgprDescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32_copy"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_lshlrev_b64"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B64,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuScalarI64MulVgprDescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.v_mul_lo_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_LO_U32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_mul_hi_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_HI_U32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_add_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuVgprMoveDescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32_copy"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_COPY,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuScalarI64CtpopSgprB32DescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.s_bcnt1_i32_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_BCNT1_I32_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.s_mov_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuScalarI64CtpopSgprB64DescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.s_bcnt1_i32_b64"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_BCNT1_I32_B64,
        },
        {
            .constraint_key = IREE_SVL("descriptor.s_mov_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuScalarI64CtpopVgprB32DescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.v_bcnt_u32_b32.src1_zero"),
            .descriptor_ref =
                LOOM_AMDGPU_DESCRIPTOR_REF_V_BCNT_U32_B32_SRC1_ZERO,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuScalarI64CtpopVgprB64DescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.v_bcnt_u32_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_BCNT_U32_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_bcnt_u32_b32.src1_zero"),
            .descriptor_ref =
                LOOM_AMDGPU_DESCRIPTOR_REF_V_BCNT_U32_B32_SRC1_ZERO,
        },
        {
            .constraint_key = IREE_SVL("descriptor.v_mov_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuI64CompareHighEqualDescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.v_cmp_eq_i32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.s_and_b64"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
        },
};

static const loom_amdgpu_i64_alu_descriptor_requirement_row_t
    kAmdgpuScalarI64AluDescriptorRequirementRows
        [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SHL + 1] = {
            [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_ADD] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuOffsetAddVgprDescriptorRequirements),
                },
            [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SUB] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuScalarI64SubVgprDescriptorRequirements),
                },
            [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_MUL_LO] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuScalarI64MulVgprDescriptorRequirements),
                },
            [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SHL] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuScalarI64ShlVgprDescriptorRequirements),
                },
};

static const loom_amdgpu_i64_alu_descriptor_requirement_row_t
    kAmdgpuAddressI64AluDescriptorRequirementRows
        [LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MADD_LO + 1] = {
            [LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_SGPR_ADD] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuOffsetAddSgprDescriptorRequirements),
                },
            [LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_ADD] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuOffsetAddVgprDescriptorRequirements),
                    .flags = LOOM_AMDGPU_ADDRESS_I64_ALU_ROW_FLAG_USES_VGPR,
                },
            [LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_SUB] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuScalarI64SubVgprDescriptorRequirements),
                    .flags = LOOM_AMDGPU_ADDRESS_I64_ALU_ROW_FLAG_USES_VGPR,
                },
            [LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MUL_LO] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuScalarI64MulVgprDescriptorRequirements),
                    .flags = LOOM_AMDGPU_ADDRESS_I64_ALU_ROW_FLAG_USES_VGPR,
                },
            [LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_SHL] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuScalarI64ShlVgprDescriptorRequirements),
                    .flags = LOOM_AMDGPU_ADDRESS_I64_ALU_ROW_FLAG_USES_VGPR,
                },
            [LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MADD_LO] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuScalarI64MulVgprDescriptorRequirements),
                    .second = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuOffsetAddVgprDescriptorRequirements),
                    .flags = LOOM_AMDGPU_ADDRESS_I64_ALU_ROW_FLAG_USES_VGPR,
                },
};

static const loom_amdgpu_descriptor_requirement_span_t
    kAmdgpuScalarI64CtpopDescriptorRequirementRows
        [LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_VGPR_B64 + 1] = {
            [LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_SGPR_B32] =
                LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                    kAmdgpuScalarI64CtpopSgprB32DescriptorRequirements),
            [LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_SGPR_B64] =
                LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                    kAmdgpuScalarI64CtpopSgprB64DescriptorRequirements),
            [LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_VGPR_B32] =
                LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                    kAmdgpuScalarI64CtpopVgprB32DescriptorRequirements),
            [LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_VGPR_B64] =
                LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                    kAmdgpuScalarI64CtpopVgprB64DescriptorRequirements),
};

static const loom_amdgpu_scalar_i64_ctpop_kind_t
    kAmdgpuScalarI64CtpopKinds[LOOM_AMDGPU_REG_CLASS_ID_VGPR +
                               1][LOOM_AMDGPU_REG_CLASS_ID_VGPR + 1][2] = {
        [LOOM_AMDGPU_REG_CLASS_ID_SGPR][LOOM_AMDGPU_REG_CLASS_ID_SGPR] =
            {
                LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_SGPR_B32,
                LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_SGPR_B64,
            },
        [LOOM_AMDGPU_REG_CLASS_ID_VGPR][LOOM_AMDGPU_REG_CLASS_ID_SGPR] =
            {
                LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_VGPR_B32,
                LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_VGPR_B64,
            },
        [LOOM_AMDGPU_REG_CLASS_ID_VGPR][LOOM_AMDGPU_REG_CLASS_ID_VGPR] =
            {
                LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_VGPR_B32,
                LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_VGPR_B64,
            },
};

static const loom_amdgpu_descriptor_ref_t
    kAmdgpuIndexCastZeroDescriptorRefs[LOOM_AMDGPU_REG_CLASS_ID_VGPR + 1] = {
        [LOOM_AMDGPU_REG_CLASS_ID_SGPR] = LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        [LOOM_AMDGPU_REG_CLASS_ID_VGPR] = LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
};

#undef LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN

static uint32_t loom_amdgpu_target_index_bitwidth(
    loom_low_lower_context_t* context) {
  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  IREE_ASSERT(bundle != NULL && bundle->snapshot != NULL);
  return bundle != NULL && bundle->snapshot != NULL
             ? bundle->snapshot->index_bitwidth
             : 0;
}

static bool loom_amdgpu_address_cmp_needs_64bit(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_op_t* source_op) {
  const loom_value_id_t lhs = loom_index_cmp_lhs(source_op);
  const loom_value_id_t rhs = loom_index_cmp_rhs(source_op);
  if (module == NULL || lhs >= module->values.count ||
      rhs >= module->values.count) {
    return false;
  }
  const loom_type_t lhs_type = loom_module_value_type(module, lhs);
  const loom_type_t rhs_type = loom_module_value_type(module, rhs);
  return loom_amdgpu_source_address_value_needs_64bit(module, fact_table, lhs,
                                                      lhs_type) ||
         loom_amdgpu_source_address_value_needs_64bit(module, fact_table, rhs,
                                                      rhs_type);
}

static bool loom_amdgpu_scalar_cmpi_has_i64_operands(
    const loom_module_t* module, const loom_op_t* source_op) {
  if (!loom_scalar_cmpi_isa(source_op)) {
    return false;
  }
  const loom_value_id_t lhs = loom_scalar_cmpi_lhs(source_op);
  const loom_value_id_t rhs = loom_scalar_cmpi_rhs(source_op);
  return lhs < module->values.count && rhs < module->values.count &&
         loom_amdgpu_type_is_i64(loom_module_value_type(module, lhs)) &&
         loom_amdgpu_type_is_i64(loom_module_value_type(module, rhs));
}

static bool loom_amdgpu_index_cmp_predicate_to_scalar(
    uint8_t index_predicate,
    loom_scalar_cmpi_predicate_t* out_scalar_predicate) {
  switch ((loom_index_cmp_predicate_t)index_predicate) {
    case LOOM_INDEX_CMP_PREDICATE_EQ:
      *out_scalar_predicate = LOOM_SCALAR_CMPI_PREDICATE_EQ;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_NE:
      *out_scalar_predicate = LOOM_SCALAR_CMPI_PREDICATE_NE;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_SLT:
      *out_scalar_predicate = LOOM_SCALAR_CMPI_PREDICATE_SLT;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_SLE:
      *out_scalar_predicate = LOOM_SCALAR_CMPI_PREDICATE_SLE;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_SGT:
      *out_scalar_predicate = LOOM_SCALAR_CMPI_PREDICATE_SGT;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_SGE:
      *out_scalar_predicate = LOOM_SCALAR_CMPI_PREDICATE_SGE;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_ULT:
      *out_scalar_predicate = LOOM_SCALAR_CMPI_PREDICATE_ULT;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_ULE:
      *out_scalar_predicate = LOOM_SCALAR_CMPI_PREDICATE_ULE;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_UGT:
      *out_scalar_predicate = LOOM_SCALAR_CMPI_PREDICATE_UGT;
      return true;
    case LOOM_INDEX_CMP_PREDICATE_UGE:
      *out_scalar_predicate = LOOM_SCALAR_CMPI_PREDICATE_UGE;
      return true;
    default:
      return false;
  }
}

typedef uint8_t loom_amdgpu_i64_compare_predicate_flags_t;

// I64 compare must include a high-word equality test before combining the
// high-word ordered compare with the low-word unsigned compare.
#define LOOM_AMDGPU_I64_COMPARE_PREDICATE_NEEDS_HIGH_EQUAL ((uint8_t)1u << 0)

typedef struct loom_amdgpu_i64_compare_predicate_descriptor_row_t {
  // High-word compare descriptor ref.
  loom_amdgpu_descriptor_ref_t high_descriptor_ref;
  // Low-word compare descriptor ref.
  loom_amdgpu_descriptor_ref_t low_descriptor_ref;
  // Scalar mask combine descriptor ref.
  loom_amdgpu_descriptor_ref_t combine_descriptor_ref;
  // Extra lowering requirements for this predicate row.
  loom_amdgpu_i64_compare_predicate_flags_t flags;
} loom_amdgpu_i64_compare_predicate_descriptor_row_t;

#define LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(                               \
    high_descriptor_ref_, low_descriptor_ref_, combine_descriptor_ref_,       \
    flags_)                                                                   \
  {                                                                           \
      .high_descriptor_ref =                                                  \
          LOOM_AMDGPU_DESCRIPTOR_REF_##high_descriptor_ref_,                  \
      .low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_##low_descriptor_ref_, \
      .combine_descriptor_ref =                                               \
          LOOM_AMDGPU_DESCRIPTOR_REF_##combine_descriptor_ref_,               \
      .flags = flags_,                                                        \
  }

static const loom_amdgpu_i64_compare_predicate_descriptor_row_t
    kAmdgpuI64ComparePredicateDescriptorRows
        [LOOM_SCALAR_CMPI_PREDICATE_COUNT_] = {
            [LOOM_SCALAR_CMPI_PREDICATE_EQ] =
                LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(
                    V_CMP_EQ_I32, V_CMP_EQ_I32, S_AND_B64, 0),
            [LOOM_SCALAR_CMPI_PREDICATE_NE] =
                LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(
                    V_CMP_NE_I32, V_CMP_NE_I32, S_OR_B64, 0),
            [LOOM_SCALAR_CMPI_PREDICATE_SLT] =
                LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(
                    V_CMP_SLT_I32, V_CMP_ULT_U32, S_OR_B64,
                    LOOM_AMDGPU_I64_COMPARE_PREDICATE_NEEDS_HIGH_EQUAL),
            [LOOM_SCALAR_CMPI_PREDICATE_SLE] =
                LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(
                    V_CMP_SLT_I32, V_CMP_ULE_U32, S_OR_B64,
                    LOOM_AMDGPU_I64_COMPARE_PREDICATE_NEEDS_HIGH_EQUAL),
            [LOOM_SCALAR_CMPI_PREDICATE_SGT] =
                LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(
                    V_CMP_SGT_I32, V_CMP_UGT_U32, S_OR_B64,
                    LOOM_AMDGPU_I64_COMPARE_PREDICATE_NEEDS_HIGH_EQUAL),
            [LOOM_SCALAR_CMPI_PREDICATE_SGE] =
                LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(
                    V_CMP_SGT_I32, V_CMP_UGE_U32, S_OR_B64,
                    LOOM_AMDGPU_I64_COMPARE_PREDICATE_NEEDS_HIGH_EQUAL),
            [LOOM_SCALAR_CMPI_PREDICATE_ULT] =
                LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(
                    V_CMP_ULT_U32, V_CMP_ULT_U32, S_OR_B64,
                    LOOM_AMDGPU_I64_COMPARE_PREDICATE_NEEDS_HIGH_EQUAL),
            [LOOM_SCALAR_CMPI_PREDICATE_ULE] =
                LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(
                    V_CMP_ULT_U32, V_CMP_ULE_U32, S_OR_B64,
                    LOOM_AMDGPU_I64_COMPARE_PREDICATE_NEEDS_HIGH_EQUAL),
            [LOOM_SCALAR_CMPI_PREDICATE_UGT] =
                LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(
                    V_CMP_UGT_U32, V_CMP_UGT_U32, S_OR_B64,
                    LOOM_AMDGPU_I64_COMPARE_PREDICATE_NEEDS_HIGH_EQUAL),
            [LOOM_SCALAR_CMPI_PREDICATE_UGE] =
                LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW(
                    V_CMP_UGT_U32, V_CMP_UGE_U32, S_OR_B64,
                    LOOM_AMDGPU_I64_COMPARE_PREDICATE_NEEDS_HIGH_EQUAL),
};

#undef LOOM_AMDGPU_I64_COMPARE_DESCRIPTOR_ROW

static bool loom_amdgpu_i64_compare_predicate_descriptors(
    loom_scalar_cmpi_predicate_t predicate,
    loom_amdgpu_descriptor_ref_t* out_high_descriptor_ref,
    loom_amdgpu_descriptor_ref_t* out_low_descriptor_ref,
    loom_amdgpu_descriptor_ref_t* out_combine_descriptor_ref,
    bool* out_needs_high_equal) {
  *out_high_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  *out_low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  *out_combine_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  *out_needs_high_equal = false;
  if (predicate >= LOOM_SCALAR_CMPI_PREDICATE_COUNT_) {
    return false;
  }
  const loom_amdgpu_i64_compare_predicate_descriptor_row_t* row =
      &kAmdgpuI64ComparePredicateDescriptorRows[predicate];
  if (row->high_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE ||
      row->low_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE ||
      row->combine_descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    return false;
  }
  *out_high_descriptor_ref = row->high_descriptor_ref;
  *out_low_descriptor_ref = row->low_descriptor_ref;
  *out_combine_descriptor_ref = row->combine_descriptor_ref;
  *out_needs_high_equal = iree_any_bit_set(
      row->flags, LOOM_AMDGPU_I64_COMPARE_PREDICATE_NEEDS_HIGH_EQUAL);
  return true;
}

static bool loom_amdgpu_i64_compare_descriptors_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_i64_compare_plan_t* plan,
    iree_string_view_t* out_constraint_key) {
  if (!loom_amdgpu_descriptor_requirements_present(
          descriptor_set, kAmdgpuVgprMoveDescriptorRequirements,
          IREE_ARRAYSIZE(kAmdgpuVgprMoveDescriptorRequirements),
          out_constraint_key)) {
    return false;
  }
  const loom_amdgpu_descriptor_requirement_t requirements[] = {
      {
          .constraint_key = IREE_SVL("descriptor.high_compare"),
          .descriptor_ref = plan->high_descriptor_ref,
      },
      {
          .constraint_key = IREE_SVL("descriptor.low_compare"),
          .descriptor_ref = plan->low_descriptor_ref,
      },
      {
          .constraint_key = IREE_SVL("descriptor.combine"),
          .descriptor_ref = plan->combine_descriptor_ref,
      },
  };
  if (!loom_amdgpu_descriptor_requirements_present(descriptor_set, requirements,
                                                   IREE_ARRAYSIZE(requirements),
                                                   out_constraint_key)) {
    return false;
  }
  if (plan->needs_high_equal) {
    return loom_amdgpu_descriptor_requirements_present(
        descriptor_set, kAmdgpuI64CompareHighEqualDescriptorRequirements,
        IREE_ARRAYSIZE(kAmdgpuI64CompareHighEqualDescriptorRequirements),
        out_constraint_key);
  }
  return true;
}

iree_status_t loom_amdgpu_low_legality_verify_address_compare(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_address_cmp_needs_64bit(
          module, loom_target_low_legality_fact_table(context), op)) {
    return iree_ok_status();
  }

  bool result_is_native_mask = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_low_legality_value_is_native_i1_mask(
      context, loom_index_cmp_result(op), &result_is_native_mask));
  if (!result_is_native_mask) {
    return iree_ok_status();
  }
  *out_handled = true;

  loom_amdgpu_i64_compare_plan_t plan = {
      .lhs = loom_index_cmp_lhs(op),
      .rhs = loom_index_cmp_rhs(op),
      .result = loom_index_cmp_result(op),
  };
  loom_scalar_cmpi_predicate_t predicate;
  if (!loom_amdgpu_index_cmp_predicate_to_scalar(loom_index_cmp_predicate(op),
                                                 &predicate) ||
      !loom_amdgpu_i64_compare_predicate_descriptors(
          predicate, &plan.high_descriptor_ref, &plan.low_descriptor_ref,
          &plan.combine_descriptor_ref, &plan.needs_high_equal)) {
    return loom_amdgpu_low_legality_reject(context, op, IREE_SV("predicate"));
  }
  iree_string_view_t constraint_key = iree_string_view_empty();
  if (loom_amdgpu_i64_compare_descriptors_supported(
          loom_target_low_legality_descriptor_set(context), &plan,
          &constraint_key)) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_reject(context, op, constraint_key);
}

iree_status_t loom_amdgpu_low_legality_verify_scalar_cmpi_i64(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_scalar_cmpi_has_i64_operands(module, op)) {
    return iree_ok_status();
  }

  bool result_is_native_mask = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_low_legality_value_is_native_i1_mask(
      context, loom_scalar_cmpi_result(op), &result_is_native_mask));
  if (!result_is_native_mask) {
    return iree_ok_status();
  }
  *out_handled = true;

  loom_amdgpu_i64_compare_plan_t plan = {
      .lhs = loom_scalar_cmpi_lhs(op),
      .rhs = loom_scalar_cmpi_rhs(op),
      .result = loom_scalar_cmpi_result(op),
  };
  if (!loom_amdgpu_i64_compare_predicate_descriptors(
          (loom_scalar_cmpi_predicate_t)loom_scalar_cmpi_predicate(op),
          &plan.high_descriptor_ref, &plan.low_descriptor_ref,
          &plan.combine_descriptor_ref, &plan.needs_high_equal)) {
    return loom_amdgpu_low_legality_reject(context, op, IREE_SV("predicate"));
  }
  iree_string_view_t constraint_key = iree_string_view_empty();
  if (loom_amdgpu_i64_compare_descriptors_supported(
          loom_target_low_legality_descriptor_set(context), &plan,
          &constraint_key)) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_reject(context, op, constraint_key);
}

static iree_status_t loom_amdgpu_emit_index_cast_range_diagnostic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t source_type, loom_type_t result_type,
    loom_value_facts_t source_facts, uint32_t index_bitwidth) {
  loom_module_t* module = loom_low_lower_context_module(context);
  static const iree_string_view_t accepted_proof_sources[] = {
      IREE_SVL("scalar.assume on the integer source before index.cast"),
      IREE_SVL("config or kernel boundary facts on the integer source"),
  };
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_lower_context_target_key(context)),
      loom_param_string(loom_low_lower_context_export_name(context)),
      loom_param_string(loom_low_lower_context_config_key(context)),
      loom_param_string(loom_low_lower_context_function_name(context)),
      loom_param_string(loom_op_name(module, source_op)),
      loom_param_type(source_type),
      loom_param_type(result_type),
      loom_param_i64(source_facts.range_lo),
      loom_param_i64(source_facts.range_hi),
      loom_param_u32(index_bitwidth),
      loom_param_i64(INT32_MIN),
      loom_param_i64(INT32_MAX),
      loom_param_string(IREE_SV("index_cast.target_width_range")),
      loom_param_string_list(accepted_proof_sources,
                             IREE_ARRAYSIZE(accepted_proof_sources)),
  };
  return loom_low_lower_emit_error_ref(context, source_op,
                                       LOOM_ERR_AMDGPU_033_REF, params,
                                       IREE_ARRAYSIZE(params));
}

iree_status_t loom_amdgpu_select_index_cast_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_index_cast_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_index_cast_plan_t){0};
  *out_selected = false;
  const loom_value_id_t source = loom_index_cast_input(source_op);
  const loom_value_id_t result = loom_index_cast_result(source_op);

  loom_type_t source_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, source, &source_low_type));
  if (!loom_low_type_is_register(source_low_type)) {
    return iree_ok_status();
  }
  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));
  if (loom_type_equal(source_low_type, result_low_type)) {
    *out_plan = (loom_amdgpu_index_cast_plan_t){
        .kind = LOOM_AMDGPU_INDEX_CAST_KIND_ALIAS,
        .source = source,
        .result = result,
    };
    *out_selected = true;
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);
  const loom_scalar_type_t source_scalar_type =
      loom_type_element_type(source_type);
  const loom_scalar_type_t result_scalar_type =
      loom_type_element_type(result_type);
  const uint32_t index_bitwidth = loom_amdgpu_target_index_bitwidth(context);
  if (index_bitwidth != 32) {
    return iree_ok_status();
  }

  loom_value_facts_t source_facts = loom_value_facts_unknown();
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (fact_table != NULL && source < module->values.count) {
    source_facts = loom_value_fact_table_lookup(fact_table, source);
  }

  switch (source_scalar_type) {
    case LOOM_SCALAR_TYPE_I64:
      if (result_scalar_type != LOOM_SCALAR_TYPE_INDEX) {
        return iree_ok_status();
      }
      if (!loom_value_facts_fit_signed_bit_count(source_facts,
                                                 (uint8_t)index_bitwidth)) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_index_cast_range_diagnostic(
            context, source_op, source_type, result_type, source_facts,
            index_bitwidth));
        *out_plan = (loom_amdgpu_index_cast_plan_t){
            .kind = LOOM_AMDGPU_INDEX_CAST_KIND_DIAGNOSTIC_REJECTED,
            .source = source,
            .result = result,
            .index_bitwidth = index_bitwidth,
        };
        *out_selected = true;
        return iree_ok_status();
      }
      *out_plan = (loom_amdgpu_index_cast_plan_t){
          .kind = LOOM_AMDGPU_INDEX_CAST_KIND_PRESERVING_LOW_32,
          .source = source,
          .result = result,
          .index_bitwidth = index_bitwidth,
      };
      *out_selected = true;
      return iree_ok_status();
    case LOOM_SCALAR_TYPE_INDEX: {
      if (result_scalar_type != LOOM_SCALAR_TYPE_I64 ||
          source_facts.range_lo < 0) {
        return iree_ok_status();
      }
      if (!loom_low_type_is_register(result_low_type)) {
        return iree_ok_status();
      }
      const uint32_t source_unit_count =
          loom_low_register_type_unit_count(source_low_type);
      const uint32_t result_unit_count =
          loom_low_register_type_unit_count(result_low_type);
      if (source_unit_count != 1 || result_unit_count != 2) {
        return iree_ok_status();
      }
      break;
    }
    default:
      return iree_ok_status();
  }

  const uint16_t source_register_class =
      loom_low_register_type_class_id(source_low_type);
  if (source_register_class > LOOM_AMDGPU_REG_CLASS_ID_VGPR ||
      source_register_class !=
          loom_low_register_type_class_id(result_low_type)) {
    return iree_ok_status();
  }
  const loom_amdgpu_descriptor_ref_t zero_descriptor_ref =
      kAmdgpuIndexCastZeroDescriptorRefs[source_register_class];
  if (!loom_amdgpu_descriptor_set_has_ref(
          loom_low_lower_context_descriptor_set(context),
          zero_descriptor_ref)) {
    return iree_ok_status();
  }
  *out_plan = (loom_amdgpu_index_cast_plan_t){
      .kind = LOOM_AMDGPU_INDEX_CAST_KIND_ZERO_EXTENDING_LOW_32,
      .source = source,
      .result = result,
      .zero_descriptor_ref = zero_descriptor_ref,
      .index_bitwidth = index_bitwidth,
  };
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_index_cmp_i64_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_i64_compare_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_i64_compare_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_amdgpu_address_cmp_needs_64bit(
          module, loom_low_lower_context_fact_table(context), source_op)) {
    return iree_ok_status();
  }

  const loom_value_id_t result = loom_index_cmp_result(source_op);
  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));
  const bool result_is_native_mask =
      loom_amdgpu_low_type_is_register_class_count(
          context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  if (!result_is_native_mask) {
    return iree_ok_status();
  }

  loom_amdgpu_i64_compare_plan_t plan = {
      .lhs = loom_index_cmp_lhs(source_op),
      .rhs = loom_index_cmp_rhs(source_op),
      .result = result,
  };
  loom_scalar_cmpi_predicate_t predicate;
  if (!loom_amdgpu_index_cmp_predicate_to_scalar(
          loom_index_cmp_predicate(source_op), &predicate) ||
      !loom_amdgpu_i64_compare_predicate_descriptors(
          predicate, &plan.high_descriptor_ref, &plan.low_descriptor_ref,
          &plan.combine_descriptor_ref, &plan.needs_high_equal)) {
    return iree_ok_status();
  }

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (!loom_amdgpu_i64_compare_descriptors_supported(
          loom_low_lower_context_descriptor_set(context), &plan,
          &constraint_key)) {
    return iree_ok_status();
  }

  *out_plan = plan;
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_scalar_i64_compare_operand_can_lower(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, bool* out_can_lower) {
  *out_can_lower = false;
  loom_type_t source_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, source, &source_low_type));
  if (!loom_low_type_is_register(source_low_type) ||
      loom_low_register_type_unit_count(source_low_type) != 2) {
    return iree_ok_status();
  }
  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, source_low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (is_vgpr) {
    *out_can_lower = true;
    return iree_ok_status();
  }
  *out_can_lower = loom_amdgpu_low_type_is_register_class(
      context, source_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_scalar_cmpi_i64_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_i64_compare_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_i64_compare_plan_t){0};
  *out_selected = false;
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_amdgpu_scalar_cmpi_has_i64_operands(module, source_op)) {
    return iree_ok_status();
  }

  const loom_value_id_t result = loom_scalar_cmpi_result(source_op);
  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));
  const bool result_is_native_mask =
      loom_amdgpu_low_type_is_register_class_count(
          context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  if (!result_is_native_mask) {
    return iree_ok_status();
  }

  bool lhs_can_lower = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_scalar_i64_compare_operand_can_lower(
      context, source_op, loom_scalar_cmpi_lhs(source_op), &lhs_can_lower));
  bool rhs_can_lower = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_scalar_i64_compare_operand_can_lower(
      context, source_op, loom_scalar_cmpi_rhs(source_op), &rhs_can_lower));
  if (!lhs_can_lower || !rhs_can_lower) {
    return iree_ok_status();
  }

  loom_amdgpu_i64_compare_plan_t plan = {
      .lhs = loom_scalar_cmpi_lhs(source_op),
      .rhs = loom_scalar_cmpi_rhs(source_op),
      .result = result,
  };
  if (!loom_amdgpu_i64_compare_predicate_descriptors(
          (loom_scalar_cmpi_predicate_t)loom_scalar_cmpi_predicate(source_op),
          &plan.high_descriptor_ref, &plan.low_descriptor_ref,
          &plan.combine_descriptor_ref, &plan.needs_high_equal)) {
    return iree_ok_status();
  }

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (!loom_amdgpu_i64_compare_descriptors_supported(
          loom_low_lower_context_descriptor_set(context), &plan,
          &constraint_key)) {
    return iree_ok_status();
  }

  *out_plan = plan;
  *out_selected = true;
  return iree_ok_status();
}

static bool loom_amdgpu_scalar_i64_alu_descriptors_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_scalar_i64_alu_kind_t kind, uint8_t shift_amount,
    iree_string_view_t* out_constraint_key) {
  if (kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_LSHR_LITERAL) {
    if (shift_amount == 0) return true;
    if (shift_amount < 32 &&
        !loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
            shift_amount)) {
      *out_constraint_key = IREE_SV("descriptor.v_lshrrev_b32_lit");
      return false;
    }
    if (shift_amount < 32 &&
        !loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
            32u - shift_amount)) {
      *out_constraint_key = IREE_SV("descriptor.v_lshlrev_b32_lit");
      return false;
    }
    if (shift_amount > 32 &&
        !loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
            shift_amount - 32u)) {
      *out_constraint_key = IREE_SV("descriptor.v_lshrrev_b32_lit");
      return false;
    }
    const loom_amdgpu_descriptor_ref_t combine_descriptor_ref =
        shift_amount < 32 ? LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32
                          : LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32;
    if (!loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                            combine_descriptor_ref)) {
      *out_constraint_key = shift_amount < 32 ? IREE_SV("descriptor.v_or_b32")
                                              : IREE_SV("descriptor.v_mov_b32");
      return false;
    }
    return true;
  }
  return loom_amdgpu_i64_alu_descriptor_requirements_present(
      descriptor_set, kAmdgpuScalarI64AluDescriptorRequirementRows,
      IREE_ARRAYSIZE(kAmdgpuScalarI64AluDescriptorRequirementRows),
      (iree_host_size_t)kind, IREE_SV("operation.scalar_i64_alu"),
      out_constraint_key);
}

static bool loom_amdgpu_scalar_i64_ctpop_descriptors_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_scalar_i64_ctpop_kind_t kind,
    iree_string_view_t* out_constraint_key) {
  return loom_amdgpu_descriptor_requirement_span_present(
      descriptor_set, kAmdgpuScalarI64CtpopDescriptorRequirementRows[kind],
      out_constraint_key);
}

static bool loom_amdgpu_address_i64_alu_kind_uses_vgpr(
    loom_amdgpu_address_i64_alu_kind_t kind) {
  if ((iree_host_size_t)kind >=
      IREE_ARRAYSIZE(kAmdgpuAddressI64AluDescriptorRequirementRows)) {
    return false;
  }
  return iree_any_bit_set(
      kAmdgpuAddressI64AluDescriptorRequirementRows[kind].flags,
      LOOM_AMDGPU_ADDRESS_I64_ALU_ROW_FLAG_USES_VGPR);
}

static bool loom_amdgpu_address_i64_alu_descriptors_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_address_i64_alu_kind_t kind,
    iree_string_view_t* out_constraint_key) {
  return loom_amdgpu_i64_alu_descriptor_requirements_present(
      descriptor_set, kAmdgpuAddressI64AluDescriptorRequirementRows,
      IREE_ARRAYSIZE(kAmdgpuAddressI64AluDescriptorRequirementRows),
      (iree_host_size_t)kind, IREE_SV("operation.address_i64_alu"),
      out_constraint_key);
}

#define LOOM_AMDGPU_OP_INDEX(op_kind) ((uint8_t)((op_kind) & 0xFFu))

typedef struct loom_amdgpu_address_i64_alu_source_layout_t {
  // Operation kind selected by this source op.
  loom_amdgpu_address_i64_alu_kind_t kind;
  // Number of fixed operands consumed by the source op.
  uint8_t operand_count;
} loom_amdgpu_address_i64_alu_source_layout_t;

static const loom_amdgpu_address_i64_alu_source_layout_t
    kAmdgpuAddressI64AluSourceLayoutByIndexOp[LOOM_OP_INDEX_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_ADD)] =
            {
                .kind = LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_ADD,
                .operand_count = 2,
            },
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_SUB)] =
            {
                .kind = LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_SUB,
                .operand_count = 2,
            },
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_MUL)] =
            {
                .kind = LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MUL_LO,
                .operand_count = 2,
            },
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_MADD)] =
            {
                .kind = LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MADD_LO,
                .operand_count = 3,
            },
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_SHLI)] =
            {
                .kind = LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_SHL,
                .operand_count = 2,
            },
};

static const loom_amdgpu_scalar_i64_alu_kind_t
    kAmdgpuScalarI64AluKindByScalarOp[LOOM_OP_SCALAR_COUNT_] = {
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_ADDI)] =
            LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_ADD,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_SUBI)] =
            LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SUB,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_MULI)] =
            LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_MUL_LO,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_SHLI)] =
            LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SHL,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_SHRUI)] =
            LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_LSHR_LITERAL,
};

#undef LOOM_AMDGPU_OP_INDEX

static bool loom_amdgpu_address_i64_alu_op(
    const loom_op_t* source_op, loom_amdgpu_address_i64_alu_kind_t* out_kind,
    loom_value_id_t* out_lhs, loom_value_id_t* out_rhs,
    loom_value_id_t* out_addend, loom_value_id_t* out_result) {
  *out_kind = LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE;
  *out_lhs = LOOM_VALUE_ID_INVALID;
  *out_rhs = LOOM_VALUE_ID_INVALID;
  *out_addend = LOOM_VALUE_ID_INVALID;
  *out_result = LOOM_VALUE_ID_INVALID;
  if (loom_op_dialect_id(source_op->kind) != LOOM_DIALECT_INDEX) return false;
  const uint8_t op_index = loom_op_dialect_index(source_op->kind);
  if (op_index >= IREE_ARRAYSIZE(kAmdgpuAddressI64AluSourceLayoutByIndexOp)) {
    return false;
  }
  const loom_amdgpu_address_i64_alu_source_layout_t* layout =
      &kAmdgpuAddressI64AluSourceLayoutByIndexOp[op_index];
  if (layout->kind == LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE) return false;
  const loom_value_id_t* operands = loom_op_const_operands(source_op);
  *out_kind = layout->kind;
  *out_lhs = operands[0];
  *out_rhs = operands[1];
  *out_addend = layout->operand_count > 2 ? operands[2] : LOOM_VALUE_ID_INVALID;
  *out_result = loom_op_const_results(source_op)[0];
  return true;
}

static bool loom_amdgpu_address_i64_alu_result_needs_wide(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t result) {
  if (result >= module->values.count) {
    return false;
  }
  const loom_type_t result_type = loom_module_value_type(module, result);
  return loom_amdgpu_source_address_value_needs_64bit(module, fact_table,
                                                      result, result_type);
}

static iree_status_t loom_amdgpu_select_address_i64_alu_kind(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_address_i64_alu_kind_t operation_kind, loom_value_id_t result,
    loom_amdgpu_address_i64_alu_kind_t* out_kind) {
  *out_kind = LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE;
  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));

  const bool result_is_vgpr64 = loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
  if (result_is_vgpr64) {
    *out_kind = operation_kind;
    return iree_ok_status();
  }

  if (operation_kind != LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_ADD) {
    return iree_ok_status();
  }
  const bool result_is_sgpr64 = loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  if (result_is_sgpr64) {
    *out_kind = LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_SGPR_ADD;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_address_i64_operand_can_materialize(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, loom_amdgpu_address_i64_alu_kind_t result_kind,
    bool* out_can_lower) {
  *out_can_lower = false;
  loom_type_t source_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, source, &source_low_type));
  if (!loom_low_type_is_register(source_low_type)) {
    return iree_ok_status();
  }
  const uint32_t unit_count =
      loom_low_register_type_unit_count(source_low_type);
  if (unit_count != 1 && unit_count != 2) {
    return iree_ok_status();
  }
  if (loom_amdgpu_address_i64_alu_kind_uses_vgpr(result_kind)) {
    const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
        context, source_low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
    if (is_vgpr) {
      *out_can_lower = true;
      return iree_ok_status();
    }
  }
  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, source_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  *out_can_lower = is_sgpr;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_address_i64_alu_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_address_i64_alu_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_address_i64_alu_plan_t){0};
  *out_selected = false;
  loom_amdgpu_address_i64_alu_kind_t operation_kind =
      LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE;
  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t addend = LOOM_VALUE_ID_INVALID;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_address_i64_alu_op(source_op, &operation_kind, &lhs, &rhs,
                                      &addend, &result)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_amdgpu_address_i64_alu_result_needs_wide(
          module, loom_low_lower_context_fact_table(context), result)) {
    return iree_ok_status();
  }

  loom_amdgpu_address_i64_alu_kind_t kind =
      LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE;
  IREE_RETURN_IF_ERROR(loom_amdgpu_select_address_i64_alu_kind(
      context, source_op, operation_kind, result, &kind));
  if (kind == LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE) {
    return iree_ok_status();
  }

  bool lhs_can_lower = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_address_i64_operand_can_materialize(
      context, source_op, lhs, kind, &lhs_can_lower));
  bool rhs_can_lower = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_address_i64_operand_can_materialize(
      context, source_op, rhs, kind, &rhs_can_lower));
  bool addend_can_lower = true;
  if (addend != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_address_i64_operand_can_materialize(
        context, source_op, addend, kind, &addend_can_lower));
  }
  if (!lhs_can_lower || !rhs_can_lower || !addend_can_lower) {
    return iree_ok_status();
  }

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (!loom_amdgpu_address_i64_alu_descriptors_supported(
          loom_low_lower_context_descriptor_set(context), kind,
          &constraint_key)) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_address_i64_alu_plan_t){
      .lhs = lhs,
      .rhs = rhs,
      .addend = addend,
      .result = result,
      .kind = kind,
  };
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_address_i64_alu(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  if (!loom_amdgpu_low_legality_context_is_amdgpu(context)) {
    return iree_ok_status();
  }
  loom_amdgpu_address_i64_alu_kind_t operation_kind =
      LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE;
  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t addend = LOOM_VALUE_ID_INVALID;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_address_i64_alu_op(op, &operation_kind, &lhs, &rhs, &addend,
                                      &result)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_address_i64_alu_result_needs_wide(
          module, loom_target_low_legality_fact_table(context), result)) {
    return iree_ok_status();
  }
  *out_handled = true;

  bool result_prefers_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_low_legality_value_prefers_vgpr(
      context, result, &result_prefers_vgpr));
  loom_amdgpu_address_i64_alu_kind_t kind = operation_kind;
  if (operation_kind == LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_ADD &&
      !result_prefers_vgpr) {
    kind = LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_SGPR_ADD;
  } else if (!result_prefers_vgpr) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("result.vgpr64"));
  }

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (loom_amdgpu_address_i64_alu_descriptors_supported(
          loom_target_low_legality_descriptor_set(context), kind,
          &constraint_key)) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_reject(context, op, constraint_key);
}

static bool loom_amdgpu_scalar_i64_alu_op(
    const loom_op_t* source_op, loom_amdgpu_scalar_i64_alu_kind_t* out_kind,
    loom_value_id_t* out_lhs, loom_value_id_t* out_rhs,
    loom_value_id_t* out_result) {
  *out_kind = LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE;
  *out_lhs = LOOM_VALUE_ID_INVALID;
  *out_rhs = LOOM_VALUE_ID_INVALID;
  *out_result = LOOM_VALUE_ID_INVALID;
  if (loom_op_dialect_id(source_op->kind) != LOOM_DIALECT_SCALAR) return false;
  const uint8_t op_index = loom_op_dialect_index(source_op->kind);
  if (op_index >= IREE_ARRAYSIZE(kAmdgpuScalarI64AluKindByScalarOp)) {
    return false;
  }
  const loom_amdgpu_scalar_i64_alu_kind_t kind =
      kAmdgpuScalarI64AluKindByScalarOp[op_index];
  if (kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE) return false;
  const loom_value_id_t* operands = loom_op_const_operands(source_op);
  *out_kind = kind;
  *out_lhs = operands[0];
  *out_rhs = operands[1];
  *out_result = loom_op_const_results(source_op)[0];
  return true;
}

static iree_status_t loom_amdgpu_scalar_i64_operand_can_materialize_as_vgpr64(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, bool* out_can_lower) {
  *out_can_lower = false;
  loom_type_t source_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, source, &source_low_type));
  if (!loom_low_type_is_register(source_low_type) ||
      loom_low_register_type_unit_count(source_low_type) != 2) {
    return iree_ok_status();
  }
  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, source_low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (is_vgpr) {
    *out_can_lower = true;
    return iree_ok_status();
  }
  *out_can_lower = loom_amdgpu_low_type_is_register_class(
      context, source_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  return iree_ok_status();
}

static bool loom_amdgpu_scalar_i64_exact_shift_amount(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value,
    uint8_t* out_shift_amount) {
  *out_shift_amount = 0;
  if (fact_table == NULL) return false;
  int64_t shift_amount = 0;
  if (!loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(fact_table, value), &shift_amount) ||
      shift_amount < 0 || shift_amount >= 64) {
    return false;
  }
  *out_shift_amount = (uint8_t)shift_amount;
  return true;
}

iree_status_t loom_amdgpu_select_scalar_i64_alu_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_scalar_i64_alu_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_scalar_i64_alu_plan_t){0};
  *out_selected = false;
  loom_amdgpu_scalar_i64_alu_kind_t kind = LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE;
  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_scalar_i64_alu_op(source_op, &kind, &lhs, &rhs, &result)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_amdgpu_type_is_i64(loom_module_value_type(module, result))) {
    return iree_ok_status();
  }

  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));
  const bool result_is_vgpr64 = loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2);
  if (!result_is_vgpr64) return iree_ok_status();

  bool lhs_can_lower = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_scalar_i64_operand_can_materialize_as_vgpr64(
      context, source_op, lhs, &lhs_can_lower));
  if (!lhs_can_lower) return iree_ok_status();

  uint8_t shift_amount = 0;
  if (kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_LSHR_LITERAL) {
    if (!loom_amdgpu_scalar_i64_exact_shift_amount(
            loom_low_lower_context_fact_table(context), rhs, &shift_amount)) {
      return iree_ok_status();
    }
  } else {
    bool rhs_can_lower = false;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_scalar_i64_operand_can_materialize_as_vgpr64(
            context, source_op, rhs, &rhs_can_lower));
    if (!rhs_can_lower) return iree_ok_status();
  }

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (!loom_amdgpu_scalar_i64_alu_descriptors_supported(
          loom_low_lower_context_descriptor_set(context), kind, shift_amount,
          &constraint_key)) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_scalar_i64_alu_plan_t){
      .kind = kind,
      .lhs = lhs,
      .rhs = rhs,
      .result = result,
      .shift_amount = shift_amount,
  };
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_scalar_i64_ctpop_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_scalar_i64_ctpop_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_scalar_i64_ctpop_plan_t){0};
  *out_selected = false;

  const loom_value_id_t source = loom_scalar_ctpopi_input(source_op);
  const loom_value_id_t result = loom_scalar_ctpopi_result(source_op);
  const loom_module_t* module = loom_low_lower_context_module(context);
  if (!loom_amdgpu_type_is_i64(loom_module_value_type(module, result))) {
    return iree_ok_status();
  }

  loom_type_t result_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op, result,
                                                   &result_low_type));
  loom_type_t source_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, source, &source_low_type));
  if (!loom_low_type_is_register(source_low_type) ||
      !loom_low_type_is_register(result_low_type)) {
    return iree_ok_status();
  }
  const uint32_t source_unit_count =
      loom_low_register_type_unit_count(source_low_type);
  if (source_unit_count != 1 && source_unit_count != 2) {
    return iree_ok_status();
  }

  const uint16_t source_register_class =
      loom_low_register_type_class_id(source_low_type);
  const uint16_t result_register_class =
      loom_low_register_type_class_id(result_low_type);
  if (source_register_class > LOOM_AMDGPU_REG_CLASS_ID_VGPR ||
      result_register_class > LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    return iree_ok_status();
  }
  const loom_amdgpu_scalar_i64_ctpop_kind_t kind =
      kAmdgpuScalarI64CtpopKinds[result_register_class][source_register_class]
                                [source_unit_count - 1];
  if (kind == LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_NONE) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_scalar_i64_ctpop_plan_t){
      .source = source,
      .result = result,
      .kind = kind,
  };
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_scalar_i64_alu(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  if (!loom_amdgpu_low_legality_context_is_amdgpu(context)) {
    return iree_ok_status();
  }
  loom_amdgpu_scalar_i64_alu_kind_t kind = LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE;
  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_scalar_i64_alu_op(op, &kind, &lhs, &rhs, &result)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_type_is_i64(loom_module_value_type(module, result))) {
    return iree_ok_status();
  }
  *out_handled = true;

  if (!loom_amdgpu_type_is_i64(loom_module_value_type(module, lhs)) ||
      !loom_amdgpu_type_is_i64(loom_module_value_type(module, rhs))) {
    return loom_amdgpu_low_legality_reject(context, op, IREE_SV("operand.i64"));
  }

  bool result_prefers_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_low_legality_value_prefers_vgpr(
      context, result, &result_prefers_vgpr));
  if (!result_prefers_vgpr) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("result.vgpr64"));
  }

  uint8_t shift_amount = 0;
  if (kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_LSHR_LITERAL &&
      !loom_amdgpu_scalar_i64_exact_shift_amount(
          loom_target_low_legality_fact_table(context), rhs, &shift_amount)) {
    return loom_amdgpu_low_legality_reject(context, op,
                                           IREE_SV("rhs.literal_shift_0_63"));
  }

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (loom_amdgpu_scalar_i64_alu_descriptors_supported(
          loom_target_low_legality_descriptor_set(context), kind, shift_amount,
          &constraint_key)) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_reject(context, op, constraint_key);
}

iree_status_t loom_amdgpu_low_legality_verify_scalar_i64_ctpop(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  if (!loom_amdgpu_low_legality_context_is_amdgpu(context)) {
    return iree_ok_status();
  }

  const loom_value_id_t result = loom_scalar_ctpopi_result(op);
  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_type_is_i64(loom_module_value_type(module, result))) {
    return iree_ok_status();
  }
  *out_handled = true;

  bool result_prefers_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_low_legality_value_prefers_vgpr(
      context, result, &result_prefers_vgpr));
  const loom_amdgpu_scalar_i64_ctpop_kind_t kinds[] = {
      result_prefers_vgpr ? LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_VGPR_B32
                          : LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_SGPR_B32,
      result_prefers_vgpr ? LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_VGPR_B64
                          : LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_SGPR_B64,
  };
  iree_string_view_t constraint_key = iree_string_view_empty();
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kinds); ++i) {
    if (!loom_amdgpu_scalar_i64_ctpop_descriptors_supported(
            loom_target_low_legality_descriptor_set(context), kinds[i],
            &constraint_key)) {
      return loom_amdgpu_low_legality_reject(context, op, constraint_key);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_low_legality_verify_scalar_remsi_i64(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  if (!loom_amdgpu_low_legality_context_is_amdgpu(context)) {
    return iree_ok_status();
  }
  if (!loom_scalar_remsi_isa(op)) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_type_is_i64(
          loom_module_value_type(module, loom_scalar_remsi_lhs(op))) ||
      !loom_amdgpu_type_is_i64(
          loom_module_value_type(module, loom_scalar_remsi_rhs(op))) ||
      !loom_amdgpu_type_is_i64(
          loom_module_value_type(module, loom_scalar_remsi_result(op)))) {
    return iree_ok_status();
  }

  *out_handled = true;
  return loom_amdgpu_low_legality_reject(
      context, op, IREE_SV("scalar_remsi.signed_i64_dynamic"));
}

iree_status_t loom_amdgpu_lower_index_cast(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_index_cast_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_INDEX_CAST_KIND_ALIAS:
      return loom_low_lower_bind_value_alias(context, plan->source,
                                             plan->result);
    case LOOM_AMDGPU_INDEX_CAST_KIND_PRESERVING_LOW_32: {
      IREE_ASSERT_EQ(plan->index_bitwidth, 32u);
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_low_lower_lookup_value(context, plan->source, &low_source));
      loom_type_t result_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
          context, source_op, plan->result, &result_type));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_source, /*lane_offset=*/0, result_type,
          &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_INDEX_CAST_KIND_ZERO_EXTENDING_LOW_32: {
      IREE_ASSERT_EQ(plan->index_bitwidth, 32u);
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_low_lower_lookup_value(context, plan->source, &low_source));
      const loom_module_t* module = loom_low_lower_context_module(context);
      const loom_type_t source_type =
          loom_module_value_type(module, low_source);
      const loom_type_t lane_type =
          loom_low_register_carrier_type_with_unit_count(source_type, 1);
      loom_value_id_t low_zero = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(context, source_op,
                                                      plan->zero_descriptor_ref,
                                                      0, lane_type, &low_zero));
      const loom_value_id_t lanes[] = {
          low_source,
          low_zero,
      };
      loom_type_t result_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
          context, source_op, plan->result, &result_type));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_build_low_register_range(
          context, source_op, lanes, IREE_ARRAYSIZE(lanes), result_type,
          &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_INDEX_CAST_KIND_DIAGNOSTIC_REJECTED:
      return iree_ok_status();
    case LOOM_AMDGPU_INDEX_CAST_KIND_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU index cast plan kind");
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_lookup_or_materialize_address_i64_operand(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, loom_amdgpu_address_i64_alu_kind_t kind,
    loom_value_id_t* out_low_value) {
  *out_low_value = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_value));

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  if (!loom_low_type_is_register(low_type)) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU address i64 ALU plan selected non-register operand");
    IREE_BUILTIN_UNREACHABLE();
  }
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);

  if (loom_amdgpu_address_i64_alu_kind_uses_vgpr(kind)) {
    const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
        context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
    if (is_vgpr && unit_count == 2) {
      *out_low_value = low_value;
      return iree_ok_status();
    }
    if (unit_count == 2) {
      return loom_amdgpu_materialize_low_vgpr_b32_registers(
          context, source_op, low_value, out_low_value);
    }
    if (unit_count == 1) {
      return loom_amdgpu_emit_vgpr64_from_u32(context, source_op, low_value,
                                              out_low_value);
    }
  } else {
    const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
        context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
    if (is_sgpr && unit_count == 2) {
      *out_low_value = low_value;
      return iree_ok_status();
    }
    if (is_sgpr && unit_count == 1) {
      return loom_amdgpu_emit_sgpr64_from_u32(context, source_op, low_value,
                                              out_low_value);
    }
  }

  IREE_ASSERT_UNREACHABLE(
      "AMDGPU address i64 ALU plan selected incompatible register type");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_amdgpu_i64_compare_operand_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value, uint32_t lane_index, loom_type_t lane_type,
    loom_value_id_t* out_low_lane) {
  *out_low_lane = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source_value, &low_source));

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_source);
  if (!loom_low_type_is_register(low_type)) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU i64 compare plan selected non-register operand");
    IREE_BUILTIN_UNREACHABLE();
  }
  const uint32_t unit_count = loom_low_register_type_unit_count(low_type);
  if (unit_count == 1 && lane_index == 1) {
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                      lane_type, out_low_lane);
  }
  if (unit_count != 1 && unit_count != 2) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU i64 compare plan selected wrong operand register count");
    IREE_BUILTIN_UNREACHABLE();
  }

  const loom_type_t source_lane_type =
      loom_low_register_carrier_type_with_unit_count(low_type, 1);
  IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_register_unit(
      context, source_op, low_source, unit_count, lane_index, source_lane_type,
      out_low_lane));
  return loom_amdgpu_materialize_low_vgpr_b32(context, source_op, *out_low_lane,
                                              out_low_lane);
}

static iree_status_t loom_amdgpu_emit_i64_compare_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_type, loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &mask_type, 1, &compare_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_i64_compare_mask_combine(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t mask_type, loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* combine_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &mask_type, 1, &combine_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(combine_op), 0);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_i64_compare(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_i64_compare_plan_t* plan) {
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));

  loom_value_id_t lhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_i64_compare_operand_lane(
      context, source_op, plan->lhs, 0, vgpr_type, &lhs_lo));
  loom_value_id_t lhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_i64_compare_operand_lane(
      context, source_op, plan->lhs, 1, vgpr_type, &lhs_hi));
  loom_value_id_t rhs_lo = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_i64_compare_operand_lane(
      context, source_op, plan->rhs, 0, vgpr_type, &rhs_lo));
  loom_value_id_t rhs_hi = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_i64_compare_operand_lane(
      context, source_op, plan->rhs, 1, vgpr_type, &rhs_hi));

  loom_value_id_t high_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i64_compare_mask(
      context, source_op, plan->high_descriptor_ref, lhs_hi, rhs_hi, mask_type,
      &high_mask));
  loom_value_id_t low_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i64_compare_mask(
      context, source_op, plan->low_descriptor_ref, lhs_lo, rhs_lo, mask_type,
      &low_mask));

  loom_value_id_t combined_low_mask = low_mask;
  if (plan->needs_high_equal) {
    loom_value_id_t high_equal_mask = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i64_compare_mask(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32, lhs_hi,
        rhs_hi, mask_type, &high_equal_mask));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i64_compare_mask_combine(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64,
        high_equal_mask, low_mask, mask_type, &combined_low_mask));
  }

  loom_value_id_t result_mask = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_i64_compare_mask_combine(
      context, source_op, plan->combine_descriptor_ref, high_mask,
      combined_low_mask, mask_type, &result_mask));
  return loom_low_lower_bind_value(context, plan->result, result_mask);
}

iree_status_t loom_amdgpu_extract_low_32_bits_as_vgpr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, loom_value_id_t* out_low_source) {
  *out_low_source = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_source_pair = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, source, &low_source_pair));
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_lane_type =
      loom_amdgpu_low_register_lane_type(module, low_source_pair);
  if (loom_type_kind(source_lane_type) == LOOM_TYPE_NONE) {
    IREE_ASSERT_UNREACHABLE(
        "AMDGPU scalar source lowered to a non-register type");
    IREE_BUILTIN_UNREACHABLE();
  }
  loom_value_id_t low_source_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_source_pair, /*lane_offset=*/0, source_lane_type,
      &low_source_lane));
  return loom_amdgpu_materialize_low_vgpr_b32_registers(
      context, source_op, low_source_lane, out_low_source);
}

iree_status_t loom_amdgpu_lower_address_i64_alu(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_address_i64_alu_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_SGPR_ADD: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->lhs, plan->kind, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->rhs, plan->kind, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr64_add(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_ADD: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->lhs, plan->kind, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->rhs, plan->kind, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_add(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_SUB: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->lhs, plan->kind, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->rhs, plan->kind, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_sub(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MUL_LO: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->lhs, plan->kind, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->rhs, plan->kind, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_mul_lo(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MADD_LO: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->lhs, plan->kind, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->rhs, plan->kind, &low_rhs));
      loom_value_id_t product = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_mul_lo(
          context, source_op, low_lhs, low_rhs, &product));
      loom_value_id_t low_addend = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->addend, plan->kind, &low_addend));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_add(
          context, source_op, product, low_addend, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_SHL: {
      loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_lookup_or_materialize_address_i64_operand(
              context, source_op, plan->lhs, plan->kind, &low_value));
      loom_value_id_t low_shift = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_32_bits_as_vgpr(
          context, source_op, plan->rhs, &low_shift));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_shl(
          context, source_op, low_value, low_shift, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU address i64 ALU plan kind");
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_vgpr64_lshr_literal(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_value, uint8_t shift_amount,
    loom_value_id_t* out_low_result) {
  *out_low_result = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_LT(shift_amount, 64u);
  if (shift_amount == 0) {
    *out_low_result = low_value;
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t result_type = loom_module_value_type(module, low_value);
  IREE_ASSERT(loom_amdgpu_low_type_is_register_class_count(
      context, result_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, 2));
  const loom_type_t lane_type =
      loom_low_register_carrier_type_with_unit_count(result_type, 1);

  loom_value_id_t low_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_value, /*offset=*/0, lane_type, &low_lane));
  loom_value_id_t high_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_value, /*offset=*/1, lane_type, &high_lane));

  loom_value_id_t result_low_lane = LOOM_VALUE_ID_INVALID;
  loom_value_id_t result_high_lane = LOOM_VALUE_ID_INVALID;
  if (shift_amount < 32) {
    loom_value_id_t shifted_low_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        shift_amount, low_lane, lane_type, &shifted_low_lane));
    loom_value_id_t carried_high_lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
        32u - shift_amount, high_lane, lane_type, &carried_high_lane));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32,
        shifted_low_lane, carried_high_lane, lane_type, &result_low_lane));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        shift_amount, high_lane, lane_type, &result_high_lane));
  } else {
    if (shift_amount == 32) {
      result_low_lane = high_lane;
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
          shift_amount - 32u, high_lane, lane_type, &result_low_lane));
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, lane_type,
        &result_high_lane));
  }

  const loom_value_id_t result_lanes[] = {
      result_low_lane,
      result_high_lane,
  };
  return loom_amdgpu_build_low_register_range(context, source_op, result_lanes,
                                              IREE_ARRAYSIZE(result_lanes),
                                              result_type, out_low_result);
}

iree_status_t loom_amdgpu_lower_scalar_i64_alu(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_i64_alu_plan_t* plan) {
  switch (plan->kind) {
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_ADD: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->lhs, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->rhs, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_add(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SUB: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->lhs, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->rhs, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_sub(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_MUL_LO: {
      loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->lhs, &low_lhs));
      loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->rhs, &low_rhs));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_mul_lo(
          context, source_op, low_lhs, low_rhs, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SHL: {
      loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->lhs, &low_value));
      loom_value_id_t low_shift = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_extract_low_32_bits_as_vgpr(
          context, source_op, plan->rhs, &low_shift));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_shl(
          context, source_op, low_value, low_shift, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_LSHR_LITERAL: {
      loom_value_id_t low_value = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_lookup_or_materialize_vgpr_i64(
          context, source_op, plan->lhs, &low_value));
      loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr64_lshr_literal(
          context, source_op, low_value, plan->shift_amount, &low_result));
      return loom_low_lower_bind_value(context, plan->result, low_result);
    }
    case LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE:
      break;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU scalar i64 ALU plan kind");
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_scalar_i64_ctpop(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_scalar_i64_ctpop_plan_t* plan) {
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(context, source_op,
                                                   plan->result, &result_type));
  const uint32_t result_unit_count =
      loom_low_register_type_unit_count(result_type);
  loom_value_id_t low_count = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_zero = LOOM_VALUE_ID_INVALID;
  switch (plan->kind) {
    case LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_SGPR_B32:
    case LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_SGPR_B64: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_low_lower_lookup_value(context, plan->source, &low_source));
      loom_type_t sgpr_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
      loom_op_t* count_op = NULL;
      const loom_value_id_t operands[] = {low_source};
      const loom_amdgpu_descriptor_ref_t descriptor_ref =
          plan->kind == LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_SGPR_B32
              ? LOOM_AMDGPU_DESCRIPTOR_REF_S_BCNT1_I32_B32
              : LOOM_AMDGPU_DESCRIPTOR_REF_S_BCNT1_I32_B64;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
          context, source_op, descriptor_ref, operands,
          IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &sgpr_type,
          1, &count_op));
      low_count = loom_value_slice_get(loom_low_op_results(count_op), 0);
      if (result_unit_count == 2) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0,
            sgpr_type, &low_zero));
      }
      break;
    }
    case LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_VGPR_B32:
    case LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_VGPR_B64: {
      loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_low_lower_lookup_value(context, plan->source, &low_source));
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32_registers(
          context, source_op, low_source, &low_source));
      loom_type_t vgpr_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
      loom_value_id_t low_source_half = low_source;
      if (plan->kind == LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_VGPR_B64) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
            context, source_op, low_source, /*lane_offset=*/0, vgpr_type,
            &low_source_half));
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_unary(
          context, source_op,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_BCNT_U32_B32_SRC1_ZERO, low_source_half,
          vgpr_type, &low_count));
      if (plan->kind == LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_VGPR_B64) {
        loom_value_id_t high_source_half = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
            context, source_op, low_source, /*lane_offset=*/1, vgpr_type,
            &high_source_half));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_BCNT_U32_B32,
            high_source_half, low_count, vgpr_type, &low_count));
      }
      if (result_unit_count == 2) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
            vgpr_type, &low_zero));
      }
      break;
    }
    case LOOM_AMDGPU_SCALAR_I64_CTPOP_KIND_NONE:
      IREE_ASSERT_UNREACHABLE("unknown AMDGPU scalar i64 ctpop plan kind");
      return iree_ok_status();
  }

  if (result_unit_count == 1) {
    return loom_low_lower_bind_value(context, plan->result, low_count);
  }
  const loom_value_id_t low_halves[] = {
      low_count,
      low_zero,
  };
  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_build_low_register_range(
      context, source_op, low_halves, IREE_ARRAYSIZE(low_halves), result_type,
      &low_result));
  return loom_low_lower_bind_value(context, plan->result, low_result);
}
