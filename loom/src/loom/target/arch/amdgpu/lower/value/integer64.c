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
#include "loom/target/arch/amdgpu/lower/source_integer_representation.h"
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
            .constraint_key = IREE_SVL("descriptor.s_add_co_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_CO_U32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.s_addc_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_ADDC_U32,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuSccWordDescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.s_mov_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.s_cselect_b32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_CSELECT_B32,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuScalarI64MulSgprDescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.s_mul_i32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_I32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.s_mul_hi_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_MUL_HI_U32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.s_add_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
        },
};

static const loom_amdgpu_descriptor_requirement_t
    kAmdgpuScalarI64SubSgprDescriptorRequirements[] = {
        {
            .constraint_key = IREE_SVL("descriptor.s_sub_co_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_SUB_CO_U32,
        },
        {
            .constraint_key = IREE_SVL("descriptor.s_subb_u32"),
            .descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_S_SUBB_U32,
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
    kAmdgpuAddressI64ShlVgprDescriptorRequirements[] = {
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

static const loom_amdgpu_i64_alu_descriptor_requirement_row_t
    kAmdgpuScalarI64AluDescriptorRequirementRows[] = {
        [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_SGPR_ADD] =
            {
                .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                    kAmdgpuOffsetAddSgprDescriptorRequirements),
            },
        [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_SGPR_MUL_LO] =
            {
                .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                    kAmdgpuScalarI64MulSgprDescriptorRequirements),
            },
        [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_SGPR_SUB] =
            {
                .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                    kAmdgpuScalarI64SubSgprDescriptorRequirements),
            },
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
        [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_ADDCARRY] =
            {
                .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                    kAmdgpuOffsetAddVgprDescriptorRequirements),
            },
        [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SUBBORROW] =
            {
                .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                    kAmdgpuScalarI64SubVgprDescriptorRequirements),
            },
        [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_SGPR_ADDCARRY] =
            {
                .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                    kAmdgpuOffsetAddSgprDescriptorRequirements),
                .second = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                    kAmdgpuSccWordDescriptorRequirements),
            },
        [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_SGPR_SUBBORROW] =
            {
                .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                    kAmdgpuScalarI64SubSgprDescriptorRequirements),
                .second = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                    kAmdgpuSccWordDescriptorRequirements),
            },
};

static const loom_amdgpu_i64_alu_descriptor_requirement_row_t
    kAmdgpuAddressI64AluDescriptorRequirementRows
        [LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_SGPR_MUL_LO + 1] = {
            [LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_SGPR_ADD] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuOffsetAddSgprDescriptorRequirements),
                },
            [LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_SGPR_MUL_LO] =
                {
                    .first = LOOM_AMDGPU_DESCRIPTOR_REQUIREMENT_SPAN(
                        kAmdgpuScalarI64MulSgprDescriptorRequirements),
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
                        kAmdgpuAddressI64ShlVgprDescriptorRequirements),
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

static const loom_amdgpu_descriptor_ref_t
    kAmdgpuIndexCastSignDescriptorRefs[LOOM_AMDGPU_REG_CLASS_ID_VGPR + 1] = {
        [LOOM_AMDGPU_REG_CLASS_ID_SGPR] =
            LOOM_AMDGPU_DESCRIPTOR_REF_S_ASHR_I32_RHS_INLINE,
        [LOOM_AMDGPU_REG_CLASS_ID_VGPR] =
            LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT,
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

bool loom_amdgpu_address_comparison_needs_pair(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_op_t* source_op) {
  const loom_value_id_t lhs = loom_index_cmp_lhs(source_op);
  const loom_value_id_t rhs = loom_index_cmp_rhs(source_op);
  const loom_type_t type = loom_module_value_type(module, lhs);
  if (loom_amdgpu_source_address_value_needs_64bit(module, fact_table, lhs,
                                                   type) ||
      loom_amdgpu_source_address_value_needs_64bit(module, fact_table, rhs,
                                                   type)) {
    return true;
  }
  if (loom_type_element_type(type) != LOOM_SCALAR_TYPE_INDEX) {
    return false;
  }
  const loom_value_facts_t lhs_facts =
      loom_value_fact_table_lookup(fact_table, lhs);
  const loom_value_facts_t rhs_facts =
      loom_value_fact_table_lookup(fact_table, rhs);
  const bool positive_high_bit =
      (lhs_facts.range_lo >= INT32_MIN && lhs_facts.range_hi > INT32_MAX) ||
      (rhs_facts.range_lo >= INT32_MIN && rhs_facts.range_hi > INT32_MAX);
  const loom_index_cmp_predicate_t predicate =
      loom_index_cmp_predicate(source_op);
  const bool signed_order = predicate >= LOOM_INDEX_CMP_PREDICATE_SLT &&
                            predicate <= LOOM_INDEX_CMP_PREDICATE_SGE;
  // Equality and unsigned order of nonnegative word magnitudes need only the
  // low words. A signed comparison or a possibly negative peer needs the high
  // words to distinguish positive magnitudes from sign-extended coordinates.
  return positive_high_bit &&
         (signed_order || lhs_facts.range_lo < 0 || rhs_facts.range_lo < 0);
}

static iree_status_t loom_amdgpu_mapped_address_value_is_wide(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value_id, bool* out_is_wide) {
  *out_is_wide = false;
  loom_type_t low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, value_id, &low_type));
  *out_is_wide = loom_low_type_is_register(low_type) &&
                 loom_low_register_type_unit_count(low_type) == 2;
  return iree_ok_status();
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

static const loom_amdgpu_descriptor_ref_t
    kAmdgpuI64CompareDescriptors[LOOM_SCALAR_CMPI_PREDICATE_COUNT_] = {
        [LOOM_SCALAR_CMPI_PREDICATE_EQ] =
            LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I64,
        [LOOM_SCALAR_CMPI_PREDICATE_NE] =
            LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I64,
        [LOOM_SCALAR_CMPI_PREDICATE_SLT] =
            LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_SLT_I64,
        [LOOM_SCALAR_CMPI_PREDICATE_SLE] =
            LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_SLE_I64,
        [LOOM_SCALAR_CMPI_PREDICATE_SGT] =
            LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_SGT_I64,
        [LOOM_SCALAR_CMPI_PREDICATE_SGE] =
            LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_SGE_I64,
        [LOOM_SCALAR_CMPI_PREDICATE_ULT] =
            LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U64,
        [LOOM_SCALAR_CMPI_PREDICATE_ULE] =
            LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULE_U64,
        [LOOM_SCALAR_CMPI_PREDICATE_UGT] =
            LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGT_U64,
        [LOOM_SCALAR_CMPI_PREDICATE_UGE] =
            LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_UGE_U64,
};

static bool loom_amdgpu_i64_compare_descriptors_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    iree_string_view_t* out_constraint_key) {
  if (!loom_amdgpu_descriptor_requirements_present(
          descriptor_set, kAmdgpuVgprMoveDescriptorRequirements,
          IREE_ARRAYSIZE(kAmdgpuVgprMoveDescriptorRequirements),
          out_constraint_key)) {
    return false;
  }
  return loom_amdgpu_descriptor_requirement_present(
      descriptor_set, IREE_SV("descriptor.compare"), descriptor_ref,
      out_constraint_key);
}

iree_status_t loom_amdgpu_low_legality_verify_address_compare(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;

  const loom_module_t* module = loom_target_low_legality_module(context);
  if (!loom_amdgpu_address_comparison_needs_pair(
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

  loom_scalar_cmpi_predicate_t predicate;
  if (!loom_amdgpu_index_cmp_predicate_to_scalar(loom_index_cmp_predicate(op),
                                                 &predicate)) {
    return loom_amdgpu_low_legality_reject(context, op, IREE_SV("predicate"));
  }
  iree_string_view_t constraint_key = iree_string_view_empty();
  if (loom_amdgpu_i64_compare_descriptors_supported(
          loom_target_low_legality_descriptor_set(context),
          kAmdgpuI64CompareDescriptors[predicate], &constraint_key)) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_reject(context, op, constraint_key);
}

iree_status_t loom_amdgpu_low_legality_verify_scalar_cmpi_i64(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;

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

  const loom_scalar_cmpi_predicate_t predicate =
      (loom_scalar_cmpi_predicate_t)loom_scalar_cmpi_predicate(op);
  iree_string_view_t constraint_key = iree_string_view_empty();
  if (loom_amdgpu_i64_compare_descriptors_supported(
          loom_target_low_legality_descriptor_set(context),
          kAmdgpuI64CompareDescriptors[predicate], &constraint_key)) {
    return iree_ok_status();
  }
  return loom_amdgpu_low_legality_reject(context, op, constraint_key);
}

static iree_status_t loom_amdgpu_emit_index_cast_range_diagnostic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t source_type, loom_type_t result_type,
    loom_value_facts_t source_facts, int64_t required_lo, int64_t required_hi) {
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
      loom_param_u32(32),
      loom_param_i64(required_lo),
      loom_param_i64(required_hi),
      loom_param_string(IREE_SV("index_cast.target_width_range")),
      loom_param_string_list(accepted_proof_sources,
                             IREE_ARRAYSIZE(accepted_proof_sources)),
  };
  return loom_low_lower_emit_error_ref(context, source_op,
                                       LOOM_ERR_AMDGPU_033_REF, params,
                                       IREE_ARRAYSIZE(params));
}

static loom_amdgpu_index_cast_kind_t loom_amdgpu_index_cast_preserving_kind(
    bool materialize_vgpr) {
  return materialize_vgpr
             ? LOOM_AMDGPU_INDEX_CAST_KIND_PRESERVING_LOW_BITS_TO_VGPR
             : LOOM_AMDGPU_INDEX_CAST_KIND_PRESERVING_LOW_BITS;
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
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);
  const loom_scalar_type_t source_scalar_type =
      loom_type_element_type(source_type);
  const loom_scalar_type_t result_scalar_type =
      loom_type_element_type(result_type);
  const uint32_t source_unit_count =
      loom_low_register_type_unit_count(source_low_type);
  const uint32_t result_unit_count =
      loom_low_register_type_unit_count(result_low_type);
  const uint16_t source_register_class =
      loom_low_register_type_class_id(source_low_type);
  const uint16_t result_register_class =
      loom_low_register_type_class_id(result_low_type);
  if (source_scalar_type == LOOM_SCALAR_TYPE_I1 &&
      loom_amdgpu_type_is_address_scalar(result_type)) {
    *out_plan = (loom_amdgpu_index_cast_plan_t){
        .kind = LOOM_AMDGPU_INDEX_CAST_KIND_PREDICATE_TO_INTEGER,
        .source = source,
        .result = result,
        .result_register_class = result_register_class,
        .result_unit_count = result_unit_count,
    };
    *out_selected = true;
    return iree_ok_status();
  }
  if ((source_scalar_type == LOOM_SCALAR_TYPE_I8 ||
       source_scalar_type == LOOM_SCALAR_TYPE_I16) &&
      loom_amdgpu_type_is_address_scalar(result_type)) {
    if (source_register_class != result_register_class ||
        (source_register_class != LOOM_AMDGPU_REG_CLASS_ID_SGPR &&
         source_register_class != LOOM_AMDGPU_REG_CLASS_ID_VGPR)) {
      return iree_ok_status();
    }
    const loom_low_representation_id_t required_representation =
        result_scalar_type == LOOM_SCALAR_TYPE_INDEX
            ? LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_SIGN_EXTENDED
            : LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_ZERO_EXTENDED;
    const loom_low_representation_id_t source_representation =
        loom_amdgpu_source_integer_representation_lookup(context, source);
    const bool needs_normalization =
        source_representation != required_representation;
    const bool sign_extend =
        required_representation ==
        LOOM_AMDGPU_NARROW_INTEGER_REPRESENTATION_SIGN_EXTENDED;
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_lower_context_descriptor_set(context);
    if (needs_normalization) {
      if (source_register_class == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
        const loom_amdgpu_descriptor_ref_t descriptor_ref =
            sign_extend ? LOOM_AMDGPU_DESCRIPTOR_REF_S_BFE_I32_LIT
                        : LOOM_AMDGPU_DESCRIPTOR_REF_S_BFE_U32_LIT;
        if (!loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                                descriptor_ref)) {
          return iree_ok_status();
        }
      } else {
        const loom_amdgpu_descriptor_ref_t descriptor_refs[] = {
            sign_extend ? LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT
                        : LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
            sign_extend ? LOOM_AMDGPU_DESCRIPTOR_REF_V_ASHRREV_I32_LIT
                        : LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
        };
        if (!loom_amdgpu_descriptor_set_has_all_refs(
                descriptor_set, descriptor_refs,
                IREE_ARRAYSIZE(descriptor_refs))) {
          return iree_ok_status();
        }
      }
    }
    const loom_amdgpu_descriptor_ref_t extension_descriptor_ref =
        sign_extend ? kAmdgpuIndexCastSignDescriptorRefs[result_register_class]
                    : kAmdgpuIndexCastZeroDescriptorRefs[result_register_class];
    if (result_unit_count == 2 &&
        !loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                            extension_descriptor_ref)) {
      return iree_ok_status();
    }
    loom_amdgpu_index_cast_kind_t kind =
        LOOM_AMDGPU_INDEX_CAST_KIND_PRESERVING_LOW_BITS;
    if (needs_normalization) {
      kind = LOOM_AMDGPU_INDEX_CAST_KIND_NORMALIZING_NARROW;
    } else if (result_unit_count == 2) {
      kind = sign_extend ? LOOM_AMDGPU_INDEX_CAST_KIND_SIGN_EXTENDING_LOW_32
                         : LOOM_AMDGPU_INDEX_CAST_KIND_ZERO_EXTENDING_LOW_32;
    }
    *out_plan = (loom_amdgpu_index_cast_plan_t){
        .kind = kind,
        .source = source,
        .result = result,
        .result_register_class = result_register_class,
        .conversion_descriptor_ref = sign_extend
                                         ? LOOM_AMDGPU_DESCRIPTOR_REF_NONE
                                         : extension_descriptor_ref,
        .result_unit_count = result_unit_count,
        .payload_bit_count =
            (uint8_t)loom_scalar_type_bitwidth(source_scalar_type),
    };
    *out_selected = true;
    return iree_ok_status();
  }
  if (loom_amdgpu_type_is_address_scalar(source_type) &&
      (result_scalar_type == LOOM_SCALAR_TYPE_I1 ||
       result_scalar_type == LOOM_SCALAR_TYPE_I8 ||
       result_scalar_type == LOOM_SCALAR_TYPE_I16)) {
    *out_plan = (loom_amdgpu_index_cast_plan_t){
        .kind = result_scalar_type == LOOM_SCALAR_TYPE_I1
                    ? LOOM_AMDGPU_INDEX_CAST_KIND_INTEGER_TO_PREDICATE
                    : LOOM_AMDGPU_INDEX_CAST_KIND_NARROWING_INTEGER,
        .source = source,
        .result = result,
        .result_register_class = result_register_class,
        .payload_bit_count =
            (uint8_t)loom_scalar_type_bitwidth(result_scalar_type),
        .result_unit_count = result_unit_count,
    };
    *out_selected = true;
    return iree_ok_status();
  }
  if (loom_type_equal(source_low_type, result_low_type)) {
    *out_plan = (loom_amdgpu_index_cast_plan_t){
        .kind = LOOM_AMDGPU_INDEX_CAST_KIND_PRESERVING_LOW_BITS,
        .source = source,
        .result = result,
        .result_register_class = result_register_class,
        .result_unit_count = result_unit_count,
    };
    *out_selected = true;
    return iree_ok_status();
  }
  if (source_unit_count == result_unit_count &&
      source_register_class == LOOM_AMDGPU_REG_CLASS_ID_SGPR &&
      result_register_class == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    *out_plan = (loom_amdgpu_index_cast_plan_t){
        .kind = LOOM_AMDGPU_INDEX_CAST_KIND_PRESERVING_LOW_BITS_TO_VGPR,
        .source = source,
        .result = result,
        .result_register_class = result_register_class,
        .result_unit_count = result_unit_count,
    };
    *out_selected = true;
    return iree_ok_status();
  }
  if ((source_register_class != LOOM_AMDGPU_REG_CLASS_ID_SGPR &&
       source_register_class != LOOM_AMDGPU_REG_CLASS_ID_VGPR) ||
      (source_register_class != result_register_class &&
       !(source_register_class == LOOM_AMDGPU_REG_CLASS_ID_SGPR &&
         result_register_class == LOOM_AMDGPU_REG_CLASS_ID_VGPR))) {
    return iree_ok_status();
  }
  const bool materialize_vgpr =
      source_register_class == LOOM_AMDGPU_REG_CLASS_ID_SGPR &&
      result_register_class == LOOM_AMDGPU_REG_CLASS_ID_VGPR;

  loom_value_facts_t source_facts = loom_value_facts_unknown();
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (fact_table != NULL && source < module->values.count) {
    source_facts = loom_value_fact_table_lookup(fact_table, source);
  }

  // Structural representation planning can widen an address value beyond its
  // local facts while a type-changing cast begins a separately planned
  // component. Materialize that physical width boundary without changing the
  // index/offset numeric conversion contract.
  if (loom_amdgpu_type_is_address_scalar(source_type) &&
      loom_amdgpu_type_is_address_scalar(result_type)) {
    if (source_unit_count == 2 && result_unit_count == 1) {
      const bool result_is_offset =
          result_scalar_type == LOOM_SCALAR_TYPE_OFFSET;
      const int64_t required_lo = result_is_offset ? 0 : INT32_MIN;
      const int64_t required_hi =
          result_is_offset ? (int64_t)UINT32_MAX : INT32_MAX;
      if (source_facts.range_lo < required_lo ||
          source_facts.range_hi > required_hi) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_index_cast_range_diagnostic(
            context, source_op, source_type, result_type, source_facts,
            required_lo, required_hi));
        *out_plan = (loom_amdgpu_index_cast_plan_t){
            .kind = LOOM_AMDGPU_INDEX_CAST_KIND_DIAGNOSTIC_REJECTED,
            .source = source,
            .result = result,
            .result_register_class = result_register_class,
            .result_unit_count = result_unit_count,
        };
        *out_selected = true;
        return iree_ok_status();
      }
      *out_plan = (loom_amdgpu_index_cast_plan_t){
          .kind = loom_amdgpu_index_cast_preserving_kind(materialize_vgpr),
          .source = source,
          .result = result,
          .result_register_class = result_register_class,
          .result_unit_count = result_unit_count,
      };
      *out_selected = true;
      return iree_ok_status();
    }
    if (source_unit_count == 1 && result_unit_count == 2) {
      const loom_amdgpu_descriptor_ref_t extension_descriptor_ref =
          kAmdgpuIndexCastZeroDescriptorRefs[result_register_class];
      if (!loom_amdgpu_descriptor_set_has_ref(
              loom_low_lower_context_descriptor_set(context),
              extension_descriptor_ref)) {
        return iree_ok_status();
      }
      *out_plan = (loom_amdgpu_index_cast_plan_t){
          .kind = LOOM_AMDGPU_INDEX_CAST_KIND_ZERO_EXTENDING_LOW_32,
          .source = source,
          .result = result,
          .result_register_class = result_register_class,
          .conversion_descriptor_ref = extension_descriptor_ref,
          .result_unit_count = result_unit_count,
      };
      *out_selected = true;
      return iree_ok_status();
    }
    return iree_ok_status();
  }

  if (source_scalar_type == LOOM_SCALAR_TYPE_I32 &&
      loom_amdgpu_type_is_address_scalar(result_type)) {
    if (source_unit_count != 1 ||
        (result_unit_count != 1 && result_unit_count != 2)) {
      return iree_ok_status();
    }
    if (result_unit_count == 1) {
      *out_plan = (loom_amdgpu_index_cast_plan_t){
          .kind = loom_amdgpu_index_cast_preserving_kind(materialize_vgpr),
          .source = source,
          .result = result,
          .result_register_class = result_register_class,
          .result_unit_count = result_unit_count,
      };
      *out_selected = true;
      return iree_ok_status();
    }
    const bool sign_extend = result_scalar_type == LOOM_SCALAR_TYPE_INDEX;
    const loom_amdgpu_descriptor_ref_t extension_descriptor_ref =
        sign_extend ? kAmdgpuIndexCastSignDescriptorRefs[result_register_class]
                    : kAmdgpuIndexCastZeroDescriptorRefs[result_register_class];
    if (!loom_amdgpu_descriptor_set_has_ref(
            loom_low_lower_context_descriptor_set(context),
            extension_descriptor_ref)) {
      return iree_ok_status();
    }
    *out_plan = (loom_amdgpu_index_cast_plan_t){
        .kind = sign_extend ? LOOM_AMDGPU_INDEX_CAST_KIND_SIGN_EXTENDING_LOW_32
                            : LOOM_AMDGPU_INDEX_CAST_KIND_ZERO_EXTENDING_LOW_32,
        .source = source,
        .result = result,
        .result_register_class = result_register_class,
        .conversion_descriptor_ref = sign_extend
                                         ? LOOM_AMDGPU_DESCRIPTOR_REF_NONE
                                         : extension_descriptor_ref,
        .result_unit_count = result_unit_count,
    };
    *out_selected = true;
    return iree_ok_status();
  }

  switch (source_scalar_type) {
    case LOOM_SCALAR_TYPE_I64: {
      if ((result_scalar_type != LOOM_SCALAR_TYPE_INDEX &&
           result_scalar_type != LOOM_SCALAR_TYPE_OFFSET) ||
          source_unit_count != 2 || result_unit_count != 1) {
        return iree_ok_status();
      }
      const bool is_offset = result_scalar_type == LOOM_SCALAR_TYPE_OFFSET;
      const int64_t required_lo = is_offset ? 0 : INT32_MIN;
      const int64_t required_hi = is_offset ? (int64_t)UINT32_MAX : INT32_MAX;
      if ((!is_offset && source_facts.range_lo < required_lo) ||
          source_facts.range_hi > required_hi) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_index_cast_range_diagnostic(
            context, source_op, source_type, result_type, source_facts,
            required_lo, required_hi));
        *out_plan = (loom_amdgpu_index_cast_plan_t){
            .kind = LOOM_AMDGPU_INDEX_CAST_KIND_DIAGNOSTIC_REJECTED,
            .source = source,
            .result = result,
            .result_register_class = result_register_class,
            .result_unit_count = result_unit_count,
        };
        *out_selected = true;
        return iree_ok_status();
      }
      *out_plan = (loom_amdgpu_index_cast_plan_t){
          .kind = loom_amdgpu_index_cast_preserving_kind(materialize_vgpr),
          .source = source,
          .result = result,
          .result_register_class = result_register_class,
          .result_unit_count = result_unit_count,
      };
      *out_selected = true;
      return iree_ok_status();
    }
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_OFFSET:
      if (result_scalar_type == LOOM_SCALAR_TYPE_I32 &&
          source_unit_count == 2 && result_unit_count == 1) {
        *out_plan = (loom_amdgpu_index_cast_plan_t){
            .kind = loom_amdgpu_index_cast_preserving_kind(materialize_vgpr),
            .source = source,
            .result = result,
            .result_register_class = result_register_class,
            .result_unit_count = result_unit_count,
        };
        *out_selected = true;
        return iree_ok_status();
      }
      if (result_scalar_type != LOOM_SCALAR_TYPE_I64 ||
          source_unit_count != 1 || result_unit_count != 2) {
        return iree_ok_status();
      }
      break;
    default:
      return iree_ok_status();
  }

  const bool sign_extend =
      source_scalar_type == LOOM_SCALAR_TYPE_INDEX && source_facts.range_lo < 0;
  const loom_amdgpu_descriptor_ref_t extension_descriptor_ref =
      sign_extend ? kAmdgpuIndexCastSignDescriptorRefs[result_register_class]
                  : kAmdgpuIndexCastZeroDescriptorRefs[result_register_class];
  if (!loom_amdgpu_descriptor_set_has_ref(
          loom_low_lower_context_descriptor_set(context),
          extension_descriptor_ref)) {
    return iree_ok_status();
  }
  *out_plan = (loom_amdgpu_index_cast_plan_t){
      .kind = sign_extend ? LOOM_AMDGPU_INDEX_CAST_KIND_SIGN_EXTENDING_LOW_32
                          : LOOM_AMDGPU_INDEX_CAST_KIND_ZERO_EXTENDING_LOW_32,
      .source = source,
      .result = result,
      .result_register_class = result_register_class,
      .conversion_descriptor_ref = sign_extend ? LOOM_AMDGPU_DESCRIPTOR_REF_NONE
                                               : extension_descriptor_ref,
      .result_unit_count = result_unit_count,
  };
  *out_selected = true;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_finalize_index_cast_plan(
    loom_low_lower_context_t* context, loom_amdgpu_index_cast_plan_t* plan) {
  if (plan->kind == LOOM_AMDGPU_INDEX_CAST_KIND_DIAGNOSTIC_REJECTED) {
    return iree_ok_status();
  }
  const loom_type_t source_type =
      loom_low_lower_value_binding_type(context, plan->source);
  const uint16_t source_register_class =
      loom_low_register_type_class_id(source_type);
  if (plan->kind == LOOM_AMDGPU_INDEX_CAST_KIND_ZERO_EXTENDING_LOW_32 ||
      plan->kind == LOOM_AMDGPU_INDEX_CAST_KIND_SIGN_EXTENDING_LOW_32) {
    // Address producers can retain a wider carrier than their preferred fact
    // mapping, including callable results. Such a carrier already contains the
    // high word; extending its low word would destroy it.
    if (loom_amdgpu_type_is_address_scalar(loom_module_value_type(
            loom_low_lower_context_module(context), plan->source)) &&
        loom_low_register_type_unit_count(source_type) == 2) {
      plan->kind = loom_amdgpu_index_cast_preserving_kind(
          source_register_class == LOOM_AMDGPU_REG_CLASS_ID_SGPR &&
          plan->result_register_class == LOOM_AMDGPU_REG_CLASS_ID_VGPR);
    } else if (source_register_class == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
      plan->result_register_class = source_register_class;
      if (plan->kind == LOOM_AMDGPU_INDEX_CAST_KIND_ZERO_EXTENDING_LOW_32) {
        plan->conversion_descriptor_ref =
            kAmdgpuIndexCastZeroDescriptorRefs[source_register_class];
      }
    }
  }
  const uint16_t register_class =
      plan->kind == LOOM_AMDGPU_INDEX_CAST_KIND_PRESERVING_LOW_BITS
          ? source_register_class
          : plan->result_register_class;
  const loom_type_t result_type = loom_low_register_type(
      loom_low_lower_context_descriptor_set(context)->stable_id, register_class,
      plan->result_unit_count);
  return loom_low_lower_plan_value_type(context, plan->result, result_type);
}

iree_status_t loom_amdgpu_select_index_cmp_i64_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_i64_compare_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_i64_compare_plan_t){0};
  *out_selected = false;
  if (!loom_amdgpu_address_comparison_needs_pair(
          loom_low_lower_context_module(context),
          loom_low_lower_context_fact_table(context), source_op)) {
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
      .operand_forms = loom_amdgpu_select_integer_operand_form(
                           loom_value_fact_table_lookup(
                               loom_low_lower_context_fact_table(context),
                               loom_index_cmp_lhs(source_op)),
                           2) |
                       (loom_amdgpu_select_integer_operand_form(
                            loom_value_fact_table_lookup(
                                loom_low_lower_context_fact_table(context),
                                loom_index_cmp_rhs(source_op)),
                            2)
                        << 2),
  };
  loom_scalar_cmpi_predicate_t predicate;
  if (!loom_amdgpu_index_cmp_predicate_to_scalar(
          loom_index_cmp_predicate(source_op), &predicate)) {
    return iree_ok_status();
  }

  plan.descriptor_ref = kAmdgpuI64CompareDescriptors[predicate];
  iree_string_view_t constraint_key = iree_string_view_empty();
  if (!loom_amdgpu_i64_compare_descriptors_supported(
          loom_low_lower_context_descriptor_set(context), plan.descriptor_ref,
          &constraint_key)) {
    return iree_ok_status();
  }

  *out_plan = plan;
  *out_selected = true;
  return loom_low_lower_plan_value_type(context, result, result_low_type);
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
  plan.descriptor_ref =
      kAmdgpuI64CompareDescriptors[loom_scalar_cmpi_predicate(source_op)];

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (!loom_amdgpu_i64_compare_descriptors_supported(
          loom_low_lower_context_descriptor_set(context), plan.descriptor_ref,
          &constraint_key)) {
    return iree_ok_status();
  }

  *out_plan = plan;
  *out_selected = true;
  return loom_low_lower_plan_value_type(context, result, result_low_type);
}

static bool loom_amdgpu_scalar_i64_alu_descriptors_supported(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_scalar_i64_alu_kind_t kind,
    iree_string_view_t* out_constraint_key) {
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
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_INDEX_SCALE)] =
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
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_ADDCARRYUI)] =
            LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_ADDCARRY,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_SUBBORROWUI)] =
            LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SUBBORROW,
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_SCALAR_MULI)] =
            LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_MUL_LO,
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
  if (loom_op_dialect_id(source_op->kind) != LOOM_DIALECT_INDEX) {
    return false;
  }
  const uint8_t op_index = loom_op_dialect_index(source_op->kind);
  if (op_index >= IREE_ARRAYSIZE(kAmdgpuAddressI64AluSourceLayoutByIndexOp)) {
    return false;
  }
  const loom_amdgpu_address_i64_alu_source_layout_t* layout =
      &kAmdgpuAddressI64AluSourceLayoutByIndexOp[op_index];
  if (layout->kind == LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE) {
    return false;
  }
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

static loom_amdgpu_address_i64_alu_kind_t loom_amdgpu_address_i64_alu_sgpr_kind(
    loom_amdgpu_address_i64_alu_kind_t operation_kind) {
  switch (operation_kind) {
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_ADD:
      return LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_SGPR_ADD;
    case LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MUL_LO:
      return LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_SGPR_MUL_LO;
    default:
      return LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE;
  }
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

  const bool result_is_sgpr64 = loom_amdgpu_low_type_is_register_class_count(
      context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2);
  if (result_is_sgpr64) {
    *out_kind = loom_amdgpu_address_i64_alu_sgpr_kind(operation_kind);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_address_i64_operand_can_materialize(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, loom_amdgpu_address_i64_alu_kind_t result_kind,
    bool* out_can_lower) {
  if (loom_amdgpu_address_i64_alu_kind_uses_vgpr(result_kind)) {
    return loom_amdgpu_value_can_materialize_as_vgpr_address(
        context, source_op, source, out_can_lower);
  }
  return loom_amdgpu_value_can_materialize_as_sgpr_address(
      context, source_op, source, out_can_lower);
}

loom_amdgpu_integer_operand_form_t loom_amdgpu_select_integer_operand_form(
    loom_value_facts_t facts, uint8_t minimum_unit_count) {
  if (minimum_unit_count == 1 &&
      loom_value_facts_fit_unsigned_bit_count(facts, 32)) {
    return LOOM_AMDGPU_INTEGER_OPERAND_UNSIGNED_WORD;
  }
  return facts.range_lo < 0 ? LOOM_AMDGPU_INTEGER_OPERAND_SIGNED_PAIR
                            : LOOM_AMDGPU_INTEGER_OPERAND_UNSIGNED_PAIR;
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

  bool result_needs_wide = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_mapped_address_value_is_wide(
      context, source_op, result, &result_needs_wide));
  if (!result_needs_wide) {
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

  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const bool has_product =
      kind == LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_SGPR_MUL_LO ||
      kind == LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MUL_LO ||
      kind == LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_VGPR_MADD_LO;
  const uint8_t minimum_unit_count = has_product ? 1 : 2;
  *out_plan = (loom_amdgpu_address_i64_alu_plan_t){
      .lhs = lhs,
      .rhs = rhs,
      .addend = addend,
      .result = result,
      .kind = kind,
      .operands =
          {
              .lhs = loom_amdgpu_select_integer_operand_form(
                  loom_value_fact_table_lookup(fact_table, lhs),
                  minimum_unit_count),
              .rhs = loom_amdgpu_select_integer_operand_form(
                  loom_value_fact_table_lookup(fact_table, rhs),
                  minimum_unit_count),
              .addend =
                  addend == LOOM_VALUE_ID_INVALID
                      ? LOOM_AMDGPU_INTEGER_OPERAND_UNSIGNED_PAIR
                      : loom_amdgpu_select_integer_operand_form(
                            loom_value_fact_table_lookup(fact_table, addend),
                            2),
          },
  };
  *out_selected = true;
  const loom_type_t result_type = loom_low_register_type(
      loom_low_lower_context_descriptor_set(context)->stable_id,
      loom_amdgpu_address_i64_alu_kind_uses_vgpr(kind)
          ? LOOM_AMDGPU_REG_CLASS_ID_VGPR
          : LOOM_AMDGPU_REG_CLASS_ID_SGPR,
      2);
  return loom_low_lower_plan_value_type(context, result, result_type);
}

iree_status_t loom_amdgpu_low_legality_verify_address_i64_alu(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
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
  if (!result_prefers_vgpr) {
    kind = loom_amdgpu_address_i64_alu_sgpr_kind(operation_kind);
    if (kind == LOOM_AMDGPU_ADDRESS_I64_ALU_KIND_NONE) {
      return loom_amdgpu_low_legality_reject(context, op,
                                             IREE_SV("result.vgpr64"));
    }
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
    loom_value_id_t* out_result, loom_value_id_t* out_predicate_result) {
  *out_kind = LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE;
  *out_lhs = LOOM_VALUE_ID_INVALID;
  *out_rhs = LOOM_VALUE_ID_INVALID;
  *out_result = LOOM_VALUE_ID_INVALID;
  *out_predicate_result = LOOM_VALUE_ID_INVALID;
  if (loom_op_dialect_id(source_op->kind) != LOOM_DIALECT_SCALAR) {
    return false;
  }
  const uint8_t op_index = loom_op_dialect_index(source_op->kind);
  if (op_index >= IREE_ARRAYSIZE(kAmdgpuScalarI64AluKindByScalarOp)) {
    return false;
  }
  const loom_amdgpu_scalar_i64_alu_kind_t kind =
      kAmdgpuScalarI64AluKindByScalarOp[op_index];
  if (kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE) {
    return false;
  }
  const loom_value_id_t* operands = loom_op_const_operands(source_op);
  *out_kind = kind;
  *out_lhs = operands[0];
  *out_rhs = operands[1];
  *out_result = loom_op_const_results(source_op)[0];
  if (kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_ADDCARRY ||
      kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SUBBORROW) {
    *out_predicate_result = loom_op_const_results(source_op)[1];
  }
  return true;
}

static loom_amdgpu_scalar_i64_alu_kind_t loom_amdgpu_scalar_i64_alu_sgpr_kind(
    loom_amdgpu_scalar_i64_alu_kind_t kind) {
  static const loom_amdgpu_scalar_i64_alu_kind_t kinds[] = {
      [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE] =
          LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE,
      [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_ADD] =
          LOOM_AMDGPU_SCALAR_I64_ALU_KIND_SGPR_ADD,
      [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SUB] =
          LOOM_AMDGPU_SCALAR_I64_ALU_KIND_SGPR_SUB,
      [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_MUL_LO] =
          LOOM_AMDGPU_SCALAR_I64_ALU_KIND_SGPR_MUL_LO,
      [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_ADDCARRY] =
          LOOM_AMDGPU_SCALAR_I64_ALU_KIND_SGPR_ADDCARRY,
      [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SUBBORROW] =
          LOOM_AMDGPU_SCALAR_I64_ALU_KIND_SGPR_SUBBORROW,
      [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_SGPR_ADDCARRY] =
          LOOM_AMDGPU_SCALAR_I64_ALU_KIND_SGPR_ADDCARRY,
      [LOOM_AMDGPU_SCALAR_I64_ALU_KIND_SGPR_SUBBORROW] =
          LOOM_AMDGPU_SCALAR_I64_ALU_KIND_SGPR_SUBBORROW,
  };
  return kinds[kind];
}

static bool loom_amdgpu_scalar_i64_alu_kind_is_carry_borrow(
    loom_amdgpu_scalar_i64_alu_kind_t kind) {
  return kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_ADDCARRY ||
         kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SUBBORROW ||
         kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_SGPR_ADDCARRY ||
         kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_SGPR_SUBBORROW;
}

static bool loom_amdgpu_scalar_i64_alu_kind_uses_vgpr(
    loom_amdgpu_scalar_i64_alu_kind_t kind) {
  return kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_ADD ||
         kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SUB ||
         kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_MUL_LO ||
         kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_ADDCARRY ||
         kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_SUBBORROW;
}

static iree_status_t loom_amdgpu_scalar_i64_operand_can_materialize(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, uint16_t result_register_class,
    bool* out_can_lower) {
  *out_can_lower = false;
  loom_type_t source_low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_low_lower_map_value(context, source_op, source, &source_low_type));
  if (!loom_low_type_is_register(source_low_type) ||
      loom_low_register_type_unit_count(source_low_type) != 2) {
    return iree_ok_status();
  }
  const uint16_t source_register_class =
      loom_low_register_type_class_id(source_low_type);
  *out_can_lower = source_register_class == result_register_class ||
                   (result_register_class == LOOM_AMDGPU_REG_CLASS_ID_VGPR &&
                    source_register_class == LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  return iree_ok_status();
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
  loom_value_id_t predicate_result = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_scalar_i64_alu_op(source_op, &kind, &lhs, &rhs, &result,
                                     &predicate_result)) {
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
  if (!result_is_vgpr64) {
    if (!loom_amdgpu_low_type_is_register_class_count(
            context, result_low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2)) {
      return iree_ok_status();
    }
    kind = loom_amdgpu_scalar_i64_alu_sgpr_kind(kind);
    if (kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE) {
      return iree_ok_status();
    }
  }
  const uint16_t result_register_class =
      loom_low_register_type_class_id(result_low_type);

  loom_type_t predicate_result_low_type = loom_type_none();
  if (loom_amdgpu_scalar_i64_alu_kind_is_carry_borrow(kind)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_low_result_type(
        context, source_op, predicate_result, &predicate_result_low_type));
    const bool predicate_type_supported =
        loom_amdgpu_scalar_i64_alu_kind_uses_vgpr(kind)
            ? loom_amdgpu_low_type_is_register_class_count(
                  context, predicate_result_low_type,
                  LOOM_AMDGPU_REG_CLASS_ID_SGPR, 2)
            : (loom_amdgpu_low_type_is_register_class_count(
                   context, predicate_result_low_type,
                   LOOM_AMDGPU_REG_CLASS_ID_SCC, 1) ||
               loom_amdgpu_low_type_is_register_class_count(
                   context, predicate_result_low_type,
                   LOOM_AMDGPU_REG_CLASS_ID_SGPR, 1));
    if (!predicate_type_supported) {
      return iree_ok_status();
    }
  }

  bool lhs_can_lower = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_scalar_i64_operand_can_materialize(
      context, source_op, lhs, result_register_class, &lhs_can_lower));
  if (!lhs_can_lower) {
    return iree_ok_status();
  }

  bool rhs_can_lower = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_scalar_i64_operand_can_materialize(
      context, source_op, rhs, result_register_class, &rhs_can_lower));
  if (!rhs_can_lower) {
    return iree_ok_status();
  }

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (!loom_amdgpu_scalar_i64_alu_descriptors_supported(
          loom_low_lower_context_descriptor_set(context), kind,
          &constraint_key)) {
    return iree_ok_status();
  }

  *out_plan = (loom_amdgpu_scalar_i64_alu_plan_t){
      .kind = kind,
      .lhs = lhs,
      .rhs = rhs,
      .result = result,
      .predicate_result = predicate_result,
  };
  if (kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_SGPR_MUL_LO ||
      kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_VGPR_MUL_LO) {
    const loom_value_fact_table_t* fact_table =
        loom_low_lower_context_fact_table(context);
    out_plan->operands.lhs = loom_amdgpu_select_integer_operand_form(
        loom_value_fact_table_lookup(fact_table, lhs), 1);
    out_plan->operands.rhs = loom_amdgpu_select_integer_operand_form(
        loom_value_fact_table_lookup(fact_table, rhs), 1);
  }
  *out_selected = true;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_plan_value_type(context, result, result_low_type));
  if (predicate_result != LOOM_VALUE_ID_INVALID) {
    return loom_low_lower_plan_value_type(context, predicate_result,
                                          predicate_result_low_type);
  }
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
      .result_unit_count =
          (uint8_t)loom_low_register_type_unit_count(result_low_type),
  };
  *out_selected = true;
  return loom_low_lower_plan_value_type(context, result, result_low_type);
}

iree_status_t loom_amdgpu_low_legality_verify_scalar_i64_alu(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  loom_amdgpu_scalar_i64_alu_kind_t kind = LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE;
  loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  loom_value_id_t predicate_result = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_scalar_i64_alu_op(op, &kind, &lhs, &rhs, &result,
                                     &predicate_result)) {
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
  if (predicate_result != LOOM_VALUE_ID_INVALID &&
      !loom_amdgpu_type_is_i1(
          loom_module_value_type(module, predicate_result))) {
    return loom_amdgpu_low_legality_reject(context, op, IREE_SV("result.i1"));
  }

  bool result_prefers_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_low_legality_value_prefers_vgpr(
      context, result, &result_prefers_vgpr));
  if (!result_prefers_vgpr) {
    kind = loom_amdgpu_scalar_i64_alu_sgpr_kind(kind);
    if (kind == LOOM_AMDGPU_SCALAR_I64_ALU_KIND_NONE) {
      return loom_amdgpu_low_legality_reject(context, op,
                                             IREE_SV("result.vgpr64"));
    }
  }

  iree_string_view_t constraint_key = iree_string_view_empty();
  if (loom_amdgpu_scalar_i64_alu_descriptors_supported(
          loom_target_low_legality_descriptor_set(context), kind,
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
