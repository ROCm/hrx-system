// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_SCALAR_OPS_H_
#define LOOM_OPS_SCALAR_OPS_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_SCALAR_ADDI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 0),
  LOOM_OP_SCALAR_SUBI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 1),
  LOOM_OP_SCALAR_MULI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 2),
  LOOM_OP_SCALAR_DIVSI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 3),
  LOOM_OP_SCALAR_DIVUI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 4),
  LOOM_OP_SCALAR_REMSI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 5),
  LOOM_OP_SCALAR_REMUI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 6),
  LOOM_OP_SCALAR_CEILDIVSI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 7),
  LOOM_OP_SCALAR_CEILDIVUI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 8),
  LOOM_OP_SCALAR_FLOORDIVSI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 9),
  LOOM_OP_SCALAR_NEGI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 10),
  LOOM_OP_SCALAR_ABSI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 11),
  LOOM_OP_SCALAR_MINSI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 12),
  LOOM_OP_SCALAR_MAXSI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 13),
  LOOM_OP_SCALAR_MINUI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 14),
  LOOM_OP_SCALAR_MAXUI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 15),
  LOOM_OP_SCALAR_FMAI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 16),
  LOOM_OP_SCALAR_ADDF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 17),
  LOOM_OP_SCALAR_SUBF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 18),
  LOOM_OP_SCALAR_MULF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 19),
  LOOM_OP_SCALAR_DIVF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 20),
  LOOM_OP_SCALAR_REMF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 21),
  LOOM_OP_SCALAR_NEGF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 22),
  LOOM_OP_SCALAR_ABSF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 23),
  LOOM_OP_SCALAR_MINIMUMF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 24),
  LOOM_OP_SCALAR_MAXIMUMF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 25),
  LOOM_OP_SCALAR_MINNUMF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 26),
  LOOM_OP_SCALAR_MAXNUMF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 27),
  LOOM_OP_SCALAR_CLAMPF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 28),
  LOOM_OP_SCALAR_COPYSIGNF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 29),
  LOOM_OP_SCALAR_EXPF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 30),
  LOOM_OP_SCALAR_EXP2F = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 31),
  LOOM_OP_SCALAR_EXPM1F = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 32),
  LOOM_OP_SCALAR_LOGF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 33),
  LOOM_OP_SCALAR_LOG2F = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 34),
  LOOM_OP_SCALAR_LOG10F = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 35),
  LOOM_OP_SCALAR_LOG1PF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 36),
  LOOM_OP_SCALAR_POWF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 37),
  LOOM_OP_SCALAR_SQRTF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 38),
  LOOM_OP_SCALAR_RSQRTF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 39),
  LOOM_OP_SCALAR_CBRTF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 40),
  LOOM_OP_SCALAR_SINF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 41),
  LOOM_OP_SCALAR_COSF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 42),
  LOOM_OP_SCALAR_SINTURNSF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 43),
  LOOM_OP_SCALAR_COSTURNSF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 44),
  LOOM_OP_SCALAR_TANF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 45),
  LOOM_OP_SCALAR_ASINF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 46),
  LOOM_OP_SCALAR_ACOSF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 47),
  LOOM_OP_SCALAR_ATANF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 48),
  LOOM_OP_SCALAR_ATAN2F = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 49),
  LOOM_OP_SCALAR_SINHF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 50),
  LOOM_OP_SCALAR_COSHF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 51),
  LOOM_OP_SCALAR_TANHF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 52),
  LOOM_OP_SCALAR_ASINHF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 53),
  LOOM_OP_SCALAR_ACOSHF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 54),
  LOOM_OP_SCALAR_ATANHF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 55),
  LOOM_OP_SCALAR_ERFF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 56),
  LOOM_OP_SCALAR_ERFCF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 57),
  LOOM_OP_SCALAR_LOGISTICF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 58),
  LOOM_OP_SCALAR_SILUF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 59),
  LOOM_OP_SCALAR_SOFTPLUSF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 60),
  LOOM_OP_SCALAR_GELUF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 61),
  LOOM_OP_SCALAR_FMAF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 62),
  LOOM_OP_SCALAR_CEILF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 63),
  LOOM_OP_SCALAR_FLOORF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 64),
  LOOM_OP_SCALAR_ROUNDF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 65),
  LOOM_OP_SCALAR_ROUNDEVENF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 66),
  LOOM_OP_SCALAR_TRUNCF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 67),
  LOOM_OP_SCALAR_CMPI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 68),
  LOOM_OP_SCALAR_CMPF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 69),
  LOOM_OP_SCALAR_ISNANF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 70),
  LOOM_OP_SCALAR_ISINFF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 71),
  LOOM_OP_SCALAR_ISFINITEF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 72),
  LOOM_OP_SCALAR_SIGNF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 73),
  LOOM_OP_SCALAR_SIGNI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 74),
  LOOM_OP_SCALAR_SITOFP = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 75),
  LOOM_OP_SCALAR_UITOFP = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 76),
  LOOM_OP_SCALAR_FPTOSI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 77),
  LOOM_OP_SCALAR_FPTOUI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 78),
  LOOM_OP_SCALAR_EXTF = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 79),
  LOOM_OP_SCALAR_FPTRUNC = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 80),
  LOOM_OP_SCALAR_EXTSI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 81),
  LOOM_OP_SCALAR_EXTUI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 82),
  LOOM_OP_SCALAR_TRUNCI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 83),
  LOOM_OP_SCALAR_BITCAST = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 84),
  LOOM_OP_SCALAR_CONSTANT = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 85),
  LOOM_OP_SCALAR_POISON = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 86),
  LOOM_OP_SCALAR_ANDI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 87),
  LOOM_OP_SCALAR_ORI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 88),
  LOOM_OP_SCALAR_XORI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 89),
  LOOM_OP_SCALAR_SHLI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 90),
  LOOM_OP_SCALAR_SHRSI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 91),
  LOOM_OP_SCALAR_SHRUI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 92),
  LOOM_OP_SCALAR_ROTLI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 93),
  LOOM_OP_SCALAR_ROTRI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 94),
  LOOM_OP_SCALAR_CTLZI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 95),
  LOOM_OP_SCALAR_CTTZI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 96),
  LOOM_OP_SCALAR_CTPOPI = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 97),
  LOOM_OP_SCALAR_BITFIELD_EXTRACTU = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 98),
  LOOM_OP_SCALAR_BITFIELD_EXTRACTS = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 99),
  LOOM_OP_SCALAR_ASSUME = LOOM_OP_KIND(LOOM_DIALECT_SCALAR, 100),
  LOOM_OP_SCALAR_COUNT_ = 101,
};

// Integer overflow behavior flags.
#define LOOM_SCALAR_INTOVERFLOWFLAGS_NSW ((uint8_t)1)
#define LOOM_SCALAR_INTOVERFLOWFLAGS_NUW ((uint8_t)2)

// IEEE 754 fast-math relaxation flags for float operations.
#define LOOM_SCALAR_FASTMATHFLAGS_REASSOC ((uint8_t)1)
#define LOOM_SCALAR_FASTMATHFLAGS_NNAN ((uint8_t)2)
#define LOOM_SCALAR_FASTMATHFLAGS_NINF ((uint8_t)4)
#define LOOM_SCALAR_FASTMATHFLAGS_NSZ ((uint8_t)8)
#define LOOM_SCALAR_FASTMATHFLAGS_ARCP ((uint8_t)16)
#define LOOM_SCALAR_FASTMATHFLAGS_CONTRACT ((uint8_t)32)
#define LOOM_SCALAR_FASTMATHFLAGS_AFN ((uint8_t)64)
#define LOOM_SCALAR_FASTMATHFLAGS_FAST ((uint8_t)127)

// Floating-point clamp NaN and comparison policy.
typedef enum loom_scalar_clampf_mode_e {
  LOOM_SCALAR_CLAMPF_MODE_ORDERED = 0,
  LOOM_SCALAR_CLAMPF_MODE_NUMBER = 1,
  LOOM_SCALAR_CLAMPF_MODE_IEEE = 2,
  LOOM_SCALAR_CLAMPF_MODE_COUNT_ = 3,
} loom_scalar_clampf_mode_t;

// GELU activation formula family.
typedef enum loom_scalar_geluf_variant_e {
  LOOM_SCALAR_GELUF_VARIANT_ERF = 0,
  LOOM_SCALAR_GELUF_VARIANT_TANH = 1,
  LOOM_SCALAR_GELUF_VARIANT_LOGISTIC = 2,
  LOOM_SCALAR_GELUF_VARIANT_COUNT_ = 3,
} loom_scalar_geluf_variant_t;

// Integer comparison predicates.
typedef enum loom_scalar_cmpi_predicate_e {
  LOOM_SCALAR_CMPI_PREDICATE_EQ = 0,
  LOOM_SCALAR_CMPI_PREDICATE_NE = 1,
  LOOM_SCALAR_CMPI_PREDICATE_SLT = 2,
  LOOM_SCALAR_CMPI_PREDICATE_SLE = 3,
  LOOM_SCALAR_CMPI_PREDICATE_SGT = 4,
  LOOM_SCALAR_CMPI_PREDICATE_SGE = 5,
  LOOM_SCALAR_CMPI_PREDICATE_ULT = 6,
  LOOM_SCALAR_CMPI_PREDICATE_ULE = 7,
  LOOM_SCALAR_CMPI_PREDICATE_UGT = 8,
  LOOM_SCALAR_CMPI_PREDICATE_UGE = 9,
  LOOM_SCALAR_CMPI_PREDICATE_COUNT_ = 10,
} loom_scalar_cmpi_predicate_t;

// Floating-point comparison predicates (IEEE 754 total order).
typedef enum loom_scalar_cmpf_predicate_e {
  LOOM_SCALAR_CMPF_PREDICATE_OEQ = 0,
  LOOM_SCALAR_CMPF_PREDICATE_OGT = 1,
  LOOM_SCALAR_CMPF_PREDICATE_OGE = 2,
  LOOM_SCALAR_CMPF_PREDICATE_OLT = 3,
  LOOM_SCALAR_CMPF_PREDICATE_OLE = 4,
  LOOM_SCALAR_CMPF_PREDICATE_ONE = 5,
  LOOM_SCALAR_CMPF_PREDICATE_ORD = 6,
  LOOM_SCALAR_CMPF_PREDICATE_UEQ = 7,
  LOOM_SCALAR_CMPF_PREDICATE_UGT = 8,
  LOOM_SCALAR_CMPF_PREDICATE_UGE = 9,
  LOOM_SCALAR_CMPF_PREDICATE_ULT = 10,
  LOOM_SCALAR_CMPF_PREDICATE_ULE = 11,
  LOOM_SCALAR_CMPF_PREDICATE_UNE = 12,
  LOOM_SCALAR_CMPF_PREDICATE_UNO = 13,
  LOOM_SCALAR_CMPF_PREDICATE_COUNT_ = 14,
} loom_scalar_cmpf_predicate_t;

// LOOM_OP_SCALAR_ADDI: Integer addition.
// %result = scalar.addi %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_addi_isa, LOOM_OP_SCALAR_ADDI)
LOOM_DEFINE_OPERAND(loom_scalar_addi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_addi_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_addi_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_addi_overflow)
iree_status_t loom_scalar_addi_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_addi_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_addi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_SUBI: Integer subtraction.
// %result = scalar.subi %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_subi_isa, LOOM_OP_SCALAR_SUBI)
LOOM_DEFINE_OPERAND(loom_scalar_subi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_subi_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_subi_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_subi_overflow)
iree_status_t loom_scalar_subi_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_subi_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_subi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_MULI: Integer multiplication.
// %result = scalar.muli %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_muli_isa, LOOM_OP_SCALAR_MULI)
LOOM_DEFINE_OPERAND(loom_scalar_muli_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_muli_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_muli_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_muli_overflow)
iree_status_t loom_scalar_muli_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_muli_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_muli_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_DIVSI: Signed integer division (rounds toward zero).
// %result = scalar.divsi %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_divsi_isa, LOOM_OP_SCALAR_DIVSI)
LOOM_DEFINE_OPERAND(loom_scalar_divsi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_divsi_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_divsi_result, 0)
iree_status_t loom_scalar_divsi_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_divsi_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_divsi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_DIVUI: Unsigned integer division.
// %result = scalar.divui %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_divui_isa, LOOM_OP_SCALAR_DIVUI)
LOOM_DEFINE_OPERAND(loom_scalar_divui_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_divui_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_divui_result, 0)
iree_status_t loom_scalar_divui_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_divui_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_divui_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_REMSI: Signed integer remainder.
// %result = scalar.remsi %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_remsi_isa, LOOM_OP_SCALAR_REMSI)
LOOM_DEFINE_OPERAND(loom_scalar_remsi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_remsi_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_remsi_result, 0)
iree_status_t loom_scalar_remsi_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_remsi_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_remsi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_REMUI: Unsigned integer remainder.
// %result = scalar.remui %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_remui_isa, LOOM_OP_SCALAR_REMUI)
LOOM_DEFINE_OPERAND(loom_scalar_remui_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_remui_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_remui_result, 0)
iree_status_t loom_scalar_remui_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_remui_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_remui_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_CEILDIVSI: Signed integer division, rounding toward positive infinity.
// %result = scalar.ceildivsi %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_ceildivsi_isa, LOOM_OP_SCALAR_CEILDIVSI)
LOOM_DEFINE_OPERAND(loom_scalar_ceildivsi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_ceildivsi_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_ceildivsi_result, 0)
iree_status_t loom_scalar_ceildivsi_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_ceildivsi_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);

// LOOM_OP_SCALAR_CEILDIVUI: Unsigned integer division, rounding toward positive infinity.
// %result = scalar.ceildivui %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_ceildivui_isa, LOOM_OP_SCALAR_CEILDIVUI)
LOOM_DEFINE_OPERAND(loom_scalar_ceildivui_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_ceildivui_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_ceildivui_result, 0)
iree_status_t loom_scalar_ceildivui_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_ceildivui_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);

// LOOM_OP_SCALAR_FLOORDIVSI: Signed integer division, rounding toward negative infinity.
// %result = scalar.floordivsi %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_floordivsi_isa, LOOM_OP_SCALAR_FLOORDIVSI)
LOOM_DEFINE_OPERAND(loom_scalar_floordivsi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_floordivsi_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_floordivsi_result, 0)
iree_status_t loom_scalar_floordivsi_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_floordivsi_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);

// LOOM_OP_SCALAR_NEGI: Integer negation.
// %result = scalar.negi %input : i32
LOOM_DEFINE_ISA(loom_scalar_negi_isa, LOOM_OP_SCALAR_NEGI)
LOOM_DEFINE_OPERAND(loom_scalar_negi_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_negi_result, 0)
iree_status_t loom_scalar_negi_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_negi_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_negi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ABSI: Integer absolute value.
// %result = scalar.absi %input : i32
LOOM_DEFINE_ISA(loom_scalar_absi_isa, LOOM_OP_SCALAR_ABSI)
LOOM_DEFINE_OPERAND(loom_scalar_absi_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_absi_result, 0)
iree_status_t loom_scalar_absi_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_absi_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_absi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_MINSI: Signed integer minimum.
// %result = scalar.minsi %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_minsi_isa, LOOM_OP_SCALAR_MINSI)
LOOM_DEFINE_OPERAND(loom_scalar_minsi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_minsi_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_minsi_result, 0)
iree_status_t loom_scalar_minsi_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_minsi_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_minsi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_MAXSI: Signed integer maximum.
// %result = scalar.maxsi %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_maxsi_isa, LOOM_OP_SCALAR_MAXSI)
LOOM_DEFINE_OPERAND(loom_scalar_maxsi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_maxsi_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_maxsi_result, 0)
iree_status_t loom_scalar_maxsi_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_maxsi_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_maxsi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_MINUI: Unsigned integer minimum.
// %result = scalar.minui %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_minui_isa, LOOM_OP_SCALAR_MINUI)
LOOM_DEFINE_OPERAND(loom_scalar_minui_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_minui_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_minui_result, 0)
iree_status_t loom_scalar_minui_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_minui_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_minui_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_MAXUI: Unsigned integer maximum.
// %result = scalar.maxui %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_maxui_isa, LOOM_OP_SCALAR_MAXUI)
LOOM_DEFINE_OPERAND(loom_scalar_maxui_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_maxui_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_maxui_result, 0)
iree_status_t loom_scalar_maxui_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_maxui_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_maxui_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_FMAI: Fused integer multiply-add: a*b + c with no intermediate overflow check.
// %result = scalar.fmai %a, %b, %c : i64
LOOM_DEFINE_ISA(loom_scalar_fmai_isa, LOOM_OP_SCALAR_FMAI)
LOOM_DEFINE_OPERAND(loom_scalar_fmai_a, 0)
LOOM_DEFINE_OPERAND(loom_scalar_fmai_b, 1)
LOOM_DEFINE_OPERAND(loom_scalar_fmai_c, 2)
LOOM_DEFINE_RESULT(loom_scalar_fmai_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_fmai_overflow)
iree_status_t loom_scalar_fmai_build(
    loom_builder_t* builder,
    uint8_t instance_flags,
    loom_value_id_t a,
    loom_value_id_t b,
    loom_value_id_t c,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_fmai_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_fmai_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ADDF: Floating-point addition.
// %result = scalar.addf %lhs, %rhs : f32
LOOM_DEFINE_ISA(loom_scalar_addf_isa, LOOM_OP_SCALAR_ADDF)
LOOM_DEFINE_OPERAND(loom_scalar_addf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_addf_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_addf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_addf_fastmath)
iree_status_t loom_scalar_addf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_addf_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_addf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_SUBF: Floating-point subtraction.
// %result = scalar.subf %lhs, %rhs : f32
LOOM_DEFINE_ISA(loom_scalar_subf_isa, LOOM_OP_SCALAR_SUBF)
LOOM_DEFINE_OPERAND(loom_scalar_subf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_subf_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_subf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_subf_fastmath)
iree_status_t loom_scalar_subf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_subf_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_subf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_MULF: Floating-point multiplication.
// %result = scalar.mulf %lhs, %rhs : f32
LOOM_DEFINE_ISA(loom_scalar_mulf_isa, LOOM_OP_SCALAR_MULF)
LOOM_DEFINE_OPERAND(loom_scalar_mulf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_mulf_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_mulf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_mulf_fastmath)
iree_status_t loom_scalar_mulf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_mulf_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_mulf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_DIVF: Floating-point division.
// %result = scalar.divf %lhs, %rhs : f32
LOOM_DEFINE_ISA(loom_scalar_divf_isa, LOOM_OP_SCALAR_DIVF)
LOOM_DEFINE_OPERAND(loom_scalar_divf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_divf_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_divf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_divf_fastmath)
iree_status_t loom_scalar_divf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_divf_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_divf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_REMF: Floating-point remainder (C fmod semantics).
// %result = scalar.remf %lhs, %rhs : f32
LOOM_DEFINE_ISA(loom_scalar_remf_isa, LOOM_OP_SCALAR_REMF)
LOOM_DEFINE_OPERAND(loom_scalar_remf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_remf_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_remf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_remf_fastmath)
iree_status_t loom_scalar_remf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_remf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_NEGF: Floating-point negation.
// %result = scalar.negf %input : f32
LOOM_DEFINE_ISA(loom_scalar_negf_isa, LOOM_OP_SCALAR_NEGF)
LOOM_DEFINE_OPERAND(loom_scalar_negf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_negf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_negf_fastmath)
iree_status_t loom_scalar_negf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_negf_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_negf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ABSF: Floating-point absolute value.
// %result = scalar.absf %input : f32
LOOM_DEFINE_ISA(loom_scalar_absf_isa, LOOM_OP_SCALAR_ABSF)
LOOM_DEFINE_OPERAND(loom_scalar_absf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_absf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_absf_fastmath)
iree_status_t loom_scalar_absf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_absf_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_absf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_MINIMUMF: IEEE 754 minimum (NaN propagates).
// %result = scalar.minimumf %lhs, %rhs : f32
LOOM_DEFINE_ISA(loom_scalar_minimumf_isa, LOOM_OP_SCALAR_MINIMUMF)
LOOM_DEFINE_OPERAND(loom_scalar_minimumf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_minimumf_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_minimumf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_minimumf_fastmath)
iree_status_t loom_scalar_minimumf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_minimumf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_MAXIMUMF: IEEE 754 maximum (NaN propagates).
// %result = scalar.maximumf %lhs, %rhs : f32
LOOM_DEFINE_ISA(loom_scalar_maximumf_isa, LOOM_OP_SCALAR_MAXIMUMF)
LOOM_DEFINE_OPERAND(loom_scalar_maximumf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_maximumf_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_maximumf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_maximumf_fastmath)
iree_status_t loom_scalar_maximumf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_maximumf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_MINNUMF: C99 fmin (NaN ignored, returns the non-NaN operand).
// %result = scalar.minnumf %lhs, %rhs : f32
LOOM_DEFINE_ISA(loom_scalar_minnumf_isa, LOOM_OP_SCALAR_MINNUMF)
LOOM_DEFINE_OPERAND(loom_scalar_minnumf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_minnumf_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_minnumf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_minnumf_fastmath)
iree_status_t loom_scalar_minnumf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_minnumf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_MAXNUMF: C99 fmax (NaN ignored, returns the non-NaN operand).
// %result = scalar.maxnumf %lhs, %rhs : f32
LOOM_DEFINE_ISA(loom_scalar_maxnumf_isa, LOOM_OP_SCALAR_MAXNUMF)
LOOM_DEFINE_OPERAND(loom_scalar_maxnumf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_maxnumf_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_maxnumf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_maxnumf_fastmath)
iree_status_t loom_scalar_maxnumf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_maxnumf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_CLAMPF: Floating-point clamp with explicit NaN/comparison policy. The ordered mode preserves strict compare/select semantics, number mode uses minnum/maxnum semantics, and ieee mode propagates NaNs.
// %result = scalar.clampf<ordered> %value, %lower, %upper : f32
LOOM_DEFINE_ISA(loom_scalar_clampf_isa, LOOM_OP_SCALAR_CLAMPF)
LOOM_DEFINE_OPERAND(loom_scalar_clampf_value, 0)
LOOM_DEFINE_OPERAND(loom_scalar_clampf_lower, 1)
LOOM_DEFINE_OPERAND(loom_scalar_clampf_upper, 2)
LOOM_DEFINE_RESULT(loom_scalar_clampf_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_scalar_clampf_mode, 0, loom_scalar_clampf_mode_t)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_clampf_fastmath)
iree_status_t loom_scalar_clampf_build(
    loom_builder_t* builder,
    loom_scalar_clampf_mode_t mode,
    uint8_t instance_flags,
    loom_value_id_t value,
    loom_value_id_t lower,
    loom_value_id_t upper,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_clampf_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_clampf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_COPYSIGNF: Copy sign of rhs onto magnitude of lhs.
// %result = scalar.copysignf %lhs, %rhs : f32
LOOM_DEFINE_ISA(loom_scalar_copysignf_isa, LOOM_OP_SCALAR_COPYSIGNF)
LOOM_DEFINE_OPERAND(loom_scalar_copysignf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_copysignf_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_copysignf_result, 0)
iree_status_t loom_scalar_copysignf_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_copysignf_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_copysignf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_EXPF: Exponential: e^x.
// %result = scalar.expf %input : f32
LOOM_DEFINE_ISA(loom_scalar_expf_isa, LOOM_OP_SCALAR_EXPF)
LOOM_DEFINE_OPERAND(loom_scalar_expf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_expf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_expf_fastmath)
iree_status_t loom_scalar_expf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_expf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_EXP2F: Base-2 exponential: 2^x.
// %result = scalar.exp2f %input : f32
LOOM_DEFINE_ISA(loom_scalar_exp2f_isa, LOOM_OP_SCALAR_EXP2F)
LOOM_DEFINE_OPERAND(loom_scalar_exp2f_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_exp2f_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_exp2f_fastmath)
iree_status_t loom_scalar_exp2f_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_exp2f_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_EXPM1F: Exponential minus one: e^x - 1 (numerically stable near 0).
// %result = scalar.expm1f %input : f32
LOOM_DEFINE_ISA(loom_scalar_expm1f_isa, LOOM_OP_SCALAR_EXPM1F)
LOOM_DEFINE_OPERAND(loom_scalar_expm1f_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_expm1f_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_expm1f_fastmath)
iree_status_t loom_scalar_expm1f_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_expm1f_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_LOGF: Natural logarithm: ln(x).
// %result = scalar.logf %input : f32
LOOM_DEFINE_ISA(loom_scalar_logf_isa, LOOM_OP_SCALAR_LOGF)
LOOM_DEFINE_OPERAND(loom_scalar_logf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_logf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_logf_fastmath)
iree_status_t loom_scalar_logf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_logf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_LOG2F: Base-2 logarithm.
// %result = scalar.log2f %input : f32
LOOM_DEFINE_ISA(loom_scalar_log2f_isa, LOOM_OP_SCALAR_LOG2F)
LOOM_DEFINE_OPERAND(loom_scalar_log2f_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_log2f_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_log2f_fastmath)
iree_status_t loom_scalar_log2f_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_log2f_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_LOG10F: Base-10 logarithm.
// %result = scalar.log10f %input : f32
LOOM_DEFINE_ISA(loom_scalar_log10f_isa, LOOM_OP_SCALAR_LOG10F)
LOOM_DEFINE_OPERAND(loom_scalar_log10f_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_log10f_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_log10f_fastmath)
iree_status_t loom_scalar_log10f_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_log10f_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_LOG1PF: Natural logarithm of 1+x: ln(1+x) (numerically stable near 0).
// %result = scalar.log1pf %input : f32
LOOM_DEFINE_ISA(loom_scalar_log1pf_isa, LOOM_OP_SCALAR_LOG1PF)
LOOM_DEFINE_OPERAND(loom_scalar_log1pf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_log1pf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_log1pf_fastmath)
iree_status_t loom_scalar_log1pf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_log1pf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_POWF: Power: x^y.
// %result = scalar.powf %lhs, %rhs : f32
LOOM_DEFINE_ISA(loom_scalar_powf_isa, LOOM_OP_SCALAR_POWF)
LOOM_DEFINE_OPERAND(loom_scalar_powf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_powf_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_powf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_powf_fastmath)
iree_status_t loom_scalar_powf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_powf_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_powf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_SQRTF: Square root.
// %result = scalar.sqrtf %input : f32
LOOM_DEFINE_ISA(loom_scalar_sqrtf_isa, LOOM_OP_SCALAR_SQRTF)
LOOM_DEFINE_OPERAND(loom_scalar_sqrtf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_sqrtf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_sqrtf_fastmath)
iree_status_t loom_scalar_sqrtf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_sqrtf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_RSQRTF: Reciprocal square root: 1/sqrt(x).
// %result = scalar.rsqrtf %input : f32
LOOM_DEFINE_ISA(loom_scalar_rsqrtf_isa, LOOM_OP_SCALAR_RSQRTF)
LOOM_DEFINE_OPERAND(loom_scalar_rsqrtf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_rsqrtf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_rsqrtf_fastmath)
iree_status_t loom_scalar_rsqrtf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_rsqrtf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_CBRTF: Cube root.
// %result = scalar.cbrtf %input : f32
LOOM_DEFINE_ISA(loom_scalar_cbrtf_isa, LOOM_OP_SCALAR_CBRTF)
LOOM_DEFINE_OPERAND(loom_scalar_cbrtf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_cbrtf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_cbrtf_fastmath)
iree_status_t loom_scalar_cbrtf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_cbrtf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_SINF: Sine.
// %result = scalar.sinf %input : f32
LOOM_DEFINE_ISA(loom_scalar_sinf_isa, LOOM_OP_SCALAR_SINF)
LOOM_DEFINE_OPERAND(loom_scalar_sinf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_sinf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_sinf_fastmath)
iree_status_t loom_scalar_sinf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_sinf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_COSF: Cosine.
// %result = scalar.cosf %input : f32
LOOM_DEFINE_ISA(loom_scalar_cosf_isa, LOOM_OP_SCALAR_COSF)
LOOM_DEFINE_OPERAND(loom_scalar_cosf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_cosf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_cosf_fastmath)
iree_status_t loom_scalar_cosf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_cosf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_SINTURNSF: Sine over turns: sin(2*pi*x), where 1.0 is one full revolution.
// %result = scalar.sinturnsf %input : f32
LOOM_DEFINE_ISA(loom_scalar_sinturnsf_isa, LOOM_OP_SCALAR_SINTURNSF)
LOOM_DEFINE_OPERAND(loom_scalar_sinturnsf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_sinturnsf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_sinturnsf_fastmath)
iree_status_t loom_scalar_sinturnsf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_sinturnsf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_COSTURNSF: Cosine over turns: cos(2*pi*x), where 1.0 is one full revolution.
// %result = scalar.costurnsf %input : f32
LOOM_DEFINE_ISA(loom_scalar_costurnsf_isa, LOOM_OP_SCALAR_COSTURNSF)
LOOM_DEFINE_OPERAND(loom_scalar_costurnsf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_costurnsf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_costurnsf_fastmath)
iree_status_t loom_scalar_costurnsf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_costurnsf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_TANF: Tangent.
// %result = scalar.tanf %input : f32
LOOM_DEFINE_ISA(loom_scalar_tanf_isa, LOOM_OP_SCALAR_TANF)
LOOM_DEFINE_OPERAND(loom_scalar_tanf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_tanf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_tanf_fastmath)
iree_status_t loom_scalar_tanf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_tanf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ASINF: Arcsine.
// %result = scalar.asinf %input : f32
LOOM_DEFINE_ISA(loom_scalar_asinf_isa, LOOM_OP_SCALAR_ASINF)
LOOM_DEFINE_OPERAND(loom_scalar_asinf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_asinf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_asinf_fastmath)
iree_status_t loom_scalar_asinf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_asinf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ACOSF: Arccosine.
// %result = scalar.acosf %input : f32
LOOM_DEFINE_ISA(loom_scalar_acosf_isa, LOOM_OP_SCALAR_ACOSF)
LOOM_DEFINE_OPERAND(loom_scalar_acosf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_acosf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_acosf_fastmath)
iree_status_t loom_scalar_acosf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_acosf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ATANF: Arctangent.
// %result = scalar.atanf %input : f32
LOOM_DEFINE_ISA(loom_scalar_atanf_isa, LOOM_OP_SCALAR_ATANF)
LOOM_DEFINE_OPERAND(loom_scalar_atanf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_atanf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_atanf_fastmath)
iree_status_t loom_scalar_atanf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_atanf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ATAN2F: Two-argument arctangent: atan2(y, x).
// %result = scalar.atan2f %lhs, %rhs : f32
LOOM_DEFINE_ISA(loom_scalar_atan2f_isa, LOOM_OP_SCALAR_ATAN2F)
LOOM_DEFINE_OPERAND(loom_scalar_atan2f_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_atan2f_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_atan2f_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_atan2f_fastmath)
iree_status_t loom_scalar_atan2f_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_atan2f_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_SINHF: Hyperbolic sine.
// %result = scalar.sinhf %input : f32
LOOM_DEFINE_ISA(loom_scalar_sinhf_isa, LOOM_OP_SCALAR_SINHF)
LOOM_DEFINE_OPERAND(loom_scalar_sinhf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_sinhf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_sinhf_fastmath)
iree_status_t loom_scalar_sinhf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_sinhf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_COSHF: Hyperbolic cosine.
// %result = scalar.coshf %input : f32
LOOM_DEFINE_ISA(loom_scalar_coshf_isa, LOOM_OP_SCALAR_COSHF)
LOOM_DEFINE_OPERAND(loom_scalar_coshf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_coshf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_coshf_fastmath)
iree_status_t loom_scalar_coshf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_coshf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_TANHF: Hyperbolic tangent.
// %result = scalar.tanhf %input : f32
LOOM_DEFINE_ISA(loom_scalar_tanhf_isa, LOOM_OP_SCALAR_TANHF)
LOOM_DEFINE_OPERAND(loom_scalar_tanhf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_tanhf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_tanhf_fastmath)
iree_status_t loom_scalar_tanhf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_tanhf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ASINHF: Inverse hyperbolic sine.
// %result = scalar.asinhf %input : f32
LOOM_DEFINE_ISA(loom_scalar_asinhf_isa, LOOM_OP_SCALAR_ASINHF)
LOOM_DEFINE_OPERAND(loom_scalar_asinhf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_asinhf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_asinhf_fastmath)
iree_status_t loom_scalar_asinhf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_asinhf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ACOSHF: Inverse hyperbolic cosine.
// %result = scalar.acoshf %input : f32
LOOM_DEFINE_ISA(loom_scalar_acoshf_isa, LOOM_OP_SCALAR_ACOSHF)
LOOM_DEFINE_OPERAND(loom_scalar_acoshf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_acoshf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_acoshf_fastmath)
iree_status_t loom_scalar_acoshf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_acoshf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ATANHF: Inverse hyperbolic tangent.
// %result = scalar.atanhf %input : f32
LOOM_DEFINE_ISA(loom_scalar_atanhf_isa, LOOM_OP_SCALAR_ATANHF)
LOOM_DEFINE_OPERAND(loom_scalar_atanhf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_atanhf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_atanhf_fastmath)
iree_status_t loom_scalar_atanhf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_atanhf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ERFF: Error function (used in GeLU activation).
// %result = scalar.erff %input : f32
LOOM_DEFINE_ISA(loom_scalar_erff_isa, LOOM_OP_SCALAR_ERFF)
LOOM_DEFINE_OPERAND(loom_scalar_erff_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_erff_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_erff_fastmath)
iree_status_t loom_scalar_erff_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_erff_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ERFCF: Complementary error function: 1 - erf(x).
// %result = scalar.erfcf %input : f32
LOOM_DEFINE_ISA(loom_scalar_erfcf_isa, LOOM_OP_SCALAR_ERFCF)
LOOM_DEFINE_OPERAND(loom_scalar_erfcf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_erfcf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_erfcf_fastmath)
iree_status_t loom_scalar_erfcf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_erfcf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_LOGISTICF: Logistic sigmoid: 1 / (1 + exp(-x)).
// %result = scalar.logisticf %input : f32
LOOM_DEFINE_ISA(loom_scalar_logisticf_isa, LOOM_OP_SCALAR_LOGISTICF)
LOOM_DEFINE_OPERAND(loom_scalar_logisticf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_logisticf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_logisticf_fastmath)
iree_status_t loom_scalar_logisticf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_logisticf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_SILUF: SiLU activation: x * logistic(x).
// %result = scalar.siluf %input : f32
LOOM_DEFINE_ISA(loom_scalar_siluf_isa, LOOM_OP_SCALAR_SILUF)
LOOM_DEFINE_OPERAND(loom_scalar_siluf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_siluf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_siluf_fastmath)
iree_status_t loom_scalar_siluf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_siluf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_SOFTPLUSF: Softplus activation: log(1 + exp(x)).
// %result = scalar.softplusf %input : f32
LOOM_DEFINE_ISA(loom_scalar_softplusf_isa, LOOM_OP_SCALAR_SOFTPLUSF)
LOOM_DEFINE_OPERAND(loom_scalar_softplusf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_softplusf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_softplusf_fastmath)
iree_status_t loom_scalar_softplusf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_softplusf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_GELUF: GELU activation preserving the chosen formula family. The logistic variant carries its scale as an explicit attribute so importers do not encode approximation identity through arithmetic constants.
// %result = scalar.geluf<erf> %input : f32
LOOM_DEFINE_ISA(loom_scalar_geluf_isa, LOOM_OP_SCALAR_GELUF)
LOOM_DEFINE_OPERAND(loom_scalar_geluf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_geluf_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_scalar_geluf_variant, 0, loom_scalar_geluf_variant_t)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_geluf_fastmath)
LOOM_DEFINE_ATTR_F64(loom_scalar_geluf_scale, 1)
enum loom_scalar_geluf_build_flag_bits_e {
  LOOM_SCALAR_GELUF_BUILD_FLAG_HAS_SCALE = 1u << 0,
};
typedef uint32_t loom_scalar_geluf_build_flags_t;
iree_status_t loom_scalar_geluf_build(
    loom_builder_t* builder,
    loom_scalar_geluf_build_flags_t build_flags,
    loom_scalar_geluf_variant_t variant,
    uint8_t instance_flags,
    loom_value_id_t input,
    loom_optional double scale,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_geluf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_scalar_geluf_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_SCALAR_FMAF: Fused multiply-add: a*b + c with single rounding.
// %result = scalar.fmaf %a, %b, %c : f32
LOOM_DEFINE_ISA(loom_scalar_fmaf_isa, LOOM_OP_SCALAR_FMAF)
LOOM_DEFINE_OPERAND(loom_scalar_fmaf_a, 0)
LOOM_DEFINE_OPERAND(loom_scalar_fmaf_b, 1)
LOOM_DEFINE_OPERAND(loom_scalar_fmaf_c, 2)
LOOM_DEFINE_RESULT(loom_scalar_fmaf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_fmaf_fastmath)
iree_status_t loom_scalar_fmaf_build(
    loom_builder_t* builder,
    uint8_t instance_flags,
    loom_value_id_t a,
    loom_value_id_t b,
    loom_value_id_t c,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_fmaf_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_fmaf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_CEILF: Round toward positive infinity.
// %result = scalar.ceilf %input : f32
LOOM_DEFINE_ISA(loom_scalar_ceilf_isa, LOOM_OP_SCALAR_CEILF)
LOOM_DEFINE_OPERAND(loom_scalar_ceilf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_ceilf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_ceilf_fastmath)
iree_status_t loom_scalar_ceilf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_ceilf_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_ceilf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_FLOORF: Round toward negative infinity.
// %result = scalar.floorf %input : f32
LOOM_DEFINE_ISA(loom_scalar_floorf_isa, LOOM_OP_SCALAR_FLOORF)
LOOM_DEFINE_OPERAND(loom_scalar_floorf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_floorf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_floorf_fastmath)
iree_status_t loom_scalar_floorf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_floorf_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_floorf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ROUNDF: Round to nearest, ties away from zero.
// %result = scalar.roundf %input : f32
LOOM_DEFINE_ISA(loom_scalar_roundf_isa, LOOM_OP_SCALAR_ROUNDF)
LOOM_DEFINE_OPERAND(loom_scalar_roundf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_roundf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_roundf_fastmath)
iree_status_t loom_scalar_roundf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_roundf_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_roundf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ROUNDEVENF: Round to nearest, ties to even (IEEE 754 default rounding).
// %result = scalar.roundevenf %input : f32
LOOM_DEFINE_ISA(loom_scalar_roundevenf_isa, LOOM_OP_SCALAR_ROUNDEVENF)
LOOM_DEFINE_OPERAND(loom_scalar_roundevenf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_roundevenf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_roundevenf_fastmath)
iree_status_t loom_scalar_roundevenf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_roundevenf_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_roundevenf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_TRUNCF: Round toward zero (C trunc).
// %result = scalar.truncf %input : f32
LOOM_DEFINE_ISA(loom_scalar_truncf_isa, LOOM_OP_SCALAR_TRUNCF)
LOOM_DEFINE_OPERAND(loom_scalar_truncf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_truncf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_truncf_fastmath)
iree_status_t loom_scalar_truncf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_truncf_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_truncf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_CMPI: Integer comparison.
// %result = scalar.cmpi eq, %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_cmpi_isa, LOOM_OP_SCALAR_CMPI)
LOOM_DEFINE_OPERAND(loom_scalar_cmpi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_cmpi_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_cmpi_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_scalar_cmpi_predicate, 0, loom_scalar_cmpi_predicate_t)
iree_status_t loom_scalar_cmpi_build(
    loom_builder_t* builder, uint8_t predicate,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t operand_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_cmpi_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_cmpi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_CMPF: Floating-point comparison.
// %result = scalar.cmpf olt, %lhs, %rhs : f32
LOOM_DEFINE_ISA(loom_scalar_cmpf_isa, LOOM_OP_SCALAR_CMPF)
LOOM_DEFINE_OPERAND(loom_scalar_cmpf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_cmpf_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_cmpf_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_scalar_cmpf_predicate, 0, loom_scalar_cmpf_predicate_t)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_cmpf_fastmath)
iree_status_t loom_scalar_cmpf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    uint8_t predicate, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t operand_type,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_cmpf_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_cmpf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ISNANF: Returns true (i1) if the operand is NaN.
// %result = scalar.isnanf %input : f32
LOOM_DEFINE_ISA(loom_scalar_isnanf_isa, LOOM_OP_SCALAR_ISNANF)
LOOM_DEFINE_OPERAND(loom_scalar_isnanf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_isnanf_result, 0)
iree_status_t loom_scalar_isnanf_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_isnanf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ISINFF: Returns true (i1) if the operand is positive or negative infinity.
// %result = scalar.isinff %input : f32
LOOM_DEFINE_ISA(loom_scalar_isinff_isa, LOOM_OP_SCALAR_ISINFF)
LOOM_DEFINE_OPERAND(loom_scalar_isinff_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_isinff_result, 0)
iree_status_t loom_scalar_isinff_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_isinff_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ISFINITEF: Returns true (i1) if the operand is finite (not NaN and not infinity).
// %result = scalar.isfinitef %input : f32
LOOM_DEFINE_ISA(loom_scalar_isfinitef_isa, LOOM_OP_SCALAR_ISFINITEF)
LOOM_DEFINE_OPERAND(loom_scalar_isfinitef_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_isfinitef_result, 0)
iree_status_t loom_scalar_isfinitef_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_isfinitef_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_SIGNF: Floating-point sign: returns -1.0, 0.0, or 1.0.
// %result = scalar.signf %input : f32
LOOM_DEFINE_ISA(loom_scalar_signf_isa, LOOM_OP_SCALAR_SIGNF)
LOOM_DEFINE_OPERAND(loom_scalar_signf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_signf_result, 0)
iree_status_t loom_scalar_signf_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_signf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_SIGNI: Integer sign: returns -1, 0, or 1.
// %result = scalar.signi %input : i32
LOOM_DEFINE_ISA(loom_scalar_signi_isa, LOOM_OP_SCALAR_SIGNI)
LOOM_DEFINE_OPERAND(loom_scalar_signi_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_signi_result, 0)
iree_status_t loom_scalar_signi_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_signi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_SITOFP: Signed integer to floating-point.
// %result = scalar.sitofp %input : i32 to f32
LOOM_DEFINE_ISA(loom_scalar_sitofp_isa, LOOM_OP_SCALAR_SITOFP)
LOOM_DEFINE_OPERAND(loom_scalar_sitofp_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_sitofp_result, 0)
iree_status_t loom_scalar_sitofp_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_sitofp_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_UITOFP: Unsigned integer to floating-point.
// %result = scalar.uitofp %input : i32 to f32
LOOM_DEFINE_ISA(loom_scalar_uitofp_isa, LOOM_OP_SCALAR_UITOFP)
LOOM_DEFINE_OPERAND(loom_scalar_uitofp_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_uitofp_result, 0)
iree_status_t loom_scalar_uitofp_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_uitofp_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_FPTOSI: Floating-point to signed integer (rounds toward zero).
// %result = scalar.fptosi %input : f32 to i32
LOOM_DEFINE_ISA(loom_scalar_fptosi_isa, LOOM_OP_SCALAR_FPTOSI)
LOOM_DEFINE_OPERAND(loom_scalar_fptosi_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_fptosi_result, 0)
iree_status_t loom_scalar_fptosi_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_fptosi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_FPTOUI: Floating-point to unsigned integer (rounds toward zero).
// %result = scalar.fptoui %input : f32 to i32
LOOM_DEFINE_ISA(loom_scalar_fptoui_isa, LOOM_OP_SCALAR_FPTOUI)
LOOM_DEFINE_OPERAND(loom_scalar_fptoui_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_fptoui_result, 0)
iree_status_t loom_scalar_fptoui_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_fptoui_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_EXTF: Float precision extension (widen): e.g. f16 to f32.
// %result = scalar.extf %input : f16 to f32
LOOM_DEFINE_ISA(loom_scalar_extf_isa, LOOM_OP_SCALAR_EXTF)
LOOM_DEFINE_OPERAND(loom_scalar_extf_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_extf_result, 0)
iree_status_t loom_scalar_extf_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_extf_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_extf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_FPTRUNC: Float precision truncation (narrow): e.g. f32 to f16.
// %result = scalar.fptrunc %input : f32 to f16
LOOM_DEFINE_ISA(loom_scalar_fptrunc_isa, LOOM_OP_SCALAR_FPTRUNC)
LOOM_DEFINE_OPERAND(loom_scalar_fptrunc_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_fptrunc_result, 0)
iree_status_t loom_scalar_fptrunc_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_fptrunc_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_fptrunc_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_EXTSI: Signed integer extension (sign-extend): e.g. i8 to i32.
// %result = scalar.extsi %input : i8 to i32
LOOM_DEFINE_ISA(loom_scalar_extsi_isa, LOOM_OP_SCALAR_EXTSI)
LOOM_DEFINE_OPERAND(loom_scalar_extsi_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_extsi_result, 0)
iree_status_t loom_scalar_extsi_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_extsi_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_extsi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_EXTUI: Unsigned integer extension (zero-extend): e.g. i8 to i32.
// %result = scalar.extui %input : i8 to i32
LOOM_DEFINE_ISA(loom_scalar_extui_isa, LOOM_OP_SCALAR_EXTUI)
LOOM_DEFINE_OPERAND(loom_scalar_extui_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_extui_result, 0)
iree_status_t loom_scalar_extui_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_extui_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_extui_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_TRUNCI: Integer truncation (narrow): e.g. i32 to i8.
// %result = scalar.trunci %input : i32 to i8
LOOM_DEFINE_ISA(loom_scalar_trunci_isa, LOOM_OP_SCALAR_TRUNCI)
LOOM_DEFINE_OPERAND(loom_scalar_trunci_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_trunci_result, 0)
iree_status_t loom_scalar_trunci_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_trunci_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_trunci_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_BITCAST: Bitwise reinterpretation: same bits, different type. No conversion.
// %result = scalar.bitcast %input : f32 to i32
LOOM_DEFINE_ISA(loom_scalar_bitcast_isa, LOOM_OP_SCALAR_BITCAST)
LOOM_DEFINE_OPERAND(loom_scalar_bitcast_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_bitcast_result, 0)
iree_status_t loom_scalar_bitcast_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_bitcast_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_bitcast_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_CONSTANT: Materialize a compile-time integer or floating-point scalar value. Logical coordinate and byte-offset constants use index.constant.
// %c42 = scalar.constant 42 : i32
LOOM_DEFINE_ISA(loom_scalar_constant_isa, LOOM_OP_SCALAR_CONSTANT)
LOOM_DEFINE_RESULT(loom_scalar_constant_result, 0)
LOOM_DEFINE_ATTR_ANY(loom_scalar_constant_value, 0)
iree_status_t loom_scalar_constant_build(
    loom_builder_t* builder,
    loom_attribute_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_constant_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_scalar_constant_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_SCALAR_POISON: Materialize a typed Loom poison scalar. Poison represents an invalid scalar observation, such as extracting a lane proven not to exist. Pure scalar ops with any poison operand canonicalize to poison of the corresponding result type. Poison is not an LLVM poison value: it must be removed by dead-code elimination or diagnosed before it reaches a store, return, kernel boundary, or target-lowering boundary.
// %p = scalar.poison : f32
LOOM_DEFINE_ISA(loom_scalar_poison_isa, LOOM_OP_SCALAR_POISON)
LOOM_DEFINE_RESULT(loom_scalar_poison_result, 0)
iree_status_t loom_scalar_poison_build(
    loom_builder_t* builder,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);

// LOOM_OP_SCALAR_ANDI: Bitwise AND.
// %result = scalar.andi %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_andi_isa, LOOM_OP_SCALAR_ANDI)
LOOM_DEFINE_OPERAND(loom_scalar_andi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_andi_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_andi_result, 0)
iree_status_t loom_scalar_andi_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_andi_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_andi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ORI: Bitwise OR.
// %result = scalar.ori %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_ori_isa, LOOM_OP_SCALAR_ORI)
LOOM_DEFINE_OPERAND(loom_scalar_ori_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_ori_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_ori_result, 0)
iree_status_t loom_scalar_ori_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_ori_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_ori_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_XORI: Bitwise XOR.
// %result = scalar.xori %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_xori_isa, LOOM_OP_SCALAR_XORI)
LOOM_DEFINE_OPERAND(loom_scalar_xori_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_xori_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_xori_result, 0)
iree_status_t loom_scalar_xori_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_xori_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_xori_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_SHLI: Left shift.
// %result = scalar.shli %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_shli_isa, LOOM_OP_SCALAR_SHLI)
LOOM_DEFINE_OPERAND(loom_scalar_shli_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_shli_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_shli_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_scalar_shli_overflow)
iree_status_t loom_scalar_shli_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_shli_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_shli_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_SHRSI: Arithmetic right shift (sign-extending).
// %result = scalar.shrsi %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_shrsi_isa, LOOM_OP_SCALAR_SHRSI)
LOOM_DEFINE_OPERAND(loom_scalar_shrsi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_shrsi_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_shrsi_result, 0)
iree_status_t loom_scalar_shrsi_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_shrsi_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_shrsi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_SHRUI: Logical right shift (zero-extending).
// %result = scalar.shrui %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_shrui_isa, LOOM_OP_SCALAR_SHRUI)
LOOM_DEFINE_OPERAND(loom_scalar_shrui_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_shrui_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_shrui_result, 0)
iree_status_t loom_scalar_shrui_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_shrui_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_shrui_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ROTLI: Left rotate.
// %result = scalar.rotli %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_rotli_isa, LOOM_OP_SCALAR_ROTLI)
LOOM_DEFINE_OPERAND(loom_scalar_rotli_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_rotli_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_rotli_result, 0)
iree_status_t loom_scalar_rotli_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_rotli_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_rotli_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ROTRI: Right rotate.
// %result = scalar.rotri %lhs, %rhs : i32
LOOM_DEFINE_ISA(loom_scalar_rotri_isa, LOOM_OP_SCALAR_ROTRI)
LOOM_DEFINE_OPERAND(loom_scalar_rotri_lhs, 0)
LOOM_DEFINE_OPERAND(loom_scalar_rotri_rhs, 1)
LOOM_DEFINE_RESULT(loom_scalar_rotri_result, 0)
iree_status_t loom_scalar_rotri_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_scalar_rotri_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_scalar_rotri_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_CTLZI: Count leading zeros.
// %result = scalar.ctlzi %input : i32
LOOM_DEFINE_ISA(loom_scalar_ctlzi_isa, LOOM_OP_SCALAR_CTLZI)
LOOM_DEFINE_OPERAND(loom_scalar_ctlzi_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_ctlzi_result, 0)
iree_status_t loom_scalar_ctlzi_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_ctlzi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_CTTZI: Count trailing zeros.
// %result = scalar.cttzi %input : i32
LOOM_DEFINE_ISA(loom_scalar_cttzi_isa, LOOM_OP_SCALAR_CTTZI)
LOOM_DEFINE_OPERAND(loom_scalar_cttzi_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_cttzi_result, 0)
iree_status_t loom_scalar_cttzi_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_cttzi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_CTPOPI: Population count (number of set bits).
// %result = scalar.ctpopi %input : i32
LOOM_DEFINE_ISA(loom_scalar_ctpopi_isa, LOOM_OP_SCALAR_CTPOPI)
LOOM_DEFINE_OPERAND(loom_scalar_ctpopi_input, 0)
LOOM_DEFINE_RESULT(loom_scalar_ctpopi_result, 0)
iree_status_t loom_scalar_ctpopi_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_ctpopi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_BITFIELD_EXTRACTU: Extract one fixed integer bitfield and zero-extend it into the result.
// %byte = scalar.bitfield.extractu %word {offset = 8, width = 8} : i32 -> i32
LOOM_DEFINE_ISA(loom_scalar_bitfield_extractu_isa, LOOM_OP_SCALAR_BITFIELD_EXTRACTU)
LOOM_DEFINE_OPERAND(loom_scalar_bitfield_extractu_source, 0)
LOOM_DEFINE_RESULT(loom_scalar_bitfield_extractu_result, 0)
LOOM_DEFINE_ATTR_I64(loom_scalar_bitfield_extractu_offset, 0)
LOOM_DEFINE_ATTR_I64(loom_scalar_bitfield_extractu_width, 1)
iree_status_t loom_scalar_bitfield_extractu_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    int64_t offset,
    int64_t width,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_bitfield_extractu_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_BITFIELD_EXTRACTS: Extract one fixed integer bitfield and sign-extend it into the result.
// %signed_byte = scalar.bitfield.extracts %word {offset = 24, width = 8} : i32 -> i32
LOOM_DEFINE_ISA(loom_scalar_bitfield_extracts_isa, LOOM_OP_SCALAR_BITFIELD_EXTRACTS)
LOOM_DEFINE_OPERAND(loom_scalar_bitfield_extracts_source, 0)
LOOM_DEFINE_RESULT(loom_scalar_bitfield_extracts_result, 0)
LOOM_DEFINE_ATTR_I64(loom_scalar_bitfield_extracts_offset, 0)
LOOM_DEFINE_ATTR_I64(loom_scalar_bitfield_extracts_width, 1)
iree_status_t loom_scalar_bitfield_extracts_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    int64_t offset,
    int64_t width,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_bitfield_extracts_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_SCALAR_ASSUME: Identity with predicate constraints on integer payload results. Use index.assume for index or offset values.
// %n2 = scalar.assume %n [mul(%n, 16)] : i64
LOOM_DEFINE_ISA(loom_scalar_assume_isa, LOOM_OP_SCALAR_ASSUME)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_scalar_assume_values, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_scalar_assume_results, 0)
iree_status_t loom_scalar_assume_build(
    loom_builder_t* builder,
    const loom_value_id_t* values,
    iree_host_size_t values_count,
    const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_scalar_assume_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_scalar_assume_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the scalar dialect.
const loom_op_vtable_t* const* loom_scalar_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the scalar dialect.
const loom_op_semantics_t* loom_scalar_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a scalar op kind, or empty metadata.
loom_op_semantics_t loom_scalar_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_SCALAR_OPS_H_
