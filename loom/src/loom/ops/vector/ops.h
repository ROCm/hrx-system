// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Generator: loom.gen.ops.c_tables.
// Regenerate: python3 loom/py/loom/gen/run.py c_tables --in-place
// clang-format off

#ifndef LOOM_OPS_VECTOR_OPS_H_
#define LOOM_OPS_VECTOR_OPS_H_

#include "loom/ops/op_defs.h"
#include "loom/ops/atomic.h"
#include "loom/ops/cache.h"
#include "loom/ops/combining.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  LOOM_OP_VECTOR_CONSTANT = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 0),
  LOOM_OP_VECTOR_POISON = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 1),
  LOOM_OP_VECTOR_EMPTY = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 2),
  LOOM_OP_VECTOR_SPLAT = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 3),
  LOOM_OP_VECTOR_IOTA = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 4),
  LOOM_OP_VECTOR_MASK_RANGE = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 5),
  LOOM_OP_VECTOR_BROADCAST = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 6),
  LOOM_OP_VECTOR_FROM_ELEMENTS = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 7),
  LOOM_OP_VECTOR_EXTRACT = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 8),
  LOOM_OP_VECTOR_INSERT = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 9),
  LOOM_OP_VECTOR_SLICE = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 10),
  LOOM_OP_VECTOR_CONCAT = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 11),
  LOOM_OP_VECTOR_TRANSPOSE = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 12),
  LOOM_OP_VECTOR_SHUFFLE = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 13),
  LOOM_OP_VECTOR_INTERLEAVE = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 14),
  LOOM_OP_VECTOR_DEINTERLEAVE = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 15),
  LOOM_OP_VECTOR_TABLE_LOOKUP = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 16),
  LOOM_OP_VECTOR_TABLE_QUANTIZE = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 17),
  LOOM_OP_VECTOR_TRANSFORM = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 18),
  LOOM_OP_VECTOR_FRAGMENT_LOAD = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 19),
  LOOM_OP_VECTOR_FRAGMENT_STORE = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 20),
  LOOM_OP_VECTOR_LOAD = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 21),
  LOOM_OP_VECTOR_STORE = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 22),
  LOOM_OP_VECTOR_LOAD_MASK = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 23),
  LOOM_OP_VECTOR_STORE_MASK = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 24),
  LOOM_OP_VECTOR_LOAD_EXPAND = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 25),
  LOOM_OP_VECTOR_STORE_COMPRESS = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 26),
  LOOM_OP_VECTOR_GATHER = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 27),
  LOOM_OP_VECTOR_SCATTER = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 28),
  LOOM_OP_VECTOR_GATHER_MASK = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 29),
  LOOM_OP_VECTOR_SCATTER_MASK = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 30),
  LOOM_OP_VECTOR_ATOMIC_REDUCE = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 31),
  LOOM_OP_VECTOR_ATOMIC_REDUCE_MASK = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 32),
  LOOM_OP_VECTOR_ATOMIC_RMW = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 33),
  LOOM_OP_VECTOR_ATOMIC_RMW_MASK = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 34),
  LOOM_OP_VECTOR_ATOMIC_CMPXCHG = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 35),
  LOOM_OP_VECTOR_SELECT = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 36),
  LOOM_OP_VECTOR_CMPI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 37),
  LOOM_OP_VECTOR_CMPF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 38),
  LOOM_OP_VECTOR_ADDF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 39),
  LOOM_OP_VECTOR_SUBF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 40),
  LOOM_OP_VECTOR_MULF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 41),
  LOOM_OP_VECTOR_DIVF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 42),
  LOOM_OP_VECTOR_REMF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 43),
  LOOM_OP_VECTOR_NEGF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 44),
  LOOM_OP_VECTOR_ABSF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 45),
  LOOM_OP_VECTOR_MINIMUMF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 46),
  LOOM_OP_VECTOR_MAXIMUMF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 47),
  LOOM_OP_VECTOR_MINNUMF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 48),
  LOOM_OP_VECTOR_MAXNUMF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 49),
  LOOM_OP_VECTOR_CLAMPF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 50),
  LOOM_OP_VECTOR_COPYSIGNF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 51),
  LOOM_OP_VECTOR_FMAF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 52),
  LOOM_OP_VECTOR_ADDI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 53),
  LOOM_OP_VECTOR_SUBI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 54),
  LOOM_OP_VECTOR_MULI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 55),
  LOOM_OP_VECTOR_DIVSI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 56),
  LOOM_OP_VECTOR_DIVUI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 57),
  LOOM_OP_VECTOR_REMSI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 58),
  LOOM_OP_VECTOR_REMUI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 59),
  LOOM_OP_VECTOR_CEILDIVSI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 60),
  LOOM_OP_VECTOR_CEILDIVUI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 61),
  LOOM_OP_VECTOR_FLOORDIVSI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 62),
  LOOM_OP_VECTOR_NEGI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 63),
  LOOM_OP_VECTOR_ABSI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 64),
  LOOM_OP_VECTOR_MINSI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 65),
  LOOM_OP_VECTOR_MAXSI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 66),
  LOOM_OP_VECTOR_MINUI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 67),
  LOOM_OP_VECTOR_MAXUI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 68),
  LOOM_OP_VECTOR_FMAI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 69),
  LOOM_OP_VECTOR_ANDI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 70),
  LOOM_OP_VECTOR_ORI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 71),
  LOOM_OP_VECTOR_XORI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 72),
  LOOM_OP_VECTOR_SHLI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 73),
  LOOM_OP_VECTOR_SHRSI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 74),
  LOOM_OP_VECTOR_SHRUI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 75),
  LOOM_OP_VECTOR_ROTLI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 76),
  LOOM_OP_VECTOR_ROTRI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 77),
  LOOM_OP_VECTOR_CTLZI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 78),
  LOOM_OP_VECTOR_CTTZI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 79),
  LOOM_OP_VECTOR_CTPOPI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 80),
  LOOM_OP_VECTOR_EXPF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 81),
  LOOM_OP_VECTOR_EXP2F = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 82),
  LOOM_OP_VECTOR_EXPM1F = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 83),
  LOOM_OP_VECTOR_LOGF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 84),
  LOOM_OP_VECTOR_LOG2F = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 85),
  LOOM_OP_VECTOR_LOG10F = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 86),
  LOOM_OP_VECTOR_LOG1PF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 87),
  LOOM_OP_VECTOR_POWF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 88),
  LOOM_OP_VECTOR_SQRTF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 89),
  LOOM_OP_VECTOR_RSQRTF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 90),
  LOOM_OP_VECTOR_CBRTF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 91),
  LOOM_OP_VECTOR_SINF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 92),
  LOOM_OP_VECTOR_COSF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 93),
  LOOM_OP_VECTOR_SINTURNSF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 94),
  LOOM_OP_VECTOR_COSTURNSF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 95),
  LOOM_OP_VECTOR_TANF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 96),
  LOOM_OP_VECTOR_ASINF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 97),
  LOOM_OP_VECTOR_ACOSF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 98),
  LOOM_OP_VECTOR_ATANF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 99),
  LOOM_OP_VECTOR_ATAN2F = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 100),
  LOOM_OP_VECTOR_SINHF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 101),
  LOOM_OP_VECTOR_COSHF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 102),
  LOOM_OP_VECTOR_TANHF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 103),
  LOOM_OP_VECTOR_ASINHF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 104),
  LOOM_OP_VECTOR_ACOSHF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 105),
  LOOM_OP_VECTOR_ATANHF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 106),
  LOOM_OP_VECTOR_ERFF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 107),
  LOOM_OP_VECTOR_ERFCF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 108),
  LOOM_OP_VECTOR_LOGISTICF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 109),
  LOOM_OP_VECTOR_SILUF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 110),
  LOOM_OP_VECTOR_SOFTPLUSF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 111),
  LOOM_OP_VECTOR_GELUF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 112),
  LOOM_OP_VECTOR_CEILF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 113),
  LOOM_OP_VECTOR_FLOORF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 114),
  LOOM_OP_VECTOR_ROUNDF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 115),
  LOOM_OP_VECTOR_ROUNDEVENF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 116),
  LOOM_OP_VECTOR_TRUNCF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 117),
  LOOM_OP_VECTOR_ISNANF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 118),
  LOOM_OP_VECTOR_ISINFF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 119),
  LOOM_OP_VECTOR_ISFINITEF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 120),
  LOOM_OP_VECTOR_SIGNF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 121),
  LOOM_OP_VECTOR_SIGNI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 122),
  LOOM_OP_VECTOR_EXTF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 123),
  LOOM_OP_VECTOR_FPTRUNC = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 124),
  LOOM_OP_VECTOR_EXTSI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 125),
  LOOM_OP_VECTOR_EXTUI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 126),
  LOOM_OP_VECTOR_TRUNCI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 127),
  LOOM_OP_VECTOR_SITOFP = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 128),
  LOOM_OP_VECTOR_UITOFP = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 129),
  LOOM_OP_VECTOR_FPTOSI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 130),
  LOOM_OP_VECTOR_FPTOUI = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 131),
  LOOM_OP_VECTOR_BITCAST = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 132),
  LOOM_OP_VECTOR_BITFIELD_EXTRACTU = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 133),
  LOOM_OP_VECTOR_BITFIELD_EXTRACTS = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 134),
  LOOM_OP_VECTOR_BITFIELD_INSERT = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 135),
  LOOM_OP_VECTOR_BITPACK = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 136),
  LOOM_OP_VECTOR_BITUNPACKU = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 137),
  LOOM_OP_VECTOR_BITUNPACKS = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 138),
  LOOM_OP_VECTOR_DOTF = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 139),
  LOOM_OP_VECTOR_DOT2F = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 140),
  LOOM_OP_VECTOR_DOT4I = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 141),
  LOOM_OP_VECTOR_DOT8I4 = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 142),
  LOOM_OP_VECTOR_DOT4F8 = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 143),
  LOOM_OP_VECTOR_MMA = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 144),
  LOOM_OP_VECTOR_REDUCE = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 145),
  LOOM_OP_VECTOR_REDUCE_AXES = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 146),
  LOOM_OP_VECTOR_DECODE = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 147),
  LOOM_OP_VECTOR_ENCODE = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 148),
  LOOM_OP_VECTOR_FRAGMENT = LOOM_OP_KIND(LOOM_DIALECT_VECTOR, 149),
  LOOM_OP_VECTOR_COUNT_ = 150,
};

// IEEE 754 fast-math relaxation flags for float operations.
#define LOOM_VECTOR_FASTMATHFLAGS_REASSOC ((uint8_t)1)
#define LOOM_VECTOR_FASTMATHFLAGS_NNAN ((uint8_t)2)
#define LOOM_VECTOR_FASTMATHFLAGS_NINF ((uint8_t)4)
#define LOOM_VECTOR_FASTMATHFLAGS_NSZ ((uint8_t)8)
#define LOOM_VECTOR_FASTMATHFLAGS_ARCP ((uint8_t)16)
#define LOOM_VECTOR_FASTMATHFLAGS_CONTRACT ((uint8_t)32)
#define LOOM_VECTOR_FASTMATHFLAGS_AFN ((uint8_t)64)
#define LOOM_VECTOR_FASTMATHFLAGS_FAST ((uint8_t)127)

// Integer overflow behavior flags.
#define LOOM_VECTOR_INTOVERFLOWFLAGS_NSW ((uint8_t)1)
#define LOOM_VECTOR_INTOVERFLOWFLAGS_NUW ((uint8_t)2)

typedef enum loom_vector_role_e {
  LOOM_VECTOR_ROLE_LHS = 0,
  LOOM_VECTOR_ROLE_RHS = 1,
  LOOM_VECTOR_ROLE_INIT = 2,
  LOOM_VECTOR_ROLE_RESULT = 3,
  LOOM_VECTOR_ROLE_COUNT_ = 4,
} loom_vector_role_t;

// NaN handling policy for table-based scalar quantization.
typedef enum loom_vector_table_quantize_nan_e {
  LOOM_VECTOR_TABLE_QUANTIZE_NAN_ZERO = 0,
  LOOM_VECTOR_TABLE_QUANTIZE_NAN_MAX = 1,
  LOOM_VECTOR_TABLE_QUANTIZE_NAN_COUNT_ = 2,
} loom_vector_table_quantize_nan_t;

// Threshold equality policy for table-based scalar quantization.
typedef enum loom_vector_table_quantize_tie_e {
  LOOM_VECTOR_TABLE_QUANTIZE_TIE_LOWER = 0,
  LOOM_VECTOR_TABLE_QUANTIZE_TIE_UPPER = 1,
  LOOM_VECTOR_TABLE_QUANTIZE_TIE_COUNT_ = 2,
} loom_vector_table_quantize_tie_t;

// Integer comparison predicates.
typedef enum loom_vector_cmpi_predicate_e {
  LOOM_VECTOR_CMPI_PREDICATE_EQ = 0,
  LOOM_VECTOR_CMPI_PREDICATE_NE = 1,
  LOOM_VECTOR_CMPI_PREDICATE_SLT = 2,
  LOOM_VECTOR_CMPI_PREDICATE_SLE = 3,
  LOOM_VECTOR_CMPI_PREDICATE_SGT = 4,
  LOOM_VECTOR_CMPI_PREDICATE_SGE = 5,
  LOOM_VECTOR_CMPI_PREDICATE_ULT = 6,
  LOOM_VECTOR_CMPI_PREDICATE_ULE = 7,
  LOOM_VECTOR_CMPI_PREDICATE_UGT = 8,
  LOOM_VECTOR_CMPI_PREDICATE_UGE = 9,
  LOOM_VECTOR_CMPI_PREDICATE_COUNT_ = 10,
} loom_vector_cmpi_predicate_t;

// Floating-point comparison predicates (IEEE 754 total order).
typedef enum loom_vector_cmpf_predicate_e {
  LOOM_VECTOR_CMPF_PREDICATE_OEQ = 0,
  LOOM_VECTOR_CMPF_PREDICATE_OGT = 1,
  LOOM_VECTOR_CMPF_PREDICATE_OGE = 2,
  LOOM_VECTOR_CMPF_PREDICATE_OLT = 3,
  LOOM_VECTOR_CMPF_PREDICATE_OLE = 4,
  LOOM_VECTOR_CMPF_PREDICATE_ONE = 5,
  LOOM_VECTOR_CMPF_PREDICATE_ORD = 6,
  LOOM_VECTOR_CMPF_PREDICATE_UEQ = 7,
  LOOM_VECTOR_CMPF_PREDICATE_UGT = 8,
  LOOM_VECTOR_CMPF_PREDICATE_UGE = 9,
  LOOM_VECTOR_CMPF_PREDICATE_ULT = 10,
  LOOM_VECTOR_CMPF_PREDICATE_ULE = 11,
  LOOM_VECTOR_CMPF_PREDICATE_UNE = 12,
  LOOM_VECTOR_CMPF_PREDICATE_UNO = 13,
  LOOM_VECTOR_CMPF_PREDICATE_COUNT_ = 14,
} loom_vector_cmpf_predicate_t;

// Floating-point clamp NaN and comparison policy.
typedef enum loom_vector_clampf_mode_e {
  LOOM_VECTOR_CLAMPF_MODE_ORDERED = 0,
  LOOM_VECTOR_CLAMPF_MODE_NUMBER = 1,
  LOOM_VECTOR_CLAMPF_MODE_IEEE = 2,
  LOOM_VECTOR_CLAMPF_MODE_COUNT_ = 3,
} loom_vector_clampf_mode_t;

// GELU activation formula family.
typedef enum loom_vector_geluf_variant_e {
  LOOM_VECTOR_GELUF_VARIANT_ERF = 0,
  LOOM_VECTOR_GELUF_VARIANT_TANH = 1,
  LOOM_VECTOR_GELUF_VARIANT_LOGISTIC = 2,
  LOOM_VECTOR_GELUF_VARIANT_COUNT_ = 3,
} loom_vector_geluf_variant_t;

// Signedness variants for four-lane i8 dot products accumulated into i32 lanes.
typedef enum loom_vector_dot4i_kind_e {
  LOOM_VECTOR_DOT4I_KIND_S8S8 = 0,
  LOOM_VECTOR_DOT4I_KIND_U8S8 = 1,
  LOOM_VECTOR_DOT4I_KIND_S8U8 = 2,
  LOOM_VECTOR_DOT4I_KIND_U8U8 = 3,
  LOOM_VECTOR_DOT4I_KIND_COUNT_ = 4,
} loom_vector_dot4i_kind_t;

// Signedness variants for packed eight-lane i4 dot products accumulated into i32 lanes.
typedef enum loom_vector_dot8i4_kind_e {
  LOOM_VECTOR_DOT8I4_KIND_S4S4 = 0,
  LOOM_VECTOR_DOT8I4_KIND_U4S4 = 1,
  LOOM_VECTOR_DOT8I4_KIND_S4U4 = 2,
  LOOM_VECTOR_DOT8I4_KIND_U4U4 = 3,
  LOOM_VECTOR_DOT8I4_KIND_COUNT_ = 4,
} loom_vector_dot8i4_kind_t;

// Format variants for packed four-lane fp8/bf8 dot products accumulated into f32 lanes.
typedef enum loom_vector_dot4f8_kind_e {
  LOOM_VECTOR_DOT4F8_KIND_FP8BF8 = 0,
  LOOM_VECTOR_DOT4F8_KIND_BF8FP8 = 1,
  LOOM_VECTOR_DOT4F8_KIND_FP8FP8 = 2,
  LOOM_VECTOR_DOT4F8_KIND_BF8BF8 = 3,
  LOOM_VECTOR_DOT4F8_KIND_COUNT_ = 4,
} loom_vector_dot4f8_kind_t;

// LOOM_OP_VECTOR_CONSTANT: Materialize a compile-time vector value whose every lane has the same scalar attribute payload. The result type supplies both the vector shape and the element type used to interpret the payload.
// %v = vector.constant 0.0 : vector<4xf32>
LOOM_DEFINE_ISA(loom_vector_constant_isa, LOOM_OP_VECTOR_CONSTANT)
LOOM_DEFINE_RESULT(loom_vector_constant_result, 0)
LOOM_DEFINE_ATTR_ANY(loom_vector_constant_value, 0)
iree_status_t loom_vector_constant_build(
    loom_builder_t* builder,
    loom_attribute_t value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_constant_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_POISON: Materialize a typed Loom poison vector. Poison represents an invalid vector value and propagates through pure vector ops until dead-code elimination removes it or a boundary diagnoses it. A zero-lane vector such as vector<0xf32> is not poison: it is an empty aggregate whose pure lane-wise computation and zero-lane memory effects should canonicalize away. Poison is introduced when IR observes something that cannot exist, such as a lane extracted from a vector proven to have zero lanes.
// %p = vector.poison : vector<4xf32>
LOOM_DEFINE_ISA(loom_vector_poison_isa, LOOM_OP_VECTOR_POISON)
LOOM_DEFINE_RESULT(loom_vector_poison_result, 0)
iree_status_t loom_vector_poison_build(
    loom_builder_t* builder,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_poison_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_EMPTY: Materialize the unique empty aggregate value for a static zero-lane vector type. Empty vectors are ordinary values, not poison, and pure zero-lane computation canonicalizes to this op.
// %v = vector.empty : vector<0xf32>
LOOM_DEFINE_ISA(loom_vector_empty_isa, LOOM_OP_VECTOR_EMPTY)
LOOM_DEFINE_RESULT(loom_vector_empty_result, 0)
iree_status_t loom_vector_empty_build(
    loom_builder_t* builder,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_empty_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_SPLAT: Replicate one scalar value to every lane of a vector result. The annotation after ':' is the result vector type; the scalar operand must already have the same element type, so conversions must be spelled with scalar/vector cast ops before or after the splat.
// %vec = vector.splat %scalar : vector<16xf32>
LOOM_DEFINE_ISA(loom_vector_splat_isa, LOOM_OP_VECTOR_SPLAT)
LOOM_DEFINE_OPERAND(loom_vector_splat_scalar, 0)
LOOM_DEFINE_RESULT(loom_vector_splat_result, 0)
iree_status_t loom_vector_splat_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t scalar,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_uniform_result_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_vector_splat_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_IOTA: Construct a vector of lane-coordinate values. Lane order is the logical row-major order of the result shape; result lane ordinal i contains base + i * step. The result element type must be index or a non-i1 integer payload, and base/step must be scalar values with the same element type. Dynamic result extents are allowed: the result type supplies the lane count symbolically and later specialization fixes the concrete number of produced coordinates.
// %lanes = vector.iota %c0 step %c1 : vector<16xindex>
LOOM_DEFINE_ISA(loom_vector_iota_isa, LOOM_OP_VECTOR_IOTA)
LOOM_DEFINE_OPERAND(loom_vector_iota_base, 0)
LOOM_DEFINE_OPERAND(loom_vector_iota_step, 1)
LOOM_DEFINE_RESULT(loom_vector_iota_result, 0)
iree_status_t loom_vector_iota_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t base,
    loom_may_consume loom_value_id_t step,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_iota_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_vector_iota_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_MASK_RANGE: Construct an i1 tail mask from an explicit scalar coordinate range. For logical lane ordinal i, the lane is true when lower_bound + i * step is strictly less than upper_bound using the coordinate domain's signed ordering. The bracketed syntax mirrors scf.for ranges because the same inclusive-lower, exclusive-upper semantics are being tested; the result vector type supplies the number and shape of lanes to test.
// %mask = vector.mask.range [%iv to %n step %c1] : index -> vector<16xi1>
LOOM_DEFINE_ISA(loom_vector_mask_range_isa, LOOM_OP_VECTOR_MASK_RANGE)
LOOM_DEFINE_OPERAND(loom_vector_mask_range_lower_bound, 0)
LOOM_DEFINE_OPERAND(loom_vector_mask_range_upper_bound, 1)
LOOM_DEFINE_OPERAND(loom_vector_mask_range_step, 2)
LOOM_DEFINE_RESULT(loom_vector_mask_range_result, 0)
iree_status_t loom_vector_mask_range_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t lower_bound,
    loom_may_consume loom_value_id_t upper_bound,
    loom_may_consume loom_value_id_t step,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_mask_range_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_BROADCAST: Broadcast a vector value to a larger-rank or same-rank vector result. Source axes align with the trailing result axes, and each static source extent must either be 1 or match the corresponding result extent.
// %wide = vector.broadcast %v : vector<4xf32> -> vector<16x4xf32>
LOOM_DEFINE_ISA(loom_vector_broadcast_isa, LOOM_OP_VECTOR_BROADCAST)
LOOM_DEFINE_OPERAND(loom_vector_broadcast_source, 0)
LOOM_DEFINE_RESULT(loom_vector_broadcast_result, 0)
iree_status_t loom_vector_broadcast_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_broadcast_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_vector_broadcast_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_FROM_ELEMENTS: Build an all-static vector from scalar element operands in logical lane order. The result vector type defines both the lane count and element type: the number of operands must equal the static element count, and every operand must have the vector element type.
// %v = vector.from_elements %a, %b, %c, %d : vector<4xf32>
LOOM_DEFINE_ISA(loom_vector_from_elements_isa, LOOM_OP_VECTOR_FROM_ELEMENTS)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_from_elements_elements, 0)
LOOM_DEFINE_RESULT(loom_vector_from_elements_result, 0)
iree_status_t loom_vector_from_elements_build(
    loom_builder_t* builder,
    loom_may_consume const loom_value_id_t* elements,
    iree_host_size_t elements_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_from_elements_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_vector_from_elements_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_EXTRACT: Extract a scalar or tail subvector from a vector at explicit leading indices. Supplying one index consumes the first source axis, two indices consume the first two axes, and consuming all axes produces a scalar element.
// %x = vector.extract %v[%i] : vector<[%n]xf32> -> f32
LOOM_DEFINE_ISA(loom_vector_extract_isa, LOOM_OP_VECTOR_EXTRACT)
LOOM_DEFINE_OPERAND(loom_vector_extract_source, 0)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_extract_indices, 1)
LOOM_DEFINE_RESULT(loom_vector_extract_result, 0)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_extract_static_indices, 0)
iree_status_t loom_vector_extract_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_extract_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_vector_extract_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_vector_extract_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_INSERT: Insert a scalar or tail subvector into a vector at explicit leading indices. The inserted value must match the destination tail shape remaining after the supplied indices, and the result type is the same as the destination type.
// %r = vector.insert %x into %v[%i] : f32, vector<[%n]xf32>
LOOM_DEFINE_ISA(loom_vector_insert_isa, LOOM_OP_VECTOR_INSERT)
LOOM_DEFINE_OPERAND(loom_vector_insert_value, 0)
LOOM_DEFINE_OPERAND(loom_vector_insert_dest, 1)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_insert_indices, 2)
LOOM_DEFINE_RESULT(loom_vector_insert_result, 0)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_insert_static_indices, 0)
iree_status_t loom_vector_insert_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t value,
    loom_may_consume loom_value_id_t dest,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_insert_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_vector_insert_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_SLICE: Extract a rank-preserving contiguous register subvector at explicit offsets. The offset list has one entry per source axis; each result axis extent describes how many lanes are kept from that source axis.
// %tail = vector.slice %v[%i] : vector<[%n]xf32> -> vector<4xf32>
LOOM_DEFINE_ISA(loom_vector_slice_isa, LOOM_OP_VECTOR_SLICE)
LOOM_DEFINE_OPERAND(loom_vector_slice_source, 0)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_slice_offsets, 1)
LOOM_DEFINE_RESULT(loom_vector_slice_result, 0)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_slice_static_offsets, 0)
iree_status_t loom_vector_slice_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    const loom_value_id_t* offsets,
    iree_host_size_t offsets_count,
    const int64_t* static_offsets,
    iree_host_size_t static_offsets_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_slice_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_vector_slice_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_CONCAT: Concatenate one or more same-rank vectors along the template axis. All non-concatenated axes must match the result shape, and when static the result axis extent must equal the sum of input extents.
// %wide = vector.concat<0> %a, %b : vector<4xf32>, vector<4xf32> -> vector<8xf32>
LOOM_DEFINE_ISA(loom_vector_concat_isa, LOOM_OP_VECTOR_CONCAT)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_concat_inputs, 0)
LOOM_DEFINE_RESULT(loom_vector_concat_result, 0)
LOOM_DEFINE_ATTR_I64(loom_vector_concat_axis, 0)
iree_status_t loom_vector_concat_build(
    loom_builder_t* builder,
    int64_t axis,
    loom_may_consume const loom_value_id_t* inputs,
    iree_host_size_t inputs_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_concat_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_vector_concat_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_TRANSPOSE: Permute vector register axes. The template list maps each result axis to a source axis: permutation[i] is the source axis used for result axis i, so <[1, 0]> maps vector<MxN> to vector<NxM>. This does not touch memory layout; it only reorders lanes in the register value.
// %t = vector.transpose<[1, 0]> %v : vector<4x8xf32> -> vector<8x4xf32>
LOOM_DEFINE_ISA(loom_vector_transpose_isa, LOOM_OP_VECTOR_TRANSPOSE)
LOOM_DEFINE_OPERAND(loom_vector_transpose_source, 0)
LOOM_DEFINE_RESULT(loom_vector_transpose_result, 0)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_transpose_permutation, 0)
iree_status_t loom_vector_transpose_build(
    loom_builder_t* builder,
    const int64_t* permutation,
    iree_host_size_t permutation_count,
    loom_may_consume loom_value_id_t source,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_transpose_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_vector_transpose_type_transfer(
    loom_type_transfer_context_t* context,
    const loom_module_t* module, loom_op_t* op);
iree_status_t loom_vector_transpose_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_SHUFFLE: Reorder a static rank-1 vector with a static lane map. Entry i of source_lanes selects the source lane for result lane i; duplicate source lanes are allowed, but the result type is the same as the source type.
// %rev = vector.shuffle<[3, 2, 1, 0]> %v : vector<4xf32>
LOOM_DEFINE_ISA(loom_vector_shuffle_isa, LOOM_OP_VECTOR_SHUFFLE)
LOOM_DEFINE_OPERAND(loom_vector_shuffle_source, 0)
LOOM_DEFINE_RESULT(loom_vector_shuffle_result, 0)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_shuffle_source_lanes, 0)
iree_status_t loom_vector_shuffle_build(
    loom_builder_t* builder,
    const int64_t* source_lanes,
    iree_host_size_t source_lanes_count,
    loom_value_id_t source,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_shuffle_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_vector_shuffle_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_vector_shuffle_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_INTERLEAVE: Interleave two same-typed vectors along the template axis. Result positions with even coordinates along that axis come from the first operand, odd coordinates come from the second operand, and the result extent on that axis is doubled.
// %r = vector.interleave<0> %lo, %hi : vector<16xi8>, vector<16xi8> -> vector<32xi8>
LOOM_DEFINE_ISA(loom_vector_interleave_isa, LOOM_OP_VECTOR_INTERLEAVE)
LOOM_DEFINE_OPERAND(loom_vector_interleave_even, 0)
LOOM_DEFINE_OPERAND(loom_vector_interleave_odd, 1)
LOOM_DEFINE_RESULT(loom_vector_interleave_result, 0)
LOOM_DEFINE_ATTR_I64(loom_vector_interleave_axis, 0)
iree_status_t loom_vector_interleave_build(
    loom_builder_t* builder,
    int64_t axis,
    loom_may_consume loom_value_id_t even,
    loom_may_consume loom_value_id_t odd,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_interleave_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_vector_interleave_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_DEINTERLEAVE: Split one vector along the template axis into two same-typed results. The first result receives even coordinates along that axis, the second receives odd coordinates, and each result extent on that axis is half of the source extent.
// %lo, %hi = vector.deinterleave<0> %r : vector<32xi8> -> vector<16xi8>, vector<16xi8>
LOOM_DEFINE_ISA(loom_vector_deinterleave_isa, LOOM_OP_VECTOR_DEINTERLEAVE)
LOOM_DEFINE_OPERAND(loom_vector_deinterleave_source, 0)
LOOM_DEFINE_VARIADIC_RESULTS(loom_vector_deinterleave_results, 0)
LOOM_DEFINE_ATTR_I64(loom_vector_deinterleave_axis, 0)
iree_status_t loom_vector_deinterleave_build(
    loom_builder_t* builder,
    int64_t axis,
    loom_may_consume loom_value_id_t source,
    const loom_type_t* result_types,
    iree_host_size_t result_count,
    const loom_tied_result_t* tied_results,
    iree_host_size_t tied_result_count,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_deinterleave_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_vector_deinterleave_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_TABLE_LOOKUP: Select values from a rank-1 register table using integer index lanes. Each result lane reads table[indices lane]; the result shape matches the index vector shape and the result element type matches the table element type.
// %values = vector.table.lookup %grid[%codes] : vector<16xf16>, vector<32xi8> -> vector<32xf16>
LOOM_DEFINE_ISA(loom_vector_table_lookup_isa, LOOM_OP_VECTOR_TABLE_LOOKUP)
LOOM_DEFINE_OPERAND(loom_vector_table_lookup_table, 0)
LOOM_DEFINE_OPERAND(loom_vector_table_lookup_indices, 1)
LOOM_DEFINE_RESULT(loom_vector_table_lookup_result, 0)
iree_status_t loom_vector_table_lookup_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t table,
    loom_may_consume loom_value_id_t indices,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_table_lookup_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_vector_table_lookup_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_TABLE_QUANTIZE: Map floating-point lanes to integer ordinal code lanes using an ordered rank-1 threshold table. For each input lane, the result code is the selected quantization bin; nan and tie attributes make NaN and threshold equality behavior explicit.
// %codes = vector.table.quantize %values, %thresholds {nan = zero, tie = lower} : vector<32xf32>, vector<15xf32> -> vector<32xi8>
LOOM_DEFINE_ISA(loom_vector_table_quantize_isa, LOOM_OP_VECTOR_TABLE_QUANTIZE)
LOOM_DEFINE_OPERAND(loom_vector_table_quantize_input, 0)
LOOM_DEFINE_OPERAND(loom_vector_table_quantize_thresholds, 1)
LOOM_DEFINE_RESULT(loom_vector_table_quantize_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_table_quantize_nan, 0, loom_vector_table_quantize_nan_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_table_quantize_tie, 1, loom_vector_table_quantize_tie_t)
iree_status_t loom_vector_table_quantize_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t input,
    loom_may_consume loom_value_id_t thresholds,
    loom_vector_table_quantize_nan_t nan,
    loom_vector_table_quantize_tie_t tie,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_table_quantize_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_vector_table_quantize_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_TRANSFORM: Apply an explicit numeric transform descriptor to vector register lanes. The transform operand is an encoding<transform> value that names the numeric mapping, such as scale/zero-point decode, whitening, or projection; verifier rules keep supported transform families and shape-changing parameters explicit. Hadamard-like families act along the last axis. `hadamard_sign` applies either an explicit per-lane sign table or deterministic seed-derived signs from the low bit of SplitMix64(seed + input lane) before the Hadamard. `sign_permute_hadamard` applies explicit signs to source lanes, gathers lanes through the explicit permutation vector, then applies the Hadamard.
// %r = vector.transform %v, %xf : vector<128xf32>, encoding<transform> -> vector<128xf32>
LOOM_DEFINE_ISA(loom_vector_transform_isa, LOOM_OP_VECTOR_TRANSFORM)
LOOM_DEFINE_OPERAND(loom_vector_transform_source, 0)
LOOM_DEFINE_OPERAND(loom_vector_transform_transform, 1)
LOOM_DEFINE_RESULT(loom_vector_transform_result, 0)
iree_status_t loom_vector_transform_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t transform,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_transform_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_vector_transform_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_FRAGMENT_LOAD: Load a target-shaped matrix fragment payload from a typed view at a full-rank logical origin. Unlike vector.load, the result vector shape is the physical fragment payload selected by role, logical matrix shape, view layout, and target legality; it is not an ordinary trailing-axis footprint of the view. The result carries fragment facts directly so vector.mma can consume it without a separate vector.fragment wrapper.
// %lhs = vector.fragment.load<lhs> %a[%row, %k0] shape [%m, %k] : view<[%M]x[%K]xf16, %layout> -> vector<16xf16>
LOOM_DEFINE_ISA(loom_vector_fragment_load_isa, LOOM_OP_VECTOR_FRAGMENT_LOAD)
LOOM_DEFINE_OPERAND(loom_vector_fragment_load_view, 0)
LOOM_DEFINE_OPERAND(loom_vector_fragment_load_rows, 1)
LOOM_DEFINE_OPERAND(loom_vector_fragment_load_columns, 2)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_fragment_load_indices, 3)
LOOM_DEFINE_RESULT(loom_vector_fragment_load_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_fragment_load_role, 0, loom_vector_role_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_fragment_load_cache_scope, 1, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_fragment_load_cache_temporal, 2, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_fragment_load_static_indices, 3)
enum loom_vector_fragment_load_build_flag_bits_e {
  LOOM_VECTOR_FRAGMENT_LOAD_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VECTOR_FRAGMENT_LOAD_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_vector_fragment_load_build_flags_t;
iree_status_t loom_vector_fragment_load_build(
    loom_builder_t* builder,
    loom_vector_fragment_load_build_flags_t build_flags,
    loom_vector_role_t role,
    loom_may_consume loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_may_consume loom_value_id_t rows,
    loom_may_consume loom_value_id_t columns,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_fragment_load_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_vector_fragment_load_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_FRAGMENT_STORE: Store a target-shaped matrix fragment payload into a typed view at a full-rank logical origin. The value is interpreted as the physical payload for the given fragment role and logical matrix shape; the store is therefore a matrix-fragment movement boundary, not an ordinary vector.store footprint.
// vector.fragment.store<result> %acc, %c[%row, %col] shape [%m, %n] : vector<8xf32>, view<[%M]x[%N]xf32, %layout>
LOOM_DEFINE_ISA(loom_vector_fragment_store_isa, LOOM_OP_VECTOR_FRAGMENT_STORE)
LOOM_DEFINE_OPERAND(loom_vector_fragment_store_value, 0)
LOOM_DEFINE_OPERAND(loom_vector_fragment_store_view, 1)
LOOM_DEFINE_OPERAND(loom_vector_fragment_store_rows, 2)
LOOM_DEFINE_OPERAND(loom_vector_fragment_store_columns, 3)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_fragment_store_indices, 4)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_fragment_store_role, 0, loom_vector_role_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_fragment_store_cache_scope, 1, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_fragment_store_cache_temporal, 2, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_fragment_store_static_indices, 3)
enum loom_vector_fragment_store_build_flag_bits_e {
  LOOM_VECTOR_FRAGMENT_STORE_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VECTOR_FRAGMENT_STORE_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_vector_fragment_store_build_flags_t;
iree_status_t loom_vector_fragment_store_build(
    loom_builder_t* builder,
    loom_vector_fragment_store_build_flags_t build_flags,
    loom_vector_role_t role,
    loom_value_id_t value,
    loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_value_id_t rows,
    loom_value_id_t columns,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_fragment_store_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_LOAD: Load a vector footprint from a typed view at a full-rank logical origin. The index list addresses the origin in view coordinates; vector axes map onto the trailing view axes, so leading view axes select a slice and trailing axes describe the loaded footprint.
// %v = vector.load %view[%row, %col] : view<[%m]x[%n]xf32, %layout> -> vector<4x8xf32>
LOOM_DEFINE_ISA(loom_vector_load_isa, LOOM_OP_VECTOR_LOAD)
LOOM_DEFINE_OPERAND(loom_vector_load_view, 0)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_load_indices, 1)
LOOM_DEFINE_RESULT(loom_vector_load_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_load_cache_scope, 0, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_load_cache_temporal, 1, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_load_static_indices, 2)
enum loom_vector_load_build_flag_bits_e {
  LOOM_VECTOR_LOAD_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VECTOR_LOAD_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_vector_load_build_flags_t;
iree_status_t loom_vector_load_build(
    loom_builder_t* builder,
    loom_vector_load_build_flags_t build_flags,
    loom_may_consume loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_load_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_STORE: Store a vector footprint into a typed view at a full-rank logical origin. The index list addresses the origin in view coordinates; vector axes map onto the trailing view axes, matching vector.load.
// vector.store %v, %view[%row, %col] : vector<4x8xf32>, view<[%m]x[%n]xf32, %layout>
LOOM_DEFINE_ISA(loom_vector_store_isa, LOOM_OP_VECTOR_STORE)
LOOM_DEFINE_OPERAND(loom_vector_store_value, 0)
LOOM_DEFINE_OPERAND(loom_vector_store_view, 1)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_store_indices, 2)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_store_cache_scope, 0, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_store_cache_temporal, 1, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_store_static_indices, 2)
enum loom_vector_store_build_flag_bits_e {
  LOOM_VECTOR_STORE_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VECTOR_STORE_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_vector_store_build_flags_t;
iree_status_t loom_vector_store_build(
    loom_builder_t* builder,
    loom_vector_store_build_flags_t build_flags,
    loom_value_id_t value,
    loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_store_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_LOAD_MASK: Masked vector load from a typed view. Mask lanes with true values perform the same access as vector.load, while false lanes do not access memory and instead take the corresponding passthrough lane.
// %v = vector.load.mask %view[%row, %col], %mask, %old : view<[%m]x[%n]xf32, %layout>, vector<4x8xi1>, vector<4x8xf32>
LOOM_DEFINE_ISA(loom_vector_load_mask_isa, LOOM_OP_VECTOR_LOAD_MASK)
LOOM_DEFINE_OPERAND(loom_vector_load_mask_view, 0)
LOOM_DEFINE_OPERAND(loom_vector_load_mask_mask, 1)
LOOM_DEFINE_OPERAND(loom_vector_load_mask_passthrough, 2)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_load_mask_indices, 3)
LOOM_DEFINE_RESULT(loom_vector_load_mask_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_load_mask_cache_scope, 0, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_load_mask_cache_temporal, 1, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_load_mask_static_indices, 2)
enum loom_vector_load_mask_build_flag_bits_e {
  LOOM_VECTOR_LOAD_MASK_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VECTOR_LOAD_MASK_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_vector_load_mask_build_flags_t;
iree_status_t loom_vector_load_mask_build(
    loom_builder_t* builder,
    loom_vector_load_mask_build_flags_t build_flags,
    loom_may_consume loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_may_consume loom_value_id_t mask,
    loom_may_consume loom_value_id_t passthrough,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_masked_memory_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_vector_load_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_STORE_MASK: Masked vector store into a typed view. True mask lanes store the corresponding value lane, and false mask lanes do not access memory and leave the destination unchanged.
// vector.store.mask %v, %view[%row, %col], %mask : vector<4x8xf32>, view<[%m]x[%n]xf32, %layout>, vector<4x8xi1>
LOOM_DEFINE_ISA(loom_vector_store_mask_isa, LOOM_OP_VECTOR_STORE_MASK)
LOOM_DEFINE_OPERAND(loom_vector_store_mask_value, 0)
LOOM_DEFINE_OPERAND(loom_vector_store_mask_view, 1)
LOOM_DEFINE_OPERAND(loom_vector_store_mask_mask, 2)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_store_mask_indices, 3)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_store_mask_cache_scope, 0, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_store_mask_cache_temporal, 1, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_store_mask_static_indices, 2)
enum loom_vector_store_mask_build_flag_bits_e {
  LOOM_VECTOR_STORE_MASK_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VECTOR_STORE_MASK_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_vector_store_mask_build_flags_t;
iree_status_t loom_vector_store_mask_build(
    loom_builder_t* builder,
    loom_vector_store_mask_build_flags_t build_flags,
    loom_value_id_t value,
    loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_value_id_t mask,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_store_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_LOAD_EXPAND: Rank-1 masked expand load from consecutive view elements. Active lanes consume memory densely in increasing lane order; inactive lanes do not consume memory and take the corresponding passthrough lane.
// %v = vector.load.expand %view[%row, %col], %mask, %old : view<[%m]x[%n]xf32, %layout>, vector<4xi1>, vector<4xf32>
LOOM_DEFINE_ISA(loom_vector_load_expand_isa, LOOM_OP_VECTOR_LOAD_EXPAND)
LOOM_DEFINE_OPERAND(loom_vector_load_expand_view, 0)
LOOM_DEFINE_OPERAND(loom_vector_load_expand_mask, 1)
LOOM_DEFINE_OPERAND(loom_vector_load_expand_passthrough, 2)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_load_expand_indices, 3)
LOOM_DEFINE_RESULT(loom_vector_load_expand_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_load_expand_cache_scope, 0, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_load_expand_cache_temporal, 1, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_load_expand_static_indices, 2)
enum loom_vector_load_expand_build_flag_bits_e {
  LOOM_VECTOR_LOAD_EXPAND_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VECTOR_LOAD_EXPAND_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_vector_load_expand_build_flags_t;
iree_status_t loom_vector_load_expand_build(
    loom_builder_t* builder,
    loom_vector_load_expand_build_flags_t build_flags,
    loom_may_consume loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_may_consume loom_value_id_t mask,
    loom_may_consume loom_value_id_t passthrough,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_load_expand_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_STORE_COMPRESS: Rank-1 masked compress store to consecutive view elements. Active lanes write densely in increasing lane order; inactive lanes do not produce memory elements.
// vector.store.compress %v, %view[%row, %col], %mask : vector<4xf32>, view<[%m]x[%n]xf32, %layout>, vector<4xi1>
LOOM_DEFINE_ISA(loom_vector_store_compress_isa, LOOM_OP_VECTOR_STORE_COMPRESS)
LOOM_DEFINE_OPERAND(loom_vector_store_compress_value, 0)
LOOM_DEFINE_OPERAND(loom_vector_store_compress_view, 1)
LOOM_DEFINE_OPERAND(loom_vector_store_compress_mask, 2)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_store_compress_indices, 3)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_store_compress_cache_scope, 0, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_store_compress_cache_temporal, 1, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_store_compress_static_indices, 2)
enum loom_vector_store_compress_build_flag_bits_e {
  LOOM_VECTOR_STORE_COMPRESS_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VECTOR_STORE_COMPRESS_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_vector_store_compress_build_flags_t;
iree_status_t loom_vector_store_compress_build(
    loom_builder_t* builder,
    loom_vector_store_compress_build_flags_t build_flags,
    loom_value_id_t value,
    loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_value_id_t mask,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_store_compress_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_GATHER: Gather a vector from per-lane signed logical offsets added to the last view axis of a full-rank view origin. Each result lane reads origin with the final coordinate adjusted by offsets[lane]; the offset vector shape matches the result shape.
// %v = vector.gather %view[%row, %col][%offsets] : view<[%m]x[%n]xf32, %layout>, vector<4xindex> -> vector<4xf32>
LOOM_DEFINE_ISA(loom_vector_gather_isa, LOOM_OP_VECTOR_GATHER)
LOOM_DEFINE_OPERAND(loom_vector_gather_view, 0)
LOOM_DEFINE_OPERAND(loom_vector_gather_offsets, 1)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_gather_indices, 2)
LOOM_DEFINE_RESULT(loom_vector_gather_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_gather_cache_scope, 0, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_gather_cache_temporal, 1, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_gather_static_indices, 2)
enum loom_vector_gather_build_flag_bits_e {
  LOOM_VECTOR_GATHER_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VECTOR_GATHER_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_vector_gather_build_flags_t;
iree_status_t loom_vector_gather_build(
    loom_builder_t* builder,
    loom_vector_gather_build_flags_t build_flags,
    loom_may_consume loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_may_consume loom_value_id_t offsets,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_gather_scatter_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_vector_gather_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_SCATTER: Non-atomic scatter of a vector to per-lane signed logical offsets added to the last view axis of a full-rank view origin. Each lane writes origin with the final coordinate adjusted by offsets[lane], and active lane addresses must be distinct because no atomic conflict resolution is implied.
// vector.scatter %v, %view[%row, %col][%offsets] : vector<4xf32>, view<[%m]x[%n]xf32, %layout>, vector<4xindex>
LOOM_DEFINE_ISA(loom_vector_scatter_isa, LOOM_OP_VECTOR_SCATTER)
LOOM_DEFINE_OPERAND(loom_vector_scatter_value, 0)
LOOM_DEFINE_OPERAND(loom_vector_scatter_view, 1)
LOOM_DEFINE_OPERAND(loom_vector_scatter_offsets, 2)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_scatter_indices, 3)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_scatter_cache_scope, 0, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_scatter_cache_temporal, 1, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_scatter_static_indices, 2)
enum loom_vector_scatter_build_flag_bits_e {
  LOOM_VECTOR_SCATTER_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VECTOR_SCATTER_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_vector_scatter_build_flags_t;
iree_status_t loom_vector_scatter_build(
    loom_builder_t* builder,
    loom_vector_scatter_build_flags_t build_flags,
    loom_value_id_t value,
    loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_value_id_t offsets,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_scatter_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_GATHER_MASK: Masked vector gather from per-lane signed logical offsets added to the last view axis. True mask lanes read the adjusted coordinate, while false mask lanes do not access memory and take the corresponding passthrough lane.
// %v = vector.gather.mask %view[%row, %col][%offsets], %mask, %old : view<[%m]x[%n]xf32, %layout>, vector<4xindex>, vector<4xi1>, vector<4xf32>
LOOM_DEFINE_ISA(loom_vector_gather_mask_isa, LOOM_OP_VECTOR_GATHER_MASK)
LOOM_DEFINE_OPERAND(loom_vector_gather_mask_view, 0)
LOOM_DEFINE_OPERAND(loom_vector_gather_mask_offsets, 1)
LOOM_DEFINE_OPERAND(loom_vector_gather_mask_mask, 2)
LOOM_DEFINE_OPERAND(loom_vector_gather_mask_passthrough, 3)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_gather_mask_indices, 4)
LOOM_DEFINE_RESULT(loom_vector_gather_mask_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_gather_mask_cache_scope, 0, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_gather_mask_cache_temporal, 1, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_gather_mask_static_indices, 2)
enum loom_vector_gather_mask_build_flag_bits_e {
  LOOM_VECTOR_GATHER_MASK_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VECTOR_GATHER_MASK_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_vector_gather_mask_build_flags_t;
iree_status_t loom_vector_gather_mask_build(
    loom_builder_t* builder,
    loom_vector_gather_mask_build_flags_t build_flags,
    loom_may_consume loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_may_consume loom_value_id_t offsets,
    loom_may_consume loom_value_id_t mask,
    loom_may_consume loom_value_id_t passthrough,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_gather_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_SCATTER_MASK: Masked non-atomic scatter. True mask lanes write the full-rank origin with the last coordinate adjusted by offsets[lane], false mask lanes do not access memory, and active lane addresses must be distinct.
// vector.scatter.mask %v, %view[%row, %col][%offsets], %mask : vector<4xf32>, view<[%m]x[%n]xf32, %layout>, vector<4xindex>, vector<4xi1>
LOOM_DEFINE_ISA(loom_vector_scatter_mask_isa, LOOM_OP_VECTOR_SCATTER_MASK)
LOOM_DEFINE_OPERAND(loom_vector_scatter_mask_value, 0)
LOOM_DEFINE_OPERAND(loom_vector_scatter_mask_view, 1)
LOOM_DEFINE_OPERAND(loom_vector_scatter_mask_offsets, 2)
LOOM_DEFINE_OPERAND(loom_vector_scatter_mask_mask, 3)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_scatter_mask_indices, 4)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_scatter_mask_cache_scope, 0, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_scatter_mask_cache_temporal, 1, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_scatter_mask_static_indices, 2)
enum loom_vector_scatter_mask_build_flag_bits_e {
  LOOM_VECTOR_SCATTER_MASK_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VECTOR_SCATTER_MASK_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_vector_scatter_mask_build_flags_t;
iree_status_t loom_vector_scatter_mask_build(
    loom_builder_t* builder,
    loom_vector_scatter_mask_build_flags_t build_flags,
    loom_value_id_t value,
    loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_value_id_t offsets,
    loom_value_id_t mask,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_scatter_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_ATOMIC_REDUCE: Atomic no-result scatter reduction/update into per-lane signed element offsets. Each lane atomically combines its value into origin + offsets[lane]; duplicate active addresses are valid and are serialized by the required ordering and scope attributes.
// vector.atomic.reduce<addi> %v, %view[%row, %col][%offsets] {ordering = relaxed, scope = workgroup} : vector<4xi32>, view<[%m]x[%n]xi32, %layout>, vector<4xindex>
LOOM_DEFINE_ISA(loom_vector_atomic_reduce_isa, LOOM_OP_VECTOR_ATOMIC_REDUCE)
LOOM_DEFINE_OPERAND(loom_vector_atomic_reduce_value, 0)
LOOM_DEFINE_OPERAND(loom_vector_atomic_reduce_view, 1)
LOOM_DEFINE_OPERAND(loom_vector_atomic_reduce_offsets, 2)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_atomic_reduce_indices, 3)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_reduce_kind, 0, loom_atomic_kind_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_reduce_ordering, 1, loom_atomic_ordering_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_reduce_scope, 2, loom_atomic_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_reduce_cache_scope, 3, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_reduce_cache_temporal, 4, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_atomic_reduce_static_indices, 5)
enum loom_vector_atomic_reduce_build_flag_bits_e {
  LOOM_VECTOR_ATOMIC_REDUCE_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VECTOR_ATOMIC_REDUCE_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_vector_atomic_reduce_build_flags_t;
iree_status_t loom_vector_atomic_reduce_build(
    loom_builder_t* builder,
    loom_vector_atomic_reduce_build_flags_t build_flags,
    loom_atomic_kind_t kind,
    loom_value_id_t value,
    loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_value_id_t offsets,
    loom_atomic_ordering_t ordering,
    loom_atomic_scope_t scope,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_atomic_reduce_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_ATOMIC_REDUCE_MASK: Masked atomic no-result scatter reduction/update. True mask lanes perform vector.atomic.reduce, while false mask lanes do not access memory.
// vector.atomic.reduce.mask<addf> %v, %view[%row, %col][%offsets], %mask {ordering = relaxed, scope = device} : vector<4xf32>, view<[%m]x[%n]xf32, %layout>, vector<4xindex>, vector<4xi1>
LOOM_DEFINE_ISA(loom_vector_atomic_reduce_mask_isa, LOOM_OP_VECTOR_ATOMIC_REDUCE_MASK)
LOOM_DEFINE_OPERAND(loom_vector_atomic_reduce_mask_value, 0)
LOOM_DEFINE_OPERAND(loom_vector_atomic_reduce_mask_view, 1)
LOOM_DEFINE_OPERAND(loom_vector_atomic_reduce_mask_offsets, 2)
LOOM_DEFINE_OPERAND(loom_vector_atomic_reduce_mask_mask, 3)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_atomic_reduce_mask_indices, 4)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_reduce_mask_kind, 0, loom_atomic_kind_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_reduce_mask_ordering, 1, loom_atomic_ordering_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_reduce_mask_scope, 2, loom_atomic_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_reduce_mask_cache_scope, 3, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_reduce_mask_cache_temporal, 4, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_atomic_reduce_mask_static_indices, 5)
enum loom_vector_atomic_reduce_mask_build_flag_bits_e {
  LOOM_VECTOR_ATOMIC_REDUCE_MASK_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VECTOR_ATOMIC_REDUCE_MASK_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_vector_atomic_reduce_mask_build_flags_t;
iree_status_t loom_vector_atomic_reduce_mask_build(
    loom_builder_t* builder,
    loom_vector_atomic_reduce_mask_build_flags_t build_flags,
    loom_atomic_kind_t kind,
    loom_value_id_t value,
    loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_value_id_t offsets,
    loom_value_id_t mask,
    loom_atomic_ordering_t ordering,
    loom_atomic_scope_t scope,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_atomic_reduce_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_ATOMIC_RMW: Atomic read-modify-write at per-lane signed element offsets. Each lane atomically combines its value with origin + offsets[lane] and the result lane is the old memory value observed by that atomic operation.
// %old = vector.atomic.rmw<addi> %v, %view[%row, %col][%offsets] {ordering = relaxed, scope = workgroup} : vector<4xi32>, view<[%m]x[%n]xi32, %layout>, vector<4xindex>
LOOM_DEFINE_ISA(loom_vector_atomic_rmw_isa, LOOM_OP_VECTOR_ATOMIC_RMW)
LOOM_DEFINE_OPERAND(loom_vector_atomic_rmw_value, 0)
LOOM_DEFINE_OPERAND(loom_vector_atomic_rmw_view, 1)
LOOM_DEFINE_OPERAND(loom_vector_atomic_rmw_offsets, 2)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_atomic_rmw_indices, 3)
LOOM_DEFINE_RESULT(loom_vector_atomic_rmw_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_rmw_kind, 0, loom_atomic_kind_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_rmw_ordering, 1, loom_atomic_ordering_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_rmw_scope, 2, loom_atomic_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_rmw_cache_scope, 3, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_rmw_cache_temporal, 4, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_atomic_rmw_static_indices, 5)
enum loom_vector_atomic_rmw_build_flag_bits_e {
  LOOM_VECTOR_ATOMIC_RMW_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VECTOR_ATOMIC_RMW_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_vector_atomic_rmw_build_flags_t;
iree_status_t loom_vector_atomic_rmw_build(
    loom_builder_t* builder,
    loom_vector_atomic_rmw_build_flags_t build_flags,
    loom_atomic_kind_t kind,
    loom_may_consume loom_value_id_t value,
    loom_may_consume loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_may_consume loom_value_id_t offsets,
    loom_atomic_ordering_t ordering,
    loom_atomic_scope_t scope,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_atomic_rmw_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_ATOMIC_RMW_MASK: Masked atomic read-modify-write. True mask lanes perform vector.atomic.rmw, while false mask lanes do not access memory and take the corresponding passthrough lane in the result.
// %old = vector.atomic.rmw.mask<addf> %v, %view[%row, %col][%offsets], %mask, %passthrough {ordering = relaxed, scope = device} : vector<4xf32>, view<[%m]x[%n]xf32, %layout>, vector<4xindex>, vector<4xi1>, vector<4xf32>
LOOM_DEFINE_ISA(loom_vector_atomic_rmw_mask_isa, LOOM_OP_VECTOR_ATOMIC_RMW_MASK)
LOOM_DEFINE_OPERAND(loom_vector_atomic_rmw_mask_value, 0)
LOOM_DEFINE_OPERAND(loom_vector_atomic_rmw_mask_view, 1)
LOOM_DEFINE_OPERAND(loom_vector_atomic_rmw_mask_offsets, 2)
LOOM_DEFINE_OPERAND(loom_vector_atomic_rmw_mask_mask, 3)
LOOM_DEFINE_OPERAND(loom_vector_atomic_rmw_mask_passthrough, 4)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_atomic_rmw_mask_indices, 5)
LOOM_DEFINE_RESULT(loom_vector_atomic_rmw_mask_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_rmw_mask_kind, 0, loom_atomic_kind_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_rmw_mask_ordering, 1, loom_atomic_ordering_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_rmw_mask_scope, 2, loom_atomic_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_rmw_mask_cache_scope, 3, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_rmw_mask_cache_temporal, 4, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_atomic_rmw_mask_static_indices, 5)
enum loom_vector_atomic_rmw_mask_build_flag_bits_e {
  LOOM_VECTOR_ATOMIC_RMW_MASK_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VECTOR_ATOMIC_RMW_MASK_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_vector_atomic_rmw_mask_build_flags_t;
iree_status_t loom_vector_atomic_rmw_mask_build(
    loom_builder_t* builder,
    loom_vector_atomic_rmw_mask_build_flags_t build_flags,
    loom_atomic_kind_t kind,
    loom_may_consume loom_value_id_t value,
    loom_may_consume loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_may_consume loom_value_id_t offsets,
    loom_may_consume loom_value_id_t mask,
    loom_may_consume loom_value_id_t passthrough,
    loom_atomic_ordering_t ordering,
    loom_atomic_scope_t scope,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_atomic_rmw_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_ATOMIC_CMPXCHG: Atomic compare-exchange at per-lane signed element offsets. Each lane compares origin + offsets[lane] with expected[lane], writes replacement[lane] on success, and returns the old memory value. Success lanes are derived by comparing old == expected.
// %old = vector.atomic.cmpxchg %expected, %replacement, %view[%row, %col][%offsets] {success_ordering = acq_rel, failure_ordering = acquire, scope = workgroup} : vector<4xi32>, view<[%m]x[%n]xi32, %layout>, vector<4xindex> -> vector<4xi32>
LOOM_DEFINE_ISA(loom_vector_atomic_cmpxchg_isa, LOOM_OP_VECTOR_ATOMIC_CMPXCHG)
LOOM_DEFINE_OPERAND(loom_vector_atomic_cmpxchg_expected, 0)
LOOM_DEFINE_OPERAND(loom_vector_atomic_cmpxchg_replacement, 1)
LOOM_DEFINE_OPERAND(loom_vector_atomic_cmpxchg_view, 2)
LOOM_DEFINE_OPERAND(loom_vector_atomic_cmpxchg_offsets, 3)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_atomic_cmpxchg_indices, 4)
LOOM_DEFINE_RESULT(loom_vector_atomic_cmpxchg_old, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_cmpxchg_success_ordering, 0, loom_atomic_ordering_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_cmpxchg_failure_ordering, 1, loom_atomic_ordering_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_cmpxchg_scope, 2, loom_atomic_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_cmpxchg_cache_scope, 3, loom_cache_scope_t)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_atomic_cmpxchg_cache_temporal, 4, loom_cache_temporal_t)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_atomic_cmpxchg_static_indices, 5)
enum loom_vector_atomic_cmpxchg_build_flag_bits_e {
  LOOM_VECTOR_ATOMIC_CMPXCHG_BUILD_FLAG_HAS_CACHE_SCOPE = 1u << 0,
  LOOM_VECTOR_ATOMIC_CMPXCHG_BUILD_FLAG_HAS_CACHE_TEMPORAL = 1u << 1,
};
typedef uint32_t loom_vector_atomic_cmpxchg_build_flags_t;
iree_status_t loom_vector_atomic_cmpxchg_build(
    loom_builder_t* builder,
    loom_vector_atomic_cmpxchg_build_flags_t build_flags,
    loom_may_consume loom_value_id_t expected,
    loom_may_consume loom_value_id_t replacement,
    loom_may_consume loom_value_id_t view,
    const loom_value_id_t* indices,
    iree_host_size_t indices_count,
    const int64_t* static_indices,
    iree_host_size_t static_indices_count,
    loom_may_consume loom_value_id_t offsets,
    loom_atomic_ordering_t success_ordering,
    loom_atomic_ordering_t failure_ordering,
    loom_atomic_scope_t scope,
    loom_optional uint8_t cache_scope,
    loom_optional uint8_t cache_temporal,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_atomic_cmpxchg_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_SELECT: Lanewise select from two same-typed vector values using an i1 mask vector. True condition lanes choose true_value; false lanes choose false_value.
// %r = vector.select %mask, %a, %b : vector<16xf32>
LOOM_DEFINE_ISA(loom_vector_select_isa, LOOM_OP_VECTOR_SELECT)
LOOM_DEFINE_OPERAND(loom_vector_select_condition, 0)
LOOM_DEFINE_OPERAND(loom_vector_select_true_value, 1)
LOOM_DEFINE_OPERAND(loom_vector_select_false_value, 2)
LOOM_DEFINE_RESULT(loom_vector_select_result, 0)
iree_status_t loom_vector_select_build(
    loom_builder_t* builder,
    loom_value_id_t condition,
    loom_value_id_t true_value,
    loom_value_id_t false_value,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_select_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_vector_select_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_CMPI: Lanewise integer comparison producing an i1 mask vector. The predicate attribute uses the scalar.cmpi predicate names and applies independently to each lane.
// %m = vector.cmpi slt, %lhs, %rhs : vector<16xi32> -> vector<16xi1>
LOOM_DEFINE_ISA(loom_vector_cmpi_isa, LOOM_OP_VECTOR_CMPI)
LOOM_DEFINE_OPERAND(loom_vector_cmpi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_cmpi_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_cmpi_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_cmpi_predicate, 0, loom_vector_cmpi_predicate_t)
iree_status_t loom_vector_cmpi_build(
    loom_builder_t* builder, uint8_t predicate,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t operand_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_comparison_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_vector_cmpi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_CMPF: Lanewise floating-point comparison producing an i1 mask vector. The predicate attribute uses the scalar.cmpf ordered/unordered predicate names and applies independently to each lane.
// %m = vector.cmpf olt, %lhs, %rhs : vector<16xf32> -> vector<16xi1>
LOOM_DEFINE_ISA(loom_vector_cmpf_isa, LOOM_OP_VECTOR_CMPF)
LOOM_DEFINE_OPERAND(loom_vector_cmpf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_cmpf_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_cmpf_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_cmpf_predicate, 0, loom_vector_cmpf_predicate_t)
iree_status_t loom_vector_cmpf_build(
    loom_builder_t* builder, uint8_t predicate,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t operand_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_cmpf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ADDF: Lanewise floating-point addition of same-typed vector operands. Optional fastmath flags carry the same per-lane floating-point permissions as scalar.addf; reduction reassociation belongs on vector.reduce instead.
// vector.addf
LOOM_DEFINE_ISA(loom_vector_addf_isa, LOOM_OP_VECTOR_ADDF)
LOOM_DEFINE_OPERAND(loom_vector_addf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_addf_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_addf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_addf_fastmath)
iree_status_t loom_vector_addf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_addf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_SUBF: Lanewise floating-point subtraction of same-typed vector operands. Optional fastmath flags carry the same per-lane floating-point permissions as scalar.subf.
// vector.subf
LOOM_DEFINE_ISA(loom_vector_subf_isa, LOOM_OP_VECTOR_SUBF)
LOOM_DEFINE_OPERAND(loom_vector_subf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_subf_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_subf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_subf_fastmath)
iree_status_t loom_vector_subf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_subf_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_vector_subf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_MULF: Lanewise floating-point multiplication of same-typed vector operands. Optional fastmath flags carry the same per-lane floating-point permissions as scalar.mulf.
// vector.mulf
LOOM_DEFINE_ISA(loom_vector_mulf_isa, LOOM_OP_VECTOR_MULF)
LOOM_DEFINE_OPERAND(loom_vector_mulf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_mulf_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_mulf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_mulf_fastmath)
iree_status_t loom_vector_mulf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_mulf_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_vector_mulf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_DIVF: Lanewise floating-point division of same-typed vector operands. Optional fastmath flags carry the same per-lane floating-point permissions as scalar.divf, including arcp for reciprocal formation.
// vector.divf
LOOM_DEFINE_ISA(loom_vector_divf_isa, LOOM_OP_VECTOR_DIVF)
LOOM_DEFINE_OPERAND(loom_vector_divf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_divf_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_divf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_divf_fastmath)
iree_status_t loom_vector_divf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_divf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_REMF: Lanewise floating-point remainder with C fmod semantics over same-typed vector operands.
// vector.remf
LOOM_DEFINE_ISA(loom_vector_remf_isa, LOOM_OP_VECTOR_REMF)
LOOM_DEFINE_OPERAND(loom_vector_remf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_remf_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_remf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_remf_fastmath)
iree_status_t loom_vector_remf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_remf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_NEGF: Lanewise floating-point negation of a same-typed vector operand.
// vector.negf
LOOM_DEFINE_ISA(loom_vector_negf_isa, LOOM_OP_VECTOR_NEGF)
LOOM_DEFINE_OPERAND(loom_vector_negf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_negf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_negf_fastmath)
iree_status_t loom_vector_negf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_negf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ABSF: Lanewise floating-point absolute value of a same-typed vector operand.
// vector.absf
LOOM_DEFINE_ISA(loom_vector_absf_isa, LOOM_OP_VECTOR_ABSF)
LOOM_DEFINE_OPERAND(loom_vector_absf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_absf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_absf_fastmath)
iree_status_t loom_vector_absf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_absf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_MINIMUMF: Lanewise IEEE 754 floating-point minimum of same-typed vector operands; NaN lanes propagate.
// vector.minimumf
LOOM_DEFINE_ISA(loom_vector_minimumf_isa, LOOM_OP_VECTOR_MINIMUMF)
LOOM_DEFINE_OPERAND(loom_vector_minimumf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_minimumf_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_minimumf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_minimumf_fastmath)
iree_status_t loom_vector_minimumf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_minimumf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_MAXIMUMF: Lanewise IEEE 754 floating-point maximum of same-typed vector operands; NaN lanes propagate.
// vector.maximumf
LOOM_DEFINE_ISA(loom_vector_maximumf_isa, LOOM_OP_VECTOR_MAXIMUMF)
LOOM_DEFINE_OPERAND(loom_vector_maximumf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_maximumf_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_maximumf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_maximumf_fastmath)
iree_status_t loom_vector_maximumf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_maximumf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_MINNUMF: Lanewise C99 fmin-style floating-point minimum of same-typed vector operands; NaN lanes select the non-NaN operand.
// vector.minnumf
LOOM_DEFINE_ISA(loom_vector_minnumf_isa, LOOM_OP_VECTOR_MINNUMF)
LOOM_DEFINE_OPERAND(loom_vector_minnumf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_minnumf_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_minnumf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_minnumf_fastmath)
iree_status_t loom_vector_minnumf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_minnumf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_MAXNUMF: Lanewise C99 fmax-style floating-point maximum of same-typed vector operands; NaN lanes select the non-NaN operand.
// vector.maxnumf
LOOM_DEFINE_ISA(loom_vector_maxnumf_isa, LOOM_OP_VECTOR_MAXNUMF)
LOOM_DEFINE_OPERAND(loom_vector_maxnumf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_maxnumf_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_maxnumf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_maxnumf_fastmath)
iree_status_t loom_vector_maxnumf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_maxnumf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_CLAMPF: Lanewise floating-point clamp with explicit NaN/comparison policy. The ordered mode preserves strict compare/select semantics, number mode uses minnum/maxnum semantics, and ieee mode propagates NaNs.
// %result = vector.clampf<ordered> %value, %lower, %upper : vector<16xf32>
LOOM_DEFINE_ISA(loom_vector_clampf_isa, LOOM_OP_VECTOR_CLAMPF)
LOOM_DEFINE_OPERAND(loom_vector_clampf_value, 0)
LOOM_DEFINE_OPERAND(loom_vector_clampf_lower, 1)
LOOM_DEFINE_OPERAND(loom_vector_clampf_upper, 2)
LOOM_DEFINE_RESULT(loom_vector_clampf_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_clampf_mode, 0, loom_vector_clampf_mode_t)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_clampf_fastmath)
iree_status_t loom_vector_clampf_build(
    loom_builder_t* builder,
    loom_vector_clampf_mode_t mode,
    uint8_t instance_flags,
    loom_value_id_t value,
    loom_value_id_t lower,
    loom_value_id_t upper,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_clampf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_COPYSIGNF: Lanewise copy sign of rhs lanes onto lhs lane magnitudes.
// vector.copysignf
LOOM_DEFINE_ISA(loom_vector_copysignf_isa, LOOM_OP_VECTOR_COPYSIGNF)
LOOM_DEFINE_OPERAND(loom_vector_copysignf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_copysignf_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_copysignf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_copysignf_fastmath)
iree_status_t loom_vector_copysignf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_copysignf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_FMAF: Lanewise fused multiply-add of same-typed floating-point vectors. Each result lane computes a*b + c with one final rounding; use separate vector.mulf/vector.addf when unfused rounding is required.
// %r = vector.fmaf %a, %b, %c : vector<16xf32>
LOOM_DEFINE_ISA(loom_vector_fmaf_isa, LOOM_OP_VECTOR_FMAF)
LOOM_DEFINE_OPERAND(loom_vector_fmaf_a, 0)
LOOM_DEFINE_OPERAND(loom_vector_fmaf_b, 1)
LOOM_DEFINE_OPERAND(loom_vector_fmaf_c, 2)
LOOM_DEFINE_RESULT(loom_vector_fmaf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_fmaf_fastmath)
iree_status_t loom_vector_fmaf_build(
    loom_builder_t* builder,
    uint8_t instance_flags,
    loom_value_id_t a,
    loom_value_id_t b,
    loom_value_id_t c,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_fmaf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ADDI: Lanewise integer addition of same-typed vector operands. Optional overflow flags state required no-wrap facts for every lane.
// vector.addi
LOOM_DEFINE_ISA(loom_vector_addi_isa, LOOM_OP_VECTOR_ADDI)
LOOM_DEFINE_OPERAND(loom_vector_addi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_addi_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_addi_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_addi_overflow)
iree_status_t loom_vector_addi_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_binary_identity_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_vector_addi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_SUBI: Lanewise integer subtraction of same-typed vector operands. Optional overflow flags state required no-wrap facts for every lane.
// vector.subi
LOOM_DEFINE_ISA(loom_vector_subi_isa, LOOM_OP_VECTOR_SUBI)
LOOM_DEFINE_OPERAND(loom_vector_subi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_subi_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_subi_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_subi_overflow)
iree_status_t loom_vector_subi_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_subi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_MULI: Lanewise integer multiplication of same-typed vector operands. Optional overflow flags state required no-wrap facts for every lane.
// vector.muli
LOOM_DEFINE_ISA(loom_vector_muli_isa, LOOM_OP_VECTOR_MULI)
LOOM_DEFINE_OPERAND(loom_vector_muli_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_muli_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_muli_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_muli_overflow)
iree_status_t loom_vector_muli_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_muli_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_DIVSI: Lanewise signed integer division of same-typed vector operands; each lane rounds toward zero.
// vector.divsi
LOOM_DEFINE_ISA(loom_vector_divsi_isa, LOOM_OP_VECTOR_DIVSI)
LOOM_DEFINE_OPERAND(loom_vector_divsi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_divsi_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_divsi_result, 0)
iree_status_t loom_vector_divsi_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_divsi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_DIVUI: Lanewise unsigned integer division of same-typed vector operands.
// vector.divui
LOOM_DEFINE_ISA(loom_vector_divui_isa, LOOM_OP_VECTOR_DIVUI)
LOOM_DEFINE_OPERAND(loom_vector_divui_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_divui_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_divui_result, 0)
iree_status_t loom_vector_divui_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_divui_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_REMSI: Lanewise signed integer remainder of same-typed vector operands.
// vector.remsi
LOOM_DEFINE_ISA(loom_vector_remsi_isa, LOOM_OP_VECTOR_REMSI)
LOOM_DEFINE_OPERAND(loom_vector_remsi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_remsi_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_remsi_result, 0)
iree_status_t loom_vector_remsi_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_remsi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_REMUI: Lanewise unsigned integer remainder of same-typed vector operands.
// vector.remui
LOOM_DEFINE_ISA(loom_vector_remui_isa, LOOM_OP_VECTOR_REMUI)
LOOM_DEFINE_OPERAND(loom_vector_remui_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_remui_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_remui_result, 0)
iree_status_t loom_vector_remui_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_remui_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_CEILDIVSI: Lanewise signed integer division rounding toward positive infinity.
// vector.ceildivsi
LOOM_DEFINE_ISA(loom_vector_ceildivsi_isa, LOOM_OP_VECTOR_CEILDIVSI)
LOOM_DEFINE_OPERAND(loom_vector_ceildivsi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_ceildivsi_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_ceildivsi_result, 0)
iree_status_t loom_vector_ceildivsi_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

// LOOM_OP_VECTOR_CEILDIVUI: Lanewise unsigned integer division rounding toward positive infinity.
// vector.ceildivui
LOOM_DEFINE_ISA(loom_vector_ceildivui_isa, LOOM_OP_VECTOR_CEILDIVUI)
LOOM_DEFINE_OPERAND(loom_vector_ceildivui_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_ceildivui_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_ceildivui_result, 0)
iree_status_t loom_vector_ceildivui_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

// LOOM_OP_VECTOR_FLOORDIVSI: Lanewise signed integer division rounding toward negative infinity.
// vector.floordivsi
LOOM_DEFINE_ISA(loom_vector_floordivsi_isa, LOOM_OP_VECTOR_FLOORDIVSI)
LOOM_DEFINE_OPERAND(loom_vector_floordivsi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_floordivsi_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_floordivsi_result, 0)
iree_status_t loom_vector_floordivsi_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

// LOOM_OP_VECTOR_NEGI: Lanewise integer negation of a same-typed vector operand.
// vector.negi
LOOM_DEFINE_ISA(loom_vector_negi_isa, LOOM_OP_VECTOR_NEGI)
LOOM_DEFINE_OPERAND(loom_vector_negi_input, 0)
LOOM_DEFINE_RESULT(loom_vector_negi_result, 0)
iree_status_t loom_vector_negi_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_negi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ABSI: Lanewise integer absolute value of a same-typed vector operand.
// vector.absi
LOOM_DEFINE_ISA(loom_vector_absi_isa, LOOM_OP_VECTOR_ABSI)
LOOM_DEFINE_OPERAND(loom_vector_absi_input, 0)
LOOM_DEFINE_RESULT(loom_vector_absi_result, 0)
iree_status_t loom_vector_absi_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_absi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_MINSI: Lanewise signed integer minimum of same-typed vector operands.
// vector.minsi
LOOM_DEFINE_ISA(loom_vector_minsi_isa, LOOM_OP_VECTOR_MINSI)
LOOM_DEFINE_OPERAND(loom_vector_minsi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_minsi_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_minsi_result, 0)
iree_status_t loom_vector_minsi_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_minsi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_MAXSI: Lanewise signed integer maximum of same-typed vector operands.
// vector.maxsi
LOOM_DEFINE_ISA(loom_vector_maxsi_isa, LOOM_OP_VECTOR_MAXSI)
LOOM_DEFINE_OPERAND(loom_vector_maxsi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_maxsi_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_maxsi_result, 0)
iree_status_t loom_vector_maxsi_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_maxsi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_MINUI: Lanewise unsigned integer minimum of same-typed vector operands.
// vector.minui
LOOM_DEFINE_ISA(loom_vector_minui_isa, LOOM_OP_VECTOR_MINUI)
LOOM_DEFINE_OPERAND(loom_vector_minui_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_minui_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_minui_result, 0)
iree_status_t loom_vector_minui_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_minui_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_MAXUI: Lanewise unsigned integer maximum of same-typed vector operands.
// vector.maxui
LOOM_DEFINE_ISA(loom_vector_maxui_isa, LOOM_OP_VECTOR_MAXUI)
LOOM_DEFINE_OPERAND(loom_vector_maxui_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_maxui_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_maxui_result, 0)
iree_status_t loom_vector_maxui_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_maxui_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_FMAI: Lanewise fused integer multiply-add a*b + c over same-typed vector operands. Optional overflow flags state required no-wrap facts for every lane.
// %r = vector.fmai %a, %b, %c : vector<16xi32>
LOOM_DEFINE_ISA(loom_vector_fmai_isa, LOOM_OP_VECTOR_FMAI)
LOOM_DEFINE_OPERAND(loom_vector_fmai_a, 0)
LOOM_DEFINE_OPERAND(loom_vector_fmai_b, 1)
LOOM_DEFINE_OPERAND(loom_vector_fmai_c, 2)
LOOM_DEFINE_RESULT(loom_vector_fmai_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_fmai_overflow)
iree_status_t loom_vector_fmai_build(
    loom_builder_t* builder,
    uint8_t instance_flags,
    loom_value_id_t a,
    loom_value_id_t b,
    loom_value_id_t c,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_fmai_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ANDI: Lanewise bitwise AND of same-typed integer vector operands.
// vector.andi
LOOM_DEFINE_ISA(loom_vector_andi_isa, LOOM_OP_VECTOR_ANDI)
LOOM_DEFINE_OPERAND(loom_vector_andi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_andi_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_andi_result, 0)
iree_status_t loom_vector_andi_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_andi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ORI: Lanewise bitwise OR of same-typed integer vector operands.
// vector.ori
LOOM_DEFINE_ISA(loom_vector_ori_isa, LOOM_OP_VECTOR_ORI)
LOOM_DEFINE_OPERAND(loom_vector_ori_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_ori_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_ori_result, 0)
iree_status_t loom_vector_ori_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_ori_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_XORI: Lanewise bitwise XOR of same-typed integer vector operands.
// vector.xori
LOOM_DEFINE_ISA(loom_vector_xori_isa, LOOM_OP_VECTOR_XORI)
LOOM_DEFINE_OPERAND(loom_vector_xori_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_xori_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_xori_result, 0)
iree_status_t loom_vector_xori_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_xori_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_SHLI: Lanewise left shift of same-typed integer vector operands.
// vector.shli
LOOM_DEFINE_ISA(loom_vector_shli_isa, LOOM_OP_VECTOR_SHLI)
LOOM_DEFINE_OPERAND(loom_vector_shli_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_shli_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_shli_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_shli_overflow)
iree_status_t loom_vector_shli_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_shli_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_SHRSI: Lanewise arithmetic right shift of same-typed integer vector operands.
// vector.shrsi
LOOM_DEFINE_ISA(loom_vector_shrsi_isa, LOOM_OP_VECTOR_SHRSI)
LOOM_DEFINE_OPERAND(loom_vector_shrsi_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_shrsi_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_shrsi_result, 0)
iree_status_t loom_vector_shrsi_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_shrsi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_SHRUI: Lanewise logical right shift of same-typed integer vector operands.
// vector.shrui
LOOM_DEFINE_ISA(loom_vector_shrui_isa, LOOM_OP_VECTOR_SHRUI)
LOOM_DEFINE_OPERAND(loom_vector_shrui_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_shrui_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_shrui_result, 0)
iree_status_t loom_vector_shrui_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_shrui_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ROTLI: Lanewise left rotate of same-typed integer vector operands.
// vector.rotli
LOOM_DEFINE_ISA(loom_vector_rotli_isa, LOOM_OP_VECTOR_ROTLI)
LOOM_DEFINE_OPERAND(loom_vector_rotli_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_rotli_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_rotli_result, 0)
iree_status_t loom_vector_rotli_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

// LOOM_OP_VECTOR_ROTRI: Lanewise right rotate of same-typed integer vector operands.
// vector.rotri
LOOM_DEFINE_ISA(loom_vector_rotri_isa, LOOM_OP_VECTOR_ROTRI)
LOOM_DEFINE_OPERAND(loom_vector_rotri_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_rotri_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_rotri_result, 0)
iree_status_t loom_vector_rotri_build(
    loom_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

// LOOM_OP_VECTOR_CTLZI: Lanewise count leading zeros over integer lanes.
// vector.ctlzi
LOOM_DEFINE_ISA(loom_vector_ctlzi_isa, LOOM_OP_VECTOR_CTLZI)
LOOM_DEFINE_OPERAND(loom_vector_ctlzi_input, 0)
LOOM_DEFINE_RESULT(loom_vector_ctlzi_result, 0)
iree_status_t loom_vector_ctlzi_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_ctlzi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_CTTZI: Lanewise count trailing zeros over integer lanes.
// vector.cttzi
LOOM_DEFINE_ISA(loom_vector_cttzi_isa, LOOM_OP_VECTOR_CTTZI)
LOOM_DEFINE_OPERAND(loom_vector_cttzi_input, 0)
LOOM_DEFINE_RESULT(loom_vector_cttzi_result, 0)
iree_status_t loom_vector_cttzi_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_cttzi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_CTPOPI: Lanewise population count over integer lanes. Each result lane is the number of set bits in the corresponding input lane and has the same integer element type as the input.
// vector.ctpopi
LOOM_DEFINE_ISA(loom_vector_ctpopi_isa, LOOM_OP_VECTOR_CTPOPI)
LOOM_DEFINE_OPERAND(loom_vector_ctpopi_input, 0)
LOOM_DEFINE_RESULT(loom_vector_ctpopi_result, 0)
iree_status_t loom_vector_ctpopi_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_ctpopi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_EXPF: Lanewise natural exponential e^x.
// vector.expf
LOOM_DEFINE_ISA(loom_vector_expf_isa, LOOM_OP_VECTOR_EXPF)
LOOM_DEFINE_OPERAND(loom_vector_expf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_expf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_expf_fastmath)
iree_status_t loom_vector_expf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_expf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_EXP2F: Lanewise base-2 exponential 2^x.
// vector.exp2f
LOOM_DEFINE_ISA(loom_vector_exp2f_isa, LOOM_OP_VECTOR_EXP2F)
LOOM_DEFINE_OPERAND(loom_vector_exp2f_input, 0)
LOOM_DEFINE_RESULT(loom_vector_exp2f_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_exp2f_fastmath)
iree_status_t loom_vector_exp2f_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_exp2f_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_EXPM1F: Lanewise exp(x)-1, preserving the scalar operation's near-zero numerical semantics.
// vector.expm1f
LOOM_DEFINE_ISA(loom_vector_expm1f_isa, LOOM_OP_VECTOR_EXPM1F)
LOOM_DEFINE_OPERAND(loom_vector_expm1f_input, 0)
LOOM_DEFINE_RESULT(loom_vector_expm1f_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_expm1f_fastmath)
iree_status_t loom_vector_expm1f_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_expm1f_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_LOGF: Lanewise natural logarithm ln(x).
// vector.logf
LOOM_DEFINE_ISA(loom_vector_logf_isa, LOOM_OP_VECTOR_LOGF)
LOOM_DEFINE_OPERAND(loom_vector_logf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_logf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_logf_fastmath)
iree_status_t loom_vector_logf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_logf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_LOG2F: Lanewise base-2 logarithm.
// vector.log2f
LOOM_DEFINE_ISA(loom_vector_log2f_isa, LOOM_OP_VECTOR_LOG2F)
LOOM_DEFINE_OPERAND(loom_vector_log2f_input, 0)
LOOM_DEFINE_RESULT(loom_vector_log2f_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_log2f_fastmath)
iree_status_t loom_vector_log2f_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_log2f_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_LOG10F: Lanewise base-10 logarithm.
// vector.log10f
LOOM_DEFINE_ISA(loom_vector_log10f_isa, LOOM_OP_VECTOR_LOG10F)
LOOM_DEFINE_OPERAND(loom_vector_log10f_input, 0)
LOOM_DEFINE_RESULT(loom_vector_log10f_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_log10f_fastmath)
iree_status_t loom_vector_log10f_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_log10f_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_LOG1PF: Lanewise log(1+x), preserving the scalar operation's near-zero numerical semantics.
// vector.log1pf
LOOM_DEFINE_ISA(loom_vector_log1pf_isa, LOOM_OP_VECTOR_LOG1PF)
LOOM_DEFINE_OPERAND(loom_vector_log1pf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_log1pf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_log1pf_fastmath)
iree_status_t loom_vector_log1pf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_log1pf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_POWF: Lanewise floating-point power lhs^rhs over same-typed vector operands.
// vector.powf
LOOM_DEFINE_ISA(loom_vector_powf_isa, LOOM_OP_VECTOR_POWF)
LOOM_DEFINE_OPERAND(loom_vector_powf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_powf_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_powf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_powf_fastmath)
iree_status_t loom_vector_powf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_powf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_SQRTF: Lanewise floating-point square root. Optional fastmath flags carry the same per-lane floating-point permissions as scalar.sqrtf.
// vector.sqrtf
LOOM_DEFINE_ISA(loom_vector_sqrtf_isa, LOOM_OP_VECTOR_SQRTF)
LOOM_DEFINE_OPERAND(loom_vector_sqrtf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_sqrtf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_sqrtf_fastmath)
iree_status_t loom_vector_sqrtf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_sqrtf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_RSQRTF: Lanewise reciprocal square root 1/sqrt(x).
// vector.rsqrtf
LOOM_DEFINE_ISA(loom_vector_rsqrtf_isa, LOOM_OP_VECTOR_RSQRTF)
LOOM_DEFINE_OPERAND(loom_vector_rsqrtf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_rsqrtf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_rsqrtf_fastmath)
iree_status_t loom_vector_rsqrtf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_rsqrtf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_CBRTF: Lanewise cube root.
// vector.cbrtf
LOOM_DEFINE_ISA(loom_vector_cbrtf_isa, LOOM_OP_VECTOR_CBRTF)
LOOM_DEFINE_OPERAND(loom_vector_cbrtf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_cbrtf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_cbrtf_fastmath)
iree_status_t loom_vector_cbrtf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_cbrtf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_SINF: Lanewise sine.
// vector.sinf
LOOM_DEFINE_ISA(loom_vector_sinf_isa, LOOM_OP_VECTOR_SINF)
LOOM_DEFINE_OPERAND(loom_vector_sinf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_sinf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_sinf_fastmath)
iree_status_t loom_vector_sinf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_sinf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_COSF: Lanewise cosine.
// vector.cosf
LOOM_DEFINE_ISA(loom_vector_cosf_isa, LOOM_OP_VECTOR_COSF)
LOOM_DEFINE_OPERAND(loom_vector_cosf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_cosf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_cosf_fastmath)
iree_status_t loom_vector_cosf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_cosf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_SINTURNSF: Lanewise sine over turns: sin(2*pi*x), where 1.0 is one full revolution.
// vector.sinturnsf
LOOM_DEFINE_ISA(loom_vector_sinturnsf_isa, LOOM_OP_VECTOR_SINTURNSF)
LOOM_DEFINE_OPERAND(loom_vector_sinturnsf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_sinturnsf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_sinturnsf_fastmath)
iree_status_t loom_vector_sinturnsf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_sinturnsf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_COSTURNSF: Lanewise cosine over turns: cos(2*pi*x), where 1.0 is one full revolution.
// vector.costurnsf
LOOM_DEFINE_ISA(loom_vector_costurnsf_isa, LOOM_OP_VECTOR_COSTURNSF)
LOOM_DEFINE_OPERAND(loom_vector_costurnsf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_costurnsf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_costurnsf_fastmath)
iree_status_t loom_vector_costurnsf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_costurnsf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_TANF: Lanewise tangent.
// vector.tanf
LOOM_DEFINE_ISA(loom_vector_tanf_isa, LOOM_OP_VECTOR_TANF)
LOOM_DEFINE_OPERAND(loom_vector_tanf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_tanf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_tanf_fastmath)
iree_status_t loom_vector_tanf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_tanf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ASINF: Lanewise arcsine.
// vector.asinf
LOOM_DEFINE_ISA(loom_vector_asinf_isa, LOOM_OP_VECTOR_ASINF)
LOOM_DEFINE_OPERAND(loom_vector_asinf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_asinf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_asinf_fastmath)
iree_status_t loom_vector_asinf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_asinf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ACOSF: Lanewise arccosine.
// vector.acosf
LOOM_DEFINE_ISA(loom_vector_acosf_isa, LOOM_OP_VECTOR_ACOSF)
LOOM_DEFINE_OPERAND(loom_vector_acosf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_acosf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_acosf_fastmath)
iree_status_t loom_vector_acosf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_acosf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ATANF: Lanewise arctangent.
// vector.atanf
LOOM_DEFINE_ISA(loom_vector_atanf_isa, LOOM_OP_VECTOR_ATANF)
LOOM_DEFINE_OPERAND(loom_vector_atanf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_atanf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_atanf_fastmath)
iree_status_t loom_vector_atanf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_atanf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ATAN2F: Lanewise two-argument arctangent atan2(lhs, rhs) over same-typed vector operands.
// vector.atan2f
LOOM_DEFINE_ISA(loom_vector_atan2f_isa, LOOM_OP_VECTOR_ATAN2F)
LOOM_DEFINE_OPERAND(loom_vector_atan2f_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_atan2f_rhs, 1)
LOOM_DEFINE_RESULT(loom_vector_atan2f_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_atan2f_fastmath)
iree_status_t loom_vector_atan2f_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_atan2f_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_SINHF: Lanewise hyperbolic sine.
// vector.sinhf
LOOM_DEFINE_ISA(loom_vector_sinhf_isa, LOOM_OP_VECTOR_SINHF)
LOOM_DEFINE_OPERAND(loom_vector_sinhf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_sinhf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_sinhf_fastmath)
iree_status_t loom_vector_sinhf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_sinhf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_COSHF: Lanewise hyperbolic cosine.
// vector.coshf
LOOM_DEFINE_ISA(loom_vector_coshf_isa, LOOM_OP_VECTOR_COSHF)
LOOM_DEFINE_OPERAND(loom_vector_coshf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_coshf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_coshf_fastmath)
iree_status_t loom_vector_coshf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_coshf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_TANHF: Lanewise hyperbolic tangent.
// vector.tanhf
LOOM_DEFINE_ISA(loom_vector_tanhf_isa, LOOM_OP_VECTOR_TANHF)
LOOM_DEFINE_OPERAND(loom_vector_tanhf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_tanhf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_tanhf_fastmath)
iree_status_t loom_vector_tanhf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_tanhf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ASINHF: Lanewise inverse hyperbolic sine.
// vector.asinhf
LOOM_DEFINE_ISA(loom_vector_asinhf_isa, LOOM_OP_VECTOR_ASINHF)
LOOM_DEFINE_OPERAND(loom_vector_asinhf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_asinhf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_asinhf_fastmath)
iree_status_t loom_vector_asinhf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_asinhf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ACOSHF: Lanewise inverse hyperbolic cosine.
// vector.acoshf
LOOM_DEFINE_ISA(loom_vector_acoshf_isa, LOOM_OP_VECTOR_ACOSHF)
LOOM_DEFINE_OPERAND(loom_vector_acoshf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_acoshf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_acoshf_fastmath)
iree_status_t loom_vector_acoshf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_acoshf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ATANHF: Lanewise inverse hyperbolic tangent.
// vector.atanhf
LOOM_DEFINE_ISA(loom_vector_atanhf_isa, LOOM_OP_VECTOR_ATANHF)
LOOM_DEFINE_OPERAND(loom_vector_atanhf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_atanhf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_atanhf_fastmath)
iree_status_t loom_vector_atanhf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_atanhf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ERFF: Lanewise error function, used by GeLU-style activations.
// vector.erff
LOOM_DEFINE_ISA(loom_vector_erff_isa, LOOM_OP_VECTOR_ERFF)
LOOM_DEFINE_OPERAND(loom_vector_erff_input, 0)
LOOM_DEFINE_RESULT(loom_vector_erff_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_erff_fastmath)
iree_status_t loom_vector_erff_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_erff_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ERFCF: Lanewise complementary error function 1-erf(x).
// vector.erfcf
LOOM_DEFINE_ISA(loom_vector_erfcf_isa, LOOM_OP_VECTOR_ERFCF)
LOOM_DEFINE_OPERAND(loom_vector_erfcf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_erfcf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_erfcf_fastmath)
iree_status_t loom_vector_erfcf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_erfcf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_LOGISTICF: Lanewise logistic sigmoid 1 / (1 + exp(-x)).
// vector.logisticf
LOOM_DEFINE_ISA(loom_vector_logisticf_isa, LOOM_OP_VECTOR_LOGISTICF)
LOOM_DEFINE_OPERAND(loom_vector_logisticf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_logisticf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_logisticf_fastmath)
iree_status_t loom_vector_logisticf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_logisticf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_SILUF: Lanewise SiLU activation x * logistic(x).
// vector.siluf
LOOM_DEFINE_ISA(loom_vector_siluf_isa, LOOM_OP_VECTOR_SILUF)
LOOM_DEFINE_OPERAND(loom_vector_siluf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_siluf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_siluf_fastmath)
iree_status_t loom_vector_siluf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_siluf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_SOFTPLUSF: Lanewise softplus activation log(1 + exp(x)).
// vector.softplusf
LOOM_DEFINE_ISA(loom_vector_softplusf_isa, LOOM_OP_VECTOR_SOFTPLUSF)
LOOM_DEFINE_OPERAND(loom_vector_softplusf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_softplusf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_softplusf_fastmath)
iree_status_t loom_vector_softplusf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_softplusf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_GELUF: Lanewise GELU activation preserving the chosen formula family. The logistic variant carries its scale as an explicit attribute so importers do not encode approximation identity through arithmetic constants.
// %result = vector.geluf<erf> %input : vector<16xf32>
LOOM_DEFINE_ISA(loom_vector_geluf_isa, LOOM_OP_VECTOR_GELUF)
LOOM_DEFINE_OPERAND(loom_vector_geluf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_geluf_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_geluf_variant, 0, loom_vector_geluf_variant_t)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_geluf_fastmath)
LOOM_DEFINE_ATTR_F64(loom_vector_geluf_scale, 1)
enum loom_vector_geluf_build_flag_bits_e {
  LOOM_VECTOR_GELUF_BUILD_FLAG_HAS_SCALE = 1u << 0,
};
typedef uint32_t loom_vector_geluf_build_flags_t;
iree_status_t loom_vector_geluf_build(
    loom_builder_t* builder,
    loom_vector_geluf_build_flags_t build_flags,
    loom_vector_geluf_variant_t variant,
    uint8_t instance_flags,
    loom_value_id_t input,
    loom_optional double scale,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_geluf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_vector_geluf_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_CEILF: Lanewise round toward positive infinity.
// vector.ceilf
LOOM_DEFINE_ISA(loom_vector_ceilf_isa, LOOM_OP_VECTOR_CEILF)
LOOM_DEFINE_OPERAND(loom_vector_ceilf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_ceilf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_ceilf_fastmath)
iree_status_t loom_vector_ceilf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_ceilf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_FLOORF: Lanewise round toward negative infinity.
// vector.floorf
LOOM_DEFINE_ISA(loom_vector_floorf_isa, LOOM_OP_VECTOR_FLOORF)
LOOM_DEFINE_OPERAND(loom_vector_floorf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_floorf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_floorf_fastmath)
iree_status_t loom_vector_floorf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_floorf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ROUNDF: Lanewise round to nearest, ties away from zero.
// vector.roundf
LOOM_DEFINE_ISA(loom_vector_roundf_isa, LOOM_OP_VECTOR_ROUNDF)
LOOM_DEFINE_OPERAND(loom_vector_roundf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_roundf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_roundf_fastmath)
iree_status_t loom_vector_roundf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_roundf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ROUNDEVENF: Lanewise round to nearest, ties to even.
// vector.roundevenf
LOOM_DEFINE_ISA(loom_vector_roundevenf_isa, LOOM_OP_VECTOR_ROUNDEVENF)
LOOM_DEFINE_OPERAND(loom_vector_roundevenf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_roundevenf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_roundevenf_fastmath)
iree_status_t loom_vector_roundevenf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_roundevenf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_TRUNCF: Lanewise round toward zero.
// vector.truncf
LOOM_DEFINE_ISA(loom_vector_truncf_isa, LOOM_OP_VECTOR_TRUNCF)
LOOM_DEFINE_OPERAND(loom_vector_truncf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_truncf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_truncf_fastmath)
iree_status_t loom_vector_truncf_build(
    loom_builder_t* builder, uint8_t instance_flags,
    loom_value_id_t input, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_truncf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ISNANF: Lanewise floating-point NaN test producing an i1 mask vector.
// vector.isnanf
LOOM_DEFINE_ISA(loom_vector_isnanf_isa, LOOM_OP_VECTOR_ISNANF)
LOOM_DEFINE_OPERAND(loom_vector_isnanf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_isnanf_result, 0)
iree_status_t loom_vector_isnanf_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_isnanf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ISINFF: Lanewise floating-point infinity test producing an i1 mask vector.
// vector.isinff
LOOM_DEFINE_ISA(loom_vector_isinff_isa, LOOM_OP_VECTOR_ISINFF)
LOOM_DEFINE_OPERAND(loom_vector_isinff_input, 0)
LOOM_DEFINE_RESULT(loom_vector_isinff_result, 0)
iree_status_t loom_vector_isinff_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_isinff_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_ISFINITEF: Lanewise floating-point finite test producing an i1 mask vector.
// vector.isfinitef
LOOM_DEFINE_ISA(loom_vector_isfinitef_isa, LOOM_OP_VECTOR_ISFINITEF)
LOOM_DEFINE_OPERAND(loom_vector_isfinitef_input, 0)
LOOM_DEFINE_RESULT(loom_vector_isfinitef_result, 0)
iree_status_t loom_vector_isfinitef_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_isfinitef_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_SIGNF: Lanewise floating-point sign, returning -1.0, 0.0, or 1.0 per lane.
// vector.signf
LOOM_DEFINE_ISA(loom_vector_signf_isa, LOOM_OP_VECTOR_SIGNF)
LOOM_DEFINE_OPERAND(loom_vector_signf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_signf_result, 0)
iree_status_t loom_vector_signf_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_signf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_SIGNI: Lanewise integer sign, returning -1, 0, or 1 per lane.
// vector.signi
LOOM_DEFINE_ISA(loom_vector_signi_isa, LOOM_OP_VECTOR_SIGNI)
LOOM_DEFINE_OPERAND(loom_vector_signi_input, 0)
LOOM_DEFINE_RESULT(loom_vector_signi_result, 0)
iree_status_t loom_vector_signi_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_signi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_EXTF: Lanewise floating-point precision extension. Source and result shapes match exactly; only the floating-point element type widens.
// vector.extf
LOOM_DEFINE_ISA(loom_vector_extf_isa, LOOM_OP_VECTOR_EXTF)
LOOM_DEFINE_OPERAND(loom_vector_extf_input, 0)
LOOM_DEFINE_RESULT(loom_vector_extf_result, 0)
iree_status_t loom_vector_extf_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_extf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_FPTRUNC: Lanewise floating-point precision truncation. Source and result shapes match exactly; only the floating-point element type narrows.
// vector.fptrunc
LOOM_DEFINE_ISA(loom_vector_fptrunc_isa, LOOM_OP_VECTOR_FPTRUNC)
LOOM_DEFINE_OPERAND(loom_vector_fptrunc_input, 0)
LOOM_DEFINE_RESULT(loom_vector_fptrunc_result, 0)
iree_status_t loom_vector_fptrunc_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

// LOOM_OP_VECTOR_EXTSI: Lanewise signed integer extension. Source and result shapes match exactly, and each source lane is sign-extended to the result element width.
// vector.extsi
LOOM_DEFINE_ISA(loom_vector_extsi_isa, LOOM_OP_VECTOR_EXTSI)
LOOM_DEFINE_OPERAND(loom_vector_extsi_input, 0)
LOOM_DEFINE_RESULT(loom_vector_extsi_result, 0)
iree_status_t loom_vector_extsi_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_extsi_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_EXTUI: Lanewise unsigned integer extension. Source and result shapes match exactly, and each source lane is zero-extended to the result element width.
// vector.extui
LOOM_DEFINE_ISA(loom_vector_extui_isa, LOOM_OP_VECTOR_EXTUI)
LOOM_DEFINE_OPERAND(loom_vector_extui_input, 0)
LOOM_DEFINE_RESULT(loom_vector_extui_result, 0)
iree_status_t loom_vector_extui_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

// LOOM_OP_VECTOR_TRUNCI: Lanewise integer truncation. Source and result shapes match exactly, and each lane keeps the low bits required by the result element width.
// vector.trunci
LOOM_DEFINE_ISA(loom_vector_trunci_isa, LOOM_OP_VECTOR_TRUNCI)
LOOM_DEFINE_OPERAND(loom_vector_trunci_input, 0)
LOOM_DEFINE_RESULT(loom_vector_trunci_result, 0)
iree_status_t loom_vector_trunci_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

// LOOM_OP_VECTOR_SITOFP: Lanewise signed integer to floating-point conversion with unchanged shape.
// vector.sitofp
LOOM_DEFINE_ISA(loom_vector_sitofp_isa, LOOM_OP_VECTOR_SITOFP)
LOOM_DEFINE_OPERAND(loom_vector_sitofp_input, 0)
LOOM_DEFINE_RESULT(loom_vector_sitofp_result, 0)
iree_status_t loom_vector_sitofp_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_sitofp_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_UITOFP: Lanewise unsigned integer to floating-point conversion with unchanged shape.
// vector.uitofp
LOOM_DEFINE_ISA(loom_vector_uitofp_isa, LOOM_OP_VECTOR_UITOFP)
LOOM_DEFINE_OPERAND(loom_vector_uitofp_input, 0)
LOOM_DEFINE_RESULT(loom_vector_uitofp_result, 0)
iree_status_t loom_vector_uitofp_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

// LOOM_OP_VECTOR_FPTOSI: Lanewise floating-point to signed integer conversion with unchanged shape.
// vector.fptosi
LOOM_DEFINE_ISA(loom_vector_fptosi_isa, LOOM_OP_VECTOR_FPTOSI)
LOOM_DEFINE_OPERAND(loom_vector_fptosi_input, 0)
LOOM_DEFINE_RESULT(loom_vector_fptosi_result, 0)
iree_status_t loom_vector_fptosi_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

// LOOM_OP_VECTOR_FPTOUI: Lanewise floating-point to unsigned integer conversion with unchanged shape.
// vector.fptoui
LOOM_DEFINE_ISA(loom_vector_fptoui_isa, LOOM_OP_VECTOR_FPTOUI)
LOOM_DEFINE_OPERAND(loom_vector_fptoui_input, 0)
LOOM_DEFINE_RESULT(loom_vector_fptoui_result, 0)
iree_status_t loom_vector_fptoui_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);

// LOOM_OP_VECTOR_BITCAST: Bitwise reinterpretation between vector register types with the same total bit count. No numeric conversion is performed; only the lane shape and element interpretation change.
// %r = vector.bitcast %input : vector<16xf32> to vector<16xi32>
LOOM_DEFINE_ISA(loom_vector_bitcast_isa, LOOM_OP_VECTOR_BITCAST)
LOOM_DEFINE_OPERAND(loom_vector_bitcast_input, 0)
LOOM_DEFINE_RESULT(loom_vector_bitcast_result, 0)
iree_status_t loom_vector_bitcast_build(
    loom_builder_t* builder, loom_value_id_t input,
    loom_type_t input_type, loom_type_t result_type,
    loom_location_id_t location, loom_op_t** out_op);
iree_status_t loom_vector_bitcast_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_BITFIELD_EXTRACTU: Extract one fixed bitfield from each integer source lane and zero-extend it into the corresponding result lane. The bitfield is identified by least-significant-bit offset and width.
// %lo = vector.bitfield.extractu %bytes {offset = 0, width = 4} : vector<16xi8> -> vector<16xi32>
LOOM_DEFINE_ISA(loom_vector_bitfield_extractu_isa, LOOM_OP_VECTOR_BITFIELD_EXTRACTU)
LOOM_DEFINE_OPERAND(loom_vector_bitfield_extractu_source, 0)
LOOM_DEFINE_RESULT(loom_vector_bitfield_extractu_result, 0)
LOOM_DEFINE_ATTR_I64(loom_vector_bitfield_extractu_offset, 0)
LOOM_DEFINE_ATTR_I64(loom_vector_bitfield_extractu_width, 1)
iree_status_t loom_vector_bitfield_extractu_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    int64_t offset,
    int64_t width,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_bitfield_extractu_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_BITFIELD_EXTRACTS: Extract one fixed bitfield from each integer source lane and sign-extend it into the corresponding result lane. The bitfield is identified by least-significant-bit offset and width.
// %signed = vector.bitfield.extracts %bytes {offset = 4, width = 4} : vector<16xi8> -> vector<16xi32>
LOOM_DEFINE_ISA(loom_vector_bitfield_extracts_isa, LOOM_OP_VECTOR_BITFIELD_EXTRACTS)
LOOM_DEFINE_OPERAND(loom_vector_bitfield_extracts_source, 0)
LOOM_DEFINE_RESULT(loom_vector_bitfield_extracts_result, 0)
LOOM_DEFINE_ATTR_I64(loom_vector_bitfield_extracts_offset, 0)
LOOM_DEFINE_ATTR_I64(loom_vector_bitfield_extracts_width, 1)
iree_status_t loom_vector_bitfield_extracts_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    int64_t offset,
    int64_t width,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_bitfield_extracts_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_BITFIELD_INSERT: Insert the low bits of each integer field lane into a fixed bitfield of the corresponding integer base lane. Bits outside the target field are preserved from the base lane.
// %packed = vector.bitfield.insert %lo into %zero {offset = 0, width = 4} : vector<16xi32>, vector<16xi8>
LOOM_DEFINE_ISA(loom_vector_bitfield_insert_isa, LOOM_OP_VECTOR_BITFIELD_INSERT)
LOOM_DEFINE_OPERAND(loom_vector_bitfield_insert_field, 0)
LOOM_DEFINE_OPERAND(loom_vector_bitfield_insert_base, 1)
LOOM_DEFINE_RESULT(loom_vector_bitfield_insert_result, 0)
LOOM_DEFINE_ATTR_I64(loom_vector_bitfield_insert_offset, 0)
LOOM_DEFINE_ATTR_I64(loom_vector_bitfield_insert_width, 1)
iree_status_t loom_vector_bitfield_insert_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t field,
    loom_may_consume loom_value_id_t base,
    int64_t offset,
    int64_t width,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_bitfield_insert_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_BITPACK: Pack the low bits of each integer source lane into a contiguous little-endian bitstream stored in integer result lanes. Source lanes are consumed in logical lane order and width gives the number of bits taken from each source lane.
// %packed = vector.bitpack<4> %codes : vector<32xi8> -> vector<16xi8>
LOOM_DEFINE_ISA(loom_vector_bitpack_isa, LOOM_OP_VECTOR_BITPACK)
LOOM_DEFINE_OPERAND(loom_vector_bitpack_source, 0)
LOOM_DEFINE_RESULT(loom_vector_bitpack_result, 0)
LOOM_DEFINE_ATTR_I64(loom_vector_bitpack_width, 0)
iree_status_t loom_vector_bitpack_build(
    loom_builder_t* builder,
    int64_t width,
    loom_may_consume loom_value_id_t source,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_bitpack_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_BITUNPACKU: Unpack unsigned fixed-width fields from a contiguous little-endian integer bitstream into zero-extended integer result lanes. Result lanes are produced in logical lane order.
// %codes = vector.bitunpacku<4> %packed : vector<16xi8> -> vector<32xi8>
LOOM_DEFINE_ISA(loom_vector_bitunpacku_isa, LOOM_OP_VECTOR_BITUNPACKU)
LOOM_DEFINE_OPERAND(loom_vector_bitunpacku_source, 0)
LOOM_DEFINE_RESULT(loom_vector_bitunpacku_result, 0)
LOOM_DEFINE_ATTR_I64(loom_vector_bitunpacku_width, 0)
iree_status_t loom_vector_bitunpacku_build(
    loom_builder_t* builder,
    int64_t width,
    loom_may_consume loom_value_id_t source,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_bitunpacku_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_BITUNPACKS: Unpack signed fixed-width fields from a contiguous little-endian integer bitstream into sign-extended integer result lanes. Result lanes are produced in logical lane order.
// %deltas = vector.bitunpacks<3> %packed : vector<12xi8> -> vector<32xi8>
LOOM_DEFINE_ISA(loom_vector_bitunpacks_isa, LOOM_OP_VECTOR_BITUNPACKS)
LOOM_DEFINE_OPERAND(loom_vector_bitunpacks_source, 0)
LOOM_DEFINE_RESULT(loom_vector_bitunpacks_result, 0)
LOOM_DEFINE_ATTR_I64(loom_vector_bitunpacks_width, 0)
iree_status_t loom_vector_bitunpacks_build(
    loom_builder_t* builder,
    int64_t width,
    loom_may_consume loom_value_id_t source,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_bitunpacks_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_DOTF: Compute a same-element floating-point dot product with an explicit scalar accumulator. Semantics are equivalent to accumulating scalar.fmaf(lhs_lane, rhs_lane, acc) over lanes in logical lane order; use vector.mulf followed by vector.reduce<addf> when separately rounded products and additions are required. The source vectors must have the same shape and element type, and the init/result scalar type matches that element type. Zero-lane inputs return init. Optional fastmath flags carry the same floating-point permissions as scalar arithmetic, including reassociation of the fused dot terms.
// %r = vector.dotf %lhs, %rhs, %acc : vector<16xf32>, vector<16xf32>, f32
LOOM_DEFINE_ISA(loom_vector_dotf_isa, LOOM_OP_VECTOR_DOTF)
LOOM_DEFINE_OPERAND(loom_vector_dotf_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_dotf_rhs, 1)
LOOM_DEFINE_OPERAND(loom_vector_dotf_init, 2)
LOOM_DEFINE_RESULT(loom_vector_dotf_result, 0)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_dotf_fastmath)
iree_status_t loom_vector_dotf_build(
    loom_builder_t* builder,
    uint8_t instance_flags,
    loom_value_id_t lhs,
    loom_value_id_t rhs,
    loom_value_id_t init,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_dotf_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_DOT2F: Group adjacent two-lane f16 or bf16 products along the last axis and add each two-product fused sum into an f32 accumulator lane. Semantics are equivalent to extending each source lane to f32, then accumulating scalar.fmaf(lhs0_f32, rhs0_f32, acc) followed by scalar.fmaf(lhs1_f32, rhs1_f32, partial) for each result lane. This models AMDGPU fdot2-style widened register dots without making f16 dot accumulation implicit in vector.dotf.
// %r = vector.dot2f %lhs, %rhs, %acc : vector<16xf16>, vector<16xf16>, vector<8xf32>
LOOM_DEFINE_ISA(loom_vector_dot2f_isa, LOOM_OP_VECTOR_DOT2F)
LOOM_DEFINE_OPERAND(loom_vector_dot2f_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_dot2f_rhs, 1)
LOOM_DEFINE_OPERAND(loom_vector_dot2f_acc, 2)
LOOM_DEFINE_RESULT(loom_vector_dot2f_result, 0)
iree_status_t loom_vector_dot2f_build(
    loom_builder_t* builder,
    loom_value_id_t lhs,
    loom_value_id_t rhs,
    loom_value_id_t acc,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_dot2f_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_DOT4I: Group adjacent four-lane i8 products along the last axis and add each four-product sum into an i32 accumulator lane. The signedness template chooses how lhs and rhs i8 lanes are interpreted, matching dp4a/VNNI-style hardware operations.
// %r = vector.dot4i<s8s8> %lhs, %rhs, %acc : vector<16xi8>, vector<16xi8>, vector<4xi32>
LOOM_DEFINE_ISA(loom_vector_dot4i_isa, LOOM_OP_VECTOR_DOT4I)
LOOM_DEFINE_OPERAND(loom_vector_dot4i_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_dot4i_rhs, 1)
LOOM_DEFINE_OPERAND(loom_vector_dot4i_acc, 2)
LOOM_DEFINE_RESULT(loom_vector_dot4i_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_dot4i_kind, 0, loom_vector_dot4i_kind_t)
iree_status_t loom_vector_dot4i_build(
    loom_builder_t* builder,
    loom_vector_dot4i_kind_t kind,
    loom_value_id_t lhs,
    loom_value_id_t rhs,
    loom_value_id_t acc,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_dot4i_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_DOT8I4: Treat each i32 source lane as a little-endian pack of eight 4-bit integer fields, multiply corresponding packed fields using the signedness template, and add the eight-product sum into the matching i32 accumulator lane. This is a packed-storage register dot: use vector.bitpack<4> when starting from unpacked byte lanes. The semantics match AMDGPU sdot8/udot8/sudot8 with clamp disabled.
// %r = vector.dot8i4<s4s4> %lhs, %rhs, %acc : vector<4xi32>
LOOM_DEFINE_ISA(loom_vector_dot8i4_isa, LOOM_OP_VECTOR_DOT8I4)
LOOM_DEFINE_OPERAND(loom_vector_dot8i4_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_dot8i4_rhs, 1)
LOOM_DEFINE_OPERAND(loom_vector_dot8i4_acc, 2)
LOOM_DEFINE_RESULT(loom_vector_dot8i4_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_dot8i4_kind, 0, loom_vector_dot8i4_kind_t)
iree_status_t loom_vector_dot8i4_build(
    loom_builder_t* builder,
    loom_vector_dot8i4_kind_t kind,
    loom_value_id_t lhs,
    loom_value_id_t rhs,
    loom_value_id_t acc,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_dot8i4_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_DOT4F8: Treat each i32 source lane as a little-endian pack of four 8-bit floating-point fields, decode fields according to the fp8/bf8 template, and add the four-product fused sum into the matching f32 accumulator lane. The fp8 spelling names the E4M3 primitive float format and bf8 names the E5M2 primitive float format. This is a packed-storage register dot matching AMDGPU dot4.f32.fp8/bf8 families without requiring unpacked f8 vector source lanes.
// %r = vector.dot4f8<fp8bf8> %lhs, %rhs, %acc : vector<4xi32>, vector<4xf32>
LOOM_DEFINE_ISA(loom_vector_dot4f8_isa, LOOM_OP_VECTOR_DOT4F8)
LOOM_DEFINE_OPERAND(loom_vector_dot4f8_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_dot4f8_rhs, 1)
LOOM_DEFINE_OPERAND(loom_vector_dot4f8_acc, 2)
LOOM_DEFINE_RESULT(loom_vector_dot4f8_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_dot4f8_kind, 0, loom_vector_dot4f8_kind_t)
iree_status_t loom_vector_dot4f8_build(
    loom_builder_t* builder,
    loom_vector_dot4f8_kind_t kind,
    loom_value_id_t lhs,
    loom_value_id_t rhs,
    loom_value_id_t acc,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_dot4f8_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_MMA: Compute a matrix multiply-accumulate over target-shaped vector fragments. The op consumes only the physical lhs, rhs, and init vectors; logical M/N/K shape, fragment role, packed storage schema, scales, codebooks, sparse metadata, and other interpretation data are carried by vector.fragment facts on those operands. Lowering queries those facts to select native matrix instructions or a reference decomposition without baking target-specific witnesses into the MMA syntax.
// %r = vector.mma %lhs, %rhs, %init : vector<8xf16>, vector<8xf16>, vector<8xf32>
LOOM_DEFINE_ISA(loom_vector_mma_isa, LOOM_OP_VECTOR_MMA)
LOOM_DEFINE_OPERAND(loom_vector_mma_lhs, 0)
LOOM_DEFINE_OPERAND(loom_vector_mma_rhs, 1)
LOOM_DEFINE_OPERAND(loom_vector_mma_init, 2)
LOOM_DEFINE_RESULT(loom_vector_mma_result, 0)
iree_status_t loom_vector_mma_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t lhs,
    loom_may_consume loom_value_id_t rhs,
    loom_may_consume loom_value_id_t init,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_mma_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// LOOM_OP_VECTOR_REDUCE: Reduce all lanes of a vector into a scalar accumulator/result using the template combining kind. The init operand and result have the same scalar type, and the combining kind must be valid for the input element type. Optional fastmath flags carry the same floating-point permissions as scalar arithmetic; contraction may fuse producer products into FMA accumulation when the producer permits it too.
// %sum = vector.reduce<addf> %v, %zero : vector<16xf32>, f32
LOOM_DEFINE_ISA(loom_vector_reduce_isa, LOOM_OP_VECTOR_REDUCE)
LOOM_DEFINE_OPERAND(loom_vector_reduce_input, 0)
LOOM_DEFINE_OPERAND(loom_vector_reduce_init, 1)
LOOM_DEFINE_RESULT(loom_vector_reduce_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_reduce_kind, 0, loom_combining_kind_t)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_reduce_fastmath)
iree_status_t loom_vector_reduce_build(
    loom_builder_t* builder,
    loom_combining_kind_t kind,
    uint8_t instance_flags,
    loom_value_id_t input,
    loom_value_id_t init,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_reduce_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_vector_reduce_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_vector_reduce_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_REDUCE_AXES: Reduce the explicit source axes of a vector while preserving the remaining axes in their original order. The init operand and result have the same type: scalar when every source axis is reduced, or a vector whose shape is the source shape with the reduced axes removed. Optional fastmath flags carry the same floating-point permissions as scalar arithmetic.
// %cols = vector.reduce.axes<addf> %src, %init axes [0] : vector<4x8xf32>, vector<8xf32>
LOOM_DEFINE_ISA(loom_vector_reduce_axes_isa, LOOM_OP_VECTOR_REDUCE_AXES)
LOOM_DEFINE_OPERAND(loom_vector_reduce_axes_input, 0)
LOOM_DEFINE_OPERAND(loom_vector_reduce_axes_init, 1)
LOOM_DEFINE_RESULT(loom_vector_reduce_axes_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_reduce_axes_kind, 0, loom_combining_kind_t)
LOOM_DEFINE_INSTANCE_FLAGS(loom_vector_reduce_axes_fastmath)
LOOM_DEFINE_ATTR_I64_ARRAY(loom_vector_reduce_axes_axes, 1)
iree_status_t loom_vector_reduce_axes_build(
    loom_builder_t* builder,
    loom_combining_kind_t kind,
    uint8_t instance_flags,
    loom_may_consume loom_value_id_t input,
    loom_may_consume loom_value_id_t init,
    const int64_t* axes,
    iree_host_size_t axes_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_reduce_axes_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_vector_reduce_axes_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_vector_reduce_axes_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_DECODE: Decode physical encoded vector payload lanes into logical numeric lanes using an explicit encoding<schema> witness. The schema value carries compact representation facts such as element format, block extent, packing order, rounding, and sparsity kind. Bulk or runtime-varying interpretation data such as scales, zero-points, codebook rows, sparse metadata, residual streams, signs, and online amax values stay visible as auxiliary SSA operands instead of being hidden inside the encoding value.
// %values = vector.decode %payload using %schema {scale = %scale : vector<1xf16>} : vector<4xi32>, encoding<schema> -> vector<32xf32>
LOOM_DEFINE_ISA(loom_vector_decode_isa, LOOM_OP_VECTOR_DECODE)
LOOM_DEFINE_OPERAND(loom_vector_decode_payload, 0)
LOOM_DEFINE_OPERAND(loom_vector_decode_schema, 1)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_decode_auxiliary, 2)
LOOM_DEFINE_RESULT(loom_vector_decode_result, 0)
LOOM_DEFINE_ATTR_DICT(loom_vector_decode_auxiliary_names, 0)
iree_status_t loom_vector_decode_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t payload,
    loom_may_consume loom_value_id_t schema,
    loom_may_consume const loom_named_value_t* auxiliary,
    iree_host_size_t auxiliary_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_decode_canonicalize(loom_op_t* op, loom_rewriter_t* rewriter);
iree_status_t loom_vector_decode_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_ENCODE: Encode logical numeric vector lanes into a physical encoded payload using an explicit encoding<schema> witness. This is the inverse boundary to vector.decode for runtime-created encoded data such as KV-cache pages, online quantization records, and target prepack buffers. Rounding, saturation, affine terms, table lookup policy, and sparse/codebook structure are described by schema facts; the actual scale/table/metadata/state values are ordinary auxiliary SSA operands.
// %payload = vector.encode %values using %schema {amax = %amax : vector<1xf32>, scale = %scale : vector<1xf16>} : vector<32xf32>, encoding<schema> -> vector<4xi32>
LOOM_DEFINE_ISA(loom_vector_encode_isa, LOOM_OP_VECTOR_ENCODE)
LOOM_DEFINE_OPERAND(loom_vector_encode_source, 0)
LOOM_DEFINE_OPERAND(loom_vector_encode_schema, 1)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_encode_auxiliary, 2)
LOOM_DEFINE_RESULT(loom_vector_encode_result, 0)
LOOM_DEFINE_ATTR_DICT(loom_vector_encode_auxiliary_names, 0)
iree_status_t loom_vector_encode_build(
    loom_builder_t* builder,
    loom_may_consume loom_value_id_t source,
    loom_may_consume loom_value_id_t schema,
    loom_may_consume const loom_named_value_t* auxiliary,
    iree_host_size_t auxiliary_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_encode_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// LOOM_OP_VECTOR_FRAGMENT: Attach a matrix-fragment interpretation to a physical vector value without changing the physical vector type. The role selects how the two shape operands are interpreted: lhs is [m, k], rhs is [k, n], and init/result are [m, n]. Dense/default fragments need only the data value and shape SSA values. Encoded fragments carry schema and scale/table/sparse metadata values in the keyed using dictionary so bulk runtime data remains ordinary SSA while lowering can consume a compact resolved fragment fact.
// %fragment = vector.fragment<lhs> %payload shape [%m, %k] : vector<4xi32>
LOOM_DEFINE_ISA(loom_vector_fragment_isa, LOOM_OP_VECTOR_FRAGMENT)
LOOM_DEFINE_OPERAND(loom_vector_fragment_data, 0)
LOOM_DEFINE_OPERAND(loom_vector_fragment_rows, 1)
LOOM_DEFINE_OPERAND(loom_vector_fragment_columns, 2)
LOOM_DEFINE_VARIADIC_OPERANDS(loom_vector_fragment_params, 3)
LOOM_DEFINE_RESULT(loom_vector_fragment_result, 0)
LOOM_DEFINE_ATTR_ENUM_TYPED(loom_vector_fragment_role, 0, loom_vector_role_t)
LOOM_DEFINE_ATTR_DICT(loom_vector_fragment_param_names, 1)
iree_status_t loom_vector_fragment_build(
    loom_builder_t* builder,
    loom_vector_role_t role,
    loom_may_consume loom_value_id_t data,
    loom_may_consume loom_value_id_t rows,
    loom_may_consume loom_value_id_t columns,
    loom_may_consume const loom_named_value_t* params,
    iree_host_size_t params_count,
    loom_optional const loom_predicate_t* predicates,
    iree_host_size_t predicates_count,
    loom_type_t result_type,
    loom_location_id_t location,
    loom_op_t** out_op);
iree_status_t loom_vector_fragment_facts(
    loom_fact_context_t* context,
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);
iree_status_t loom_vector_fragment_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

// Returns the vtable array for the vector dialect.
const loom_op_vtable_t* const* loom_vector_dialect_vtables(
    iree_host_size_t* out_count);

// Returns the dense semantic metadata array for the vector dialect.
const loom_op_semantics_t* loom_vector_dialect_op_semantics(
    iree_host_size_t* out_count);

// Returns semantic metadata for a vector op kind, or empty metadata.
loom_op_semantics_t loom_vector_op_semantics(
    loom_op_kind_t kind);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_VECTOR_OPS_H_
