// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/module_emitter.h"

#include <inttypes.h>
#include <stdio.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/attribute.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/atomic.h"
#include "loom/ops/llvmir/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/llvmir/descriptors/descriptors.h"
#include "loom/target/emit/llvmir/builder.h"
#include "loom/target/emit/llvmir/target_env.h"
#include "loom/target/launch.h"
#include "loom/target/registers.h"

#define LOOM_LLVMIR_LOW_EMITTER_KEY IREE_SV("llvmir.low")
#define LOOM_LLVMIR_GENERIC_CORE_DESCRIPTOR_SET_KEY \
  IREE_SV("llvmir.generic.core")
#define LOOM_LLVMIR_MAX_VECTOR_LANES 32

typedef enum loom_llvmir_emit_core_type_e {
  LOOM_LLVMIR_EMIT_CORE_TYPE_I1 = 0,
  LOOM_LLVMIR_EMIT_CORE_TYPE_I8 = 1,
  LOOM_LLVMIR_EMIT_CORE_TYPE_I16 = 2,
  LOOM_LLVMIR_EMIT_CORE_TYPE_I32 = 3,
  LOOM_LLVMIR_EMIT_CORE_TYPE_I64 = 4,
  LOOM_LLVMIR_EMIT_CORE_TYPE_F16 = 5,
  LOOM_LLVMIR_EMIT_CORE_TYPE_BF16 = 6,
  LOOM_LLVMIR_EMIT_CORE_TYPE_F32 = 7,
  LOOM_LLVMIR_EMIT_CORE_TYPE_F64 = 8,
  LOOM_LLVMIR_EMIT_CORE_TYPE_PTR = 9,
} loom_llvmir_emit_core_type_t;

typedef enum loom_llvmir_emit_memory_flag_bits_e {
  LOOM_LLVMIR_EMIT_MEMORY_FLAG_LOAD = 1u << 0,
  LOOM_LLVMIR_EMIT_MEMORY_FLAG_INDEXED = 1u << 1,
} loom_llvmir_emit_memory_flag_bits_t;
typedef uint8_t loom_llvmir_emit_memory_flags_t;

typedef enum loom_llvmir_emit_atomic_flag_bits_e {
  LOOM_LLVMIR_EMIT_ATOMIC_FLAG_RETURN_VALUE = 1u << 0,
  LOOM_LLVMIR_EMIT_ATOMIC_FLAG_INDEXED = 1u << 1,
  LOOM_LLVMIR_EMIT_ATOMIC_FLAG_CMPXCHG = 1u << 2,
} loom_llvmir_emit_atomic_flag_bits_t;
typedef uint8_t loom_llvmir_emit_atomic_flags_t;

typedef enum loom_llvmir_emit_const_kind_e {
  LOOM_LLVMIR_EMIT_CONST_KIND_INTEGER = 0,
  LOOM_LLVMIR_EMIT_CONST_KIND_FLOAT_BITS = 1,
} loom_llvmir_emit_const_kind_t;

typedef enum loom_llvmir_emit_shuffle_kind_e {
  LOOM_LLVMIR_EMIT_SHUFFLE_KIND_EXPLICIT = 0,
  LOOM_LLVMIR_EMIT_SHUFFLE_KIND_SLICE = 1,
} loom_llvmir_emit_shuffle_kind_t;

typedef enum loom_llvmir_emit_kernel_query_kind_e {
  LOOM_LLVMIR_EMIT_KERNEL_QUERY_WORKITEM_ID = 0,
  LOOM_LLVMIR_EMIT_KERNEL_QUERY_WORKGROUP_ID = 1,
  LOOM_LLVMIR_EMIT_KERNEL_QUERY_WORKGROUP_SIZE = 2,
  LOOM_LLVMIR_EMIT_KERNEL_QUERY_WORKITEM_DISPATCH_ID = 3,
  LOOM_LLVMIR_EMIT_KERNEL_QUERY_SUBGROUP_SIZE = 4,
  LOOM_LLVMIR_EMIT_KERNEL_QUERY_SUBGROUP_COUNT = 5,
} loom_llvmir_emit_kernel_query_kind_t;

typedef enum loom_llvmir_emit_dimension_e {
  LOOM_LLVMIR_EMIT_DIMENSION_X = 0,
  LOOM_LLVMIR_EMIT_DIMENSION_Y = 1,
  LOOM_LLVMIR_EMIT_DIMENSION_Z = 2,
} loom_llvmir_emit_dimension_t;

typedef struct loom_llvmir_emit_binary_info_t {
  // Generated descriptor reference ordinal.
  uint32_t descriptor_ref;
  // LLVMIR builder binary opcode.
  loom_llvmir_binop_t binop;
} loom_llvmir_emit_binary_info_t;

typedef struct loom_llvmir_emit_binary_intrinsic_info_t {
  // Generated descriptor reference ordinal.
  uint32_t descriptor_ref;
  // LLVM intrinsic symbol name.
  iree_string_view_t intrinsic_name;
} loom_llvmir_emit_binary_intrinsic_info_t;

typedef struct loom_llvmir_emit_unary_info_t {
  // Generated descriptor reference ordinal.
  uint32_t descriptor_ref;
  // LLVMIR builder unary opcode.
  loom_llvmir_unop_t unop;
} loom_llvmir_emit_unary_info_t;

typedef struct loom_llvmir_emit_unary_intrinsic_info_t {
  // Generated descriptor reference ordinal.
  uint32_t descriptor_ref;
  // LLVM intrinsic symbol name.
  iree_string_view_t intrinsic_name;
} loom_llvmir_emit_unary_intrinsic_info_t;

typedef struct loom_llvmir_emit_ternary_intrinsic_info_t {
  // Generated descriptor reference ordinal.
  uint32_t descriptor_ref;
  // LLVM intrinsic symbol name.
  iree_string_view_t intrinsic_name;
} loom_llvmir_emit_ternary_intrinsic_info_t;

typedef struct loom_llvmir_emit_const_info_t {
  // Generated descriptor reference ordinal.
  uint32_t descriptor_ref;
  // Scalar value type stored by the constant.
  loom_llvmir_emit_core_type_t value_type;
  // Number of scalar units in the constant result.
  uint32_t unit_count;
  // Immediate payload interpretation for the constant.
  loom_llvmir_emit_const_kind_t kind;
} loom_llvmir_emit_const_info_t;

typedef struct loom_llvmir_emit_concat_info_t {
  // Generated descriptor reference ordinal.
  uint32_t descriptor_ref;
  // Number of vector operands concatenated into the result.
  uint32_t input_count;
} loom_llvmir_emit_concat_info_t;

typedef struct loom_llvmir_emit_compare_info_t {
  // Generated descriptor reference ordinal.
  uint32_t descriptor_ref;
  // True when |predicate| stores an LLVM floating-point comparison predicate.
  bool is_float;
  // LLVM comparison predicate stored as loom_llvmir_icmp_predicate_t or
  // loom_llvmir_fcmp_predicate_t according to |is_float|.
  uint8_t predicate;
} loom_llvmir_emit_compare_info_t;

typedef struct loom_llvmir_emit_cast_info_t {
  // Generated descriptor reference ordinal.
  uint32_t descriptor_ref;
  // LLVMIR builder cast opcode.
  loom_llvmir_cast_op_t cast_op;
} loom_llvmir_emit_cast_info_t;

typedef struct loom_llvmir_emit_memory_info_t {
  // Generated descriptor reference ordinal.
  uint32_t descriptor_ref;
  // Loaded or stored value type family.
  loom_llvmir_emit_core_type_t value_type;
  // Number of vector lanes in the loaded or stored value.
  uint32_t unit_count;
  // Memory operation flags selecting load/store and indexed addressing.
  loom_llvmir_emit_memory_flags_t flags;
} loom_llvmir_emit_memory_info_t;

typedef struct loom_llvmir_emit_atomic_info_t {
  // Generated descriptor reference ordinal.
  uint32_t descriptor_ref;
  // Scalar value type used by the atomic operation.
  loom_llvmir_emit_core_type_t value_type;
  // LLVMIR builder atomic read-modify-write opcode for non-cmpxchg forms.
  loom_llvmir_atomic_rmw_op_t op;
  // Atomic operation flags selecting result and indexed addressing forms.
  loom_llvmir_emit_atomic_flags_t flags;
} loom_llvmir_emit_atomic_info_t;

typedef struct loom_llvmir_emit_alloca_info_t {
  // Generated descriptor reference ordinal.
  uint32_t descriptor_ref;
  // Source scratch memory space represented by the descriptor.
  loom_value_fact_memory_space_t memory_space;
} loom_llvmir_emit_alloca_info_t;

typedef struct loom_llvmir_emit_kernel_query_info_t {
  // Generated descriptor reference ordinal.
  uint32_t descriptor_ref;
  // Logical kernel query represented by the descriptor.
  loom_llvmir_emit_kernel_query_kind_t kind;
  // Query coordinate dimension.
  loom_llvmir_emit_dimension_t dimension;
} loom_llvmir_emit_kernel_query_info_t;

typedef struct loom_llvmir_emit_module_state_t {
  // Module containing the emitted low functions.
  loom_module_t* module;
  // Low descriptor registry used to resolve target-bound packets.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Runtime-selected target overlay, or empty for source-selected targets.
  loom_target_selection_t target_selection;
  // Structured diagnostic emitter for user IR failures.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // Case/module scratch arena.
  iree_arena_allocator_t* scratch_arena;
  // Optional registry of linked target profiles for kernel projection.
  const loom_llvmir_target_profile_registry_t* target_profile_registry;
  // Cached symbol facts shared by target resolution for every function.
  loom_symbol_fact_table_t symbol_facts;
  // Structured LLVMIR module being built.
  loom_llvmir_module_t* llvmir_module;
  // Number of low functions emitted into |llvmir_module|.
  iree_host_size_t function_count;
  // Number of error diagnostics emitted during semantic emission.
  iree_host_size_t error_count;
} loom_llvmir_emit_module_state_t;

typedef struct loom_llvmir_emit_function_state_t {
  // Module-emission state shared by every emitted function.
  loom_llvmir_emit_module_state_t* module_state;
  // Module containing the emitted low function.
  loom_module_t* module;
  // Target-low function definition being emitted.
  loom_op_t* function_op;
  // Target-low function body being emitted.
  const loom_region_t* body;
  // Resolved target record and descriptor set for |function_op|.
  const loom_low_resolved_target_t* target;
  // Function-local target profile storage derived from |target|.
  loom_llvmir_target_profile_storage_t target_profile_storage;
  // Function-local LLVMIR target profile derived from |target_profile_storage|.
  const loom_llvmir_target_profile_t* target_profile;
  // Function symbol name used in diagnostics.
  iree_string_view_t function_name;
  // Case/function scratch arena.
  iree_arena_allocator_t* scratch_arena;
  // Structured LLVMIR module being built.
  loom_llvmir_module_t* llvmir_module;
  // Structured LLVMIR function being built.
  loom_llvmir_function_t* llvmir_function;
  // Structured LLVMIR block currently receiving instructions.
  loom_llvmir_block_t* llvmir_block;
  // Function-local Loom value to LLVMIR value table.
  loom_llvmir_value_id_t* value_map;
  // Number of entries in |value_map|.
  iree_host_size_t value_map_count;
  // Number of LLVMIR return values expected by this function.
  uint16_t result_count;
} loom_llvmir_emit_function_state_t;

#define LOOM_LLVMIR_BINARY_INFO(suffix, op) \
  {LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_##suffix, LOOM_LLVMIR_BINOP_##op}

#define LOOM_LLVMIR_MASK_BINARY_INFOS(suffix)   \
  LOOM_LLVMIR_BINARY_INFO(AND_##suffix, AND),   \
      LOOM_LLVMIR_BINARY_INFO(OR_##suffix, OR), \
      LOOM_LLVMIR_BINARY_INFO(XOR_##suffix, XOR)

#define LOOM_LLVMIR_SHIFT_BINARY_INFOS(suffix)      \
  LOOM_LLVMIR_BINARY_INFO(SHL_##suffix, SHL),       \
      LOOM_LLVMIR_BINARY_INFO(LSHR_##suffix, LSHR), \
      LOOM_LLVMIR_BINARY_INFO(ASHR_##suffix, ASHR)

#define LOOM_LLVMIR_BITWISE_BINARY_INFOS(suffix) \
  LOOM_LLVMIR_MASK_BINARY_INFOS(suffix), LOOM_LLVMIR_SHIFT_BINARY_INFOS(suffix)

#define LOOM_LLVMIR_INTEGER_BASE_BINARY_INFOS(suffix) \
  LOOM_LLVMIR_BINARY_INFO(ADD_##suffix, ADD),         \
      LOOM_LLVMIR_BINARY_INFO(SUB_##suffix, SUB),     \
      LOOM_LLVMIR_BINARY_INFO(MUL_##suffix, MUL),     \
      LOOM_LLVMIR_BITWISE_BINARY_INFOS(suffix)

#define LOOM_LLVMIR_INTEGER_BINARY_INFOS(suffix)    \
  LOOM_LLVMIR_INTEGER_BASE_BINARY_INFOS(suffix),    \
      LOOM_LLVMIR_BINARY_INFO(UDIV_##suffix, UDIV), \
      LOOM_LLVMIR_BINARY_INFO(SDIV_##suffix, SDIV), \
      LOOM_LLVMIR_BINARY_INFO(UREM_##suffix, UREM), \
      LOOM_LLVMIR_BINARY_INFO(SREM_##suffix, SREM)

#define LOOM_LLVMIR_FLOAT_BINARY_INFOS(suffix)     \
  LOOM_LLVMIR_BINARY_INFO(ADD_##suffix, FADD),     \
      LOOM_LLVMIR_BINARY_INFO(SUB_##suffix, FSUB), \
      LOOM_LLVMIR_BINARY_INFO(MUL_##suffix, FMUL), \
      LOOM_LLVMIR_BINARY_INFO(DIV_##suffix, FDIV)

#define LOOM_LLVMIR_VECTOR_BINARY_INFOS(lanes)              \
  LOOM_LLVMIR_MASK_BINARY_INFOS(V##lanes##I1),              \
      LOOM_LLVMIR_BITWISE_BINARY_INFOS(V##lanes##I8),       \
      LOOM_LLVMIR_BITWISE_BINARY_INFOS(V##lanes##I16),      \
      LOOM_LLVMIR_INTEGER_BASE_BINARY_INFOS(V##lanes##I32), \
      LOOM_LLVMIR_BINARY_INFO(ADD_V##lanes##F32, FADD),     \
      LOOM_LLVMIR_BINARY_INFO(SUB_V##lanes##F32, FSUB),     \
      LOOM_LLVMIR_BINARY_INFO(MUL_V##lanes##F32, FMUL),     \
      LOOM_LLVMIR_BINARY_INFO(DIV_V##lanes##F32, FDIV)

static const loom_llvmir_emit_binary_info_t kBinaryInfos[] = {
    LOOM_LLVMIR_MASK_BINARY_INFOS(I1),
    LOOM_LLVMIR_BITWISE_BINARY_INFOS(I8),
    LOOM_LLVMIR_BITWISE_BINARY_INFOS(I16),
    LOOM_LLVMIR_INTEGER_BINARY_INFOS(I32),
    LOOM_LLVMIR_INTEGER_BINARY_INFOS(I64),
    LOOM_LLVMIR_FLOAT_BINARY_INFOS(F32),
    LOOM_LLVMIR_FLOAT_BINARY_INFOS(F64),
    LOOM_LLVMIR_VECTOR_BINARY_INFOS(2),
    LOOM_LLVMIR_VECTOR_BINARY_INFOS(3),
    LOOM_LLVMIR_VECTOR_BINARY_INFOS(4),
    LOOM_LLVMIR_VECTOR_BINARY_INFOS(8),
    LOOM_LLVMIR_VECTOR_BINARY_INFOS(16),
};

#undef LOOM_LLVMIR_VECTOR_BINARY_INFOS
#undef LOOM_LLVMIR_FLOAT_BINARY_INFOS
#undef LOOM_LLVMIR_INTEGER_BINARY_INFOS
#undef LOOM_LLVMIR_INTEGER_BASE_BINARY_INFOS
#undef LOOM_LLVMIR_BITWISE_BINARY_INFOS
#undef LOOM_LLVMIR_SHIFT_BINARY_INFOS
#undef LOOM_LLVMIR_MASK_BINARY_INFOS
#undef LOOM_LLVMIR_BINARY_INFO

#define LOOM_LLVMIR_BINARY_INTRINSIC_INFO(suffix, name) \
  {LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_##suffix, IREE_SVL(name)}

#define LOOM_LLVMIR_MINNUM_INFO(suffix, name) \
  LOOM_LLVMIR_BINARY_INTRINSIC_INFO(MINNUM_##suffix, "llvm.minnum." name)

#define LOOM_LLVMIR_MAXNUM_INFO(suffix, name) \
  LOOM_LLVMIR_BINARY_INTRINSIC_INFO(MAXNUM_##suffix, "llvm.maxnum." name)

static const loom_llvmir_emit_binary_intrinsic_info_t kBinaryIntrinsicInfos[] =
    {
        LOOM_LLVMIR_MINNUM_INFO(F32, "f32"),
        LOOM_LLVMIR_MINNUM_INFO(F64, "f64"),
        LOOM_LLVMIR_MINNUM_INFO(V2F32, "v2f32"),
        LOOM_LLVMIR_MINNUM_INFO(V3F32, "v3f32"),
        LOOM_LLVMIR_MINNUM_INFO(V4F32, "v4f32"),
        LOOM_LLVMIR_MINNUM_INFO(V8F32, "v8f32"),
        LOOM_LLVMIR_MINNUM_INFO(V16F32, "v16f32"),
        LOOM_LLVMIR_MAXNUM_INFO(F32, "f32"),
        LOOM_LLVMIR_MAXNUM_INFO(F64, "f64"),
        LOOM_LLVMIR_MAXNUM_INFO(V2F32, "v2f32"),
        LOOM_LLVMIR_MAXNUM_INFO(V3F32, "v3f32"),
        LOOM_LLVMIR_MAXNUM_INFO(V4F32, "v4f32"),
        LOOM_LLVMIR_MAXNUM_INFO(V8F32, "v8f32"),
        LOOM_LLVMIR_MAXNUM_INFO(V16F32, "v16f32"),
};

#undef LOOM_LLVMIR_MAXNUM_INFO
#undef LOOM_LLVMIR_MINNUM_INFO
#undef LOOM_LLVMIR_BINARY_INTRINSIC_INFO

#define LOOM_LLVMIR_UNARY_INFO(suffix, op) \
  {LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_##suffix, LOOM_LLVMIR_UNOP_##op}

#define LOOM_LLVMIR_NEG_INFO(suffix) LOOM_LLVMIR_UNARY_INFO(NEG_##suffix, FNEG)

static const loom_llvmir_emit_unary_info_t kUnaryInfos[] = {
    LOOM_LLVMIR_NEG_INFO(F32),    LOOM_LLVMIR_NEG_INFO(F64),
    LOOM_LLVMIR_NEG_INFO(V2F32),  LOOM_LLVMIR_NEG_INFO(V3F32),
    LOOM_LLVMIR_NEG_INFO(V4F32),  LOOM_LLVMIR_NEG_INFO(V8F32),
    LOOM_LLVMIR_NEG_INFO(V16F32),
};

#undef LOOM_LLVMIR_NEG_INFO
#undef LOOM_LLVMIR_UNARY_INFO

#define LOOM_LLVMIR_UNARY_INTRINSIC_INFO(suffix, name) \
  {LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_##suffix, IREE_SVL(name)}

#define LOOM_LLVMIR_FABS_INFO(suffix, name) \
  LOOM_LLVMIR_UNARY_INTRINSIC_INFO(ABS_##suffix, "llvm.fabs." name)

static const loom_llvmir_emit_unary_intrinsic_info_t kUnaryIntrinsicInfos[] = {
    LOOM_LLVMIR_FABS_INFO(F32, "f32"),
    LOOM_LLVMIR_FABS_INFO(F64, "f64"),
    LOOM_LLVMIR_FABS_INFO(V2F32, "v2f32"),
    LOOM_LLVMIR_FABS_INFO(V3F32, "v3f32"),
    LOOM_LLVMIR_FABS_INFO(V4F32, "v4f32"),
    LOOM_LLVMIR_FABS_INFO(V8F32, "v8f32"),
    LOOM_LLVMIR_FABS_INFO(V16F32, "v16f32"),
};

#undef LOOM_LLVMIR_FABS_INFO
#undef LOOM_LLVMIR_UNARY_INTRINSIC_INFO

#define LOOM_LLVMIR_TERNARY_INTRINSIC_INFO(suffix, name) \
  {LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_##suffix, IREE_SVL(name)}

#define LOOM_LLVMIR_FMA_INFO(suffix, name) \
  LOOM_LLVMIR_TERNARY_INTRINSIC_INFO(FMA_##suffix, "llvm.fma." name)

static const loom_llvmir_emit_ternary_intrinsic_info_t
    kTernaryIntrinsicInfos[] = {
        LOOM_LLVMIR_FMA_INFO(F32, "f32"),
        LOOM_LLVMIR_FMA_INFO(F64, "f64"),
        LOOM_LLVMIR_FMA_INFO(V2F32, "v2f32"),
        LOOM_LLVMIR_FMA_INFO(V3F32, "v3f32"),
        LOOM_LLVMIR_FMA_INFO(V4F32, "v4f32"),
        LOOM_LLVMIR_FMA_INFO(V8F32, "v8f32"),
        LOOM_LLVMIR_FMA_INFO(V16F32, "v16f32"),
};

#undef LOOM_LLVMIR_FMA_INFO
#undef LOOM_LLVMIR_TERNARY_INTRINSIC_INFO

#define LOOM_LLVMIR_ICMP_INFO(suffix, predicate)                         \
  {LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_CMP_##predicate##_##suffix, false, \
   LOOM_LLVMIR_ICMP_##predicate}

#define LOOM_LLVMIR_FCMP_INFO(suffix, predicate)                        \
  {LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_CMP_##predicate##_##suffix, true, \
   LOOM_LLVMIR_FCMP_##predicate}

#define LOOM_LLVMIR_ICMP_INFOS(suffix)                                        \
  LOOM_LLVMIR_ICMP_INFO(suffix, EQ), LOOM_LLVMIR_ICMP_INFO(suffix, NE),       \
      LOOM_LLVMIR_ICMP_INFO(suffix, SLT), LOOM_LLVMIR_ICMP_INFO(suffix, SLE), \
      LOOM_LLVMIR_ICMP_INFO(suffix, SGT), LOOM_LLVMIR_ICMP_INFO(suffix, SGE), \
      LOOM_LLVMIR_ICMP_INFO(suffix, ULT), LOOM_LLVMIR_ICMP_INFO(suffix, ULE), \
      LOOM_LLVMIR_ICMP_INFO(suffix, UGT), LOOM_LLVMIR_ICMP_INFO(suffix, UGE)

#define LOOM_LLVMIR_FCMP_INFOS(suffix)                                        \
  LOOM_LLVMIR_FCMP_INFO(suffix, OEQ), LOOM_LLVMIR_FCMP_INFO(suffix, OGT),     \
      LOOM_LLVMIR_FCMP_INFO(suffix, OGE), LOOM_LLVMIR_FCMP_INFO(suffix, OLT), \
      LOOM_LLVMIR_FCMP_INFO(suffix, OLE), LOOM_LLVMIR_FCMP_INFO(suffix, ONE), \
      LOOM_LLVMIR_FCMP_INFO(suffix, ORD), LOOM_LLVMIR_FCMP_INFO(suffix, UEQ), \
      LOOM_LLVMIR_FCMP_INFO(suffix, UGT), LOOM_LLVMIR_FCMP_INFO(suffix, UGE), \
      LOOM_LLVMIR_FCMP_INFO(suffix, ULT), LOOM_LLVMIR_FCMP_INFO(suffix, ULE), \
      LOOM_LLVMIR_FCMP_INFO(suffix, UNE), LOOM_LLVMIR_FCMP_INFO(suffix, UNO)

#define LOOM_LLVMIR_VECTOR_COMPARE_INFOS(lanes) \
  LOOM_LLVMIR_ICMP_INFOS(V##lanes##I32), LOOM_LLVMIR_FCMP_INFOS(V##lanes##F32)

static const loom_llvmir_emit_compare_info_t kCompareInfos[] = {
    LOOM_LLVMIR_ICMP_INFOS(I32),          LOOM_LLVMIR_ICMP_INFOS(I64),
    LOOM_LLVMIR_FCMP_INFOS(F32),          LOOM_LLVMIR_FCMP_INFOS(F64),
    LOOM_LLVMIR_VECTOR_COMPARE_INFOS(2),  LOOM_LLVMIR_VECTOR_COMPARE_INFOS(3),
    LOOM_LLVMIR_VECTOR_COMPARE_INFOS(4),  LOOM_LLVMIR_VECTOR_COMPARE_INFOS(8),
    LOOM_LLVMIR_VECTOR_COMPARE_INFOS(16),
};

#undef LOOM_LLVMIR_VECTOR_COMPARE_INFOS
#undef LOOM_LLVMIR_FCMP_INFOS
#undef LOOM_LLVMIR_ICMP_INFOS
#undef LOOM_LLVMIR_FCMP_INFO
#undef LOOM_LLVMIR_ICMP_INFO

#define LOOM_LLVMIR_CAST_INFO(stem, source, result, op) \
  {LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_##stem##_##source##_##result, op}

#define LOOM_LLVMIR_SCALAR_CAST_INFOS()                                        \
  LOOM_LLVMIR_CAST_INFO(TRUNC, I32, I8, LOOM_LLVMIR_CAST_TRUNCATE),            \
      LOOM_LLVMIR_CAST_INFO(TRUNC, I32, I16, LOOM_LLVMIR_CAST_TRUNCATE),       \
      LOOM_LLVMIR_CAST_INFO(TRUNC, I64, I32, LOOM_LLVMIR_CAST_TRUNCATE),       \
      LOOM_LLVMIR_CAST_INFO(SEXT, I8, I32, LOOM_LLVMIR_CAST_SIGN_EXTEND),      \
      LOOM_LLVMIR_CAST_INFO(SEXT, I16, I32, LOOM_LLVMIR_CAST_SIGN_EXTEND),     \
      LOOM_LLVMIR_CAST_INFO(SEXT, I32, I64, LOOM_LLVMIR_CAST_SIGN_EXTEND),     \
      LOOM_LLVMIR_CAST_INFO(ZEXT, I8, I32, LOOM_LLVMIR_CAST_ZERO_EXTEND),      \
      LOOM_LLVMIR_CAST_INFO(ZEXT, I16, I32, LOOM_LLVMIR_CAST_ZERO_EXTEND),     \
      LOOM_LLVMIR_CAST_INFO(ZEXT, I32, I64, LOOM_LLVMIR_CAST_ZERO_EXTEND),     \
      LOOM_LLVMIR_CAST_INFO(SITOFP, I8, F32,                                   \
                            LOOM_LLVMIR_CAST_SIGNED_INT_TO_FP),                \
      LOOM_LLVMIR_CAST_INFO(SITOFP, I32, F32,                                  \
                            LOOM_LLVMIR_CAST_SIGNED_INT_TO_FP),                \
      LOOM_LLVMIR_CAST_INFO(SITOFP, I64, F64,                                  \
                            LOOM_LLVMIR_CAST_SIGNED_INT_TO_FP),                \
      LOOM_LLVMIR_CAST_INFO(UITOFP, I8, F32,                                   \
                            LOOM_LLVMIR_CAST_UNSIGNED_INT_TO_FP),              \
      LOOM_LLVMIR_CAST_INFO(UITOFP, I32, F32,                                  \
                            LOOM_LLVMIR_CAST_UNSIGNED_INT_TO_FP),              \
      LOOM_LLVMIR_CAST_INFO(UITOFP, I64, F64,                                  \
                            LOOM_LLVMIR_CAST_UNSIGNED_INT_TO_FP),              \
      LOOM_LLVMIR_CAST_INFO(FPTOSI, F32, I32,                                  \
                            LOOM_LLVMIR_CAST_FP_TO_SIGNED_INT),                \
      LOOM_LLVMIR_CAST_INFO(FPTOSI, F64, I64,                                  \
                            LOOM_LLVMIR_CAST_FP_TO_SIGNED_INT),                \
      LOOM_LLVMIR_CAST_INFO(FPTOUI, F32, I32,                                  \
                            LOOM_LLVMIR_CAST_FP_TO_UNSIGNED_INT),              \
      LOOM_LLVMIR_CAST_INFO(FPTOUI, F64, I64,                                  \
                            LOOM_LLVMIR_CAST_FP_TO_UNSIGNED_INT),              \
      LOOM_LLVMIR_CAST_INFO(FPTRUNC, F32, F16, LOOM_LLVMIR_CAST_FP_TRUNCATE),  \
      LOOM_LLVMIR_CAST_INFO(FPTRUNC, F32, BF16, LOOM_LLVMIR_CAST_FP_TRUNCATE), \
      LOOM_LLVMIR_CAST_INFO(FPTRUNC, F64, F32, LOOM_LLVMIR_CAST_FP_TRUNCATE),  \
      LOOM_LLVMIR_CAST_INFO(FPEXT, F16, F32, LOOM_LLVMIR_CAST_FP_EXTEND),      \
      LOOM_LLVMIR_CAST_INFO(FPEXT, BF16, F32, LOOM_LLVMIR_CAST_FP_EXTEND),     \
      LOOM_LLVMIR_CAST_INFO(FPEXT, F32, F64, LOOM_LLVMIR_CAST_FP_EXTEND),      \
      LOOM_LLVMIR_CAST_INFO(BITCAST, I16, F16, LOOM_LLVMIR_CAST_BITCAST),      \
      LOOM_LLVMIR_CAST_INFO(BITCAST, I16, BF16, LOOM_LLVMIR_CAST_BITCAST),     \
      LOOM_LLVMIR_CAST_INFO(BITCAST, F16, I16, LOOM_LLVMIR_CAST_BITCAST),      \
      LOOM_LLVMIR_CAST_INFO(BITCAST, BF16, I16, LOOM_LLVMIR_CAST_BITCAST),     \
      LOOM_LLVMIR_CAST_INFO(BITCAST, F16, BF16, LOOM_LLVMIR_CAST_BITCAST),     \
      LOOM_LLVMIR_CAST_INFO(BITCAST, BF16, F16, LOOM_LLVMIR_CAST_BITCAST),     \
      LOOM_LLVMIR_CAST_INFO(BITCAST, I32, F32, LOOM_LLVMIR_CAST_BITCAST),      \
      LOOM_LLVMIR_CAST_INFO(BITCAST, F32, I32, LOOM_LLVMIR_CAST_BITCAST),      \
      LOOM_LLVMIR_CAST_INFO(BITCAST, I64, F64, LOOM_LLVMIR_CAST_BITCAST),      \
      LOOM_LLVMIR_CAST_INFO(BITCAST, F64, I64, LOOM_LLVMIR_CAST_BITCAST)

#define LOOM_LLVMIR_VECTOR_CAST_INFOS(lanes)                        \
  LOOM_LLVMIR_CAST_INFO(SEXT, V##lanes##I32, V##lanes##I64,         \
                        LOOM_LLVMIR_CAST_SIGN_EXTEND),              \
      LOOM_LLVMIR_CAST_INFO(ZEXT, V##lanes##I32, V##lanes##I64,     \
                            LOOM_LLVMIR_CAST_ZERO_EXTEND),          \
      LOOM_LLVMIR_CAST_INFO(TRUNC, V##lanes##I64, V##lanes##I32,    \
                            LOOM_LLVMIR_CAST_TRUNCATE),             \
      LOOM_LLVMIR_CAST_INFO(SITOFP, V##lanes##I32, V##lanes##F32,   \
                            LOOM_LLVMIR_CAST_SIGNED_INT_TO_FP),     \
      LOOM_LLVMIR_CAST_INFO(UITOFP, V##lanes##I32, V##lanes##F32,   \
                            LOOM_LLVMIR_CAST_UNSIGNED_INT_TO_FP),   \
      LOOM_LLVMIR_CAST_INFO(FPTOSI, V##lanes##F32, V##lanes##I32,   \
                            LOOM_LLVMIR_CAST_FP_TO_SIGNED_INT),     \
      LOOM_LLVMIR_CAST_INFO(FPTOUI, V##lanes##F32, V##lanes##I32,   \
                            LOOM_LLVMIR_CAST_FP_TO_UNSIGNED_INT),   \
      LOOM_LLVMIR_CAST_INFO(FPTRUNC, V##lanes##F32, V##lanes##F16,  \
                            LOOM_LLVMIR_CAST_FP_TRUNCATE),          \
      LOOM_LLVMIR_CAST_INFO(FPTRUNC, V##lanes##F32, V##lanes##BF16, \
                            LOOM_LLVMIR_CAST_FP_TRUNCATE),          \
      LOOM_LLVMIR_CAST_INFO(FPEXT, V##lanes##F16, V##lanes##F32,    \
                            LOOM_LLVMIR_CAST_FP_EXTEND),            \
      LOOM_LLVMIR_CAST_INFO(FPEXT, V##lanes##BF16, V##lanes##F32,   \
                            LOOM_LLVMIR_CAST_FP_EXTEND),            \
      LOOM_LLVMIR_CAST_INFO(BITCAST, V##lanes##I32, V##lanes##F32,  \
                            LOOM_LLVMIR_CAST_BITCAST),              \
      LOOM_LLVMIR_CAST_INFO(BITCAST, V##lanes##F32, V##lanes##I32,  \
                            LOOM_LLVMIR_CAST_BITCAST),              \
      LOOM_LLVMIR_CAST_INFO(BITCAST, V##lanes##I64, V##lanes##F64,  \
                            LOOM_LLVMIR_CAST_BITCAST),              \
      LOOM_LLVMIR_CAST_INFO(BITCAST, V##lanes##F64, V##lanes##I64,  \
                            LOOM_LLVMIR_CAST_BITCAST)

#define LOOM_LLVMIR_BITCAST_RESHAPE_INFOS()                                  \
  LOOM_LLVMIR_CAST_INFO(BITCAST, I32, V4I8, LOOM_LLVMIR_CAST_BITCAST),       \
      LOOM_LLVMIR_CAST_INFO(BITCAST, V2I32, V8I8, LOOM_LLVMIR_CAST_BITCAST), \
      LOOM_LLVMIR_CAST_INFO(BITCAST, V2F16, I32, LOOM_LLVMIR_CAST_BITCAST),  \
      LOOM_LLVMIR_CAST_INFO(BITCAST, V2BF16, I32, LOOM_LLVMIR_CAST_BITCAST)

static const loom_llvmir_emit_cast_info_t kCastInfos[] = {
    LOOM_LLVMIR_SCALAR_CAST_INFOS(),     LOOM_LLVMIR_VECTOR_CAST_INFOS(2),
    LOOM_LLVMIR_VECTOR_CAST_INFOS(3),    LOOM_LLVMIR_VECTOR_CAST_INFOS(4),
    LOOM_LLVMIR_VECTOR_CAST_INFOS(8),    LOOM_LLVMIR_VECTOR_CAST_INFOS(16),
    LOOM_LLVMIR_BITCAST_RESHAPE_INFOS(),
};

#undef LOOM_LLVMIR_BITCAST_RESHAPE_INFOS
#undef LOOM_LLVMIR_VECTOR_CAST_INFOS
#undef LOOM_LLVMIR_SCALAR_CAST_INFOS
#undef LOOM_LLVMIR_CAST_INFO

#define LOOM_LLVMIR_SELECT_REF(suffix) \
  LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_SELECT_##suffix

#define LOOM_LLVMIR_VECTOR_SELECT_REFS(lanes)                                  \
  LOOM_LLVMIR_SELECT_REF(V##lanes##I8), LOOM_LLVMIR_SELECT_REF(V##lanes##I16), \
      LOOM_LLVMIR_SELECT_REF(V##lanes##I32),                                   \
      LOOM_LLVMIR_SELECT_REF(V##lanes##I64),                                   \
      LOOM_LLVMIR_SELECT_REF(V##lanes##F16),                                   \
      LOOM_LLVMIR_SELECT_REF(V##lanes##BF16),                                  \
      LOOM_LLVMIR_SELECT_REF(V##lanes##F32),                                   \
      LOOM_LLVMIR_SELECT_REF(V##lanes##F64)

static const uint32_t kSelectDescriptorRefs[] = {
    LOOM_LLVMIR_SELECT_REF(I1),        LOOM_LLVMIR_SELECT_REF(I32),
    LOOM_LLVMIR_SELECT_REF(I64),       LOOM_LLVMIR_SELECT_REF(F32),
    LOOM_LLVMIR_SELECT_REF(F64),       LOOM_LLVMIR_VECTOR_SELECT_REFS(2),
    LOOM_LLVMIR_VECTOR_SELECT_REFS(3), LOOM_LLVMIR_VECTOR_SELECT_REFS(4),
    LOOM_LLVMIR_VECTOR_SELECT_REFS(8), LOOM_LLVMIR_VECTOR_SELECT_REFS(16),
};

#undef LOOM_LLVMIR_VECTOR_SELECT_REFS
#undef LOOM_LLVMIR_SELECT_REF

#define LOOM_LLVMIR_KERNEL_QUERY_INFO(query, suffix, kind, dimension)  \
  {LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_KERNEL_##query##_##suffix, kind, \
   dimension}

#define LOOM_LLVMIR_KERNEL_SCALAR_QUERY_INFO(query, kind)   \
  {LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_KERNEL_##query, kind, \
   LOOM_LLVMIR_EMIT_DIMENSION_X}

#define LOOM_LLVMIR_KERNEL_QUERY_INFOS(query, kind)                            \
  LOOM_LLVMIR_KERNEL_QUERY_INFO(query, X, kind, LOOM_LLVMIR_EMIT_DIMENSION_X), \
      LOOM_LLVMIR_KERNEL_QUERY_INFO(query, Y, kind,                            \
                                    LOOM_LLVMIR_EMIT_DIMENSION_Y),             \
      LOOM_LLVMIR_KERNEL_QUERY_INFO(query, Z, kind,                            \
                                    LOOM_LLVMIR_EMIT_DIMENSION_Z)

static const loom_llvmir_emit_kernel_query_info_t kKernelQueryInfos[] = {
    LOOM_LLVMIR_KERNEL_QUERY_INFOS(WORKITEM_ID,
                                   LOOM_LLVMIR_EMIT_KERNEL_QUERY_WORKITEM_ID),
    LOOM_LLVMIR_KERNEL_QUERY_INFOS(WORKGROUP_ID,
                                   LOOM_LLVMIR_EMIT_KERNEL_QUERY_WORKGROUP_ID),
    LOOM_LLVMIR_KERNEL_QUERY_INFOS(
        WORKGROUP_SIZE, LOOM_LLVMIR_EMIT_KERNEL_QUERY_WORKGROUP_SIZE),
    LOOM_LLVMIR_KERNEL_QUERY_INFOS(
        WORKITEM_DISPATCH_ID,
        LOOM_LLVMIR_EMIT_KERNEL_QUERY_WORKITEM_DISPATCH_ID),
    LOOM_LLVMIR_KERNEL_SCALAR_QUERY_INFO(
        SUBGROUP_SIZE, LOOM_LLVMIR_EMIT_KERNEL_QUERY_SUBGROUP_SIZE),
    LOOM_LLVMIR_KERNEL_SCALAR_QUERY_INFO(
        SUBGROUP_COUNT, LOOM_LLVMIR_EMIT_KERNEL_QUERY_SUBGROUP_COUNT),
};

#undef LOOM_LLVMIR_KERNEL_QUERY_INFOS
#undef LOOM_LLVMIR_KERNEL_SCALAR_QUERY_INFO
#undef LOOM_LLVMIR_KERNEL_QUERY_INFO

#define LOOM_LLVMIR_CONST_INFO(suffix, type, units, const_kind) \
  {LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_CONST_##suffix, type, units, const_kind}

#define LOOM_LLVMIR_VECTOR_CONST_INFOS(lanes)                                 \
  LOOM_LLVMIR_CONST_INFO(V##lanes##I8, LOOM_LLVMIR_EMIT_CORE_TYPE_I8, lanes,  \
                         LOOM_LLVMIR_EMIT_CONST_KIND_INTEGER),                \
      LOOM_LLVMIR_CONST_INFO(V##lanes##I16, LOOM_LLVMIR_EMIT_CORE_TYPE_I16,   \
                             lanes, LOOM_LLVMIR_EMIT_CONST_KIND_INTEGER),     \
      LOOM_LLVMIR_CONST_INFO(V##lanes##I32, LOOM_LLVMIR_EMIT_CORE_TYPE_I32,   \
                             lanes, LOOM_LLVMIR_EMIT_CONST_KIND_INTEGER),     \
      LOOM_LLVMIR_CONST_INFO(V##lanes##I64, LOOM_LLVMIR_EMIT_CORE_TYPE_I64,   \
                             lanes, LOOM_LLVMIR_EMIT_CONST_KIND_INTEGER),     \
      LOOM_LLVMIR_CONST_INFO(V##lanes##F16, LOOM_LLVMIR_EMIT_CORE_TYPE_F16,   \
                             lanes, LOOM_LLVMIR_EMIT_CONST_KIND_FLOAT_BITS),  \
      LOOM_LLVMIR_CONST_INFO(V##lanes##BF16, LOOM_LLVMIR_EMIT_CORE_TYPE_BF16, \
                             lanes, LOOM_LLVMIR_EMIT_CONST_KIND_FLOAT_BITS),  \
      LOOM_LLVMIR_CONST_INFO(V##lanes##F32, LOOM_LLVMIR_EMIT_CORE_TYPE_F32,   \
                             lanes, LOOM_LLVMIR_EMIT_CONST_KIND_FLOAT_BITS),  \
      LOOM_LLVMIR_CONST_INFO(V##lanes##F64, LOOM_LLVMIR_EMIT_CORE_TYPE_F64,   \
                             lanes, LOOM_LLVMIR_EMIT_CONST_KIND_FLOAT_BITS)

static const loom_llvmir_emit_const_info_t kConstInfos[] = {
    LOOM_LLVMIR_CONST_INFO(I1, LOOM_LLVMIR_EMIT_CORE_TYPE_I1, 1,
                           LOOM_LLVMIR_EMIT_CONST_KIND_INTEGER),
    LOOM_LLVMIR_CONST_INFO(I8, LOOM_LLVMIR_EMIT_CORE_TYPE_I8, 1,
                           LOOM_LLVMIR_EMIT_CONST_KIND_INTEGER),
    LOOM_LLVMIR_CONST_INFO(I16, LOOM_LLVMIR_EMIT_CORE_TYPE_I16, 1,
                           LOOM_LLVMIR_EMIT_CONST_KIND_INTEGER),
    LOOM_LLVMIR_CONST_INFO(I32, LOOM_LLVMIR_EMIT_CORE_TYPE_I32, 1,
                           LOOM_LLVMIR_EMIT_CONST_KIND_INTEGER),
    LOOM_LLVMIR_CONST_INFO(I64, LOOM_LLVMIR_EMIT_CORE_TYPE_I64, 1,
                           LOOM_LLVMIR_EMIT_CONST_KIND_INTEGER),
    LOOM_LLVMIR_CONST_INFO(F16, LOOM_LLVMIR_EMIT_CORE_TYPE_F16, 1,
                           LOOM_LLVMIR_EMIT_CONST_KIND_FLOAT_BITS),
    LOOM_LLVMIR_CONST_INFO(BF16, LOOM_LLVMIR_EMIT_CORE_TYPE_BF16, 1,
                           LOOM_LLVMIR_EMIT_CONST_KIND_FLOAT_BITS),
    LOOM_LLVMIR_CONST_INFO(F32, LOOM_LLVMIR_EMIT_CORE_TYPE_F32, 1,
                           LOOM_LLVMIR_EMIT_CONST_KIND_FLOAT_BITS),
    LOOM_LLVMIR_CONST_INFO(F64, LOOM_LLVMIR_EMIT_CORE_TYPE_F64, 1,
                           LOOM_LLVMIR_EMIT_CONST_KIND_FLOAT_BITS),
    LOOM_LLVMIR_VECTOR_CONST_INFOS(2),
    LOOM_LLVMIR_VECTOR_CONST_INFOS(3),
    LOOM_LLVMIR_VECTOR_CONST_INFOS(4),
    LOOM_LLVMIR_VECTOR_CONST_INFOS(8),
    LOOM_LLVMIR_VECTOR_CONST_INFOS(16),
};

#undef LOOM_LLVMIR_VECTOR_CONST_INFOS
#undef LOOM_LLVMIR_CONST_INFO

#define LOOM_LLVMIR_VECTOR_REF(prefix, suffix) \
  LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_##prefix##_##suffix

#define LOOM_LLVMIR_STRUCTURAL_VECTOR_REFS(prefix, lanes) \
  LOOM_LLVMIR_VECTOR_REF(prefix, V##lanes##I1),           \
      LOOM_LLVMIR_VECTOR_REF(prefix, V##lanes##I8),       \
      LOOM_LLVMIR_VECTOR_REF(prefix, V##lanes##I16),      \
      LOOM_LLVMIR_VECTOR_REF(prefix, V##lanes##I32),      \
      LOOM_LLVMIR_VECTOR_REF(prefix, V##lanes##I64),      \
      LOOM_LLVMIR_VECTOR_REF(prefix, V##lanes##F16),      \
      LOOM_LLVMIR_VECTOR_REF(prefix, V##lanes##BF16),     \
      LOOM_LLVMIR_VECTOR_REF(prefix, V##lanes##F32),      \
      LOOM_LLVMIR_VECTOR_REF(prefix, V##lanes##F64)

#define LOOM_LLVMIR_ALL_STRUCTURAL_VECTOR_REFS(prefix) \
  LOOM_LLVMIR_STRUCTURAL_VECTOR_REFS(prefix, 2),       \
      LOOM_LLVMIR_STRUCTURAL_VECTOR_REFS(prefix, 3),   \
      LOOM_LLVMIR_STRUCTURAL_VECTOR_REFS(prefix, 4),   \
      LOOM_LLVMIR_STRUCTURAL_VECTOR_REFS(prefix, 8),   \
      LOOM_LLVMIR_STRUCTURAL_VECTOR_REFS(prefix, 16),  \
      LOOM_LLVMIR_STRUCTURAL_VECTOR_REFS(prefix, 32)

#define LOOM_LLVMIR_DYNAMIC_INSERT_VECTOR_REFS(lanes)         \
  LOOM_LLVMIR_VECTOR_REF(INSERT_DYNAMIC, V##lanes##I1),       \
      LOOM_LLVMIR_VECTOR_REF(INSERT_DYNAMIC, V##lanes##I8),   \
      LOOM_LLVMIR_VECTOR_REF(INSERT_DYNAMIC, V##lanes##I16),  \
      LOOM_LLVMIR_VECTOR_REF(INSERT_DYNAMIC, V##lanes##I32),  \
      LOOM_LLVMIR_VECTOR_REF(INSERT_DYNAMIC, V##lanes##I64),  \
      LOOM_LLVMIR_VECTOR_REF(INSERT_DYNAMIC, V##lanes##F16),  \
      LOOM_LLVMIR_VECTOR_REF(INSERT_DYNAMIC, V##lanes##BF16), \
      LOOM_LLVMIR_VECTOR_REF(INSERT_DYNAMIC, V##lanes##F32),  \
      LOOM_LLVMIR_VECTOR_REF(INSERT_DYNAMIC, V##lanes##F64)

#define LOOM_LLVMIR_ALL_DYNAMIC_INSERT_VECTOR_REFS() \
  LOOM_LLVMIR_DYNAMIC_INSERT_VECTOR_REFS(2),         \
      LOOM_LLVMIR_DYNAMIC_INSERT_VECTOR_REFS(3),     \
      LOOM_LLVMIR_DYNAMIC_INSERT_VECTOR_REFS(4),     \
      LOOM_LLVMIR_DYNAMIC_INSERT_VECTOR_REFS(8),     \
      LOOM_LLVMIR_DYNAMIC_INSERT_VECTOR_REFS(16),    \
      LOOM_LLVMIR_DYNAMIC_INSERT_VECTOR_REFS(32)

static const uint32_t kSplatDescriptorRefs[] = {
    LOOM_LLVMIR_ALL_STRUCTURAL_VECTOR_REFS(SPLAT),
};

static const uint32_t kFromElementsDescriptorRefs[] = {
    LOOM_LLVMIR_ALL_STRUCTURAL_VECTOR_REFS(FROM_ELEMENTS),
};

static const uint32_t kExtractDescriptorRefs[] = {
    LOOM_LLVMIR_ALL_STRUCTURAL_VECTOR_REFS(EXTRACT),
};

static const uint32_t kInsertDescriptorRefs[] = {
    LOOM_LLVMIR_ALL_STRUCTURAL_VECTOR_REFS(INSERT),
};

static const uint32_t kDynamicInsertDescriptorRefs[] = {
    LOOM_LLVMIR_ALL_DYNAMIC_INSERT_VECTOR_REFS(),
};

static const uint32_t kShuffleDescriptorRefs[] = {
    LOOM_LLVMIR_ALL_STRUCTURAL_VECTOR_REFS(SHUFFLE),
};

#define LOOM_LLVMIR_CONCAT_REF(suffix) \
  {LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_CONCAT_V4##suffix##_V16##suffix, 4}

static const loom_llvmir_emit_concat_info_t kConcatInfos[] = {
    LOOM_LLVMIR_CONCAT_REF(I1),   LOOM_LLVMIR_CONCAT_REF(I8),
    LOOM_LLVMIR_CONCAT_REF(I16),  LOOM_LLVMIR_CONCAT_REF(I32),
    LOOM_LLVMIR_CONCAT_REF(I64),  LOOM_LLVMIR_CONCAT_REF(F16),
    LOOM_LLVMIR_CONCAT_REF(BF16), LOOM_LLVMIR_CONCAT_REF(F32),
    LOOM_LLVMIR_CONCAT_REF(F64),
};

#define LOOM_LLVMIR_SLICE_REF(source_lanes, result_lanes, suffix) \
  LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_SLICE_V##source_lanes##suffix##_V##result_lanes##suffix

static const uint32_t kSliceDescriptorRefs[] = {
    LOOM_LLVMIR_SLICE_REF(4, 2, I1),   LOOM_LLVMIR_SLICE_REF(4, 2, I8),
    LOOM_LLVMIR_SLICE_REF(4, 2, I16),  LOOM_LLVMIR_SLICE_REF(4, 2, I32),
    LOOM_LLVMIR_SLICE_REF(4, 2, I64),  LOOM_LLVMIR_SLICE_REF(4, 2, F16),
    LOOM_LLVMIR_SLICE_REF(4, 2, BF16), LOOM_LLVMIR_SLICE_REF(4, 2, F32),
    LOOM_LLVMIR_SLICE_REF(4, 2, F64),  LOOM_LLVMIR_SLICE_REF(16, 4, I1),
    LOOM_LLVMIR_SLICE_REF(16, 4, I8),  LOOM_LLVMIR_SLICE_REF(16, 4, I16),
    LOOM_LLVMIR_SLICE_REF(16, 4, I32), LOOM_LLVMIR_SLICE_REF(16, 4, I64),
    LOOM_LLVMIR_SLICE_REF(16, 4, F16), LOOM_LLVMIR_SLICE_REF(16, 4, BF16),
    LOOM_LLVMIR_SLICE_REF(16, 4, F32), LOOM_LLVMIR_SLICE_REF(16, 4, F64),
};

#undef LOOM_LLVMIR_SLICE_REF
#undef LOOM_LLVMIR_CONCAT_REF
#undef LOOM_LLVMIR_ALL_DYNAMIC_INSERT_VECTOR_REFS
#undef LOOM_LLVMIR_DYNAMIC_INSERT_VECTOR_REFS
#undef LOOM_LLVMIR_ALL_STRUCTURAL_VECTOR_REFS
#undef LOOM_LLVMIR_STRUCTURAL_VECTOR_REFS
#undef LOOM_LLVMIR_VECTOR_REF

#define LOOM_LLVMIR_MEMORY_INFO(ref, type, units, flags) \
  {LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_##ref, type, units, flags}

#define LOOM_LLVMIR_MEMORY_INFOS(suffix, type, units)                    \
  LOOM_LLVMIR_MEMORY_INFO(LOAD_##suffix, type, units,                    \
                          LOOM_LLVMIR_EMIT_MEMORY_FLAG_LOAD),            \
      LOOM_LLVMIR_MEMORY_INFO(STORE_##suffix, type, units, 0),           \
      LOOM_LLVMIR_MEMORY_INFO(LOAD_INDEXED_##suffix, type, units,        \
                              LOOM_LLVMIR_EMIT_MEMORY_FLAG_LOAD |        \
                                  LOOM_LLVMIR_EMIT_MEMORY_FLAG_INDEXED), \
      LOOM_LLVMIR_MEMORY_INFO(STORE_INDEXED_##suffix, type, units,       \
                              LOOM_LLVMIR_EMIT_MEMORY_FLAG_INDEXED)

#define LOOM_LLVMIR_VECTOR_MEMORY_INFOS(lanes, suffix, type) \
  LOOM_LLVMIR_MEMORY_INFOS(V##lanes##suffix, type, lanes)

#define LOOM_LLVMIR_ALL_MEMORY_INFOS(suffix, type)      \
  LOOM_LLVMIR_MEMORY_INFOS(suffix, type, 1),            \
      LOOM_LLVMIR_VECTOR_MEMORY_INFOS(2, suffix, type), \
      LOOM_LLVMIR_VECTOR_MEMORY_INFOS(3, suffix, type), \
      LOOM_LLVMIR_VECTOR_MEMORY_INFOS(4, suffix, type), \
      LOOM_LLVMIR_VECTOR_MEMORY_INFOS(8, suffix, type), \
      LOOM_LLVMIR_VECTOR_MEMORY_INFOS(16, suffix, type)

static const loom_llvmir_emit_memory_info_t kMemoryInfos[] = {
    LOOM_LLVMIR_ALL_MEMORY_INFOS(I8, LOOM_LLVMIR_EMIT_CORE_TYPE_I8),
    LOOM_LLVMIR_ALL_MEMORY_INFOS(I16, LOOM_LLVMIR_EMIT_CORE_TYPE_I16),
    LOOM_LLVMIR_ALL_MEMORY_INFOS(I32, LOOM_LLVMIR_EMIT_CORE_TYPE_I32),
    LOOM_LLVMIR_ALL_MEMORY_INFOS(I64, LOOM_LLVMIR_EMIT_CORE_TYPE_I64),
    LOOM_LLVMIR_ALL_MEMORY_INFOS(F16, LOOM_LLVMIR_EMIT_CORE_TYPE_F16),
    LOOM_LLVMIR_ALL_MEMORY_INFOS(BF16, LOOM_LLVMIR_EMIT_CORE_TYPE_BF16),
    LOOM_LLVMIR_ALL_MEMORY_INFOS(F32, LOOM_LLVMIR_EMIT_CORE_TYPE_F32),
    LOOM_LLVMIR_ALL_MEMORY_INFOS(F64, LOOM_LLVMIR_EMIT_CORE_TYPE_F64),
};

#undef LOOM_LLVMIR_ALL_MEMORY_INFOS
#undef LOOM_LLVMIR_VECTOR_MEMORY_INFOS
#undef LOOM_LLVMIR_MEMORY_INFOS
#undef LOOM_LLVMIR_MEMORY_INFO

#define LOOM_LLVMIR_ATOMIC_INFO(form, op_suffix, type_suffix, type, op, flags)              \
  {LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_ATOMIC_##form##_##op_suffix##_##type_suffix,          \
   type, op, flags},                                                                        \
  {                                                                                         \
    LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_ATOMIC_##form##_INDEXED_##op_suffix##_##type_suffix, \
        type, op, flags | LOOM_LLVMIR_EMIT_ATOMIC_FLAG_INDEXED                              \
  }

#define LOOM_LLVMIR_ATOMIC_CMPXCHG_INFO(type_suffix, type)                   \
  {LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_ATOMIC_CMPXCHG_##type_suffix, type,    \
   LOOM_LLVMIR_ATOMIC_RMW_XCHG,                                              \
   LOOM_LLVMIR_EMIT_ATOMIC_FLAG_RETURN_VALUE |                               \
       LOOM_LLVMIR_EMIT_ATOMIC_FLAG_CMPXCHG},                                \
  {                                                                          \
    LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_ATOMIC_CMPXCHG_INDEXED_##type_suffix, \
        type, LOOM_LLVMIR_ATOMIC_RMW_XCHG,                                   \
        LOOM_LLVMIR_EMIT_ATOMIC_FLAG_RETURN_VALUE |                          \
            LOOM_LLVMIR_EMIT_ATOMIC_FLAG_CMPXCHG |                           \
            LOOM_LLVMIR_EMIT_ATOMIC_FLAG_INDEXED                             \
  }

#define LOOM_LLVMIR_INTEGER_ATOMIC_REDUCE_INFOS(type_suffix, type) \
  LOOM_LLVMIR_ATOMIC_INFO(REDUCE, ADD, type_suffix, type,          \
                          LOOM_LLVMIR_ATOMIC_RMW_ADD, 0),          \
      LOOM_LLVMIR_ATOMIC_INFO(REDUCE, SUB, type_suffix, type,      \
                              LOOM_LLVMIR_ATOMIC_RMW_SUB, 0),      \
      LOOM_LLVMIR_ATOMIC_INFO(REDUCE, AND, type_suffix, type,      \
                              LOOM_LLVMIR_ATOMIC_RMW_AND, 0),      \
      LOOM_LLVMIR_ATOMIC_INFO(REDUCE, OR, type_suffix, type,       \
                              LOOM_LLVMIR_ATOMIC_RMW_OR, 0),       \
      LOOM_LLVMIR_ATOMIC_INFO(REDUCE, XOR, type_suffix, type,      \
                              LOOM_LLVMIR_ATOMIC_RMW_XOR, 0),      \
      LOOM_LLVMIR_ATOMIC_INFO(REDUCE, MIN, type_suffix, type,      \
                              LOOM_LLVMIR_ATOMIC_RMW_MIN, 0),      \
      LOOM_LLVMIR_ATOMIC_INFO(REDUCE, MAX, type_suffix, type,      \
                              LOOM_LLVMIR_ATOMIC_RMW_MAX, 0),      \
      LOOM_LLVMIR_ATOMIC_INFO(REDUCE, UMIN, type_suffix, type,     \
                              LOOM_LLVMIR_ATOMIC_RMW_UMIN, 0),     \
      LOOM_LLVMIR_ATOMIC_INFO(REDUCE, UMAX, type_suffix, type,     \
                              LOOM_LLVMIR_ATOMIC_RMW_UMAX, 0)

#define LOOM_LLVMIR_INTEGER_ATOMIC_RMW_INFOS(type_suffix, type)           \
  LOOM_LLVMIR_ATOMIC_INFO(RMW, XCHG, type_suffix, type,                   \
                          LOOM_LLVMIR_ATOMIC_RMW_XCHG,                    \
                          LOOM_LLVMIR_EMIT_ATOMIC_FLAG_RETURN_VALUE),     \
      LOOM_LLVMIR_ATOMIC_INFO(RMW, ADD, type_suffix, type,                \
                              LOOM_LLVMIR_ATOMIC_RMW_ADD,                 \
                              LOOM_LLVMIR_EMIT_ATOMIC_FLAG_RETURN_VALUE), \
      LOOM_LLVMIR_ATOMIC_INFO(RMW, SUB, type_suffix, type,                \
                              LOOM_LLVMIR_ATOMIC_RMW_SUB,                 \
                              LOOM_LLVMIR_EMIT_ATOMIC_FLAG_RETURN_VALUE), \
      LOOM_LLVMIR_ATOMIC_INFO(RMW, AND, type_suffix, type,                \
                              LOOM_LLVMIR_ATOMIC_RMW_AND,                 \
                              LOOM_LLVMIR_EMIT_ATOMIC_FLAG_RETURN_VALUE), \
      LOOM_LLVMIR_ATOMIC_INFO(RMW, OR, type_suffix, type,                 \
                              LOOM_LLVMIR_ATOMIC_RMW_OR,                  \
                              LOOM_LLVMIR_EMIT_ATOMIC_FLAG_RETURN_VALUE), \
      LOOM_LLVMIR_ATOMIC_INFO(RMW, XOR, type_suffix, type,                \
                              LOOM_LLVMIR_ATOMIC_RMW_XOR,                 \
                              LOOM_LLVMIR_EMIT_ATOMIC_FLAG_RETURN_VALUE), \
      LOOM_LLVMIR_ATOMIC_INFO(RMW, MIN, type_suffix, type,                \
                              LOOM_LLVMIR_ATOMIC_RMW_MIN,                 \
                              LOOM_LLVMIR_EMIT_ATOMIC_FLAG_RETURN_VALUE), \
      LOOM_LLVMIR_ATOMIC_INFO(RMW, MAX, type_suffix, type,                \
                              LOOM_LLVMIR_ATOMIC_RMW_MAX,                 \
                              LOOM_LLVMIR_EMIT_ATOMIC_FLAG_RETURN_VALUE), \
      LOOM_LLVMIR_ATOMIC_INFO(RMW, UMIN, type_suffix, type,               \
                              LOOM_LLVMIR_ATOMIC_RMW_UMIN,                \
                              LOOM_LLVMIR_EMIT_ATOMIC_FLAG_RETURN_VALUE), \
      LOOM_LLVMIR_ATOMIC_INFO(RMW, UMAX, type_suffix, type,               \
                              LOOM_LLVMIR_ATOMIC_RMW_UMAX,                \
                              LOOM_LLVMIR_EMIT_ATOMIC_FLAG_RETURN_VALUE)

#define LOOM_LLVMIR_FLOAT_ATOMIC_REDUCE_INFOS(type_suffix, type)   \
  LOOM_LLVMIR_ATOMIC_INFO(REDUCE, FADD, type_suffix, type,         \
                          LOOM_LLVMIR_ATOMIC_RMW_FADD, 0),         \
      LOOM_LLVMIR_ATOMIC_INFO(REDUCE, FMINIMUM, type_suffix, type, \
                              LOOM_LLVMIR_ATOMIC_RMW_FMINIMUM, 0), \
      LOOM_LLVMIR_ATOMIC_INFO(REDUCE, FMAXIMUM, type_suffix, type, \
                              LOOM_LLVMIR_ATOMIC_RMW_FMAXIMUM, 0), \
      LOOM_LLVMIR_ATOMIC_INFO(REDUCE, FMIN, type_suffix, type,     \
                              LOOM_LLVMIR_ATOMIC_RMW_FMIN, 0),     \
      LOOM_LLVMIR_ATOMIC_INFO(REDUCE, FMAX, type_suffix, type,     \
                              LOOM_LLVMIR_ATOMIC_RMW_FMAX, 0)

#define LOOM_LLVMIR_FLOAT_ATOMIC_RMW_INFOS(type_suffix, type)             \
  LOOM_LLVMIR_ATOMIC_INFO(RMW, XCHG, type_suffix, type,                   \
                          LOOM_LLVMIR_ATOMIC_RMW_XCHG,                    \
                          LOOM_LLVMIR_EMIT_ATOMIC_FLAG_RETURN_VALUE),     \
      LOOM_LLVMIR_ATOMIC_INFO(RMW, FADD, type_suffix, type,               \
                              LOOM_LLVMIR_ATOMIC_RMW_FADD,                \
                              LOOM_LLVMIR_EMIT_ATOMIC_FLAG_RETURN_VALUE), \
      LOOM_LLVMIR_ATOMIC_INFO(RMW, FMINIMUM, type_suffix, type,           \
                              LOOM_LLVMIR_ATOMIC_RMW_FMINIMUM,            \
                              LOOM_LLVMIR_EMIT_ATOMIC_FLAG_RETURN_VALUE), \
      LOOM_LLVMIR_ATOMIC_INFO(RMW, FMAXIMUM, type_suffix, type,           \
                              LOOM_LLVMIR_ATOMIC_RMW_FMAXIMUM,            \
                              LOOM_LLVMIR_EMIT_ATOMIC_FLAG_RETURN_VALUE), \
      LOOM_LLVMIR_ATOMIC_INFO(RMW, FMIN, type_suffix, type,               \
                              LOOM_LLVMIR_ATOMIC_RMW_FMIN,                \
                              LOOM_LLVMIR_EMIT_ATOMIC_FLAG_RETURN_VALUE), \
      LOOM_LLVMIR_ATOMIC_INFO(RMW, FMAX, type_suffix, type,               \
                              LOOM_LLVMIR_ATOMIC_RMW_FMAX,                \
                              LOOM_LLVMIR_EMIT_ATOMIC_FLAG_RETURN_VALUE)

#define LOOM_LLVMIR_INTEGER_ATOMIC_INFOS(type_suffix, type)    \
  LOOM_LLVMIR_INTEGER_ATOMIC_REDUCE_INFOS(type_suffix, type),  \
      LOOM_LLVMIR_INTEGER_ATOMIC_RMW_INFOS(type_suffix, type), \
      LOOM_LLVMIR_ATOMIC_CMPXCHG_INFO(type_suffix, type)

#define LOOM_LLVMIR_FLOAT_ATOMIC_INFOS(type_suffix, type)   \
  LOOM_LLVMIR_FLOAT_ATOMIC_REDUCE_INFOS(type_suffix, type), \
      LOOM_LLVMIR_FLOAT_ATOMIC_RMW_INFOS(type_suffix, type)

static const loom_llvmir_emit_atomic_info_t kAtomicInfos[] = {
    LOOM_LLVMIR_INTEGER_ATOMIC_INFOS(I8, LOOM_LLVMIR_EMIT_CORE_TYPE_I8),
    LOOM_LLVMIR_INTEGER_ATOMIC_INFOS(I16, LOOM_LLVMIR_EMIT_CORE_TYPE_I16),
    LOOM_LLVMIR_INTEGER_ATOMIC_INFOS(I32, LOOM_LLVMIR_EMIT_CORE_TYPE_I32),
    LOOM_LLVMIR_INTEGER_ATOMIC_INFOS(I64, LOOM_LLVMIR_EMIT_CORE_TYPE_I64),
    LOOM_LLVMIR_FLOAT_ATOMIC_INFOS(F32, LOOM_LLVMIR_EMIT_CORE_TYPE_F32),
    LOOM_LLVMIR_FLOAT_ATOMIC_INFOS(F64, LOOM_LLVMIR_EMIT_CORE_TYPE_F64),
};

#undef LOOM_LLVMIR_FLOAT_ATOMIC_INFOS
#undef LOOM_LLVMIR_INTEGER_ATOMIC_INFOS
#undef LOOM_LLVMIR_FLOAT_ATOMIC_RMW_INFOS
#undef LOOM_LLVMIR_FLOAT_ATOMIC_REDUCE_INFOS
#undef LOOM_LLVMIR_INTEGER_ATOMIC_RMW_INFOS
#undef LOOM_LLVMIR_INTEGER_ATOMIC_REDUCE_INFOS
#undef LOOM_LLVMIR_ATOMIC_CMPXCHG_INFO
#undef LOOM_LLVMIR_ATOMIC_INFO

static const loom_llvmir_emit_alloca_info_t kAllocaInfos[] = {
    {LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_ALLOCA_PRIVATE_I8,
     LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE},
    {LLVMIR_GENERIC_CORE_DESCRIPTOR_REF_ALLOCA_WORKGROUP_I8,
     LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP},
};

static iree_string_view_t loom_llvmir_emit_string_or_empty(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_string_view_t loom_llvmir_emit_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  return loom_low_diagnostic_symbol_name(module, symbol_ref);
}

static iree_string_view_t loom_llvmir_emit_value_name(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id >= module->values.count) return iree_string_view_empty();
  const loom_value_t* value = loom_module_value(module, value_id);
  return loom_llvmir_emit_string_or_empty(module, value->name_id);
}

static const loom_named_attr_t* loom_llvmir_emit_find_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  const loom_string_id_t name_id = loom_module_lookup_string(module, name);
  if (name_id == LOOM_STRING_ID_INVALID) return NULL;
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (attr->name_id == name_id) return attr;
  }
  return NULL;
}

static loom_named_attr_slice_t loom_llvmir_emit_packet_attrs(
    const loom_low_resolved_descriptor_packet_t* packet,
    uint16_t* out_attrs_attr_index) {
  switch (packet->kind) {
    case LOOM_LOW_DESCRIPTOR_PACKET_CONST:
      *out_attrs_attr_index = loom_low_const_attrs_ATTR_INDEX;
      return loom_low_const_attrs(packet->op);
    case LOOM_LOW_DESCRIPTOR_PACKET_OP:
      *out_attrs_attr_index = loom_low_op_attrs_ATTR_INDEX;
      return loom_low_op_attrs(packet->op);
    case LOOM_LOW_DESCRIPTOR_PACKET_NONE:
      break;
  }
  *out_attrs_attr_index = LOOM_ATTR_INDEX_NONE;
  return loom_named_attr_slice_empty();
}

static iree_status_t loom_llvmir_emit_diagnostic(
    loom_llvmir_emit_function_state_t* state, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count) {
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  IREE_RETURN_IF_ERROR(
      iree_diagnostic_emit(state->module_state->diagnostic_emitter, &emission));
  ++state->module_state->error_count;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_emit_shape_diagnostic(
    loom_llvmir_emit_function_state_t* state, const loom_op_t* op,
    iree_string_view_t subject_kind, uint32_t actual_count,
    uint32_t expected_count) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_string(LOOM_LLVMIR_LOW_EMITTER_KEY),
      loom_param_string(subject_kind),
      loom_param_u32(actual_count),
      loom_param_u32(expected_count),
  };
  return loom_llvmir_emit_diagnostic(state, op, LOOM_ERR_TARGET_054, params,
                                     IREE_ARRAYSIZE(params));
}

static iree_status_t loom_llvmir_emit_value_type_diagnostic(
    loom_llvmir_emit_function_state_t* state, const loom_op_t* op,
    loom_value_id_t value_id, iree_string_view_t value_kind,
    iree_string_view_t expected_constraint) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_string(value_kind),
      loom_param_string(
          loom_low_diagnostic_value_name(state->module, value_id)),
      loom_param_type(loom_module_value_type(state->module, value_id)),
      loom_param_string(LOOM_LLVMIR_LOW_EMITTER_KEY),
      loom_param_string(expected_constraint),
  };
  return loom_llvmir_emit_diagnostic(state, op, LOOM_ERR_TARGET_056, params,
                                     IREE_ARRAYSIZE(params));
}

static iree_status_t loom_llvmir_emit_missing_descriptor_diagnostic(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_with_field_ref(
          loom_param_string(packet->key),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    packet->key_attr_index)),
      loom_param_string(state->target->descriptor_set_key),
  };
  return loom_llvmir_emit_diagnostic(state, packet->op, LOOM_ERR_TARGET_045,
                                     params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_llvmir_emit_unsupported_descriptor_diagnostic(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_with_field_ref(
          loom_param_string(packet->key),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    packet->key_attr_index)),
      loom_param_string(state->target->descriptor_set_key),
      loom_param_string(LOOM_LLVMIR_LOW_EMITTER_KEY),
  };
  return loom_llvmir_emit_diagnostic(state, packet->op, LOOM_ERR_TARGET_053,
                                     params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_llvmir_emit_missing_immediate_diagnostic(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    iree_string_view_t immediate_name, uint16_t attrs_attr_index) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_with_field_ref(
          loom_param_string(packet->key),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    packet->key_attr_index)),
      loom_param_with_field_ref(
          loom_param_string(immediate_name),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attrs_attr_index)),
  };
  return loom_llvmir_emit_diagnostic(state, packet->op, LOOM_ERR_TARGET_047,
                                     params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_llvmir_emit_immediate_kind_diagnostic(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    iree_string_view_t immediate_name, uint16_t attrs_attr_index,
    loom_attr_kind_t actual_kind) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_with_field_ref(
          loom_param_string(packet->key),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    packet->key_attr_index)),
      loom_param_with_field_ref(
          loom_param_string(immediate_name),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attrs_attr_index)),
      loom_param_u32(actual_kind),
      loom_param_string(IREE_SV("i64")),
  };
  return loom_llvmir_emit_diagnostic(state, packet->op, LOOM_ERR_TARGET_049,
                                     params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_llvmir_emit_register_class_diagnostic(
    loom_llvmir_emit_function_state_t* state, const loom_op_t* op,
    loom_value_id_t value_id, iree_string_view_t value_kind,
    loom_diagnostic_field_ref_t field_ref) {
  const loom_type_t type = loom_module_value_type(state->module, value_id);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(state->function_name),
      loom_param_string(value_kind),
      loom_param_string(
          loom_low_diagnostic_value_name(state->module, value_id)),
      loom_param_with_field_ref(loom_param_type(type), field_ref),
      loom_param_string(state->target->descriptor_set_key),
  };
  return loom_llvmir_emit_diagnostic(state, op, LOOM_ERR_TARGET_042, params,
                                     IREE_ARRAYSIZE(params));
}

static iree_status_t loom_llvmir_emit_read_i64_immediate(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    iree_string_view_t immediate_name, bool* out_present, int64_t* out_value) {
  *out_present = false;
  uint16_t attrs_attr_index = LOOM_ATTR_INDEX_NONE;
  const loom_named_attr_slice_t attrs =
      loom_llvmir_emit_packet_attrs(packet, &attrs_attr_index);
  const loom_named_attr_t* attr =
      loom_llvmir_emit_find_attr(state->module, attrs, immediate_name);
  if (!attr) {
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_missing_immediate_diagnostic(
        state, packet, immediate_name, attrs_attr_index));
    return iree_ok_status();
  }
  if (attr->value.kind != LOOM_ATTR_I64) {
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_immediate_kind_diagnostic(
        state, packet, immediate_name, attrs_attr_index, attr->value.kind));
    return iree_ok_status();
  }
  *out_value = loom_attr_as_i64(attr->value);
  *out_present = true;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_emit_read_optional_i64_immediate(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    iree_string_view_t immediate_name, int64_t* inout_value) {
  uint16_t attrs_attr_index = LOOM_ATTR_INDEX_NONE;
  const loom_named_attr_slice_t attrs =
      loom_llvmir_emit_packet_attrs(packet, &attrs_attr_index);
  const loom_named_attr_t* attr =
      loom_llvmir_emit_find_attr(state->module, attrs, immediate_name);
  if (!attr) return iree_ok_status();
  if (attr->value.kind != LOOM_ATTR_I64) {
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_immediate_kind_diagnostic(
        state, packet, immediate_name, attrs_attr_index, attr->value.kind));
    return iree_ok_status();
  }
  *inout_value = loom_attr_as_i64(attr->value);
  return iree_ok_status();
}

static iree_status_t loom_llvmir_emit_core_scalar_type(
    loom_llvmir_module_t* module, loom_llvmir_emit_core_type_t type,
    uint32_t pointer_address_space, loom_llvmir_type_id_t* out_type_id) {
  switch (type) {
    case LOOM_LLVMIR_EMIT_CORE_TYPE_I1:
      return loom_llvmir_module_get_integer_type(module, 1, out_type_id);
    case LOOM_LLVMIR_EMIT_CORE_TYPE_I8:
      return loom_llvmir_module_get_integer_type(module, 8, out_type_id);
    case LOOM_LLVMIR_EMIT_CORE_TYPE_I16:
      return loom_llvmir_module_get_integer_type(module, 16, out_type_id);
    case LOOM_LLVMIR_EMIT_CORE_TYPE_I32:
      return loom_llvmir_module_get_integer_type(module, 32, out_type_id);
    case LOOM_LLVMIR_EMIT_CORE_TYPE_I64:
      return loom_llvmir_module_get_integer_type(module, 64, out_type_id);
    case LOOM_LLVMIR_EMIT_CORE_TYPE_F16:
      return loom_llvmir_module_get_float_type(module, LOOM_LLVMIR_FLOAT_F16,
                                               out_type_id);
    case LOOM_LLVMIR_EMIT_CORE_TYPE_BF16:
      return loom_llvmir_module_get_float_type(module, LOOM_LLVMIR_FLOAT_BF16,
                                               out_type_id);
    case LOOM_LLVMIR_EMIT_CORE_TYPE_F32:
      return loom_llvmir_module_get_float_type(module, LOOM_LLVMIR_FLOAT_F32,
                                               out_type_id);
    case LOOM_LLVMIR_EMIT_CORE_TYPE_F64:
      return loom_llvmir_module_get_float_type(module, LOOM_LLVMIR_FLOAT_F64,
                                               out_type_id);
    case LOOM_LLVMIR_EMIT_CORE_TYPE_PTR:
      return loom_llvmir_module_get_pointer_type(module, pointer_address_space,
                                                 out_type_id);
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "unknown LLVMIR low core type %d", (int)type);
}

static iree_status_t loom_llvmir_emit_core_type(
    loom_llvmir_module_t* module, loom_llvmir_emit_core_type_t type,
    uint32_t unit_count, uint32_t pointer_address_space,
    loom_llvmir_type_id_t* out_type_id) {
  loom_llvmir_type_id_t scalar_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_core_scalar_type(
      module, type, pointer_address_space, &scalar_type));
  if (unit_count == 1) {
    *out_type_id = scalar_type;
    return iree_ok_status();
  }
  return loom_llvmir_module_get_vector_type(module, unit_count, scalar_type,
                                            out_type_id);
}

static bool loom_llvmir_emit_core_type_byte_count(
    loom_llvmir_emit_core_type_t type, uint32_t* out_byte_count) {
  switch (type) {
    case LOOM_LLVMIR_EMIT_CORE_TYPE_I1:
      *out_byte_count = 1;
      return true;
    case LOOM_LLVMIR_EMIT_CORE_TYPE_I8:
      *out_byte_count = 1;
      return true;
    case LOOM_LLVMIR_EMIT_CORE_TYPE_I16:
    case LOOM_LLVMIR_EMIT_CORE_TYPE_F16:
    case LOOM_LLVMIR_EMIT_CORE_TYPE_BF16:
      *out_byte_count = 2;
      return true;
    case LOOM_LLVMIR_EMIT_CORE_TYPE_I32:
    case LOOM_LLVMIR_EMIT_CORE_TYPE_F32:
      *out_byte_count = 4;
      return true;
    case LOOM_LLVMIR_EMIT_CORE_TYPE_I64:
    case LOOM_LLVMIR_EMIT_CORE_TYPE_F64:
      *out_byte_count = 8;
      return true;
    case LOOM_LLVMIR_EMIT_CORE_TYPE_PTR:
      return false;
  }
  return false;
}

static const loom_llvmir_emit_alloca_info_t* loom_llvmir_emit_lookup_alloca(
    uint32_t descriptor_ref) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kAllocaInfos); ++i) {
    if (kAllocaInfos[i].descriptor_ref == descriptor_ref) {
      return &kAllocaInfos[i];
    }
  }
  return NULL;
}

static bool loom_llvmir_emit_memory_space_address_space(
    const loom_llvmir_target_profile_t* profile,
    loom_value_fact_memory_space_t memory_space, uint32_t* out_address_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
      *out_address_space = profile->target_env->address_spaces.generic;
      return true;
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
      *out_address_space = profile->target_env->address_spaces.global;
      return true;
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      *out_address_space = profile->target_env->address_spaces.local;
      return true;
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
      *out_address_space = profile->target_env->address_spaces.private_memory;
      return true;
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
      *out_address_space = profile->target_env->address_spaces.constant;
      return true;
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
      if (profile->target_env->address_spaces.buffer_resource == UINT32_MAX) {
        return false;
      }
      *out_address_space = profile->target_env->address_spaces.buffer_resource;
      return true;
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
      return false;
  }
  return false;
}

static bool loom_llvmir_emit_low_value_is_pointer_register(
    loom_llvmir_emit_function_state_t* state, loom_value_id_t value_id) {
  if (value_id >= state->module->values.count) return false;
  const loom_type_t type = loom_module_value_type(state->module, value_id);
  const loom_low_register_type_resolver_t resolver =
      loom_low_register_type_resolver_for_descriptor_set(
          state->target->descriptor_set);
  uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  if (!loom_low_register_type_resolver_try_resolve(&resolver, type,
                                                   &reg_class_id, NULL)) {
    return false;
  }
  return reg_class_id == LLVMIR_GENERIC_CORE_REG_CLASS_ID_PTR &&
         loom_low_register_type_unit_count(type) == 1;
}

static iree_status_t loom_llvmir_emit_pointer_address_space_for_low_value(
    loom_llvmir_emit_function_state_t* state, const loom_op_t* op,
    loom_value_id_t value_id, iree_string_view_t value_kind,
    iree_string_view_t expected_constraint, uint32_t* out_address_space,
    bool* out_supported) {
  *out_address_space =
      state->target_profile->target_env->address_spaces.generic;
  *out_supported = true;
  if (!loom_llvmir_emit_low_value_is_pointer_register(state, value_id)) {
    *out_supported = false;
    return loom_llvmir_emit_value_type_diagnostic(
        state, op, value_id, value_kind, expected_constraint);
  }

  const loom_value_t* value = loom_module_value(state->module, value_id);
  if (loom_value_is_block_arg(value)) {
    if (state->target_profile->kind == LOOM_LLVMIR_TARGET_PROFILE_KERNEL) {
      *out_supported = false;
      return loom_llvmir_emit_value_type_diagnostic(
          state, op, value_id, value_kind,
          IREE_SV("low.resource<hal_binding> pointer"));
    }
    return iree_ok_status();
  }

  const loom_op_t* def_op = loom_value_def_op(value);
  if (def_op != NULL && loom_low_resource_isa(def_op)) {
    switch (loom_low_resource_import_kind(def_op)) {
      case LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER:
        *out_address_space =
            state->target_profile->target_env->address_spaces.generic;
        return iree_ok_status();
      case LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING:
        *out_address_space =
            state->target_profile->target_env->address_spaces.global;
        return iree_ok_status();
      case LOOM_LOW_RESOURCE_IMPORT_KIND_VM_STATE:
      case LOOM_LOW_RESOURCE_IMPORT_KIND_VM_IMPORT:
      default:
        *out_supported = false;
        return loom_llvmir_emit_value_type_diagnostic(
            state, op, value_id, value_kind,
            IREE_SV("native_pointer or hal_binding pointer resource"));
    }
  }
  if (def_op != NULL) {
    loom_low_resolved_descriptor_packet_t packet = {0};
    IREE_RETURN_IF_ERROR(loom_low_resolve_descriptor_packet(
        state->module, state->target, def_op, &packet));
    if (packet.descriptor != NULL) {
      const uint32_t descriptor_ref =
          loom_low_descriptor_set_descriptor_ordinal(
              state->target->descriptor_set, packet.descriptor);
      const loom_llvmir_emit_alloca_info_t* alloca_info =
          loom_llvmir_emit_lookup_alloca(descriptor_ref);
      if (alloca_info != NULL &&
          !loom_llvmir_emit_memory_space_address_space(
              state->target_profile, alloca_info->memory_space,
              out_address_space)) {
        *out_supported = false;
        return loom_llvmir_emit_value_type_diagnostic(
            state, op, value_id, value_kind,
            IREE_SV("LLVMIR-addressable scratch pointer"));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_emit_type_for_low_value(
    loom_llvmir_emit_function_state_t* state, const loom_op_t* op,
    loom_value_id_t value_id, iree_string_view_t value_kind,
    loom_diagnostic_field_ref_t field_ref, loom_llvmir_type_id_t* out_type_id) {
  *out_type_id = LOOM_LLVMIR_TYPE_ID_INVALID;
  const loom_type_t type = loom_module_value_type(state->module, value_id);
  const loom_low_register_type_resolver_t resolver =
      loom_low_register_type_resolver_for_descriptor_set(
          state->target->descriptor_set);
  uint16_t reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  if (!loom_low_register_type_resolver_try_resolve(&resolver, type,
                                                   &reg_class_id, NULL)) {
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_register_class_diagnostic(
        state, op, value_id, value_kind, field_ref));
    return iree_ok_status();
  }
  loom_llvmir_emit_core_type_t core_type = LOOM_LLVMIR_EMIT_CORE_TYPE_I1;
  switch (reg_class_id) {
    case LLVMIR_GENERIC_CORE_REG_CLASS_ID_I1:
      core_type = LOOM_LLVMIR_EMIT_CORE_TYPE_I1;
      break;
    case LLVMIR_GENERIC_CORE_REG_CLASS_ID_I8:
      core_type = LOOM_LLVMIR_EMIT_CORE_TYPE_I8;
      break;
    case LLVMIR_GENERIC_CORE_REG_CLASS_ID_I16:
      core_type = LOOM_LLVMIR_EMIT_CORE_TYPE_I16;
      break;
    case LLVMIR_GENERIC_CORE_REG_CLASS_ID_I32:
      core_type = LOOM_LLVMIR_EMIT_CORE_TYPE_I32;
      break;
    case LLVMIR_GENERIC_CORE_REG_CLASS_ID_I64:
      core_type = LOOM_LLVMIR_EMIT_CORE_TYPE_I64;
      break;
    case LLVMIR_GENERIC_CORE_REG_CLASS_ID_F16:
      core_type = LOOM_LLVMIR_EMIT_CORE_TYPE_F16;
      break;
    case LLVMIR_GENERIC_CORE_REG_CLASS_ID_BF16:
      core_type = LOOM_LLVMIR_EMIT_CORE_TYPE_BF16;
      break;
    case LLVMIR_GENERIC_CORE_REG_CLASS_ID_F32:
      core_type = LOOM_LLVMIR_EMIT_CORE_TYPE_F32;
      break;
    case LLVMIR_GENERIC_CORE_REG_CLASS_ID_F64:
      core_type = LOOM_LLVMIR_EMIT_CORE_TYPE_F64;
      break;
    case LLVMIR_GENERIC_CORE_REG_CLASS_ID_PTR:
      core_type = LOOM_LLVMIR_EMIT_CORE_TYPE_PTR;
      break;
    default: {
      IREE_RETURN_IF_ERROR(loom_llvmir_emit_register_class_diagnostic(
          state, op, value_id, value_kind, field_ref));
      return iree_ok_status();
    }
  }
  uint32_t pointer_address_space = 0;
  if (core_type == LOOM_LLVMIR_EMIT_CORE_TYPE_PTR) {
    bool supported = true;
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_pointer_address_space_for_low_value(
        state, op, value_id, value_kind, IREE_SV("reg<llvmir.ptr>"),
        &pointer_address_space, &supported));
    if (!supported) return iree_ok_status();
  }
  return loom_llvmir_emit_core_type(state->llvmir_module, core_type,
                                    loom_low_register_type_unit_count(type),
                                    pointer_address_space, out_type_id);
}

static iree_status_t loom_llvmir_emit_descriptor_set_diagnostic(
    loom_llvmir_emit_module_state_t* state, const loom_op_t* op,
    iree_string_view_t function_name,
    const loom_low_resolved_target_t* target) {
  const iree_string_view_t descriptor_set_key = loom_low_descriptor_set_string(
      target->descriptor_set, target->descriptor_set->key_string_offset);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(function_name),
      loom_param_string(descriptor_set_key),
      loom_param_string(LOOM_LLVMIR_LOW_EMITTER_KEY),
      loom_param_string(LOOM_LLVMIR_GENERIC_CORE_DESCRIPTOR_SET_KEY),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_TARGET_055,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(
      iree_diagnostic_emit(state->diagnostic_emitter, &emission));
  ++state->error_count;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_emit_projection_diagnostic(
    loom_llvmir_emit_function_state_t* state) {
  const loom_target_bundle_storage_t* bundle_storage =
      &state->target->bundle_storage;
  const loom_diagnostic_param_t params[] = {
      loom_param_string(state->target->target_name),
      loom_param_string(bundle_storage->export_plan.name),
      loom_param_string(bundle_storage->config.name),
      loom_param_string(LOOM_LLVMIR_LOW_EMITTER_KEY),
      loom_param_string(loom_target_codegen_format_name(
          bundle_storage->snapshot.codegen_format)),
      loom_param_string(
          loom_target_abi_kind_name(bundle_storage->export_plan.abi_kind)),
  };
  return loom_llvmir_emit_diagnostic(state, state->function_op,
                                     LOOM_ERR_TARGET_036, params,
                                     IREE_ARRAYSIZE(params));
}

static iree_status_t loom_llvmir_emit_no_functions_diagnostic(
    loom_llvmir_emit_module_state_t* state) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(LOOM_LLVMIR_LOW_EMITTER_KEY),
  };
  const loom_diagnostic_emission_t emission = {
      .error = LOOM_ERR_TARGET_011,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(
      iree_diagnostic_emit(state->diagnostic_emitter, &emission));
  ++state->error_count;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_emit_lookup_value(
    loom_llvmir_emit_function_state_t* state, loom_value_id_t value_id,
    loom_llvmir_value_id_t* out_value_id) {
  if (value_id >= state->value_map_count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "low value %" PRIu32 " is outside the LLVMIR value map", value_id);
  }
  *out_value_id = state->value_map[value_id];
  if (*out_value_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "low value %" PRIu32 " has no LLVMIR definition",
                            value_id);
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_emit_define_value(
    loom_llvmir_emit_function_state_t* state, loom_value_id_t value_id,
    loom_llvmir_value_id_t llvmir_value_id) {
  if (value_id >= state->value_map_count) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "low value %" PRIu32 " is outside the LLVMIR value map", value_id);
  }
  state->value_map[value_id] = llvmir_value_id;
  return iree_ok_status();
}

static const loom_llvmir_emit_binary_info_t* loom_llvmir_emit_lookup_binary(
    uint32_t descriptor_ref) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kBinaryInfos); ++i) {
    if (kBinaryInfos[i].descriptor_ref == descriptor_ref) {
      return &kBinaryInfos[i];
    }
  }
  return NULL;
}

static const loom_llvmir_emit_binary_intrinsic_info_t*
loom_llvmir_emit_lookup_binary_intrinsic(uint32_t descriptor_ref) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kBinaryIntrinsicInfos); ++i) {
    if (kBinaryIntrinsicInfos[i].descriptor_ref == descriptor_ref) {
      return &kBinaryIntrinsicInfos[i];
    }
  }
  return NULL;
}

static const loom_llvmir_emit_unary_info_t* loom_llvmir_emit_lookup_unary(
    uint32_t descriptor_ref) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kUnaryInfos); ++i) {
    if (kUnaryInfos[i].descriptor_ref == descriptor_ref) {
      return &kUnaryInfos[i];
    }
  }
  return NULL;
}

static const loom_llvmir_emit_unary_intrinsic_info_t*
loom_llvmir_emit_lookup_unary_intrinsic(uint32_t descriptor_ref) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kUnaryIntrinsicInfos); ++i) {
    if (kUnaryIntrinsicInfos[i].descriptor_ref == descriptor_ref) {
      return &kUnaryIntrinsicInfos[i];
    }
  }
  return NULL;
}

static const loom_llvmir_emit_ternary_intrinsic_info_t*
loom_llvmir_emit_lookup_ternary_intrinsic(uint32_t descriptor_ref) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kTernaryIntrinsicInfos);
       ++i) {
    if (kTernaryIntrinsicInfos[i].descriptor_ref == descriptor_ref) {
      return &kTernaryIntrinsicInfos[i];
    }
  }
  return NULL;
}

static const loom_llvmir_emit_const_info_t* loom_llvmir_emit_lookup_const(
    uint32_t descriptor_ref) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kConstInfos); ++i) {
    if (kConstInfos[i].descriptor_ref == descriptor_ref) {
      return &kConstInfos[i];
    }
  }
  return NULL;
}

static const loom_llvmir_emit_compare_info_t* loom_llvmir_emit_lookup_compare(
    uint32_t descriptor_ref) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kCompareInfos); ++i) {
    if (kCompareInfos[i].descriptor_ref == descriptor_ref) {
      return &kCompareInfos[i];
    }
  }
  return NULL;
}

static const loom_llvmir_emit_cast_info_t* loom_llvmir_emit_lookup_cast(
    uint32_t descriptor_ref) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kCastInfos); ++i) {
    if (kCastInfos[i].descriptor_ref == descriptor_ref) {
      return &kCastInfos[i];
    }
  }
  return NULL;
}

static const loom_llvmir_emit_concat_info_t* loom_llvmir_emit_lookup_concat(
    uint32_t descriptor_ref) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kConcatInfos); ++i) {
    if (kConcatInfos[i].descriptor_ref == descriptor_ref) {
      return &kConcatInfos[i];
    }
  }
  return NULL;
}

static const loom_llvmir_emit_kernel_query_info_t*
loom_llvmir_emit_lookup_kernel_query(uint32_t descriptor_ref) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kKernelQueryInfos); ++i) {
    if (kKernelQueryInfos[i].descriptor_ref == descriptor_ref) {
      return &kKernelQueryInfos[i];
    }
  }
  return NULL;
}

static bool loom_llvmir_emit_descriptor_ref_in(
    uint32_t descriptor_ref, const uint32_t* descriptor_refs,
    iree_host_size_t descriptor_ref_count) {
  for (iree_host_size_t i = 0; i < descriptor_ref_count; ++i) {
    if (descriptor_refs[i] == descriptor_ref) return true;
  }
  return false;
}

static bool loom_llvmir_emit_is_select(uint32_t descriptor_ref) {
  return loom_llvmir_emit_descriptor_ref_in(
      descriptor_ref, kSelectDescriptorRefs,
      IREE_ARRAYSIZE(kSelectDescriptorRefs));
}

static const loom_llvmir_emit_memory_info_t* loom_llvmir_emit_lookup_memory(
    uint32_t descriptor_ref) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kMemoryInfos); ++i) {
    if (kMemoryInfos[i].descriptor_ref == descriptor_ref) {
      return &kMemoryInfos[i];
    }
  }
  return NULL;
}

static const loom_llvmir_emit_atomic_info_t* loom_llvmir_emit_lookup_atomic(
    uint32_t descriptor_ref) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kAtomicInfos); ++i) {
    if (kAtomicInfos[i].descriptor_ref == descriptor_ref) {
      return &kAtomicInfos[i];
    }
  }
  return NULL;
}

static iree_status_t loom_llvmir_emit_prepare_packet_result(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    loom_llvmir_type_id_t* out_result_type, loom_value_id_t* out_result_value) {
  if (packet->op->result_count != 1) {
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_shape_diagnostic(
        state, packet->op, IREE_SV("packet_result"), packet->op->result_count,
        1));
    *out_result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
    *out_result_value = LOOM_VALUE_ID_INVALID;
    return iree_ok_status();
  }
  const loom_value_id_t result_value = loom_op_const_results(packet->op)[0];
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_type_for_low_value(
      state, packet->op, result_value, IREE_SV("result"),
      loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, 0),
      out_result_type));
  *out_result_value = result_value;
  return iree_ok_status();
}

static uint32_t loom_llvmir_emit_low_value_unit_count(
    loom_llvmir_emit_function_state_t* state, loom_value_id_t value_id) {
  return loom_low_register_type_unit_count(
      loom_module_value_type(state->module, value_id));
}

static iree_status_t loom_llvmir_emit_binary(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_llvmir_emit_binary_info_t* info) {
  if (packet->op->operand_count != 2) {
    return loom_llvmir_emit_shape_diagnostic(state, packet->op,
                                             IREE_SV("packet_operand"),
                                             packet->op->operand_count, 2);
  }
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_value_id_t result_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_prepare_packet_result(
      state, packet, &result_type, &result_value));
  if (result_type == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();

  loom_llvmir_value_id_t lhs = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t rhs = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_lookup_value(
      state, loom_op_const_operands(packet->op)[0], &lhs));
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_lookup_value(
      state, loom_op_const_operands(packet->op)[1], &rhs));
  int64_t fast_math_flags = 0;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_read_optional_i64_immediate(
      state, packet, IREE_SV("fast_math_flags"), &fast_math_flags));
  loom_llvmir_value_id_t llvmir_result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_binop(
      state->llvmir_block,
      &(loom_llvmir_binop_desc_t){
          .result_name =
              loom_llvmir_emit_value_name(state->module, result_value),
          .result_type = result_type,
          .op = info->binop,
          .lhs = lhs,
          .rhs = rhs,
          .fast_math_flags = (loom_llvmir_fast_math_flags_t)fast_math_flags,
      },
      &llvmir_result));
  return loom_llvmir_emit_define_value(state, result_value, llvmir_result);
}

static iree_status_t loom_llvmir_emit_unary(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_llvmir_emit_unary_info_t* info) {
  if (packet->op->operand_count != 1) {
    return loom_llvmir_emit_shape_diagnostic(state, packet->op,
                                             IREE_SV("packet_operand"),
                                             packet->op->operand_count, 1);
  }
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_value_id_t result_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_prepare_packet_result(
      state, packet, &result_type, &result_value));
  if (result_type == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();

  loom_llvmir_value_id_t input = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_lookup_value(
      state, loom_op_const_operands(packet->op)[0], &input));
  loom_llvmir_value_id_t llvmir_result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_unop(
      state->llvmir_block,
      &(loom_llvmir_unop_desc_t){
          .result_name =
              loom_llvmir_emit_value_name(state->module, result_value),
          .result_type = result_type,
          .op = info->unop,
          .value = input,
      },
      &llvmir_result));
  return loom_llvmir_emit_define_value(state, result_value, llvmir_result);
}

static iree_status_t loom_llvmir_emit_declare_same_type_intrinsic(
    loom_llvmir_emit_function_state_t* state, iree_string_view_t name,
    loom_llvmir_type_id_t type_id, uint32_t parameter_count,
    loom_llvmir_function_t** out_function) {
  *out_function = loom_llvmir_module_find_function(state->llvmir_module, name);
  if (*out_function != NULL) return iree_ok_status();

  loom_llvmir_function_desc_t desc = {
      .kind = LOOM_LLVMIR_FUNCTION_DECLARATION,
      .name = name,
      .return_type = type_id,
      .linkage = LOOM_LLVMIR_LINKAGE_DEFAULT,
      .calling_convention = LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT,
      .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_function(state->llvmir_module,
                                                       &desc, out_function));
  loom_llvmir_value_id_t ignored = LOOM_LLVMIR_VALUE_ID_INVALID;
  for (uint32_t i = 0; i < parameter_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
        *out_function, &(loom_llvmir_parameter_desc_t){.type_id = type_id},
        &ignored));
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_emit_binary_intrinsic(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_llvmir_emit_binary_intrinsic_info_t* info) {
  if (packet->op->operand_count != 2) {
    return loom_llvmir_emit_shape_diagnostic(state, packet->op,
                                             IREE_SV("packet_operand"),
                                             packet->op->operand_count, 2);
  }
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_value_id_t result_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_prepare_packet_result(
      state, packet, &result_type, &result_value));
  if (result_type == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();

  const loom_value_id_t* operands = loom_op_const_operands(packet->op);
  loom_llvmir_value_id_t args[2] = {
      LOOM_LLVMIR_VALUE_ID_INVALID,
      LOOM_LLVMIR_VALUE_ID_INVALID,
  };
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(args); ++i) {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_emit_lookup_value(state, operands[i], &args[i]));
  }

  loom_llvmir_function_t* intrinsic = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_declare_same_type_intrinsic(
      state, info->intrinsic_name, result_type, IREE_ARRAYSIZE(args),
      &intrinsic));
  loom_llvmir_value_id_t llvmir_result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_call(
      state->llvmir_block,
      &(loom_llvmir_call_desc_t){
          .result_name =
              loom_llvmir_emit_value_name(state->module, result_value),
          .callee = loom_llvmir_function_id(intrinsic),
          .args = args,
          .arg_count = IREE_ARRAYSIZE(args),
      },
      &llvmir_result));
  return loom_llvmir_emit_define_value(state, result_value, llvmir_result);
}

static iree_status_t loom_llvmir_emit_unary_intrinsic(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_llvmir_emit_unary_intrinsic_info_t* info) {
  if (packet->op->operand_count != 1) {
    return loom_llvmir_emit_shape_diagnostic(state, packet->op,
                                             IREE_SV("packet_operand"),
                                             packet->op->operand_count, 1);
  }
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_value_id_t result_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_prepare_packet_result(
      state, packet, &result_type, &result_value));
  if (result_type == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();

  loom_llvmir_value_id_t arg = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_lookup_value(
      state, loom_op_const_operands(packet->op)[0], &arg));
  loom_llvmir_function_t* intrinsic = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_declare_same_type_intrinsic(
      state, info->intrinsic_name, result_type, 1, &intrinsic));
  loom_llvmir_value_id_t llvmir_result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_call(
      state->llvmir_block,
      &(loom_llvmir_call_desc_t){
          .result_name =
              loom_llvmir_emit_value_name(state->module, result_value),
          .callee = loom_llvmir_function_id(intrinsic),
          .args = &arg,
          .arg_count = 1,
      },
      &llvmir_result));
  return loom_llvmir_emit_define_value(state, result_value, llvmir_result);
}

static iree_status_t loom_llvmir_emit_ternary_intrinsic(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_llvmir_emit_ternary_intrinsic_info_t* info) {
  if (packet->op->operand_count != 3) {
    return loom_llvmir_emit_shape_diagnostic(state, packet->op,
                                             IREE_SV("packet_operand"),
                                             packet->op->operand_count, 3);
  }
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_value_id_t result_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_prepare_packet_result(
      state, packet, &result_type, &result_value));
  if (result_type == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();

  const loom_value_id_t* operands = loom_op_const_operands(packet->op);
  loom_llvmir_value_id_t args[3] = {
      LOOM_LLVMIR_VALUE_ID_INVALID,
      LOOM_LLVMIR_VALUE_ID_INVALID,
      LOOM_LLVMIR_VALUE_ID_INVALID,
  };
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(args); ++i) {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_emit_lookup_value(state, operands[i], &args[i]));
  }

  loom_llvmir_function_t* intrinsic = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_declare_same_type_intrinsic(
      state, info->intrinsic_name, result_type, IREE_ARRAYSIZE(args),
      &intrinsic));
  loom_llvmir_value_id_t llvmir_result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_call(
      state->llvmir_block,
      &(loom_llvmir_call_desc_t){
          .result_name =
              loom_llvmir_emit_value_name(state->module, result_value),
          .callee = loom_llvmir_function_id(intrinsic),
          .args = args,
          .arg_count = IREE_ARRAYSIZE(args),
      },
      &llvmir_result));
  return loom_llvmir_emit_define_value(state, result_value, llvmir_result);
}

static iree_status_t loom_llvmir_emit_compare(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_llvmir_emit_compare_info_t* info) {
  if (packet->op->operand_count != 2) {
    return loom_llvmir_emit_shape_diagnostic(state, packet->op,
                                             IREE_SV("packet_operand"),
                                             packet->op->operand_count, 2);
  }
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_value_id_t result_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_prepare_packet_result(
      state, packet, &result_type, &result_value));
  if (result_type == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();

  loom_llvmir_value_id_t lhs = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t rhs = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_lookup_value(
      state, loom_op_const_operands(packet->op)[0], &lhs));
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_lookup_value(
      state, loom_op_const_operands(packet->op)[1], &rhs));

  loom_llvmir_value_id_t llvmir_result = LOOM_LLVMIR_VALUE_ID_INVALID;
  if (info->is_float) {
    IREE_RETURN_IF_ERROR(loom_llvmir_build_fcmp(
        state->llvmir_block,
        &(loom_llvmir_fcmp_desc_t){
            .result_name =
                loom_llvmir_emit_value_name(state->module, result_value),
            .result_type = result_type,
            .predicate = (loom_llvmir_fcmp_predicate_t)info->predicate,
            .lhs = lhs,
            .rhs = rhs,
        },
        &llvmir_result));
  } else {
    IREE_RETURN_IF_ERROR(loom_llvmir_build_icmp(
        state->llvmir_block,
        &(loom_llvmir_icmp_desc_t){
            .result_name =
                loom_llvmir_emit_value_name(state->module, result_value),
            .result_type = result_type,
            .predicate = (loom_llvmir_icmp_predicate_t)info->predicate,
            .lhs = lhs,
            .rhs = rhs,
        },
        &llvmir_result));
  }
  return loom_llvmir_emit_define_value(state, result_value, llvmir_result);
}

static iree_status_t loom_llvmir_emit_cast(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_llvmir_emit_cast_info_t* info) {
  if (packet->op->operand_count != 1) {
    return loom_llvmir_emit_shape_diagnostic(state, packet->op,
                                             IREE_SV("packet_operand"),
                                             packet->op->operand_count, 1);
  }
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_value_id_t result_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_prepare_packet_result(
      state, packet, &result_type, &result_value));
  if (result_type == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();

  loom_llvmir_value_id_t value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_lookup_value(
      state, loom_op_const_operands(packet->op)[0], &value));
  loom_llvmir_value_id_t llvmir_result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_cast(
      state->llvmir_block,
      &(loom_llvmir_cast_desc_t){
          .result_name =
              loom_llvmir_emit_value_name(state->module, result_value),
          .result_type = result_type,
          .op = info->cast_op,
          .value = value,
      },
      &llvmir_result));
  return loom_llvmir_emit_define_value(state, result_value, llvmir_result);
}

static iree_status_t loom_llvmir_emit_select(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet) {
  if (packet->op->operand_count != 3) {
    return loom_llvmir_emit_shape_diagnostic(state, packet->op,
                                             IREE_SV("packet_operand"),
                                             packet->op->operand_count, 3);
  }
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_value_id_t result_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_prepare_packet_result(
      state, packet, &result_type, &result_value));
  if (result_type == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();

  const loom_value_id_t* operands = loom_op_const_operands(packet->op);
  loom_llvmir_value_id_t condition = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t true_value = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t false_value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_emit_lookup_value(state, operands[0], &condition));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_emit_lookup_value(state, operands[1], &true_value));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_emit_lookup_value(state, operands[2], &false_value));
  loom_llvmir_value_id_t llvmir_result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_select(
      state->llvmir_block,
      &(loom_llvmir_select_desc_t){
          .result_name =
              loom_llvmir_emit_value_name(state->module, result_value),
          .result_type = result_type,
          .condition = condition,
          .true_value = true_value,
          .false_value = false_value,
      },
      &llvmir_result));
  return loom_llvmir_emit_define_value(state, result_value, llvmir_result);
}

static iree_status_t loom_llvmir_emit_i64_constant(
    loom_llvmir_emit_function_state_t* state, int64_t value,
    loom_llvmir_value_id_t* out_value) {
  loom_llvmir_type_id_t i64_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(state->llvmir_module, 64, &i64_type));
  return loom_llvmir_module_add_integer_constant(state->llvmir_module, i64_type,
                                                 (uint64_t)value, out_value);
}

static iree_string_view_t loom_llvmir_emit_kernel_coordinate_intrinsic(
    const loom_llvmir_target_profile_t* profile,
    loom_llvmir_emit_kernel_query_kind_t kind,
    loom_llvmir_emit_dimension_t dimension) {
  if (profile->kind != LOOM_LLVMIR_TARGET_PROFILE_KERNEL) {
    return iree_string_view_empty();
  }
  const loom_llvmir_kernel_coordinate_intrinsics_t* intrinsics =
      &profile->kernel.coordinate_intrinsics;
  switch (kind) {
    case LOOM_LLVMIR_EMIT_KERNEL_QUERY_WORKITEM_ID:
      switch (dimension) {
        case LOOM_LLVMIR_EMIT_DIMENSION_X:
          return intrinsics->workitem_id_x;
        case LOOM_LLVMIR_EMIT_DIMENSION_Y:
          return intrinsics->workitem_id_y;
        case LOOM_LLVMIR_EMIT_DIMENSION_Z:
          return intrinsics->workitem_id_z;
      }
      break;
    case LOOM_LLVMIR_EMIT_KERNEL_QUERY_WORKGROUP_ID:
      switch (dimension) {
        case LOOM_LLVMIR_EMIT_DIMENSION_X:
          return intrinsics->workgroup_id_x;
        case LOOM_LLVMIR_EMIT_DIMENSION_Y:
          return intrinsics->workgroup_id_y;
        case LOOM_LLVMIR_EMIT_DIMENSION_Z:
          return intrinsics->workgroup_id_z;
      }
      break;
    case LOOM_LLVMIR_EMIT_KERNEL_QUERY_WORKGROUP_SIZE:
    case LOOM_LLVMIR_EMIT_KERNEL_QUERY_WORKITEM_DISPATCH_ID:
    case LOOM_LLVMIR_EMIT_KERNEL_QUERY_SUBGROUP_SIZE:
    case LOOM_LLVMIR_EMIT_KERNEL_QUERY_SUBGROUP_COUNT:
      break;
  }
  return iree_string_view_empty();
}

static uint32_t loom_llvmir_emit_workgroup_size_dimension(
    const loom_llvmir_workgroup_size_t* size,
    loom_llvmir_emit_dimension_t dimension) {
  switch (dimension) {
    case LOOM_LLVMIR_EMIT_DIMENSION_X:
      return size->x;
    case LOOM_LLVMIR_EMIT_DIMENSION_Y:
      return size->y;
    case LOOM_LLVMIR_EMIT_DIMENSION_Z:
      return size->z;
  }
  return 0;
}

static bool loom_llvmir_emit_flat_workgroup_size(
    const loom_llvmir_workgroup_size_t* size, uint32_t* out_flat_size) {
  *out_flat_size = 0;
  if (size->x == 0 || size->y == 0 || size->z == 0) {
    return false;
  }
  const uint64_t flat_size = (uint64_t)size->x * size->y * size->z;
  if (flat_size == 0 || flat_size > UINT32_MAX) {
    return false;
  }
  *out_flat_size = (uint32_t)flat_size;
  return true;
}

static uint32_t loom_llvmir_emit_ceil_div_u32(uint32_t numerator,
                                              uint32_t denominator) {
  IREE_ASSERT_NE(denominator, 0u);
  return numerator == 0 ? 0 : 1u + (numerator - 1u) / denominator;
}

static iree_status_t loom_llvmir_emit_declare_i32_zero_arg_intrinsic(
    loom_llvmir_emit_function_state_t* state, iree_string_view_t name,
    loom_llvmir_function_t** out_function) {
  *out_function = loom_llvmir_module_find_function(state->llvmir_module, name);
  if (*out_function != NULL) return iree_ok_status();

  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(state->llvmir_module, 32, &i32_type));
  loom_llvmir_function_desc_t desc = {
      .kind = LOOM_LLVMIR_FUNCTION_DECLARATION,
      .name = name,
      .return_type = i32_type,
      .linkage = LOOM_LLVMIR_LINKAGE_DEFAULT,
      .calling_convention = LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT,
      .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
  };
  return loom_llvmir_module_add_function(state->llvmir_module, &desc,
                                         out_function);
}

static iree_status_t loom_llvmir_emit_i32_kernel_coordinate(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    loom_llvmir_emit_kernel_query_kind_t kind,
    loom_llvmir_emit_dimension_t dimension, loom_llvmir_value_id_t* out_value) {
  iree_string_view_t intrinsic_name =
      loom_llvmir_emit_kernel_coordinate_intrinsic(state->target_profile, kind,
                                                   dimension);
  if (iree_string_view_is_empty(intrinsic_name)) {
    *out_value = LOOM_LLVMIR_VALUE_ID_INVALID;
    return loom_llvmir_emit_unsupported_descriptor_diagnostic(state, packet);
  }
  loom_llvmir_function_t* intrinsic = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_declare_i32_zero_arg_intrinsic(
      state, intrinsic_name, &intrinsic));
  return loom_llvmir_build_call(
      state->llvmir_block,
      &(loom_llvmir_call_desc_t){
          .callee = loom_llvmir_function_id(intrinsic),
      },
      out_value);
}

static iree_status_t loom_llvmir_emit_i64_kernel_coordinate(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    loom_llvmir_emit_kernel_query_kind_t kind,
    loom_llvmir_emit_dimension_t dimension, loom_llvmir_type_id_t result_type,
    iree_string_view_t result_name, loom_llvmir_value_id_t* out_value) {
  loom_llvmir_value_id_t i32_coordinate = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_i32_kernel_coordinate(
      state, packet, kind, dimension, &i32_coordinate));
  if (i32_coordinate == LOOM_LLVMIR_VALUE_ID_INVALID) {
    *out_value = LOOM_LLVMIR_VALUE_ID_INVALID;
    return iree_ok_status();
  }
  return loom_llvmir_build_cast(state->llvmir_block,
                                &(loom_llvmir_cast_desc_t){
                                    .result_name = result_name,
                                    .result_type = result_type,
                                    .op = LOOM_LLVMIR_CAST_ZERO_EXTEND,
                                    .value = i32_coordinate,
                                },
                                out_value);
}

static iree_status_t loom_llvmir_emit_workitem_dispatch_id(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    loom_llvmir_emit_dimension_t dimension, loom_llvmir_type_id_t result_type,
    iree_string_view_t result_name, loom_llvmir_value_id_t* out_value) {
  loom_llvmir_value_id_t workgroup_id = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_i64_kernel_coordinate(
      state, packet, LOOM_LLVMIR_EMIT_KERNEL_QUERY_WORKGROUP_ID, dimension,
      result_type, iree_string_view_empty(), &workgroup_id));
  if (workgroup_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
    *out_value = LOOM_LLVMIR_VALUE_ID_INVALID;
    return iree_ok_status();
  }

  loom_llvmir_value_id_t workitem_id = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_i64_kernel_coordinate(
      state, packet, LOOM_LLVMIR_EMIT_KERNEL_QUERY_WORKITEM_ID, dimension,
      result_type, iree_string_view_empty(), &workitem_id));
  if (workitem_id == LOOM_LLVMIR_VALUE_ID_INVALID) {
    *out_value = LOOM_LLVMIR_VALUE_ID_INVALID;
    return iree_ok_status();
  }

  const uint32_t workgroup_size = loom_llvmir_emit_workgroup_size_dimension(
      &state->target_profile->kernel.required_workgroup_size, dimension);
  if (workgroup_size == 0) {
    *out_value = LOOM_LLVMIR_VALUE_ID_INVALID;
    return loom_llvmir_emit_shape_diagnostic(
        state, packet->op, IREE_SV("workgroup_size"), workgroup_size, 1);
  }

  loom_llvmir_value_id_t scaled_workgroup_id = workgroup_id;
  if (workgroup_size != 1) {
    loom_llvmir_value_id_t size_constant = LOOM_LLVMIR_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_llvmir_emit_i64_constant(state, workgroup_size, &size_constant));
    IREE_RETURN_IF_ERROR(
        loom_llvmir_build_binop(state->llvmir_block,
                                &(loom_llvmir_binop_desc_t){
                                    .result_type = result_type,
                                    .op = LOOM_LLVMIR_BINOP_MUL,
                                    .lhs = workgroup_id,
                                    .rhs = size_constant,
                                },
                                &scaled_workgroup_id));
  }

  return loom_llvmir_build_binop(state->llvmir_block,
                                 &(loom_llvmir_binop_desc_t){
                                     .result_name = result_name,
                                     .result_type = result_type,
                                     .op = LOOM_LLVMIR_BINOP_ADD,
                                     .lhs = scaled_workgroup_id,
                                     .rhs = workitem_id,
                                 },
                                 out_value);
}

static iree_status_t loom_llvmir_emit_kernel_query(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_llvmir_emit_kernel_query_info_t* info) {
  if (packet->op->operand_count != 0) {
    return loom_llvmir_emit_shape_diagnostic(state, packet->op,
                                             IREE_SV("packet_operand"),
                                             packet->op->operand_count, 0);
  }
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_value_id_t result_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_prepare_packet_result(
      state, packet, &result_type, &result_value));
  if (result_type == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();

  loom_llvmir_value_id_t llvmir_result = LOOM_LLVMIR_VALUE_ID_INVALID;
  switch (info->kind) {
    case LOOM_LLVMIR_EMIT_KERNEL_QUERY_WORKITEM_ID:
    case LOOM_LLVMIR_EMIT_KERNEL_QUERY_WORKGROUP_ID: {
      IREE_RETURN_IF_ERROR(loom_llvmir_emit_i64_kernel_coordinate(
          state, packet, info->kind, info->dimension, result_type,
          loom_llvmir_emit_value_name(state->module, result_value),
          &llvmir_result));
      break;
    }
    case LOOM_LLVMIR_EMIT_KERNEL_QUERY_WORKGROUP_SIZE: {
      const uint32_t workgroup_size = loom_llvmir_emit_workgroup_size_dimension(
          &state->target_profile->kernel.required_workgroup_size,
          info->dimension);
      if (workgroup_size == 0) {
        return loom_llvmir_emit_shape_diagnostic(
            state, packet->op, IREE_SV("workgroup_size"), workgroup_size, 1);
      }
      IREE_RETURN_IF_ERROR(
          loom_llvmir_emit_i64_constant(state, workgroup_size, &llvmir_result));
      break;
    }
    case LOOM_LLVMIR_EMIT_KERNEL_QUERY_WORKITEM_DISPATCH_ID: {
      IREE_RETURN_IF_ERROR(loom_llvmir_emit_workitem_dispatch_id(
          state, packet, info->dimension, result_type,
          loom_llvmir_emit_value_name(state->module, result_value),
          &llvmir_result));
      break;
    }
    case LOOM_LLVMIR_EMIT_KERNEL_QUERY_SUBGROUP_SIZE: {
      const uint32_t subgroup_size =
          state->target_profile->kernel.subgroup_size;
      if (subgroup_size == 0) {
        return loom_llvmir_emit_shape_diagnostic(
            state, packet->op, IREE_SV("subgroup_size"), subgroup_size, 1);
      }
      IREE_RETURN_IF_ERROR(
          loom_llvmir_emit_i64_constant(state, subgroup_size, &llvmir_result));
      break;
    }
    case LOOM_LLVMIR_EMIT_KERNEL_QUERY_SUBGROUP_COUNT: {
      const uint32_t subgroup_size =
          state->target_profile->kernel.subgroup_size;
      if (subgroup_size == 0) {
        return loom_llvmir_emit_shape_diagnostic(
            state, packet->op, IREE_SV("subgroup_size"), subgroup_size, 1);
      }
      uint32_t flat_workgroup_size = 0;
      if (!loom_llvmir_emit_flat_workgroup_size(
              &state->target_profile->kernel.required_workgroup_size,
              &flat_workgroup_size)) {
        return loom_llvmir_emit_shape_diagnostic(state, packet->op,
                                                 IREE_SV("workgroup_size"),
                                                 flat_workgroup_size, 1);
      }
      const uint32_t subgroup_count =
          loom_llvmir_emit_ceil_div_u32(flat_workgroup_size, subgroup_size);
      IREE_RETURN_IF_ERROR(
          loom_llvmir_emit_i64_constant(state, subgroup_count, &llvmir_result));
      break;
    }
  }
  if (llvmir_result == LOOM_LLVMIR_VALUE_ID_INVALID) return iree_ok_status();
  return loom_llvmir_emit_define_value(state, result_value, llvmir_result);
}

static iree_status_t loom_llvmir_emit_insert_vector_lane(
    loom_llvmir_emit_function_state_t* state, loom_llvmir_type_id_t result_type,
    loom_llvmir_value_id_t vector, loom_llvmir_value_id_t element,
    uint32_t lane, iree_string_view_t result_name,
    loom_llvmir_value_id_t* out_value) {
  loom_llvmir_value_id_t lane_index = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_i64_constant(state, lane, &lane_index));
  return loom_llvmir_build_insert_element(state->llvmir_block,
                                          &(loom_llvmir_insert_element_desc_t){
                                              .result_name = result_name,
                                              .result_type = result_type,
                                              .vector = vector,
                                              .element = element,
                                              .index = lane_index,
                                          },
                                          out_value);
}

static iree_status_t loom_llvmir_emit_extract_vector_lane(
    loom_llvmir_emit_function_state_t* state, loom_llvmir_type_id_t result_type,
    loom_llvmir_value_id_t vector, uint32_t lane,
    iree_string_view_t result_name, loom_llvmir_value_id_t* out_value) {
  loom_llvmir_value_id_t lane_index = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_i64_constant(state, lane, &lane_index));
  return loom_llvmir_build_extract_element(
      state->llvmir_block,
      &(loom_llvmir_extract_element_desc_t){
          .result_name = result_name,
          .result_type = result_type,
          .vector = vector,
          .index = lane_index,
      },
      out_value);
}

static iree_status_t loom_llvmir_emit_const(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_llvmir_emit_const_info_t* info) {
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_value_id_t result_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_prepare_packet_result(
      state, packet, &result_type, &result_value));
  if (result_type == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();

  const uint32_t result_unit_count =
      loom_llvmir_emit_low_value_unit_count(state, result_value);
  if (result_unit_count != info->unit_count) {
    return loom_llvmir_emit_shape_diagnostic(
        state, packet->op, IREE_SV("vector_lane_count"), result_unit_count,
        info->unit_count);
  }
  if (info->unit_count > LOOM_LLVMIR_MAX_VECTOR_LANES) {
    return loom_llvmir_emit_shape_diagnostic(
        state, packet->op, IREE_SV("vector_lane_count"), info->unit_count,
        LOOM_LLVMIR_MAX_VECTOR_LANES);
  }

  int64_t immediate = 0;
  bool has_immediate = false;
  switch (info->kind) {
    case LOOM_LLVMIR_EMIT_CONST_KIND_INTEGER: {
      IREE_RETURN_IF_ERROR(loom_llvmir_emit_read_i64_immediate(
          state, packet, IREE_SV("value"), &has_immediate, &immediate));
      break;
    }
    case LOOM_LLVMIR_EMIT_CONST_KIND_FLOAT_BITS: {
      IREE_RETURN_IF_ERROR(loom_llvmir_emit_read_i64_immediate(
          state, packet, IREE_SV("bits"), &has_immediate, &immediate));
      break;
    }
  }
  if (!has_immediate) return iree_ok_status();

  loom_llvmir_value_id_t constant = LOOM_LLVMIR_VALUE_ID_INVALID;
  switch (info->kind) {
    case LOOM_LLVMIR_EMIT_CONST_KIND_INTEGER: {
      if (info->unit_count == 1) {
        IREE_RETURN_IF_ERROR(loom_llvmir_module_add_integer_constant(
            state->llvmir_module, result_type, (uint64_t)immediate, &constant));
      } else {
        uint64_t values[LOOM_LLVMIR_MAX_VECTOR_LANES] = {0};
        for (uint32_t i = 0; i < info->unit_count; ++i) {
          values[i] = (uint64_t)immediate;
        }
        IREE_RETURN_IF_ERROR(loom_llvmir_module_add_integer_vector_constant(
            state->llvmir_module, result_type, values, info->unit_count,
            &constant));
      }
      break;
    }
    case LOOM_LLVMIR_EMIT_CONST_KIND_FLOAT_BITS: {
      if (info->unit_count == 1) {
        IREE_RETURN_IF_ERROR(loom_llvmir_module_add_float_bits_constant(
            state->llvmir_module, result_type, (uint64_t)immediate, &constant));
      } else {
        loom_llvmir_type_id_t scalar_type = LOOM_LLVMIR_TYPE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_llvmir_emit_core_scalar_type(
            state->llvmir_module, info->value_type,
            /*pointer_address_space=*/0, &scalar_type));
        loom_llvmir_value_id_t scalar_constant = LOOM_LLVMIR_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_llvmir_module_add_float_bits_constant(
            state->llvmir_module, scalar_type, (uint64_t)immediate,
            &scalar_constant));
        IREE_RETURN_IF_ERROR(loom_llvmir_module_add_poison_constant(
            state->llvmir_module, result_type, &constant));
        for (uint32_t i = 0; i < info->unit_count; ++i) {
          loom_llvmir_value_id_t next = LOOM_LLVMIR_VALUE_ID_INVALID;
          IREE_RETURN_IF_ERROR(loom_llvmir_emit_insert_vector_lane(
              state, result_type, constant, scalar_constant, i,
              i + 1 == info->unit_count
                  ? loom_llvmir_emit_value_name(state->module, result_value)
                  : iree_string_view_empty(),
              &next));
          constant = next;
        }
      }
      break;
    }
  }

  return loom_llvmir_emit_define_value(state, result_value, constant);
}

static iree_status_t loom_llvmir_emit_splat(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet) {
  if (packet->op->operand_count != 1) {
    return loom_llvmir_emit_shape_diagnostic(state, packet->op,
                                             IREE_SV("packet_operand"),
                                             packet->op->operand_count, 1);
  }
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_value_id_t result_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_prepare_packet_result(
      state, packet, &result_type, &result_value));
  if (result_type == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();

  const uint32_t lane_count =
      loom_llvmir_emit_low_value_unit_count(state, result_value);
  if (lane_count > LOOM_LLVMIR_MAX_VECTOR_LANES) {
    return loom_llvmir_emit_shape_diagnostic(
        state, packet->op, IREE_SV("vector_lane_count"), lane_count,
        LOOM_LLVMIR_MAX_VECTOR_LANES);
  }

  loom_llvmir_value_id_t scalar = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_lookup_value(
      state, loom_op_const_operands(packet->op)[0], &scalar));
  loom_llvmir_value_id_t current = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_poison_constant(
      state->llvmir_module, result_type, &current));
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_llvmir_value_id_t next = LOOM_LLVMIR_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_insert_vector_lane(
        state, result_type, current, scalar, i,
        i + 1 == lane_count
            ? loom_llvmir_emit_value_name(state->module, result_value)
            : iree_string_view_empty(),
        &next));
    current = next;
  }
  return loom_llvmir_emit_define_value(state, result_value, current);
}

static iree_status_t loom_llvmir_emit_from_elements(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet) {
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_value_id_t result_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_prepare_packet_result(
      state, packet, &result_type, &result_value));
  if (result_type == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();

  const uint32_t lane_count =
      loom_llvmir_emit_low_value_unit_count(state, result_value);
  if (packet->op->operand_count != lane_count) {
    return loom_llvmir_emit_shape_diagnostic(
        state, packet->op, IREE_SV("packet_operand"), packet->op->operand_count,
        lane_count);
  }
  if (lane_count > LOOM_LLVMIR_MAX_VECTOR_LANES) {
    return loom_llvmir_emit_shape_diagnostic(
        state, packet->op, IREE_SV("vector_lane_count"), lane_count,
        LOOM_LLVMIR_MAX_VECTOR_LANES);
  }

  loom_llvmir_value_id_t current = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_poison_constant(
      state->llvmir_module, result_type, &current));
  const loom_value_id_t* operands = loom_op_const_operands(packet->op);
  for (uint32_t i = 0; i < lane_count; ++i) {
    loom_llvmir_value_id_t element = LOOM_LLVMIR_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_llvmir_emit_lookup_value(state, operands[i], &element));
    loom_llvmir_value_id_t next = LOOM_LLVMIR_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_insert_vector_lane(
        state, result_type, current, element, i,
        i + 1 == lane_count
            ? loom_llvmir_emit_value_name(state->module, result_value)
            : iree_string_view_empty(),
        &next));
    current = next;
  }
  return loom_llvmir_emit_define_value(state, result_value, current);
}

static iree_status_t loom_llvmir_emit_concat(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_llvmir_emit_concat_info_t* info) {
  if (packet->op->operand_count != info->input_count) {
    return loom_llvmir_emit_shape_diagnostic(
        state, packet->op, IREE_SV("packet_operand"), packet->op->operand_count,
        info->input_count);
  }
  loom_llvmir_type_id_t result_type_id = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_value_id_t result_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_prepare_packet_result(
      state, packet, &result_type_id, &result_value));
  if (result_type_id == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();

  uint32_t result_lane_count = 0;
  loom_llvmir_type_id_t result_element_type_id = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_vector_type_info(
      state->llvmir_module, result_type_id, &result_lane_count,
      &result_element_type_id));

  loom_llvmir_value_id_t current = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_poison_constant(
      state->llvmir_module, result_type_id, &current));

  uint32_t result_lane = 0;
  const loom_value_id_t* operands = loom_op_const_operands(packet->op);
  for (uint32_t input_ordinal = 0; input_ordinal < info->input_count;
       ++input_ordinal) {
    const loom_value_id_t input_value = operands[input_ordinal];
    loom_llvmir_type_id_t input_type_id = LOOM_LLVMIR_TYPE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_type_for_low_value(
        state, packet->op, input_value, IREE_SV("input"),
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, input_ordinal),
        &input_type_id));
    if (input_type_id == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();
    uint32_t input_lane_count = 0;
    loom_llvmir_type_id_t input_element_type_id = LOOM_LLVMIR_TYPE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_llvmir_module_get_vector_type_info(
        state->llvmir_module, input_type_id, &input_lane_count,
        &input_element_type_id));
    if (input_element_type_id != result_element_type_id) {
      return loom_llvmir_emit_shape_diagnostic(state, packet->op,
                                               IREE_SV("input_type"),
                                               input_type_id, result_type_id);
    }

    loom_llvmir_value_id_t input = LOOM_LLVMIR_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_llvmir_emit_lookup_value(state, input_value, &input));
    for (uint32_t input_lane = 0; input_lane < input_lane_count;
         ++input_lane, ++result_lane) {
      if (result_lane >= result_lane_count) {
        return loom_llvmir_emit_shape_diagnostic(
            state, packet->op, IREE_SV("result_lane"), result_lane,
            result_lane_count);
      }
      loom_llvmir_value_id_t element = LOOM_LLVMIR_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_llvmir_emit_extract_vector_lane(
          state, result_element_type_id, input, input_lane,
          iree_string_view_empty(), &element));
      loom_llvmir_value_id_t next = LOOM_LLVMIR_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_llvmir_emit_insert_vector_lane(
          state, result_type_id, current, element, result_lane,
          result_lane + 1 == result_lane_count
              ? loom_llvmir_emit_value_name(state->module, result_value)
              : iree_string_view_empty(),
          &next));
      current = next;
    }
  }
  if (result_lane != result_lane_count) {
    return loom_llvmir_emit_shape_diagnostic(state, packet->op,
                                             IREE_SV("result_lane_count"),
                                             result_lane, result_lane_count);
  }
  return loom_llvmir_emit_define_value(state, result_value, current);
}

static iree_status_t loom_llvmir_emit_extract(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet) {
  if (packet->op->operand_count != 1) {
    return loom_llvmir_emit_shape_diagnostic(state, packet->op,
                                             IREE_SV("packet_operand"),
                                             packet->op->operand_count, 1);
  }
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_value_id_t result_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_prepare_packet_result(
      state, packet, &result_type, &result_value));
  if (result_type == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();

  int64_t lane = 0;
  bool has_lane = false;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_read_i64_immediate(
      state, packet, IREE_SV("lane"), &has_lane, &lane));
  if (!has_lane) return iree_ok_status();

  const loom_value_id_t* operands = loom_op_const_operands(packet->op);
  loom_llvmir_value_id_t source = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_emit_lookup_value(state, operands[0], &source));
  loom_llvmir_value_id_t llvmir_result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_extract_vector_lane(
      state, result_type, source, (uint32_t)lane,
      loom_llvmir_emit_value_name(state->module, result_value),
      &llvmir_result));
  return loom_llvmir_emit_define_value(state, result_value, llvmir_result);
}

static iree_status_t loom_llvmir_emit_insert(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet) {
  if (packet->op->operand_count != 2) {
    return loom_llvmir_emit_shape_diagnostic(state, packet->op,
                                             IREE_SV("packet_operand"),
                                             packet->op->operand_count, 2);
  }
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_value_id_t result_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_prepare_packet_result(
      state, packet, &result_type, &result_value));
  if (result_type == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();

  int64_t lane = 0;
  bool has_lane = false;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_read_i64_immediate(
      state, packet, IREE_SV("lane"), &has_lane, &lane));
  if (!has_lane) return iree_ok_status();

  const loom_value_id_t* operands = loom_op_const_operands(packet->op);
  loom_llvmir_value_id_t dest = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_emit_lookup_value(state, operands[0], &dest));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_emit_lookup_value(state, operands[1], &value));
  loom_llvmir_value_id_t llvmir_result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_insert_vector_lane(
      state, result_type, dest, value, (uint32_t)lane,
      loom_llvmir_emit_value_name(state->module, result_value),
      &llvmir_result));
  return loom_llvmir_emit_define_value(state, result_value, llvmir_result);
}

static iree_status_t loom_llvmir_emit_dynamic_insert(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet) {
  if (packet->op->operand_count != 3) {
    return loom_llvmir_emit_shape_diagnostic(state, packet->op,
                                             IREE_SV("packet_operand"),
                                             packet->op->operand_count, 3);
  }
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_value_id_t result_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_prepare_packet_result(
      state, packet, &result_type, &result_value));
  if (result_type == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();

  const loom_value_id_t* operands = loom_op_const_operands(packet->op);
  loom_llvmir_value_id_t dest = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t value = LOOM_LLVMIR_VALUE_ID_INVALID;
  loom_llvmir_value_id_t index = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_emit_lookup_value(state, operands[0], &dest));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_emit_lookup_value(state, operands[1], &value));
  IREE_RETURN_IF_ERROR(
      loom_llvmir_emit_lookup_value(state, operands[2], &index));
  loom_llvmir_value_id_t llvmir_result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_insert_element(
      state->llvmir_block,
      &(loom_llvmir_insert_element_desc_t){
          .result_name =
              loom_llvmir_emit_value_name(state->module, result_value),
          .result_type = result_type,
          .vector = dest,
          .element = value,
          .index = index,
      },
      &llvmir_result));
  return loom_llvmir_emit_define_value(state, result_value, llvmir_result);
}

static iree_status_t loom_llvmir_emit_read_shuffle_mask(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet, uint32_t lane_count,
    uint64_t* out_lanes, bool* out_present) {
  *out_present = false;
  for (uint32_t i = 0; i < lane_count; ++i) {
    char lane_name_buffer[16] = {0};
    const int lane_name_length = snprintf(
        lane_name_buffer, IREE_ARRAYSIZE(lane_name_buffer), "lane%" PRIu32, i);
    if (lane_name_length < 0 || (iree_host_size_t)lane_name_length >=
                                    IREE_ARRAYSIZE(lane_name_buffer)) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "failed to format LLVMIR shuffle lane name");
    }
    int64_t lane = 0;
    bool has_lane = false;
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_read_i64_immediate(
        state, packet,
        iree_make_string_view(lane_name_buffer,
                              (iree_host_size_t)lane_name_length),
        &has_lane, &lane));
    if (!has_lane) return iree_ok_status();
    out_lanes[i] = (uint64_t)lane;
  }
  *out_present = true;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_emit_shuffle_like(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    loom_llvmir_emit_shuffle_kind_t kind) {
  if (packet->op->operand_count != 1) {
    return loom_llvmir_emit_shape_diagnostic(state, packet->op,
                                             IREE_SV("packet_operand"),
                                             packet->op->operand_count, 1);
  }
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_value_id_t result_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_prepare_packet_result(
      state, packet, &result_type, &result_value));
  if (result_type == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();

  const uint32_t lane_count =
      loom_llvmir_emit_low_value_unit_count(state, result_value);
  if (lane_count > LOOM_LLVMIR_MAX_VECTOR_LANES) {
    return loom_llvmir_emit_shape_diagnostic(
        state, packet->op, IREE_SV("vector_lane_count"), lane_count,
        LOOM_LLVMIR_MAX_VECTOR_LANES);
  }

  uint64_t lanes[LOOM_LLVMIR_MAX_VECTOR_LANES] = {0};
  switch (kind) {
    case LOOM_LLVMIR_EMIT_SHUFFLE_KIND_EXPLICIT: {
      bool has_lanes = false;
      IREE_RETURN_IF_ERROR(loom_llvmir_emit_read_shuffle_mask(
          state, packet, lane_count, lanes, &has_lanes));
      if (!has_lanes) return iree_ok_status();
      break;
    }
    case LOOM_LLVMIR_EMIT_SHUFFLE_KIND_SLICE: {
      int64_t offset = 0;
      bool has_offset = false;
      IREE_RETURN_IF_ERROR(loom_llvmir_emit_read_i64_immediate(
          state, packet, IREE_SV("offset"), &has_offset, &offset));
      if (!has_offset) return iree_ok_status();
      for (uint32_t i = 0; i < lane_count; ++i) {
        lanes[i] = (uint64_t)(offset + i);
      }
      break;
    }
  }

  const loom_value_id_t source_value = loom_op_const_operands(packet->op)[0];
  loom_llvmir_type_id_t source_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_type_for_low_value(
      state, packet->op, source_value, IREE_SV("source"),
      loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, 0),
      &source_type));
  if (source_type == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();

  loom_llvmir_value_id_t source = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_emit_lookup_value(state, source_value, &source));
  loom_llvmir_value_id_t poison = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_poison_constant(
      state->llvmir_module, source_type, &poison));

  loom_llvmir_type_id_t i32_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(state->llvmir_module, 32, &i32_type));
  loom_llvmir_type_id_t mask_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_vector_type(
      state->llvmir_module, lane_count, i32_type, &mask_type));
  loom_llvmir_value_id_t mask = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_integer_vector_constant(
      state->llvmir_module, mask_type, lanes, lane_count, &mask));

  loom_llvmir_value_id_t llvmir_result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_shuffle_vector(
      state->llvmir_block,
      &(loom_llvmir_shuffle_vector_desc_t){
          .result_name =
              loom_llvmir_emit_value_name(state->module, result_value),
          .result_type = result_type,
          .lhs = source,
          .rhs = poison,
          .mask = mask,
      },
      &llvmir_result));
  return loom_llvmir_emit_define_value(state, result_value, llvmir_result);
}

static iree_status_t loom_llvmir_emit_byte_pointer(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_llvmir_emit_memory_info_t* info, loom_value_id_t base_value_id,
    loom_llvmir_value_id_t base, loom_llvmir_value_id_t* out_pointer) {
  int64_t byte_offset = 0;
  bool has_byte_offset = false;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_read_i64_immediate(
      state, packet, IREE_SV("byte_offset"), &has_byte_offset, &byte_offset));
  if (!has_byte_offset) {
    *out_pointer = LOOM_LLVMIR_VALUE_ID_INVALID;
    return iree_ok_status();
  }

  const bool indexed =
      iree_any_bit_set(info->flags, LOOM_LLVMIR_EMIT_MEMORY_FLAG_INDEXED);
  if (!indexed && byte_offset == 0) {
    *out_pointer = base;
    return iree_ok_status();
  }

  loom_llvmir_value_id_t byte_index = LOOM_LLVMIR_VALUE_ID_INVALID;
  if (indexed) {
    int64_t byte_stride = 0;
    bool has_byte_stride = false;
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_read_i64_immediate(
        state, packet, IREE_SV("byte_stride"), &has_byte_stride, &byte_stride));
    if (!has_byte_stride) {
      *out_pointer = LOOM_LLVMIR_VALUE_ID_INVALID;
      return iree_ok_status();
    }

    loom_llvmir_value_id_t index = LOOM_LLVMIR_VALUE_ID_INVALID;
    const uint32_t index_operand_index =
        iree_any_bit_set(info->flags, LOOM_LLVMIR_EMIT_MEMORY_FLAG_LOAD) ? 1
                                                                         : 2;
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_lookup_value(
        state, loom_op_const_operands(packet->op)[index_operand_index],
        &index));
    if (byte_stride == 1) {
      byte_index = index;
    } else {
      loom_llvmir_type_id_t i64_type = LOOM_LLVMIR_TYPE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_llvmir_module_get_integer_type(
          state->llvmir_module, 64, &i64_type));
      loom_llvmir_value_id_t stride_value = LOOM_LLVMIR_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_llvmir_emit_i64_constant(state, byte_stride, &stride_value));
      IREE_RETURN_IF_ERROR(
          loom_llvmir_build_binop(state->llvmir_block,
                                  &(loom_llvmir_binop_desc_t){
                                      .result_type = i64_type,
                                      .op = LOOM_LLVMIR_BINOP_MUL,
                                      .lhs = index,
                                      .rhs = stride_value,
                                  },
                                  &byte_index));
    }

    if (byte_offset != 0) {
      loom_llvmir_type_id_t i64_type = LOOM_LLVMIR_TYPE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_llvmir_module_get_integer_type(
          state->llvmir_module, 64, &i64_type));
      loom_llvmir_value_id_t offset_value = LOOM_LLVMIR_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_llvmir_emit_i64_constant(state, byte_offset, &offset_value));
      loom_llvmir_value_id_t indexed_offset = LOOM_LLVMIR_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(
          loom_llvmir_build_binop(state->llvmir_block,
                                  &(loom_llvmir_binop_desc_t){
                                      .result_type = i64_type,
                                      .op = LOOM_LLVMIR_BINOP_ADD,
                                      .lhs = byte_index,
                                      .rhs = offset_value,
                                  },
                                  &indexed_offset));
      byte_index = indexed_offset;
    }
  } else {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_emit_i64_constant(state, byte_offset, &byte_index));
  }

  loom_llvmir_type_id_t i8_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  loom_llvmir_type_id_t ptr_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(state->llvmir_module, 8, &i8_type));
  uint32_t pointer_address_space = 0;
  bool supported = true;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_pointer_address_space_for_low_value(
      state, packet->op, base_value_id, IREE_SV("memory_pointer"),
      IREE_SV("reg<llvmir.ptr> memory pointer"), &pointer_address_space,
      &supported));
  if (!supported) {
    *out_pointer = LOOM_LLVMIR_VALUE_ID_INVALID;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_pointer_type(
      state->llvmir_module, pointer_address_space, &ptr_type));
  return loom_llvmir_build_gep(state->llvmir_block,
                               &(loom_llvmir_gep_desc_t){
                                   .result_type = ptr_type,
                                   .element_type = i8_type,
                                   .base = base,
                                   .indices = &byte_index,
                                   .index_count = 1,
                               },
                               out_pointer);
}

static iree_status_t loom_llvmir_emit_alloca(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_llvmir_emit_alloca_info_t* info) {
  if (packet->op->operand_count != 1) {
    return loom_llvmir_emit_shape_diagnostic(state, packet->op,
                                             IREE_SV("packet_operand"),
                                             packet->op->operand_count, 1);
  }
  if (packet->op->result_count != 1) {
    return loom_llvmir_emit_shape_diagnostic(state, packet->op,
                                             IREE_SV("packet_result"),
                                             packet->op->result_count, 1);
  }

  int64_t base_alignment = 0;
  bool has_base_alignment = false;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_read_i64_immediate(
      state, packet, IREE_SV("base_alignment"), &has_base_alignment,
      &base_alignment));
  if (!has_base_alignment) return iree_ok_status();

  uint32_t pointer_address_space = 0;
  if (!loom_llvmir_emit_memory_space_address_space(
          state->target_profile, info->memory_space, &pointer_address_space)) {
    return loom_llvmir_emit_shape_diagnostic(
        state, packet->op, IREE_SV("memory_space_addressable"), 0, 1);
  }

  loom_llvmir_type_id_t i8_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(state->llvmir_module, 8, &i8_type));
  loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_pointer_type(
      state->llvmir_module, pointer_address_space, &result_type));

  loom_llvmir_value_id_t byte_length = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_lookup_value(
      state, loom_op_const_operands(packet->op)[0], &byte_length));

  const loom_value_id_t result_value = loom_op_const_results(packet->op)[0];
  loom_llvmir_value_id_t llvmir_result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_build_alloca(
      state->llvmir_block,
      &(loom_llvmir_alloca_desc_t){
          .result_name =
              loom_llvmir_emit_value_name(state->module, result_value),
          .result_type = result_type,
          .element_type = i8_type,
          .count = byte_length,
          .alignment = (uint32_t)base_alignment,
      },
      &llvmir_result));
  return loom_llvmir_emit_define_value(state, result_value, llvmir_result);
}

static bool loom_llvmir_emit_atomic_ordering(
    int64_t ordering, loom_llvmir_atomic_ordering_t* out_ordering) {
  switch ((loom_atomic_ordering_t)ordering) {
    case LOOM_ATOMIC_ORDERING_RELAXED:
      *out_ordering = LOOM_LLVMIR_ATOMIC_ORDERING_MONOTONIC;
      return true;
    case LOOM_ATOMIC_ORDERING_ACQUIRE:
      *out_ordering = LOOM_LLVMIR_ATOMIC_ORDERING_ACQUIRE;
      return true;
    case LOOM_ATOMIC_ORDERING_RELEASE:
      *out_ordering = LOOM_LLVMIR_ATOMIC_ORDERING_RELEASE;
      return true;
    case LOOM_ATOMIC_ORDERING_ACQ_REL:
      *out_ordering = LOOM_LLVMIR_ATOMIC_ORDERING_ACQ_REL;
      return true;
    case LOOM_ATOMIC_ORDERING_SEQ_CST:
      *out_ordering = LOOM_LLVMIR_ATOMIC_ORDERING_SEQ_CST;
      return true;
    case LOOM_ATOMIC_ORDERING_COUNT_:
      return false;
  }
  return false;
}

static bool loom_llvmir_emit_atomic_sync_scope(
    int64_t scope, iree_string_view_t* out_sync_scope) {
  switch ((loom_atomic_scope_t)scope) {
    case LOOM_ATOMIC_SCOPE_THREAD:
      *out_sync_scope = IREE_SV("singlethread");
      return true;
    case LOOM_ATOMIC_SCOPE_SUBGROUP:
      *out_sync_scope = IREE_SV("subgroup");
      return true;
    case LOOM_ATOMIC_SCOPE_WORKGROUP:
      *out_sync_scope = IREE_SV("workgroup");
      return true;
    case LOOM_ATOMIC_SCOPE_DEVICE:
    case LOOM_ATOMIC_SCOPE_SYSTEM:
      *out_sync_scope = iree_string_view_empty();
      return true;
    case LOOM_ATOMIC_SCOPE_COUNT_:
      return false;
  }
  return false;
}

static iree_status_t loom_llvmir_emit_memory(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_llvmir_emit_memory_info_t* info) {
  const bool is_load =
      iree_any_bit_set(info->flags, LOOM_LLVMIR_EMIT_MEMORY_FLAG_LOAD);
  const uint32_t expected_operands =
      (is_load ? 1u : 2u) +
      (iree_any_bit_set(info->flags, LOOM_LLVMIR_EMIT_MEMORY_FLAG_INDEXED)
           ? 1u
           : 0u);
  if (packet->op->operand_count != expected_operands) {
    return loom_llvmir_emit_shape_diagnostic(
        state, packet->op, IREE_SV("packet_operand"), packet->op->operand_count,
        expected_operands);
  }
  if (packet->op->result_count != (is_load ? 1u : 0u)) {
    return loom_llvmir_emit_shape_diagnostic(
        state, packet->op, IREE_SV("packet_result"), packet->op->result_count,
        is_load ? 1u : 0u);
  }

  loom_llvmir_value_id_t base = LOOM_LLVMIR_VALUE_ID_INVALID;
  const uint32_t pointer_operand_index = is_load ? 0 : 1;
  const loom_value_id_t base_value_id =
      loom_op_const_operands(packet->op)[pointer_operand_index];
  IREE_RETURN_IF_ERROR(
      loom_llvmir_emit_lookup_value(state, base_value_id, &base));
  loom_llvmir_value_id_t pointer = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_byte_pointer(
      state, packet, info, base_value_id, base, &pointer));
  if (pointer == LOOM_LLVMIR_VALUE_ID_INVALID) return iree_ok_status();

  if (is_load) {
    loom_llvmir_type_id_t result_type = LOOM_LLVMIR_TYPE_ID_INVALID;
    loom_value_id_t result_value = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_prepare_packet_result(
        state, packet, &result_type, &result_value));
    if (result_type == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();

    loom_llvmir_value_id_t llvmir_result = LOOM_LLVMIR_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_llvmir_build_load(
        state->llvmir_block,
        &(loom_llvmir_load_desc_t){
            .result_name =
                loom_llvmir_emit_value_name(state->module, result_value),
            .result_type = result_type,
            .pointer = pointer,
        },
        &llvmir_result));
    return loom_llvmir_emit_define_value(state, result_value, llvmir_result);
  }

  loom_llvmir_type_id_t value_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_core_type(
      state->llvmir_module, info->value_type, info->unit_count,
      /*pointer_address_space=*/0, &value_type));
  loom_llvmir_value_id_t value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_lookup_value(
      state, loom_op_const_operands(packet->op)[0], &value));
  return loom_llvmir_build_store(state->llvmir_block,
                                 &(loom_llvmir_store_desc_t){
                                     .value = value,
                                     .pointer = pointer,
                                 });
}

static iree_status_t loom_llvmir_emit_atomic(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet,
    const loom_llvmir_emit_atomic_info_t* info) {
  const bool indexed =
      iree_any_bit_set(info->flags, LOOM_LLVMIR_EMIT_ATOMIC_FLAG_INDEXED);
  const bool is_cmpxchg =
      iree_any_bit_set(info->flags, LOOM_LLVMIR_EMIT_ATOMIC_FLAG_CMPXCHG);
  const bool returns_value =
      iree_any_bit_set(info->flags, LOOM_LLVMIR_EMIT_ATOMIC_FLAG_RETURN_VALUE);
  const uint32_t expected_operands =
      (is_cmpxchg ? 3u : 2u) + (indexed ? 1u : 0u);
  if (packet->op->operand_count != expected_operands) {
    return loom_llvmir_emit_shape_diagnostic(
        state, packet->op, IREE_SV("packet_operand"), packet->op->operand_count,
        expected_operands);
  }
  if (packet->op->result_count != (returns_value ? 1u : 0u)) {
    return loom_llvmir_emit_shape_diagnostic(
        state, packet->op, IREE_SV("packet_result"), packet->op->result_count,
        returns_value ? 1u : 0u);
  }

  loom_llvmir_atomic_ordering_t success_ordering =
      LOOM_LLVMIR_ATOMIC_ORDERING_MONOTONIC;
  loom_llvmir_atomic_ordering_t failure_ordering =
      LOOM_LLVMIR_ATOMIC_ORDERING_MONOTONIC;
  if (is_cmpxchg) {
    int64_t success_ordering_attr = 0;
    bool has_success_ordering = false;
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_read_i64_immediate(
        state, packet, IREE_SV("success_ordering"), &has_success_ordering,
        &success_ordering_attr));
    if (!has_success_ordering) return iree_ok_status();
    if (!loom_llvmir_emit_atomic_ordering(success_ordering_attr,
                                          &success_ordering)) {
      return loom_llvmir_emit_shape_diagnostic(
          state, packet->op, IREE_SV("atomic_success_ordering"),
          (uint32_t)success_ordering_attr, LOOM_ATOMIC_ORDERING_COUNT_);
    }
    int64_t failure_ordering_attr = 0;
    bool has_failure_ordering = false;
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_read_i64_immediate(
        state, packet, IREE_SV("failure_ordering"), &has_failure_ordering,
        &failure_ordering_attr));
    if (!has_failure_ordering) return iree_ok_status();
    if (!loom_llvmir_emit_atomic_ordering(failure_ordering_attr,
                                          &failure_ordering)) {
      return loom_llvmir_emit_shape_diagnostic(
          state, packet->op, IREE_SV("atomic_failure_ordering"),
          (uint32_t)failure_ordering_attr, LOOM_ATOMIC_ORDERING_COUNT_);
    }
  } else {
    int64_t ordering_attr = 0;
    bool has_ordering = false;
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_read_i64_immediate(
        state, packet, IREE_SV("ordering"), &has_ordering, &ordering_attr));
    if (!has_ordering) return iree_ok_status();
    if (!loom_llvmir_emit_atomic_ordering(ordering_attr, &success_ordering)) {
      return loom_llvmir_emit_shape_diagnostic(
          state, packet->op, IREE_SV("atomic_ordering"),
          (uint32_t)ordering_attr, LOOM_ATOMIC_ORDERING_COUNT_);
    }
  }

  int64_t scope_attr = 0;
  bool has_scope = false;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_read_i64_immediate(
      state, packet, IREE_SV("scope"), &has_scope, &scope_attr));
  if (!has_scope) return iree_ok_status();
  iree_string_view_t sync_scope = iree_string_view_empty();
  if (!loom_llvmir_emit_atomic_sync_scope(scope_attr, &sync_scope)) {
    return loom_llvmir_emit_shape_diagnostic(
        state, packet->op, IREE_SV("atomic_scope"), (uint32_t)scope_attr,
        LOOM_ATOMIC_SCOPE_COUNT_);
  }

  loom_llvmir_value_id_t base = LOOM_LLVMIR_VALUE_ID_INVALID;
  const uint32_t pointer_operand_index = is_cmpxchg ? 2u : 1u;
  const loom_value_id_t base_value_id =
      loom_op_const_operands(packet->op)[pointer_operand_index];
  IREE_RETURN_IF_ERROR(
      loom_llvmir_emit_lookup_value(state, base_value_id, &base));
  const loom_llvmir_emit_memory_info_t memory_info = {
      .value_type = info->value_type,
      .unit_count = 1,
      .flags = indexed ? LOOM_LLVMIR_EMIT_MEMORY_FLAG_INDEXED : 0,
  };
  loom_llvmir_value_id_t pointer = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_byte_pointer(
      state, packet, &memory_info, base_value_id, base, &pointer));
  if (pointer == LOOM_LLVMIR_VALUE_ID_INVALID) return iree_ok_status();

  loom_llvmir_type_id_t value_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_core_type(
      state->llvmir_module, info->value_type, /*unit_count=*/1,
      /*pointer_address_space=*/0, &value_type));

  uint32_t alignment = 0;
  if (!loom_llvmir_emit_core_type_byte_count(info->value_type, &alignment)) {
    return loom_llvmir_emit_shape_diagnostic(
        state, packet->op, IREE_SV("atomic_value_type"), 0, 1);
  }

  loom_value_id_t result_value = LOOM_VALUE_ID_INVALID;
  iree_string_view_t result_name = iree_string_view_empty();
  if (returns_value) {
    result_value = loom_op_const_results(packet->op)[0];
    result_name = loom_llvmir_emit_value_name(state->module, result_value);
  }

  if (is_cmpxchg) {
    loom_llvmir_value_id_t expected = LOOM_LLVMIR_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_lookup_value(
        state, loom_op_const_operands(packet->op)[0], &expected));
    loom_llvmir_value_id_t replacement = LOOM_LLVMIR_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_lookup_value(
        state, loom_op_const_operands(packet->op)[1], &replacement));
    loom_llvmir_type_id_t i1_type = LOOM_LLVMIR_TYPE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_llvmir_module_get_integer_type(state->llvmir_module, 1, &i1_type));
    const loom_llvmir_type_id_t aggregate_elements[] = {value_type, i1_type};
    loom_llvmir_type_id_t aggregate_type = LOOM_LLVMIR_TYPE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_llvmir_module_get_struct_type(
        state->llvmir_module, aggregate_elements,
        IREE_ARRAYSIZE(aggregate_elements), &aggregate_type));
    loom_llvmir_value_id_t aggregate = LOOM_LLVMIR_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_llvmir_build_cmpxchg(state->llvmir_block,
                                  &(loom_llvmir_cmpxchg_desc_t){
                                      .result_type = aggregate_type,
                                      .value_type = value_type,
                                      .pointer = pointer,
                                      .expected = expected,
                                      .replacement = replacement,
                                      .success_ordering = success_ordering,
                                      .failure_ordering = failure_ordering,
                                      .sync_scope = sync_scope,
                                      .alignment = alignment,
                                  },
                                  &aggregate));
    loom_llvmir_value_id_t old_value = LOOM_LLVMIR_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_llvmir_build_extract_value(state->llvmir_block,
                                        &(loom_llvmir_extract_value_desc_t){
                                            .result_name = result_name,
                                            .result_type = value_type,
                                            .aggregate = aggregate,
                                            .index = 0,
                                        },
                                        &old_value));
    return loom_llvmir_emit_define_value(state, result_value, old_value);
  }

  loom_llvmir_value_id_t value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_lookup_value(
      state, loom_op_const_operands(packet->op)[0], &value));
  loom_llvmir_value_id_t llvmir_result = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_build_atomic_rmw(state->llvmir_block,
                                   &(loom_llvmir_atomic_rmw_desc_t){
                                       .result_name = result_name,
                                       .result_type = value_type,
                                       .op = info->op,
                                       .pointer = pointer,
                                       .value = value,
                                       .ordering = success_ordering,
                                       .sync_scope = sync_scope,
                                       .alignment = alignment,
                                   },
                                   &llvmir_result));
  if (!returns_value) return iree_ok_status();
  return loom_llvmir_emit_define_value(state, result_value, llvmir_result);
}

static iree_status_t loom_llvmir_emit_packet(
    loom_llvmir_emit_function_state_t* state,
    const loom_low_resolved_descriptor_packet_t* packet) {
  if (!packet->descriptor) {
    return loom_llvmir_emit_missing_descriptor_diagnostic(state, packet);
  }
  const uint32_t descriptor_ref = loom_low_descriptor_set_descriptor_ordinal(
      state->target->descriptor_set, packet->descriptor);
  if (descriptor_ref == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "resolved LLVMIR low descriptor is not a row in "
                            "the selected descriptor set");
  }

  const loom_llvmir_emit_const_info_t* const_info =
      loom_llvmir_emit_lookup_const(descriptor_ref);
  if (const_info) {
    return loom_llvmir_emit_const(state, packet, const_info);
  }

  const loom_llvmir_emit_binary_info_t* binary_info =
      loom_llvmir_emit_lookup_binary(descriptor_ref);
  if (binary_info) {
    return loom_llvmir_emit_binary(state, packet, binary_info);
  }

  const loom_llvmir_emit_binary_intrinsic_info_t* binary_intrinsic_info =
      loom_llvmir_emit_lookup_binary_intrinsic(descriptor_ref);
  if (binary_intrinsic_info) {
    return loom_llvmir_emit_binary_intrinsic(state, packet,
                                             binary_intrinsic_info);
  }

  const loom_llvmir_emit_unary_info_t* unary_info =
      loom_llvmir_emit_lookup_unary(descriptor_ref);
  if (unary_info) {
    return loom_llvmir_emit_unary(state, packet, unary_info);
  }

  const loom_llvmir_emit_unary_intrinsic_info_t* unary_intrinsic_info =
      loom_llvmir_emit_lookup_unary_intrinsic(descriptor_ref);
  if (unary_intrinsic_info) {
    return loom_llvmir_emit_unary_intrinsic(state, packet,
                                            unary_intrinsic_info);
  }

  const loom_llvmir_emit_ternary_intrinsic_info_t* ternary_intrinsic_info =
      loom_llvmir_emit_lookup_ternary_intrinsic(descriptor_ref);
  if (ternary_intrinsic_info) {
    return loom_llvmir_emit_ternary_intrinsic(state, packet,
                                              ternary_intrinsic_info);
  }

  const loom_llvmir_emit_compare_info_t* compare_info =
      loom_llvmir_emit_lookup_compare(descriptor_ref);
  if (compare_info) {
    return loom_llvmir_emit_compare(state, packet, compare_info);
  }

  const loom_llvmir_emit_cast_info_t* cast_info =
      loom_llvmir_emit_lookup_cast(descriptor_ref);
  if (cast_info) {
    return loom_llvmir_emit_cast(state, packet, cast_info);
  }

  if (loom_llvmir_emit_is_select(descriptor_ref)) {
    return loom_llvmir_emit_select(state, packet);
  }

  const loom_llvmir_emit_kernel_query_info_t* kernel_query_info =
      loom_llvmir_emit_lookup_kernel_query(descriptor_ref);
  if (kernel_query_info) {
    return loom_llvmir_emit_kernel_query(state, packet, kernel_query_info);
  }

  if (loom_llvmir_emit_descriptor_ref_in(
          descriptor_ref, kSplatDescriptorRefs,
          IREE_ARRAYSIZE(kSplatDescriptorRefs))) {
    return loom_llvmir_emit_splat(state, packet);
  }

  if (loom_llvmir_emit_descriptor_ref_in(
          descriptor_ref, kFromElementsDescriptorRefs,
          IREE_ARRAYSIZE(kFromElementsDescriptorRefs))) {
    return loom_llvmir_emit_from_elements(state, packet);
  }

  if (loom_llvmir_emit_descriptor_ref_in(
          descriptor_ref, kExtractDescriptorRefs,
          IREE_ARRAYSIZE(kExtractDescriptorRefs))) {
    return loom_llvmir_emit_extract(state, packet);
  }

  if (loom_llvmir_emit_descriptor_ref_in(
          descriptor_ref, kInsertDescriptorRefs,
          IREE_ARRAYSIZE(kInsertDescriptorRefs))) {
    return loom_llvmir_emit_insert(state, packet);
  }

  if (loom_llvmir_emit_descriptor_ref_in(
          descriptor_ref, kDynamicInsertDescriptorRefs,
          IREE_ARRAYSIZE(kDynamicInsertDescriptorRefs))) {
    return loom_llvmir_emit_dynamic_insert(state, packet);
  }

  if (loom_llvmir_emit_descriptor_ref_in(
          descriptor_ref, kShuffleDescriptorRefs,
          IREE_ARRAYSIZE(kShuffleDescriptorRefs))) {
    return loom_llvmir_emit_shuffle_like(
        state, packet, LOOM_LLVMIR_EMIT_SHUFFLE_KIND_EXPLICIT);
  }

  const loom_llvmir_emit_concat_info_t* concat_info =
      loom_llvmir_emit_lookup_concat(descriptor_ref);
  if (concat_info) {
    return loom_llvmir_emit_concat(state, packet, concat_info);
  }

  if (loom_llvmir_emit_descriptor_ref_in(
          descriptor_ref, kSliceDescriptorRefs,
          IREE_ARRAYSIZE(kSliceDescriptorRefs))) {
    return loom_llvmir_emit_shuffle_like(state, packet,
                                         LOOM_LLVMIR_EMIT_SHUFFLE_KIND_SLICE);
  }

  const loom_llvmir_emit_alloca_info_t* alloca_info =
      loom_llvmir_emit_lookup_alloca(descriptor_ref);
  if (alloca_info) {
    return loom_llvmir_emit_alloca(state, packet, alloca_info);
  }

  const loom_llvmir_emit_memory_info_t* memory_info =
      loom_llvmir_emit_lookup_memory(descriptor_ref);
  if (memory_info) {
    return loom_llvmir_emit_memory(state, packet, memory_info);
  }

  const loom_llvmir_emit_atomic_info_t* atomic_info =
      loom_llvmir_emit_lookup_atomic(descriptor_ref);
  if (atomic_info) {
    return loom_llvmir_emit_atomic(state, packet, atomic_info);
  }

  return loom_llvmir_emit_unsupported_descriptor_diagnostic(state, packet);
}

static iree_status_t loom_llvmir_emit_copy(
    loom_llvmir_emit_function_state_t* state, const loom_op_t* op) {
  loom_llvmir_value_id_t source = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_emit_lookup_value(state, loom_low_copy_source(op), &source));
  return loom_llvmir_emit_define_value(state, loom_low_copy_result(op), source);
}

static iree_status_t loom_llvmir_emit_return(
    loom_llvmir_emit_function_state_t* state, const loom_op_t* op) {
  const loom_value_slice_t values = loom_low_return_values(op);
  if (values.count != state->result_count) {
    return loom_llvmir_emit_shape_diagnostic(state, op,
                                             IREE_SV("return_operand"),
                                             values.count, state->result_count);
  }
  if (values.count == 0) {
    return loom_llvmir_build_ret_void(state->llvmir_block);
  }
  loom_llvmir_value_id_t value = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_emit_lookup_value(state, values.values[0], &value));
  return loom_llvmir_build_ret(state->llvmir_block, value);
}

static iree_status_t loom_llvmir_emit_low_op(
    loom_llvmir_emit_function_state_t* state, const loom_op_t* op) {
  if (loom_low_return_isa(op)) {
    return loom_llvmir_emit_return(state, op);
  }
  if (loom_low_resource_isa(op)) {
    return iree_ok_status();
  }
  if (loom_low_copy_isa(op)) {
    return loom_llvmir_emit_copy(state, op);
  }

  loom_low_resolved_descriptor_packet_t packet = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_descriptor_packet(
      state->module, state->target, op, &packet));
  if (packet.kind != LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    return loom_llvmir_emit_packet(state, &packet);
  }
  return loom_llvmir_emit_shape_diagnostic(state, op, IREE_SV("packet_kind"), 0,
                                           1);
}

static iree_status_t loom_llvmir_emit_initialize_value_map(
    loom_llvmir_emit_function_state_t* state) {
  state->value_map_count = state->module->values.count;
  if (state->value_map_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, state->value_map_count, sizeof(*state->value_map),
      (void**)&state->value_map));
  for (iree_host_size_t i = 0; i < state->value_map_count; ++i) {
    state->value_map[i] = LOOM_LLVMIR_VALUE_ID_INVALID;
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_emit_resource_parameter(
    loom_llvmir_emit_function_state_t* state, const loom_op_t* resource_op) {
  const loom_value_id_t result_value = loom_low_resource_result(resource_op);
  uint32_t pointer_address_space = 0;
  bool supported = true;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_pointer_address_space_for_low_value(
      state, resource_op, result_value, IREE_SV("resource"),
      IREE_SV("native_pointer or hal_binding pointer resource"),
      &pointer_address_space, &supported));
  if (!supported) return iree_ok_status();

  loom_llvmir_type_id_t parameter_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_pointer_type(
      state->llvmir_module, pointer_address_space, &parameter_type));

  loom_llvmir_attr_t
      binding_attrs[LOOM_LLVMIR_TARGET_PROFILE_MAX_KERNEL_BINDING_ATTR_COUNT] =
          {{0}};
  iree_host_size_t binding_attr_count = 0;
  if (state->target_profile->kind == LOOM_LLVMIR_TARGET_PROFILE_KERNEL &&
      loom_low_resource_import_kind(resource_op) ==
          LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING) {
    loom_llvmir_target_profile_kernel_binding_attrs(
        state->target_profile, binding_attrs, &binding_attr_count);
  }

  loom_llvmir_value_id_t parameter = LOOM_LLVMIR_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
      state->llvmir_function,
      &(loom_llvmir_parameter_desc_t){
          .type_id = parameter_type,
          .name = loom_llvmir_emit_value_name(state->module, result_value),
          .attrs = binding_attrs,
          .attr_count = binding_attr_count,
      },
      &parameter));
  return loom_llvmir_emit_define_value(state, result_value, parameter);
}

static iree_status_t loom_llvmir_emit_resource_parameters(
    loom_llvmir_emit_function_state_t* state) {
  const loom_block_t* entry_block = loom_region_const_entry_block(state->body);
  loom_op_t* op = NULL;
  loom_block_for_each_op(entry_block, op) {
    if (!loom_low_resource_isa(op)) continue;
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_resource_parameter(state, op));
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_emit_function_signature(
    loom_llvmir_emit_function_state_t* state) {
  const loom_value_slice_t results =
      loom_low_func_def_results(state->function_op);
  state->result_count = results.count;
  if (results.count > 1) {
    return loom_llvmir_emit_shape_diagnostic(state, state->function_op,
                                             IREE_SV("function_result"),
                                             results.count, 1);
  }

  loom_llvmir_type_id_t return_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  if (results.count == 0) {
    IREE_RETURN_IF_ERROR(
        loom_llvmir_module_get_void_type(state->llvmir_module, &return_type));
  } else {
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_type_for_low_value(
        state, state->function_op, results.values[0], IREE_SV("result"),
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, 0),
        &return_type));
  }
  if (return_type == LOOM_LLVMIR_TYPE_ID_INVALID) return iree_ok_status();

  const loom_block_t* entry_block = loom_region_const_entry_block(state->body);
  loom_llvmir_type_id_t* arg_types = NULL;
  if (entry_block->arg_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->scratch_arena, entry_block->arg_count,
                                  sizeof(*arg_types), (void**)&arg_types));
  }
  bool valid_signature = true;
  for (uint16_t i = 0; i < entry_block->arg_count; ++i) {
    const loom_value_id_t arg_value = entry_block->arg_ids[i];
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_type_for_low_value(
        state, state->function_op, arg_value, IREE_SV("parameter"),
        loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, i),
        &arg_types[i]));
    if (arg_types[i] == LOOM_LLVMIR_TYPE_ID_INVALID) {
      valid_signature = false;
    }
  }
  if (!valid_signature) return iree_ok_status();

  const iree_string_view_t export_symbol =
      state->target->bundle_storage.export_plan.export_symbol;
  const iree_string_view_t function_name =
      iree_string_view_is_empty(export_symbol) ? state->function_name
                                               : export_symbol;
  loom_llvmir_attr_group_id_t attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID;
  loom_llvmir_calling_convention_t calling_convention =
      LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT;
  if (state->target_profile->kind == LOOM_LLVMIR_TARGET_PROFILE_KERNEL) {
    IREE_RETURN_IF_ERROR(loom_llvmir_target_profile_add_kernel_attr_group(
        state->llvmir_module, state->target_profile, &attr_group_id));
    calling_convention = state->target_profile->kernel.calling_convention;
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_module_add_function(
      state->llvmir_module,
      &(loom_llvmir_function_desc_t){
          .kind = LOOM_LLVMIR_FUNCTION_DEFINITION,
          .name = function_name,
          .return_type = return_type,
          .linkage = state->target_profile->exported_linkage,
          .calling_convention = calling_convention,
          .attr_group_id = attr_group_id,
      },
      &state->llvmir_function));

  for (uint16_t i = 0; i < entry_block->arg_count; ++i) {
    const loom_value_id_t arg_value = entry_block->arg_ids[i];
    loom_llvmir_value_id_t parameter = LOOM_LLVMIR_VALUE_ID_INVALID;
    loom_llvmir_attr_t binding_attrs
        [LOOM_LLVMIR_TARGET_PROFILE_MAX_KERNEL_BINDING_ATTR_COUNT] = {{0}};
    iree_host_size_t binding_attr_count = 0;
    if (state->target_profile->kind == LOOM_LLVMIR_TARGET_PROFILE_KERNEL &&
        loom_llvmir_emit_low_value_is_pointer_register(state, arg_value)) {
      loom_llvmir_target_profile_kernel_binding_attrs(
          state->target_profile, binding_attrs, &binding_attr_count);
    }
    IREE_RETURN_IF_ERROR(loom_llvmir_function_add_parameter(
        state->llvmir_function,
        &(loom_llvmir_parameter_desc_t){
            .type_id = arg_types[i],
            .name = loom_llvmir_emit_value_name(state->module, arg_value),
            .attrs = binding_attrs,
            .attr_count = binding_attr_count,
        },
        &parameter));
    IREE_RETURN_IF_ERROR(
        loom_llvmir_emit_define_value(state, arg_value, parameter));
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_resource_parameters(state));
  if (state->target_profile->kind == LOOM_LLVMIR_TARGET_PROFILE_KERNEL) {
    IREE_RETURN_IF_ERROR(loom_llvmir_target_profile_attach_kernel_metadata(
        state->llvmir_function, state->target_profile));
  }

  return loom_llvmir_function_add_block(state->llvmir_function,
                                        IREE_SV("entry"), &state->llvmir_block);
}

static iree_status_t loom_llvmir_emit_function_body(
    loom_llvmir_emit_function_state_t* state) {
  if (state->body->block_count != 1) {
    return loom_llvmir_emit_shape_diagnostic(state, state->function_op,
                                             IREE_SV("function_block"),
                                             state->body->block_count, 1);
  }
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_function_signature(state));
  if (state->llvmir_block == NULL) return iree_ok_status();

  const loom_block_t* block = loom_region_const_entry_block(state->body);
  bool has_terminator = false;
  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    IREE_RETURN_IF_ERROR(loom_llvmir_emit_low_op(state, op));
    if (state->module_state->error_count != 0) return iree_ok_status();
    if (loom_low_return_isa(op)) {
      has_terminator = true;
      break;
    }
  }
  if (!has_terminator) {
    return loom_llvmir_emit_shape_diagnostic(state, state->function_op,
                                             IREE_SV("terminator"), 0, 1);
  }
  return iree_ok_status();
}

static iree_string_view_t loom_llvmir_emit_optional_string_attr(
    const loom_module_t* module, const loom_op_t* op, uint8_t attr_index) {
  if (attr_index >= op->attribute_count) return iree_string_view_empty();
  const loom_attribute_t attr = loom_op_const_attrs(op)[attr_index];
  if (attr.kind != LOOM_ATTR_STRING) return iree_string_view_empty();
  return loom_llvmir_emit_string_or_empty(module, loom_attr_as_string_id(attr));
}

static uint32_t loom_llvmir_emit_workgroup_size_dimension_count(
    const loom_target_workgroup_size_t* size) {
  return (size->x != 0 ? 1u : 0u) + (size->y != 0 ? 1u : 0u) +
         (size->z != 0 ? 1u : 0u);
}

static iree_status_t loom_llvmir_emit_prepare_function_profile(
    loom_llvmir_emit_function_state_t* state) {
  const loom_target_snapshot_t* snapshot =
      &state->target->bundle_storage.snapshot;
  const loom_target_export_plan_t* export_plan =
      &state->target->bundle_storage.export_plan;
  iree_string_view_t target_triple = iree_string_view_empty();
  iree_string_view_t data_layout = iree_string_view_empty();
  iree_string_view_t target_cpu = iree_string_view_empty();
  iree_string_view_t target_features = iree_string_view_empty();
  if (state->target->target_op &&
      loom_llvmir_target_isa(state->target->target_op)) {
    target_triple = loom_llvmir_emit_optional_string_attr(
        state->module, state->target->target_op,
        loom_llvmir_target_triple_ATTR_INDEX);
    data_layout = loom_llvmir_emit_optional_string_attr(
        state->module, state->target->target_op,
        loom_llvmir_target_data_layout_ATTR_INDEX);
    target_cpu = loom_llvmir_emit_optional_string_attr(
        state->module, state->target->target_op,
        loom_llvmir_target_cpu_ATTR_INDEX);
    target_features = loom_llvmir_emit_optional_string_attr(
        state->module, state->target->target_op,
        loom_llvmir_target_features_ATTR_INDEX);
  }

  loom_llvmir_target_env_t projected_env = {
      .name = snapshot->name,
      .target_triple = target_triple,
      .data_layout = data_layout,
      .object_format = LOOM_LLVMIR_OBJECT_FORMAT_UNKNOWN,
      .default_pointer_bitwidth = snapshot->default_pointer_bitwidth,
      .index_bitwidth = snapshot->index_bitwidth,
      .offset_bitwidth = snapshot->offset_bitwidth,
      .address_spaces =
          {
              .generic = snapshot->memory_spaces.generic,
              .global = snapshot->memory_spaces.global,
              .local = snapshot->memory_spaces.workgroup,
              .constant = snapshot->memory_spaces.constant,
              .private_memory = snapshot->memory_spaces.private_memory,
              .buffer_resource = snapshot->memory_spaces.descriptor,
          },
  };
  loom_llvmir_target_profile_t projected_profile = {
      .name = export_plan->name,
      .target_env = &projected_env,
      .kind = LOOM_LLVMIR_TARGET_PROFILE_HOST_OBJECT,
      .target_cpu = target_cpu,
      .target_features = target_features,
  };

  switch (export_plan->abi_kind) {
    case LOOM_TARGET_ABI_OBJECT_FUNCTION:
      break;
    case LOOM_TARGET_ABI_HAL_KERNEL: {
      const loom_target_workgroup_size_t* required_workgroup_size =
          &export_plan->hal_kernel.required_workgroup_size;
      if (!loom_target_workgroup_size_is_concrete(required_workgroup_size)) {
        return loom_llvmir_emit_shape_diagnostic(
            state, state->function_op, IREE_SV("hal_kernel_workgroup_size"),
            loom_llvmir_emit_workgroup_size_dimension_count(
                required_workgroup_size),
            3);
      }
      const loom_llvmir_target_profile_t* provider_profile = NULL;
      const loom_llvmir_target_profile_projection_request_t request = {
          .bundle = &state->target->bundle_storage.bundle,
          .target_triple = target_triple,
      };
      if (!loom_llvmir_target_profile_registry_project_bundle(
              state->module_state->target_profile_registry, &request,
              &provider_profile)) {
        return loom_llvmir_emit_projection_diagnostic(state);
      }
      projected_env = *provider_profile->target_env;
      if (!iree_string_view_is_empty(target_triple)) {
        projected_env.target_triple = target_triple;
      }
      if (!iree_string_view_is_empty(data_layout)) {
        projected_env.data_layout = data_layout;
      }
      projected_profile = *provider_profile;
      projected_profile.target_env = &projected_env;
      if (!iree_string_view_is_empty(target_cpu)) {
        projected_profile.target_cpu = target_cpu;
      }
      if (!iree_string_view_is_empty(target_features)) {
        projected_profile.target_features = target_features;
      }
      break;
    }
    default:
      return loom_llvmir_emit_projection_diagnostic(state);
  }

  loom_llvmir_target_profile_storage_initialize_from_bundle(
      &state->target->bundle_storage.bundle, &projected_profile,
      &state->target_profile_storage);
  state->target_profile = &state->target_profile_storage.profile;
  return iree_ok_status();
}

static iree_status_t loom_llvmir_emit_prepare_module(
    loom_llvmir_emit_module_state_t* state,
    const loom_llvmir_target_profile_t* profile, iree_allocator_t allocator) {
  if (state->llvmir_module != NULL) return iree_ok_status();

  loom_llvmir_target_config_t config = {0};
  loom_llvmir_target_profile_module_config(profile, iree_string_view_empty(),
                                           &config);
  config.producer = IREE_SV("loom");
  return loom_llvmir_module_allocate(&config, allocator, &state->llvmir_module);
}

static iree_status_t loom_llvmir_emit_low_function_into_module(
    loom_llvmir_emit_module_state_t* module_state, loom_op_t* low_function_op,
    iree_allocator_t allocator) {
  if (!loom_low_function_def_isa(low_function_op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "LLVMIR emission requires a low function "
                            "definition");
  }

  loom_low_resolved_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target_with_facts(
      module_state->module, &module_state->symbol_facts, low_function_op,
      module_state->descriptor_registry, module_state->target_selection,
      module_state->diagnostic_emitter, &target));
  if (target.descriptor_set == NULL) {
    ++module_state->error_count;
    return iree_ok_status();
  }
  if (target.descriptor_set->stable_id !=
      LLVMIR_GENERIC_CORE_DESCRIPTOR_SET_ID) {
    return loom_llvmir_emit_descriptor_set_diagnostic(
        module_state, low_function_op,
        loom_low_diagnostic_function_name(module_state->module,
                                          low_function_op),
        &target);
  }

  loom_llvmir_emit_function_state_t function_state = {
      .module_state = module_state,
      .module = module_state->module,
      .function_op = low_function_op,
      .body = loom_low_function_const_body(low_function_op),
      .target = &target,
      .function_name = loom_low_diagnostic_function_name(module_state->module,
                                                         low_function_op),
      .scratch_arena = module_state->scratch_arena,
  };
  IREE_RETURN_IF_ERROR(
      loom_llvmir_emit_prepare_function_profile(&function_state));
  if (module_state->error_count != 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_prepare_module(
      module_state, function_state.target_profile, allocator));
  function_state.llvmir_module = module_state->llvmir_module;
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_initialize_value_map(&function_state));
  IREE_RETURN_IF_ERROR(loom_llvmir_emit_function_body(&function_state));
  if (module_state->error_count == 0) {
    ++module_state->function_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_emit_low_module_options_validate(
    const loom_llvmir_emit_low_module_options_t* options) {
  if (options == NULL || options->entry_count == 0) {
    return iree_ok_status();
  }
  if (options->entry_ops == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected LLVMIR low module entries require an entry op list");
  }
  for (iree_host_size_t i = 0; i < options->entry_count; ++i) {
    if (options->entry_ops[i] == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "selected LLVMIR low module entry list contains a null op");
    }
  }
  return iree_ok_status();
}

static bool loom_llvmir_emit_low_module_options_selects_entry(
    const loom_llvmir_emit_low_module_options_t* options,
    loom_op_t* low_function_op) {
  if (options == NULL || options->entry_count == 0) {
    return true;
  }
  for (iree_host_size_t i = 0; i < options->entry_count; ++i) {
    if (options->entry_ops[i] == low_function_op) {
      return true;
    }
  }
  return false;
}

void loom_llvmir_emit_low_module_options_initialize(
    loom_llvmir_emit_low_module_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  *out_options = (loom_llvmir_emit_low_module_options_t){0};
}

iree_status_t loom_llvmir_emit_low_module(
    loom_module_t* module,
    const loom_low_descriptor_registry_t* descriptor_registry,
    loom_target_selection_t target_selection,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* scratch_arena,
    const loom_llvmir_emit_low_module_options_t* options,
    loom_llvmir_module_t** out_module, iree_allocator_t allocator) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(descriptor_registry);
  IREE_ASSERT_ARGUMENT(scratch_arena);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;

  IREE_RETURN_IF_ERROR(loom_llvmir_emit_low_module_options_validate(options));
  loom_llvmir_emit_module_state_t state = {
      .module = module,
      .descriptor_registry = descriptor_registry,
      .target_selection = target_selection,
      .diagnostic_emitter = diagnostic_emitter,
      .scratch_arena = scratch_arena,
      .target_profile_registry =
          options ? options->target_profile_registry : NULL,
  };
  loom_symbol_fact_table_initialize(&state.symbol_facts, scratch_arena);

  iree_status_t status = iree_ok_status();
  loom_op_t* op = NULL;
  loom_block_for_each_op(loom_module_block(module), op) {
    if (!loom_low_function_def_isa(op) ||
        !loom_llvmir_emit_low_module_options_selects_entry(options, op)) {
      continue;
    }
    status = loom_llvmir_emit_low_function_into_module(&state, op, allocator);
    if (!iree_status_is_ok(status) || state.error_count != 0) break;
  }

  if (iree_status_is_ok(status) && state.error_count == 0 &&
      state.function_count == 0) {
    status = loom_llvmir_emit_no_functions_diagnostic(&state);
  }
  if (iree_status_is_ok(status) && state.error_count == 0) {
    *out_module = state.llvmir_module;
    state.llvmir_module = NULL;
  }
  loom_llvmir_module_free(state.llvmir_module);
  return status;
}
