// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable generated source-to-target-low lowering-rule table schema.
//
// Targets use these records to describe source op guards and descriptor-backed
// Low packet emission as compact .rodata. Matching and emission are separate
// interpreters declared by rule_match.h and rule_emit.h.

#ifndef LOOM_CODEGEN_LOW_LOWER_RULES_H_
#define LOOM_CODEGEN_LOW_LOWER_RULES_H_

#include "iree/base/api.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/error/error_defs.h"
#include "loom/ir/encoding.h"
#include "loom/ir/ir.h"
#include "loom/util/string_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint16_t loom_low_lower_descriptor_ref_t;

#define LOOM_LOW_LOWER_DESCRIPTOR_REF_NONE UINT16_MAX

// Returns a scalar element-type bit for type-pattern masks.
#define LOOM_LOW_LOWER_SCALAR_TYPE_BIT(type) \
  ((loom_scalar_type_set_t)(1u << (uint32_t)(type)))

// Bitset of fields checked by a type pattern.
typedef uint8_t loom_low_lower_type_pattern_flags_t;

// Type kind must match type_kind.
#define LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_KIND ((uint8_t)1u << 0)
// Scalar or shaped element type must be in element_type_mask.
#define LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_ELEMENT ((uint8_t)1u << 1)
// Shaped rank must match rank.
#define LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_RANK ((uint8_t)1u << 2)
// First shaped dimension must equal shape.exact.dim0.
#define LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0 ((uint8_t)1u << 3)
// First shaped dimension must be inside the referenced range row.
#define LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM0_RANGE ((uint8_t)1u << 4)
// Second shaped dimension must equal shape.exact.dim1.
#define LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM1 ((uint8_t)1u << 5)
// Total static shaped element count must be inside the referenced range row.
#define LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_ELEMENT_COUNT_RANGE \
  ((uint8_t)1u << 6)

typedef struct loom_low_lower_type_pattern_t {
  // Type fields this pattern checks.
  loom_low_lower_type_pattern_flags_t flags;
  // Required type kind when the KIND flag is set.
  loom_type_kind_t type_kind;
  // Required rank when the RANK flag is set.
  uint8_t rank;
  // Allowed element scalar types when the ELEMENT flag is set.
  loom_scalar_type_set_t element_type_mask;
  // Shape constraint selected by the static-shape flags. Python generation
  // proves that exact dimensions and a range reference are mutually exclusive.
  union {
    // Exact static dimensions used by STATIC_DIM0 and STATIC_DIM1.
    struct {
      // Required first static dimension.
      uint16_t dim0;
      // Required second static dimension.
      uint16_t dim1;
    } exact;
    // One-based wide range row used by the range flags.
    uint16_t range_ref;
  } shape;
} loom_low_lower_type_pattern_t;
static_assert(sizeof(loom_low_lower_type_pattern_t) == 10,
              "lower type-pattern rows must remain compact");

typedef struct loom_low_lower_type_pattern_range_t {
  // Inclusive minimum dimension or total static element count.
  uint64_t minimum;
  // Inclusive maximum dimension or total static element count.
  uint64_t maximum;
} loom_low_lower_type_pattern_range_t;
static_assert(sizeof(loom_low_lower_type_pattern_range_t) == 16,
              "lower type-pattern range rows must be 16 bytes");

typedef uint8_t loom_low_lower_value_ref_kind_t;

enum loom_low_lower_value_ref_kind_e {
  // Invalid or uninitialized value reference.
  LOOM_LOW_LOWER_VALUE_REF_INVALID = 0,
  // Source op operand field |index|, element |element_index|.
  LOOM_LOW_LOWER_VALUE_REF_OPERAND = 1,
  // Source op result field |index|, element |element_index|.
  LOOM_LOW_LOWER_VALUE_REF_RESULT = 2,
  // Rule-local temporary low value at |index|.
  LOOM_LOW_LOWER_VALUE_REF_TEMPORARY = 3,
  // Dynamic source-memory term at |index|.
  LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_TERM = 4,
  // Dynamic byte offset materialized from all selected source-memory terms.
  LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET = 5,
  // Complete target address materialized from one source-memory plan.
  LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ADDRESS = 6,
  // Storage root selected by one source-memory plan.
  LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_ROOT = 7,
  // Complete byte offset including the selected source-memory static bias.
  LOOM_LOW_LOWER_VALUE_REF_SOURCE_MEMORY_BYTE_OFFSET = 8,
  // Exact whole-vector lane origin of source operand field |index|, element
  // |element_index|. Selection proves the origin is available and identity
  // mapped before emission consumes it.
  LOOM_LOW_LOWER_VALUE_REF_EXACT_LANE_ORIGIN_OPERAND = 9,
  // Exact scalar origin shared by every element of source operand field
  // |index|, element |element_index|. Selection proves the indexed uniform
  // origin is available before emission consumes it. The origin may precede
  // an exact widening conversion and have a narrower element type.
  LOOM_LOW_LOWER_VALUE_REF_EXACT_UNIFORM_ELEMENT_ORIGIN_OPERAND = 10,
  // Scalar of the operand's element type shared by every element of source
  // operand field |index|, element |element_index|. This retains widening
  // conversions when the consumer requires the current arithmetic precision.
  LOOM_LOW_LOWER_VALUE_REF_UNIFORM_ELEMENT_ORIGIN_OPERAND = 11,
  // Maximum value-ref kind plus one.
  LOOM_LOW_LOWER_VALUE_REF_COUNT_,
};

static_assert(LOOM_LOW_LOWER_VALUE_REF_COUNT_ <= UINT8_MAX,
              "lower value-ref kinds must fit in uint8_t storage");

// Returns true when the materializer can produce a low value for the source
// value without emitting IR. Selection uses this to keep diagnostics tied to
// the same value-ref row consumed during emission.
typedef iree_status_t (*loom_low_lower_can_materialize_value_fn_t)(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, bool* out_can_materialize);

// Emits or returns the low value consumed by a descriptor operand for the
// source value. The callback only runs when a value-ref row explicitly names
// it.
typedef iree_status_t (*loom_low_lower_materialize_value_fn_t)(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, loom_value_id_t* out_low_value_id);

// Retains a use's source-dependent choices after producer carriers are final.
// The returned opaque recipe belongs to the function arena or immutable target
// storage. NULL is a valid recipe; no source analysis survives into emission.
typedef iree_status_t (*loom_low_lower_prepare_value_materialization_fn_t)(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    const void** out_plan);

// Executes the retained recipe without consulting source analysis. The source
// identity remains available for looking up its canonical emitted carrier.
typedef iree_status_t (*loom_low_lower_emit_value_materialization_fn_t)(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_value_id, const void* plan,
    loom_value_id_t* out_low_value_id);

// Returns the exact carrier produced by a selected materializer. Source
// producers have published their bindings; this query does not emit or choose
// a preferred source mapping.
typedef loom_type_t (*loom_low_lower_materialized_value_type_fn_t)(
    const loom_low_lower_context_t* context, loom_value_id_t source_value_id);

typedef struct loom_low_lower_value_materializer_t {
  // Selection-time predicate proving this materializer can handle the source
  // value without emitting IR.
  loom_low_lower_can_materialize_value_fn_t can_materialize;
  // Native operand type selected before materialization begins.
  loom_low_lower_materialized_value_type_fn_t result_type;
  // Optional preparation of source-dependent choices for this operand use.
  // NULL selects direct emission, which needs no source analysis.
  loom_low_lower_prepare_value_materialization_fn_t prepare;
  // Emission consumes either a retained recipe or only the bound Low carrier.
  union {
    // Used when prepare is NULL; may inspect Low values and target descriptors.
    loom_low_lower_materialize_value_fn_t direct;
    // Used when prepare is present; receives its recipe, including NULL.
    loom_low_lower_emit_value_materialization_fn_t planned;
  } emit;
} loom_low_lower_value_materializer_t;

typedef struct loom_low_lower_rule_descriptor_ref_t {
  // Rule-set string reference for the stable descriptor key.
  loom_string_ref_t key_string_ref;
} loom_low_lower_rule_descriptor_ref_t;
static_assert(sizeof(loom_low_lower_rule_descriptor_ref_t) == 4,
              "loom_low_lower_rule_descriptor_ref_t must be 4 bytes");

typedef struct loom_low_lower_value_ref_t {
  // Source value namespace being referenced.
  loom_low_lower_value_ref_kind_t kind;
  // Source node containing the referenced field. Zero selects the rule root.
  uint8_t source_node_index;
  // Ordinal within the namespace selected by |kind|. For operand and result
  // refs this is the source field index; |element_index| selects within that
  // field.
  uint16_t index;
  // Element ordinal within the operand or result field selected by |index|.
  uint16_t element_index;
  // One-based materializer table row used when this source ref is consumed as a
  // low operand. Zero means direct source-to-low value lookup.
  uint16_t materializer_index;
} loom_low_lower_value_ref_t;
static_assert(sizeof(loom_low_lower_value_ref_t) == 8,
              "loom_low_lower_value_ref_t must be 8 bytes");

#define LOOM_LOW_LOWER_MAX_SOURCE_NODES 8

// Number of low bits occupied by the related source-node count in a packed
// rule source-node span. The root is implicit, so at most seven rows are
// stored for an eight-node source graph.
#define LOOM_LOW_LOWER_SOURCE_NODE_COUNT_BITS 3
#define LOOM_LOW_LOWER_SOURCE_NODE_COUNT_MASK \
  ((1u << LOOM_LOW_LOWER_SOURCE_NODE_COUNT_BITS) - 1u)
#define LOOM_LOW_LOWER_SOURCE_NODE_SPAN(start, count)                        \
  ((uint16_t)(((uint16_t)(start) << LOOM_LOW_LOWER_SOURCE_NODE_COUNT_BITS) | \
              ((uint16_t)(count) & LOOM_LOW_LOWER_SOURCE_NODE_COUNT_MASK)))

typedef uint8_t loom_low_lower_source_node_relation_t;

enum loom_low_lower_source_node_relation_e {
  // The node is the adjacent sole ordinary user of a parent result.
  LOOM_LOW_LOWER_SOURCE_NODE_ADJACENT_UNIQUE_USER = 0,
  // The node is the adjacent sole-use definition of a parent operand.
  LOOM_LOW_LOWER_SOURCE_NODE_ADJACENT_DEFINITION = 1,
};

typedef struct loom_low_lower_source_node_t {
  // SSA relation used to find this source op.
  loom_low_lower_source_node_relation_t relation;
  // Previously resolved node that owns parent_value_ref_index.
  uint8_t parent_node_index;
  // Required source op kind for this node.
  loom_op_kind_t source_op_kind;
  // Value-ref row selecting the connection in the parent node.
  uint16_t parent_value_ref_index;
  // Value-ref row selecting the same connection in this node.
  uint16_t node_value_ref_index;
  // First node-local guard ref.
  uint16_t guard_start;
  // Number of node-local guard refs.
  uint16_t guard_count;
} loom_low_lower_source_node_t;
static_assert(sizeof(loom_low_lower_source_node_t) == 12,
              "loom_low_lower_source_node_t must be 12 bytes");

typedef uint8_t loom_low_lower_attr_copy_kind_t;

enum loom_low_lower_attr_copy_kind_e {
  // Copy the source op attribute directly into the emitted low packet.
  LOOM_LOW_LOWER_ATTR_COPY_DIRECT = 0,
  // Copy one i64_array element as an i64 attribute into the emitted low packet.
  LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_ELEMENT = 1,
  // Packs contiguous i64_array elements into an i64 attribute, with the first
  // source element occupying the least-significant bitfield.
  LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_PACK_ELEMENTS = 2,
  // Emits literal_i64 as an i64 attribute into the emitted low packet.
  LOOM_LOW_LOWER_ATTR_COPY_I64_LITERAL = 3,
  // Emits an exact integer source value fact as an i64 packet attribute.
  LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64 = 4,
  // Emits the negated exact integer source value fact as an i64 packet
  // attribute.
  LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64_NEGATE = 5,
  // Emits log2 of an exact positive power-of-two integer source value fact as
  // an i64 packet attribute.
  LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64_LOG2 = 6,
  // Emits one less than an exact positive integer source value fact as an i64
  // packet attribute.
  LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64_MINUS_ONE = 7,
  // Emits the unsigned magic multiplier for an exact u32 divisor source value
  // fact as an i64 packet attribute. literal_i64 selects multiplier width minus
  // 32: zero retains the corrected high32 recipe; 32 projects ceil(2^64 / d),
  // supporting both high64 quotient and low64/high64 direct remainder.
  LOOM_LOW_LOWER_ATTR_COPY_VALUE_U32_DIVISOR_MAGIC_MULTIPLIER = 8,
  // Emits the unsigned 32-bit magic post-shift plus literal_i64 for an exact
  // divisor source value fact. A full uncorrected 64-bit product adds 32;
  // a high-half 32-bit product (after any correction) adds zero.
  LOOM_LOW_LOWER_ATTR_COPY_VALUE_U32_DIVISOR_MAGIC_SHIFT = 9,
  // Emits an exact signed i32 source value fact as a zero-extended u32 packet
  // attribute bit pattern.
  LOOM_LOW_LOWER_ATTR_COPY_VALUE_I32_AS_U32_BITS = 10,
  // Encodes an exact float fact in the source value's scalar element format
  // as a zero-extended bit pattern. Uniform vectors use their element value;
  // exact canonical NaNs are materializable, but unknown NaN payloads are not.
  LOOM_LOW_LOWER_ATTR_COPY_VALUE_FLOAT_BITS = 11,
  // Expands one i64_array lane ordinal into one byte-lane immediate:
  // source_attr[source_element_index] * source_element_count + literal_i64.
  LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_LANE_BYTE = 12,
  // Emits the selected source-memory static byte offset as an i64 attribute.
  LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_STATIC_BYTE_OFFSET = 13,
  // Emits one selected source-memory dynamic term byte stride as an i64
  // attribute.
  LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_DYNAMIC_BYTE_STRIDE = 14,
  // Emits a source enum attribute ordinal as an i64 packet attribute.
  LOOM_LOW_LOWER_ATTR_COPY_ENUM_ORDINAL = 15,
  // Emits the source op instance flag bitmask as an i64 packet attribute.
  LOOM_LOW_LOWER_ATTR_COPY_SOURCE_OP_INSTANCE_FLAGS = 16,
  // Packs contiguous integer or enum source attributes into an i64 attribute,
  // with the first source attribute occupying the least-significant bitfield.
  LOOM_LOW_LOWER_ATTR_COPY_ATTRS_PACK_CONSECUTIVE = 17,
  // Emits a u32 low-bit mask with width read from one i64 source attribute.
  LOOM_LOW_LOWER_ATTR_COPY_I64_LOW_BIT_MASK = 18,
  // Emits a u32 low-bit mask shifted by another i64 source attribute.
  LOOM_LOW_LOWER_ATTR_COPY_I64_SHIFTED_LOW_BIT_MASK = 19,
  // Emits the u32 inverse of a shifted low-bit mask.
  LOOM_LOW_LOWER_ATTR_COPY_I64_SHIFTED_LOW_BIT_CLEAR_MASK = 20,
  // Emits literal_i64 minus one i64 source attribute.
  LOOM_LOW_LOWER_ATTR_COPY_I64_LITERAL_MINUS_ATTR = 21,
  // Emits literal_i64 minus two i64 source attributes.
  LOOM_LOW_LOWER_ATTR_COPY_I64_LITERAL_MINUS_ATTRS = 22,
  // Emits selected source-memory static byte offset divided by literal_i64.
  LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_STATIC_BYTE_OFFSET_QUOTIENT = 23,
  // Emits selected source-memory static byte offset modulo literal_i64.
  LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_STATIC_BYTE_OFFSET_REMAINDER = 24,
  // Emits log2 of a verified positive power-of-two i64 source attribute.
  LOOM_LOW_LOWER_ATTR_COPY_I64_LOG2 = 25,
  // Emits one i64 source attribute minus literal_i64.
  LOOM_LOW_LOWER_ATTR_COPY_I64_ATTR_MINUS_LITERAL = 26,
  // Emits selected source-memory static byte offset plus literal_i64.
  LOOM_LOW_LOWER_ATTR_COPY_SOURCE_MEMORY_STATIC_BYTE_OFFSET_PLUS_LITERAL = 27,
  // Emits one signed i32 word from an exact i64 source value bit pattern.
  LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64_I32_WORD = 28,
  // Emits one signed i32 word from an exact f64 source value bit pattern.
  LOOM_LOW_LOWER_ATTR_COPY_VALUE_FLOAT_AS_F64_I32_WORD = 29,
  // Emits exact f32 source value bits reinterpreted as a signed i32 packet
  // attribute.
  LOOM_LOW_LOWER_ATTR_COPY_VALUE_FLOAT_AS_F32_I32 = 30,
  // Emits one i64_array element plus literal_i64.
  LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_ELEMENT_PLUS_LITERAL = 31,
  // Emits unsigned 32-bit reciprocal multiplier bits as a signed i32 packet
  // attribute. The divisor's unsigned arithmetic domain is unchanged.
  LOOM_LOW_LOWER_ATTR_COPY_VALUE_U32_DIVISOR_MAGIC_MULTIPLIER_AS_I32 = 32,
  // Emits the source value static dimension selected by source_element_index,
  // multiplied by source_element_count, then adds literal_i64.
  LOOM_LOW_LOWER_ATTR_COPY_VALUE_TYPE_STATIC_DIM_SCALED = 33,
  // Emits literal_i64 minus the source value static dimension selected by
  // source_element_index, multiplied by source_element_count.
  LOOM_LOW_LOWER_ATTR_COPY_VALUE_TYPE_LITERAL_MINUS_STATIC_DIM_SCALED = 34,
  // Emits an i32 low-bit mask whose width is the source value static dimension
  // selected by source_element_index, multiplied by source_element_count, then
  // offset by literal_i64.
  LOOM_LOW_LOWER_ATTR_COPY_VALUE_TYPE_STATIC_DIM_LOW_BITS_MASK = 35,
  // Emits exact f64 source value bits reinterpreted as a signed i64 packet
  // attribute.
  LOOM_LOW_LOWER_ATTR_COPY_VALUE_FLOAT_AS_F64_I64 = 36,
  // Maps a source enum ordinal through a generated packed immediate table.
  LOOM_LOW_LOWER_ATTR_COPY_ENUM_REMAP = 37,
  // Emits one i64_array element divided by a positive literal_i64.
  LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_ELEMENT_QUOTIENT = 38,
  // Emits one i64_array element modulo a positive literal_i64.
  LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_ELEMENT_REMAINDER = 39,
  // Packs eight byte selectors for one PSHUFB source segment. Selectors outside
  // the segment use the high-bit zeroing form.
  LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_SHUFFLE_MASK_CHUNK = 40,
  // Interns all i64_array elements as fixed-width little-endian data and emits
  // the reserved read-only data symbol.
  LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_READ_ONLY_ELEMENTS = 41,
  // Interns byte selectors relative to one fixed source segment and emits the
  // reserved read-only data symbol.
  LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_READ_ONLY_BYTE_SEGMENT = 42,
  // Interns one parity of byte selectors as word indices plus 0/8-bit shifts
  // and emits the reserved read-only data symbol.
  LOOM_LOW_LOWER_ATTR_COPY_I64_ARRAY_READ_ONLY_BYTE_WORDS = 43,
  // Emits the mathematical exponent of an exact signed floating-point power
  // of two as an i64 packet attribute.
  LOOM_LOW_LOWER_ATTR_COPY_VALUE_FLOAT_POWER_OF_TWO_EXPONENT = 44,
  // Emits the negated mathematical exponent of an exact signed floating-point
  // power of two as an i64 packet attribute.
  LOOM_LOW_LOWER_ATTR_COPY_VALUE_FLOAT_POWER_OF_TWO_NEGATED_EXPONENT = 45,
  // Maximum attribute-copy kind plus one.
  LOOM_LOW_LOWER_ATTR_COPY_COUNT_,
};

static_assert(LOOM_LOW_LOWER_ATTR_COPY_COUNT_ <= UINT8_MAX,
              "lower attribute-copy kinds must fit in uint8_t storage");

typedef struct loom_low_lower_attr_copy_t {
  // Literal value emitted by I64_LITERAL rows, byte offset used by
  // I64_ARRAY_LANE_BYTE rows, source segment offset used by
  // SHUFFLE_MASK_CHUNK and READ_ONLY_BYTE_SEGMENT rows, divisor used by
  // SOURCE_MEMORY quotient and remainder rows, divisor used by
  // I64_ARRAY_ELEMENT quotient and remainder rows, the width adjustment for
  // divisor magic projections, or the low 63 bits of a packed enum-remap
  // table.
  int64_t literal_i64;
  // Rule-set string reference for the target low packet attribute name.
  loom_string_ref_t target_name_string_ref;
  // Kind-selected secondary source index.
  union {
    // Second source op attribute ordinal consumed by two-attr projections, or
    // low 16 bits of the packed enum-remap upper word.
    uint16_t other_source_attr_index;
    // Second source value-ref row consumed by two-value projections.
    uint16_t other_value_ref_index;
  };
  // First source i64_array element ordinal, byte parity, i32 word ordinal, or
  // shaped dimension ordinal consumed by the projection row, or high 16 bits
  // of the packed enum-remap upper word.
  uint16_t source_element_index;
  // Source value-ref table row consumed by value projection rows.
  uint16_t value_ref_index;
  // Attribute projection operation to perform.
  loom_low_lower_attr_copy_kind_t kind;
  // Primary source op attribute ordinal consumed by projection rows.
  uint8_t source_attr_index;
  // Number of source elements consumed by PACK_ELEMENTS rows, byte stride used
  // by I64_ARRAY_LANE_BYTE, SHUFFLE_MASK_CHUNK, and READ_ONLY_BYTE_SEGMENT
  // rows, scale used by VALUE_TYPE rows, or source enum case span used by
  // ENUM_REMAP rows.
  uint8_t source_element_count;
  // Bit width of each packed or read-only source element, enum-remap value, or
  // source byte segment length for SHUFFLE_MASK_CHUNK and
  // READ_ONLY_BYTE_SEGMENT rows.
  uint8_t source_element_bit_width;
  // Low bit position of the projected or packed value in the emitted i64.
  uint8_t target_bit_offset;
  // Dynamic source-memory term ordinal consumed by SOURCE_MEMORY rows.
  uint8_t dynamic_term_index;
} loom_low_lower_attr_copy_t;
static_assert(sizeof(loom_low_lower_attr_copy_t) == 24,
              "loom_low_lower_attr_copy_t must be 24 bytes");

typedef uint8_t loom_low_lower_diagnostic_param_kind_t;

#define LOOM_LOW_LOWER_MAX_DIAGNOSTIC_PARAMS 16
#define LOOM_LOW_LOWER_TARGET_CONTEXT_PARAM_COUNT 5

enum loom_low_lower_diagnostic_param_kind_e {
  // Target bundle key selected for lowering.
  LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_TARGET_KEY = 0,
  // Target export name selected for lowering.
  LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_EXPORT_NAME = 1,
  // Target config key selected for lowering.
  LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_CONFIG_KEY = 2,
  // Source function name being lowered.
  LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_FUNCTION_NAME = 3,
  // Source operation name that failed contract selection.
  LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_SOURCE_OP_NAME = 4,
  // String literal stored in string_value.
  LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_STRING_LITERAL = 5,
  // Type of the value referenced by value_ref_index.
  LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_VALUE_TYPE = 6,
  // Signed integer literal stored in i64_value.
  LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_I64_LITERAL = 7,
  // Unsigned 32-bit literal stored in u32_value.
  LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_U32_LITERAL = 8,
  // Unsigned 64-bit literal stored in u64_value.
  LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_U64_LITERAL = 9,
  // Boolean literal stored in bool_value.
  LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_BOOL_LITERAL = 10,
  // Minimum source memory alignment proven for the source op.
  LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_SOURCE_MEMORY_MINIMUM_ALIGNMENT = 11,
};

typedef struct loom_low_lower_diagnostic_param_t {
  // Parameter projection operation.
  loom_low_lower_diagnostic_param_kind_t kind;
  // Reserved padding available to future compact projection metadata.
  uint8_t reserved[7];
  // Projection payload selected by kind.
  union {
    // Rule-set string reference for STRING_LITERAL payloads.
    loom_string_ref_t string_value_ref;
    // Source value-ref row consumed by VALUE_TYPE rows.
    uint16_t value_ref_index;
    // Signed literal payload for I64_LITERAL rows.
    int64_t i64_value;
    // Unsigned literal payload for U32_LITERAL rows.
    uint32_t u32_value;
    // Unsigned literal payload for U64_LITERAL rows.
    uint64_t u64_value;
    // Boolean literal payload for BOOL_LITERAL rows.
    bool bool_value;
  } value;
} loom_low_lower_diagnostic_param_t;
static_assert(sizeof(loom_low_lower_diagnostic_param_t) == 16,
              "loom_low_lower_diagnostic_param_t must be 16 bytes");

// Ordinal into a rule set's interned diagnostic-parameter table.
typedef uint16_t loom_low_lower_diagnostic_param_ref_t;

enum loom_low_lower_diagnostic_flag_bits_e {
  // Materialize the canonical five target-context parameters before stored
  // parameter rows.
  LOOM_LOW_LOWER_DIAGNOSTIC_FLAG_IMPLICIT_TARGET_CONTEXT = 1u << 0,
};
typedef uint8_t loom_low_lower_diagnostic_flags_t;

typedef struct loom_low_lower_diagnostic_t {
  // Stable structured diagnostic identity.
  loom_error_ref_t error_ref;
  // First stored parameter projection ref.
  uint16_t param_start;
  // Total number of materialized parameters, including implicit context.
  uint8_t param_count;
  // Parameter projection behavior flags.
  loom_low_lower_diagnostic_flags_t flags;
} loom_low_lower_diagnostic_t;
static_assert(sizeof(loom_low_lower_diagnostic_t) == 6,
              "loom_low_lower_diagnostic_t must be 6 bytes");

#define LOOM_LOW_LOWER_DIAGNOSTIC_NONE UINT16_MAX

typedef uint16_t loom_low_lower_memory_space_mask_t;

typedef enum loom_low_lower_source_memory_root_kind_e {
  // Root memory value may have any source provenance.
  LOOM_LOW_LOWER_SOURCE_MEMORY_ROOT_ANY = 0,
  // Root memory value must be a source function block argument.
  LOOM_LOW_LOWER_SOURCE_MEMORY_ROOT_BLOCK_ARGUMENT = 1,
  // Root memory value must be produced by buffer.alloca.
  LOOM_LOW_LOWER_SOURCE_MEMORY_ROOT_ALLOCA = 2,
} loom_low_lower_source_memory_root_kind_t;

typedef enum loom_low_lower_source_memory_address_layout_e {
  // The source-memory row accepts any proven or unproven address layout.
  LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_LAYOUT_ANY = 0,
  // The source-memory row requires a proven compact row-major layout.
  LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_LAYOUT_COMPACT_ROW_MAJOR = 1,
} loom_low_lower_source_memory_address_layout_t;

typedef enum loom_low_lower_source_memory_address_base_e {
  // Complete addresses use the source-memory storage root.
  LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_BASE_ROOT = 0,
  // Complete addresses use the planner-owned materialized base view.
  LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_BASE_VIEW = 1,
} loom_low_lower_source_memory_address_base_t;

typedef enum loom_low_lower_source_memory_address_coordinate_e {
  // Address coordinates use the semantic offset carrier.
  LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_COORDINATE_OFFSET = 0,
  // Address coordinates use the semantic index carrier.
  LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_COORDINATE_INDEX = 1,
} loom_low_lower_source_memory_address_coordinate_t;

#define LOOM_LOW_LOWER_MEMORY_SPACE_UNKNOWN \
  ((loom_low_lower_memory_space_mask_t)1u   \
   << LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN)
#define LOOM_LOW_LOWER_MEMORY_SPACE_GLOBAL \
  ((loom_low_lower_memory_space_mask_t)1u  \
   << LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL)
#define LOOM_LOW_LOWER_MEMORY_SPACE_WORKGROUP \
  ((loom_low_lower_memory_space_mask_t)1u     \
   << LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP)
#define LOOM_LOW_LOWER_MEMORY_SPACE_PRIVATE \
  ((loom_low_lower_memory_space_mask_t)1u   \
   << LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE)
#define LOOM_LOW_LOWER_MEMORY_SPACE_CONSTANT \
  ((loom_low_lower_memory_space_mask_t)1u    \
   << LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT)
#define LOOM_LOW_LOWER_MEMORY_SPACE_HOST \
  ((loom_low_lower_memory_space_mask_t)1u << LOOM_VALUE_FACT_MEMORY_SPACE_HOST)
#define LOOM_LOW_LOWER_MEMORY_SPACE_DESCRIPTOR \
  ((loom_low_lower_memory_space_mask_t)1u      \
   << LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR)
#define LOOM_LOW_LOWER_MEMORY_SPACE_GENERIC \
  ((loom_low_lower_memory_space_mask_t)1u   \
   << LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC)

#define LOOM_LOW_LOWER_SOURCE_MEMORY_DYNAMIC_TERM_COUNT_ANY UINT8_MAX
#define LOOM_LOW_LOWER_SOURCE_MEMORY_DYNAMIC_VIEW_BASE_TERM_COUNT_ANY UINT8_MAX

typedef uint8_t loom_low_lower_source_memory_flags_t;

// Accept any byte stride for selected dynamic source-memory terms.
#define LOOM_LOW_LOWER_SOURCE_MEMORY_FLAG_DYNAMIC_BYTE_STRIDE_ANY \
  ((loom_low_lower_source_memory_flags_t)1u << 0)
// Accept selected dynamic source-memory terms with dynamic stride values.
#define LOOM_LOW_LOWER_SOURCE_MEMORY_FLAG_DYNAMIC_STRIDE_VALUES \
  ((loom_low_lower_source_memory_flags_t)1u << 1)
// Consume an original source index only while canonicalization has not moved a
// static contribution from it into the source-memory static byte offset.
#define LOOM_LOW_LOWER_SOURCE_MEMORY_FLAG_PRESERVE_SOURCE_INDEX \
  ((loom_low_lower_source_memory_flags_t)1u << 2)
// Accept any advisory source cache policy.
#define LOOM_LOW_LOWER_SOURCE_MEMORY_FLAG_CACHE_POLICY_ANY \
  ((loom_low_lower_source_memory_flags_t)1u << 3)

// Converts a fixed-width canonical integer term to the address carrier.
// Symbolic analysis preserves signed numeric values (zero/one for i1), even
// when the term originated in an unsigned offset cast. Narrowing is modular;
// source-memory matching owns the complete-address representability proof.
typedef struct loom_low_lower_source_memory_integer_conversion_t {
  // Target features required when this source domain is materialized.
  uint64_t required_features;
  // Literal selector for a descriptor with an explicit conversion immediate.
  int64_t immediate_value;
  // Selector name, or LOOM_STRING_REF_NONE for no selector.
  loom_string_ref_t immediate_string_ref;
  // Numeric conversion, or NONE for compatible carriers with low-unit
  // projection.
  loom_low_lower_descriptor_ref_t descriptor_ref;
  // One for unary conversion; three for i1 select(predicate, one, zero).
  uint8_t input_count;
} loom_low_lower_source_memory_integer_conversion_t;
static_assert(sizeof(loom_low_lower_source_memory_integer_conversion_t) == 24,
              "source-memory integer conversion must be 24 bytes");

// Defines canonical byte-offset arithmetic in the constant descriptor's
// integer carrier. Source terms use the same conversions whether consumed
// directly by a target descriptor or composed with the arithmetic descriptors.
// Source-memory matching owns the complete-address range proof, including
// modular narrowing.
typedef struct loom_low_lower_source_memory_byte_offset_materializer_t {
  // Shared immediate field for arithmetic constants and the static bias.
  loom_string_ref_t constant_immediate_string_ref;
  // Descriptor ref defining the arithmetic carrier and materializing constants.
  loom_low_lower_descriptor_ref_t constant_descriptor_ref;
  // Descriptor ref used to materialize additions in the arithmetic carrier.
  loom_low_lower_descriptor_ref_t add_descriptor_ref;
  // Descriptor ref used to materialize multiplies in the arithmetic carrier.
  loom_low_lower_descriptor_ref_t multiply_descriptor_ref;
  // Descriptor ref used to accumulate a multiplied term, or NONE.
  loom_low_lower_descriptor_ref_t multiply_add_descriptor_ref;
  // Descriptor ref used to materialize the complete static bias, or NONE.
  loom_low_lower_descriptor_ref_t static_bias_descriptor_ref;
  // Descriptor ref used to materialize shifts in the arithmetic carrier.
  loom_low_lower_descriptor_ref_t shift_left_descriptor_ref;
  // Conversions indexed by source scalar kind minus LOOM_SCALAR_TYPE_I1.
  // Address-domain terms already use their target-selected carrier. An absent
  // fixed-width conversion requires the same register class as the arithmetic
  // carrier and permits projection of wider physical register tuples.
  loom_low_lower_source_memory_integer_conversion_t
      integer_conversions[LOOM_SCALAR_TYPE_I64 - LOOM_SCALAR_TYPE_I1 + 1];
} loom_low_lower_source_memory_byte_offset_materializer_t;
static_assert(sizeof(loom_low_lower_source_memory_byte_offset_materializer_t) ==
                  136,
              "source-memory byte-offset materializer must be 136 bytes");

typedef struct loom_low_lower_source_memory_address_materializer_t {
  // Minimum accepted complete address coordinate.
  int64_t coordinate_minimum;
  // Maximum accepted complete address coordinate.
  int64_t coordinate_maximum;
  // Number of bytes represented by one materialized address coordinate unit.
  uint32_t coordinate_unit_byte_count;
  // Rule-set string reference for the coordinate constant immediate field.
  loom_string_ref_t const_coordinate_immediate_string_ref;
  // Descriptor ref used to materialize a complete address coordinate constant.
  loom_low_lower_descriptor_ref_t const_coordinate_descriptor_ref;
  // Descriptor ref used to add complete address coordinate values.
  loom_low_lower_descriptor_ref_t add_coordinate_descriptor_ref;
  // Descriptor ref used to multiply complete address coordinate values.
  loom_low_lower_descriptor_ref_t mul_coordinate_descriptor_ref;
  // Descriptor ref used to shift complete address coordinate values.
  loom_low_lower_descriptor_ref_t shl_coordinate_descriptor_ref;
  // Numeric signed-index conversion to the coordinate carrier.
  loom_low_lower_descriptor_ref_t index_to_coordinate_descriptor_ref;
  // Descriptor ref used to materialize the final target address.
  loom_low_lower_descriptor_ref_t address_descriptor_ref;
  // Source-memory value used as the complete-address base.
  uint8_t base_kind;
  // Semantic source type carried by materialized address coordinates.
  uint8_t coordinate_type;
  // Required fixed-integer conversions, indexed by kind minus I1. Unlike byte
  // offsets, complete addresses reject domains without a declared conversion.
  loom_low_lower_source_memory_integer_conversion_t
      integer_conversions[LOOM_SCALAR_TYPE_I64 - LOOM_SCALAR_TYPE_I1 + 1];
} loom_low_lower_source_memory_address_materializer_t;
static_assert(sizeof(loom_low_lower_source_memory_address_materializer_t) ==
                  160,
              "source-memory address materializer must be 160 bytes");

#define LOOM_LOW_LOWER_SOURCE_MEMORY_MATERIALIZER_NONE ((uint8_t)0)

// Interned diagnostic selection for one source-memory constraint family.
typedef struct loom_low_lower_source_memory_diagnostics_t {
  // Diagnostic selected for each exact source-memory rejection reason.
  uint16_t rejection_diagnostic_indices
      [LOOM_LOW_SOURCE_MEMORY_REJECTION_REASON_COUNT];
} loom_low_lower_source_memory_diagnostics_t;
static_assert(sizeof(loom_low_lower_source_memory_diagnostics_t) == 60,
              "source-memory diagnostic rows must remain compact");

typedef struct loom_low_lower_source_memory_t {
  // Source memory operation category required by this row.
  uint8_t operation_kind;
  // Source provenance required for the root memory value.
  uint8_t root_kind;
  // Target-independent address-layout classification required by this row.
  uint8_t address_layout;
  // Required number of dynamic address terms.
  uint8_t dynamic_term_count;
  // Minimum accepted dynamic term count when dynamic_term_count is ANY.
  uint8_t dynamic_term_count_minimum;
  // Required number of dynamic view-base terms, or ANY if unconstrained.
  uint8_t dynamic_view_base_term_count;
  // Required provenance for each dynamic address term.
  uint8_t dynamic_index_source;
  // Required unsigned width of the complete byte offset, or zero if
  // unconstrained.
  uint8_t byte_offset_unsigned_bit_count;
  // Required unsigned width excluding the static bias, or zero if
  // unconstrained.
  uint8_t dynamic_offset_unsigned_bit_count;
  // One-based dynamic byte-offset materializer row, or zero when unused.
  uint8_t byte_offset_materializer_ordinal;
  // One-based complete-address materializer row, or zero when unused.
  uint8_t address_materializer_ordinal;
  // Bitfield of source-memory row option bits.
  loom_low_lower_source_memory_flags_t flags;
  // Rule-set source-memory diagnostic row index.
  uint16_t diagnostics_index;
  // Accepted target-independent source memory spaces.
  loom_low_lower_memory_space_mask_t memory_space_mask;
  // Rule-set source-memory shape row index.
  uint16_t shape_index;
  // Required static number of vector lanes addressed by the operation.
  uint16_t vector_lane_count;
  // Minimum required final address byte alignment, or zero if unconstrained.
  uint16_t minimum_alignment;
  // Required byte count of one addressed view element.
  uint8_t element_byte_count;
  // Required source cache-policy build flags unless CACHE_POLICY_ANY is set.
  uint8_t cache_policy_build_flags;
} loom_low_lower_source_memory_t;
static_assert(sizeof(loom_low_lower_source_memory_t) == 24,
              "source-memory rows must remain compact");

typedef struct loom_low_lower_source_memory_shape_t {
  // Required byte stride between adjacent vector lanes.
  int64_t vector_lane_byte_stride;
  // Minimum accepted static byte offset from the storage root.
  int64_t static_byte_offset_minimum;
  // Maximum accepted static byte offset from the storage root.
  int64_t static_byte_offset_maximum;
  // Required byte stride for each dynamic address term unless ANY is set.
  int64_t dynamic_byte_stride;
} loom_low_lower_source_memory_shape_t;
static_assert(sizeof(loom_low_lower_source_memory_shape_t) == 32,
              "source-memory shape rows must be 32 bytes");

typedef enum loom_low_lower_guard_kind_e {
  // Invalid or uninitialized guard.
  LOOM_LOW_LOWER_GUARD_INVALID = 0,
  // Source operand/result type must match a type pattern.
  LOOM_LOW_LOWER_GUARD_VALUE_TYPE = 1,
  // Source attribute kind must match attr_kind.
  LOOM_LOW_LOWER_GUARD_ATTR_KIND = 2,
  // Source enum attribute value must match u64.
  LOOM_LOW_LOWER_GUARD_ATTR_ENUM_EQ = 3,
  // Source i64 attribute value must fall in [minimum_i64, maximum_i64].
  LOOM_LOW_LOWER_GUARD_ATTR_I64_RANGE = 4,
  // Selected descriptor set must contain descriptor_ref and its required
  // features must be enabled by the target bundle.
  LOOM_LOW_LOWER_GUARD_DESCRIPTOR_AVAILABLE = 5,
  // Source value ref must be accepted by its configured materializer.
  LOOM_LOW_LOWER_GUARD_VALUE_MATERIALIZABLE = 6,
  // Source value ref must map to a low register with register_class_id.
  LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_CLASS = 7,
  // Source value ref must have a static dim0 divisible by u64.
  LOOM_LOW_LOWER_GUARD_VALUE_STATIC_DIM0_MULTIPLE = 8,
  // Source value refs must map to low registers with equal unit counts.
  LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_UNIT_COUNT_EQ = 9,
  // Source i64_array attribute must contain exactly u64 elements.
  LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_COUNT_EQ = 10,
  // Source i64_array attribute element u64 must fall in
  // [minimum_i64, maximum_i64].
  LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_ELEMENT_RANGE = 11,
  // All source i64_array attribute elements must fall in
  // [minimum_i64, maximum_i64].
  LOOM_LOW_LOWER_GUARD_ATTR_I64_ARRAY_ELEMENTS_RANGE = 12,
  // Source value facts must prove every non-floating integer element fits in a
  // signed integer with |u64| bits.
  LOOM_LOW_LOWER_GUARD_VALUE_SIGNED_BIT_COUNT = 13,
  // Source value facts must prove every non-floating integer element fits in an
  // unsigned integer with |u64| bits.
  LOOM_LOW_LOWER_GUARD_VALUE_UNSIGNED_BIT_COUNT = 14,
  // Source value facts must be an exact non-floating integer.
  LOOM_LOW_LOWER_GUARD_VALUE_EXACT_I64 = 15,
  // Source value facts plus payload.addend must be an exact positive
  // power-of-two integer, without signed overflow.
  LOOM_LOW_LOWER_GUARD_VALUE_EXACT_POWER_OF_TWO_I64 = 16,
  // Source divisor and numerator facts select the reciprocal arithmetic shape
  // in u64, a loom_low_lower_unsigned_divisor_magic_kind_t.
  LOOM_LOW_LOWER_GUARD_VALUE_U32_DIVISOR_MAGIC_KIND = 17,
  // Source value facts must be an exact floating-point value.
  LOOM_LOW_LOWER_GUARD_VALUE_EXACT_FLOAT = 18,
  // Source value facts must prove every non-floating integer element is
  // contained in [minimum_i64, maximum_i64].
  LOOM_LOW_LOWER_GUARD_VALUE_I64_RANGE = 19,
  // Source operand segment starting at selector.attribute.attr_index must
  // contain exactly payload.u64 operands.
  LOOM_LOW_LOWER_GUARD_OPERAND_SEGMENT_COUNT_EQ = 20,
  // Source op instance flags must contain every bit in u64.
  LOOM_LOW_LOWER_GUARD_INSTANCE_FLAGS_HAS_ALL = 21,
  // Source value facts must prove an exact float equal to the declared-width
  // rounding of the f64 literal bit pattern in u64.
  LOOM_LOW_LOWER_GUARD_VALUE_FLOAT_EQUALS = 22,
  // Source value facts must prove every integer element is <= every integer
  // element in the other source value facts.
  LOOM_LOW_LOWER_GUARD_VALUE_I64_RANGE_LE = 23,
  // Source value facts must prove every integer element is >= every integer
  // element in the other source value facts.
  LOOM_LOW_LOWER_GUARD_VALUE_I64_RANGE_GE = 24,
  // Source value's retained schema or type storage element format must match
  // u64.
  LOOM_LOW_LOWER_GUARD_VALUE_STORAGE_ELEMENT_FORMAT = 25,
  // Packed integer storage payload bits must equal source lane count times a
  // source i64 width attribute.
  LOOM_LOW_LOWER_GUARD_VALUE_PACKED_INTEGER_PAYLOAD_FROM_LANES = 26,
  // Packed integer storage payload bits divided by a source i64 width
  // attribute must equal the result lane count.
  LOOM_LOW_LOWER_GUARD_VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD = 27,
  // Source value must have no ordinary operand uses. Type uses are ignored.
  LOOM_LOW_LOWER_GUARD_VALUE_NO_USES = 28,
  // Source value ref must map to a low register with exactly |u64| units.
  LOOM_LOW_LOWER_GUARD_LOW_VALUE_REGISTER_UNIT_COUNT = 29,
  // Source vector.extract op must have a supported source/result/index shape.
  LOOM_LOW_LOWER_GUARD_VECTOR_EXTRACT_SHAPE = 30,
  // Source value refs must have equal static shaped element counts.
  LOOM_LOW_LOWER_GUARD_VALUE_STATIC_ELEMENT_COUNT_EQ = 31,
  // Source buffer/view reference facts must name a space present in u64.
  LOOM_LOW_LOWER_GUARD_VALUE_MEMORY_SPACE = 32,
  // Source op instance flags must contain no bits in u64.
  LOOM_LOW_LOWER_GUARD_INSTANCE_FLAGS_HAS_NONE = 33,
  // Selected target subgroup size must be known and fall in the inclusive
  // payload i64 range.
  LOOM_LOW_LOWER_GUARD_TARGET_SUBGROUP_SIZE_RANGE = 34,
  // Source value must have no ordinary operand use that can dynamically
  // execute after the source operation. Type uses are ignored.
  LOOM_LOW_LOWER_GUARD_VALUE_NO_USES_AFTER = 35,
  // Source value's complete encoded-operand schema must equal the rule-set
  // storage_operand_schemas row selected by selector.value.parameter_index.
  LOOM_LOW_LOWER_GUARD_VALUE_STORAGE_OPERAND_SCHEMA = 36,
  // Retained source value facts prove that the value cannot be NaN.
  LOOM_LOW_LOWER_GUARD_VALUE_NOT_NAN = 37,
  // Source enum attribute value must be present in the u64 bit set.
  LOOM_LOW_LOWER_GUARD_ATTR_ENUM_IN = 38,
  // Source value facts must be an exact signed floating-point power of two
  // whose mathematical exponent is in the inclusive payload i64 range.
  LOOM_LOW_LOWER_GUARD_VALUE_EXACT_POWER_OF_TWO_FLOAT = 39,
  // Two source i64 attributes must sum to payload.i64 without signed overflow.
  LOOM_LOW_LOWER_GUARD_ATTR_I64_SUM_EQ = 40,
  // Source flags permit subnormal flushing or retained facts prove that the
  // selected value cannot be subnormal.
  LOOM_LOW_LOWER_GUARD_VALUE_NOT_SUBNORMAL_OR_INSTANCE_FLAGS_HAS_ALL = 41,
  // Source value must use the physical representation ID in payload.u64.
  // Contract-only queries accept an unselected representation so legality
  // remains independent of function-local representation planning.
  LOOM_LOW_LOWER_GUARD_LOW_VALUE_REPRESENTATION = 42,
  // Maximum guard kind value plus one.
  LOOM_LOW_LOWER_GUARD_COUNT_,
} loom_low_lower_guard_kind_t;

static_assert(LOOM_LOW_LOWER_GUARD_COUNT_ <= UINT8_MAX,
              "lower guard kinds must fit in uint8_t storage");
static_assert(LOOM_ATTR_COUNT_ <= UINT8_MAX,
              "attribute kinds must fit in uint8_t guard storage");

typedef union loom_low_lower_guard_payload_t {
  // Required enum value, divisor recipe kind, expected count, bit-count limit,
  // register unit count, exact f64 bit pattern, flag mask, storage element
  // format, or memory-space mask.
  uint64_t u64;
  // Signed bias applied before testing an exact power of two.
  int64_t addend;
  // Required signed integer value.
  int64_t i64;
  // Inclusive signed range payload.
  struct {
    // Inclusive lower bound.
    int64_t minimum;
    // Inclusive upper bound.
    int64_t maximum;
  } i64_range;
  // Packed integer storage-shape payload.
  struct {
    // Required storage payload multiple or maximum storage unit count.
    uint32_t storage_payload_multiple;
    // Bit count in one packed storage unit.
    uint32_t storage_unit_bit_count;
    // Maximum permitted lane count, or zero when not constrained.
    uint32_t maximum_lane_count;
    // Reserved storage available to future packed integer guards.
    uint32_t reserved;
  } packed_integer;
} loom_low_lower_guard_payload_t;
static_assert(sizeof(loom_low_lower_guard_payload_t) == 16,
              "loom_low_lower_guard_payload_t must be 16 bytes");

typedef struct loom_low_lower_guard_t {
  // Guard operation to evaluate, stored as a loom_low_lower_guard_kind_t.
  uint8_t kind;
  // Required attribute kind for ATTR_KIND guards, stored as a
  // loom_attr_kind_t.
  uint8_t attr_kind;
  // Diagnostic table index emitted when this guard rejects.
  uint16_t diagnostic_index;
  // Kind-selected source and table indices.
  union {
    // Indices used by guards over source values.
    struct {
      // Primary source value-ref table index.
      uint16_t value_ref_index;
      // Second source value-ref table index used by pairwise guards.
      uint16_t other_value_ref_index;
      // Kind-selected type-pattern, register-class, storage-schema, or source
      // attribute index.
      uint16_t parameter_index;
    } value;
    // Indices used by guards over source attributes.
    struct {
      // Source attribute or operand-segment ordinal.
      uint16_t attr_index;
      // Kind-selected second attribute or array-element ordinal.
      union {
        // Second source attribute ordinal used by pairwise attribute guards.
        uint16_t other_attr_index;
        // Source i64-array element ordinal used by element-range guards.
        uint16_t element_index;
      };
      // Reserved storage available to future attribute guards.
      uint16_t reserved;
    } attribute;
    // Rule-set-local descriptor ref used by DESCRIPTOR_AVAILABLE guards.
    struct {
      // Descriptor table reference.
      loom_low_lower_descriptor_ref_t descriptor_ref;
      // Reserved storage available to future descriptor guards.
      uint16_t reserved[2];
    } descriptor;
  } selector;
  // One-based kind-selected payload row. Zero means no payload.
  uint16_t payload_ordinal;
} loom_low_lower_guard_t;
static_assert(sizeof(loom_low_lower_guard_t) == 12,
              "loom_low_lower_guard_t must be 12 bytes");

// Ordinal into a rule set's interned guard table.
typedef uint16_t loom_low_lower_guard_ref_t;

typedef uint8_t loom_low_lower_emit_kind_t;

enum loom_low_lower_emit_kind_e {
  // Invalid or uninitialized emit action.
  LOOM_LOW_LOWER_EMIT_INVALID = 0,
  // Emits a descriptor-backed low.op.
  LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP = 1,
  // Emits a descriptor-backed low.const with copied attributes.
  LOOM_LOW_LOWER_EMIT_DESCRIPTOR_CONST = 2,
  // Slices register-range operands at lane 0 and emits one descriptor-backed
  // low.op.
  LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_FIRST_LANE = 3,
  // Materializes operands once, slices register ranges by descriptor packet
  // widths, emits one low.op per packet, and concatenates each result. A
  // one-packet input is reused; aggregate inputs and results define the packet
  // count. A one-packet result type inherits the aggregate input packet count.
  LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE = 4,
  // Executes the final contiguous emit-program tail once per source lane,
  // slicing operands and typing lane-local temporaries by descriptor packet
  // widths, then concatenates the final lane results. Ordinary emits before
  // the tail provide shared setup values; one-packet operands are broadcast.
  LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_PER_LANE_SEQUENCE = 5,
  // Slices register-range operands, emits one descriptor-backed low.op per
  // register lane, and threads one scalar accumulator operand through the
  // emitted results.
  LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_ACCUMULATE_LANES = 6,
  // Projects consecutive register units with low.slice.
  LOOM_LOW_LOWER_EMIT_REGISTER_SLICE = 7,
  // Concatenates register values with low.concat.
  LOOM_LOW_LOWER_EMIT_REGISTER_CONCAT = 8,
  // Copies a register value into a compatible register class with low.copy.
  LOOM_LOW_LOWER_EMIT_REGISTER_COPY = 9,
  // Transfers a temporary register value to a fresh identity with low.move.
  LOOM_LOW_LOWER_EMIT_REGISTER_MOVE = 10,
  // Maximum emit kind plus one.
  LOOM_LOW_LOWER_EMIT_COUNT_,
};

static_assert(LOOM_LOW_LOWER_EMIT_COUNT_ <= UINT8_MAX,
              "lower emit kinds must fit in uint8_t storage");

typedef uint8_t loom_low_lower_emit_flags_t;

// Swaps emitted descriptor operands 0 and 1 after operand lookup/copy/slicing.
#define LOOM_LOW_LOWER_EMIT_FLAG_SWAP_OPERANDS_0_1 ((uint8_t)1u << 0)
// Binds emitted low results to result_bind_ref_start instead of
// result_type.value_ref_start.
#define LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS ((uint8_t)1u << 1)
// Maps emitted low result types from exact type-pattern rows instead of
// result_type.value_ref_start source value refs.
#define LOOM_LOW_LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN ((uint8_t)1u << 2)
// Seeds DESCRIPTOR_OP_ACCUMULATE_LANES from lane 0 of its accumulator operand
// and starts descriptor emission at lane 1.
#define LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_SEED_FIRST_LANE ((uint8_t)1u << 3)
// Builds DESCRIPTOR_OP_ACCUMULATE_LANES as a balanced tree over source lanes.
#define LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_TREE_BALANCED ((uint8_t)1u << 4)
// Starts DESCRIPTOR_OP_ACCUMULATE_LANES at lane 1. The accumulator operand must
// already contain the folded lane 0 contribution.
#define LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_SKIP_FIRST_LANE ((uint8_t)1u << 5)
// Synthesizes emitted low result types from descriptor result operand rows.
#define LOOM_LOW_LOWER_EMIT_FLAG_RESULT_DESCRIPTOR_TYPE ((uint8_t)1u << 6)
// Records the source access summary for an emitted descriptor with memory
// effects. Address arithmetic consumes the plan without recording an access.
#define LOOM_LOW_LOWER_EMIT_FLAG_RECORD_SOURCE_MEMORY ((uint8_t)1u << 7)

typedef uint8_t loom_low_lower_operand_materialization_t;

enum loom_low_lower_operand_materialization_e {
  // Emits descriptor operands exactly as resolved by the rule.
  LOOM_LOW_LOWER_OPERAND_MATERIALIZATION_DIRECT = 0,
  // Lets the target materialize relationships across the complete operand
  // group after copies, lane projection, and operand permutation.
  LOOM_LOW_LOWER_OPERAND_MATERIALIZATION_TARGET = 1,
};

typedef struct loom_low_lower_emit_t {
  // Emit action to perform.
  loom_low_lower_emit_kind_t kind;
  // Emit behavior flags.
  loom_low_lower_emit_flags_t flags;
  // Rule-set-local low descriptor ref consumed by the selected descriptor set.
  loom_low_lower_descriptor_ref_t descriptor_ref;
  // First value-ref table row copied as a low operand.
  uint16_t operand_ref_start;
  // Bitmask of emitted low operand ordinals copied through low.copy before the
  // descriptor op consumes them. The copy uses the descriptor operand's
  // concrete register class, supporting destructive/tied operands and fixed
  // physical-register constraints without changing source SSA values.
  uint16_t copy_operand_mask;
  // Result-type table range selected by RESULT_TYPE_PATTERN.
  union {
    // First value-ref table row mapped as a low result. Descriptor result
    // types reference source results; structural emits can also inherit an
    // earlier temporary's carrier. BIND_RESULTS_TO_REFS makes
    // result_bind_ref_start control where emitted results are bound.
    uint16_t value_ref_start;
    // First exact type-pattern table row mapped as a low result type.
    uint16_t type_pattern_start;
  } result_type;
  // First value-ref table row receiving emitted low results when
  // BIND_RESULTS_TO_REFS is set.
  uint16_t result_bind_ref_start;
  // One-based source-memory row consumed by this descriptor emit. Zero means
  // the emit does not consume a source memory plan. RECORD_SOURCE_MEMORY marks
  // memory packets; address-only emits use the plan without recording it.
  uint16_t source_memory_ordinal;
  // Kind-specific descriptor or structural emit payload.
  union {
    // Descriptor-backed emit table ranges.
    struct {
      // First attr-copy table row emitted onto the low packet.
      uint16_t attr_copy_start;
      // First tied-result table row forwarded to the low packet builder.
      uint16_t tied_result_start;
    } descriptor;
    // Structural register operation parameters.
    struct {
      // Register-unit offset consumed by REGISTER_SLICE.
      uint16_t offset;
      // Explicit result register-unit count consumed by REGISTER_SLICE. Zero
      // maps the result type through the normal result reference or type
      // pattern.
      uint16_t unit_count;
    } structural;
  } payload;
  // Number of low operands to copy from value-ref rows.
  uint16_t operand_ref_count : 3;
  // Number of low results to map and bind.
  uint16_t result_ref_count : 2;
  // Number of attributes copied onto the low packet.
  uint16_t attr_copy_count : 5;
  // Number of tied-result rows forwarded to the low packet builder.
  uint16_t tied_result_count : 1;
  // Operand ordinal that carries the threaded scalar accumulator for
  // DESCRIPTOR_OP_ACCUMULATE_LANES.
  uint16_t accumulator_operand_index : 2;
  // Operand-group materialization applied before descriptor emission.
  uint16_t operand_materialization : 1;
  // At least one attribute projects a private immutable payload. Planning
  // retains its data ID; execution publishes the symbol used by the packet.
  uint16_t has_read_only_data_attributes : 1;
  // Reserved storage available to future emit parameters.
  uint16_t reserved_count_bits : 1;
} loom_low_lower_emit_t;
static_assert(sizeof(loom_low_lower_emit_t) == 20,
              "loom_low_lower_emit_t must be 20 bytes");

// Returns the table reference receiving one emitted result. Type selection
// and SSA materialization share this exact result-to-source correspondence.
static inline uint16_t loom_low_lower_rule_emit_result_bind_ref_index(
    const loom_low_lower_emit_t* emit, uint16_t result_ordinal) {
  const uint16_t start =
      iree_any_bit_set(emit->flags,
                       LOOM_LOW_LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS)
          ? emit->result_bind_ref_start
          : emit->result_type.value_ref_start;
  return (uint16_t)(start + result_ordinal);
}

// Ordinal into a rule set's interned emit table.
typedef uint16_t loom_low_lower_emit_ref_t;

typedef uint8_t loom_low_lower_rule_flags_t;

// Rule row is a read-only target contract case and must not be selected as an
// emission program by source-to-low.
#define LOOM_LOW_LOWER_RULE_FLAG_CONTRACT_ONLY \
  ((loom_low_lower_rule_flags_t)1u << 0)

// Alias-ref pair names operand and result fields whose values alias by ordinal.
#define LOOM_LOW_LOWER_RULE_FLAG_ORDINAL_VALUE_ALIAS \
  ((loom_low_lower_rule_flags_t)1u << 1)

// Rule row has no structured report key.
#define LOOM_LOW_LOWER_RULE_REPORT_KEY_NONE ((uint16_t)0)

// Emission rule has no descriptor-bearing primary emit.
#define LOOM_LOW_LOWER_RULE_PRIMARY_EMIT_NONE ((uint16_t)UINT16_MAX)

typedef struct loom_low_lower_rule_t {
  // Packed first related source-node row and row count. The root source op is
  // implicit node zero.
  uint16_t source_node_span;
  // First guard-ref row for this rule.
  uint16_t guard_start;
  // Rule action range selected by its nonzero count.
  union {
    // First emit-reference row for this rule's program.
    uint16_t emit_start;
    // First value-ref pair whose source operand aliases a source result.
    uint16_t alias_ref_start;
    // First value-ref table row whose source result is intentionally erased.
    uint16_t elide_ref_start;
  } action;
  // Number of emit-reference rows for this rule's program.
  uint16_t emit_count;
  // Action-specific metadata sharing storage across disjoint rule kinds.
  union {
    // Metadata for descriptor-emission rules.
    struct {
      // Ordinal of the descriptor emit representing the rule's primary action.
      uint16_t primary_emit_ordinal;
    } emit;
    // Metadata for value alias and elision rules.
    struct {
      // Number of operand/result alias pairs consumed by this rule.
      uint8_t alias_ref_count;
      // Number of source result refs erased by this rule.
      uint8_t elide_ref_count;
    } value;
  } metadata;
  // One-based report-key table ordinal. Zero means the selected rule has no
  // stable strategy key for compile reports.
  uint8_t report_key_ordinal;
  // Number of liveness-packed rule-local temporary slots.
  uint8_t temporary_count;
  // Rule behavior flags.
  loom_low_lower_rule_flags_t flags;
  // Number of guard refs for this rule.
  uint8_t guard_count;
} loom_low_lower_rule_t;
static_assert(sizeof(loom_low_lower_rule_t) == 14,
              "loom_low_lower_rule_t must be 14 bytes");

static inline uint16_t loom_low_lower_rule_source_node_start(
    const loom_low_lower_rule_t* rule) {
  return (uint16_t)(rule->source_node_span >>
                    LOOM_LOW_LOWER_SOURCE_NODE_COUNT_BITS);
}

static inline uint8_t loom_low_lower_rule_source_node_count(
    const loom_low_lower_rule_t* rule) {
  return (uint8_t)(rule->source_node_span &
                   LOOM_LOW_LOWER_SOURCE_NODE_COUNT_MASK);
}

typedef struct loom_low_lower_rule_span_t {
  // Source op kind covered by this contiguous rule range.
  loom_op_kind_t source_op_kind;
  // First rule table row for source_op_kind.
  uint16_t rule_start;
  // Number of rules for source_op_kind.
  uint16_t rule_count;
} loom_low_lower_rule_span_t;

typedef uint16_t loom_low_lower_rule_set_flags_t;

// Rule set can answer read-only target contract queries before source-to-low
// emission. Rule sets without this flag are emission helpers whose legality is
// still owned by target-local family analysis.
#define LOOM_LOW_LOWER_RULE_SET_FLAG_TARGET_CONTRACT_QUERY \
  ((loom_low_lower_rule_set_flags_t)1u << 0)

typedef struct loom_low_lower_rule_set_t {
  // Rule-set behavior flags.
  loom_low_lower_rule_set_flags_t flags;
  // Packed generated strings referenced by rule-set-local table rows.
  loom_string_pool_t string_pool;
  // Source op kind to rule-span lookup table sorted by source_op_kind.
  const loom_low_lower_rule_span_t* spans;
  // Number of rows in spans.
  uint16_t span_count;
  // Rule rows referenced by spans.
  const loom_low_lower_rule_t* rules;
  // Number of rows in rules.
  uint16_t rule_count;
  // Rule-set string references referenced by one-based report-key ordinals.
  const loom_string_ref_t* report_key_string_refs;
  // Number of rows in report_key_string_refs.
  uint16_t report_key_count;
  // Type-pattern rows referenced by guards.
  const loom_low_lower_type_pattern_t* type_patterns;
  // Number of rows in type_patterns.
  uint16_t type_pattern_count;
  // Interned wide ranges referenced by type-pattern rows.
  const loom_low_lower_type_pattern_range_t* type_pattern_ranges;
  // Number of rows in type_pattern_ranges.
  uint16_t type_pattern_range_count;
  // Source value-reference rows referenced by guards and emits.
  const loom_low_lower_value_ref_t* value_refs;
  // Number of rows in value_refs.
  uint16_t value_ref_count;
  // Related source-op rows referenced by rules.
  const loom_low_lower_source_node_t* source_nodes;
  // Number of rows in source_nodes.
  uint16_t source_node_count;
  // Target-owned value materializers referenced by one-based value refs.
  const loom_low_lower_value_materializer_t* materializers;
  // Number of rows in materializers.
  uint16_t materializer_count;
  // Source-memory rows referenced by emits.
  const loom_low_lower_source_memory_t* source_memories;
  // Number of rows in source_memories.
  uint16_t source_memory_count;
  // Interned shape rows referenced by source_memories.
  const loom_low_lower_source_memory_shape_t* source_memory_shapes;
  // Number of rows in source_memory_shapes.
  uint16_t source_memory_shape_count;
  // Interned diagnostic selections referenced by source memories.
  const loom_low_lower_source_memory_diagnostics_t* source_memory_diagnostics;
  // Number of rows in source_memory_diagnostics.
  uint16_t source_memory_diagnostic_count;
  // Interned dynamic byte-offset materializers referenced by source memories.
  const loom_low_lower_source_memory_byte_offset_materializer_t*
      source_memory_byte_offset_materializers;
  // Number of rows in source_memory_byte_offset_materializers.
  uint8_t source_memory_byte_offset_materializer_count;
  // Interned complete-address materializers referenced by source memories.
  const loom_low_lower_source_memory_address_materializer_t*
      source_memory_address_materializers;
  // Number of rows in source_memory_address_materializers.
  uint8_t source_memory_address_materializer_count;
  // Descriptor refs referenced by guards and emits.
  const loom_low_lower_rule_descriptor_ref_t* descriptor_refs;
  // Number of rows in descriptor_refs.
  uint16_t descriptor_ref_count;
  // Interned diagnostic parameter projection rows.
  const loom_low_lower_diagnostic_param_t* diagnostic_params;
  // Number of rows in diagnostic_params.
  uint16_t diagnostic_param_count;
  // Diagnostic parameter refs addressed by diagnostic param spans.
  const loom_low_lower_diagnostic_param_ref_t* diagnostic_param_refs;
  // Number of rows in diagnostic_param_refs.
  uint16_t diagnostic_param_ref_count;
  // Interned immediate and range payloads referenced by guards.
  const loom_low_lower_guard_payload_t* guard_payloads;
  // Number of rows in guard_payloads.
  uint16_t guard_payload_count;
  // Interned guard rows referenced by guard_refs.
  const loom_low_lower_guard_t* guards;
  // Number of rows in guards.
  uint16_t guard_count;
  // Exact encoded-operand schemas referenced by guards.
  const loom_encoding_operand_summary_t* storage_operand_schemas;
  // Number of rows in storage_operand_schemas.
  uint16_t storage_operand_schema_count;
  // Guard refs addressed by rule guard spans.
  const loom_low_lower_guard_ref_t* guard_refs;
  // Number of rows in guard_refs.
  uint16_t guard_ref_count;
  // Attribute-copy rows referenced by emits.
  const loom_low_lower_attr_copy_t* attr_copies;
  // Number of rows in attr_copies.
  uint16_t attr_copy_count;
  // Tied-result rows referenced by emits.
  const loom_tied_result_t* tied_results;
  // Number of rows in tied_results.
  uint16_t tied_result_count;
  // Emit refs addressed by rule emit-program spans.
  const loom_low_lower_emit_ref_t* emit_refs;
  // Number of rows in emit_refs.
  uint16_t emit_ref_count;
  // Interned emit rows referenced by emit_refs.
  const loom_low_lower_emit_t* emits;
  // Number of rows in emits.
  uint16_t emit_count;
  // Diagnostic rows referenced by guards.
  const loom_low_lower_diagnostic_t* diagnostics;
  // Number of rows in diagnostics.
  uint16_t diagnostic_count;
} loom_low_lower_rule_set_t;

// Resolves one emit-program position to its interned emit row.
static inline const loom_low_lower_emit_t* loom_low_lower_rule_set_emit_at(
    const loom_low_lower_rule_set_t* rule_set, uint16_t emit_ref_index) {
  return &rule_set->emits[rule_set->emit_refs[emit_ref_index]];
}

// Resolves the trusted one-based payload referenced by |guard|.
static inline const loom_low_lower_guard_payload_t*
loom_low_lower_rule_set_guard_payload(const loom_low_lower_rule_set_t* rule_set,
                                      const loom_low_lower_guard_t* guard) {
  return &rule_set->guard_payloads[guard->payload_ordinal - 1];
}

// Resolves the trusted byte-offset materializer referenced by |source_memory|.
static inline const loom_low_lower_source_memory_byte_offset_materializer_t*
loom_low_lower_rule_set_source_memory_byte_offset_materializer(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_source_memory_t* source_memory) {
  return &rule_set->source_memory_byte_offset_materializers
              [source_memory->byte_offset_materializer_ordinal - 1];
}

// Resolves the trusted diagnostic selection referenced by |source_memory|.
static inline const loom_low_lower_source_memory_diagnostics_t*
loom_low_lower_rule_set_source_memory_diagnostics(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_source_memory_t* source_memory) {
  return &rule_set->source_memory_diagnostics[source_memory->diagnostics_index];
}

// Resolves the trusted complete-address materializer referenced by
// |source_memory|.
static inline const loom_low_lower_source_memory_address_materializer_t*
loom_low_lower_rule_set_source_memory_address_materializer(
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_source_memory_t* source_memory) {
  return &rule_set->source_memory_address_materializers
              [source_memory->address_materializer_ordinal - 1];
}

// Returns the static view for a trusted rule-set string reference.
static inline iree_string_view_t loom_low_lower_rule_set_string(
    const loom_low_lower_rule_set_t* rule_set, loom_string_ref_t string_ref) {
  return loom_string_pool_get(&rule_set->string_pool, string_ref);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_RULES_H_
