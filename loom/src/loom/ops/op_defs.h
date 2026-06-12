// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core type definitions for the loom op infrastructure.
//
// This header defines the types that both hand-written and generated dialect
// code depend on: attributes, format elements, field descriptors, op vtables,
// and supporting enums. Generated per-dialect ops.h files include this header.
// Pass authors include the per-dialect ops.h, which transitively includes this.
//
// The op system is table-driven: each op kind has a vtable in .rodata
// containing format element arrays (4-byte instructions for the printer/parser
// interpreter), field descriptors, trait bits, and a B-string name. Adding ops
// adds .rodata tables, not .text code.

#ifndef LOOM_OPS_OP_DEFS_H_
#define LOOM_OPS_OP_DEFS_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/ir/semantics.h"
#include "loom/ir/types.h"
#include "loom/util/bstring.h"

// Annotation for parameters that may be NULL or zero. No-op macro that
// improves readability at call sites and in generated declarations.
#define loom_optional

// Annotation for operand parameters that may be consumed by a tied result.
// The operand's storage may be reused by a result of this op (linear
// ownership transfer). After calling the builder, the annotated operand
// must not be used if it was tied to a result.
#define loom_may_consume

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Value slices
//===----------------------------------------------------------------------===//

// A typed range of value IDs. Returned by variadic operand/result accessors
// and used by generic pass infrastructure. One type for all variadic fields
// across all ops.
typedef struct loom_value_slice_t {
  loom_value_id_t* values;
  uint16_t count;
} loom_value_slice_t;

// A typed range of region pointers. Returned by variadic region accessors.
typedef struct loom_region_slice_t {
  loom_region_t** regions;
  uint8_t count;
} loom_region_slice_t;

// A typed range of successor block pointers. Returned by variadic successor
// accessors on CFG terminators.
typedef struct loom_successor_slice_t {
  loom_block_t** blocks;
  uint8_t count;
} loom_successor_slice_t;

// Returns the value ID at |index| in the slice. |index| must be less than
// |slice.count|.
static inline loom_value_id_t loom_value_slice_get(loom_value_slice_t slice,
                                                   uint16_t index) {
  IREE_ASSERT(index < slice.count);
  return slice.values[index];
}

// Sets the value ID at |index| in the slice. |index| must be less than
// |slice.count|.
static inline void loom_value_slice_set(loom_value_slice_t slice,
                                        uint16_t index, loom_value_id_t value) {
  IREE_ASSERT(index < slice.count);
  slice.values[index] = value;
}

// Replaces all occurrences of |old_value| with |new_value| in the slice.
static inline void loom_value_slice_replace(loom_value_slice_t slice,
                                            loom_value_id_t old_value,
                                            loom_value_id_t new_value) {
  for (uint16_t i = 0; i < slice.count; ++i) {
    if (slice.values[i] == old_value) {
      slice.values[i] = new_value;
    }
  }
}

// A keyed SSA value used by OperandDict builders. The name is static metadata
// identifying the dictionary entry; the value remains an ordinary operand and
// participates in the normal use-def lists.
typedef struct loom_named_value_t {
  // Interned dictionary key in the destination module string table.
  loom_string_id_t name_id;
  // Reserved for stable ABI alignment and must be zero.
  uint32_t reserved;
  // SSA value ID stored in the variadic operand segment.
  loom_value_id_t value_id;
} loom_named_value_t;

// A borrowed range of named SSA values.
typedef struct loom_named_value_slice_t {
  // Borrowed entry pointer. NULL is valid only when count is zero.
  const loom_named_value_t* entries;
  // Number of entries in the borrowed range.
  iree_host_size_t count;
} loom_named_value_slice_t;

// Bundles |entries| and |count| as a named value slice.
static inline loom_named_value_slice_t loom_make_named_value_slice(
    const loom_named_value_t* entries, iree_host_size_t count) {
  loom_named_value_slice_t slice = {
      /*.entries=*/count > 0 ? entries : NULL,
      /*.count=*/count,
  };
  return slice;
}

//===----------------------------------------------------------------------===//
// Format elements
//===----------------------------------------------------------------------===//

// Instructions for the format-element-walking printer and parser. The generic
// printer has one switch statement over format element kinds. Each op's format
// element array is the instruction stream for that switch. Adding ops adds
// .rodata format arrays, not .text code.
enum loom_format_kind_e {
  // Single operand reference: %name.
  LOOM_FORMAT_KIND_OPERAND_REF = 0,
  // Variadic operand references: %a, %b, %c.
  LOOM_FORMAT_KIND_OPERAND_REFS = 1,
  // Attribute value: 42, 3.14, slt, "hello".
  LOOM_FORMAT_KIND_ATTR_VALUE = 2,
  // Symbol reference attribute: @name.
  LOOM_FORMAT_KIND_SYMBOL_REF = 3,
  // Type of an operand: f32, tile<4xf32>.
  LOOM_FORMAT_KIND_OPERAND_TYPE = 4,
  // Type of a result: f32, tile<4xf32>.
  LOOM_FORMAT_KIND_RESULT_TYPE = 5,
  // Types of a variadic operand: f32, tile<4xf32>, i32.
  LOOM_FORMAT_KIND_OPERAND_TYPES = 6,
  // Result type list with tied handling: -> (type, %op as type).
  LOOM_FORMAT_KIND_RESULT_TYPE_LIST = 7,
  // Literal keyword token: , : -> to step else do.
  LOOM_FORMAT_KIND_KEYWORD = 8,
  // Optional attribute dictionary: {key = value, ...}.
  // If data has LOOM_ATTR_DICT_FORMAT_INLINE_ATTRS, dictionary entries map to
  // declared op attrs by key and field_index is ignored. Otherwise field_index
  // references one named LOOM_ATTR_DICT attribute.
  LOOM_FORMAT_KIND_ATTR_DICT = 9,
  // Nested region. data = loom_region_syntax_t selector.
  LOOM_FORMAT_KIND_REGION = 10,
  // Mixed static/dynamic index list: [0, %x, 4].
  // field_index = dynamic operand index, data = LOOM_FORMAT_INDEX_LIST_DATA.
  LOOM_FORMAT_KIND_INDEX_LIST = 11,
  // Named value bindings: (%a = %x : type, ...).
  // data = binding kind (CAPTURE or ELEMENT).
  LOOM_FORMAT_KIND_BINDING_LIST = 12,
  // Function argument definitions: (%a: type, %b: type).
  LOOM_FORMAT_KIND_FUNC_ARGS = 13,
  // Where-clause predicates: [mul(%M, 16), ...].
  LOOM_FORMAT_KIND_PREDICATE_LIST = 14,
  // Optional group marker. field_index = anchor field index.
  // data = (skip_count << 2) | anchor_category.
  // The walker skips |skip_count| elements when the anchor is absent.
  LOOM_FORMAT_KIND_OPTIONAL_GROUP = 15,
  // Suppress space before the next token.
  LOOM_FORMAT_KIND_GLUE = 16,
  // Per-instance flags in angle brackets: <flag1|flag2>.
  // Glued to the preceding token (op name). field_index indexes into
  // the vtable's instance_flags_case_names. Reads/writes op->instance_flags.
  LOOM_FORMAT_KIND_FLAGS = 17,

  // Op kind reference in angle brackets: <tile.contract>.
  // Glued to the preceding token (op name). The field_index references
  // a string attribute storing the op name.
  LOOM_FORMAT_KIND_OP_REF = 18,

  // Single result type without parentheses: type.
  // For ops with exactly one non-variadic result where parenthesized
  // list syntax would be misleading. No tied-result support — use
  // RESULT_TYPE_LIST for ops that need tied-result syntax.
  LOOM_FORMAT_KIND_RESULT_TYPE_SINGLE = 19,

  // Scoped group of format elements. Pushes a new name scope before
  // processing children and pops it after. Within the scope, type
  // parsing uses definition mode: [%name] creates new index-typed
  // values rather than requiring existing names. Used for global
  // definitions where type annotations introduce named type variables.
  // data = child_count (number of following format elements in scope).
  LOOM_FORMAT_KIND_SCOPE = 20,

  // Required compile-time op parameter in angle brackets: <add>.
  // Glued to the preceding token (op name). The field_index references
  // an ordinary attribute parsed with the attr descriptor.
  LOOM_FORMAT_KIND_TEMPLATE_PARAM = 21,

  // Required compile-time op parameter plus optional instance flags:
  // <add> or <add, flag1|flag2>. Glued to the preceding token (op name).
  // field_index references the ordinary parameter attribute. Instance flags
  // read/write op->instance_flags.
  LOOM_FORMAT_KIND_TEMPLATE_PARAM_FLAGS = 22,

  // Keyed variadic operand dictionary: {key = %value : type, ...}.
  // field_index = variadic operand start, data = dict attr field index storing
  // key -> operand ordinal relative to field_index. The dict stores only
  // integer ordinals, never SSA value IDs.
  LOOM_FORMAT_KIND_OPERAND_DICT = 23,

  // Static-attribute-keyed value table:
  // {0 = (%a, %b), 1 = (%c, %d)} default(%x, %y).
  // field_index = variadic operand start, data = i64 array attr field index
  // storing the row keys. The operand field stores row payloads flattened in
  // row-major order, followed by one default row.
  LOOM_FORMAT_KIND_ATTR_TABLE = 24,

  // Static-attribute-keyed region table:
  // { case 0 { ... } case 1 { ... } default { ... } }.
  // field_index = variadic case region start. data packs the i64 array attr
  // field index storing row keys and the fixed default region index using
  // LOOM_FORMAT_REGION_TABLE_DATA.
  LOOM_FORMAT_KIND_REGION_TABLE = 25,

  // Region entry block arguments: (%a: type, %b: type).
  // field_index = region index whose entry block args are printed or parsed.
  LOOM_FORMAT_KIND_BLOCK_ARGS = 26,

  // CFG successor block reference: ^label.
  // field_index = successor index whose target block is printed or parsed.
  LOOM_FORMAT_KIND_SUCCESSOR_REF = 27,

  // Descriptor key reference in angle brackets: <amdgpu.v_add_u32>. The
  // field_index references the diagnostic string attribute and data references
  // the descriptor-set-local ordinal attribute. Text parsing stores -1 because
  // descriptor refs are resolved after target binding selects a descriptor set.
  LOOM_FORMAT_KIND_DESCRIPTOR_REF = 28,

  // Stable symbolic key reference in angle brackets: <source.key>. The
  // field_index references the diagnostic string attribute and data references
  // the derived i64 stable-key attribute. This is for non-descriptor symbolic
  // domains; descriptor-backed packets use LOOM_FORMAT_KIND_DESCRIPTOR_REF.
  LOOM_FORMAT_KIND_STABLE_KEY_REF = 29,

  // Variadic operand references with adjacent type annotations:
  // %a: type, %b: type.
  LOOM_FORMAT_KIND_OPERAND_TYPED_REFS = 30,
};
typedef uint8_t loom_format_kind_t;

// Individual flag bits packed into INDEX_LIST format element data.
enum loom_format_index_list_data_bits_e {
  LOOM_FORMAT_INDEX_LIST_DATA_NO_LEADING_GLUE = 1u << 15,
};

// Mask for the static attribute field index packed into INDEX_LIST data.
#define LOOM_FORMAT_INDEX_LIST_ATTR_INDEX_MASK ((uint16_t)0x7FFFu)
#define LOOM_FORMAT_INDEX_LIST_DATA(static_attr_index, leading_glue) \
  ((uint16_t)((uint16_t)(static_attr_index) |                        \
              ((leading_glue) ? 0u                                   \
                              : LOOM_FORMAT_INDEX_LIST_DATA_NO_LEADING_GLUE)))
#define LOOM_FORMAT_INDEX_LIST_STATIC_ATTR_INDEX(data) \
  ((uint16_t)((data) & LOOM_FORMAT_INDEX_LIST_ATTR_INDEX_MASK))
#define LOOM_FORMAT_INDEX_LIST_HAS_LEADING_GLUE(data) \
  (!iree_any_bit_set((data), LOOM_FORMAT_INDEX_LIST_DATA_NO_LEADING_GLUE))

// Surface syntax selected by a REGION format element. This affects only text
// parsing/printing; the in-memory representation is always an ordinary
// loom_region_t.
typedef enum loom_region_syntax_e {
  // Canonical braced region: { block+ }.
  LOOM_REGION_SYNTAX_DEFAULT = 0,
  // Test-only alternate region syntax: do { block+ }.
  LOOM_REGION_SYNTAX_TEST_DO = 1,
  // Descriptor-backed target-low assembly syntax: asm<descriptor.set> { ... }.
  LOOM_REGION_SYNTAX_LOW_ASM = 2,
  // Canonical braced region by default, with optional target-low asm syntax.
  LOOM_REGION_SYNTAX_LOW_ASM_OPTIONAL = 3,
  // Pass pipeline syntax. Currently canonical braced form; the friendly
  // parser/printer selects the same in-memory pass.* operations.
  LOOM_REGION_SYNTAX_PIPELINE = 4,
} loom_region_syntax_t;

#define LOOM_FORMAT_REGION_TABLE_DATA(keys_attr_index, default_region_index) \
  ((uint16_t)(((uint16_t)(default_region_index) << 8) |                      \
              (uint16_t)(keys_attr_index)))
#define LOOM_FORMAT_REGION_TABLE_KEYS_ATTR_INDEX(data) \
  ((uint8_t)((data) & 0xFF))
#define LOOM_FORMAT_REGION_TABLE_DEFAULT_REGION_INDEX(data) \
  ((uint8_t)(((data) >> 8) & 0xFF))

// A 4-byte printer/parser instruction. An op with 12 format elements uses
// 48 bytes of .rodata. For 200 ops, total format tables are ~10KB.
//
// The kind field determines the instruction. The field_index selects which
// operand, result, attribute, or region this element references. The data
// field is kind-specific:
//   KEYWORD:        keyword ID (loom_keyword_id_t).
//   INDEX_LIST:     LOOM_FORMAT_INDEX_LIST_DATA(static attr index, glue).
//   OPERAND_DICT:   dict attribute field index storing key -> operand ordinal.
//   ATTR_TABLE:     i64 array attr field index storing row keys.
//   REGION_TABLE:   packed keys attr index and fixed default region index.
//   REGION:         loom_region_syntax_t parser/printer selector.
//   BINDING_LIST:   binding kind (CAPTURE=0, ELEMENT=1).
//   OPTIONAL_GROUP: (skip_count << 2) | anchor_category.
typedef struct loom_format_element_t {
  loom_format_kind_t kind;
  uint8_t field_index;
  uint16_t data;
} loom_format_element_t;

static_assert(sizeof(loom_format_element_t) == 4,
              "loom_format_element_t must be exactly 4 bytes");

// Data field flags for RESULT_TYPE_LIST elements. Stored in the
// element's data field.
enum loom_result_type_list_flag_bits_e {
  // Wrap result types in parentheses: (type, type).
  // When clear, result types are bare: type, type.
  LOOM_RESULT_TYPE_LIST_PARENS = 1u << 0,
};

// Data field flags for ATTR_DICT elements. Stored in the element's data field.
enum loom_attr_dict_format_flag_bits_e {
  // The dictionary contains ordinary declared op attributes that were not
  // otherwise printed by the format, instead of a single named dict attribute.
  LOOM_ATTR_DICT_FORMAT_INLINE_ATTRS = 1u << 0,
};

// Anchor categories for OPTIONAL_GROUP elements. Encoded in the low
// 2 bits of the data field. Tells the format walker what kind of field
// to check for presence.
enum loom_anchor_category_e {
  // Variadic operand: present if operand_count > fixed_operand_count.
  LOOM_ANCHOR_OPERAND = 0,
  // Optional attribute: present if the attribute is not LOOM_ATTR_ABSENT.
  LOOM_ANCHOR_ATTR = 1,
  // Region: present if region pointer is non-null.
  LOOM_ANCHOR_REGION = 2,
  // Results: present if result_count > 0.
  LOOM_ANCHOR_RESULTS = 3,
};

//===----------------------------------------------------------------------===//
// Keywords
//===----------------------------------------------------------------------===//

// Format keywords (punctuation + text). All format specs across all dialects
// reference keywords by ID. Generated from KEYWORD_MAP in c_tables.py —
// do not edit manually. Append new keywords to KEYWORD_MAP and regenerate.
typedef enum loom_keyword_id_e {
#include "loom/ops/keyword_enum.inc"
  LOOM_KW_COUNT_,
} loom_keyword_id_t;

// Returns the B-string for |keyword_id|, or NULL if |keyword_id| is out of
// range. E.g., LOOM_KW_TO -> "\x02to", LOOM_KW_STEP -> "\x04step".
loom_bstring_t loom_keyword_bstring(loom_keyword_id_t keyword_id);

//===----------------------------------------------------------------------===//
// Type constraints, traits, and binding kinds
//===----------------------------------------------------------------------===//

// Abstract type categories for operand/result descriptors. The verifier
// checks that the actual type of a value satisfies the constraint.
typedef enum loom_type_constraint_e {
  LOOM_TYPE_CONSTRAINT_TILE = 0,
  LOOM_TYPE_CONSTRAINT_TENSOR,
  LOOM_TYPE_CONSTRAINT_INTEGER,
  LOOM_TYPE_CONSTRAINT_FLOAT,
  LOOM_TYPE_CONSTRAINT_SCALAR,
  LOOM_TYPE_CONSTRAINT_INDEX,
  LOOM_TYPE_CONSTRAINT_OFFSET,
  LOOM_TYPE_CONSTRAINT_ADDRESS,
  LOOM_TYPE_CONSTRAINT_ANY,
  LOOM_TYPE_CONSTRAINT_GROUP,
  LOOM_TYPE_CONSTRAINT_ANY_ENCODING,
  LOOM_TYPE_CONSTRAINT_POOL,
  LOOM_TYPE_CONSTRAINT_REGISTER,
  // Exactly i1. Used for comparison results and boolean predicates.
  // Unlike INTEGER (which accepts any integer width), this requires
  // the type to be specifically i1.
  LOOM_TYPE_CONSTRAINT_I1,
  LOOM_TYPE_CONSTRAINT_VECTOR,
  // Vector type with rank 1.
  LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR,
  // Vector type with an all-static shape.
  LOOM_TYPE_CONSTRAINT_ALL_STATIC_VECTOR,
  // Vector type with an all-static rank-1 shape.
  LOOM_TYPE_CONSTRAINT_ALL_STATIC_RANK_ONE_VECTOR,
  LOOM_TYPE_CONSTRAINT_VIEW,
  LOOM_TYPE_CONSTRAINT_BUFFER,
  // Shaped type with an integer element type.
  LOOM_TYPE_CONSTRAINT_INTEGER_ELEMENT,
  // Shaped type with a floating-point element type.
  LOOM_TYPE_CONSTRAINT_FLOAT_ELEMENT,
  // Scalar index type or non-i1 integer type.
  LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_SCALAR,
  // Shaped type with index or non-i1 integer element type.
  LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT,
  // Shaped type with element type i1.
  LOOM_TYPE_CONSTRAINT_I1_ELEMENT,
  // Shaped type with element type i8.
  LOOM_TYPE_CONSTRAINT_I8_ELEMENT,
  // Shaped type with element type i32.
  LOOM_TYPE_CONSTRAINT_I32_ELEMENT,
  // Shaped type with element type f16 or bf16.
  LOOM_TYPE_CONSTRAINT_F16_OR_BF16_ELEMENT,
  // Shaped type with element type f32.
  LOOM_TYPE_CONSTRAINT_F32_ELEMENT,
  // Encoding type with address-layout role.
  LOOM_TYPE_CONSTRAINT_ENCODING_LAYOUT,
  // Encoding type with storage-schema role.
  LOOM_TYPE_CONSTRAINT_ENCODING_SCHEMA,
  // Encoding type with physical-storage role.
  LOOM_TYPE_CONSTRAINT_ENCODING_STORAGE,
  // Encoding type with numeric-transform role.
  LOOM_TYPE_CONSTRAINT_ENCODING_TRANSFORM,
  // Function-local byte storage handle.
  LOOM_TYPE_CONSTRAINT_STORAGE,
  LOOM_TYPE_CONSTRAINT_COUNT_,
} loom_type_constraint_t;

// Returns the display name for a type constraint (e.g., "tile", "integer").
const char* loom_type_constraint_name(loom_type_constraint_t constraint);

// Returns true if |type| satisfies the abstract |constraint|.
// Used by the verifier for operand/result type checking and by
// builders for debug-mode assertions.
bool loom_type_satisfies_constraint(loom_type_t type,
                                    loom_type_constraint_t constraint);

//===----------------------------------------------------------------------===//
// Semantic constraints
//===----------------------------------------------------------------------===//
//
// Constraints express relationships between an op's fields that the
// verifier checks. Each op's vtable points to an array of constraint
// entries. The verifier walks the array, interpreting each by kind.
//
// Per-op cost: 16 bytes .rodata per constraint, zero .text. The
// interpreter is one switch statement (~12 cases). Adding constraint
// kinds adds a case to the switch, not code per op.
//
// Field reference categories.
enum loom_field_category_e {
  LOOM_FIELD_OPERAND = 0,
  LOOM_FIELD_RESULT = 1,
  LOOM_FIELD_ATTR = 2,
  LOOM_FIELD_REGION = 3,
};

// Packs a (category, index) pair into a field reference.
#define LOOM_FIELD_REF(category, index) \
  ((loom_field_ref_t)(((category) << 6) | ((index) & 0x3F)))

// Extracts the category (0-3) from a packed field reference.
#define LOOM_FIELD_REF_CATEGORY(ref) ((ref) >> 6)

// Extracts the index (0-63) from a packed field reference.
#define LOOM_FIELD_REF_INDEX(ref) ((ref) & 0x3F)

// Constraint kinds for the table-driven constraint interpreter.
// Constraint relation: how values are compared.
enum loom_constraint_relation_e {
  // Every element of every listed field has the same property as the
  // first element of the first field. Despite the name, this is "all
  // elements equal the reference", not strict pairwise comparison.
  // Variadic fields are walked elementwise. Args: 1+ value fields.
  // Used by SameType, SameElementType, SameShape, SameEncoding,
  // RanksMatch.
  LOOM_RELATION_PAIRWISE_EQ = 0,

  // All elements of a single variadic value field share the same
  // property. Args: 1 variadic value field. Used by AllShapesMatch.
  LOOM_RELATION_ALL_SAME,

  // Every value in every listed field satisfies a type constraint.
  // The property slot stores a loom_type_constraint_t. Args: 1+ value
  // fields. Used by Has*Element and Has*Vector constraints.
  LOOM_RELATION_FIELD_SATISFIES,

  // Every entry block argument of a region satisfies a type constraint.
  // The property slot stores a loom_type_constraint_t. Args: (region field).
  // Used by BlockArgsSatisfy.
  LOOM_RELATION_REGION_ARGS_SATISFY,

  // An i64 attribute satisfies a relation-specific predicate stored in the
  // property slot. Args: (i64 attr field). Used by PositiveBitWidthAttr.
  LOOM_RELATION_ATTR_I64_PREDICATE,

  // An attribute literal payload kind matches the scalar element type of a
  // value field. Args: (attr field, value field). Used by
  // AttrMatchesElementType.
  LOOM_RELATION_ATTR_MATCHES_ELEMENT_TYPE,

  // The scalar or shaped element bit width of the first value field is
  // strictly ordered against the second value field. The property slot stores
  // the order predicate. Args: (checked value field, reference value field).
  // Used by ElementWidthGreaterThan and ElementWidthLessThan.
  LOOM_RELATION_ELEMENT_WIDTH_ORDER,

  // The scalar or shaped element bit width of the first value field is at least
  // the i64 attribute value. Args: (checked value field, i64 attr field). Used
  // by ElementWidthAtLeastAttr.
  LOOM_RELATION_ELEMENT_WIDTH_AT_LEAST_ATTR,

  // A static bit range described by offset and width i64 attributes fits within
  // the scalar or shaped element bit width of a value field. Args: (checked
  // value field, offset i64 attr field, width i64 attr field). Used by
  // BitRangeWithinElementWidth.
  LOOM_RELATION_BIT_RANGE_WITHIN_ELEMENT_WIDTH,

  // Two value fields have the same total bit count, allowing static and simple
  // symbolic dynamic shaped types. Args: (lhs value field, rhs value field).
  // Used by TotalBitCountEqual.
  LOOM_RELATION_TOTAL_BIT_COUNT_EQUAL,

  // A payload value field with a fixed bit-width attribute has the same static
  // bit count as a storage value field. The fourth arg is the value field used
  // for the mismatch diagnostic. Args: (payload value field, width i64 attr
  // field, storage value field, diagnostic value field). Used by
  // PackedPayloadBitCountMatchesStorage and
  // UnpackedPayloadBitCountMatchesStorage.
  LOOM_RELATION_PAYLOAD_BIT_COUNT_MATCHES_STORAGE,

  // The element count of a variadic value field equals the rank of a
  // shaped value field. Args: (shaped value field, variadic value
  // field). Used by OffsetCountMatchesRank.
  LOOM_RELATION_COUNT_MATCHES_RANK,

  // The element count of a variadic value field equals the static element
  // count of a shaped value field. Args: (shaped value field, variadic value
  // field). Used by ValueCountMatchesStaticElementCount.
  LOOM_RELATION_COUNT_MATCHES_STATIC_ELEMENT_COUNT,

  // An i64 attribute's value falls within [0, rank) of a shaped value
  // field. Args: (shaped value field, i64 attr field). Used by
  // DimIndexInBounds.
  LOOM_RELATION_ATTR_IN_RANGE_RANK,

  // A region's entry block argument count matches the element count of a
  // variadic value field or another region's entry block args. Args: (region
  // field, variadic value field | region field). Used by BlockArgCount.
  LOOM_RELATION_REGION_ARG_COUNT,

  // Each region entry block argument's property matches the corresponding
  // element of a variadic value field or another region's entry block args at
  // the same position. Args: (region field, variadic value field | region
  // field). Used by BlockArgsMatchTypes and BlockArgsMatchElementTypes.
  LOOM_RELATION_REGION_ARG_MATCH,

  // A region's terminator (yield) operand count matches the element
  // count of a variadic value field. Args: (region field, variadic
  // value field). Used by YieldCountMatchesResults.
  LOOM_RELATION_YIELD_COUNT,

  // Each region terminator (yield) operand's property matches the
  // corresponding element of a variadic value field at the same
  // position. Args: (region field, variadic value field). Used by
  // YieldTypesMatchResults and YieldElementTypesMatchResults.
  LOOM_RELATION_YIELD_MATCH,

  // Two variadic value fields agree position-by-position. The two
  // fields must have the same element count, and the property at
  // each position must be equal. Diagnoses a count mismatch with one
  // error and per-position property mismatches with one error each.
  // Args: (variadic value field, variadic value field). Used by
  // IterArgsMatchResults.
  LOOM_RELATION_VARIADIC_MATCH,

  // Result vector shape equals source vector shape with the last axis divided
  // by a small static group size stored in the property slot.
  // Args: (source vector field, result vector field). Used by
  // LastAxisGroupedBy.
  LOOM_RELATION_LAST_AXIS_GROUPED_BY,

  // Register unit counts in the first variadic register field sum to the
  // register unit count of the second register field. Args: (summed register
  // value field, result register value field). Used by RegisterUnitsSumTo.
  LOOM_RELATION_REGISTER_UNIT_COUNT_SUM,

  LOOM_RELATION_COUNT_,
};
typedef uint8_t loom_constraint_relation_t;

// Constraint property: which aspect of a type to compare.
enum loom_constraint_property_e {
  // Full type equality, including shape, element type, and encoding.
  LOOM_PROPERTY_TYPE = 0,
  // Type kind only: scalar vs tile vs tensor vs vector vs view, etc.
  LOOM_PROPERTY_KIND = 1,
  // Element type only — ignores shape and encoding. For shaped types
  // this lets a constraint compare a tile<...x f32> against a scalar
  // f32 successfully.
  LOOM_PROPERTY_ELEMENT_TYPE = 2,
  // Encoding only — ignores shape and element type. Used to check
  // that two tiles share an encoding regardless of element type.
  LOOM_PROPERTY_ENCODING = 3,
  // Shape only — ignores element type and encoding. Two tiles with
  // identical dimensions but different element types match.
  LOOM_PROPERTY_SHAPE = 4,
  // Rank only — ignores dimension sizes, element type, and encoding.
  // Two shaped types with the same number of dimensions match.
  LOOM_PROPERTY_RANK = 5,
  // Element bit width of the first relation field is strictly greater than the
  // second relation field. Used by ElementWidthGreaterThan.
  LOOM_PROPERTY_ELEMENT_WIDTH_GREATER_THAN = 6,
  // Element bit width of the first relation field is strictly less than the
  // second relation field. Used by ElementWidthLessThan.
  LOOM_PROPERTY_ELEMENT_WIDTH_LESS_THAN = 7,
  // Attribute value is a positive bit width. Used by PositiveBitWidthAttr.
  LOOM_PROPERTY_BIT_WIDTH_POSITIVE = 8,
  // Element bit width is at least an i64 attribute. Used by
  // ElementWidthAtLeastAttr.
  LOOM_PROPERTY_ELEMENT_WIDTH_AT_LEAST_ATTR = 9,
  // Static bit range is contained within an element width. Used by
  // BitRangeWithinElementWidth.
  LOOM_PROPERTY_BIT_RANGE_WITHIN_ELEMENT_WIDTH = 10,
  // Total value bit count. Used by TotalBitCountEqual.
  LOOM_PROPERTY_TOTAL_BIT_COUNT = 11,
  // Packed payload bit count equals storage bit count. Used by
  // PackedPayloadBitCountMatchesStorage.
  LOOM_PROPERTY_PACKED_PAYLOAD_BIT_COUNT_MATCHES_STORAGE = 12,
  // Unpacked payload bit count equals storage bit count. Used by
  // UnpackedPayloadBitCountMatchesStorage.
  LOOM_PROPERTY_UNPACKED_PAYLOAD_BIT_COUNT_MATCHES_STORAGE = 13,
  // Register allocation class, ignoring the number of units. Used by
  // SameRegisterClass.
  LOOM_PROPERTY_REGISTER_CLASS = 14,
  // Register unit count. Used by RegisterUnitsSumTo.
  LOOM_PROPERTY_REGISTER_UNIT_COUNT = 15,
  LOOM_PROPERTY_COUNT_,
};
typedef uint8_t loom_constraint_property_t;

const char* loom_constraint_relation_name(loom_constraint_relation_t relation);
const char* loom_constraint_property_name(loom_constraint_property_t property);

// A table-driven semantic constraint entry. 10 bytes.
//
// Each op's vtable points to an array of these. The verifier walks
// the array, interpreting each constraint by (relation, property).
// Per-op cost: 10 bytes .rodata per constraint, zero .text code.
typedef struct loom_constraint_t {
  // Relation interpreter opcode.
  loom_constraint_relation_t relation;
  // Property or small data payload interpreted by the relation.
  loom_constraint_property_t property;
  // Number of valid field references in args.
  uint8_t arg_count;
  // Reserved for stable table layout and must be zero.
  uint8_t reserved;
  // Field references consumed by the relation, padded with zero.
  loom_field_ref_t args[4];
  // LOOM_ERROR_REF-packed override, or LOOM_ERROR_REF_NONE for the default.
  uint16_t error_ref;
} loom_constraint_t;

static_assert(sizeof(loom_constraint_t) == 10,
              "loom_constraint_t must be 10 bytes");

//===----------------------------------------------------------------------===//
// Field descriptors
//===----------------------------------------------------------------------===//

// How an operand carries a managed resource into an operation.
enum loom_ownership_carrier_e {
  // Operand has no ownership contract.
  LOOM_OWNERSHIP_CARRIER_NONE = 0,
  // Operand carries the resource handle value itself.
  LOOM_OWNERSHIP_CARRIER_BY_VALUE = 1,
  // Operand carries an addressable slot or object that may contain a resource.
  LOOM_OWNERSHIP_CARRIER_BY_REFERENCE = 2,
};
typedef uint8_t loom_ownership_carrier_t;

// Ownership action applied to an operand field.
enum loom_operand_ownership_effect_e {
  // Operand has no ownership action.
  LOOM_OPERAND_OWNERSHIP_NONE = 0,
  // Operand is observed without transferring ownership.
  LOOM_OPERAND_OWNERSHIP_BORROW = 1,
  // Operand ownership transfers into the operation.
  LOOM_OPERAND_OWNERSHIP_CONSUME = 2,
  // Operation retains the operand resource.
  LOOM_OPERAND_OWNERSHIP_RETAIN = 3,
  // Operation releases one owned reference to the operand resource.
  LOOM_OPERAND_OWNERSHIP_RELEASE = 4,
  // Operation drops compiler ownership without emitting a release.
  LOOM_OPERAND_OWNERSHIP_DISCARD = 5,
  // Operand resource escapes to an untracked owner.
  LOOM_OPERAND_OWNERSHIP_ESCAPE = 6,
};
typedef uint8_t loom_operand_ownership_effect_t;

// Ownership action applied to a result field.
enum loom_result_ownership_effect_e {
  // Result has no ownership action.
  LOOM_RESULT_OWNERSHIP_NONE = 0,
  // Result creates a fresh owned resource.
  LOOM_RESULT_OWNERSHIP_FRESH = 1,
  // Result borrows a resource owned elsewhere.
  LOOM_RESULT_OWNERSHIP_BORROWED = 2,
  // Result is an owned retained reference to an existing resource.
  LOOM_RESULT_OWNERSHIP_RETAINED = 3,
  // Result aliases an operand resource without consuming it.
  LOOM_RESULT_OWNERSHIP_ALIAS = 4,
  // Result receives ownership from a tied operand.
  LOOM_RESULT_OWNERSHIP_TIED = 5,
};
typedef uint8_t loom_result_ownership_effect_t;

#define LOOM_RESULT_OWNERSHIP_SOURCE_FIELD_NONE UINT8_MAX

#define LOOM_OWNERSHIP_SOURCE_OPERAND_NONE UINT16_MAX

enum loom_operand_flag_bits_e {
  LOOM_OPERAND_VARIADIC = 1u << 0,
  LOOM_OPERAND_OPTIONAL = 1u << 1,
  LOOM_OPERAND_READS = 1u << 2,
  LOOM_OPERAND_WRITES = 1u << 3,
};
typedef uint8_t loom_operand_flags_t;

enum loom_result_flag_bits_e {
  LOOM_RESULT_VARIADIC = 1u << 0,
  LOOM_RESULT_ALLOCATES = 1u << 1,
};
typedef uint8_t loom_result_flags_t;

enum loom_attr_flag_bits_e {
  LOOM_ATTR_OPTIONAL = 1u << 0,
  // Enum values are ordinal-preserving across bytecode and generic
  // verification. Op-specific verifiers still own sentinel and consumer
  // support checks.
  LOOM_ATTR_OPEN_ENUM = 1u << 1,
  // Text formats may omit this required scalar attribute when the present
  // value equals the zero/false default. Parsers restore the explicit value.
  LOOM_ATTR_ELIDE_DEFAULT = 1u << 2,
};
typedef uint8_t loom_attr_flags_t;

enum loom_region_flag_bits_e {
  // Region must contain exactly one block.
  LOOM_REGION_SINGLE_BLOCK = 1u << 0,
  // Region may be absent when it is part of a trailing optional suffix.
  LOOM_REGION_OPTIONAL = 1u << 1,
  // Region entry block arguments are projected from the op's FuncArgs
  // signature. The projected values have matching names and types but remain
  // distinct SSA values owned by this region.
  LOOM_REGION_PROJECT_FUNC_ARGS = 1u << 2,
  // Buffer entry block arguments seed target-independent global memory facts.
  // This is a region signature contract, not a property of the generic buffer
  // type.
  LOOM_REGION_GLOBAL_BUFFER_ARGS = 1u << 3,
};
typedef uint8_t loom_region_flags_t;

// Per-operand metadata in the op vtable.
typedef struct loom_operand_descriptor_t {
  // Author-facing DSL operand field name used in diagnostics.
  loom_bstring_t name;
  // Required type category for each operand in this field.
  loom_type_constraint_t type_constraint;
  // Operand representation, effect, and variadic flags.
  loom_operand_flags_t flags;
  // Ownership action applied to each value in this operand field.
  loom_operand_ownership_effect_t ownership_effect;
  // Carrier mode for the operand ownership action.
  loom_ownership_carrier_t ownership_carrier;
} loom_operand_descriptor_t;

static_assert(sizeof(loom_operand_descriptor_t) == 16,
              "loom_operand_descriptor_t must be 16 bytes");

// Per-result metadata in the op vtable.
typedef struct loom_result_descriptor_t {
  // Author-facing DSL result field name used in diagnostics.
  loom_bstring_t name;
  // Required type category for each result in this field.
  loom_type_constraint_t type_constraint;
  // Result representation and variadic flags.
  loom_result_flags_t flags;
  // Ownership action applied to each value in this result field.
  loom_result_ownership_effect_t ownership_effect;
  // Source operand field index for aliasing result effects.
  uint8_t ownership_source_operand_index;
} loom_result_descriptor_t;

static_assert(sizeof(loom_result_descriptor_t) == 16,
              "loom_result_descriptor_t must be 16 bytes");

typedef uint32_t loom_symbol_interface_flags_t;

enum loom_symbol_interface_bits_e {
  // Symbol implements the generated function-like interface.
  LOOM_SYMBOL_INTERFACE_FUNC_LIKE = 1u << 0,
  // Symbol implements the generated global-like contract.
  LOOM_SYMBOL_INTERFACE_GLOBAL = 1u << 1,
  // Symbol names a target executable/package-like entity.
  LOOM_SYMBOL_INTERFACE_EXECUTABLE = 1u << 2,
  // Symbol names a generic module-level record.
  LOOM_SYMBOL_INTERFACE_RECORD = 1u << 3,
  // Symbol names a target environment record.
  LOOM_SYMBOL_INTERFACE_TARGET = 1u << 4,
  // Symbol names a compile/link-time configuration value.
  LOOM_SYMBOL_INTERFACE_CONFIG = 1u << 5,
};

// Generated metadata for an op that defines a module symbol.
typedef struct loom_symbol_definition_descriptor_t {
  // Human-readable symbol class used in diagnostics.
  loom_bstring_t name;
  // Attribute index of the symbol identity field on the defining op.
  uint8_t name_attr_index;
  // Structural symbol interfaces implemented by this definition.
  loom_symbol_interface_flags_t interfaces;
  // Existing bytecode payload kind, or LOOM_SYMBOL_NONE if not serializable
  // through the current SYMBOLS section.
  loom_symbol_kind_t bytecode_kind;
  // Optional domain that computes typed facts for symbols defined by this op.
  const loom_symbol_fact_domain_t* fact_domain;
} loom_symbol_definition_descriptor_t;

// Generated metadata for a symbol-reference attribute.
typedef struct loom_symbol_reference_descriptor_t {
  // Human-readable expected symbol class used in diagnostics.
  loom_bstring_t name;
  // Structural symbol interfaces accepted by this reference.
  loom_symbol_interface_flags_t interfaces;
} loom_symbol_reference_descriptor_t;

static inline iree_string_view_t loom_symbol_definition_descriptor_name(
    const loom_symbol_definition_descriptor_t* descriptor) {
  return descriptor ? loom_bstring_view(descriptor->name) : IREE_SV("unknown");
}

static inline iree_string_view_t loom_symbol_reference_descriptor_name(
    const loom_symbol_reference_descriptor_t* descriptor) {
  return descriptor ? loom_bstring_view(descriptor->name) : IREE_SV("symbol");
}

static inline bool loom_symbol_definition_implements(
    const loom_symbol_definition_descriptor_t* descriptor,
    loom_symbol_interface_flags_t interfaces) {
  return descriptor && interfaces &&
         iree_any_bit_set(descriptor->interfaces, interfaces);
}

static inline bool loom_symbol_implements(
    const loom_symbol_t* symbol, loom_symbol_interface_flags_t interfaces) {
  return symbol &&
         loom_symbol_definition_implements(symbol->definition, interfaces);
}

static inline loom_symbol_kind_t loom_symbol_bytecode_kind(
    const loom_symbol_t* symbol) {
  if (!symbol) return LOOM_SYMBOL_NONE;
  return symbol->definition ? symbol->definition->bytecode_kind : symbol->kind;
}

// Per-attribute metadata in the op vtable.
typedef struct loom_attr_descriptor_t {
  // Author-facing DSL attribute field name used in diagnostics.
  loom_bstring_t name;
  // Runtime attribute payload kind.
  loom_attr_kind_t attr_kind;
  // Attribute structural flags such as optional.
  loom_attr_flags_t flags;
  // Number of enum keyword slots in |enum_case_names|.
  uint8_t enum_case_count;
  // Dense enum value to keyword table, or NULL for non-enum attrs.
  const loom_bstring_t* enum_case_names;
  // Expected symbol target contract, or NULL for non-symbol-reference attrs.
  const loom_symbol_reference_descriptor_t* symbol_ref;
} loom_attr_descriptor_t;

// Returns the attribute name as a string view.
static inline iree_string_view_t loom_attr_descriptor_name(
    const loom_attr_descriptor_t* descriptor) {
  return loom_bstring_view(descriptor->name);
}

// Returns the explicit zero/false scalar value implied by ELIDE_DEFAULT.
static inline loom_attribute_t loom_attr_descriptor_default_value(
    const loom_attr_descriptor_t* descriptor) {
  switch ((loom_attr_kind_t)descriptor->attr_kind) {
    case LOOM_ATTR_I64:
      return loom_attr_i64(0);
    case LOOM_ATTR_BOOL:
      return loom_attr_bool(false);
    default:
      return loom_attr_absent();
  }
}

// Returns true when |attr| is the elidable text default for |descriptor|.
static inline bool loom_attr_descriptor_elides_value(
    const loom_attr_descriptor_t* descriptor, const loom_attribute_t* attr) {
  if (!iree_any_bit_set(descriptor->flags, LOOM_ATTR_ELIDE_DEFAULT)) {
    return false;
  }
  loom_attribute_t default_value =
      loom_attr_descriptor_default_value(descriptor);
  return !loom_attr_is_absent(default_value) &&
         loom_attribute_equal(attr, &default_value);
}

// Per-region metadata in the op vtable.
typedef struct loom_region_descriptor_t {
  // Required explicit terminator kind, or LOOM_OP_KIND_UNKNOWN if any
  // terminator kind is allowed.
  loom_op_kind_t terminator;

  // Op kind that the text parser may synthesize and the text printer may elide,
  // or LOOM_OP_KIND_UNKNOWN when the text form must spell every terminator.
  // In-memory IR must still contain a materialized terminator op.
  loom_op_kind_t implicit_terminator;

  // Region structure flags such as single-block enforcement.
  loom_region_flags_t flags;
} loom_region_descriptor_t;

static_assert(sizeof(loom_region_descriptor_t) == 6,
              "loom_region_descriptor_t must be 6 bytes");

// Generated structural placement metadata for an op kind.
typedef struct loom_op_placement_descriptor_t {
  // Op kinds that must be the direct parent op.
  const loom_op_kind_t* required_parents;
  // Op kinds that must appear somewhere in the parent-op chain.
  const loom_op_kind_t* required_ancestors;
  // Op kinds that must not appear anywhere in the parent-op chain.
  const loom_op_kind_t* forbidden_ancestors;
  // Number of entries in |required_parents|.
  uint8_t required_parent_count;
  // Number of entries in |required_ancestors|.
  uint8_t required_ancestor_count;
  // Number of entries in |forbidden_ancestors|.
  uint8_t forbidden_ancestor_count;
} loom_op_placement_descriptor_t;

// Returns a dialect vtable array and writes |vtable_count| to |out_count| when
// requested. Generated dialect accessors use this so the count-output contract
// is owned by the op-definition runtime instead of duplicated in generated C.
const loom_op_vtable_t* const* loom_dialect_vtable_array(
    const loom_op_vtable_t* const* vtables, iree_host_size_t vtable_count,
    iree_host_size_t* out_count);

// Returns a dense dialect semantic metadata array and writes |semantic_count|
// to |out_count| when requested.
const loom_op_semantics_t* loom_dialect_semantics_array(
    const loom_op_semantics_t* semantics, iree_host_size_t semantic_count,
    iree_host_size_t* out_count);

// Returns semantic metadata for |kind| within |dialect_id| from a dense
// dialect-local semantic table, or empty metadata when the kind belongs to a
// different dialect or has no row in |semantics|.
loom_op_semantics_t loom_dialect_semantics_lookup(
    loom_op_kind_t kind, loom_dialect_id_t dialect_id,
    const loom_op_semantics_t* semantics, iree_host_size_t semantic_count);

// Returns the descriptor for an actual region slot. For ops with a trailing
// variadic region field, fixed slots use their exact descriptor and every
// variadic slot reuses the final descriptor entry.
const loom_region_descriptor_t* loom_op_vtable_region_descriptor(
    const loom_op_vtable_t* vtable, uint8_t region_index);

// Returns the module-local symbol reference defined by |op|. Returns false for
// non-symbol ops, malformed symbol attributes, cross-module refs, or refs
// outside the module symbol table.
bool loom_op_defining_symbol_ref(const loom_module_t* module,
                                 const loom_op_t* op,
                                 loom_symbol_ref_t* out_ref);

// Returns true when operand descriptors name independent operand segments
// stored over the op's flat operand array.
static inline bool loom_op_vtable_has_segmented_operands(
    const loom_op_vtable_t* vtable) {
  return vtable && iree_any_bit_set(vtable->vtable_flags,
                                    LOOM_OP_VTABLE_SEGMENTED_OPERANDS);
}

// Returns the number of operand segment counts stored on an instance of this
// op kind, or zero for non-segmented ops.
static inline uint8_t loom_op_vtable_operand_segment_count(
    const loom_op_vtable_t* vtable) {
  return loom_op_vtable_has_segmented_operands(vtable)
             ? loom_op_vtable_operand_descriptor_count(vtable)
             : 0;
}

// Resolves an author-facing operand field to its flat operand span. For
// non-segmented ops this preserves the historical fixed-plus-trailing-variadic
// layout. For segmented ops, |field_index| indexes the operand descriptor array
// and the returned span is derived from the op's segment-count tail storage.
loom_value_slice_t loom_op_operand_field_span(const loom_op_vtable_t* vtable,
                                              const loom_op_t* op,
                                              uint8_t field_index);

// Returns true when an operand field has at least one value.
bool loom_op_operand_field_present(const loom_op_vtable_t* vtable,
                                   const loom_op_t* op, uint8_t field_index);

// Maps a flat operand index back to the operand descriptor that owns it.
// Returns false if the index is out of range or the op kind has no descriptor
// metadata.
bool loom_op_operand_descriptor_at(
    const loom_op_vtable_t* vtable, const loom_op_t* op, uint16_t operand_index,
    const loom_operand_descriptor_t** out_descriptor, uint8_t* out_field_index,
    uint16_t* out_element_index);

// Binding kind for BindingList format elements.
typedef enum loom_binding_kind_e {
  // Block arg has the same type as the operand.
  LOOM_BINDING_CAPTURE = 0,
  // Block arg has the element type of the operand.
  LOOM_BINDING_ELEMENT = 1,
} loom_binding_kind_t;

//===----------------------------------------------------------------------===//
// Value dereference helpers
//===----------------------------------------------------------------------===//

// Resolves an op's operand value ID to the value struct in the module's
// value table. |index| is the operand position (0-based).
//
// Usage:
//   loom_value_t* lhs = loom_op_operand_value(module, addi_op, 0);
//   if (loom_type_kind(lhs->type) == LOOM_TYPE_TILE) { ... }
static inline loom_value_t* loom_op_operand_value(const loom_module_t* module,
                                                  const loom_op_t* op,
                                                  uint16_t index) {
  IREE_ASSERT(index < op->operand_count);
  loom_value_id_t value_id = loom_op_operands(op)[index];
  IREE_ASSERT(value_id < module->values.count);
  return &module->values.entries[value_id];
}

// Resolves an op's result value ID to the value struct in the module's
// value table. |index| is the result position (0-based).
//
// Usage:
//   loom_value_t* result = loom_op_result_value(module, addi_op, 0);
//   if (result->use_count == 0) { /* dead result, candidate for DCE */ }
static inline loom_value_t* loom_op_result_value(const loom_module_t* module,
                                                 const loom_op_t* op,
                                                 uint16_t index) {
  IREE_ASSERT(index < op->result_count);
  loom_value_id_t value_id = loom_op_results(op)[index];
  IREE_ASSERT(value_id < module->values.count);
  return &module->values.entries[value_id];
}

// Returns a pointer to the single use entry if the value has exactly
// one use, or NULL if it has zero or more than one use. The returned
// pointer is valid until the next use-list mutation on this value.
//
// Usage (fusion pattern — "does this tile feed exactly one consumer?"):
//   const loom_use_t* use = loom_value_single_use(tile_value);
//   if (use && loom_test_map_isa(loom_use_user_op(*use))) {
//     // Fuse into the map.
//   }
static inline const loom_use_t* loom_value_single_use(
    const loom_value_t* value) {
  if (value->use_count != 1) return NULL;
  return &loom_value_uses(value)[0];
}

//===----------------------------------------------------------------------===//
// Effect query helpers
//===----------------------------------------------------------------------===//

// Returns the cached effective trait flags for |op|.
//
// IR constructors initialize this word from the op vtable construction default.
// Descriptor-backed and callback-backed ops stamp richer per-instance traits at
// construction or mutation boundaries so generic passes can query effects with
// one op-local load.
loom_trait_flags_t loom_op_effective_traits(const loom_module_t* module,
                                            const loom_op_t* op);

// Refreshes callback-backed effective traits for |op| after attrs or instance
// flags have changed. Ops with no effective-traits callback keep their current
// trait word; descriptor-backed ops are stamped explicitly by target-low
// construction helpers.
void loom_op_refresh_effective_traits(const loom_module_t* module,
                                      loom_op_t* op);

// Returns true if |op| may write to a resource or has unknown effects.
bool loom_op_may_write(const loom_module_t* module, const loom_op_t* op);

// Returns true if any live op nested under |op|'s regions is a compiler hint.
// Hints are not semantic memory effects, but ordinary DCE and canonicalization
// must preserve them until an explicit hint-stripping pass removes them.
bool loom_op_regions_have_hints(const loom_module_t* module,
                                const loom_op_t* op);

// Returns true if every result of |op| has zero operand uses and no external
// value type references. Type references carried by another result of |op| do
// not keep the whole op alive.
bool loom_op_results_unused(const loom_module_t* module, const loom_op_t* op);

// Returns true if |op| is trivially dead: it has results, does not
// write to any resource, has no unknown effects, and every result is
// unused. Read-only and non-deterministic ops without writes are dead
// when unused — a read with no observer is a no-op.
bool loom_op_is_trivially_dead(const loom_module_t* module,
                               const loom_op_t* op);

// Walks SSA value references embedded in all value types owned by |op|'s
// subtree.
//
// This includes result types on |op| and nested ops, plus block argument types
// in nested regions. Erase and DCE paths use this before unlinking a subtree so
// providers of dynamic dimensions or SSA encodings get rechecked after the
// carrier values disappear.
iree_status_t loom_op_walk_subtree_type_refs(
    const loom_module_t* module, const loom_op_t* op,
    loom_type_value_ref_callback_t callback, void* user_data);

//===----------------------------------------------------------------------===//
// CallLike interface helpers
//===----------------------------------------------------------------------===//

// Returns true if |call| refers to a valid direct call-like op. A cast via
// loom_call_like_cast() returns {NULL, NULL} on failure. All accessors below
// tolerate a NULL vtable and return safe defaults.
static inline bool loom_call_like_isa(loom_call_like_t call) {
  return call.op != NULL;
}

// Casts |op| to loom_call_like_t if it implements the CallLike interface.
// Returns {NULL, NULL} if |op| is NULL or does not implement it.
loom_call_like_t loom_call_like_cast(const loom_module_t* module,
                                     loom_op_t* op);

// Returns the direct callee symbol ref, or {0, 0} if |call| is not valid.
loom_symbol_ref_t loom_call_like_callee(loom_call_like_t call);

// Returns the trailing call argument slice, or an empty slice if |call| is not
// valid or the recorded offset is malformed for the op instance.
loom_value_slice_t loom_call_like_operands(loom_call_like_t call);

// Returns the trailing call result slice, or an empty slice if |call| is not
// valid or the recorded offset is malformed for the op instance.
loom_value_slice_t loom_call_like_results(loom_call_like_t call);

// Returns the operand offset where call arguments begin.
uint16_t loom_call_like_operand_offset(loom_call_like_t call);

// Returns the result offset where call results begin.
uint16_t loom_call_like_result_offset(loom_call_like_t call);

// Returns the purity attr value (0 = unspecified, nonzero = pure).
uint8_t loom_call_like_purity(loom_call_like_t call);

// Returns the temperature attr value (0 = unspecified).
uint8_t loom_call_like_temperature(loom_call_like_t call);

// Returns the inline policy attr value (0 = unspecified).
uint8_t loom_call_like_inline_policy(loom_call_like_t call);

// Returns the semantic class of the call-like op.
loom_call_like_kind_t loom_call_like_kind(loom_call_like_t call);

//===----------------------------------------------------------------------===//
// FuncLike interface helpers
//===----------------------------------------------------------------------===//

// Returns true if |func| refers to a valid func-like op. A cast via
// loom_func_like_cast() returns {NULL, NULL} on failure. All accessor
// helpers below tolerate a NULL vtable and return safe defaults (NULL,
// 0, or empty) so callers do not need to check loom_func_like_isa()
// before every call.
static inline bool loom_func_like_isa(loom_func_like_t func) {
  return func.op != NULL;
}

// Casts |op| to loom_func_like_t if it implements the FuncLike interface.
// Returns {NULL, NULL} if |op| is NULL or does not implement it. Safe to
// call unconditionally — callers check the result with loom_func_like_isa().
loom_func_like_t loom_func_like_cast(const loom_module_t* module,
                                     loom_op_t* op);

// Returns the body region of a func-like op, or NULL for bodyless ops
// (func.decl, func.ukernel) or if |func| is not valid.
loom_region_t* loom_func_like_body(loom_func_like_t func);

// Returns the body region index, or LOOM_REGION_INDEX_NONE for bodyless ops or
// invalid func-like references.
uint8_t loom_func_like_body_region_index(loom_func_like_t func);

// Returns the number of root regions owned by a func-like op.
uint8_t loom_func_like_region_count(loom_func_like_t func);

// Returns the root region at |region_index|, or NULL for invalid inputs.
loom_region_t* loom_func_like_region(loom_func_like_t func,
                                     uint8_t region_index);

// Returns true when |region_index| names the body region.
bool loom_func_like_region_is_body(loom_func_like_t func, uint8_t region_index);

// Returns true when |region_index| projects the function signature arguments
// into a distinct root-region entry block.
bool loom_func_like_region_projects_args(const loom_module_t* module,
                                         loom_func_like_t func,
                                         uint8_t region_index);

// Returns the purity attr value (0 = unspecified, nonzero = pure).
uint8_t loom_func_like_purity(loom_func_like_t func);

// Returns the temperature attr value (0 = unspecified).
uint8_t loom_func_like_temperature(loom_func_like_t func);

// Returns the inline policy attr value (0 = unspecified).
uint8_t loom_func_like_inline_policy(loom_func_like_t func);

// Returns the visibility attr value (0 = private, nonzero = public).
uint8_t loom_func_like_visibility(loom_func_like_t func);

// Returns the calling convention attr value (0 = default/host).
uint8_t loom_func_like_cc(loom_func_like_t func);

// Returns the callee symbol ref for a func-like op, or {0, 0} if
// |func| is not valid.
loom_symbol_ref_t loom_func_like_callee(loom_func_like_t func);

// Returns the import module string ID for a func-like op, or
// LOOM_STRING_ID_INVALID if absent.
loom_string_id_t loom_func_like_import_module(loom_func_like_t func);

// Returns the import symbol string ID for a func-like op, or
// LOOM_STRING_ID_INVALID if absent.
loom_string_id_t loom_func_like_import_symbol(loom_func_like_t func);

// Returns the target record symbol ref for a func-like op, or null if
// |func| has no target contract.
loom_symbol_ref_t loom_func_like_target(loom_func_like_t func);

// Returns the target ABI enum value, or 0 if |func| has no explicit ABI.
uint8_t loom_func_like_abi(loom_func_like_t func);

// Returns the target ABI payload attrs, or an empty slice if absent.
loom_named_attr_slice_t loom_func_like_abi_attrs(loom_func_like_t func);

// Returns the export symbol string ID, or LOOM_STRING_ID_INVALID if absent.
loom_string_id_t loom_func_like_export_symbol(loom_func_like_t func);

// Returns the export payload attrs, or an empty slice if absent.
loom_named_attr_slice_t loom_func_like_export_attrs(loom_func_like_t func);

// Returns true and assigns the export linkage enum value when present.
bool loom_func_like_export_linkage(loom_func_like_t func, uint8_t* out_linkage);

// Returns the function argument value IDs and their count. For ops
// with a body region, args are the entry block's block arguments.
// For declaration-style ops, args are stored as the op's operands.
// Returns NULL and sets |out_count| to 0 if |func| is not valid.
const loom_value_id_t* loom_func_like_arg_ids(loom_func_like_t func,
                                              uint16_t* out_count);

// Returns the predicate list and count for a func-like op. Sets |out_count|
// to 0 and returns NULL for ops with no predicate list attr or if |func| is
// not valid.
const loom_predicate_t* loom_func_like_predicates(loom_func_like_t func,
                                                  uint16_t* out_count);

// Returns the implements string ID for template/ukernel ops — the name of the
// op kind this function provides an implementation for. Returns
// LOOM_STRING_ID_INVALID for def/decl ops, ops with no implements attr, or
// if |func| is not valid.
loom_string_id_t loom_func_like_implements(loom_func_like_t func);

// Returns the dispatch priority for template/ukernel ops. Returns 0 for
// def/decl ops, ops with no priority attr, or if |func| is not valid.
int64_t loom_func_like_priority(loom_func_like_t func);

//===----------------------------------------------------------------------===//
// TargetLike interface helpers
//===----------------------------------------------------------------------===//

// Returns true if |target| refers to a valid target-like op. A cast via
// loom_target_like_cast() returns {NULL, NULL} on failure. All accessor helpers
// below tolerate a NULL vtable and return safe defaults.
static inline bool loom_target_like_isa(loom_target_like_t target) {
  return target.op != NULL;
}

// Casts |op| to loom_target_like_t if it implements the TargetLike interface.
// Returns {NULL, NULL} if |op| is NULL or does not implement it. Safe to call
// unconditionally; callers check the result with loom_target_like_isa().
loom_target_like_t loom_target_like_cast(const loom_module_t* module,
                                         const loom_op_t* op);

// Returns the symbol ref naming a target-like op, or null if |target| is not
// valid.
loom_symbol_ref_t loom_target_like_symbol(loom_target_like_t target);

// Returns the typed selector attr that chooses the target row used as the
// projection base, or an absent attr if |target| is not valid.
loom_attribute_t loom_target_like_selector(loom_target_like_t target);

// Returns the target-specific extension attrs, or an empty slice if absent.
loom_named_attr_slice_t loom_target_like_extension_attrs(
    loom_target_like_t target);

// Returns the opaque target-family projection descriptor, or NULL if absent.
const loom_target_like_descriptor_t* loom_target_like_descriptor(
    loom_target_like_t target);

//===----------------------------------------------------------------------===//
// LoopLike interface
//===----------------------------------------------------------------------===//

// Returns true if |loop| refers to a valid loop-like op. All accessor
// helpers below tolerate a NULL vtable and return safe defaults.
static inline bool loom_loop_like_isa(loom_loop_like_t loop) {
  return loop.op != NULL;
}

// Casts |op| to loom_loop_like_t if it implements the LoopLike interface.
// Returns {NULL, NULL} if |op| is NULL or does not implement it. Safe to
// call unconditionally — callers check the result with loom_loop_like_isa().
loom_loop_like_t loom_loop_like_cast(const loom_module_t* module,
                                     loom_op_t* op);

// Returns the primary body region of a loop-like op, or NULL if |loop|
// is not valid.
loom_region_t* loom_loop_like_body(loom_loop_like_t loop);

// Returns the condition region of a loop-like op, or NULL for loops
// without a separate condition region (scf.for) or if |loop| is not
// valid. For scf.while this returns the "before" region.
loom_region_t* loom_loop_like_condition_region(loom_loop_like_t loop);

// Returns the induction variable value ID for a loop-like op, or
// LOOM_VALUE_ID_INVALID for loops without an induction variable
// (scf.while) or if |loop| is not valid. The IV is a block argument
// on the body region's entry block.
loom_value_id_t loom_loop_like_iv(loom_loop_like_t loop);

// Returns the initial values for loop-carried state. The LoopLike vtable names
// the author-facing operand field; this accessor resolves that field through
// the op layout so segmented policy operands do not leak into analyses.
loom_value_slice_t loom_loop_like_iter_args(loom_loop_like_t loop);

// Returns the lower-bound operand value ID for counted loops, or
// LOOM_VALUE_ID_INVALID for non-counted loops or malformed op instances.
loom_value_id_t loom_loop_like_lower_bound(loom_loop_like_t loop);

// Returns the upper-bound operand value ID for counted loops, or
// LOOM_VALUE_ID_INVALID for non-counted loops or malformed op instances.
loom_value_id_t loom_loop_like_upper_bound(loom_loop_like_t loop);

// Returns the step operand value ID for counted loops, or
// LOOM_VALUE_ID_INVALID for non-counted loops or malformed op instances.
loom_value_id_t loom_loop_like_step(loom_loop_like_t loop);

// Returns true when all counted-loop range operands are present.
bool loom_loop_like_has_counted_range(loom_loop_like_t loop);

//===----------------------------------------------------------------------===//
// RegionBranch interface
//===----------------------------------------------------------------------===//

// Returns true if |branch| refers to a valid region-branch op. All
// accessor helpers below tolerate a NULL vtable and return safe defaults.
static inline bool loom_region_branch_isa(loom_region_branch_t branch) {
  return branch.op != NULL;
}

// Casts |op| to loom_region_branch_t if it implements the RegionBranch
// interface. Returns {NULL, NULL} if |op| is NULL or does not implement
// it. Safe to call unconditionally — callers check the result with
// loom_region_branch_isa().
loom_region_branch_t loom_region_branch_cast(const loom_module_t* module,
                                             loom_op_t* op);

// Returns the selector operand value ID for a region-branch op, or
// LOOM_VALUE_ID_INVALID if |branch| is not valid. For scf.if this is
// the i1 condition; for scf.switch this is the index selector.
loom_value_id_t loom_region_branch_selector(loom_region_branch_t branch);

// Returns the branch region at |region_index|, or NULL for malformed inputs.
// Region 0 is the first physical region on the op; dialect-specific accessors
// define whether that is a default, then, or other semantic branch.
loom_region_t* loom_region_branch_region(const loom_module_t* module,
                                         loom_region_branch_t branch,
                                         uint8_t region_index);

// Returns the single-block terminator for a branch region when it matches the
// region descriptor's required terminator kind. Returns NULL for malformed
// inputs, multi-block regions, missing terminators, or wrong terminator kinds.
loom_op_t* loom_region_branch_region_terminator(const loom_module_t* module,
                                                loom_region_branch_t branch,
                                                uint8_t region_index);

// Returns true when a branch region consists only of its terminator and that
// terminator forwards exactly |expected_count| operands. The returned slice
// aliases the terminator operands and is valid until the op is rewritten.
bool loom_region_branch_region_yield_only_operands(
    const loom_module_t* module, loom_region_branch_t branch,
    uint8_t region_index, uint16_t expected_count,
    loom_value_slice_t* out_values);

//===----------------------------------------------------------------------===//
// MemoryAccess interface
//===----------------------------------------------------------------------===//

// Returns true if |access| refers to a valid memory-access op. All accessor
// helpers below tolerate a NULL vtable and return safe defaults.
static inline bool loom_memory_access_isa(loom_memory_access_t access) {
  return access.op != NULL;
}

// Casts |op| to loom_memory_access_t if it implements the MemoryAccess
// interface. Returns {NULL, NULL} if |op| is NULL or does not implement it.
// Safe to call unconditionally; callers check the result with
// loom_memory_access_isa().
loom_memory_access_t loom_memory_access_cast(const loom_module_t* module,
                                             const loom_op_t* op);

// Returns the accessed view or memory-object operand.
loom_value_id_t loom_memory_access_view(loom_memory_access_t access);

// Returns the written value or atomic update contribution operand.
loom_value_id_t loom_memory_access_value(loom_memory_access_t access);

// Returns the compare-exchange expected-value operand.
loom_value_id_t loom_memory_access_expected(loom_memory_access_t access);

// Returns the compare-exchange replacement-value operand.
loom_value_id_t loom_memory_access_replacement(loom_memory_access_t access);

// Returns the lane/activity mask operand.
loom_value_id_t loom_memory_access_mask(loom_memory_access_t access);

// Returns the passthrough operand for inactive result lanes.
loom_value_id_t loom_memory_access_passthrough(loom_memory_access_t access);

// Returns the per-lane offsets operand.
loom_value_id_t loom_memory_access_offsets(loom_memory_access_t access);

// Returns the dynamic logical-origin index operand slice.
loom_value_slice_t loom_memory_access_dynamic_indices(
    loom_memory_access_t access);

// Returns the static logical-origin indices attr.
loom_attribute_t loom_memory_access_static_indices(loom_memory_access_t access);

// Returns the optional cache/coherency-scope attr.
loom_attribute_t loom_memory_access_cache_scope(loom_memory_access_t access);

// Returns the optional temporal cache-policy attr.
loom_attribute_t loom_memory_access_cache_temporal(loom_memory_access_t access);

// Returns the atomic update-kind attr.
loom_attribute_t loom_memory_access_atomic_kind(loom_memory_access_t access);

// Returns the single atomic memory-ordering attr.
loom_attribute_t loom_memory_access_atomic_ordering(
    loom_memory_access_t access);

// Returns the compare-exchange success memory-ordering attr.
loom_attribute_t loom_memory_access_atomic_success_ordering(
    loom_memory_access_t access);

// Returns the compare-exchange failure memory-ordering attr.
loom_attribute_t loom_memory_access_atomic_failure_ordering(
    loom_memory_access_t access);

// Returns the atomic synchronization-scope attr.
loom_attribute_t loom_memory_access_atomic_scope(loom_memory_access_t access);

//===----------------------------------------------------------------------===//
// Op definition macros
//===----------------------------------------------------------------------===//
//
// Generated per-dialect ops.h files use these macros to define typed
// inline accessor functions for each op.
//
// Usage in generated ops.h:
//
//   // LOOM_OP_TEST_ADDI: Test binary integer op.
//   // %result = test.addi %lhs, %rhs : type
//   LOOM_DEFINE_ISA(loom_test_addi_isa, LOOM_OP_TEST_ADDI)
//   LOOM_DEFINE_OPERAND(loom_test_addi_lhs, 0)
//   LOOM_DEFINE_OPERAND(loom_test_addi_rhs, 1)
//   LOOM_DEFINE_RESULT(loom_test_addi_result, 0)

// Defines a function that checks if an op is of a specific kind.
#define LOOM_DEFINE_ISA(func_name, kind_enum)         \
  static inline bool func_name(const loom_op_t* op) { \
    return op != NULL && op->kind == (kind_enum);     \
  }

// Defines a function that reads a fixed operand by index.
#define LOOM_DEFINE_OPERAND(func_name, index)                    \
  static inline loom_value_id_t func_name(const loom_op_t* op) { \
    return loom_op_operands(op)[(index)];                        \
  }

// Defines functions that query and read an optional fixed operand by index.
#define LOOM_DEFINE_OPTIONAL_OPERAND(func_name, index)                \
  enum { func_name##_OPERAND_INDEX = (index) };                       \
  static inline bool func_name##_is_present(const loom_op_t* op) {    \
    return op->operand_count > (index);                               \
  }                                                                   \
  static inline loom_value_id_t func_name(const loom_op_t* op) {      \
    return func_name##_is_present(op) ? loom_op_operands(op)[(index)] \
                                      : LOOM_VALUE_ID_INVALID;        \
  }

// Defines a function that reads a fixed result by index.
#define LOOM_DEFINE_RESULT(func_name, index)                     \
  static inline loom_value_id_t func_name(const loom_op_t* op) { \
    return loom_op_results(op)[(index)];                         \
  }

// Defines a function that returns the variadic operand tail as a value
// slice. |fixed_count| is the number of non-variadic operands before
// the variadic tail.
#define LOOM_DEFINE_VARIADIC_OPERANDS(func_name, fixed_count)       \
  static inline loom_value_slice_t func_name(const loom_op_t* op) { \
    loom_value_slice_t slice;                                       \
    slice.values = loom_op_operands(op) + (fixed_count);            \
    slice.count = (uint16_t)(op->operand_count - (fixed_count));    \
    return slice;                                                   \
  }

// Defines a function that reads a single segmented operand field. The op kind
// must store operand segment counts in trailing storage.
#define LOOM_DEFINE_SEGMENTED_OPERAND(func_name, field_index)            \
  static inline loom_value_id_t func_name(const loom_op_t* op) {         \
    const uint16_t* counts = loom_op_const_operand_segment_counts(op);   \
    uint16_t start = 0;                                                  \
    for (uint8_t _i = 0; _i < (field_index); ++_i) {                     \
      start += counts[_i];                                               \
    }                                                                    \
    return counts[(field_index)] > 0 ? loom_op_const_operands(op)[start] \
                                     : LOOM_VALUE_ID_INVALID;            \
  }

// Defines functions that query and read an optional segmented operand field.
#define LOOM_DEFINE_SEGMENTED_OPTIONAL_OPERAND(func_name, field_index)  \
  enum { func_name##_OPERAND_FIELD_INDEX = (field_index) };             \
  static inline bool func_name##_is_present(const loom_op_t* op) {      \
    return loom_op_const_operand_segment_counts(op)[(field_index)] > 0; \
  }                                                                     \
  static inline loom_value_id_t func_name(const loom_op_t* op) {        \
    const uint16_t* counts = loom_op_const_operand_segment_counts(op);  \
    if (counts[(field_index)] == 0) return LOOM_VALUE_ID_INVALID;       \
    uint16_t start = 0;                                                 \
    for (uint8_t _i = 0; _i < (field_index); ++_i) {                    \
      start += counts[_i];                                              \
    }                                                                   \
    return loom_op_const_operands(op)[start];                           \
  }

// Defines a function that returns a segmented operand field as a value slice.
#define LOOM_DEFINE_SEGMENTED_OPERANDS(func_name, field_index)         \
  static inline loom_value_slice_t func_name(const loom_op_t* op) {    \
    const uint16_t* counts = loom_op_const_operand_segment_counts(op); \
    uint16_t start = 0;                                                \
    for (uint8_t _i = 0; _i < (field_index); ++_i) {                   \
      start += counts[_i];                                             \
    }                                                                  \
    return (loom_value_slice_t){                                       \
        /*.values=*/loom_op_operands(op) + start,                      \
        /*.count=*/counts[(field_index)],                              \
    };                                                                 \
  }

// Defines a function that returns the variadic result tail as a value
// slice. |fixed_count| is the number of non-variadic results before
// the variadic tail.
#define LOOM_DEFINE_VARIADIC_RESULTS(func_name, fixed_count)        \
  static inline loom_value_slice_t func_name(const loom_op_t* op) { \
    loom_value_slice_t slice;                                       \
    slice.values = loom_op_results(op) + (fixed_count);             \
    slice.count = (uint16_t)(op->result_count - (fixed_count));     \
    return slice;                                                   \
  }

// Defines a function that reads a region by index.
#define LOOM_DEFINE_REGION(func_name, index)                    \
  static inline loom_region_t* func_name(const loom_op_t* op) { \
    return loom_op_regions(op)[(index)];                        \
  }

// Defines a function that reads an optional region by index.
#define LOOM_DEFINE_OPTIONAL_REGION(func_name, index)                        \
  static inline loom_region_t* func_name(const loom_op_t* op) {              \
    return (index) < op->region_count ? loom_op_regions(op)[(index)] : NULL; \
  }

// Defines a function that reads a successor block by index.
#define LOOM_DEFINE_SUCCESSOR(func_name, index)                \
  static inline loom_block_t* func_name(const loom_op_t* op) { \
    return loom_op_successors(op)[(index)];                    \
  }

// Defines a function that returns the variadic region tail as a slice.
// |fixed_count| is the number of non-variadic regions before the tail.
#define LOOM_DEFINE_VARIADIC_REGIONS(func_name, fixed_count)         \
  static inline loom_region_slice_t func_name(const loom_op_t* op) { \
    loom_region_slice_t slice;                                       \
    slice.regions = loom_op_regions(op) + (fixed_count);             \
    slice.count = (uint8_t)(op->region_count - (fixed_count));       \
    return slice;                                                    \
  }

// Defines a function that returns the variadic successor tail as a slice.
// |fixed_count| is the number of non-variadic successors before the tail.
#define LOOM_DEFINE_VARIADIC_SUCCESSORS(func_name, fixed_count)         \
  static inline loom_successor_slice_t func_name(const loom_op_t* op) { \
    loom_successor_slice_t slice;                                       \
    slice.blocks = loom_op_successors(op) + (fixed_count);              \
    slice.count = (uint8_t)(op->successor_count - (fixed_count));       \
    return slice;                                                       \
  }

// Each LOOM_DEFINE_ATTR_* macro defines both a typed accessor function
// and a compile-time constant for the attribute's index in the attr
// array: func_name##_ATTR_INDEX. This lets canonicalize callbacks use
// loom_rewriter_set_attr(rewriter, op, loom_foo_bar_ATTR_INDEX, value)
// without hardcoding magic numbers.

// Defines a function that reads an i64 attribute by index.
#define LOOM_DEFINE_ATTR_I64(func_name, index)           \
  enum { func_name##_ATTR_INDEX = (index) };             \
  static inline int64_t func_name(const loom_op_t* op) { \
    return loom_attr_as_i64(loom_op_attrs(op)[(index)]); \
  }

// Defines a function that reads an f64 attribute by index.
#define LOOM_DEFINE_ATTR_F64(func_name, index)           \
  enum { func_name##_ATTR_INDEX = (index) };             \
  static inline double func_name(const loom_op_t* op) {  \
    return loom_attr_as_f64(loom_op_attrs(op)[(index)]); \
  }

// Defines a function that reads an enum attribute by index.
// Returns the enum case index as a uint8_t.
#define LOOM_DEFINE_ATTR_ENUM(func_name, index)           \
  enum { func_name##_ATTR_INDEX = (index) };              \
  static inline uint8_t func_name(const loom_op_t* op) {  \
    return loom_attr_as_enum(loom_op_attrs(op)[(index)]); \
  }

// Defines a function that reads an enum attribute by index.
// Returns the enum case index as |enum_type|.
#define LOOM_DEFINE_ATTR_ENUM_TYPED(func_name, index, enum_type)     \
  enum { func_name##_ATTR_INDEX = (index) };                         \
  static inline enum_type func_name(const loom_op_t* op) {           \
    return (enum_type)loom_attr_as_enum(loom_op_attrs(op)[(index)]); \
  }

// Defines a function that reads a symbol attribute by index.
#define LOOM_DEFINE_ATTR_SYMBOL(func_name, index)                  \
  enum { func_name##_ATTR_INDEX = (index) };                       \
  static inline loom_symbol_ref_t func_name(const loom_op_t* op) { \
    return loom_attr_as_symbol(loom_op_attrs(op)[(index)]);        \
  }

// Defines a function that reads a string attribute by index.
#define LOOM_DEFINE_ATTR_STRING(func_name, index)                 \
  enum { func_name##_ATTR_INDEX = (index) };                      \
  static inline loom_string_id_t func_name(const loom_op_t* op) { \
    return loom_attr_as_string_id(loom_op_attrs(op)[(index)]);    \
  }

// Defines a function that reads a bool attribute by index.
#define LOOM_DEFINE_ATTR_BOOL(func_name, index)           \
  enum { func_name##_ATTR_INDEX = (index) };              \
  static inline bool func_name(const loom_op_t* op) {     \
    return loom_attr_as_bool(loom_op_attrs(op)[(index)]); \
  }

// Defines a function that reads a static encoding attribute by index.
#define LOOM_DEFINE_ATTR_ENCODING(func_name, index)              \
  enum { func_name##_ATTR_INDEX = (index) };                     \
  static inline uint16_t func_name(const loom_op_t* op) {        \
    return loom_attr_as_encoding_id(loom_op_attrs(op)[(index)]); \
  }

// Defines a function that reads a type-table attribute by index.
#define LOOM_DEFINE_ATTR_TYPE(func_name, index)                 \
  enum { func_name##_ATTR_INDEX = (index) };                    \
  static inline loom_type_id_t func_name(const loom_op_t* op) { \
    return loom_attr_as_type_id(loom_op_attrs(op)[(index)]);    \
  }

// Defines a function that reads an i64 array attribute by index.
#define LOOM_DEFINE_ATTR_I64_ARRAY(func_name, index)              \
  enum { func_name##_ATTR_INDEX = (index) };                      \
  static inline loom_attribute_t func_name(const loom_op_t* op) { \
    return loom_op_attrs(op)[(index)];                            \
  }

// Defines a function that reads a DICT attribute by index.
#define LOOM_DEFINE_ATTR_DICT(func_name, index)                          \
  enum { func_name##_ATTR_INDEX = (index) };                             \
  static inline loom_named_attr_slice_t func_name(const loom_op_t* op) { \
    return loom_attr_as_dict(loom_op_attrs(op)[(index)]);                \
  }

// Defines a function that reads a generic attribute payload by index.
#define LOOM_DEFINE_ATTR_ANY(func_name, index)                    \
  enum { func_name##_ATTR_INDEX = (index) };                      \
  static inline loom_attribute_t func_name(const loom_op_t* op) { \
    return loom_op_attrs(op)[(index)];                            \
  }

// Defines a function that reads the per-instance flags byte.
// Used for fast-math flags (float ops) and overflow flags (integer ops).
#define LOOM_DEFINE_INSTANCE_FLAGS(func_name)            \
  static inline uint8_t func_name(const loom_op_t* op) { \
    return op->instance_flags;                           \
  }

//===----------------------------------------------------------------------===//
// Builder
//===----------------------------------------------------------------------===//

// Saved insertion point for nested region construction. Stack-allocated.
// Use loom_builder_save/loom_builder_restore to switch between blocks.
// The parent_op field is saved/restored so that ops created inside
// nested regions get the correct ancestry on their parent_op pointer.
typedef struct loom_builder_ip_t {
  // Block to insert into.
  loom_block_t* block;
  // Op whose region contains |block|. NULL at the module level.
  // Stamped onto every op created at this insertion point as
  // op->parent_op, giving O(depth) ancestry queries.
  loom_op_t* parent_op;
  // Live op to insert before. NULL means append to the end of the block.
  loom_op_t* before_op;
} loom_builder_ip_t;

// Callback invoked when an op is fully constructed (after finalize_op).
// Used by the rewriter to add newly created ops to its worklist.
typedef iree_status_t (*loom_builder_op_fn_t)(void* user_data, loom_op_t* op);
typedef struct loom_builder_callback_t {
  loom_builder_op_fn_t fn;
  void* user_data;
} loom_builder_callback_t;

// The builder is the single API surface for IR construction.
//
// It bundles the target module (for value/string/type tables), the
// allocation arena, and the current insertion point into one object.
// Every generated op builder takes a loom_builder_t* as its first
// parameter. Generic helpers (inlining, cloning, pattern rewriting)
// take a builder without knowing the source or target module.
//
// The arena is separate from the module's arena so that callers can
// control storage lifetime: linking into a fresh arena, per-thread
// arenas during parallel compilation, etc.
//
// The optional on_op_finalized callback fires after finalize_op
// completes (uses registered, def pointers set). NULL when unused.
typedef struct loom_builder_t {
  loom_module_t* module;
  iree_arena_allocator_t* arena;
  loom_builder_ip_t ip;
  loom_builder_callback_t on_op_finalized;
  // Pre-allocated result value_ids for the next op build. When
  // reserved_result_count > 0, loom_builder_define_value consumes
  // from this array instead of allocating new value_ids. Cleared
  // by loom_builder_finalize_op after verifying all were consumed.
  const loom_value_id_t* reserved_result_ids;
  iree_host_size_t reserved_result_count;
  iree_host_size_t reserved_result_next;
} loom_builder_t;

// Initializes a builder that appends to |block|.
void loom_builder_initialize(loom_module_t* module,
                             iree_arena_allocator_t* arena, loom_block_t* block,
                             loom_builder_t* out_builder);

// Moves the insertion point to before |op| and inherits |op|'s parent op.
// Used during rewrites: insert replacement ops, then erase the original.
void loom_builder_set_before(loom_builder_t* builder, const loom_op_t* op);

// Moves the insertion point to after |op| and inherits |op|'s parent op.
void loom_builder_set_after(loom_builder_t* builder, const loom_op_t* op);

// Moves the insertion point to the end of |block| (append mode).
// Does not change parent_op — use loom_builder_enter_region when
// entering a nested region to ensure correct parent ancestry.
void loom_builder_set_block(loom_builder_t* builder, loom_block_t* block);

// Enters a nested region for building ops inside it. Saves the
// current insertion point (use loom_builder_restore to return),
// sets the insertion block to the region's entry block, and sets
// parent_op so that ops created inside inherit correct ancestry.
//
//   loom_builder_ip_t saved = loom_builder_enter_region(
//       &builder, parent_op, region);
//   // ... build ops inside region ...
//   loom_builder_restore(&builder, saved);
loom_builder_ip_t loom_builder_enter_region(loom_builder_t* builder,
                                            loom_op_t* parent_op,
                                            loom_region_t* region);

// Builds the required yield-style terminator for |region_index| using
// |values| as its forwarded operands. RegionBranch terminators are required to
// be operand-only terminators; dialects needing richer terminators need a more
// specific interface before participating in generic branch factoring.
iree_status_t loom_region_branch_build_region_terminator(
    loom_builder_t* builder, const loom_module_t* module,
    loom_region_branch_t branch, uint8_t region_index,
    const loom_value_id_t* values, iree_host_size_t value_count,
    loom_location_id_t location, loom_op_t** out_op);

// Saves the current insertion point. Returns a value that can be
// passed to loom_builder_restore to return to this position.
loom_builder_ip_t loom_builder_save(const loom_builder_t* builder);

// Restores a previously saved insertion point.
void loom_builder_restore(loom_builder_t* builder, loom_builder_ip_t ip);

// Pre-allocates |count| result value_ids in the module's value table.
// The values are real entries with uninitialized types. The next |count|
// calls to loom_builder_define_value (typically from a generated builder)
// will assign types to these values instead of allocating fresh ones.
// loom_builder_finalize_op verifies all reserved results were consumed.
//
// This enables constructing result types that reference other results
// by value_id before the build call:
//
//   loom_value_id_t result_ids[2];
//   loom_builder_reserve_results(&builder, 2, result_ids);
//   loom_type_t output_type = loom_type_shaped_1d(
//       LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32,
//       loom_dim_pack_dynamic(result_ids[1]), 0);
//   loom_type_t result_types[] = {output_type, index_type};
//   loom_test_deflate_build(&builder, input, result_types, 2, ...);
//
iree_status_t loom_builder_reserve_results(loom_builder_t* builder,
                                           iree_host_size_t count,
                                           loom_value_id_t* out_result_ids);

// Creates a fresh value in the module's value table with the given type.
// Returns the value ID. The value has no defining op yet (set by the
// builder when the op is inserted). If results were reserved via
// loom_builder_reserve_results, consumes the next reserved id and
// assigns the type to it.
iree_status_t loom_builder_define_value(loom_builder_t* builder,
                                        loom_type_t type,
                                        loom_value_id_t* out_value_id);

// Interns a string in the module's string table. Returns the string ID.
// Identical strings share the same ID.
iree_status_t loom_builder_intern_string(loom_builder_t* builder,
                                         iree_string_view_t string,
                                         loom_string_id_t* out_string_id);

// Canonicalizes a keyed variadic operand dictionary.
//
// Writes the sorted operand values into |operand_storage| and writes a
// canonical DICT attribute to |out_names_attr| mapping each key to its operand
// ordinal relative to |operand_storage|. Empty dictionaries produce an absent
// names attribute so optional OperandDict fields print nothing.
iree_status_t loom_builder_set_operand_dict(
    loom_builder_t* builder, loom_named_value_slice_t named_values,
    loom_value_id_t* operand_storage, loom_attribute_t* out_names_attr);

// Checks that a generated builder count fits the target storage width.
iree_status_t loom_builder_check_count_range(iree_host_size_t count,
                                             iree_host_size_t max_count,
                                             iree_string_view_t label);

// Copies an i64-array attribute payload into the builder arena.
iree_status_t loom_builder_copy_i64_array_attr_storage(loom_builder_t* builder,
                                                       const int64_t* values,
                                                       iree_host_size_t count,
                                                       iree_string_view_t label,
                                                       int64_t** out_storage);

// Copies a predicate-list attribute payload into the builder arena.
iree_status_t loom_builder_copy_predicate_list_attr_storage(
    loom_builder_t* builder, const loom_predicate_t* predicates,
    iree_host_size_t count, iree_string_view_t label,
    loom_predicate_t** out_storage);

// Creates a one-block region and installs it into |op| at |region_index|.
iree_status_t loom_builder_create_region(loom_builder_t* builder, loom_op_t* op,
                                         uint8_t region_index,
                                         loom_block_t** out_entry_block);

// Creates a fresh value with the given type and adds it as a block
// argument. Convenience wrapper for the define_value + block_add_arg
// sequence that generated builders use when auto-creating regions.
iree_status_t loom_builder_define_block_arg(loom_builder_t* builder,
                                            loom_block_t* block,
                                            loom_type_t type,
                                            loom_value_id_t* out_value_id);

// Defines one op result value and writes the result id to |out_result|.
iree_status_t loom_builder_define_result(loom_builder_t* builder,
                                         loom_type_t result_type,
                                         loom_value_id_t* out_result);

// Defines |result_count| op result values into |result_storage|.
iree_status_t loom_builder_define_results(loom_builder_t* builder,
                                          const loom_type_t* result_types,
                                          iree_host_size_t result_count,
                                          loom_value_id_t* result_storage);

// Copies tied-result metadata into |op|.
iree_status_t loom_builder_copy_tied_results(
    const loom_tied_result_t* tied_results, iree_host_size_t tied_result_count,
    loom_op_t* op);

// Allocates an op with the given field counts and inserts it at the
// builder's current insertion point. This is the low-level primitive
// that generated builders call. The caller fills in trailing data
// (operands, results, regions, tied results, attributes) through the
// accessor functions above. Successor storage is omitted.
iree_status_t loom_builder_allocate_op(
    loom_builder_t* builder, loom_op_kind_t kind, uint16_t operand_count,
    uint16_t result_count, uint8_t region_count, uint16_t tied_result_count,
    uint8_t attribute_count, loom_location_id_t location, loom_op_t** out_op);

// Allocates an op with explicit successor storage and inserts it at the
// builder's current insertion point. Successor slots are semantic block
// targets used by CFG terminators; labels remain display names and parser
// syntax. The caller fills successors through loom_op_successors(op), then
// fills the ordinary trailing fields through their accessors.
iree_status_t loom_builder_allocate_op_with_successors(
    loom_builder_t* builder, loom_op_kind_t kind, uint16_t operand_count,
    uint16_t result_count, uint8_t successor_count, uint8_t region_count,
    uint16_t tied_result_count, uint8_t attribute_count,
    loom_location_id_t location, loom_op_t** out_op);

// Allocates an op with segmented operand metadata. |operand_segment_counts|
// must have one entry per operand descriptor on op kinds whose vtable has
// LOOM_OP_VTABLE_SEGMENTED_OPERANDS. The counts must sum to |operand_count|.
iree_status_t loom_builder_allocate_segmented_op(
    loom_builder_t* builder, loom_op_kind_t kind, uint16_t operand_count,
    const uint16_t* operand_segment_counts, uint8_t operand_segment_count,
    uint16_t result_count, uint8_t region_count, uint16_t tied_result_count,
    uint8_t attribute_count, loom_location_id_t location, loom_op_t** out_op);

// Allocates a segmented-operand op with explicit successor storage.
iree_status_t loom_builder_allocate_segmented_op_with_successors(
    loom_builder_t* builder, loom_op_kind_t kind, uint16_t operand_count,
    const uint16_t* operand_segment_counts, uint8_t operand_segment_count,
    uint16_t result_count, uint8_t successor_count, uint8_t region_count,
    uint16_t tied_result_count, uint8_t attribute_count,
    loom_location_id_t location, loom_op_t** out_op);

// Removes selected results from a variadic-result op and compacts trailing
// storage in-place.
//
// |remove_results| has one entry per current result. Removed result values must
// have no operand uses and no incoming type uses. Dropped values remain in the
// module value table but no longer carry defining-op identity or outgoing
// type-use records. Kept result values keep their IDs and receive updated
// definition indices. Tied results targeting removed result slots are rejected;
// kept tied result indices are remapped.
iree_status_t loom_op_remove_results(loom_module_t* module, loom_op_t* op,
                                     const bool* remove_results,
                                     iree_arena_allocator_t* scratch_arena,
                                     uint16_t* out_removed_count);

// Erases an op: removes all use records for the op's operands, verifies
// that every result has no operand uses or external type uses (caller must
// RAUW results first), drops type-use records carried by result and nested
// block-argument types, then marks the op dead. Dead ops are skipped by
// enumeration macros and will not be serialized. The memory is not freed
// (arena-owned). Returns
// IREE_STATUS_FAILED_PRECONDITION if any result still has uses.
iree_status_t loom_op_erase(loom_module_t* module, loom_op_t* op);

// Removes a closed set of non-entry blocks from |region| and compacts the
// region block table in place.
//
// |remove_blocks| must contain exactly |remove_block_count| entries, one per
// current block index in |region|. Entry block removal is rejected. Any kept op
// successor targeting a removed block is rejected. Values defined by removed
// block arguments or removed op subtrees may only have operand and type uses
// inside the removed set; callers must retarget or replace external uses before
// removing blocks.
//
// Removed block/op/value objects remain arena-owned for diagnostics, but the
// blocks are detached from the region, their operations are marked dead, and
// block-argument identity/type-use records are dropped.
iree_status_t loom_region_remove_blocks(loom_module_t* module,
                                        loom_region_t* region,
                                        const bool* remove_blocks,
                                        uint16_t remove_block_count,
                                        uint16_t* out_removed_count);

//===----------------------------------------------------------------------===//
// Use-def list maintenance
//===----------------------------------------------------------------------===//
//
// These functions keep value use lists always-correct. All operand
// mutations must go through these functions (or the builder finalize
// path) to maintain the invariant that every operand has a corresponding
// use entry on the referenced value.

// Adds a use record: |user_op| uses value |value_id| at |operand_index|.
// Handles inline-to-overflow transition via arena allocation on the module.
iree_status_t loom_value_add_use(loom_module_t* module,
                                 loom_value_id_t value_id, loom_op_t* user_op,
                                 uint16_t operand_index);

// Removes a use record: |user_op| no longer uses |value_id| at
// |operand_index|. Scans the use list for the matching entry, swaps
// with last, decrements use_count. Returns IREE_STATUS_NOT_FOUND if
// no matching entry exists (indicates a use-list bookkeeping bug).
// No overflow-to-inline transition (arena cannot free the overflow
// array; loom_module_compute_uses handles repack).
iree_status_t loom_value_remove_use(loom_module_t* module,
                                    loom_value_id_t value_id,
                                    loom_op_t* user_op, uint16_t operand_index);

// Finalizes a newly-built op: registers all operand uses and performs
// any other per-op bookkeeping. Called as the tail return from every
// builder: `return loom_builder_finalize_op(builder, *out_op);`
iree_status_t loom_builder_finalize_op(loom_builder_t* builder, loom_op_t* op);

// Records SSA value references embedded in |op|'s attributes. Attribute
// references are tracked as a conservative per-value bit so RAUW can avoid
// scanning operation attributes when replacing ordinary operand-only values.
iree_status_t loom_module_note_op_attribute_value_refs(loom_module_t* module,
                                                       const loom_op_t* op);

// Links a symbol-defining op to its symbol table entry using the op's generated
// symbol definition descriptor. Sets the symbol's defining op, definition
// descriptor, and legacy bytecode kind. Idempotent.
void loom_module_link_symbol_defining_op(loom_module_t* module, loom_op_t* op,
                                         const loom_op_vtable_t* vtable);

// Changes an operand on an existing op, maintaining use lists. Removes
// the use from the old value, writes the new value ID, and adds a use
// to the new value. Skips LOOM_VALUE_ID_INVALID for both old and new.
iree_status_t loom_op_set_operand(loom_module_t* module, loom_op_t* op,
                                  uint16_t operand_index,
                                  loom_value_id_t new_value_id);

// Replaces all uses of |old_id| with |new_id|. Walks old's operand use list,
// patches each user op's operand slot, bulk-transfers those use entries to
// new's list, and rewrites SSA references embedded in value types and operation
// attributes. No-op if old_id == new_id.
iree_status_t loom_value_replace_all_uses_with(loom_module_t* module,
                                               loom_value_id_t old_id,
                                               loom_value_id_t new_id);

// Same as replace_all_uses_with, but skips uses where the user op is
// |except_op|. This filtered form only rewrites operand slots; embedded type
// references have no user op to predicate against. Used during pattern rewrites
// where the replacement op also references the old value.
iree_status_t loom_value_replace_all_uses_except(loom_module_t* module,
                                                 loom_value_id_t old_id,
                                                 loom_value_id_t new_id,
                                                 const loom_op_t* except_op);

// Predicate-based RAUW. Replaces operand uses of |old_id| with |new_id| only
// where |predicate| returns true for the user op. Embedded type references are
// intentionally not rewritten by this filtered form.
typedef bool (*loom_use_predicate_fn)(const loom_op_t* user_op,
                                      void* user_data);
iree_status_t loom_value_replace_uses_if(loom_module_t* module,
                                         loom_value_id_t old_id,
                                         loom_value_id_t new_id,
                                         loom_use_predicate_fn predicate,
                                         void* user_data);

// Rebuilds all use lists from scratch by walking every live op in the
// module. Clears all values' use data, then re-adds uses from operands.
// Used after parsing (the parser fills operands but not use lists) and
// as a recovery path after bulk IR mutations.
iree_status_t loom_module_compute_uses(loom_module_t* module);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_OP_DEFS_H_
