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
#include "loom/ir/attribute_schema.h"
#include "loom/ir/ir.h"
#include "loom/ir/semantics.h"
#include "loom/ir/type_constraint.h"
#include "loom/ir/types.h"
#include "loom/ops/attribute_accessors.h"
#include "loom/ops/format.h"
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
// Semantic constraints
//===----------------------------------------------------------------------===//
//
// Constraints express relationships between an op's fields that the
// verifier checks. Each op's vtable points to an array of constraint
// entries. The verifier walks the array, interpreting each by kind.
//
// Per-op cost: 10 bytes .rodata per constraint, zero .text. Adding constraint
// kinds extends one shared interpreter, not every op's generated code.
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

  // The number of values forwarded after a condition region terminator's
  // leading predicate matches another region's entry block argument count.
  // The third arg names the value field that validates the target block args;
  // malformed target args are diagnosed by their own region constraints.
  // Args: (condition region field, target region field, target variadic value
  // field). Used by ConditionForwardedCountMatchesBlockArgs.
  LOOM_RELATION_CONDITION_FORWARD_COUNT,

  // Each value forwarded after a condition region terminator's leading
  // predicate has the same type as the corresponding entry block argument of
  // another region. The third arg names the value field that validates the
  // target block args. Args: (condition region field, target region field,
  // target variadic value field). Used by
  // ConditionForwardedTypesMatchBlockArgs.
  LOOM_RELATION_CONDITION_FORWARD_MATCH,

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

  // A canonical dictionary maps unique names to a permutation of relative
  // operand ordinals. Args: (variadic operand field, optional dict attr field).
  // These rows form the vtable's operand_dictionary_count prefix and run during
  // structural verification, before the ordinary relation interpreter.
  LOOM_RELATION_OPERAND_DICTIONARY,

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

// A table-driven semantic constraint entry. 10 bytes.
//
// Each op's vtable points to an array of these. Operand dictionary rows form
// a prefix checked during structural verification; the remaining rows are
// interpreted by (relation, property) during semantic verification.
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
  // Result receives the exact ownership state of a consumed operand.
  LOOM_RESULT_OWNERSHIP_MOVED = 6,
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

// Semantic role of an operand field independent of its author-facing name.
enum loom_operand_role_e {
  // Operand has no special cross-op semantic role.
  LOOM_OPERAND_ROLE_NONE = 0,
  // Operand controls a branch or region transfer.
  LOOM_OPERAND_ROLE_CONTROL_CONDITION = 1,
  // Operand selects between value payloads.
  LOOM_OPERAND_ROLE_SELECT_CONDITION = 2,
  // Operand is one arm of a value-selecting operation. For operations with
  // multiple results, payload operands are row-major result tuples: payload
  // operand N selects result N modulo the result count.
  LOOM_OPERAND_ROLE_SELECT_PAYLOAD = 3,
  // Operand is broadcast into every element of a composite result.
  LOOM_OPERAND_ROLE_BROADCAST_SOURCE = 4,
  // Operand contributes one logical element to a composite result.
  LOOM_OPERAND_ROLE_COMPOSITE_ELEMENT = 5,
  // Operand is widened by a floating-point precision extension.
  LOOM_OPERAND_ROLE_FLOAT_EXTENSION_SOURCE = 6,
};
typedef uint8_t loom_operand_role_t;

enum loom_operand_role_mask_bits_e {
  LOOM_OPERAND_ROLE_MASK_CONTROL_CONDITION =
      1u << LOOM_OPERAND_ROLE_CONTROL_CONDITION,
  LOOM_OPERAND_ROLE_MASK_SELECT_CONDITION =
      1u << LOOM_OPERAND_ROLE_SELECT_CONDITION,
  LOOM_OPERAND_ROLE_MASK_SELECT_PAYLOAD = 1u
                                          << LOOM_OPERAND_ROLE_SELECT_PAYLOAD,
  LOOM_OPERAND_ROLE_MASK_BROADCAST_SOURCE =
      1u << LOOM_OPERAND_ROLE_BROADCAST_SOURCE,
  LOOM_OPERAND_ROLE_MASK_COMPOSITE_ELEMENT =
      1u << LOOM_OPERAND_ROLE_COMPOSITE_ELEMENT,
  LOOM_OPERAND_ROLE_MASK_FLOAT_EXTENSION_SOURCE =
      1u << LOOM_OPERAND_ROLE_FLOAT_EXTENSION_SOURCE,
};
typedef uint8_t loom_operand_role_mask_t;

static inline loom_operand_role_mask_t loom_operand_role_mask_bit(
    loom_operand_role_t role) {
  return role == LOOM_OPERAND_ROLE_NONE || role >= 8
             ? 0
             : (loom_operand_role_mask_t)(1u << role);
}

enum loom_result_flag_bits_e {
  LOOM_RESULT_VARIADIC = 1u << 0,
  LOOM_RESULT_ALLOCATES = 1u << 1,
};
typedef uint8_t loom_result_flags_t;

enum loom_region_flag_bits_e {
  // Region must contain exactly one block.
  LOOM_REGION_SINGLE_BLOCK = 1u << 0,
  // Region may be absent when it is part of a trailing optional suffix.
  LOOM_REGION_OPTIONAL = 1u << 1,
  // Region entry block arguments are projected from the op's FuncArgs
  // signature. Projected values have matching names and types, with type
  // references to signature peers remapped to the corresponding region-owned
  // arguments. Each region owns distinct SSA values.
  LOOM_REGION_PROJECT_FUNC_ARGS = 1u << 2,
  // Buffer entry block arguments seed target-independent global memory facts.
  // This is a region signature contract, not a property of the generic buffer
  // type.
  LOOM_REGION_GLOBAL_BUFFER_ARGS = 1u << 3,
  // Scalar entry block arguments are identical across the workgroup. This is
  // a region signature contract and does not apply to nested region arguments.
  LOOM_REGION_WORKGROUP_UNIFORM_ARGS = 1u << 4,
  // Scalar entry block arguments are identical across every workgroup in a
  // workgroup cluster. This implies workgroup uniformity.
  LOOM_REGION_CLUSTER_UNIFORM_ARGS = 1u << 5,
  // Every directly observable effect in this region or a nested region must
  // be represented by an op carrying LOOM_TRAIT_COMMAND_EFFECT.
  LOOM_REGION_COMMAND_EFFECTS_ONLY = 1u << 6,
  // The parent format declares the entry block arguments before this region.
  // Generated from signature/region syntax so printers need no format scan.
  LOOM_REGION_PARENT_DECLARED_ARGS = 1u << 7,
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
  // Semantic role of this operand field.
  loom_operand_role_t role;
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

typedef uint16_t loom_symbol_definition_flags_t;

enum loom_symbol_definition_flag_bits_e {
  // Symbol op declares a contract that a provider definition may satisfy.
  LOOM_SYMBOL_DEFINITION_FLAG_DECLARATION = 1u << 0,
  // Symbol exists only for test or benchmark tooling.
  LOOM_SYMBOL_DEFINITION_FLAG_TEST_ONLY = 1u << 1,
};

// Generated metadata for an op that defines a module symbol.
typedef struct loom_symbol_definition_descriptor_t {
  // Human-readable symbol class used in diagnostics.
  loom_bstring_t name;
  // Structural symbol interfaces implemented by this definition.
  loom_symbol_interface_flags_t interfaces;
  // Definition roles such as declaration status.
  loom_symbol_definition_flags_t flags;
  // Attribute index of the symbol identity field on the defining op.
  uint8_t name_attr_index;
  // Attribute index plus one for optional public visibility, or 0 if absent.
  uint8_t visibility_attr_index_plus_one;
  // Attribute index plus one for the optional retain marker, or 0 if absent.
  uint8_t retain_attr_index_plus_one;
  // Attribute index plus one for an enum product carrier, or 0 if this symbol
  // definition has no product-carrier contract.
  uint8_t product_carrier_attr_index_plus_one;
  // Result index plus one for a typed-value contract, or 0 if absent.
  uint8_t value_contract_result_index_plus_one;
  // Attribute index plus one for an exact contract value, or 0 if absent.
  uint8_t value_contract_value_attr_index_plus_one;
  // Attribute index plus one for contract predicates, or 0 if absent.
  uint8_t value_contract_predicates_attr_index_plus_one;
  // Existing bytecode payload kind, or LOOM_SYMBOL_NONE if not serializable
  // through the current SYMBOLS section.
  loom_symbol_kind_t bytecode_kind;
  // Region index plus one containing a kernel workload signature, or 0 when
  // the signature is stored as operands or the symbol is not a kernel.
  uint8_t kernel_workload_region_index_plus_one;
  // Operand field index plus one containing a kernel workload signature, or 0
  // when the signature is stored in a region or the symbol is not a kernel.
  uint8_t kernel_workload_operand_field_index_plus_one;
  // Optional domain that computes typed facts for symbols defined by this op.
  const loom_symbol_fact_domain_t* fact_domain;
} loom_symbol_definition_descriptor_t;

static_assert(sizeof(loom_symbol_definition_descriptor_t) == 32,
              "loom_symbol_definition_descriptor_t must be 32 bytes");

static inline iree_string_view_t loom_symbol_definition_descriptor_name(
    const loom_symbol_definition_descriptor_t* descriptor) {
  return descriptor ? loom_bstring_view(descriptor->name) : IREE_SV("unknown");
}

static inline bool loom_symbol_definition_implements(
    const loom_symbol_definition_descriptor_t* descriptor,
    loom_symbol_interface_flags_t interfaces) {
  return descriptor && interfaces &&
         iree_any_bit_set(descriptor->interfaces, interfaces);
}

static inline bool loom_symbol_definition_satisfies(
    const loom_symbol_definition_descriptor_t* descriptor,
    loom_symbol_interface_flags_t required_interfaces) {
  return descriptor &&
         iree_all_bits_set(descriptor->interfaces, required_interfaces);
}

static inline bool loom_symbol_definition_is_declaration(
    const loom_symbol_definition_descriptor_t* descriptor) {
  return descriptor &&
         iree_any_bit_set(descriptor->flags,
                          LOOM_SYMBOL_DEFINITION_FLAG_DECLARATION);
}

static inline bool loom_symbol_definition_is_test_only(
    const loom_symbol_definition_descriptor_t* descriptor) {
  return descriptor && iree_any_bit_set(descriptor->flags,
                                        LOOM_SYMBOL_DEFINITION_FLAG_TEST_ONLY);
}

static inline uint8_t loom_symbol_definition_visibility_attr_index(
    const loom_symbol_definition_descriptor_t* descriptor) {
  return descriptor && descriptor->visibility_attr_index_plus_one
             ? descriptor->visibility_attr_index_plus_one - 1
             : LOOM_ATTR_INDEX_NONE;
}

static inline bool loom_symbol_definition_has_value_contract(
    const loom_symbol_definition_descriptor_t* descriptor) {
  return descriptor && descriptor->value_contract_result_index_plus_one != 0;
}

static inline uint8_t loom_symbol_definition_product_carrier_attr_index(
    const loom_symbol_definition_descriptor_t* descriptor) {
  return descriptor && descriptor->product_carrier_attr_index_plus_one
             ? descriptor->product_carrier_attr_index_plus_one - 1
             : LOOM_ATTR_INDEX_NONE;
}

// Projects the product carrier of one symbol-defining operation without
// inspecting its body. An absent optional carrier attribute is enum value 0.
static inline loom_symbol_product_carrier_t
loom_symbol_definition_product_carrier(
    const loom_symbol_definition_descriptor_t* descriptor,
    const loom_op_t* defining_op) {
  const uint8_t attr_index =
      loom_symbol_definition_product_carrier_attr_index(descriptor);
  if (defining_op == NULL || attr_index == LOOM_ATTR_INDEX_NONE) {
    return LOOM_SYMBOL_PRODUCT_CARRIER_UNCLASSIFIED;
  }
  const loom_attribute_t carrier_attr =
      loom_op_const_attrs(defining_op)[attr_index];
  return loom_attr_is_absent(carrier_attr) ? 0
                                           : loom_attr_as_enum(carrier_attr);
}

static inline uint8_t loom_symbol_definition_value_contract_result_index(
    const loom_symbol_definition_descriptor_t* descriptor) {
  return descriptor && descriptor->value_contract_result_index_plus_one
             ? descriptor->value_contract_result_index_plus_one - 1
             : LOOM_RESULT_INDEX_NONE;
}

static inline uint8_t loom_symbol_definition_value_contract_value_attr_index(
    const loom_symbol_definition_descriptor_t* descriptor) {
  return descriptor && descriptor->value_contract_value_attr_index_plus_one
             ? descriptor->value_contract_value_attr_index_plus_one - 1
             : LOOM_ATTR_INDEX_NONE;
}

static inline uint8_t
loom_symbol_definition_value_contract_predicates_attr_index(
    const loom_symbol_definition_descriptor_t* descriptor) {
  return descriptor && descriptor->value_contract_predicates_attr_index_plus_one
             ? descriptor->value_contract_predicates_attr_index_plus_one - 1
             : LOOM_ATTR_INDEX_NONE;
}

static inline bool loom_symbol_implements(
    const loom_symbol_t* symbol, loom_symbol_interface_flags_t interfaces) {
  return symbol &&
         loom_symbol_definition_implements(symbol->definition, interfaces);
}

static inline loom_symbol_kind_t loom_symbol_bytecode_kind(
    const loom_symbol_t* symbol) {
  if (!symbol) {
    return LOOM_SYMBOL_NONE;
  }
  return symbol->definition ? symbol->definition->bytecode_kind : symbol->kind;
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

// Returns the module-local symbol ID defined by |op| using its already-resolved
// |vtable| metadata. Returns LOOM_SYMBOL_ID_INVALID for non-symbol ops,
// cross-module refs, or refs outside the module symbol table.
loom_symbol_id_t loom_op_defining_symbol_id(const loom_module_t* module,
                                            const loom_op_t* op,
                                            const loom_op_vtable_t* vtable);

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

// Resolves an author-facing result field to its flat result span. For
// variadic result fields the returned span covers the trailing variadic tail.
loom_value_slice_t loom_op_result_field_span(const loom_op_vtable_t* vtable,
                                             const loom_op_t* op,
                                             uint8_t field_index);

// Maps a flat operand index back to the operand descriptor that owns it.
// Returns false if the index is out of range or the op kind has no descriptor
// metadata.
bool loom_op_operand_descriptor_at(
    const loom_op_vtable_t* vtable, const loom_op_t* op, uint16_t operand_index,
    const loom_operand_descriptor_t** out_descriptor, uint8_t* out_field_index,
    uint16_t* out_element_index);

// Returns the semantic role for a flat operand index, or
// LOOM_OPERAND_ROLE_NONE when the op has no descriptor metadata or the operand
// has no declared role.
loom_operand_role_t loom_op_operand_role_at(const loom_op_vtable_t* vtable,
                                            const loom_op_t* op,
                                            uint16_t operand_index);

// Returns the semantic role for a flat operand index using the op's module
// vtable, or LOOM_OPERAND_ROLE_NONE when unavailable.
loom_operand_role_t loom_op_operand_role(const loom_module_t* module,
                                         const loom_op_t* op,
                                         uint16_t operand_index);

// Returns true when the op operand at |operand_index| has |role|.
bool loom_op_operand_has_role(const loom_module_t* module, const loom_op_t* op,
                              uint16_t operand_index, loom_operand_role_t role);

// Returns the first operand value with |role|, if present.
bool loom_op_first_operand_with_role(const loom_module_t* module,
                                     const loom_op_t* op,
                                     loom_operand_role_t role,
                                     loom_value_id_t* out_value_id);

// Returns true when |op| defines |value_id| as one of its results.
bool loom_op_defines_value(const loom_op_t* op, loom_value_id_t value_id);

// Binding kind for BindingList format elements.
typedef enum loom_binding_kind_e {
  // Block arg has the same type as the operand.
  LOOM_BINDING_CAPTURE = 0,
  // Block arg has the element type of the operand.
  LOOM_BINDING_ELEMENT = 1,
} loom_binding_kind_t;

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

// Returns true if every result of |op| has zero operand uses and no attribute
// or value type references from outside |op|. References carried by |op|'s
// own attributes or result types do not keep the whole op alive.
bool loom_op_results_unused(const loom_module_t* module, const loom_op_t* op);

// Returns true if |op| is trivially dead: it has results, does not
// write to any resource, has no unknown effects, and every result is
// unused. Read-only and non-deterministic ops without writes are dead
// when unused — a read with no observer is a no-op.
bool loom_op_is_trivially_dead(const loom_module_t* module,
                               const loom_op_t* op);

//===----------------------------------------------------------------------===//
// CallLike interface helpers
//===----------------------------------------------------------------------===//

// Returns true when source expansion must retain |op|'s control-flow context.
// This includes CONTEXTUAL operations and semantic calls whose callsite or
// callee requires inlining. It reads the direct symbol's declared policy, never
// its body. Runtime calls and Low calls do not require source context merely
// because their arguments may later become more precise. This classification
// does not prevent erasing unused pure work or expanding a selected body.
bool loom_op_requires_source_context(const loom_module_t* module,
                                     const loom_op_t* op);

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

// Casts const |op| to loom_call_like_t if it implements the CallLike
// interface. The returned interface view does not encode transitive constness;
// callers must preserve the access discipline of the input operation.
static inline loom_call_like_t loom_call_like_const_cast(
    const loom_module_t* module, const loom_op_t* op) {
  return loom_call_like_cast(module, (loom_op_t*)op);
}

// Returns true when |call| is an ordinary semantic call whose callable
// operands and results occupy the complete flat operation boundary. Such calls
// can be lowered without dialect-specific prefix or region semantics.
bool loom_call_like_is_direct_semantic(loom_call_like_t call);

// Returns the direct callee symbol ref, or {0, 0} if |call| is not valid.
loom_symbol_ref_t loom_call_like_callee(loom_call_like_t call);

// Retargets a verified direct call to |callee|.
//
// The call and symbol reference must belong to |module|. This refreshes
// callback-backed effective traits and transitive effect summaries;
// callers remain responsible for invalidating any higher-level analyses.
void loom_call_like_set_callee(loom_module_t* module, loom_call_like_t call,
                               loom_symbol_ref_t callee);

// Returns the trailing call argument slice, or an empty slice if |call| is not
// valid or the recorded field is malformed for the op instance.
loom_value_slice_t loom_call_like_operands(loom_call_like_t call);

// Returns the trailing call result slice, or an empty slice if |call| is not
// valid or the recorded offset is malformed for the op instance.
loom_value_slice_t loom_call_like_results(loom_call_like_t call);

// Resolves the flat operand offset where call arguments begin.
uint16_t loom_call_like_operand_offset(loom_call_like_t call);

// Returns the result offset where call results begin.
uint16_t loom_call_like_result_offset(loom_call_like_t call);

// Returns the purity attr value (0 = unspecified, nonzero = pure).
uint8_t loom_call_like_purity(loom_call_like_t call);

// Returns the temperature attr value (0 = unspecified).
uint8_t loom_call_like_temperature(loom_call_like_t call);

// Returns the authored inline policy.
loom_inline_policy_t loom_call_like_inline_policy(loom_call_like_t call);

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

// Casts const |op| to loom_func_like_t if it implements the FuncLike
// interface. Returns {NULL, NULL} if |op| is NULL or does not implement it.
// The returned interface view does not encode transitive constness; callers
// must preserve the access discipline of the input operation.
static inline loom_func_like_t loom_func_like_const_cast(
    const loom_module_t* module, const loom_op_t* op) {
  return loom_func_like_cast(module, (loom_op_t*)op);
}

// Returns the body region of a func-like op, or NULL for bodyless ops
// (func.decl, template.ukernel) or if |func| is not valid.
loom_region_t* loom_func_like_body(loom_func_like_t func);

// Returns the structural descriptor for the body region, or NULL for bodyless
// ops or invalid func-like references.
const loom_region_descriptor_t* loom_func_like_body_region_descriptor(
    const loom_module_t* module, loom_func_like_t func);

// Returns true when |op| is the declared terminator kind in a block directly
// owned by |func|'s body. A callable exit's complete operand list is the
// function result tuple. A same-kind terminator in a nested region is not a
// callable exit.
bool loom_func_like_op_is_body_exit(const loom_module_t* module,
                                    loom_func_like_t func, const loom_op_t* op);

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

// Returns the authored inline policy.
loom_inline_policy_t loom_func_like_inline_policy(loom_func_like_t func);

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

// Retargets a verified target-assignable function to |target|.
//
// The function and symbol reference must belong to |module|. This refreshes
// callback-backed effective traits and transitive effect summaries; callers
// remain responsible for invalidating any higher-level analyses.
void loom_func_like_set_target(loom_module_t* module, loom_func_like_t func,
                               loom_symbol_ref_t target);

// Sets whether |func| survives symbol pruning as a module-boundary root.
//
// The function must define a symbol with a retain attribute. This updates both
// the operation contract and the module's maintained symbol flags.
void loom_func_like_set_retained(loom_module_t* module, loom_func_like_t func,
                                 bool retained);

// Returns the authored representation-contract key for a func-like op, or
// LOOM_STRING_ID_INVALID when none is present.
loom_string_id_t loom_func_like_repr_contract(loom_func_like_t func);

// Returns the target ABI enum value, or 0 if |func| has no explicit ABI.
uint8_t loom_func_like_abi(loom_func_like_t func);

// Returns the target ABI payload attrs, or an empty slice if absent.
loom_named_attr_slice_t loom_func_like_abi_attrs(loom_func_like_t func);

// Returns the artifact export name override, or LOOM_STRING_ID_INVALID to use
// the source symbol name. An override does not change visibility or entry
// roles.
loom_string_id_t loom_func_like_export_symbol(loom_func_like_t func);

// Returns the export payload attrs, or an empty slice if absent.
loom_named_attr_slice_t loom_func_like_export_attrs(loom_func_like_t func);

// Returns true when |func| is a source-level or target-low kernel entry.
// Kernel entries are exported by symbol name even without an explicit export
// symbol attribute.
bool loom_func_like_is_kernel_entry(loom_func_like_t func);

// Returns true for dispatchable kernels and entries, including declarations.
bool loom_func_like_is_kernel(loom_func_like_t func);

// Returns true for kernels and public functions that are not imports.
// Export name and payload attributes do not make a private function public.
bool loom_func_like_is_exported(loom_func_like_t func);

// Returns true when all possible callers and references to |func| are owned by
// the current module. Imports, public functions, and kernels are
// externally reachable.
bool loom_func_like_is_module_internal(loom_func_like_t func);

// Returns true and assigns the export linkage enum value when present.
bool loom_func_like_export_linkage(loom_func_like_t func, uint8_t* out_linkage);

// Returns the function argument value IDs and their count. For ops
// with a body region, args are the entry block's block arguments.
// For declaration-style ops, args are stored as the op's operands.
// Returns NULL and sets |out_count| to 0 if |func| is not valid.
const loom_value_id_t* loom_func_like_arg_ids(loom_func_like_t func,
                                              uint16_t* out_count);

// Returns the workload signature value IDs for a kernel definition or
// declaration. Definitions source the signature from their configuration
// region and declarations source it from their designated operand field.
// Returns an empty slice for non-kernel symbols.
loom_value_slice_t loom_kernel_workload_arg_ids(const loom_module_t* module,
                                                const loom_op_t* op);

// Returns the predicate list and count for a func-like op. Sets |out_count|
// to 0 and returns NULL for ops with no predicate list attr or if |func| is
// not valid.
const loom_predicate_t* loom_func_like_predicates(loom_func_like_t func,
                                                  uint16_t* out_count);

// Returns the authored proof requirements for a provider function. Returns an
// empty slice for non-provider function kinds and providers without
// requirements.
loom_parameterized_attr_array_t loom_func_like_requires(loom_func_like_t func);

// Returns the number of leading function arguments consumed while
// materializing the function-like artifact. Returns zero when the function
// has no distinct specialization arguments or |func| is invalid.
int64_t loom_func_like_specialization_count(loom_func_like_t func);

// Returns the template family implemented by |func|, or an invalid reference
// when the function-like symbol is not a template provider.
loom_symbol_ref_t loom_func_like_template_family(loom_func_like_t func);

// Returns the dispatch priority for concrete providers. Returns 0 for ops with
// no priority attr or if |func| is not valid.
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

// Per-instance execution-semantics modifiers for memory-access operations.
enum loom_memory_access_flag_bits_e {
  // Requires every dynamic access to execute as an independently observable
  // operation and preserves its order relative to other volatile accesses. A
  // target may decompose one logical access into multiple physical accesses.
  // This does not provide atomicity, synchronization, or a cache-coherence
  // guarantee.
  LOOM_MEMORY_ACCESS_FLAG_VOLATILE = 1u << 0,
  // Requires floating-point atomic addition to preserve subnormal inputs and
  // results. Without this flag the target may flush subnormals. Selection
  // consumes this numerical requirement; it is not a physical Low access flag.
  LOOM_MEMORY_ACCESS_FLAG_NOFTZ = 1u << 1,
};
typedef uint8_t loom_memory_access_flags_t;

// Derives effective traits for a memory-access operation from its instance
// flags. Volatile accesses carry ObservableEffect in addition to their static
// read or write effects.
loom_trait_flags_t loom_memory_access_effective_traits(const loom_op_t* op);

// Returns true if |access| refers to a valid memory-access op. All accessor
// helpers below tolerate a NULL op vtable and return safe defaults.
static inline bool loom_memory_access_isa(loom_memory_access_t access) {
  return access.op != NULL && access.op_vtable != NULL &&
         access.op_vtable->memory_access != NULL;
}

// Casts |op| to loom_memory_access_t if it implements the MemoryAccess
// interface. Returns {NULL, NULL} if |op| is NULL or does not implement it.
// Safe to call unconditionally; callers check the result with
// loom_memory_access_isa().
loom_memory_access_t loom_memory_access_cast(const loom_module_t* module,
                                             const loom_op_t* op);

// Returns the memory operation family represented by the op shape.
loom_memory_access_operation_kind_t loom_memory_access_operation_kind(
    loom_memory_access_t access);

// Returns the per-access execution semantics. MemoryAccess operations use the
// shared memory-access vocabulary for their instance flags; the op generator
// enforces that contract even for operations without optional flags.
loom_memory_access_flags_t loom_memory_access_flags(
    loom_memory_access_t access);

// Returns true when the operand at |operand_index| is a written value,
// compare-exchange expected value, or compare-exchange replacement value.
bool loom_memory_access_operand_index_is_payload(loom_memory_access_t access,
                                                 uint16_t operand_index);

// Returns the accessed view or memory-object operand.
loom_value_id_t loom_memory_access_view(loom_memory_access_t access);

// Returns the physical byte-offset operand, or INVALID for logical accesses.
loom_value_id_t loom_memory_access_byte_offset(loom_memory_access_t access);

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

// Callback invoked after an op's direct fields are finalized. Region-owning op
// builders create their regions before invoking this callback, but callers may
// populate those regions afterward. Used by the rewriter to add newly created
// ops to its worklist.
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
// The optional on_op_finalized callback fires after finalize_op completes
// (uses registered, def pointers set). Nested regions may still be empty at
// this point. NULL when unused.
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
// Callable builders consume argument signature identities before result
// identities. Other region entry arguments are fresh, independent definitions.
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
// names attribute so optional OperandDict fields print nothing. Input entries
// are borrowed and may appear in any order; duplicate keys are rejected. The
// module owns the final name entries, with no retained sorting scratch. Sorting
// takes linear comparisons for ordered input and O(n log n) in the worst case.
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

// Copies an enum-array attribute payload into the builder arena.
iree_status_t loom_builder_copy_enum_array_attr_storage(
    loom_builder_t* builder, loom_enum_array_t values, iree_string_view_t label,
    const uint8_t** out_storage);

// Validates, canonicalizes, and copies a signed enum-set attribute payload into
// the builder arena.
iree_status_t loom_builder_copy_signed_enum_set_attr_storage(
    loom_builder_t* builder, loom_signed_enum_set_t set,
    iree_string_view_t label, const uint64_t** out_storage,
    uint16_t* out_word_count);

// Copies a symbol-array attribute payload into the builder arena.
iree_status_t loom_builder_copy_symbol_array_attr_storage(
    loom_builder_t* builder, loom_symbol_ref_array_t values,
    iree_string_view_t label, const loom_symbol_ref_t** out_storage);

// Copies a predicate-list attribute payload into the builder arena.
iree_status_t loom_builder_copy_predicate_list_attr_storage(
    loom_builder_t* builder, const loom_predicate_t* predicates,
    iree_host_size_t count, iree_string_view_t label,
    loom_predicate_t** out_storage);

// Copies a byte attribute payload into the builder arena.
iree_status_t loom_builder_copy_bytes_attr_storage(loom_builder_t* builder,
                                                   iree_const_byte_span_t bytes,
                                                   iree_string_view_t label,
                                                   const uint8_t** out_storage);

// Creates a one-block region and installs it into |op| at |region_index|.
iree_status_t loom_builder_create_region(loom_builder_t* builder, loom_op_t* op,
                                         uint8_t region_index,
                                         loom_block_t** out_entry_block);

// Creates a fresh value with the given type and adds it as a block argument.
// Region arguments never consume identities reserved for the enclosing op's
// results, even when generated builders create regions before those results.
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
// have no operand, attribute, or incoming type uses. Dropped values remain in
// the module value table but no longer carry defining-op identity or outgoing
// type-use records. Kept result values keep their IDs and receive updated
// definition indices. Tied results targeting removed result slots are rejected;
// kept tied result indices are remapped.
iree_status_t loom_op_remove_results(loom_module_t* module, loom_op_t* op,
                                     const bool* remove_results,
                                     iree_arena_allocator_t* scratch_arena,
                                     uint16_t* out_removed_count);

// Erases an op after verifying that every result has no operand uses or
// attribute/type uses from outside the op (caller must RAUW results first).
// Removes operand and attribute use records, drops type-use records carried by
// results, owned declaration arguments and nested block arguments, then marks
// the op dead. Result values retain their defining op and result index as
// immutable producer provenance. Dead ops are skipped by enumeration macros
// and will not be serialized. The memory is not freed (arena-owned). Returns
// IREE_STATUS_FAILED_PRECONDITION if any result still has uses.
iree_status_t loom_op_erase(loom_module_t* module, loom_op_t* op);

// Removes a closed set of non-entry blocks from |region| and compacts the
// region block table in place.
//
// |remove_blocks| must contain exactly |remove_block_count| entries, one per
// current block index in |region|. Entry block removal is rejected. Any kept op
// successor targeting a removed block is rejected. Values defined by removed
// block arguments or removed op subtrees may only have operand, type and
// attribute uses inside the removed set; callers must retarget or replace
// external uses before removing blocks. Closure and scratch-allocation failures
// leave IR unchanged and |out_removed_count| zero.
//
// Uses O(B) space in |scratch_arena| to index B removed nested blocks; flat
// block sets allocate no scratch. Validation is O(S + (B + E) log(B + 2)) for
// S inspected IR nodes/value slots and E incoming references/successors.
//
// Removed block/op/value objects remain arena-owned for diagnostics, but the
// blocks are detached from the region, their operations are marked dead, and
// block-argument identity/type-use records are dropped.
iree_status_t loom_region_remove_blocks(loom_module_t* module,
                                        loom_region_t* region,
                                        const bool* remove_blocks,
                                        uint16_t remove_block_count,
                                        iree_arena_allocator_t* scratch_arena,
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
// |operand_index|. Reads the operand's retained use index, swaps with the last
// entry, and updates the moved operand's index in O(1). Returns
// IREE_STATUS_NOT_FOUND if the index does not name the matching entry.
// No overflow-to-inline transition (arena cannot free the overflow
// array; loom_module_compute_uses handles repack).
iree_status_t loom_value_remove_use(loom_module_t* module,
                                    loom_value_id_t value_id,
                                    loom_op_t* user_op, uint16_t operand_index);

// Finalizes a newly-built op: registers all operand uses and performs
// any other per-op bookkeeping. Called as the tail return from every
// builder: `return loom_builder_finalize_op(builder, *out_op);`
iree_status_t loom_builder_finalize_op(loom_builder_t* builder, loom_op_t* op);

// Replaces an attribute on a constructed operation, maintaining exact SSA
// attribute-use records, effective traits and direct semantic summaries.
// Raw attribute writes are only valid before finalization or when preserving
// the exact SSA reference set (for example remapping symbol IDs).
iree_status_t loom_op_set_attr(loom_module_t* module, loom_op_t* op,
                               uint8_t attribute_index,
                               loom_attribute_t attribute);

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
// attributes with one shared immutable substitution context. Each embedded
// reference owner publishes consistently with its index, but an allocation
// failure can leave earlier owners changed. Ordinary operands are transferred
// only after all embedded references succeed. No-op if old_id == new_id.
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
