// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Declarative operation text formats and compact assembly aliases.

#ifndef LOOM_OPS_FORMAT_H_
#define LOOM_OPS_FORMAT_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/util/bstring.h"

#ifdef __cplusplus
extern "C" {
#endif

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
  // Result type list with tied handling: (type, %operand as type).
  // Symbol results may prefix either form with a local %result: binder.
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
  // data = optional start/end i64 attribute indices packed by
  // LOOM_FORMAT_FUNC_ARGS_DATA. The boundaries project a contiguous slice
  // from a body-backed function signature.
  LOOM_FORMAT_KIND_FUNC_ARGS = 13,
  // Where-clause predicates: [multiple_of(%M, 16), ...].
  LOOM_FORMAT_KIND_PREDICATE_LIST = 14,
  // Optional group marker. field_index = anchor field index.
  // data = LOOM_FORMAT_OPTIONAL_GROUP_DATA. The walker skips |skip_count|
  // elements when the anchor presence does not match the group polarity.
  LOOM_FORMAT_KIND_OPTIONAL_GROUP = 15,
  // Suppress space before the next token.
  LOOM_FORMAT_KIND_GLUE = 16,
  // Per-instance flags in angle brackets: <flag1|flag2>.
  // Glued to the preceding token (op name). Uses the vtable's instance_flags
  // descriptor and reads/writes op->instance_flags. field_index is unused.
  LOOM_FORMAT_KIND_FLAGS = 17,

  // Bare symbolic key in angle brackets: <tile.contract>.
  // Glued to the preceding token. The field_index references a string
  // attribute storing the canonical key spelling.
  LOOM_FORMAT_KIND_KEY_REF = 18,

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
  // data optionally packs attribute indices with LOOM_FORMAT_BLOCK_ARGS_DATA.
  // The boundaries project a contiguous slice of one entry signature.
  LOOM_FORMAT_KIND_BLOCK_ARGS = 26,

  // CFG successor block reference: ^label.
  // field_index = successor index whose target block is printed or parsed.
  LOOM_FORMAT_KIND_SUCCESSOR_REF = 27,

  // Representation-scoped enum in angle brackets: <amdgpu.v_add_u32>. The
  // field_index references one SCOPED_ENUM attribute. Text parsing resolves the
  // stable spelling to a dense ordinal in the enclosing function contract.
  LOOM_FORMAT_KIND_SCOPED_ENUM_REF = 28,

  // Stable symbolic key reference in angle brackets: <source.key>. The
  // field_index references the diagnostic string attribute and data references
  // the derived i64 stable-key attribute. This is for non-descriptor symbolic
  // domains; descriptor-backed packets use LOOM_FORMAT_KIND_SCOPED_ENUM_REF.
  LOOM_FORMAT_KIND_STABLE_KEY_REF = 29,

  // Variadic operand references with adjacent type annotations:
  // %a: type, %b: type.
  LOOM_FORMAT_KIND_OPERAND_TYPED_REFS = 30,

  // Known-family parameterized attribute payload in angle brackets:
  // <mode = fast, scopes = [workgroup]>. The field_index references a
  // PARAMETERIZED attribute constrained to one exact family. The family name
  // is carried by the descriptor and omitted from text.
  LOOM_FORMAT_KIND_ATTR_PARAMS = 31,

  // Variadic byte-length operands paired with static alignments:
  // [align(16) %a, align(256) %b]. field_index references the variadic
  // operand field and data references its i64 array alignment attribute.
  LOOM_FORMAT_KIND_ALIGNED_REFS = 32,

  // Structured operands: (%value, [(%value, %value)], none).
  // field_index is the variadic operand field. data is the i64-array shape
  // attribute governed by the operation's OperandTree semantic constraint.
  LOOM_FORMAT_KIND_TREE_REFS = 33,
};
typedef uint8_t loom_format_kind_t;

// The high bit distinguishes groups selected by an absent anchor. Remaining
// bits hold the child count and two-bit anchor category.
#define LOOM_FORMAT_OPTIONAL_GROUP_INVERTED ((uint16_t)(1u << 15))
#define LOOM_FORMAT_OPTIONAL_GROUP_SKIP_MASK \
  ((uint16_t)~LOOM_FORMAT_OPTIONAL_GROUP_INVERTED)
#define LOOM_FORMAT_OPTIONAL_GROUP_DATA(skip_count, anchor_category, inverted) \
  ((uint16_t)((((uint16_t)(skip_count) << 2) &                                 \
               LOOM_FORMAT_OPTIONAL_GROUP_SKIP_MASK) |                         \
              (uint16_t)(anchor_category) |                                    \
              ((inverted) ? LOOM_FORMAT_OPTIONAL_GROUP_INVERTED : 0u)))
#define LOOM_FORMAT_OPTIONAL_GROUP_SKIP_COUNT(data) \
  ((uint16_t)(((data) & LOOM_FORMAT_OPTIONAL_GROUP_SKIP_MASK) >> 2))
#define LOOM_FORMAT_OPTIONAL_GROUP_IS_INVERTED(data) \
  iree_any_bit_set((data), LOOM_FORMAT_OPTIONAL_GROUP_INVERTED)

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

// Packs optional start/end attribute indices for a FUNC_ARGS element. Each
// byte stores index + 1 so zero remains the absent sentinel.
#define LOOM_FORMAT_FUNC_ARGS_ATTR_BYTE(attr_index) \
  ((uint16_t)((attr_index) == LOOM_ATTR_INDEX_NONE  \
                  ? 0u                              \
                  : ((uint16_t)(attr_index) + 1u)))
#define LOOM_FORMAT_FUNC_ARGS_DATA(start_attr_index, end_attr_index)     \
  ((uint16_t)((LOOM_FORMAT_FUNC_ARGS_ATTR_BYTE(start_attr_index) << 8) | \
              LOOM_FORMAT_FUNC_ARGS_ATTR_BYTE(end_attr_index)))
#define LOOM_FORMAT_FUNC_ARGS_START_ATTR_INDEX(data) \
  ((uint8_t)(((data) >> 8) - 1u))
#define LOOM_FORMAT_FUNC_ARGS_END_ATTR_INDEX(data) \
  ((uint8_t)(((data) & 0xFFu) - 1u))

// BLOCK_ARGS uses the same packed signature-slice representation as
// FUNC_ARGS. Separate names keep format consumers explicit about which
// signature owns the boundary attributes.
#define LOOM_FORMAT_BLOCK_ARGS_DATA(start_attr_index, end_attr_index) \
  LOOM_FORMAT_FUNC_ARGS_DATA(start_attr_index, end_attr_index)
#define LOOM_FORMAT_BLOCK_ARGS_START_ATTR_INDEX(data) \
  LOOM_FORMAT_FUNC_ARGS_START_ATTR_INDEX(data)
#define LOOM_FORMAT_BLOCK_ARGS_END_ATTR_INDEX(data) \
  LOOM_FORMAT_FUNC_ARGS_END_ATTR_INDEX(data)

// Surface syntax selected by a REGION format element. This affects only text
// parsing/printing; the in-memory representation is always an ordinary
// loom_region_t.
typedef enum loom_region_syntax_e {
  // Canonical braced region: { block+ }.
  LOOM_REGION_SYNTAX_DEFAULT = 0,
  // Test-only alternate region syntax: do { block+ }.
  LOOM_REGION_SYNTAX_TEST_DO = 1,
  // Canonical braced region by default, with optional target-low asm syntax.
  LOOM_REGION_SYNTAX_LOW_ASM_OPTIONAL = 2,
  // Pass pipeline syntax. Currently canonical braced form; the friendly
  // parser/printer selects the same in-memory pass.* operations.
  LOOM_REGION_SYNTAX_PIPELINE = 3,
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
//   BINDING_LIST:   loom_binding_list_flag_bits_e.
//   FUNC_ARGS:      packed optional start/end i64 attribute indices.
//   OPTIONAL_GROUP: (skip_count << 2) | anchor_category.
typedef struct loom_format_element_t {
  // Interpreter instruction kind.
  loom_format_kind_t kind;
  // Declaration-owned field ordinal, interpreted according to kind.
  uint8_t field_index;
  // Kind-specific packed payload described above.
  uint16_t data;
} loom_format_element_t;

static_assert(sizeof(loom_format_element_t) == 4,
              "loom_format_element_t must be exactly 4 bytes");

// Data field flags for BINDING_LIST elements. Stored in the element's data
// field.
enum loom_binding_list_flag_bits_e {
  // Block arguments have the same types as their captured operands.
  LOOM_BINDING_LIST_CAPTURE = 0u,
  // Block arguments have the element types of their captured operands.
  LOOM_BINDING_LIST_ELEMENT = 1u << 0,
  // Type annotations describe block arguments and may reference peer block
  // arguments. Otherwise annotations describe the captured operands.
  LOOM_BINDING_LIST_ANNOTATES_BLOCK_ARGS = 1u << 1,
};

// Data field flags for RESULT_TYPE_LIST elements. Stored in the
// element's data field.
enum loom_result_type_list_flag_bits_e {
  // Wrap result types in parentheses: (type, type).
  // When clear, result types are bare: type, type.
  LOOM_RESULT_TYPE_LIST_PARENS = 1u << 0,
  // Print and parse one type shared by every result.
  LOOM_RESULT_TYPE_LIST_UNIFORM = 1u << 1,
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
// reference keywords by ID. Generated from KEYWORD_MAP in
// loom.gen.assembly.tokens; new keywords append to that map.
typedef enum loom_keyword_id_e {
#include "loom/ops/keyword_enum.inc"
  LOOM_KW_COUNT_,
} loom_keyword_id_t;

// Returns the B-string for |keyword_id|, or NULL if |keyword_id| is out of
// range. E.g., LOOM_KW_TO -> "\x02to", LOOM_KW_STEP -> "\x04step".
loom_bstring_t loom_keyword_bstring(loom_keyword_id_t keyword_id);

// Borrowed instruction stream selected for one parse or print operation.
typedef struct loom_format_t {
  // Static generated instructions, or NULL for an empty format.
  const loom_format_element_t* elements;
  // Number of instructions in the selected stream.
  uint16_t count;
} loom_format_t;

// Alternate spelling of an operation inside a target assembly region. Field
// indices address the canonical operation layout; builders and IR are shared.
typedef struct loom_op_assembly_format_t {
  // Short source mnemonic, or NULL to retain the canonical operation name.
  loom_bstring_t name;
  // Alternate instructions when has_custom_format is true.
  const loom_format_element_t* elements;
  // Canonical operation kind constructed by the ordinary parser.
  loom_op_kind_t kind;
  // Number of alternate instructions, including zero for an empty grammar.
  uint16_t element_count;
  // False reuses the canonical operation format without duplicating its data.
  bool has_custom_format;
} loom_op_assembly_format_t;

// Generated dialect-local assembly vocabulary. Rows are ordered by mnemonic;
// printing uses the dense op-index map and never searches source names.
typedef struct loom_op_assembly_format_table_t {
  // Static rows ordered lexicographically by mnemonic.
  const loom_op_assembly_format_t* entries;
  // One row index per registered dialect op; UINT8_MAX means canonical syntax.
  const uint8_t* op_indices;
  // Number of rows in entries.
  uint16_t entry_count;
  // Dialect owning the op-index map.
  uint8_t dialect_id;
} loom_op_assembly_format_table_t;

// Looks up an authored short mnemonic at the text parsing boundary.
const loom_op_assembly_format_t* loom_op_assembly_format_lookup_name(
    const loom_op_assembly_format_table_t* table, iree_string_view_t name);

// Returns an assembly spelling for a registered operation kind, or NULL for
// canonical syntax. The kind must belong to a registered dialect operation.
const loom_op_assembly_format_t* loom_op_assembly_format_lookup_kind(
    const loom_op_assembly_format_table_t* table, loom_op_kind_t kind);

// Selects the instruction stream once, before ordinary format interpretation.
static inline loom_format_t loom_op_format(
    const loom_op_vtable_t* vtable, const loom_op_assembly_format_t* assembly) {
  if (assembly && assembly->has_custom_format) {
    return (loom_format_t){assembly->elements, assembly->element_count};
  }
  return (loom_format_t){vtable->format_elements, vtable->format_element_count};
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_FORMAT_H_
