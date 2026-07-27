// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Loom IR attribute values and symbol/predicate payloads.
//
// Attributes are 16-byte tagged values used in op trailing storage, encoding
// parameters, and any API that needs a small typed scalar/aggregate payload.
// Pointer-valued attributes reference arena-owned immutable storage. DICT
// attributes have an additional canonicalization contract: non-empty entries
// are sorted by key spelling, duplicate-free, and recursively canonical.

#ifndef LOOM_IR_ATTRIBUTE_H_
#define LOOM_IR_ATTRIBUTE_H_

#include "iree/base/api.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Symbol references
//===----------------------------------------------------------------------===//

#define LOOM_SYMBOL_ID_INVALID ((uint16_t)UINT16_MAX)
#define LOOM_MODULE_ID_INVALID ((uint16_t)UINT16_MAX)

// Identifies a module-local symbol.
//
// Valid in-memory refs use module_id = 0 and symbol_id indexes into the
// current module's symbol table. LOOM_MODULE_ID_INVALID is used only for the
// null sentinel returned by loom_symbol_ref_null(). The module_id field keeps
// the payload ABI aligned with serialized/linker symbol refs that may carry a
// non-local module selector.
typedef struct loom_symbol_ref_t {
  uint16_t module_id;
  uint16_t symbol_id;
} loom_symbol_ref_t;

// Returns true if |ref| is not the null sentinel. Module-locality and
// symbol-id range are separate verifier invariants.
static inline bool loom_symbol_ref_is_valid(loom_symbol_ref_t ref) {
  return ref.symbol_id != LOOM_SYMBOL_ID_INVALID;
}

static inline loom_symbol_ref_t loom_symbol_ref_null(void) {
  loom_symbol_ref_t ref = {LOOM_MODULE_ID_INVALID, LOOM_SYMBOL_ID_INVALID};
  return ref;
}

//===----------------------------------------------------------------------===//
// Predicates
//===----------------------------------------------------------------------===//

// Predicate kind byte. Evaluated at runtime for dynamic dimensions and stored
// in loom_predicate_t::kind.
typedef uint8_t loom_predicate_kind_t;

enum loom_predicate_kind_e {
  // eq(a, b)
  LOOM_PREDICATE_EQ = 0,
  // ne(a, b)
  LOOM_PREDICATE_NE = 1,
  // lt(a, b)
  LOOM_PREDICATE_LT = 2,
  // le(a, b)
  LOOM_PREDICATE_LE = 3,
  // gt(a, b)
  LOOM_PREDICATE_GT = 4,
  // ge(a, b)
  LOOM_PREDICATE_GE = 5,
  // mul(a, n): a is multiple of n.
  LOOM_PREDICATE_MUL = 6,
  // min(a, n): a >= n.
  LOOM_PREDICATE_MIN = 7,
  // max(a, n): a <= n.
  LOOM_PREDICATE_MAX = 8,
  // pow2(a): a is power of 2.
  LOOM_PREDICATE_POW2 = 9,
  // range(a, lo, hi): lo <= a <= hi.
  LOOM_PREDICATE_RANGE = 10,
  // not_nan(a): a is not NaN.
  LOOM_PREDICATE_NOT_NAN = 11,
  // not_inf(a): a is not positive or negative infinity.
  LOOM_PREDICATE_NOT_INF = 12,
  // finite(a): a is not NaN or infinity.
  LOOM_PREDICATE_FINITE = 13,
  // Number of predicate kinds.
  LOOM_PREDICATE_COUNT_,
};

// Predicate argument tag byte: immediate constant or SSA value reference.
typedef uint8_t loom_predicate_arg_tag_t;

enum loom_predicate_arg_tag_e {
  // Unused slot.
  LOOM_PRED_ARG_NONE = 0,
  // SSA value ID.
  LOOM_PRED_ARG_VALUE = 1,
  // Integer constant.
  LOOM_PRED_ARG_CONST = 2,
  // Number of predicate argument tags.
  LOOM_PRED_ARG_COUNT_,
};

// A single predicate constraint. 32 bytes, arena-allocated.
//
// Predicates constrain dynamic dimension values in where clauses and
// assume ops. Each predicate has a kind (eq, mul, range, etc.)
// and 1-3 arguments. Arguments are tagged: SSA value references or
// integer constants.
typedef struct loom_predicate_t {
  // Predicate kind.
  loom_predicate_kind_t kind;
  // Number of active argument slots.
  uint8_t arg_count;
  // Argument tag per active argument slot.
  loom_predicate_arg_tag_t arg_tags[3];
  // Pad to an 8-byte boundary.
  uint8_t reserved[3];
  // SSA value ID or constant payload per active argument slot.
  int64_t args[3];
} loom_predicate_t;

static_assert(sizeof(loom_predicate_t) == 32,
              "loom_predicate_t must be 32 bytes");

// Returns the canonical text spelling for |kind| or NULL if |kind| is outside
// the predicate vocabulary.
const char* loom_predicate_kind_name(uint8_t kind);

// Returns the exact arity for |kind| or UINT8_MAX if |kind| is outside the
// predicate vocabulary.
uint8_t loom_predicate_kind_argument_count(uint8_t kind);

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

typedef struct loom_named_attr_t loom_named_attr_t;

// A borrowed range of named attributes.
//
// This is the pass-facing view of DICT attribute entries and the input shape
// for canonical dict builders. Keeping the pointer/count pair bundled avoids
// the usual "which order were those parameters in again?" foot-gun at rebuild
// callsites.
typedef struct loom_named_attr_slice_t {
  const loom_named_attr_t* entries;
  iree_host_size_t count;
} loom_named_attr_slice_t;

static inline loom_named_attr_slice_t loom_named_attr_slice_empty(void) {
  loom_named_attr_slice_t slice = {0};
  return slice;
}

static inline loom_named_attr_slice_t loom_make_named_attr_slice(
    const loom_named_attr_t* entries, iree_host_size_t count) {
  loom_named_attr_slice_t slice = {
      /*.entries=*/count > 0 ? entries : NULL,
      /*.count=*/count,
  };
  return slice;
}

// Attribute value kind tag. Determines which union member of
// loom_attribute_t is active.
typedef enum loom_attr_kind_e {
  // All-zero optional-absent sentinel. This is not a concrete payload value in
  // required attributes or DICT entries.
  LOOM_ATTR_ABSENT = 0,
  // 64-bit signed integer.
  LOOM_ATTR_I64 = 1,
  // 64-bit IEEE 754 double.
  LOOM_ATTR_F64 = 2,
  // Interned string reference (loom_string_id_t).
  LOOM_ATTR_STRING = 3,
  // Boolean (payload raw field is 0 or 1).
  LOOM_ATTR_BOOL = 4,
  // Enum case index (payload raw field is the case ordinal).
  LOOM_ATTR_ENUM = 5,
  // Arena-allocated int64_t array (count in header, pointer in payload).
  LOOM_ATTR_I64_ARRAY = 6,
  // Symbol reference (module_id + symbol_id packed in payload).
  LOOM_ATTR_SYMBOL = 7,
  // Type table index (uint32_t in payload).
  LOOM_ATTR_TYPE = 8,
  // Arena-allocated predicate list (count in header, pointer in payload).
  LOOM_ATTR_PREDICATE_LIST = 9,
  // Arena-allocated dictionary of named attributes (count in header,
  // pointer to loom_named_attr_t array in payload). Non-empty dict entries
  // are canonicalized by key spelling, contain no duplicate keys, and nested
  // DICT values recursively follow the same invariant. Used by ops with
  // AttrDict format elements for extensible metadata.
  LOOM_ATTR_DICT = 10,
  // Static encoding table index (1-based uint16_t in payload).
  LOOM_ATTR_ENCODING = 11,
  // Arena-allocated arbitrary byte payload (length in reserved_1, pointer in
  // payload). Used for compact compiler-owned metadata blobs and small
  // executable data payloads. Large externally stored resources should use the
  // bytecode resource table once materialized instead of this inline form.
  LOOM_ATTR_BYTES = 12,
  // Descriptor wildcard for format/parser/verification metadata. This is not a
  // concrete payload kind in loom_attribute_t values.
  LOOM_ATTR_ANY = 13,
  LOOM_ATTR_COUNT_,
} loom_attr_kind_t;

// Maximum supported nesting depth for DICT attribute values. This keeps
// recursive verification, equality, and hashing bounded while still covering
// practical metadata shapes.
#define LOOM_ATTR_DICT_MAX_NESTING_DEPTH 16

// A 16-byte tagged value. Used in operation trailing data, encoding
// parameters, and anywhere a typed scalar/aggregate is needed.
//
// The kind tag (byte 0) determines which union member is active.
// For I64_ARRAY and PREDICATE_LIST, the count field holds the element count and
// the payload holds a pointer to arena-allocated storage. For BYTES, reserved_1
// holds the byte length and the payload holds a pointer to arena-allocated
// immutable bytes.
typedef struct loom_attribute_t {
  uint8_t kind;
  uint8_t reserved_0;
  uint16_t count;
  uint32_t reserved_1;
  union {
    int64_t i64;
    double f64;
    loom_string_id_t string_id;
    loom_symbol_ref_t symbol;
    int64_t* i64_array;
    loom_type_id_t type_id;
    uint32_t encoding_id;
    loom_predicate_t* predicate_list;
    const loom_named_attr_t* dict_entries;
    const uint8_t* bytes;
    uint64_t raw;
  };
} loom_attribute_t;

static_assert(sizeof(loom_attribute_t) == 16,
              "loom_attribute_t must be exactly 16 bytes");

// Constructs the all-zero optional-absent sentinel.
static inline loom_attribute_t loom_attr_absent(void) {
  loom_attribute_t attr = {0};
  return attr;
}

// Initializes a present attribute shell.
static inline loom_attribute_t loom_attr_make_present(
    loom_attr_kind_t attr_kind) {
  loom_attribute_t attr = {
      /*.kind=*/(uint8_t)attr_kind,
  };
  return attr;
}

// Constructs an integer attribute.
static inline loom_attribute_t loom_attr_i64(int64_t value) {
  loom_attribute_t attr = loom_attr_make_present(LOOM_ATTR_I64);
  attr.i64 = value;
  return attr;
}

// Constructs a floating-point attribute.
static inline loom_attribute_t loom_attr_f64(double value) {
  loom_attribute_t attr = loom_attr_make_present(LOOM_ATTR_F64);
  attr.f64 = value;
  return attr;
}

// Constructs a string attribute from an interned string ID.
static inline loom_attribute_t loom_attr_string(loom_string_id_t string_id) {
  loom_attribute_t attr = loom_attr_make_present(LOOM_ATTR_STRING);
  attr.string_id = string_id;
  return attr;
}

// Constructs a boolean attribute.
static inline loom_attribute_t loom_attr_bool(bool value) {
  loom_attribute_t attr = loom_attr_make_present(LOOM_ATTR_BOOL);
  attr.raw = value ? 1 : 0;
  return attr;
}

// Constructs an enum attribute from a case index.
static inline loom_attribute_t loom_attr_enum(uint8_t case_index) {
  loom_attribute_t attr = loom_attr_make_present(LOOM_ATTR_ENUM);
  attr.raw = case_index;
  return attr;
}

// Constructs a symbol reference attribute.
static inline loom_attribute_t loom_attr_symbol(loom_symbol_ref_t ref) {
  loom_attribute_t attr = loom_attr_make_present(LOOM_ATTR_SYMBOL);
  attr.symbol = ref;
  return attr;
}

// Constructs a type reference attribute from a type table index.
static inline loom_attribute_t loom_attr_type(loom_type_id_t type_id) {
  loom_attribute_t attr = loom_attr_make_present(LOOM_ATTR_TYPE);
  attr.type_id = type_id;
  return attr;
}

// Constructs a static encoding attribute from a 1-based module encoding ID.
static inline loom_attribute_t loom_attr_encoding(uint16_t encoding_id) {
  loom_attribute_t attr = loom_attr_make_present(LOOM_ATTR_ENCODING);
  attr.encoding_id = encoding_id;
  return attr;
}

// Constructs a byte payload attribute. The bytes array must be arena-allocated
// and outlive the attribute.
static inline loom_attribute_t loom_attr_bytes(const uint8_t* bytes,
                                               uint32_t byte_length) {
  loom_attribute_t attr = loom_attr_make_present(LOOM_ATTR_BYTES);
  attr.reserved_1 = byte_length;
  attr.bytes = byte_length > 0 ? bytes : NULL;
  return attr;
}

// Constructs an integer array attribute. The values array must be
// arena-allocated and outlive the attribute.
static inline loom_attribute_t loom_attr_i64_array(int64_t* values,
                                                   uint16_t count) {
  loom_attribute_t attr = loom_attr_make_present(LOOM_ATTR_I64_ARRAY);
  attr.count = count;
  attr.i64_array = values;
  return attr;
}

// Constructs a predicate list attribute. The predicates array must be
// arena-allocated and outlive the attribute.
static inline loom_attribute_t loom_attr_predicate_list(
    loom_predicate_t* predicates, uint16_t count) {
  loom_attribute_t attr = loom_attr_make_present(LOOM_ATTR_PREDICATE_LIST);
  attr.count = count;
  attr.predicate_list = predicates;
  return attr;
}

// Constructs a dictionary attribute from an already-canonical entry array.
// This helper is an unchecked representation wrapper: it does not sort,
// dedupe, copy, or verify |entries|. Non-empty entries must already be sorted
// by key spelling, duplicate-free, recursively canonical, and lifetime-stable.
// Use loom_module_make_canonical_attr_dict for normal construction from
// unsorted or temporary input.
static inline loom_attribute_t loom_make_canonical_attr_dict(
    const loom_named_attr_t* entries, iree_host_size_t count) {
  IREE_ASSERT(count <= UINT16_MAX);
  loom_attribute_t attr = loom_attr_make_present(LOOM_ATTR_DICT);
  attr.count = (uint16_t)count;
  attr.dict_entries = count > 0 ? entries : NULL;
  return attr;
}

// Returns true if |attr| is the optional-absent sentinel.
static inline bool loom_attr_is_absent(loom_attribute_t attr) {
  return attr.kind == LOOM_ATTR_ABSENT;
}

// Returns true if |attr| is a valid literal payload for a value whose scalar
// element type is |scalar_type|. If non-null, |out_expected_kind| receives the
// primary expected attribute kind for diagnostics.
bool loom_attr_matches_scalar_type(loom_attribute_t attr,
                                   loom_scalar_type_t scalar_type,
                                   loom_attr_kind_t* out_expected_kind);

// A named attribute: interned name paired with a typed value.
// Used for encoding parameters (block=32, layout="nchw"),
// dictionary attributes on ops, and anywhere else a key-value
// attribute pair is needed.
struct loom_named_attr_t {
  loom_string_id_t name_id;
  uint32_t reserved;
  loom_attribute_t value;
};

// A named attribute update applied to an existing DICT attribute.
//
// When |remove| is true, the key is deleted and |value| is ignored. Otherwise
// the key is inserted or replaced with |value|.
typedef struct loom_named_attr_update_t {
  loom_string_id_t name_id;
  bool remove;
  loom_attribute_t value;
} loom_named_attr_update_t;

// A borrowed range of named attribute updates.
typedef struct loom_named_attr_update_slice_t {
  const loom_named_attr_update_t* updates;
  iree_host_size_t count;
} loom_named_attr_update_slice_t;

static inline loom_named_attr_update_t loom_named_attr_replace(
    loom_string_id_t name_id, loom_attribute_t value) {
  loom_named_attr_update_t update = {
      /*.name_id=*/name_id,
      /*.remove=*/false,
      /*.value=*/value,
  };
  return update;
}

static inline loom_named_attr_update_t loom_named_attr_remove(
    loom_string_id_t name_id) {
  loom_named_attr_update_t update = {
      /*.name_id=*/name_id,
      /*.remove=*/true,
      /*.value=*/{0},
  };
  return update;
}

static inline loom_named_attr_update_slice_t loom_make_named_attr_update_slice(
    const loom_named_attr_update_t* updates, iree_host_size_t count) {
  loom_named_attr_update_slice_t slice = {
      /*.updates=*/count > 0 ? updates : NULL,
      /*.count=*/count,
  };
  return slice;
}

// Returns true if two attributes are structurally equal.
//
// Pointer-valued kinds compare by pointee contents, not pointer identity. For
// DICT attributes this comparison is positional over the canonical entry order,
// so callers should build or verify the dict invariant at construction
// boundaries instead of using unsorted ad hoc storage.
bool loom_attribute_equal(const loom_attribute_t* a, const loom_attribute_t* b);

// Returns a content-based hash of an attribute.
//
// Guarantee: if loom_attribute_equal(a, b), then
// loom_attribute_hash(a) == loom_attribute_hash(b).
uint32_t loom_attribute_hash(const loom_attribute_t* attr);

//===----------------------------------------------------------------------===//
// Attribute accessors
//===----------------------------------------------------------------------===//

// Returns the integer value of an I64 attribute.
static inline int64_t loom_attr_as_i64(loom_attribute_t attr) {
  return attr.i64;
}

// Returns the floating-point value of an F64 attribute.
static inline double loom_attr_as_f64(loom_attribute_t attr) {
  return attr.f64;
}

// Returns the string ID of a STRING attribute.
static inline loom_string_id_t loom_attr_as_string_id(loom_attribute_t attr) {
  return attr.string_id;
}

// Returns the boolean value of a BOOL attribute.
static inline bool loom_attr_as_bool(loom_attribute_t attr) {
  return attr.raw != 0;
}

// Returns the enum case index of an ENUM attribute.
static inline uint8_t loom_attr_as_enum(loom_attribute_t attr) {
  return (uint8_t)attr.raw;
}

// Returns the symbol reference of a SYMBOL attribute. Absent optional symbol
// attributes read as the explicit null symbol instead of the all-zero storage
// payload, which could otherwise alias module-local symbol ordinal 0.
static inline loom_symbol_ref_t loom_attr_as_symbol(loom_attribute_t attr) {
  if (loom_attr_is_absent(attr)) {
    return loom_symbol_ref_null();
  }
  return attr.symbol;
}

// Returns the type table index of a TYPE attribute.
static inline loom_type_id_t loom_attr_as_type_id(loom_attribute_t attr) {
  return attr.type_id;
}

// Returns the 1-based module encoding ID of an ENCODING attribute.
static inline uint16_t loom_attr_as_encoding_id(loom_attribute_t attr) {
  return (uint16_t)attr.encoding_id;
}

// Returns the byte payload of a BYTES attribute.
static inline iree_const_byte_span_t loom_attr_as_bytes(loom_attribute_t attr) {
  if (loom_attr_is_absent(attr)) {
    return iree_make_const_byte_span(NULL, 0);
  }
  IREE_ASSERT(attr.kind == LOOM_ATTR_BYTES);
  return iree_make_const_byte_span(attr.bytes, attr.reserved_1);
}

// Returns the dictionary entries of a DICT attribute.
//
// A zero-initialized optional attribute slot is interpreted as an empty slice
// so generated accessors remain ergonomic for optional AttrDict fields.
static inline loom_named_attr_slice_t loom_attr_as_dict(loom_attribute_t attr) {
  if (loom_attr_is_absent(attr)) {
    return loom_make_named_attr_slice(NULL, 0);
  }
  IREE_ASSERT(attr.kind == LOOM_ATTR_DICT);
  return loom_make_named_attr_slice(attr.dict_entries, attr.count);
}

#ifdef __cplusplus
}
#endif

#endif  // LOOM_IR_ATTRIBUTE_H_
