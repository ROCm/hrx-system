// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured error definitions for the loom compiler infrastructure.
//
// This is the foundational error layer. It owns severity levels, error
// domains, emitter identifiers, and the error definition type. All
// diagnostic subsystems (text format, verifier, bytecode reader) depend
// on this — it depends on none of them.
//
// Error definitions are .rodata tables generated from the Python error
// DSL (loom/py/loom/error/*.py). Each error has a stable symbolic ID,
// (domain, code) identity, typed parameter schema, message template, and
// optional fix hint. The tables are generated from Python so readable catalogs
// remain the source of truth while C consumers see compact rodata.
//
// Materialized diagnostics, lightweight emitters, and verifier hooks all
// build on these definitions. A build that excludes the text format still
// uses the same error tables for structured reporting through JSON or
// programmatic sinks.

#ifndef LOOM_ERROR_ERROR_DEFS_H_
#define LOOM_ERROR_ERROR_DEFS_H_

#include "iree/base/api.h"
#include "loom/error/emitter.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

// Diagnostic severity levels. Shared across all subsystems.
typedef enum loom_diagnostic_severity_e {
  LOOM_DIAGNOSTIC_ERROR = 0,
  LOOM_DIAGNOSTIC_WARNING = 1,
  LOOM_DIAGNOSTIC_REMARK = 2,
  LOOM_DIAGNOSTIC_COUNT_,
} loom_diagnostic_severity_t;

// Semantic error domain. Describes *what* the error is about, not which
// subsystem emitted it. An agent filtering for type errors doesn't care
// whether the verifier or the parser caught it.
typedef enum loom_error_domain_e {
  LOOM_ERROR_DOMAIN_TYPE = 0,       // Type mismatches, constraint violations.
  LOOM_ERROR_DOMAIN_SHAPE = 1,      // Rank mismatches, shape inconsistencies.
  LOOM_ERROR_DOMAIN_SUBRANGE = 2,   // Offset/size counts, bounds violations.
  LOOM_ERROR_DOMAIN_ENCODING = 3,   // Encoding mismatches.
  LOOM_ERROR_DOMAIN_STRUCTURE = 4,  // Count errors, terminators, regions.
  LOOM_ERROR_DOMAIN_DOMINANCE = 5,  // Undefined values, use-after-consume.
  LOOM_ERROR_DOMAIN_SYMBOL = 6,     // Unresolved references.
  LOOM_ERROR_DOMAIN_PARSE = 7,      // Syntax errors, tokenization.
  LOOM_ERROR_DOMAIN_BYTECODE = 8,   // Format errors, version mismatches.
  LOOM_ERROR_DOMAIN_FOLD = 9,       // Poison folding, canonicalization.
  LOOM_ERROR_DOMAIN_LOWERING = 10,  // Pass legality and unsupported mappings.
  LOOM_ERROR_DOMAIN_BACKEND = 11,   // Target codegen feedback and emission.
  LOOM_ERROR_DOMAIN_TARGET = 12,    // Target contract and legality failures.
  LOOM_ERROR_DOMAIN_AMDGPU = 13,    // AMDGPU legality and lowering failures.
  LOOM_ERROR_DOMAIN_X86 = 14,       // X86 legality and lowering failures.
  LOOM_ERROR_DOMAIN_WASM = 15,      // WebAssembly legality and lowering.
  LOOM_ERROR_DOMAIN_SPIRV = 16,     // SPIR-V legality and lowering failures.
  LOOM_ERROR_DOMAIN_COUNT_,
} loom_error_domain_t;

// A compact stable reference to a generated error definition. It packs a
// non-zero domain ordinal and code into 16 bits so generated static tables can
// name diagnostics without requiring public storage symbols.
typedef uint16_t loom_error_ref_t;

#define LOOM_ERROR_REF_NONE ((loom_error_ref_t)0u)
#define LOOM_ERROR_REF_CODE_BITS 10u
#define LOOM_ERROR_REF_CODE_MASK \
  ((uint16_t)((1u << LOOM_ERROR_REF_CODE_BITS) - 1u))
#define LOOM_ERROR_REF(domain, code)                  \
  ((loom_error_ref_t)(((((uint16_t)(domain)) + 1u)    \
                       << LOOM_ERROR_REF_CODE_BITS) | \
                      ((uint16_t)(code))))

static inline bool loom_error_ref_is_set(loom_error_ref_t ref) {
  return ref != LOOM_ERROR_REF_NONE;
}

static inline loom_error_domain_t loom_error_ref_domain(loom_error_ref_t ref) {
  return (loom_error_domain_t)((ref >> LOOM_ERROR_REF_CODE_BITS) - 1u);
}

static inline uint16_t loom_error_ref_code(loom_error_ref_t ref) {
  return ref & LOOM_ERROR_REF_CODE_MASK;
}

// Which subsystem emitted the diagnostic. Context for debugging and
// filtering, not part of the error identity.
typedef enum loom_emitter_e {
  LOOM_EMITTER_VERIFIER = 0,
  LOOM_EMITTER_PARSER = 1,
  LOOM_EMITTER_BYTECODE_READER = 2,
  LOOM_EMITTER_PASS = 3,
  LOOM_EMITTER_BUILDER = 4,
  LOOM_EMITTER_COUNT_,
} loom_emitter_t;

// Parameter kind for diagnostic parameters. Determines how the parameter
// is stored in a loom_diagnostic_param_t and how it is rendered.
typedef enum loom_param_kind_e {
  LOOM_PARAM_STRING = 0,       // iree_string_view_t
  LOOM_PARAM_I64 = 1,          // int64_t
  LOOM_PARAM_U32 = 2,          // uint32_t
  LOOM_PARAM_BOOL = 3,         // bool
  LOOM_PARAM_TYPE = 4,         // loom_type_t (rendered by type printer)
  LOOM_PARAM_U64 = 5,          // uint64_t
  LOOM_PARAM_STRING_LIST = 6,  // Span of iree_string_view_t values.
  LOOM_PARAM_COUNT_,
} loom_param_kind_t;

// Returns the severity name string ("error", "warning", "remark").
const char* loom_diagnostic_severity_name(loom_diagnostic_severity_t severity);

// Returns the domain name string (e.g., "TYPE", "SHAPE").
const char* loom_error_domain_name(loom_error_domain_t domain);

// Parses a domain name string (e.g., "TYPE", "SHAPE") into its enum
// value. Returns true if the name was recognized, false otherwise.
// The match is exact and case-sensitive.
bool loom_error_domain_from_name(iree_string_view_t name,
                                 loom_error_domain_t* out_domain);

// Returns the emitter name string (e.g., "verifier", "parser").
const char* loom_emitter_name(loom_emitter_t emitter);

//===----------------------------------------------------------------------===//
// Error definition types
//===----------------------------------------------------------------------===//

// Parameter definition in .rodata. Describes one typed placeholder in
// an error's message template.
typedef struct loom_error_param_def_t {
  const char* name;        // Placeholder name (e.g., "operand_name").
  loom_param_kind_t kind;  // How to store and render the parameter.
} loom_error_param_def_t;

// Error definition in .rodata. One per error code. The (domain, code)
// pair is the stable identity — codes are never reused within a domain.
//
// The message_template uses {param_name} placeholders that reference
// the param_defs array. The rendering function expands placeholders by
// substituting the corresponding runtime parameter value.
typedef struct loom_error_def_t {
  const char* error_id;  // Stable symbolic ID, e.g. "ERR_TYPE_001".
  loom_error_domain_t domain;
  loom_diagnostic_severity_t severity;
  uint16_t code;
  const char* summary;            // One-line description.
  const char* message_template;   // "{field_a} type {type_a} ..."
  const char* fix_hint_template;  // NULL if no fix hint.
  const loom_error_param_def_t* param_defs;
  uint8_t param_count;
} loom_error_def_t;

// Generated domain-local catalog. Code zero is unused so diagnostic code N is a
// direct index into errors_by_code[N].
typedef struct loom_error_domain_catalog_t {
  // Domain owned by this catalog.
  loom_error_domain_t domain;
  // Number of entries in errors_by_code, including the unused code zero slot.
  uint16_t code_count;
  // Code-indexed error definition table. Missing codes have NULL entries.
  const loom_error_def_t* const* errors_by_code;
} loom_error_domain_catalog_t;

// Composed error catalog used to resolve compact refs carried by generated
// tables. Composition happens at package/provider boundaries so core code can
// resolve injected target refs without depending on optional target catalogs.
typedef struct loom_error_catalog_t {
  // Domain-indexed catalog shards. Absent domains have NULL entries.
  const loom_error_domain_catalog_t* domains[LOOM_ERROR_DOMAIN_COUNT_];
  // Optional catalog searched when this catalog does not contain a domain/code.
  const struct loom_error_catalog_t* fallback_catalog;
} loom_error_catalog_t;

typedef struct loom_diagnostic_string_list_t {
  // Values in stable display and serialization order.
  const iree_string_view_t* values;
  // Number of values in the list.
  iree_host_size_t count;
} loom_diagnostic_string_list_t;

// A runtime parameter value, matching an error_def's param_defs entry.
// The kind is redundant with the def but avoids chasing the def pointer
// during rendering.
typedef struct loom_diagnostic_param_t {
  loom_param_kind_t kind;  // Active parameter value variant.
  // Optional structured metadata describing the op field this param names.
  loom_diagnostic_field_ref_t field_ref;
  union {
    iree_string_view_t string;                  // LOOM_PARAM_STRING payload.
    int64_t i64;                                // LOOM_PARAM_I64 payload.
    uint32_t u32;                               // LOOM_PARAM_U32 payload.
    bool boolean;                               // LOOM_PARAM_BOOL payload.
    loom_type_t type;                           // LOOM_PARAM_TYPE payload.
    uint64_t u64;                               // LOOM_PARAM_U64 payload.
    loom_diagnostic_string_list_t string_list;  // LOOM_PARAM_STRING_LIST.
  };
} loom_diagnostic_param_t;

// Inline constructors for diagnostic params. Portable across C and C++
// without relying on designated initializers for anonymous union members.

static inline loom_diagnostic_param_t loom_param_string(
    iree_string_view_t value) {
  loom_diagnostic_param_t param;
  memset(&param, 0, sizeof(param));
  param.kind = LOOM_PARAM_STRING;
  param.string = value;
  return param;
}

static inline loom_diagnostic_param_t loom_param_i64(int64_t value) {
  loom_diagnostic_param_t param;
  memset(&param, 0, sizeof(param));
  param.kind = LOOM_PARAM_I64;
  param.i64 = value;
  return param;
}

static inline loom_diagnostic_param_t loom_param_u32(uint32_t value) {
  loom_diagnostic_param_t param;
  memset(&param, 0, sizeof(param));
  param.kind = LOOM_PARAM_U32;
  param.u32 = value;
  return param;
}

static inline loom_diagnostic_param_t loom_param_u64(uint64_t value) {
  loom_diagnostic_param_t param;
  memset(&param, 0, sizeof(param));
  param.kind = LOOM_PARAM_U64;
  param.u64 = value;
  return param;
}

static inline loom_diagnostic_param_t loom_param_string_list(
    const iree_string_view_t* values, iree_host_size_t count) {
  loom_diagnostic_param_t param;
  memset(&param, 0, sizeof(param));
  param.kind = LOOM_PARAM_STRING_LIST;
  param.string_list.values = values;
  param.string_list.count = count;
  return param;
}

static inline loom_diagnostic_param_t loom_param_bool(bool value) {
  loom_diagnostic_param_t param;
  memset(&param, 0, sizeof(param));
  param.kind = LOOM_PARAM_BOOL;
  param.boolean = value;
  return param;
}

static inline loom_diagnostic_param_t loom_param_type(loom_type_t value) {
  loom_diagnostic_param_t param;
  memset(&param, 0, sizeof(param));
  param.kind = LOOM_PARAM_TYPE;
  param.type = value;
  return param;
}

// Attaches structured field-ref metadata to an existing diagnostic parameter.
// The parameter's rendered value remains unchanged; sinks can consume the
// metadata for highlighting and machine-oriented JSON metadata.
static inline loom_diagnostic_param_t loom_param_with_field_ref(
    loom_diagnostic_param_t param, loom_diagnostic_field_ref_t field_ref) {
  param.field_ref = field_ref;
  return param;
}

// Returns the error def for a (domain, code) pair in |catalog|'s fallback
// chain, or NULL if no catalog in the chain contains that domain/code.
const loom_error_def_t* loom_error_catalog_lookup(
    const loom_error_catalog_t* catalog, loom_error_domain_t domain,
    uint16_t code);

// Returns the error def for a compact reference in |catalog|'s fallback chain,
// or NULL if |ref| is empty or names no definition in the chain.
const loom_error_def_t* loom_error_catalog_lookup_ref(
    const loom_error_catalog_t* catalog, loom_error_ref_t ref);

// Returns the error def for a (domain, code) pair, or NULL if not found.
const loom_error_def_t* loom_error_def_lookup(loom_error_domain_t domain,
                                              uint16_t code);

// Returns the error def for a compact reference, or NULL if |ref| is
// LOOM_ERROR_REF_NONE or names no generated definition.
const loom_error_def_t* loom_error_def_lookup_ref(loom_error_ref_t ref);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ERROR_ERROR_DEFS_H_
