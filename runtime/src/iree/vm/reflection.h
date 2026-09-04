// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_REFLECTION_H_
#define IREE_VM_REFLECTION_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Typed Metadata
//===----------------------------------------------------------------------===//

// Append-only stable metadata value types. The 16-bit semantic ID domain is
// carried by uint32_t. BOOL is one canonical 0/1 byte; I64/U64/F64 are
// unaligned little-endian bytes; UTF8 may contain NUL bytes. Unknown nonzero
// IDs remain valid opaque byte spans.
enum iree_vm_metadata_value_type_e {
  // Invalid type reserved for zero-initialized outputs.
  IREE_VM_METADATA_VALUE_TYPE_INVALID = 0u,
  // One byte containing canonical zero or one.
  IREE_VM_METADATA_VALUE_TYPE_BOOL = 1u,
  // Eight little-endian signed integer bytes.
  IREE_VM_METADATA_VALUE_TYPE_I64 = 2u,
  // Eight little-endian unsigned integer bytes.
  IREE_VM_METADATA_VALUE_TYPE_U64 = 3u,
  // Eight little-endian IEEE binary64 bytes.
  IREE_VM_METADATA_VALUE_TYPE_F64 = 4u,
  // Valid UTF-8 bytes that may contain NUL.
  IREE_VM_METADATA_VALUE_TYPE_UTF8 = 5u,
  // Uninterpreted bytes.
  IREE_VM_METADATA_VALUE_TYPE_BYTES = 6u,
};
typedef uint32_t iree_vm_metadata_value_type_t;

typedef struct iree_vm_metadata_value_t {
  // Stable value encoding.
  iree_vm_metadata_value_type_t type;
  // Stable module-lifetime bytes.
  iree_const_byte_span_t data;
} iree_vm_metadata_value_t;

typedef struct iree_vm_metadata_entry_t {
  // Stable nonempty UTF-8 key without NUL, strictly ordered in its scope.
  iree_string_view_t key;
  // Stable typed value.
  iree_vm_metadata_value_t value;
} iree_vm_metadata_entry_t;

//===----------------------------------------------------------------------===//
// Callable Descriptions
//===----------------------------------------------------------------------===//

enum iree_vm_signature_type_kind_e {
  // Invalid kind reserved for zero-initialized outputs.
  IREE_VM_SIGNATURE_TYPE_KIND_INVALID = 0u,
  // Exact iree_vm_scalar_type_t payload.
  IREE_VM_SIGNATURE_TYPE_KIND_SCALAR = 1u,
  // Canonical iree_vm_ref_type_t payload.
  IREE_VM_SIGNATURE_TYPE_KIND_REF = 2u,
  // Borrowed module-bound iree_vm_callable_type_t payload.
  IREE_VM_SIGNATURE_TYPE_KIND_FUNCTION = 3u,
};
typedef uint32_t iree_vm_signature_type_kind_t;

// Resolved public reflection type.
typedef struct iree_vm_signature_type_t {
  // Active payload kind.
  iree_vm_signature_type_kind_t kind;
  // Resolved payload selected by |kind|.
  union {
    // Exact scalar type.
    iree_vm_scalar_type_t scalar;
    // Canonical exact ref descriptor.
    iree_vm_ref_type_t ref;
    // Reachable module-bound callable type.
    iree_vm_callable_type_t callable;
  } value;
} iree_vm_signature_type_t;

typedef struct iree_vm_signature_type_span_t {
  // Contiguous resolved types in caller-owned query storage.
  const iree_vm_signature_type_t* data;
  // Number of resolved types in |data|.
  iree_host_size_t count;
} iree_vm_signature_type_span_t;

typedef struct iree_vm_signature_field_t {
  // Required resolved VM type.
  iree_vm_signature_type_t type;
  // Source-level name anchored at this field, or empty when unavailable.
  iree_string_view_t name;
  // Source-level type anchored at this field, or empty when unavailable.
  // Multi-field source aggregates anchor their text at the first VM field.
  iree_string_view_t authored_type;
} iree_vm_signature_field_t;

typedef struct iree_vm_signature_field_span_t {
  // Contiguous fields in caller-owned query storage.
  const iree_vm_signature_field_t* data;
  // Number of fields in |data|.
  iree_host_size_t count;
} iree_vm_signature_field_span_t;

typedef struct iree_vm_signature_field_storage_t {
  // Writable fields in caller-owned query storage.
  iree_vm_signature_field_t* data;
  // Actual available field count.
  iree_host_size_t count;
} iree_vm_signature_field_storage_t;

typedef struct iree_vm_import_description_t {
  // Exact import target.
  iree_vm_import_target_t target;
  // Import behavior flags.
  iree_vm_module_import_flags_t flags;
  // Expected callable behavior.
  iree_vm_callable_type_flags_t callable_flags;
  // Source-ordered machine argument fields.
  iree_vm_signature_field_span_t arguments;
  // Source-ordered machine result fields.
  iree_vm_signature_field_span_t results;
  // Declaration documentation, or empty when unavailable.
  iree_string_view_t documentation;
  // Complete authored function type, or empty when unavailable.
  iree_string_view_t authored_type;
} iree_vm_import_description_t;

typedef struct iree_vm_export_description_t {
  // Exact public alias name.
  iree_string_view_t name;
  // Actual public callable behavior.
  iree_vm_callable_type_flags_t callable_flags;
  // Source-ordered machine argument fields.
  iree_vm_signature_field_span_t arguments;
  // Source-ordered machine result fields.
  iree_vm_signature_field_span_t results;
  // Declaration documentation, or empty when unavailable.
  iree_string_view_t documentation;
  // Complete authored function type, or empty when unavailable.
  iree_string_view_t authored_type;
} iree_vm_export_description_t;

typedef struct iree_vm_callable_type_description_t {
  // Permitted callable behavior.
  iree_vm_callable_type_flags_t flags;
  // Source-ordered resolved argument types.
  iree_vm_signature_type_span_t arguments;
  // Source-ordered resolved result types.
  iree_vm_signature_type_span_t results;
} iree_vm_callable_type_description_t;

//===----------------------------------------------------------------------===//
// Provider Reflection Queries
//===----------------------------------------------------------------------===//

enum iree_vm_module_declaration_kind_e {
  // Invalid kind reserved for zero-initialized queries.
  IREE_VM_MODULE_DECLARATION_KIND_INVALID = 0u,
  // Public import declaration.
  IREE_VM_MODULE_DECLARATION_KIND_IMPORT = 1u,
  // Public export declaration.
  IREE_VM_MODULE_DECLARATION_KIND_EXPORT = 2u,
};
typedef uint32_t iree_vm_module_declaration_kind_t;

typedef struct iree_vm_module_declaration_t {
  // Import or export declaration kind.
  iree_vm_module_declaration_kind_t kind;
  // Flat declaration ordinal.
  iree_host_size_t ordinal;
} iree_vm_module_declaration_t;

// Correlated presentation inputs passed to a module provider.
typedef struct iree_vm_module_presentation_query_t {
  // Declaration whose optional presentation is requested.
  iree_vm_module_declaration_t declaration;
  // Argument-then-result fields, or empty for a size probe.
  iree_vm_signature_field_storage_t fields;
  // Max-aligned transient tail, or empty for a size probe.
  iree_byte_span_t transient_storage;
} iree_vm_module_presentation_query_t;

// Optional provider-authored presentation returned from one query.
typedef struct iree_vm_module_presentation_t {
  // Exact transient-tail bytes required after max-alignment padding.
  iree_host_size_t required_transient_storage_size;
  // Declaration documentation, or empty when unavailable.
  iree_string_view_t documentation;
  // Complete authored function type, or empty when unavailable.
  iree_string_view_t authored_type;
} iree_vm_module_presentation_t;

enum iree_vm_module_metadata_scope_kind_e {
  // Invalid kind reserved for zero-initialized queries.
  IREE_VM_MODULE_METADATA_SCOPE_KIND_INVALID = 0u,
  // Module-level public metadata.
  IREE_VM_MODULE_METADATA_SCOPE_KIND_MODULE = 1u,
  // Metadata belonging to one public import alias.
  IREE_VM_MODULE_METADATA_SCOPE_KIND_IMPORT = 2u,
  // Metadata belonging to one public export alias.
  IREE_VM_MODULE_METADATA_SCOPE_KIND_EXPORT = 3u,
};
typedef uint32_t iree_vm_module_metadata_scope_kind_t;

typedef struct iree_vm_module_metadata_scope_t {
  // Module, import, or export scope.
  iree_vm_module_metadata_scope_kind_t kind;
  // Import/export ordinal, or zero for module scope.
  iree_host_size_t ordinal;
} iree_vm_module_metadata_scope_t;

typedef struct iree_vm_module_metadata_query_t {
  // Selected public metadata scope.
  iree_vm_module_metadata_scope_t scope;
  // Valid entry ordinal within that scope.
  iree_host_size_t ordinal;
} iree_vm_module_metadata_query_t;

//===----------------------------------------------------------------------===//
// Public Reflection
//===----------------------------------------------------------------------===//

// Import, export, and callable descriptions use a two-call caller-storage
// protocol. First pass empty storage and a null description to obtain the exact
// required byte size. Allocate that many bytes with normal iree_allocator_t
// alignment, then repeat the query with the storage and output description.
// Supplying insufficient storage is not an error: the call reports the required
// size and leaves both storage and description untouched.
//
// Returned field/type spans point into caller storage. Presentation strings may
// point into immutable module storage or the supplied transient tail. The
// borrowed declaration's module and the caller storage must therefore remain
// live until all description views are discarded. A query invokes provider
// presentation exactly once per call and never allocates.

// Describes one valid borrowed import identity using the caller-storage
// protocol above.
IREE_API_EXPORT iree_status_t iree_vm_import_query_description(
    iree_vm_import_t import_value, iree_byte_span_t storage,
    iree_host_size_t* out_required_storage_size,
    iree_vm_import_description_t* out_description);

// Describes one valid borrowed export identity using the caller-storage
// protocol above.
IREE_API_EXPORT iree_status_t iree_vm_export_query_description(
    iree_vm_export_t export_value, iree_byte_span_t storage,
    iree_host_size_t* out_required_storage_size,
    iree_vm_export_description_t* out_description);

// Describes one valid borrowed structural callable identity. Its storage
// requirement contains only resolved signature types and no presentation tail.
IREE_API_EXPORT iree_status_t iree_vm_callable_type_query_description(
    iree_vm_callable_type_t callable_type, iree_byte_span_t storage,
    iree_host_size_t* out_required_storage_size,
    iree_vm_callable_type_description_t* out_description);

// Returns the module-scope metadata count of nonnull |module|.
IREE_API_EXPORT iree_host_size_t
iree_vm_module_metadata_count(const iree_vm_module_t* module);

// Returns the metadata count of a valid borrowed import identity.
IREE_API_EXPORT iree_host_size_t
iree_vm_import_metadata_count(iree_vm_import_t import_value);

// Returns the metadata count of a valid borrowed export identity.
IREE_API_EXPORT iree_host_size_t
iree_vm_export_metadata_count(iree_vm_export_t export_value);

// Returns one module metadata entry. Failure leaves |out_entry| untouched.
IREE_API_EXPORT iree_status_t iree_vm_module_metadata_by_ordinal(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_metadata_entry_t* out_entry);

// Returns one import metadata entry. Failure leaves |out_entry| untouched.
IREE_API_EXPORT iree_status_t iree_vm_import_metadata_by_ordinal(
    iree_vm_import_t import_value, iree_host_size_t ordinal,
    iree_vm_metadata_entry_t* out_entry);

// Returns one export metadata entry. Failure leaves |out_entry| untouched.
IREE_API_EXPORT iree_status_t iree_vm_export_metadata_by_ordinal(
    iree_vm_export_t export_value, iree_host_size_t ordinal,
    iree_vm_metadata_entry_t* out_entry);

// Looks up one exact module metadata key. Absence succeeds with |out_found|
// false and leaves |out_value| untouched; failure leaves both untouched.
IREE_API_EXPORT iree_status_t iree_vm_module_try_lookup_metadata(
    const iree_vm_module_t* module, iree_string_view_t key, bool* out_found,
    iree_vm_metadata_value_t* out_value);

// Looks up one exact import metadata key with the same transactional outputs.
IREE_API_EXPORT iree_status_t iree_vm_import_try_lookup_metadata(
    iree_vm_import_t import_value, iree_string_view_t key, bool* out_found,
    iree_vm_metadata_value_t* out_value);

// Looks up one exact export metadata key with the same transactional outputs.
IREE_API_EXPORT iree_status_t iree_vm_export_try_lookup_metadata(
    iree_vm_export_t export_value, iree_string_view_t key, bool* out_found,
    iree_vm_metadata_value_t* out_value);

// Extracts one canonical BOOL. Failure leaves |out_value| untouched.
IREE_API_EXPORT iree_status_t iree_vm_bool_from_metadata_value(
    iree_vm_metadata_value_t value, bool* out_value);

// Extracts one little-endian I64. Failure leaves |out_value| untouched.
IREE_API_EXPORT iree_status_t iree_vm_i64_from_metadata_value(
    iree_vm_metadata_value_t value, int64_t* out_value);

// Extracts one little-endian U64. Failure leaves |out_value| untouched.
IREE_API_EXPORT iree_status_t iree_vm_u64_from_metadata_value(
    iree_vm_metadata_value_t value, uint64_t* out_value);

// Extracts one little-endian F64. Failure leaves |out_value| untouched.
IREE_API_EXPORT iree_status_t iree_vm_f64_from_metadata_value(
    iree_vm_metadata_value_t value, double* out_value);

// Returns one borrowed valid UTF-8 view. Failure leaves the output untouched.
IREE_API_EXPORT iree_status_t iree_vm_string_view_from_metadata_value(
    iree_vm_metadata_value_t value, iree_string_view_t* out_value);

// Returns one borrowed opaque byte span. Failure leaves the output untouched.
IREE_API_EXPORT iree_status_t iree_vm_const_byte_span_from_metadata_value(
    iree_vm_metadata_value_t value, iree_const_byte_span_t* out_value);

// Common provider callback for declarations with no optional presentation. It
// still runs for zero-field and zero-transient-storage queries.
IREE_API_EXPORT void iree_vm_module_query_presentation_none(
    const iree_vm_module_t* module,
    const iree_vm_module_presentation_query_t* query,
    iree_vm_module_presentation_t* out_presentation);
// Required no-op metadata callback for providers whose scopes are all empty.
// No valid ordinal query reaches it.
IREE_API_EXPORT void iree_vm_module_metadata_by_ordinal_none(
    const iree_vm_module_t* module,
    const iree_vm_module_metadata_query_t* query,
    iree_vm_metadata_entry_t* out_entry);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_REFLECTION_H_
