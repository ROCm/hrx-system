// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_MODULE_H_
#define IREE_VM_MODULE_H_

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/execution.h"
#include "iree/vm/ref.h"
#include "iree/vm/scalar.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_vm_module_t iree_vm_module_t;
typedef struct iree_vm_module_vtable_t iree_vm_module_vtable_t;

// Immutable provider-defined unit of composition. Implementations embed
// iree_vm_module_t at offset zero and own the vtable, descriptor, declaration
// storage, and private representation for the complete module lifetime. A
// published module may receive concurrent callbacks for independent processes;
// all mutable program-instance state lives in the process slices below.

// Optional reflection payloads are completed by reflection.h. The provider
// vtable passes only pointers and does not require their representations.
typedef struct iree_vm_metadata_entry_t iree_vm_metadata_entry_t;
typedef struct iree_vm_module_metadata_query_t iree_vm_module_metadata_query_t;
typedef struct iree_vm_module_presentation_t iree_vm_module_presentation_t;
typedef struct iree_vm_module_presentation_query_t
    iree_vm_module_presentation_query_t;

//===----------------------------------------------------------------------===//
// Immutable Declarations
//===----------------------------------------------------------------------===//

// Four-byte module-local machine signature type. Kind values 0x0001 through
// 0x00FF are defined iree_vm_scalar_type_t IDs.
typedef uint16_t iree_vm_module_signature_type_kind_t;
enum iree_vm_module_signature_type_kind_e {
  // Invalid type reserved for zero-initialized declarations.
  IREE_VM_MODULE_SIGNATURE_TYPE_KIND_INVALID = 0x0000,
  // Module-local ref type named by |type_ordinal|.
  IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF = 0x0100,
  // Module-local callable type named by |type_ordinal|.
  IREE_VM_MODULE_SIGNATURE_TYPE_KIND_FUNCTION = 0x0200,
};

typedef struct iree_vm_module_signature_type_t {
  // Scalar ID, REF, or FUNCTION.
  iree_vm_module_signature_type_kind_t kind;
  // Module-local ref/callable ordinal, or zero for a scalar.
  uint16_t type_ordinal;
} iree_vm_module_signature_type_t;
static_assert(sizeof(iree_vm_module_signature_type_t) == 4,
              "module signature types must remain four bytes");

// One source-ordered side of a machine signature and its exact physical-bank
// partition. Providers preserve these counts from construction or verified
// bytecode so linking never needs to rediscover them from |data|.
typedef struct iree_vm_module_signature_side_t {
  // Stable source-ordered machine signature fields.
  const iree_vm_module_signature_type_t* data;
  // Number of signature fields in |data|.
  uint16_t count;
  // Number of fields carried in the value bank.
  uint16_t value_count;
  // Number of fields carried in the ref bank.
  uint16_t ref_count;
  // Number of fields carried in the function bank.
  uint16_t function_count;
} iree_vm_module_signature_side_t;
static_assert(sizeof(void*) != 8 ||
                  sizeof(iree_vm_module_signature_side_t) == 16,
              "64-bit module signature sides must remain 16 bytes");

typedef struct iree_vm_module_signature_t {
  // Source-ordered machine arguments.
  iree_vm_module_signature_side_t arguments;
  // Source-ordered machine results.
  iree_vm_module_signature_side_t results;
} iree_vm_module_signature_t;

enum iree_vm_module_import_flag_bits_e {
  IREE_VM_MODULE_IMPORT_FLAG_NONE = 0u,
  // Absence of the target is permitted during program linking.
  IREE_VM_MODULE_IMPORT_FLAG_OPTIONAL = 1u << 0,
};
typedef uint32_t iree_vm_module_import_flags_t;

enum iree_vm_callable_type_flag_bits_e {
  IREE_VM_CALLABLE_TYPE_FLAG_NONE = 0u,
  // Callables of this type are permitted to suspend.
  IREE_VM_CALLABLE_TYPE_FLAG_MAY_YIELD = 1u << 0,
};
typedef uint32_t iree_vm_callable_type_flags_t;

// Transient semantic query result for one run of imports targeting a module.
typedef struct iree_vm_module_import_group_t {
  // Exact target module name.
  iree_string_view_t target_module_name;
  // First flat import ordinal in the group.
  iree_host_size_t first_import_ordinal;
  // Nonzero number of imports in the group.
  iree_host_size_t import_count;
} iree_vm_module_import_group_t;

// Transient semantic query result for one import declaration.
typedef struct iree_vm_module_import_declaration_t {
  // Exact target module name.
  iree_string_view_t target_module_name;
  // Exact target export name.
  iree_string_view_t target_export_name;
  // Module-local callable type defining the expected contract.
  iree_host_size_t callable_type_ordinal;
  // Import behavior flags.
  iree_vm_module_import_flags_t flags;
  // Stable metadata entry count for this declaration.
  iree_host_size_t metadata_count;
} iree_vm_module_import_declaration_t;

// Transient semantic query result for one export declaration.
typedef struct iree_vm_module_export_declaration_t {
  // Exact public export name.
  iree_string_view_t export_name;
  // Module-local callable type defining the public contract.
  iree_host_size_t callable_type_ordinal;
  // Private module-local execution target.
  iree_host_size_t function_ordinal;
  // Stable metadata entry count for this declaration.
  iree_host_size_t metadata_count;
} iree_vm_module_export_declaration_t;

// Transient semantic query result for one canonical structural callable type.
// Tables are strictly ordered by nesting depth, signature, then flags; nested
// function types name only earlier rows.
typedef struct iree_vm_module_callable_type_declaration_t {
  // Exact structural signature.
  iree_vm_module_signature_t signature;
  // Permitted callable behavior.
  iree_vm_callable_type_flags_t flags;
  // Maximum nested callable depth, zero for a leaf signature.
  uint16_t nesting_depth;
  // Reserved zero bits for future callable attributes.
  uint16_t reserved;
} iree_vm_module_callable_type_declaration_t;

// Borrowed identity of one public import declaration.
typedef struct iree_vm_import_t {
  // Module owning the declaration.
  const iree_vm_module_t* module;
  // Flat import ordinal.
  iree_host_size_t ordinal;
} iree_vm_import_t;

// Borrowed identity of one public export declaration.
typedef struct iree_vm_export_t {
  // Module owning the declaration.
  const iree_vm_module_t* module;
  // Export ordinal.
  iree_host_size_t ordinal;
} iree_vm_export_t;

// Borrowed identity of one structural callable type.
typedef struct iree_vm_callable_type_t {
  // Module owning the structural type.
  const iree_vm_module_t* module;
  // Callable-type ordinal.
  iree_host_size_t ordinal;
} iree_vm_callable_type_t;

// Exact target named by an import declaration.
typedef struct iree_vm_import_target_t {
  // Target module name.
  iree_string_view_t module_name;
  // Target export name.
  iree_string_view_t export_name;
} iree_vm_import_target_t;

//===----------------------------------------------------------------------===//
// Fixed Module Descriptor
//===----------------------------------------------------------------------===//

// Aggregate physical fields across all callable-type signatures. Counts may
// conservatively include the same structural signature more than once when
// callable declarations differ only in behavior flags.
typedef struct iree_vm_module_callable_field_counts_t {
  // Number of value-bank fields.
  iree_host_size_t value_count;
  // Number of ref-bank fields.
  iree_host_size_t ref_count;
  // Number of function-bank fields.
  iree_host_size_t function_count;
} iree_vm_module_callable_field_counts_t;

typedef struct iree_vm_module_counts_t {
  // Private executable-function domain.
  iree_host_size_t function_count;
  // Canonical structural callable-type domain.
  iree_host_size_t callable_type_count;
  // Sorted import-group domain.
  iree_host_size_t import_group_count;
  // Flat import declaration domain.
  iree_host_size_t import_count;
  // Public export declaration domain.
  iree_host_size_t export_count;
  // Module-scope metadata entry count.
  iree_host_size_t metadata_count;
  // Aggregate physical fields across all callable-type signatures.
  iree_vm_module_callable_field_counts_t callable_fields;
} iree_vm_module_counts_t;

enum iree_vm_module_flag_bits_e {
  IREE_VM_MODULE_FLAG_NONE = 0u,
  // The module has executable canonical ref types and may join a program.
  IREE_VM_MODULE_FLAG_LINKABLE = 1u << 0,
};
typedef uint32_t iree_vm_module_flags_t;

// Fixed immutable facts read directly by common VM code. The implementation
// owns and keeps the descriptor and every borrowed field live and immutable.
typedef struct iree_vm_module_descriptor_t {
  // Nonempty exact module link name.
  iree_string_view_t name;
  // Generic module capabilities.
  iree_vm_module_flags_t flags;
  // Canonical type handles in module-local ordinal order.
  iree_vm_ref_type_span_t ref_types;
  // Fixed declaration counts.
  iree_vm_module_counts_t counts;
  // Max-aligned opaque bytes required in every process.
  iree_host_size_t process_storage_size;
} iree_vm_module_descriptor_t;

//===----------------------------------------------------------------------===//
// Provider ABI
//===----------------------------------------------------------------------===//

// Generic ref-counted module base embedded at offset zero by implementations.
struct iree_vm_module_t {
  // Intrusive owner count published last by module initialization.
  iree_atomic_ref_count_t ref_count;
  // Immutable generic implementation vtable.
  const iree_vm_module_vtable_t* vtable;
  // Immutable implementation-owned descriptor.
  const iree_vm_module_descriptor_t* descriptor;
};
static_assert(offsetof(iree_vm_module_t, ref_count) == 0,
              "module ref count must remain at offset zero");
static_assert(sizeof(void*) != 8 || sizeof(iree_vm_module_t) == 24,
              "64-bit module bases must remain 24 bytes");
static_assert(sizeof(void*) != 4 || sizeof(iree_vm_module_t) == 12,
              "32-bit module bases must remain 12 bytes");

// Constructs one exact zeroed process-storage span synchronously. A present
// callback is invoked even when the span is empty. It cannot execute guest
// code, call another module, or yield. On failure it releases its own partial
// work; core detaches only earlier attached modules.
typedef iree_status_t(IREE_API_PTR* iree_vm_module_attach_state_fn_t)(
    iree_vm_module_t* module, iree_byte_span_t zeroed_storage,
    iree_allocator_t host_allocator);

// Validates that one attached process-storage span is publishable after the
// executable initializer completes. A present callback is invoked even when
// the span is empty. Failure leaves a representation accepted by
// |detach_state|.
typedef iree_status_t(IREE_API_PTR* iree_vm_module_seal_state_fn_t)(
    iree_vm_module_t* module, iree_byte_span_t storage);

// Releases one attached or sealed span without failing or freeing its
// container. Process-construction failure and final release invoke callbacks in
// reverse attachment order, including present callbacks for empty spans.
typedef void(IREE_API_PTR* iree_vm_module_detach_state_fn_t)(
    iree_vm_module_t* module, iree_byte_span_t storage);

// Populates one complete import group for an already validated ordinal.
typedef void(IREE_API_PTR* iree_vm_module_query_import_group_fn_t)(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_group);

// Populates one complete import declaration for an already validated ordinal.
typedef void(IREE_API_PTR* iree_vm_module_query_import_fn_t)(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_import);

// Populates one complete export declaration for an already validated ordinal.
typedef void(IREE_API_PTR* iree_vm_module_query_export_fn_t)(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_export);

// Populates one callable declaration for an already validated ordinal.
typedef void(IREE_API_PTR* iree_vm_module_query_callable_type_fn_t)(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_callable_type);

// Queries optional authored presentation without allocating. Every call
// reports exact transient-tail bytes. A complete fill requires exact field
// storage and sufficient transient storage; otherwise no storage is touched.
// Complete fills write every name/authored type but never the VM type. Views
// are valid UTF-8 without NUL and may borrow immutable module storage or the
// supplied transient tail.
typedef void(IREE_API_PTR* iree_vm_module_query_presentation_fn_t)(
    const iree_vm_module_t* module,
    const iree_vm_module_presentation_query_t* query,
    iree_vm_module_presentation_t* out_presentation);

// Populates one stable metadata entry for a validated scope and ordinal.
typedef void(IREE_API_PTR* iree_vm_module_metadata_by_ordinal_fn_t)(
    const iree_vm_module_t* module,
    const iree_vm_module_metadata_query_t* query,
    iree_vm_metadata_entry_t* out_entry);

// Initial generic module provider ABI version.
enum { IREE_VM_MODULE_ABI_VERSION_0 = 0 };

// Versioned generic module implementation interface. State lifecycle callbacks
// are independently nullable identity operations; every other callback is
// required in version zero. Common VM code allocates and routes process storage
// but never interprets it.
struct iree_vm_module_vtable_t {
  // Accessible bytes in this vtable.
  uint32_t structure_size;
  // Incompatible provider ABI version.
  uint32_t abi_version;
  // Private final-owner callback.
  void(IREE_API_PTR* destroy)(iree_vm_module_t* module);
  // Starts one module-local function.
  iree_vm_module_function_start_fn_t function_start;
  // Resumes one module-owned top frame.
  iree_vm_module_function_resume_fn_t function_resume;
  // Constructs one zeroed opaque process slice.
  iree_vm_module_attach_state_fn_t attach_state;
  // Validates one attached slice for publication.
  iree_vm_module_seal_state_fn_t seal_state;
  // Releases one attached or sealed slice.
  iree_vm_module_detach_state_fn_t detach_state;
  // Queries one import group by valid ordinal.
  iree_vm_module_query_import_group_fn_t query_import_group;
  // Queries one import declaration by valid ordinal.
  iree_vm_module_query_import_fn_t query_import;
  // Queries one export declaration by valid ordinal.
  iree_vm_module_query_export_fn_t query_export;
  // Queries one structural callable type by valid ordinal.
  iree_vm_module_query_callable_type_fn_t query_callable_type;
  // Queries optional authored presentation.
  iree_vm_module_query_presentation_fn_t query_presentation;
  // Queries stable typed metadata by valid scope and ordinal.
  iree_vm_module_metadata_by_ordinal_fn_t metadata_by_ordinal;
};

#define IREE_VM_MODULE_VTABLE_V0_REQUIRED_SIZE              \
  (offsetof(iree_vm_module_vtable_t, metadata_by_ordinal) + \
   sizeof(iree_vm_module_metadata_by_ordinal_fn_t))

static_assert(sizeof(void*) != 8 ||
                  offsetof(iree_vm_module_vtable_t, attach_state) == 32,
              "64-bit module vtable hot prefix must remain 32 bytes");

// Failure-atomically validates and publishes the first module owner. The
// factory calls this only after its immutable vtable, descriptor, declaration
// storage, and private representation are complete. Common validation is
// allocation-free and covers only implementation-neutral facts. Success makes
// |destroy| responsible for final cleanup; failure does not call |destroy| and
// leaves all cleanup with the unpublished factory object.
IREE_API_EXPORT iree_status_t
iree_vm_module_initialize(const iree_vm_module_vtable_t* vtable,
                          const iree_vm_module_descriptor_t* descriptor,
                          iree_vm_module_t* out_module);

// Retains |module| for the caller. A null module is ignored.
IREE_API_EXPORT void iree_vm_module_retain(iree_vm_module_t* module);

// Releases |module| from the caller. A null module is ignored.
IREE_API_EXPORT void iree_vm_module_release(iree_vm_module_t* module);

//===----------------------------------------------------------------------===//
// Structural Queries
//===----------------------------------------------------------------------===//

// Returns the stable link name of nonnull |module|.
IREE_API_EXPORT iree_string_view_t
iree_vm_module_name(const iree_vm_module_t* module);

// Returns the flat public import count of nonnull |module|.
IREE_API_EXPORT iree_host_size_t
iree_vm_module_import_count(const iree_vm_module_t* module);

// Returns the public export count of nonnull |module|.
IREE_API_EXPORT iree_host_size_t
iree_vm_module_export_count(const iree_vm_module_t* module);

// Returns the private executable-function count of nonnull |module|.
IREE_API_EXPORT iree_host_size_t
iree_vm_module_function_count(const iree_vm_module_t* module);

// Returns the module-local canonical ref-type count of nonnull |module|.
IREE_API_EXPORT iree_host_size_t
iree_vm_module_ref_type_count(const iree_vm_module_t* module);

// Returns one canonical ref type by module-local ordinal. Failure leaves
// |out_type| untouched.
IREE_API_EXPORT iree_status_t iree_vm_module_ref_type_by_ordinal(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_ref_type_t* out_type);

// Returns one borrowed import identity. Failure leaves |out_import| untouched.
IREE_API_EXPORT iree_status_t iree_vm_module_import_by_ordinal(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_import_t* out_import);

// Returns one borrowed export identity. Failure leaves |out_export| untouched.
IREE_API_EXPORT iree_status_t iree_vm_module_export_by_ordinal(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_export_t* out_export);

// Looks up one exact name in the sorted export directory. An absent name
// returns NOT_FOUND. Failure leaves |out_export| untouched.
IREE_API_EXPORT iree_status_t iree_vm_module_lookup_export(
    const iree_vm_module_t* module, iree_string_view_t name,
    iree_vm_export_t* out_export);

// Returns one import group. Failure leaves |out_group| untouched.
IREE_API_EXPORT iree_status_t iree_vm_module_query_import_group(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_group);

// Returns one import declaration. Failure leaves |out_import| untouched.
IREE_API_EXPORT iree_status_t iree_vm_module_query_import(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_import);

// Returns one export declaration. Failure leaves |out_export| untouched.
IREE_API_EXPORT iree_status_t iree_vm_module_query_export(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_export);

// Returns one callable declaration. Failure leaves the output untouched.
IREE_API_EXPORT iree_status_t iree_vm_module_query_callable_type(
    const iree_vm_module_t* module, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_callable_type);

// Returns the stable target of a valid borrowed import identity.
IREE_API_EXPORT iree_vm_import_target_t
iree_vm_import_target(iree_vm_import_t import_value);

// Returns the stable name of a valid borrowed export identity.
IREE_API_EXPORT iree_string_view_t
iree_vm_export_name(iree_vm_export_t export_value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_MODULE_H_
