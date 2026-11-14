// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_IMPORT_CXX_H_
#define LOOMC_IMPORT_CXX_H_

#include "loomc/module.h"

/// @file
/// Optional native C/C++ source import into ordinary Loom modules.
///
/// One configured translation unit produces one compilation-unit module.
/// Multiple kernels and ordinary functions share the module. Build scheduling,
/// host/device partitioning, and linking remain above this interface.
///
/// The implementation uses the standalone cxx frontend and native Loom APIs;
/// no LLVM or Python runtime is required. Enable the cxx importer package when
/// building the library and link the optional import/cxx component.

#ifdef __cplusplus
extern "C" {
#endif

/// Source integer and pointer layout, independent of the importing process.
typedef enum loomc_cxx_data_model_e {
  /// 32-bit int; 64-bit long and pointers.
  LOOMC_CXX_DATA_MODEL_LP64 = 0,
  /// 32-bit int and long; 64-bit pointers.
  LOOMC_CXX_DATA_MODEL_LLP64 = 1,
  /// 32-bit int, long, and pointers.
  LOOMC_CXX_DATA_MODEL_ILP32 = 2,
} loomc_cxx_data_model_t;

/// Translation-unit floating-point transformation policy.
///
/// Source fastmath attributes may replace this default for individual
/// functions or statements.
typedef enum loomc_cxx_fp_model_e {
  /// Preserve floating-point value and evaluation-order semantics.
  LOOMC_CXX_FP_MODEL_PRECISE = 0,
  /// Permit the complete Loom fast-math contract.
  LOOMC_CXX_FP_MODEL_FAST = 1,
} loomc_cxx_fp_model_t;

/// Include delivery policy.
typedef enum loomc_cxx_import_flag_bits_e {
  /// Omit the embedded system include root; use caller paths/providers only.
  LOOMC_CXX_IMPORT_FLAG_NO_BUILTIN_INCLUDES = 1u << 0,
  /// Hosted single-source profile: the unit is a host compile for the triple
  /// (Linux/glibc/libc++ predefines, exceptions disabled unless EXCEPTIONS).
  LOOMC_CXX_IMPORT_FLAG_HOSTED = 1u << 1,
  /// Hosted profile only: parse with C++ exceptions enabled, as a host compile
  /// with -fexceptions. Device import still rejects exception handling in the
  /// code it translates.
  LOOMC_CXX_IMPORT_FLAG_EXCEPTIONS = 1u << 2,
} loomc_cxx_import_flag_bits_t;
/// Bitmask of loomc_cxx_import_flag_bits_t values.
typedef uint32_t loomc_cxx_import_flags_t;

/// Resolves a candidate include path to immutable source bytes.
///
/// Paths follow ordinary quote, user, system, and include_next search rules.
/// Candidate paths use generic '/' separators on every host.
/// Return OK with a NULL output for a missing candidate. Other failures return
/// status with a NULL output. A found source uses UNKNOWN format; its contents
/// supply the header and the candidate path determines include identity.
///
/// @param user_data Caller state from the provider descriptor.
/// @param path Candidate path borrowed for this callback.
/// @param out_source Receives one retained source or NULL for a missing path.
/// @return OK for found/missing candidates, or a provider failure status.
///
/// @ownership
/// A successful non-NULL output transfers one retained reference to the
/// importer. A cache can retain its own reference and return another. The
/// importer releases the reference once it has copied the bytes.
///
/// @thread_safety
/// Calls are synchronous and sequential within one import. A provider shared
/// by concurrent imports must synchronize its own mutable state.
typedef loomc_status_t(LOOMC_API_PTR* loomc_cxx_source_provider_fn_t)(
    void* user_data, loomc_string_view_t path, loomc_source_t** out_source);

/// Optional source provider replacing filesystem include reads.
typedef struct loomc_cxx_source_provider_t {
  /// Callback, or NULL to read ordinary filesystem paths.
  loomc_cxx_source_provider_fn_t fn;
  /// Caller-owned state borrowed until import returns.
  void* user_data;
} loomc_cxx_source_provider_t;

/// Maps a physical frontend path to the logical source identifier retained in
/// diagnostics and imported modules.
///
/// The physical path remains the identity used for include lookup. Returning
/// the same logical identifier for distinct physical sources is invalid.
///
/// @param user_data Caller state from the mapper descriptor.
/// @param path Physical frontend path borrowed for this callback.
/// @param out_identifier Receives the logical diagnostic/module identity.
/// @return OK when the path was mapped, or a mapping failure status.
///
/// @lifetime
/// The returned identifier must remain valid until the next mapper callback.
/// The importer copies it before another callback can occur.
///
/// @thread_safety
/// Calls are synchronous and sequential within one import. A mapper shared by
/// concurrent imports must synchronize its own mutable state.
typedef loomc_status_t(LOOMC_API_PTR* loomc_cxx_source_path_mapper_fn_t)(
    void* user_data, loomc_string_view_t path,
    loomc_string_view_t* out_identifier);

/// Optional logical-identity mapper for retained frontend source paths.
typedef struct loomc_cxx_source_path_mapper_t {
  /// Callback, or NULL to retain physical frontend paths unchanged.
  loomc_cxx_source_path_mapper_fn_t fn;
  /// Caller-owned state borrowed until import returns.
  void* user_data;
} loomc_cxx_source_path_mapper_t;

/// Preprocessor definition applied after the frontend's predefined macros.
typedef struct loomc_cxx_define_t {
  /// Macro name, including parameters for a function-like macro.
  loomc_string_view_t name;
  /// Replacement token text; empty defines an empty macro.
  loomc_string_view_t value;
} loomc_cxx_define_t;

/// Immutable process-local C/C++ semantic preamble.
///
/// A preamble owns the parsed state and exact source snapshot of one leading
/// include prefix. It does not retain the source, context, import options, or
/// callback state used to create it. Retained handles may be shared by
/// concurrent imports; each import restores into a fresh translation unit.
typedef struct loomc_cxx_preamble_t loomc_cxx_preamble_t;

/// Configuration for one source translation unit.
///
/// Zero initialization selects C++26, LP64, precise math, visible definitions
/// as roots, and the embedded facade system root when built into the library.
/// All strings, arrays, and callback state are borrowed until import returns.
typedef struct loomc_cxx_import_options_t {
  /// Structure type; LOOMC_STRUCTURE_TYPE_CXX_IMPORT_OPTIONS when nonzero.
  loomc_structure_type_t type;
  /// Size of this structure in bytes, or zero.
  loomc_host_size_t structure_size;
  /// Reserved extension chain. Must be NULL.
  const void* next;
  /// Frontend standard spelling, such as c23 or c++26. Empty selects c++26,
  /// or c++20 for hosted import.
  loomc_string_view_t standard;
  /// Source ABI triple, independent of output target selection.
  loomc_string_view_t triple;
  /// Source integer and pointer layout.
  loomc_cxx_data_model_t data_model;
  /// Default floating-point transformation policy.
  loomc_cxx_fp_model_t fp_model;
  /// Include delivery flags.
  loomc_cxx_import_flags_t flags;
  /// Optional include provider; embedded headers are resolved separately.
  loomc_cxx_source_provider_t source_provider;
  /// Optional mapping from physical paths to retained logical identities.
  loomc_cxx_source_path_mapper_t source_path_mapper;
  /// User include directories, searched after quoted local includes.
  const loomc_string_view_t* include_paths;
  /// Number of user include directories.
  loomc_host_size_t include_path_count;
  /// System include directories, searched before the embedded system root.
  const loomc_string_view_t* system_include_paths;
  /// Number of system include directories.
  loomc_host_size_t system_include_path_count;
  /// Ordered macro definitions.
  const loomc_cxx_define_t* defines;
  /// Number of macro definitions.
  loomc_host_size_t define_count;
  /// Macro names removed, in order, after the definitions.
  const loomc_string_view_t* undefines;
  /// Number of removed macro names.
  loomc_host_size_t undefine_count;
  /// Qualified source function names to export. Empty exports visible concrete
  /// definitions; reached helpers remain private when roots are explicit.
  const loomc_string_view_t* roots;
  /// Number of explicitly selected roots.
  loomc_host_size_t root_count;
  /// Optional immutable preamble borrowed until import returns. An exact source
  /// prefix or frontend-configuration mismatch uses ordinary cold parsing.
  const loomc_cxx_preamble_t* preamble;

  /// Logical module identity, independent of the physical source path. Empty
  /// uses the source identifier. This does not change include lookup or source
  /// diagnostics. Requires `structure_size` to include this field.
  loomc_string_view_t module_name;
} loomc_cxx_import_options_t;

/// Creates an immutable semantic preamble from leading include directives.
///
/// The representative source must begin with at least one reusable include.
/// The handle copies its matching source prefix, resolved headers, retained
/// misses, and serialized frontend state. Later imports can vary source text
/// after that prefix and import-only policy such as roots or floating-point
/// lowering while retaining the same language, ABI, include-search, define,
/// and embedded-header configuration.
///
/// @param context Immutable context defining diagnostic source retention.
/// @param source Representative source with UNKNOWN format.
/// @param options Source configuration, or NULL for defaults.
/// @param allocator Host allocator owning the returned handle and result.
/// @param out_preamble Receives one owned preamble after successful admission.
/// @param out_result Receives one owned result with retained diagnostics.
/// @return OK for successful creation or diagnosed source rejection. Rejection
/// returns a failed result and NULL preamble. A source without a leading
/// include returns FAILED_PRECONDITION with NULL outputs.
///
/// @ownership
/// Release outputs with loomc_cxx_preamble_release and loomc_result_release.
/// The caller owns catalog lookup, caching, and eviction policy.
///
/// @lifetime
/// Source/include bytes, options, and callback state may be released on return.
/// The returned handle is process-local and has no persisted wire format.
///
/// @thread_safety
/// The immutable handle may be retained, released, and used by concurrent
/// imports. The source provider and path mapper follow their documented
/// per-invocation synchronization requirements during creation.
LOOMC_API_EXPORT loomc_status_t loomc_cxx_preamble_create(
    loomc_context_t* context, const loomc_source_t* source,
    const loomc_cxx_import_options_t* options, loomc_allocator_t allocator,
    loomc_cxx_preamble_t** out_preamble, loomc_result_t** out_result);

/// Retains `preamble` for another owner. Passing NULL is allowed.
///
/// @param preamble Preamble to retain, or NULL.
///
/// Retain/release operations are safe to perform from multiple threads.
LOOMC_API_EXPORT void loomc_cxx_preamble_retain(loomc_cxx_preamble_t* preamble);

/// Releases `preamble` from one owner. Passing NULL is allowed.
///
/// @param preamble Preamble to release, or NULL.
///
/// The immutable preamble is destroyed after its final reference is released.
LOOMC_API_EXPORT void loomc_cxx_preamble_release(
    loomc_cxx_preamble_t* preamble);

/// Imports C/C++ source into a verified High IR module, without cleanup passes.
///
/// @param context Immutable context containing the core dialects.
/// @param workspace Workspace backing the returned module's arena storage.
/// @param source Source with UNKNOWN format. Its identifier labels diagnostics
/// and supplies the main file's path for quoted include lookup.
/// @param options Source configuration, or NULL for defaults.
/// @param allocator Host allocator for public handles and temporary adapters.
/// @param out_module Receives one owned module on successful source admission.
/// @param out_result Receives one owned result with retained diagnostics.
/// @return OK for successful import or diagnosed source rejection. Rejection
/// returns a failed result and NULL module. Infrastructure/API failures return
/// non-OK status and NULL outputs. C++ exceptions never cross this boundary.
///
/// @ownership
/// Release outputs with loomc_module_release and loomc_result_release. The
/// module retains its context/workspace. The result owns diagnostic contents.
///
/// @lifetime
/// Source/include bytes, options, and callback state may be released on return.
/// Neither the module nor the result retains borrowed frontend or AST storage.
///
/// @thread_safety
/// Each import requires exclusive access to its workspace. Concurrent imports
/// may share an immutable context and immutable source handles.
LOOMC_API_EXPORT loomc_status_t loomc_module_import_cxx(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_source_t* source, const loomc_cxx_import_options_t* options,
    loomc_allocator_t allocator, loomc_module_t** out_module,
    loomc_result_t** out_result);

/// One preprocessed and type-checked C/C++ translation unit.
///
/// A unit lets several consumers share one parse of a source: a single-source
/// compiler driver generates host code from the unit's frontend state
/// (loom/import/cxx/unit.h, through the native interop of a statically linked
/// tool) and imports its device code with loomc_module_import_cxx_unit. A unit
/// is mutable: consumers may complete template instantiations in it.
typedef struct loomc_cxx_unit_t loomc_cxx_unit_t;

/// Preprocesses and checks one C/C++ translation unit without importing it.
///
/// @param context Immutable context defining diagnostic source retention.
/// @param source Source with UNKNOWN format. Its identifier labels diagnostics
/// and supplies the main file's path for quoted include lookup.
/// @param options Source configuration, or NULL for defaults: standard,
/// triple, data model, include paths, defines, source provider, path mapper,
/// preamble, and the HOSTED and NO_BUILTIN_INCLUDES flags.
/// @param allocator Host allocator owning the result.
/// @param out_unit Receives one owned unit after successful admission.
/// @param out_result Receives one owned result with retained diagnostics.
/// @return OK for successful parsing or diagnosed source rejection. Rejection
/// returns a failed result and NULL unit. Infrastructure/API failures return
/// non-OK status and NULL outputs.
///
/// @ownership
/// Release outputs with loomc_cxx_unit_release and loomc_result_release.
///
/// @lifetime
/// Source/include bytes, options, and callback state may be released on return:
/// the source provider is consulted only while preprocessing.
///
/// @thread_safety
/// A unit is used by one thread at a time.
LOOMC_API_EXPORT loomc_status_t loomc_cxx_unit_parse(
    loomc_context_t* context, const loomc_source_t* source,
    const loomc_cxx_import_options_t* options, loomc_allocator_t allocator,
    loomc_cxx_unit_t** out_unit, loomc_result_t** out_result);

/// Destroys `unit` and its frontend state. Passing NULL is allowed.
///
/// @param unit Unit to destroy, or NULL.
LOOMC_API_EXPORT void loomc_cxx_unit_release(loomc_cxx_unit_t* unit);

/// Imports a parsed C/C++ translation unit into a verified High IR module,
/// without cleanup passes, as loomc_module_import_cxx does after parsing.
///
/// @param context Immutable context containing the core dialects.
/// @param workspace Workspace backing the returned module's arena storage.
/// @param unit Unit from loomc_cxx_unit_parse. It is not reparsed.
/// @param options Import policy, or NULL for defaults: roots, floating-point
/// model and path mapper. Source configuration
/// fields were fixed when the unit was parsed and are ignored.
/// @param allocator Host allocator for public handles and temporary adapters.
/// @param out_module Receives one owned module on successful translation.
/// @param out_result Receives one owned result with retained diagnostics.
/// @return OK for successful import or diagnosed rejection, as
/// loomc_module_import_cxx. A unit can be imported again.
///
/// @ownership
/// Release outputs with loomc_module_release and loomc_result_release.
///
/// @thread_safety
/// Requires exclusive access to the unit and the workspace.
LOOMC_API_EXPORT loomc_status_t loomc_module_import_cxx_unit(
    loomc_context_t* context, loomc_workspace_t* workspace,
    loomc_cxx_unit_t* unit, const loomc_cxx_import_options_t* options,
    loomc_allocator_t allocator, loomc_module_t** out_module,
    loomc_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_IMPORT_CXX_H_
