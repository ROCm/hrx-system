// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_EXECUTABLE_LOADER_H_
#define IREE_HAL_LOCAL_EXECUTABLE_LOADER_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_executable_import_provider_t
//===----------------------------------------------------------------------===//

enum iree_hal_executable_import_resolution_bits_e {
  // One or more missing optional symbols.
  IREE_HAL_EXECUTABLE_IMPORT_RESOLUTION_MISSING_OPTIONAL = 1u << 0,
  // TODO(benvanik): could add JIT feedback here ("may be slow path" etc) to
  // propagate warnings up.
};
typedef uint32_t iree_hal_executable_import_resolution_t;

typedef iree_status_t(
    IREE_API_PTR* iree_hal_executable_import_provider_resolve_fn_t)(
    void* self, iree_host_size_t count, const char* const* symbol_names,
    void** out_fn_ptrs, void** out_fn_contexts,
    iree_hal_executable_import_resolution_t* out_resolution);

// Interface used to resolve executable imports at load-time.
// This virtualizes some external provider and does not take ownership of the
// instance: callers must ensure that the provider remains valid for the
// lifetime of the executable loader that it is providing for.
typedef struct iree_hal_executable_import_provider_t {
  // TODO(benvanik): version field.
  IREE_API_UNSTABLE

  // User-defined pointer passed to all functions.
  void* self;

  // Resolves |count| imports given |symbol_names| and stores pointers to their
  // implementation in |out_fn_ptrs| and optional contexts in |out_fn_contexts|.
  iree_hal_executable_import_provider_resolve_fn_t resolve;
} iree_hal_executable_import_provider_t;

static inline iree_hal_executable_import_provider_t
iree_hal_executable_import_provider_null(void) {
  iree_hal_executable_import_provider_t provider = {NULL, NULL};
  return provider;
}

// Resolves |count| imports given |symbol_names| and stores pointers to their
// implementation in |out_fn_ptrs| and optional contexts in |out_fn_contexts|.
//
// A symbol name starting with `?` indicates that the symbol is optional and is
// allowed to be resolved to NULL. Such cases will always return OK but set the
// IREE_HAL_EXECUTABLE_IMPORT_RESOLUTION_MISSING_OPTIONAL resolution bit.
//
// Any already resolved function pointers will be skipped and left unmodified.
// When there's only partial availability of required imports any available
// ones will still be populated and NOT_FOUND will is returned. This allows for
// looping over multiple providers to populate what they can and only fails out
// if all providers return NOT_FOUND for a required import.
//
// Symbol names must be sorted alphabetically so if we cared we could use this
// information to more efficiently resolve the symbols from providers (O(n)
// walk vs potential O(nlogn)/O(n^2)).
iree_status_t iree_hal_executable_import_provider_try_resolve(
    const iree_hal_executable_import_provider_t import_provider,
    iree_host_size_t count, const char* const* symbol_names, void** out_fn_ptrs,
    void** out_fn_contexts,
    iree_hal_executable_import_resolution_t* out_resolution);

// Returns true if the import |symbol_name| is optional.
static inline bool iree_hal_executable_import_is_optional(
    const char* symbol_name) {
  // A `?` prefix indicates the symbol is optional and can be NULL.
  // Since the strings are NUL terminated we know there's always 1 char and
  // we can just test that for the prefix.
  return symbol_name ? (symbol_name[0] == '?') : false;
}

//===----------------------------------------------------------------------===//
// iree_hal_executable_loader_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_executable_loader_vtable_t
    iree_hal_executable_loader_vtable_t;

// Interface for compiled executable loader implementations.
// A loader may be as simple as something that resolves function pointers in the
// local executable for statically linked executables or as complex as a custom
// relocatable ELF loader. Loaders are registered and persist for each device
// they are attached to and may keep internal caches or memoize resources shared
// by multiple loaded executables.
//
// Thread-safe - multiple threads may load executables (including the *same*
// executable) simultaneously.
typedef struct iree_hal_executable_loader_t {
  iree_atomic_ref_count_t ref_count;
  const iree_hal_executable_loader_vtable_t* vtable;
  iree_hal_executable_import_provider_t import_provider;
} iree_hal_executable_loader_t;

// Initializes the base iree_hal_executable_loader_t type.
// Called by subclasses upon allocating their loader.
void iree_hal_executable_loader_initialize(
    const void* vtable, iree_hal_executable_import_provider_t import_provider,
    iree_hal_executable_loader_t* out_base_loader);

// Retains the given |executable_loader| for the caller.
void iree_hal_executable_loader_retain(
    iree_hal_executable_loader_t* executable_loader);

// Releases the given |executable_loader| from the caller.
void iree_hal_executable_loader_release(
    iree_hal_executable_loader_t* executable_loader);

// Returns true if the loader can load native artifacts for |target|.
//
// This is a pure capability query over a target borrowed from the owning
// device spec. Artifact bytes may still be malformed or incompatible and are
// classified separately.
bool iree_hal_executable_loader_query_target_support(
    iree_hal_executable_loader_t* executable_loader,
    const iree_hal_executable_target_t* target);

// Returns immutable executable facts owned by |executable_loader|.
//
// The returned record borrows storage from |executable_loader| and remains
// valid until the loader is destroyed. Callers that need to store the facts
// beyond the loader lifetime must copy them.
void iree_hal_executable_loader_query_spec(
    iree_hal_executable_loader_t* executable_loader,
    iree_hal_device_executable_spec_t* out_executable_spec);

// Returns true if any loader in the list supports |target|.
bool iree_hal_query_any_executable_loader_target_support(
    iree_host_size_t loader_count, iree_hal_executable_loader_t** loaders,
    const iree_hal_executable_target_t* target);

// Returns true when |executable_loader| claims |load_params->executable_data|
// for |target|.
//
// Claims are status-free routing decisions. A loader claims malformed data
// when its identifying bytes are present so that its load call can return a
// specific diagnostic. A false result means the loader will not inspect or
// load the bytes.
bool iree_hal_executable_loader_claims_executable(
    iree_hal_executable_loader_t* executable_loader,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params);

// Loads a previously claimed executable.
//
// Failures are terminal and must not be used to fall through to another
// loader. Callers must establish target support and byte ownership first.
iree_status_t iree_hal_executable_loader_load(
    iree_hal_executable_loader_t* executable_loader,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params,
    iree_host_size_t worker_capacity, iree_hal_executable_t** out_executable);

// Selects the unique loader claiming |load_params| for |target| and loads it.
//
// Returns NOT_FOUND when no target-compatible loader claims the bytes and
// FAILED_PRECONDITION when multiple loaders claim them. A claimed load failure
// is returned directly without trying another loader.
iree_status_t iree_hal_executable_loader_select_and_load(
    iree_host_size_t loader_count, iree_hal_executable_loader_t** loaders,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params,
    iree_host_size_t worker_capacity, iree_hal_executable_t** out_executable);

//===----------------------------------------------------------------------===//
// iree_hal_executable_loader_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_executable_loader_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_executable_loader_t* executable_loader);

  bool(IREE_API_PTR* query_target_support)(
      iree_hal_executable_loader_t* executable_loader,
      const iree_hal_executable_target_t* target);

  void(IREE_API_PTR* query_spec)(
      iree_hal_executable_loader_t* executable_loader,
      iree_hal_device_executable_spec_t* out_executable_spec);

  bool(IREE_API_PTR* claims_executable)(
      iree_hal_executable_loader_t* executable_loader,
      const iree_hal_executable_target_t* target,
      const iree_hal_executable_load_params_t* load_params);

  iree_status_t(IREE_API_PTR* load)(
      iree_hal_executable_loader_t* executable_loader,
      const iree_hal_executable_target_t* target,
      const iree_hal_executable_load_params_t* load_params,
      iree_host_size_t worker_capacity, iree_hal_executable_t** out_executable);
} iree_hal_executable_loader_vtable_t;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_EXECUTABLE_LOADER_H_
