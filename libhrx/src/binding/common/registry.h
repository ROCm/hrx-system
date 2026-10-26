// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_REGISTRY_H_
#define LIBHRX_SRC_BINDING_COMMON_REGISTRY_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_context_symbol_map_t
    iree_hal_streaming_context_symbol_map_t;
typedef struct iree_hal_streaming_managed_storage_t
    iree_hal_streaming_managed_storage_t;
typedef struct iree_hal_streaming_module_t iree_hal_streaming_module_t;
typedef struct iree_hal_streaming_module_registration_t
    iree_hal_streaming_module_registration_t;
typedef struct iree_hal_streaming_symbol_t iree_hal_streaming_symbol_t;

// Symbol type enumeration.
typedef enum iree_hal_streaming_symbol_type_e {
  IREE_HAL_STREAMING_SYMBOL_TYPE_UNDEFINED = 0,  // Deleted/invalid entry.
  IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION = 1,
  IREE_HAL_STREAMING_SYMBOL_TYPE_GLOBAL = 2,
  IREE_HAL_STREAMING_SYMBOL_TYPE_DATA = 3,
} iree_hal_streaming_symbol_type_t;

//===----------------------------------------------------------------------===//
// Symbol tagging
//===----------------------------------------------------------------------===//
//
// We use pointer tagging to quickly identify symbols returned from our registry
// vs raw device pointers from the driver API. This avoids the slow lookup path
// for device pointers.
//
// We use bits 48-55 (8 bits) which are safe across x86-64, ARM64, and RISC-V:
// - x86-64: non-canonical address bits (must be sign-extended from bit 47)
// - ARM64: top byte ignore (TBI) feature ignores bits 56-63
// - RISC-V: similar to x86-64 canonical addressing
//
// This gives us 8 bits for tagging, which is plenty for our needs.

#define IREE_HAL_STREAMING_SYMBOL_TAG_SHIFT 48
#define IREE_HAL_STREAMING_SYMBOL_TAG_MASK 0x00FF000000000000ULL
#define IREE_HAL_STREAMING_SYMBOL_TAG_VALUE 0x00EE000000000000ULL

// Tags a symbol pointer to mark it as coming from our registry.
static inline iree_hal_streaming_symbol_t* iree_hal_streaming_symbol_tag(
    iree_hal_streaming_symbol_t* symbol) {
  uintptr_t ptr = (uintptr_t)symbol;
  // Clear bits 48-55 and set our tag value.
  ptr = (ptr & 0xFF00FFFFFFFFFFFFULL) | IREE_HAL_STREAMING_SYMBOL_TAG_VALUE;
  return (iree_hal_streaming_symbol_t*)ptr;
}

// Checks if a pointer has our tag.
static inline bool iree_hal_streaming_symbol_has_tag(const void* ptr) {
  return ((uintptr_t)ptr & IREE_HAL_STREAMING_SYMBOL_TAG_MASK) ==
         IREE_HAL_STREAMING_SYMBOL_TAG_VALUE;
}

// Removes tag to get original pointer.
static inline iree_hal_streaming_symbol_t* iree_hal_streaming_symbol_untag(
    const void* ptr) {
  uintptr_t untagged = (uintptr_t)ptr;
  // Clear our tag bits (48-55).
  untagged = untagged & 0xFF00FFFFFFFFFFFFULL;
  // Restore sign extension: if bit 47 is set, set bits 48-63.
  if (untagged & 0x0000800000000000ULL) {
    untagged |= 0xFFFF000000000000ULL;
  }
  return (iree_hal_streaming_symbol_t*)untagged;
}

// Type for tagged symbol pointers to prevent accidental dereferencing.
typedef uintptr_t iree_hal_streaming_tagged_symbol_ptr_t;

//===----------------------------------------------------------------------===//
// Symbol registry
//===----------------------------------------------------------------------===//

// Complete registration information for a symbol.
// This contains all the metadata needed to create device-specific symbols.
// Used for both functions and variables (differentiated by type).
typedef struct iree_hal_streaming_symbol_registration_t {
  // Host-side pointer (function or variable).
  void* host_pointer;
  // Symbol type.
  iree_hal_streaming_symbol_type_t type;
  // Device name (for compilation/lookup).
  // Points directly to the string in the fat binary - must remain valid for
  // the lifetime of the registration.
  const char* device_name;
  // Module registration that owns this symbol.
  iree_hal_streaming_module_registration_t* module;
  union {
    // Function-specific metadata (only valid if type == FUNCTION).
    struct {
      uint32_t thread_limit;
      uint32_t block_dim[3];
      uint32_t grid_dim[3];
      uint32_t shared_size_bytes;
    } function;
    // Variable-specific metadata (only valid if type == GLOBAL/DATA).
    struct {
      // Compiler-owned host pointer slot receiving |managed_storage->data|, or
      // NULL for ordinary globals.
      void** publication_slot;
      // Process-wide host backing shared by every context import, or NULL for
      // ordinary globals. Retained by this registration.
      iree_hal_streaming_managed_storage_t* managed_storage;
      // Logical byte length of the variable.
      size_t size;
      // Required byte alignment of the variable.
      uint32_t alignment;
    } variable;
  } params;
} iree_hal_streaming_symbol_registration_t;

// Module registration tracking registered modules and their symbols.
typedef struct iree_hal_streaming_module_registration_t {
  // Fat binary data pointer (opaque, interpretation depends on platform).
  const void* module_binary;
  // Array of symbol registrations owned by this module.
  iree_hal_streaming_symbol_registration_t* symbols;
  iree_host_size_t symbol_count;
  iree_host_size_t symbol_capacity;
} iree_hal_streaming_module_registration_t;

// Global registry that holds all symbol registrations and manages local
// per-context hash maps.
// Typically one per process, created on demand by HIP bindings.
//
// Thread-safe: modules and symbols can be registered/unregistered from any
// thread.
typedef struct iree_hal_streaming_global_symbol_registry_t {
  iree_allocator_t host_allocator;
  iree_slim_mutex_t mutex;

  // All registered modules (array of pointers for stable addresses).
  iree_hal_streaming_module_registration_t** modules;
  iree_host_size_t module_count;
  iree_host_size_t module_capacity;

  // Linked list of all context maps for notifications.
  iree_hal_streaming_context_symbol_map_t* context_maps_head;
} iree_hal_streaming_global_symbol_registry_t;

// Borrows the process registry, initialized once without heap allocation.
// Registrations survive runtime reinitialization. This registry has static
// storage; only independently allocated registries use the free function.
iree_hal_streaming_global_symbol_registry_t*
iree_hal_streaming_global_symbol_registry(void);

// Allocates a new global symbol registry.
// Callers must manage global lifetime to ensure that we don't mix registries
// from different binding layers.
iree_status_t iree_hal_streaming_global_symbol_registry_allocate(
    iree_allocator_t host_allocator,
    iree_hal_streaming_global_symbol_registry_t** out_registry);

// Frees a global symbol registry.
void iree_hal_streaming_global_symbol_registry_free(
    iree_hal_streaming_global_symbol_registry_t* registry);

// Registers a module binary with the registry.
// Returns an opaque handle that should be passed to unregister.
iree_status_t iree_hal_streaming_global_symbol_registry_register_module(
    iree_hal_streaming_global_symbol_registry_t* registry,
    const void* module_binary,
    iree_hal_streaming_module_registration_t** out_module);

// Unregisters a module and all its symbols.
iree_status_t iree_hal_streaming_global_symbol_registry_unregister_module(
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_hal_streaming_module_registration_t* module);

// Registers a function within a module.
iree_status_t iree_hal_streaming_global_symbol_registry_insert_function(
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_hal_streaming_module_registration_t* module, void* host_function,
    const char* device_name, uint32_t thread_limit, uint32_t shared_size_bytes);

// Registers a global variable within a module.
iree_status_t iree_hal_streaming_global_symbol_registry_insert_variable(
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_hal_streaming_module_registration_t* module, void* host_variable,
    const char* device_name, size_t size, uint32_t alignment);

// Registers a managed global variable within a module.
iree_status_t iree_hal_streaming_global_symbol_registry_insert_managed_variable(
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_hal_streaming_module_registration_t* module, void* host_variable,
    void** publication_slot, const char* device_name, size_t size,
    uint32_t alignment);

// Looks up the registration type for a host-side variable pointer.
bool iree_hal_streaming_global_symbol_registry_query_variable(
    iree_hal_streaming_global_symbol_registry_t* registry, void* host_variable,
    iree_hal_streaming_symbol_type_t* out_type, size_t* out_size);

// Initializes a context-specific symbol map.
// It will be registered with the given global |registry| until it is
// deinitialized.
iree_status_t iree_hal_streaming_context_symbol_map_initialize(
    iree_hal_streaming_context_t* context, iree_host_size_t initial_capacity,
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_allocator_t host_allocator,
    iree_hal_streaming_context_symbol_map_t* out_map);

// Deinitializes a context symbol map.
void iree_hal_streaming_context_symbol_map_deinitialize(
    iree_hal_streaming_context_symbol_map_t* map);

// Looks up a symbol in the context map.
// If not found:
// - Checks global registry for registration
// - Loads the module executable into the context
// - Inserts all symbols from the module into the context map
// Returns NOT_FOUND if the host pointer has no live registration.
iree_status_t iree_hal_streaming_context_symbol_map_lookup(
    iree_hal_streaming_context_symbol_map_t* map, void* host_pointer,
    iree_hal_streaming_symbol_t** out_symbol,
    iree_hal_streaming_module_t** out_module);

// Tracks a module loaded into a context symbol map.
typedef struct iree_hal_streaming_context_module_entry_t {
  // Module registration used for lazy-load identity, or NULL once retired.
  iree_hal_streaming_module_registration_t* registration;
  // Compiled module retained until the context is destroyed.
  iree_hal_streaming_module_t* module;
  // Linked list pointers.
  struct iree_hal_streaming_context_module_entry_t* next;
} iree_hal_streaming_context_module_entry_t;

typedef struct iree_hal_streaming_context_symbol_entry_t {
  // Host pointer key used by generated HIP registration code.
  void* key;
  // Compiled symbol associated with the registration key.
  iree_hal_streaming_symbol_t* symbol;
} iree_hal_streaming_context_symbol_entry_t;

// Per-context cache of compiled symbols shared by all threads using the
// context. Mutations include lazy module loading and table growth.
typedef struct iree_hal_streaming_context_symbol_map_t {
  // Serializes table access and the loaded-module list.
  iree_slim_mutex_t mutex;
  // Hash table: host pointer -> compiled symbol on the context device.
  iree_hal_streaming_context_symbol_entry_t* entries;
  iree_host_size_t capacity;
  iree_host_size_t count;

  // List of modules loaded into this context.
  iree_hal_streaming_context_module_entry_t* modules;

  // Notification list linkage.
  struct iree_hal_streaming_context_symbol_map_t* next;
  struct iree_hal_streaming_context_symbol_map_t* prev;

  // Associated context (not owned).
  iree_hal_streaming_context_t* context;

  // Global registry the map is tracking.
  iree_hal_streaming_global_symbol_registry_t* registry;

  iree_allocator_t host_allocator;
} iree_hal_streaming_context_symbol_map_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_COMMON_REGISTRY_H_
