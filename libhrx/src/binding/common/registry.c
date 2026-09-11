// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "common/module.h"
#include "iree/base/internal/math.h"
#include "iree/base/threading/call_once.h"
#include "iree/base/threading/mutex.h"

static void iree_hal_streaming_context_symbol_map_expunge_module(
    iree_hal_streaming_context_symbol_map_t* map,
    iree_hal_streaming_module_registration_t* registration);

static iree_status_t iree_hal_streaming_managed_storage_create(
    iree_allocator_t host_allocator, iree_host_size_t size,
    iree_host_size_t alignment,
    iree_hal_streaming_managed_storage_t** out_storage) {
  *out_storage = NULL;
  iree_hal_streaming_managed_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*storage),
                                             (void**)&storage));
  iree_atomic_ref_count_init(&storage->ref_count);
  storage->host_allocator = host_allocator;
  iree_status_t status = iree_allocator_malloc_aligned(
      host_allocator, size, alignment, /*offset=*/0, &storage->data);
  if (iree_status_is_ok(status)) {
    *out_storage = storage;
  } else {
    iree_allocator_free(host_allocator, storage);
  }
  return status;
}

void iree_hal_streaming_managed_storage_retain(
    iree_hal_streaming_managed_storage_t* storage) {
  if (!storage) return;
  iree_atomic_ref_count_inc(&storage->ref_count);
}

void iree_hal_streaming_managed_storage_release(
    iree_hal_streaming_managed_storage_t* storage) {
  if (storage && iree_atomic_ref_count_dec(&storage->ref_count) == 1) {
    iree_allocator_t host_allocator = storage->host_allocator;
    iree_allocator_free_aligned(host_allocator, storage->data);
    iree_allocator_free(host_allocator, storage);
  }
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

#define IREE_HAL_STREAMING_SYMBOL_REGISTRY_DEFAULT_CAPACITY 64
#define IREE_HAL_STREAMING_SYMBOL_MAP_DEFAULT_CAPACITY 16

// Indicates an empty entry (implicitly terminating a chain).
#define IREE_HAL_STREAMING_SYMBOL_MAP_EMPTY_KEY ((void*)0)
// Indicates a deleted entry (linear scan must proceed).
#define IREE_HAL_STREAMING_SYMBOL_MAP_TOMBSTONE_KEY ((void*)1)

static inline bool iree_hal_streaming_symbol_map_is_valid_key(void* key) {
  return key != IREE_HAL_STREAMING_SYMBOL_MAP_EMPTY_KEY &&
         key != IREE_HAL_STREAMING_SYMBOL_MAP_TOMBSTONE_KEY;
}

// Hash function for host pointers. This is a MurmurHash3-style finalizer that
// mixes aligned function pointer bits well enough for the open-addressed table.
static inline uint64_t iree_hal_streaming_symbol_pointer_hash(void* ptr) {
  // Simple hash for pointers - mix bits.
  uint64_t hash = (uint64_t)ptr;
  hash ^= hash >> 33;
  hash *= 0xff51afd7ed558ccdull;
  hash ^= hash >> 33;
  hash *= 0xc4ceb9fe1a85ec53ull;
  hash ^= hash >> 33;
  return hash;
}

//===----------------------------------------------------------------------===//
// Global Symbol Registry
//===----------------------------------------------------------------------===//

// Static global registry instance, lazily initialized.
static iree_hal_streaming_global_symbol_registry_t*
    iree_hal_streaming_global_symbol_registry_ptr = NULL;

// One-time initialization function for the global registry.
static void iree_hal_streaming_initialize_global_registry(void) {
  iree_status_t status = iree_hal_streaming_global_symbol_registry_allocate(
      iree_allocator_system(), &iree_hal_streaming_global_symbol_registry_ptr);
  if (!iree_status_is_ok(status)) {
    // Log error but continue - registry will be NULL.
    iree_status_fprint(stderr, status);
    iree_status_free(status);
  }
}

iree_hal_streaming_global_symbol_registry_t*
iree_hal_streaming_global_symbol_registry(void) {
  static iree_once_flag once = IREE_ONCE_FLAG_INIT;
  iree_call_once(&once, iree_hal_streaming_initialize_global_registry);
  IREE_ASSERT(iree_hal_streaming_global_symbol_registry_ptr);
  return iree_hal_streaming_global_symbol_registry_ptr;
}

iree_status_t iree_hal_streaming_global_symbol_registry_allocate(
    iree_allocator_t host_allocator,
    iree_hal_streaming_global_symbol_registry_t** out_registry) {
  IREE_ASSERT_ARGUMENT(out_registry);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_registry = NULL;

  // Allocate registry.
  iree_hal_streaming_global_symbol_registry_t* registry = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*registry),
                                (void**)&registry));
  registry->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&registry->mutex);

  // Allocate initial module pointer array.
  registry->module_capacity = 16;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, registry->module_capacity * sizeof(registry->modules[0]),
      (void**)&registry->modules);

  if (iree_status_is_ok(status)) {
    *out_registry = registry;
  } else {
    iree_hal_streaming_global_symbol_registry_free(registry);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_streaming_global_symbol_registry_free(
    iree_hal_streaming_global_symbol_registry_t* registry) {
  if (!registry) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = registry->host_allocator;

  // Free module registrations and their symbols.
  for (iree_host_size_t i = 0; i < registry->module_count; ++i) {
    if (registry->modules[i]) {
      for (iree_host_size_t j = 0; j < registry->modules[i]->symbol_count;
           ++j) {
        iree_hal_streaming_symbol_registration_t* symbol =
            &registry->modules[i]->symbols[j];
        if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_DATA) continue;
        iree_hal_streaming_managed_storage_t* managed_storage =
            symbol->params.variable.managed_storage;
        if (managed_storage && symbol->params.variable.publication_slot &&
            *symbol->params.variable.publication_slot ==
                managed_storage->data) {
          *symbol->params.variable.publication_slot = symbol->host_pointer;
        }
        iree_hal_streaming_managed_storage_release(managed_storage);
      }
      iree_allocator_free(host_allocator, registry->modules[i]->symbols);
      iree_allocator_free(host_allocator, registry->modules[i]);
    }
  }
  iree_allocator_free(host_allocator, registry->modules);

  iree_slim_mutex_deinitialize(&registry->mutex);
  iree_allocator_free(host_allocator, registry);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_streaming_global_symbol_registry_grow_unsafe(
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_host_size_t new_capacity) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t allocation_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          new_capacity, sizeof(registry->modules[0]), &allocation_size))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "global symbol registry size overflow");
  }
  iree_hal_streaming_module_registration_t** new_modules = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(registry->host_allocator, allocation_size,
                                (void**)&new_modules));
  memcpy(new_modules, registry->modules,
         registry->module_count * sizeof(new_modules[0]));
  iree_allocator_free(registry->host_allocator, registry->modules);

  registry->modules = new_modules;
  registry->module_capacity = new_capacity;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_global_symbol_registry_register_module(
    iree_hal_streaming_global_symbol_registry_t* registry,
    const void* module_binary,
    iree_hal_streaming_module_registration_t** out_module) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(out_module);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_module = NULL;

  iree_slim_mutex_lock(&registry->mutex);

  // Grow the module table if needed.
  iree_status_t status = iree_ok_status();
  if (registry->module_count >= registry->module_capacity) {
    iree_host_size_t new_capacity = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(registry->module_capacity, 2,
                                                  &new_capacity))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "global symbol registry capacity overflow");
    } else {
      status = iree_hal_streaming_global_symbol_registry_grow_unsafe(
          registry, new_capacity);
    }
  }

  // Allocate a new module registration dynamically.
  iree_hal_streaming_module_registration_t* module = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(registry->host_allocator, sizeof(*module),
                                   (void**)&module);
  }

  if (iree_status_is_ok(status)) {
    module->module_binary = module_binary;

    // Allocate initial symbols array.
    module->symbol_capacity = 32;
    status = iree_allocator_malloc(
        registry->host_allocator,
        module->symbol_capacity * sizeof(module->symbols[0]),
        (void**)&module->symbols);
  }

  if (iree_status_is_ok(status)) {
    registry->modules[registry->module_count++] = module;
    *out_module = module;
  } else {
    if (module) {
      iree_allocator_free(registry->host_allocator, module->symbols);
      iree_allocator_free(registry->host_allocator, module);
    }
  }

  iree_slim_mutex_unlock(&registry->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_global_symbol_registry_unregister_module(
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_hal_streaming_module_registration_t* module) {
  IREE_ASSERT_ARGUMENT(registry);
  if (!module) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&registry->mutex);

  // Find the module in the list.
  iree_host_size_t module_index = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < registry->module_count; ++i) {
    if (registry->modules[i] == module) {
      module_index = i;
      break;
    }
  }
  if (module_index == IREE_HOST_SIZE_MAX) {
    iree_slim_mutex_unlock(&registry->mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_NOT_FOUND, "module not registered");
  }

  // Notify all context maps to remove symbols from this module.
  iree_hal_streaming_context_symbol_map_t* context_map =
      registry->context_maps_head;
  while (context_map) {
    iree_hal_streaming_context_symbol_map_expunge_module(context_map, module);
    context_map = context_map->next;
  }

  // Context modules have released their imports, so the shared host backing is
  // no longer visible to a device. Restore the compiler slot before freeing it.
  for (iree_host_size_t i = 0; i < module->symbol_count; ++i) {
    iree_hal_streaming_symbol_registration_t* symbol = &module->symbols[i];
    if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_DATA) continue;
    iree_hal_streaming_managed_storage_t* managed_storage =
        symbol->params.variable.managed_storage;
    if (managed_storage && symbol->params.variable.publication_slot &&
        *symbol->params.variable.publication_slot == managed_storage->data) {
      *symbol->params.variable.publication_slot = symbol->host_pointer;
    }
    iree_hal_streaming_managed_storage_release(managed_storage);
  }

  // Free the module registration.
  iree_allocator_free(registry->host_allocator, module->symbols);
  iree_allocator_free(registry->host_allocator, module);

  // Remove the module pointer by shifting remaining pointers in the list.
  if (module_index < registry->module_count - 1) {
    memmove(&registry->modules[module_index],
            &registry->modules[module_index + 1],
            (registry->module_count - module_index - 1) *
                sizeof(registry->modules[0]));
  }
  --registry->module_count;

  iree_slim_mutex_unlock(&registry->mutex);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_module_registration_grow_unsafe(
    iree_hal_streaming_module_registration_t* module,
    iree_host_size_t new_capacity, iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t allocation_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          new_capacity, sizeof(module->symbols[0]), &allocation_size))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "module symbol registry size overflow");
  }
  iree_hal_streaming_symbol_registration_t* new_symbols = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, allocation_size,
                                (void**)&new_symbols));
  memcpy(new_symbols, module->symbols,
         module->symbol_count * sizeof(module->symbols[0]));
  iree_allocator_free(host_allocator, module->symbols);
  module->symbols = new_symbols;
  module->symbol_capacity = new_capacity;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_global_symbol_registry_insert_function(
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_hal_streaming_module_registration_t* module, void* host_function,
    const char* device_name, uint32_t thread_limit,
    uint32_t shared_size_bytes) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(host_function);
  IREE_ASSERT_ARGUMENT(device_name);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&registry->mutex);

  // Check if we need to grow the module's symbols array.
  iree_status_t status = iree_ok_status();
  if (module->symbol_count >= module->symbol_capacity) {
    iree_host_size_t new_capacity = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(module->symbol_capacity, 2,
                                                  &new_capacity))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "module symbol registry capacity overflow");
    } else {
      status = iree_hal_streaming_module_registration_grow_unsafe(
          module, new_capacity, registry->host_allocator);
    }
  }

  if (iree_status_is_ok(status)) {
    // Add symbol to module's symbols array.
    iree_hal_streaming_symbol_registration_t* symbol =
        &module->symbols[module->symbol_count++];

    // Fill in registration.
    symbol->host_pointer = host_function;
    symbol->type = IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION;
    symbol->device_name = device_name;  // direct pointer, no copy
    symbol->module = module;
    symbol->params.function.thread_limit = thread_limit;
    symbol->params.function.shared_size_bytes = shared_size_bytes;
  }

  iree_slim_mutex_unlock(&registry->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t
iree_hal_streaming_global_symbol_registry_insert_variable_with_type(
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_hal_streaming_module_registration_t* module, void* host_variable,
    const char* device_name, size_t size, uint32_t alignment,
    iree_hal_streaming_symbol_type_t symbol_type, void** publication_slot) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(host_variable);
  IREE_ASSERT_ARGUMENT(device_name);
  IREE_ASSERT_ARGUMENT(symbol_type == IREE_HAL_STREAMING_SYMBOL_TYPE_GLOBAL ||
                       symbol_type == IREE_HAL_STREAMING_SYMBOL_TYPE_DATA);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&registry->mutex);

  // Check if we need to grow the module's symbols array.
  iree_status_t status = iree_ok_status();
  if (module->symbol_count >= module->symbol_capacity) {
    iree_host_size_t new_capacity = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(module->symbol_capacity, 2,
                                                  &new_capacity))) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "module symbol registry capacity overflow");
    } else {
      status = iree_hal_streaming_module_registration_grow_unsafe(
          module, new_capacity, registry->host_allocator);
    }
  }

  iree_hal_streaming_managed_storage_t* managed_storage = NULL;
  if (iree_status_is_ok(status) &&
      symbol_type == IREE_HAL_STREAMING_SYMBOL_TYPE_DATA) {
    if (IREE_UNLIKELY(!publication_slot || size == 0)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "managed variable requires a publication slot and nonzero size");
    } else if (IREE_UNLIKELY(alignment != 0 &&
                             !iree_host_size_is_power_of_two(
                                 (iree_host_size_t)alignment))) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "managed variable alignment must be a power of "
                                "two");
    }
    const iree_host_size_t allocation_size =
        iree_max((iree_host_size_t)size, (iree_host_size_t)8);
    const iree_host_size_t allocation_alignment =
        iree_max((iree_host_size_t)alignment, (iree_host_size_t)4096);
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_managed_storage_create(
          registry->host_allocator, allocation_size, allocation_alignment,
          &managed_storage);
    }
    if (iree_status_is_ok(status)) {
      memcpy(managed_storage->data, host_variable, size);
    }
  }

  if (iree_status_is_ok(status)) {
    // Add symbol to module's symbols array.
    iree_hal_streaming_symbol_registration_t* symbol =
        &module->symbols[module->symbol_count++];

    // Fill in registration (device_name points directly to fat binary string).
    symbol->host_pointer = host_variable;
    symbol->type = symbol_type;
    symbol->device_name = device_name;  // direct pointer, no copy
    symbol->module = module;
    symbol->params.variable.publication_slot = publication_slot;
    symbol->params.variable.managed_storage = managed_storage;
    symbol->params.variable.size = size;
    symbol->params.variable.alignment = alignment;
    managed_storage = NULL;
    if (publication_slot)
      *publication_slot = symbol->params.variable.managed_storage->data;
  }
  iree_hal_streaming_managed_storage_release(managed_storage);

  iree_slim_mutex_unlock(&registry->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_global_symbol_registry_insert_variable(
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_hal_streaming_module_registration_t* module, void* host_variable,
    const char* device_name, size_t size, uint32_t alignment) {
  return iree_hal_streaming_global_symbol_registry_insert_variable_with_type(
      registry, module, host_variable, device_name, size, alignment,
      IREE_HAL_STREAMING_SYMBOL_TYPE_GLOBAL, /*publication_slot=*/NULL);
}

iree_status_t iree_hal_streaming_global_symbol_registry_insert_managed_variable(
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_hal_streaming_module_registration_t* module, void* host_variable,
    void** publication_slot, const char* device_name, size_t size,
    uint32_t alignment) {
  IREE_ASSERT_ARGUMENT(publication_slot);
  return iree_hal_streaming_global_symbol_registry_insert_variable_with_type(
      registry, module, host_variable, device_name, size, alignment,
      IREE_HAL_STREAMING_SYMBOL_TYPE_DATA, publication_slot);
}

bool iree_hal_streaming_global_symbol_registry_query_variable(
    iree_hal_streaming_global_symbol_registry_t* registry, void* host_variable,
    iree_hal_streaming_symbol_type_t* out_type, size_t* out_size) {
  if (out_type) *out_type = IREE_HAL_STREAMING_SYMBOL_TYPE_UNDEFINED;
  if (out_size) *out_size = 0;
  if (!registry || !host_variable) return false;

  bool found = false;
  iree_slim_mutex_lock(&registry->mutex);
  for (iree_host_size_t i = 0; i < registry->module_count && !found; ++i) {
    iree_hal_streaming_module_registration_t* module = registry->modules[i];
    if (!module) continue;
    for (iree_host_size_t j = 0; j < module->symbol_count; ++j) {
      const iree_hal_streaming_symbol_registration_t* symbol =
          &module->symbols[j];
      const iree_hal_streaming_managed_storage_t* managed_storage =
          symbol->params.variable.managed_storage;
      const bool is_managed_storage =
          symbol->type == IREE_HAL_STREAMING_SYMBOL_TYPE_DATA &&
          managed_storage && managed_storage->data == host_variable;
      if ((symbol->host_pointer != host_variable && !is_managed_storage) ||
          (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_GLOBAL &&
           symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_DATA)) {
        continue;
      }
      if (out_type) *out_type = symbol->type;
      if (out_size) *out_size = symbol->params.variable.size;
      found = true;
      break;
    }
  }
  iree_slim_mutex_unlock(&registry->mutex);
  return found;
}

// Slowly looks up a registration by host pointer.
// We assume that this only ever happens on a context map miss and that we have
// very few of those after startup.
// Returns NULL if not found.
// The caller must hold |registry->mutex|.
static const iree_hal_streaming_symbol_registration_t*
iree_hal_streaming_global_symbol_registry_lookup_unsafe(
    iree_hal_streaming_global_symbol_registry_t* registry, void* host_pointer) {
  if (!registry || !host_pointer) return NULL;

  // Linear scan through all modules and their symbols.
  const iree_hal_streaming_symbol_registration_t* result = NULL;
  for (iree_host_size_t i = 0; i < registry->module_count; ++i) {
    iree_hal_streaming_module_registration_t* module = registry->modules[i];
    if (!module) continue;
    for (iree_host_size_t j = 0; j < module->symbol_count; ++j) {
      const iree_hal_streaming_symbol_registration_t* symbol =
          &module->symbols[j];
      const iree_hal_streaming_managed_storage_t* managed_storage =
          symbol->params.variable.managed_storage;
      const bool is_managed_storage =
          symbol->type == IREE_HAL_STREAMING_SYMBOL_TYPE_DATA &&
          managed_storage && managed_storage->data == host_pointer;
      if (symbol->host_pointer == host_pointer || is_managed_storage) {
        result = &module->symbols[j];
        break;
      }
    }
    if (result) break;
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Context Symbol Map
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_context_symbol_map_initialize(
    iree_hal_streaming_context_t* context, iree_host_size_t initial_capacity,
    iree_hal_streaming_global_symbol_registry_t* registry,
    iree_allocator_t host_allocator,
    iree_hal_streaming_context_symbol_map_t* out_map) {
  IREE_ASSERT_ARGUMENT(out_map);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_map, 0, sizeof(*out_map));

  out_map->context = context;
  out_map->host_allocator = host_allocator;

  // Start with a small capacity.
  if (initial_capacity == 0) {
    initial_capacity = IREE_HAL_STREAMING_SYMBOL_MAP_DEFAULT_CAPACITY;
  }

  // Round capacity up to the next power of 2 (if not already).
  out_map->capacity =
      (iree_host_size_t)iree_math_round_up_to_pow2_u64(initial_capacity);
  iree_host_size_t entries_size = 0;
  if (IREE_UNLIKELY(out_map->capacity == 0 ||
                    !iree_host_size_checked_mul(out_map->capacity,
                                                sizeof(out_map->entries[0]),
                                                &entries_size))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "symbol map allocation size overflow");
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, entries_size,
                                (void**)&out_map->entries));
  memset(out_map->entries, 0, entries_size);
  iree_slim_mutex_initialize(&out_map->mutex);

  // Register with the global registry (so we can listen for notifications).
  iree_slim_mutex_lock(&registry->mutex);
  out_map->registry = registry;
  out_map->next = registry->context_maps_head;
  if (registry->context_maps_head) {
    registry->context_maps_head->prev = out_map;
  }
  registry->context_maps_head = out_map;
  iree_slim_mutex_unlock(&registry->mutex);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_streaming_context_symbol_map_deinitialize(
    iree_hal_streaming_context_symbol_map_t* map) {
  if (!map || !map->entries) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = map->host_allocator;
  iree_hal_streaming_global_symbol_registry_t* registry = map->registry;

  // Unlink before releasing map-owned state. The registry lock excludes module
  // unregistration while the map lock drains active lookups.
  if (registry) iree_slim_mutex_lock(&registry->mutex);
  iree_slim_mutex_lock(&map->mutex);

  if (registry) {
    if (map->prev) {
      map->prev->next = map->next;
    } else if (registry->context_maps_head == map) {
      registry->context_maps_head = map->next;
    }
    if (map->next) map->next->prev = map->prev;
  }

  // Release all loaded modules.
  iree_hal_streaming_context_module_entry_t* module_entry = map->modules;
  while (module_entry) {
    iree_hal_streaming_context_module_entry_t* next = module_entry->next;
    iree_hal_streaming_module_release(module_entry->module);
    iree_allocator_free(host_allocator, module_entry);
    module_entry = next;
  }

  iree_allocator_free(host_allocator, map->entries);
  map->entries = NULL;
  iree_slim_mutex_unlock(&map->mutex);
  if (registry) iree_slim_mutex_unlock(&registry->mutex);
  iree_slim_mutex_deinitialize(&map->mutex);

  IREE_TRACE_ZONE_END(z0);
}

// Grows the hash table to accommodate at least the specified capacity.
// Rehashes all existing entries into the new table.
static iree_status_t iree_hal_streaming_context_symbol_map_grow(
    iree_hal_streaming_context_symbol_map_t* map,
    iree_host_size_t new_min_capacity) {
  IREE_ASSERT_ARGUMENT(map);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Round up to the next power of 2 for optimal hash distribution.
  iree_host_size_t new_capacity =
      (iree_host_size_t)iree_math_round_up_to_pow2_u64(new_min_capacity);
  if (new_capacity <= map->capacity) {
    if (IREE_UNLIKELY(
            !iree_host_size_checked_mul(map->capacity, 2, &new_capacity))) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "context symbol map capacity overflow");
    }
  }

  // Allocate the new table.
  iree_host_size_t allocation_size = 0;
  if (IREE_UNLIKELY(new_capacity == 0 ||
                    !iree_host_size_checked_mul(new_capacity,
                                                sizeof(map->entries[0]),
                                                &allocation_size))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "context symbol map size overflow");
  }
  iree_hal_streaming_context_symbol_entry_t* new_entries = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(map->host_allocator, allocation_size,
                                (void**)&new_entries));
  memset(new_entries, 0, allocation_size);

  // Rehash all existing entries into the new table.
  for (iree_host_size_t i = 0; i < map->capacity; ++i) {
    void* key = map->entries[i].key;
    if (!iree_hal_streaming_symbol_map_is_valid_key(key)) {
      continue;  // Skip empty and tombstone entries.
    }

    // Find a slot in the new table.
    const uint64_t hash = iree_hal_streaming_symbol_pointer_hash(key);
    iree_host_size_t index = hash & (new_capacity - 1);
    for (iree_host_size_t j = 0; j < new_capacity; ++j) {
      if (new_entries[index].key == IREE_HAL_STREAMING_SYMBOL_MAP_EMPTY_KEY) {
        new_entries[index].key = key;
        new_entries[index].symbol = map->entries[i].symbol;
        break;
      }
      index = (index + 1) & (new_capacity - 1);
    }
  }

  // Free the old table and swap with the new one.
  iree_allocator_free(map->host_allocator, map->entries);
  map->entries = new_entries;
  map->capacity = new_capacity;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_context_symbol_map_insert(
    iree_hal_streaming_context_symbol_map_t* map, void* host_pointer,
    iree_hal_streaming_symbol_t* symbol) {
  if (IREE_UNLIKELY(
          !iree_hal_streaming_symbol_map_is_valid_key(host_pointer))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "registered symbol has an invalid host pointer");
  }

  const uint64_t hash = iree_hal_streaming_symbol_pointer_hash(host_pointer);
  iree_host_size_t index = hash & (map->capacity - 1);
  for (iree_host_size_t i = 0; i < map->capacity; ++i) {
    iree_hal_streaming_context_symbol_entry_t* entry = &map->entries[index];
    if (entry->key == host_pointer && entry->symbol == symbol) {
      return iree_ok_status();
    }
    if (entry->key == IREE_HAL_STREAMING_SYMBOL_MAP_EMPTY_KEY ||
        entry->key == IREE_HAL_STREAMING_SYMBOL_MAP_TOMBSTONE_KEY) {
      entry->key = host_pointer;
      entry->symbol = symbol;
      ++map->count;
      return iree_ok_status();
    }
    index = (index + 1) & (map->capacity - 1);
  }
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "symbol map has no free entries");
}

// Loads a module and populates the context symbol map with its symbols.
// This is called when a symbol is requested for use in a context but the module
// hasn't been instantiated in it yet.
static iree_status_t iree_hal_streaming_context_symbol_map_prepare_module(
    iree_hal_streaming_context_symbol_map_t* map,
    iree_hal_streaming_module_registration_t* registration) {
  IREE_ASSERT_ARGUMENT(map);
  IREE_ASSERT_ARGUMENT(registration);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Check if we've already loaded this module.
  for (iree_hal_streaming_context_module_entry_t* module_entry = map->modules;
       module_entry != NULL; module_entry = module_entry->next) {
    if (module_entry->registration == registration) {
      // Registrations are immutable after their containing module has finished
      // initialization, so an existing module entry is already complete.
      IREE_TRACE_ZONE_END(z0);
      return iree_ok_status();
    }
  }

  // Managed registrations have a stable allocation distinct from the initial
  // host key. Reserve one extra alias for each such registration.
  iree_host_size_t insertion_count = registration->symbol_count;
  for (iree_host_size_t i = 0; i < registration->symbol_count; ++i) {
    if (registration->symbols[i].type == IREE_HAL_STREAMING_SYMBOL_TYPE_DATA &&
        registration->symbols[i].params.variable.managed_storage &&
        IREE_UNLIKELY(!iree_host_size_checked_add(insertion_count, 1,
                                                  &insertion_count))) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "context symbol count overflow");
    }
  }

  // Grow the hash table to fit our new total count, if needed.
  iree_host_size_t new_count = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(map->count, insertion_count,
                                                &new_count))) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "context symbol count overflow");
  }
  const iree_host_size_t maximum_count = map->capacity - map->capacity / 4;
  if (new_count > maximum_count) {
    // Need to resize the hash table to maintain good load factor.
    // We want to keep the load factor below 75% for good performance.
    const iree_host_size_t growth =
        new_count / 3 + (new_count % 3 != 0 ? 1 : 0);
    iree_host_size_t new_min_capacity = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_add(new_count, growth,
                                                  &new_min_capacity))) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "context symbol map capacity overflow");
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_context_symbol_map_grow(map, new_min_capacity));
  }

  // Allocate tracking entry.
  iree_hal_streaming_context_module_entry_t* entry = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(map->host_allocator, sizeof(*entry),
                                (void**)&entry));
  entry->registration = registration;
  entry->module = NULL;

  // Module not loaded yet - load it from the registered fat binary.
  iree_hal_executable_load_flags_t load_flags =
      IREE_HAL_EXECUTABLE_LOAD_FLAG_ALLOW_OPTIMIZATION;
  iree_const_byte_span_t module_data =
      iree_make_const_byte_span((const uint8_t*)registration->module_binary,
                                /*infer*/ 0);
  iree_status_t status =
      iree_hal_streaming_module_create_from_memory_borrowing_context(
          map->context, load_flags, module_data, map->host_allocator,
          &entry->module);
  if (iree_status_is_ok(status)) {
    // Insert all symbols from the module into the hash table.
    for (iree_host_size_t i = 0;
         iree_status_is_ok(status) && i < registration->symbol_count; ++i) {
      // Get the registered symbol's device name
      iree_string_view_t registered_name =
          iree_make_cstring_view(registration->symbols[i].device_name);
      void* symbol_host_ptr = registration->symbols[i].host_pointer;

      // Find the corresponding compiled symbol in the module by name.
      iree_hal_streaming_symbol_t* symbol = NULL;
      bool found = false;
      switch (registration->symbols[i].type) {
        case IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION:
          for (iree_host_size_t j = 0; j < entry->module->symbol_count; ++j) {
            if (iree_string_view_equal(registered_name,
                                       entry->module->symbols[j].name)) {
              symbol = &entry->module->symbols[j];
              break;
            }
          }
          break;
        case IREE_HAL_STREAMING_SYMBOL_TYPE_GLOBAL:
        case IREE_HAL_STREAMING_SYMBOL_TYPE_DATA:
          status = iree_hal_streaming_module_try_lookup_global_symbol(
              entry->module, registration->symbols[i].device_name, &found,
              &symbol);
          break;
        default:
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "unsupported registered symbol type %d",
                                    registration->symbols[i].type);
          break;
      }
      if (!iree_status_is_ok(status)) {
        break;
      }
      if (symbol && symbol->type != registration->symbols[i].type) {
        if (!(registration->symbols[i].type ==
                  IREE_HAL_STREAMING_SYMBOL_TYPE_DATA &&
              symbol->type == IREE_HAL_STREAMING_SYMBOL_TYPE_GLOBAL)) {
          status = iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "registered symbol `%.*s` type mismatch (expected %d, got %d)",
              (int)registered_name.size, registered_name.data,
              registration->symbols[i].type, symbol->type);
          break;
        }
      }
      if (!symbol) {
        // A fat binary may register stubs for symbols omitted from the selected
        // target image. Leave those stubs unresolved without rejecting the
        // other symbols in the image.
        fprintf(stderr,
                "[WARN] registered symbol `%.*s` not found in module with %zu "
                "symbols\n",
                (int)registered_name.size, registered_name.data,
                entry->module->symbol_count);
        // Skip this symbol and continue with the rest.
        continue;
      }

      iree_hal_streaming_managed_storage_t* managed_storage =
          registration->symbols[i].params.variable.managed_storage;
      if (managed_storage) {
        status = iree_hal_streaming_module_bind_registered_managed_global(
            entry->module, registration->symbols[i].device_name,
            managed_storage, registration->symbols[i].params.variable.size,
            &symbol);
        if (!iree_status_is_ok(status)) break;
      }

      status = iree_hal_streaming_context_symbol_map_insert(
          map, symbol_host_ptr, symbol);
      if (iree_status_is_ok(status) && managed_storage &&
          managed_storage->data != symbol_host_ptr) {
        status = iree_hal_streaming_context_symbol_map_insert(
            map, managed_storage->data, symbol);
      }
    }
  }

  if (iree_status_is_ok(status)) {
    // Link module into listing.
    entry->next = map->modules;
    map->modules = entry;
  } else {
    // Module population is transactional. Remove entries inserted before the
    // failure while their owning compiled module is still alive.
    for (iree_host_size_t i = 0; i < map->capacity; ++i) {
      iree_hal_streaming_context_symbol_entry_t* symbol_entry =
          &map->entries[i];
      if (!iree_hal_streaming_symbol_map_is_valid_key(symbol_entry->key) ||
          !symbol_entry->symbol ||
          symbol_entry->symbol->module != entry->module) {
        continue;
      }
      symbol_entry->key = IREE_HAL_STREAMING_SYMBOL_MAP_TOMBSTONE_KEY;
      symbol_entry->symbol = NULL;
      --map->count;
    }
    iree_hal_streaming_module_release(entry->module);
    iree_allocator_free(map->host_allocator, entry);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Removes all host-pointer mappings for a retired module registration.
// Called when a module is unregistered from the global registry.
// Returns without mutation if the module was never registered.
static void iree_hal_streaming_context_symbol_map_expunge_module(
    iree_hal_streaming_context_symbol_map_t* map,
    iree_hal_streaming_module_registration_t* registration) {
  IREE_ASSERT_ARGUMENT(map);
  IREE_ASSERT_ARGUMENT(registration);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&map->mutex);

  // Find and unlink the module from the loaded modules list.
  iree_hal_streaming_context_module_entry_t** module_link = &map->modules;
  while (*module_link) {
    if ((*module_link)->registration == registration) {
      break;
    }
    module_link = &(*module_link)->next;
  }
  iree_hal_streaming_context_module_entry_t* module_entry = *module_link;
  if (!module_entry) {
    // Module was not loaded in this context, no-op.
    iree_slim_mutex_unlock(&map->mutex);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Remove every key backed by the module, including the initial and published
  // aliases of managed variables. Weak host stubs may be shared by multiple
  // images, so module ownership rather than key identity selects entries.
  for (iree_host_size_t i = 0; i < map->capacity; ++i) {
    iree_hal_streaming_context_symbol_entry_t* entry = &map->entries[i];
    if (!iree_hal_streaming_symbol_map_is_valid_key(entry->key) ||
        !entry->symbol || entry->symbol->module != module_entry->module) {
      continue;
    }
    entry->key = IREE_HAL_STREAMING_SYMBOL_MAP_TOMBSTONE_KEY;
    entry->symbol = NULL;
    --map->count;
  }

  // Lookups and graph nodes retain the module explicitly, while recorded
  // command buffers retain the executable. Remove the cache's ownership now
  // that no new host-pointer lookup can find this module.
  *module_link = module_entry->next;

  iree_slim_mutex_unlock(&map->mutex);
  iree_hal_streaming_module_release(module_entry->module);
  iree_allocator_free(map->host_allocator, module_entry);
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_streaming_context_symbol_map_lookup(
    iree_hal_streaming_context_symbol_map_t* map, void* host_pointer,
    iree_hal_streaming_symbol_t** out_symbol,
    iree_hal_streaming_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(map);
  IREE_ASSERT_ARGUMENT(out_symbol);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_symbol = NULL;
  *out_module = NULL;

  // Check for invalid keys.
  if (!host_pointer ||
      !iree_hal_streaming_symbol_map_is_valid_key(host_pointer)) {
    *out_symbol = NULL;
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid host pointer");
  }

  // Fast path: check the context-local map without touching the process-wide
  // registry.
  const uint64_t hash = iree_hal_streaming_symbol_pointer_hash(host_pointer);
  iree_slim_mutex_lock(&map->mutex);
  iree_host_size_t index = hash & (map->capacity - 1);
  for (iree_host_size_t i = 0; i < map->capacity; ++i) {
    const void* entry_key = map->entries[index].key;
    if (entry_key == host_pointer) {
      *out_symbol = map->entries[index].symbol;
      *out_module = (*out_symbol)->module;
      iree_hal_streaming_module_retain(*out_module);
      iree_slim_mutex_unlock(&map->mutex);
      return iree_ok_status();
    } else if (entry_key == IREE_HAL_STREAMING_SYMBOL_MAP_EMPTY_KEY) {
      break;  // not found in local map
    }
    index = (index + 1) & (map->capacity - 1);  // continue linear probe
  }
  iree_slim_mutex_unlock(&map->mutex);

  // Slow path lock order is registry then map, matching module unregistration.
  // Recheck the map after acquiring both locks because another thread may have
  // loaded the module while this thread was waiting.
  iree_slim_mutex_lock(&map->registry->mutex);
  iree_slim_mutex_lock(&map->mutex);
  index = hash & (map->capacity - 1);
  for (iree_host_size_t i = 0; i < map->capacity; ++i) {
    const void* entry_key = map->entries[index].key;
    if (entry_key == host_pointer) {
      *out_symbol = map->entries[index].symbol;
      *out_module = (*out_symbol)->module;
      iree_hal_streaming_module_retain(*out_module);
      iree_slim_mutex_unlock(&map->mutex);
      iree_slim_mutex_unlock(&map->registry->mutex);
      return iree_ok_status();
    } else if (entry_key == IREE_HAL_STREAMING_SYMBOL_MAP_EMPTY_KEY) {
      break;
    }
    index = (index + 1) & (map->capacity - 1);
  }

  const iree_hal_streaming_symbol_registration_t* registration =
      iree_hal_streaming_global_symbol_registry_lookup_unsafe(map->registry,
                                                              host_pointer);
  if (!registration) {
    *out_symbol = NULL;
    iree_slim_mutex_unlock(&map->mutex);
    iree_slim_mutex_unlock(&map->registry->mutex);
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "host pointer is not a registered symbol");
  }

  // Ensure the module is loaded and its symbols are in the hash table.
  iree_status_t status = iree_hal_streaming_context_symbol_map_prepare_module(
      map, registration->module);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&map->mutex);
    iree_slim_mutex_unlock(&map->registry->mutex);
    return iree_status_annotate_f(
        status, "preparing statically registered module for context");
  }

  // Now look up again in the hash table using the original hash.
  // The symbol should be there now after module preparation.
  // Note: We must recompute the index because the hash table may have been
  // resized during prepare_module.
  index = hash & (map->capacity - 1);
  for (iree_host_size_t i = 0; i < map->capacity; ++i) {
    const void* entry_key = map->entries[index].key;
    if (entry_key == host_pointer) {
      *out_symbol = map->entries[index].symbol;
      *out_module = (*out_symbol)->module;
      iree_hal_streaming_module_retain(*out_module);
      iree_slim_mutex_unlock(&map->mutex);
      iree_slim_mutex_unlock(&map->registry->mutex);
      return iree_ok_status();
    } else if (entry_key == IREE_HAL_STREAMING_SYMBOL_MAP_EMPTY_KEY) {
      break;  // still not found (shouldn't happen)
    }
    index = (index + 1) & (map->capacity - 1);  // continue linear probe
  }

  *out_symbol = NULL;
  iree_slim_mutex_unlock(&map->mutex);
  iree_slim_mutex_unlock(&map->registry->mutex);
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "registered symbol `%s` is absent from the module",
                          registration->device_name);
}
