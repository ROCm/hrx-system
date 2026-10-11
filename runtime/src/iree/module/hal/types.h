// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULE_HAL_TYPES_H_
#define IREE_MODULE_HAL_TYPES_H_

#include "iree/base/api.h"
#include "iree/hal/buffer.h"
#include "iree/hal/buffer_view.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/device_group.h"
#include "iree/hal/pool.h"
#include "iree/vm/environment.h"
#include "iree/vm/variant.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Append-only ordinals in the native "hal" reference family. These are provider
// ordinals, independent of a bytecode image's sorted reference type table.
enum iree_hal_module_type_ordinal_e {
  // Exact HAL buffer, including its allocation range and access rights.
  IREE_HAL_MODULE_TYPE_BUFFER = 0,
  // Exact HAL buffer view, retaining its buffer and shape/element metadata.
  IREE_HAL_MODULE_TYPE_BUFFER_VIEW = 1,
  // Device group and its immutable topology.
  IREE_HAL_MODULE_TYPE_DEVICE_GROUP = 2,
  // Queue with an owning device-group anchor.
  IREE_HAL_MODULE_TYPE_QUEUE = 3,
  // Command buffer with an owning device-group anchor.
  IREE_HAL_MODULE_TYPE_COMMAND_BUFFER = 4,
  // Semaphore with an owning device-group anchor.
  IREE_HAL_MODULE_TYPE_SEMAPHORE = 5,
  // Pool with an owning device-group anchor.
  IREE_HAL_MODULE_TYPE_POOL = 6,
  // Number of HAL types known by this consumer API version.
  IREE_HAL_MODULE_TYPE_COUNT = 7,
};
typedef uint32_t iree_hal_module_type_ordinal_t;

// Consumer-owned resolved prefix of the native "hal" family. Fields remain
// dense and in append-only ordinal order. Descriptors, strings and finalizer
// code are borrowed from the provider scope, which must outlive every module,
// reference and variant using them, including outputs escaped from a process.
typedef struct iree_hal_module_types_t {
  // Canonical descriptor for an actual iree_hal_buffer_t pointer.
  iree_vm_ref_type_t buffer;
  // Canonical descriptor for an actual iree_hal_buffer_view_t pointer.
  iree_vm_ref_type_t buffer_view;
  // Canonical descriptor for an iree_hal_vm_device_group_t wrapper.
  iree_vm_ref_type_t device_group;
  // Canonical descriptor for an iree_hal_vm_queue_t wrapper.
  iree_vm_ref_type_t queue;
  // Canonical descriptor for an iree_hal_vm_command_buffer_t wrapper.
  iree_vm_ref_type_t command_buffer;
  // Canonical descriptor for an iree_hal_vm_semaphore_t wrapper.
  iree_vm_ref_type_t semaphore;
  // Canonical descriptor for an iree_hal_vm_pool_t wrapper.
  iree_vm_ref_type_t pool;
} iree_hal_module_types_t;

static_assert(offsetof(iree_hal_module_types_t, buffer) ==
                  IREE_HAL_MODULE_TYPE_BUFFER * sizeof(iree_vm_ref_type_t),
              "HAL buffer type must remain at ordinal zero");
static_assert(offsetof(iree_hal_module_types_t, buffer_view) ==
                  IREE_HAL_MODULE_TYPE_BUFFER_VIEW * sizeof(iree_vm_ref_type_t),
              "HAL buffer view type must remain at ordinal one");
static_assert(sizeof(iree_hal_module_types_t) ==
                  IREE_HAL_MODULE_TYPE_COUNT * sizeof(iree_vm_ref_type_t),
              "HAL type fields must remain dense");

// Registers this provider's complete canonical "hal" table and returns the
// borrowed table in |out_table|. The hosting executable/library calls this once
// per environment; other consumers resolve the registered table. Duplicate
// registration fails even for the same provider. Failure leaves |out_table|
// untouched. Matches iree_vm_ref_type_table_register_fn_t.
//
// Buffers and views share their HAL intrusive owner count with VM references.
// Other handles use the cold wrappers below to anchor their native parents.
IREE_API_EXPORT iree_status_t
iree_hal_module_register_types(iree_vm_environment_t* environment,
                               const iree_vm_ref_type_table_t** out_table);

// Resolves the prefix known by this consumer. |table| must come from successful
// environment registration or lookup. Newer providers may append types. Failure
// leaves |out_types| untouched. This inline resolver uses the consumer's
// prefix, independently of the provider library version.
static inline iree_status_t iree_hal_module_types_resolve(
    const iree_vm_ref_type_table_t* table, iree_hal_module_types_t* out_types) {
  if (!table || !out_types) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "table and out_types are required");
  }
  if (!iree_string_view_equal(table->namespace_name, IREE_SV("hal")) ||
      table->types.count < IREE_HAL_MODULE_TYPE_COUNT) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL type prefix is unavailable");
  }
  const iree_hal_module_types_t types = {
      iree_vm_ref_type_storage_at(table->types, IREE_HAL_MODULE_TYPE_BUFFER),
      iree_vm_ref_type_storage_at(table->types,
                                  IREE_HAL_MODULE_TYPE_BUFFER_VIEW),
      iree_vm_ref_type_storage_at(table->types,
                                  IREE_HAL_MODULE_TYPE_DEVICE_GROUP),
      iree_vm_ref_type_storage_at(table->types, IREE_HAL_MODULE_TYPE_QUEUE),
      iree_vm_ref_type_storage_at(table->types,
                                  IREE_HAL_MODULE_TYPE_COMMAND_BUFFER),
      iree_vm_ref_type_storage_at(table->types, IREE_HAL_MODULE_TYPE_SEMAPHORE),
      iree_vm_ref_type_storage_at(table->types, IREE_HAL_MODULE_TYPE_POOL),
  };
  if (!iree_string_view_equal(types.buffer->type_name, IREE_SV("buffer")) ||
      !iree_string_view_equal(types.buffer_view->type_name,
                              IREE_SV("buffer_view")) ||
      !iree_string_view_equal(types.device_group->type_name,
                              IREE_SV("device_group")) ||
      !iree_string_view_equal(types.queue->type_name, IREE_SV("queue")) ||
      !iree_string_view_equal(types.command_buffer->type_name,
                              IREE_SV("command_buffer")) ||
      !iree_string_view_equal(types.semaphore->type_name,
                              IREE_SV("semaphore")) ||
      !iree_string_view_equal(types.pool->type_name, IREE_SV("pool"))) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL type ordinal mismatch");
  }
  *out_types = types;
  return iree_ok_status();
}

IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_buffer, iree_hal_module_types_t, buffer,
                             iree_hal_buffer_t)
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_buffer_view, iree_hal_module_types_t,
                             buffer_view, iree_hal_buffer_view_t)

// Cold handles own both the native resource and its parent group. The native
// object is released before the group, so finalizers may access their device.
// These wrappers only protect object ownership: callers still retain pools,
// backing storage and other borrowed native state through pending work.
// Wrapping retains inputs and leaves |out_ref| untouched on failure. The native
// resource must belong to |group|. Neither input may be null. The returned ref
// owns the wrapper and uses this library's canonical registered type table.
#define IREE_HAL_VM_DECLARE_GROUP_RESOURCE(name)                            \
  typedef struct iree_hal_vm_##name##_t {                                   \
    /* Offset-zero VM owner count. */                                       \
    iree_vm_ref_object_t object;                                            \
    /* Owned native resource, released before the group. */                 \
    iree_hal_##name##_t* name;                                              \
    /* Owned parent group anchoring native devices and queue families. */   \
    iree_hal_device_group_t* group;                                         \
    /* Allocator used to free this cold wrapper. */                         \
    iree_allocator_t host_allocator;                                        \
  } iree_hal_vm_##name##_t;                                                 \
  IREE_API_EXPORT iree_status_t iree_hal_vm_##name##_wrap(                  \
      iree_hal_device_group_t* group, iree_hal_##name##_t* name,            \
      iree_allocator_t host_allocator, iree_vm_ref_t* out_ref);             \
  IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_vm_##name, iree_hal_module_types_t, \
                               name, iree_hal_vm_##name##_t)

IREE_HAL_VM_DECLARE_GROUP_RESOURCE(queue)
IREE_HAL_VM_DECLARE_GROUP_RESOURCE(command_buffer)
IREE_HAL_VM_DECLARE_GROUP_RESOURCE(semaphore)
IREE_HAL_VM_DECLARE_GROUP_RESOURCE(pool)
#undef IREE_HAL_VM_DECLARE_GROUP_RESOURCE

typedef struct iree_hal_vm_device_group_t {
  // Offset-zero VM owner count.
  iree_vm_ref_object_t object;
  // Owned native group, released through the native HAL API.
  iree_hal_device_group_t* group;
  // Allocator used to free this cold wrapper.
  iree_allocator_t host_allocator;
} iree_hal_vm_device_group_t;

// Wraps and retains a non-null group. Failure leaves |out_ref| untouched.
IREE_API_EXPORT iree_status_t iree_hal_vm_device_group_wrap(
    iree_hal_device_group_t* group, iree_allocator_t host_allocator,
    iree_vm_ref_t* out_ref);

IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_vm_device_group, iree_hal_module_types_t,
                             device_group, iree_hal_vm_device_group_t)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULE_HAL_TYPES_H_
