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
#include "iree/hal/fence.h"
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
  // Exact HAL fence retaining the semaphores of its completion set.
  IREE_HAL_MODULE_TYPE_FENCE = 2,
  // Number of HAL types known by this consumer API version.
  IREE_HAL_MODULE_TYPE_COUNT = 3,
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
  // Canonical descriptor for an actual iree_hal_fence_t pointer.
  iree_vm_ref_type_t fence;
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
static_assert(offsetof(iree_hal_module_types_t, fence) ==
                  IREE_HAL_MODULE_TYPE_FENCE * sizeof(iree_vm_ref_type_t),
              "HAL fence type must remain at ordinal two");

// Registers this provider's complete canonical "hal" table and returns the
// borrowed table in |out_table|. The hosting executable/library calls this once
// per environment; other consumers resolve the registered table. Duplicate
// registration fails even for the same provider. Failure leaves |out_table|
// untouched. Matches iree_vm_ref_type_table_register_fn_t.
//
// Objects share their existing HAL intrusive owner count with VM references;
// there is no wrapper allocation or second owner count. Final buffer release
// follows HAL recycling; views and fences use their normal HAL destruction.
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
      iree_vm_ref_type_storage_at(table->types, IREE_HAL_MODULE_TYPE_FENCE),
  };
  if (!iree_string_view_equal(types.buffer->type_name, IREE_SV("buffer")) ||
      !iree_string_view_equal(types.buffer_view->type_name,
                              IREE_SV("buffer_view")) ||
      !iree_string_view_equal(types.fence->type_name, IREE_SV("fence"))) {
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
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_fence, iree_hal_module_types_t, fence,
                             iree_hal_fence_t)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_MODULE_HAL_TYPES_H_
