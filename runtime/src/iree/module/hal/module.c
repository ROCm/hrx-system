// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/module/hal/module.h"

#include "iree/module/hal/native_schema.h"
#include "iree/module/hal/queue.h"
#include "iree/vm/reflection.h"

typedef struct iree_hal_module_t {
  // Common reference-counted owner of this immutable module.
  iree_vm_module_t base;
  // Published descriptor borrowing storage within this allocation.
  iree_vm_module_descriptor_t descriptor;
  // Resolved HAL identities retained as immutable native-entry context.
  iree_hal_module_types_t types;
  // Native directory with no ordinary VM callable exports or imports.
  iree_vm_native_module_t native;
  // One exact-version entry per declared HAL method, independent of shapes.
  iree_vm_native_export_t exports[IREE_HAL_NATIVE_METHOD_COUNT];
  // Allocator used for final module storage release.
  iree_allocator_t host_allocator;
} iree_hal_module_t;

static void iree_hal_module_destroy(iree_vm_module_t* base) {
  iree_hal_module_t* module = (iree_hal_module_t*)base;
  iree_allocator_free(module->host_allocator, module);
}

static const iree_vm_module_vtable_t kVtable = {
    .structure_size = sizeof(kVtable),
    .abi_version = IREE_VM_MODULE_ABI_VERSION_0,
    .destroy = iree_hal_module_destroy,
    .query_presentation = iree_vm_module_query_presentation_none,
    .metadata_by_ordinal = iree_vm_module_metadata_by_ordinal_none,
};

IREE_API_EXPORT iree_status_t iree_hal_module_create(
    iree_vm_environment_t* environment, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  if (!out_module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_module is required");
  }
  *out_module = NULL;
  if (!environment) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "environment is required");
  }
  const iree_vm_ref_type_table_t* table =
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("hal"));
  if (!table) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL reference types must be registered first");
  }
  iree_hal_module_types_t types;
  IREE_RETURN_IF_ERROR(iree_hal_module_types_resolve(table, &types));
  iree_hal_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module));
  module->host_allocator = host_allocator;
  module->types = types;
  module->exports[IREE_HAL_NATIVE_QUEUE_COPY] =
      iree_hal_vm_queue_copy_export(&module->types);
  module->native = (iree_vm_native_module_t){
      .exports = module->exports,
      .export_count = IREE_ARRAYSIZE(module->exports)};
  module->descriptor = (iree_vm_module_descriptor_t){
      .name = IREE_SV("hal"),
      .flags = IREE_VM_MODULE_FLAG_LINKABLE,
      .ref_types = {(const iree_vm_ref_type_t*)&module->types,
                    IREE_HAL_MODULE_TYPE_COUNT},
      .native = &module->native,
  };
  iree_status_t status =
      iree_vm_module_initialize(&kVtable, &module->descriptor, &module->base);
  if (iree_status_is_ok(status)) {
    *out_module = &module->base;
  } else {
    iree_allocator_free(host_allocator, module);
  }
  return status;
}
