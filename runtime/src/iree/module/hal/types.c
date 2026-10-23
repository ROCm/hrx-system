// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/module/hal/types.h"

static_assert(offsetof(iree_hal_buffer_t, resource) == 0 &&
                  offsetof(iree_hal_resource_t, ref_count) == 0,
              "HAL buffers must expose the intrusive count at offset zero");
static_assert(sizeof(((iree_hal_resource_t*)0)->ref_count) ==
                  IREE_VM_REF_OBJECT_SIZE,
              "HAL and VM must share the intrusive owner count");
static_assert(offsetof(iree_hal_fence_t, ref_count) == 0 &&
                  sizeof(((iree_hal_fence_t*)0)->ref_count) ==
                      IREE_VM_REF_OBJECT_SIZE,
              "HAL fences must share the VM intrusive owner count");

static void iree_hal_module_buffer_destroy(void* object) {
  // The VM already consumed the final owner, so destroy without decrementing
  // the shared intrusive reference count again.
  iree_hal_buffer_destroy((iree_hal_buffer_t*)object);
}

static void iree_hal_module_buffer_view_destroy(void* object) {
  iree_hal_buffer_view_destroy((iree_hal_buffer_view_t*)object);
}

static void iree_hal_module_fence_destroy(void* object) {
  iree_hal_fence_destroy((iree_hal_fence_t*)object);
}

static const iree_vm_ref_type_table_t iree_hal_module_type_table_;
static const iree_vm_ref_type_descriptor_t iree_hal_module_buffer_type_ = {
    iree_hal_module_buffer_destroy,
    &iree_hal_module_type_table_,
    IREE_SVL("buffer"),
};
static const iree_vm_ref_type_descriptor_t iree_hal_module_buffer_view_type_ = {
    iree_hal_module_buffer_view_destroy,
    &iree_hal_module_type_table_,
    IREE_SVL("buffer_view"),
};
static const iree_vm_ref_type_descriptor_t iree_hal_module_fence_type_ = {
    iree_hal_module_fence_destroy,
    &iree_hal_module_type_table_,
    IREE_SVL("fence"),
};
static const iree_hal_module_types_t iree_hal_module_types_ = {
    &iree_hal_module_buffer_type_,
    &iree_hal_module_buffer_view_type_,
    &iree_hal_module_fence_type_,
};
static const iree_vm_ref_type_table_t iree_hal_module_type_table_ = {
    sizeof(iree_hal_module_type_table_),
    IREE_VM_REF_TYPE_TABLE_FLAG_NONE,
    IREE_SVL("hal"),
    {&iree_hal_module_types_, IREE_HAL_MODULE_TYPE_COUNT},
};

IREE_API_EXPORT iree_status_t
iree_hal_module_register_types(iree_vm_environment_t* environment,
                               const iree_vm_ref_type_table_t** out_table) {
  if (!out_table) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_table is required");
  }
  IREE_RETURN_IF_ERROR(iree_vm_environment_register_ref_type_table(
      environment, &iree_hal_module_type_table_));
  *out_table = &iree_hal_module_type_table_;
  return iree_ok_status();
}
