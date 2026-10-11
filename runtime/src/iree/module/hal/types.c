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

static void iree_hal_module_buffer_destroy(void* object) {
  // The VM already consumed the final owner, so destroy without decrementing
  // the shared intrusive reference count again.
  iree_hal_buffer_destroy((iree_hal_buffer_t*)object);
}

static void iree_hal_module_buffer_view_destroy(void* object) {
  iree_hal_buffer_view_destroy((iree_hal_buffer_view_t*)object);
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

#define IREE_HAL_VM_DEFINE_GROUP_RESOURCE(name, ordinal)                    \
  static_assert(offsetof(iree_hal_module_types_t, name) ==                  \
                    ordinal * sizeof(iree_vm_ref_type_t),                   \
                "HAL type ordinal must remain stable");                     \
  static void iree_hal_vm_##name##_destroy(void* object) {                  \
    iree_hal_vm_##name##_t* wrapper = object;                               \
    iree_hal_##name##_release(wrapper->name);                               \
    iree_hal_device_group_release(wrapper->group);                          \
    iree_allocator_free(wrapper->host_allocator, wrapper);                  \
  }                                                                         \
  static const iree_vm_ref_type_descriptor_t iree_hal_vm_##name##_type_ = { \
      iree_hal_vm_##name##_destroy, &iree_hal_module_type_table_,           \
      IREE_SVL(#name)};                                                     \
  IREE_API_EXPORT iree_status_t iree_hal_vm_##name##_wrap(                  \
      iree_hal_device_group_t* group, iree_hal_##name##_t* name,            \
      iree_allocator_t host_allocator, iree_vm_ref_t* out_ref) {            \
    if (!group || !name || !out_ref) {                                      \
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,                 \
                              "group, resource and out_ref are required");  \
    }                                                                       \
    iree_hal_vm_##name##_t* wrapper = NULL;                                 \
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(                             \
        host_allocator, sizeof(*wrapper), (void**)&wrapper));               \
    iree_vm_ref_object_initialize(&wrapper->object);                        \
    wrapper->name = name;                                                   \
    wrapper->group = group;                                                 \
    wrapper->host_allocator = host_allocator;                               \
    iree_hal_##name##_retain(name);                                         \
    iree_hal_device_group_retain(group);                                    \
    void* object = wrapper;                                                 \
    *out_ref =                                                              \
        iree_vm_ref_from_ptr_move(&object, &iree_hal_vm_##name##_type_);    \
    return iree_ok_status();                                                \
  }

IREE_HAL_VM_DEFINE_GROUP_RESOURCE(queue, IREE_HAL_MODULE_TYPE_QUEUE)
IREE_HAL_VM_DEFINE_GROUP_RESOURCE(command_buffer,
                                  IREE_HAL_MODULE_TYPE_COMMAND_BUFFER)
IREE_HAL_VM_DEFINE_GROUP_RESOURCE(semaphore, IREE_HAL_MODULE_TYPE_SEMAPHORE)
IREE_HAL_VM_DEFINE_GROUP_RESOURCE(pool, IREE_HAL_MODULE_TYPE_POOL)
#undef IREE_HAL_VM_DEFINE_GROUP_RESOURCE

static_assert(offsetof(iree_hal_module_types_t, device_group) ==
                  IREE_HAL_MODULE_TYPE_DEVICE_GROUP *
                      sizeof(iree_vm_ref_type_t),
              "HAL type ordinal must remain stable");

static void iree_hal_vm_device_group_destroy(void* object) {
  iree_hal_vm_device_group_t* wrapper = object;
  iree_hal_device_group_release(wrapper->group);
  iree_allocator_free(wrapper->host_allocator, wrapper);
}

static const iree_vm_ref_type_descriptor_t iree_hal_vm_device_group_type_ = {
    iree_hal_vm_device_group_destroy,
    &iree_hal_module_type_table_,
    IREE_SVL("device_group"),
};

IREE_API_EXPORT iree_status_t iree_hal_vm_device_group_wrap(
    iree_hal_device_group_t* group, iree_allocator_t host_allocator,
    iree_vm_ref_t* out_ref) {
  if (!group || !out_ref) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "group and out_ref are required");
  }
  iree_hal_vm_device_group_t* wrapper = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*wrapper),
                                             (void**)&wrapper));
  iree_vm_ref_object_initialize(&wrapper->object);
  wrapper->group = group;
  wrapper->host_allocator = host_allocator;
  iree_hal_device_group_retain(group);
  void* object = wrapper;
  *out_ref =
      iree_vm_ref_from_ptr_move(&object, &iree_hal_vm_device_group_type_);
  return iree_ok_status();
}

static const iree_hal_module_types_t iree_hal_module_types_ = {
    &iree_hal_module_buffer_type_,     &iree_hal_module_buffer_view_type_,
    &iree_hal_vm_device_group_type_,   &iree_hal_vm_queue_type_,
    &iree_hal_vm_command_buffer_type_, &iree_hal_vm_semaphore_type_,
    &iree_hal_vm_pool_type_,
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
