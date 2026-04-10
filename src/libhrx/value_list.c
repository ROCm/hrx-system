// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_internal.h"

#include <stdlib.h>

hrx_status_t hrx_value_list_create(size_t capacity,
                                     hrx_value_list_t* list) {
  if (!list) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "list is NULL");
  }
  *list = NULL;

  hrx_value_list_t created = (hrx_value_list_t)calloc(1, sizeof(*created));
  if (!created) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                            "failed to allocate value list");
  }

  iree_status_t status = iree_vm_list_create(
      iree_vm_make_undefined_type_def(), (iree_host_size_t)capacity,
      iree_allocator_system(), &created->vm_list);
  if (!iree_status_is_ok(status)) {
    free(created);
    return hrx_status_from_iree(status);
  }

  iree_atomic_ref_count_init(&created->ref_count);
  *list = created;
  return hrx_ok_status();
}

void hrx_value_list_retain(hrx_value_list_t list) {
  iree_vm_list_retain(list->vm_list);
  iree_atomic_ref_count_inc(&list->ref_count);
}

void hrx_value_list_release(hrx_value_list_t list) {
  iree_vm_list_release(list->vm_list);
  if (iree_atomic_ref_count_dec(&list->ref_count) == 1) {
    free(list);
  }
}

hrx_status_t hrx_value_list_size(hrx_value_list_t list, size_t* size) {
  if (!list || !size) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "list or size is NULL");
  }
  *size = (size_t)iree_vm_list_size(list->vm_list);
  return hrx_ok_status();
}

hrx_status_t hrx_value_list_push_i64(hrx_value_list_t list, int64_t value) {
  if (!list) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "list is NULL");
  }
  iree_vm_value_t vm_value = iree_vm_value_make_i64(value);
  return hrx_status_from_iree(
      iree_vm_list_push_value(list->vm_list, &vm_value));
}

hrx_status_t hrx_value_list_get_i64(hrx_value_list_t list, size_t index,
                                      int64_t* value) {
  if (!list || !value) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "list or value is NULL");
  }
  iree_vm_value_t vm_value;
  iree_status_t status = iree_vm_list_get_value_as(
      list->vm_list, (iree_host_size_t)index, IREE_VM_VALUE_TYPE_I64,
      &vm_value);
  if (!iree_status_is_ok(status)) {
    return hrx_status_from_iree(status);
  }
  *value = vm_value.i64;
  return hrx_ok_status();
}

hrx_status_t hrx_value_list_push_null_ref(hrx_value_list_t list) {
  if (!list) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "list is NULL");
  }
  iree_vm_ref_t ref = iree_vm_ref_null();
  return hrx_status_from_iree(
      iree_vm_list_push_ref_move(list->vm_list, &ref));
}

hrx_status_t hrx_value_list_push_buffer(hrx_value_list_t list,
                                          hrx_buffer_t buffer) {
  if (!list || !buffer) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "list or buffer is NULL");
  }
  iree_vm_ref_t ref = iree_hal_buffer_retain_ref(buffer->hal_buffer);
  return hrx_status_from_iree(
      iree_vm_list_push_ref_move(list->vm_list, &ref));
}

hrx_status_t hrx_value_list_push_buffer_view(
    hrx_value_list_t list, hrx_buffer_view_t buffer_view) {
  if (!list || !buffer_view) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "list or buffer_view is NULL");
  }
  iree_vm_ref_t ref =
      iree_hal_buffer_view_retain_ref(buffer_view->hal_buffer_view);
  return hrx_status_from_iree(
      iree_vm_list_push_ref_move(list->vm_list, &ref));
}

hrx_status_t hrx_value_list_push_fence(hrx_value_list_t list,
                                         hrx_fence_t fence) {
  if (!list || !fence) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                            "list or fence is NULL");
  }
  iree_vm_ref_t ref = iree_hal_fence_retain_ref(fence->hal_fence);
  return hrx_status_from_iree(
      iree_vm_list_push_ref_move(list->vm_list, &ref));
}
