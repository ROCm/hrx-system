// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_internal.h"

#include <stdlib.h>

pyre_status_t pyre_value_list_create(size_t capacity,
                                     pyre_value_list_t* list) {
  if (!list) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "list is NULL");
  }
  *list = NULL;

  pyre_value_list_t created = (pyre_value_list_t)calloc(1, sizeof(*created));
  if (!created) {
    return pyre_make_status(PYRE_STATUS_OUT_OF_MEMORY,
                            "failed to allocate value list");
  }

  iree_status_t status = iree_vm_list_create(
      iree_vm_make_undefined_type_def(), (iree_host_size_t)capacity,
      iree_allocator_system(), &created->vm_list);
  if (!iree_status_is_ok(status)) {
    free(created);
    return pyre_status_from_iree(status);
  }

  iree_atomic_ref_count_init(&created->ref_count);
  *list = created;
  return pyre_ok_status();
}

void pyre_value_list_retain(pyre_value_list_t list) {
  iree_vm_list_retain(list->vm_list);
  iree_atomic_ref_count_inc(&list->ref_count);
}

void pyre_value_list_release(pyre_value_list_t list) {
  iree_vm_list_release(list->vm_list);
  if (iree_atomic_ref_count_dec(&list->ref_count) == 1) {
    free(list);
  }
}

pyre_status_t pyre_value_list_size(pyre_value_list_t list, size_t* size) {
  if (!list || !size) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "list or size is NULL");
  }
  *size = (size_t)iree_vm_list_size(list->vm_list);
  return pyre_ok_status();
}

pyre_status_t pyre_value_list_push_i64(pyre_value_list_t list, int64_t value) {
  if (!list) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "list is NULL");
  }
  iree_vm_value_t vm_value = iree_vm_value_make_i64(value);
  return pyre_status_from_iree(
      iree_vm_list_push_value(list->vm_list, &vm_value));
}

pyre_status_t pyre_value_list_get_i64(pyre_value_list_t list, size_t index,
                                      int64_t* value) {
  if (!list || !value) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "list or value is NULL");
  }
  iree_vm_value_t vm_value;
  iree_status_t status = iree_vm_list_get_value_as(
      list->vm_list, (iree_host_size_t)index, IREE_VM_VALUE_TYPE_I64,
      &vm_value);
  if (!iree_status_is_ok(status)) {
    return pyre_status_from_iree(status);
  }
  *value = vm_value.i64;
  return pyre_ok_status();
}

pyre_status_t pyre_value_list_push_null_ref(pyre_value_list_t list) {
  if (!list) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "list is NULL");
  }
  iree_vm_ref_t ref = iree_vm_ref_null();
  return pyre_status_from_iree(
      iree_vm_list_push_ref_move(list->vm_list, &ref));
}

pyre_status_t pyre_value_list_push_buffer(pyre_value_list_t list,
                                          pyre_buffer_t buffer) {
  if (!list || !buffer) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "list or buffer is NULL");
  }
  iree_vm_ref_t ref = iree_hal_buffer_retain_ref(buffer->hal_buffer);
  return pyre_status_from_iree(
      iree_vm_list_push_ref_move(list->vm_list, &ref));
}

pyre_status_t pyre_value_list_push_buffer_view(
    pyre_value_list_t list, pyre_buffer_view_t buffer_view) {
  if (!list || !buffer_view) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "list or buffer_view is NULL");
  }
  iree_vm_ref_t ref =
      iree_hal_buffer_view_retain_ref(buffer_view->hal_buffer_view);
  return pyre_status_from_iree(
      iree_vm_list_push_ref_move(list->vm_list, &ref));
}

pyre_status_t pyre_value_list_push_fence(pyre_value_list_t list,
                                         pyre_fence_t fence) {
  if (!list || !fence) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "list or fence is NULL");
  }
  iree_vm_ref_t ref = iree_hal_fence_retain_ref(fence->hal_fence);
  return pyre_status_from_iree(
      iree_vm_list_push_ref_move(list->vm_list, &ref));
}
