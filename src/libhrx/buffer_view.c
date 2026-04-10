// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_internal.h"

#include <stdlib.h>

hrx_status_t hrx_buffer_view_create(hrx_buffer_t buffer, size_t shape_rank,
                                    const int64_t *shape,
                                    hrx_element_type_t element_type,
                                    hrx_encoding_type_t encoding_type,
                                    hrx_buffer_view_t *buffer_view) {
  if (!buffer || !buffer_view) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "buffer or buffer_view is NULL");
  }
  *buffer_view = NULL;
  if (shape_rank > 0 && !shape) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "shape is NULL for non-zero rank");
  }

  hrx_buffer_view_t created = (hrx_buffer_view_t)calloc(1, sizeof(*created));
  if (!created) {
    return hrx_make_status(HRX_STATUS_OUT_OF_MEMORY,
                           "failed to allocate buffer_view");
  }

  iree_hal_dim_t stack_shape[8];
  iree_hal_dim_t *iree_shape = stack_shape;
  if (shape_rank > IREE_ARRAYSIZE(stack_shape)) {
    iree_status_t alloc_status = iree_allocator_malloc(
        iree_allocator_system(), shape_rank * sizeof(*iree_shape),
        (void **)&iree_shape);
    if (!iree_status_is_ok(alloc_status)) {
      free(created);
      return hrx_status_from_iree(alloc_status);
    }
  }

  for (size_t i = 0; i < shape_rank; ++i) {
    if (shape[i] < 0) {
      if (iree_shape != stack_shape) {
        iree_allocator_free(iree_allocator_system(), iree_shape);
      }
      free(created);
      return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                             "shape dimensions must be non-negative");
    }
    iree_shape[i] = (iree_hal_dim_t)shape[i];
  }

  iree_status_t status = iree_hal_buffer_view_create(
      buffer->hal_buffer, (iree_host_size_t)shape_rank, iree_shape,
      (iree_hal_element_type_t)element_type,
      (iree_hal_encoding_type_t)encoding_type, iree_allocator_system(),
      &created->hal_buffer_view);

  if (iree_shape != stack_shape) {
    iree_allocator_free(iree_allocator_system(), iree_shape);
  }

  if (!iree_status_is_ok(status)) {
    free(created);
    return hrx_status_from_iree(status);
  }

  iree_atomic_ref_count_init(&created->ref_count);
  *buffer_view = created;
  return hrx_ok_status();
}

void hrx_buffer_view_retain(hrx_buffer_view_t buffer_view) {
  iree_hal_buffer_view_retain(buffer_view->hal_buffer_view);
  iree_atomic_ref_count_inc(&buffer_view->ref_count);
}

void hrx_buffer_view_release(hrx_buffer_view_t buffer_view) {
  iree_hal_buffer_view_release(buffer_view->hal_buffer_view);
  if (iree_atomic_ref_count_dec(&buffer_view->ref_count) == 1) {
    free(buffer_view);
  }
}

hrx_status_t hrx_buffer_view_rank(hrx_buffer_view_t buffer_view, size_t *rank) {
  if (!buffer_view || !rank) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "buffer_view or rank is NULL");
  }
  *rank = (size_t)iree_hal_buffer_view_shape_rank(buffer_view->hal_buffer_view);
  return hrx_ok_status();
}

hrx_status_t hrx_buffer_view_dim(hrx_buffer_view_t buffer_view, size_t dim,
                                 int64_t *value) {
  if (!buffer_view || !value) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT,
                           "buffer_view or value is NULL");
  }
  size_t rank =
      (size_t)iree_hal_buffer_view_shape_rank(buffer_view->hal_buffer_view);
  if (dim >= rank) {
    return hrx_make_status(HRX_STATUS_OUT_OF_RANGE,
                           "buffer_view dim out of range");
  }
  *value = (int64_t)iree_hal_buffer_view_shape_dim(buffer_view->hal_buffer_view,
                                                   (iree_host_size_t)dim);
  return hrx_ok_status();
}
