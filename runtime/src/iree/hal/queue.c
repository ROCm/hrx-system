// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/queue.h"

#include "iree/hal/detail.h"

//===----------------------------------------------------------------------===//
// iree_hal_queue_family_t
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_hal_queue_family_ordinal_t
iree_hal_queue_family_ordinal(const iree_hal_queue_family_t* queue_family) {
  IREE_ASSERT_ARGUMENT(queue_family);
  return queue_family->ordinal;
}

IREE_API_EXPORT void iree_hal_queue_family_initialize(
    iree_hal_queue_family_ordinal_t ordinal,
    iree_hal_queue_family_t* out_queue_family) {
  IREE_ASSERT_ARGUMENT(out_queue_family);
  out_queue_family->ordinal = ordinal;
}

//===----------------------------------------------------------------------===//
// iree_hal_queue_t
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(queue);

IREE_API_EXPORT const iree_hal_queue_family_t* iree_hal_queue_family(
    const iree_hal_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  return queue->queue_family;
}

IREE_API_EXPORT void iree_hal_queue_initialize(
    const iree_hal_queue_family_t* queue_family,
    const iree_hal_queue_vtable_t* vtable, iree_hal_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(queue_family);
  IREE_ASSERT_ARGUMENT(vtable);
  IREE_ASSERT_ARGUMENT(out_queue);
  iree_hal_resource_initialize(vtable, &out_queue->resource);
  out_queue->queue_family = queue_family;
}
