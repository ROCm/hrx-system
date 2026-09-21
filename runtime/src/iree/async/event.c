// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/event.h"

#include "iree/async/proactor.h"

IREE_API_EXPORT iree_status_t iree_async_event_create(
    iree_async_proactor_t* proactor, iree_async_event_t** out_event) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = proactor->vtable->create_event(proactor, out_event);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Destroys the event, releasing backend resources and closing platform handles.
// Routed through the proactor that created this event.
static void iree_async_event_destroy(iree_async_event_t* event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  event->proactor->vtable->destroy_event(event->proactor, event);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_async_event_retain(iree_async_event_t* event) {
  if (event) {
    iree_atomic_ref_count_inc(&event->ref_count);
  }
}

IREE_API_EXPORT void iree_async_event_release(iree_async_event_t* event) {
  if (event && iree_atomic_ref_count_dec(&event->ref_count) == 1) {
    iree_async_event_destroy(event);
  }
}

IREE_API_EXPORT void iree_async_event_set(iree_async_event_t* event) {
  iree_async_event_native_set(&event->native);
}

IREE_API_EXPORT iree_status_t
iree_async_event_consume(iree_async_event_t* event) {
  return iree_async_event_native_consume(&event->native);
}

IREE_API_EXPORT void iree_async_event_native_deinitialize(
    iree_async_event_native_t* event) {
#if defined(IREE_ASYNC_HAVE_FD)
  if (event->wait_primitive.type == event->signal_primitive.type &&
      event->wait_primitive.value.fd == event->signal_primitive.value.fd) {
    event->signal_primitive = iree_async_primitive_none();
  }
#elif defined(IREE_ASYNC_HAVE_WIN32_HANDLE)
  if (event->wait_primitive.type == event->signal_primitive.type &&
      event->wait_primitive.value.win32_handle ==
          event->signal_primitive.value.win32_handle) {
    event->signal_primitive = iree_async_primitive_none();
  }
#endif  // native handle type
  iree_async_primitive_close(&event->signal_primitive);
  iree_async_primitive_close(&event->wait_primitive);
}
