// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Bridge between the public HRX API and binding code that uses IREE types and
// status conventions internally.

#ifndef HRX_STREAMING_BRIDGE_H_
#define HRX_STREAMING_BRIDGE_H_

#include <stdlib.h>

#include "hrx_runtime.h"
#include "iree/base/api.h"

//===----------------------------------------------------------------------===//
// Status bridging: hrx_status_t <-> iree_status_t
//
// NULL is OK for both types so the success path is free. For non-OK
// statuses we cannot simply reinterpret-cast: hrx_status_t is a pointer
// to a heap-allocated hrx_status_s, while iree_status_t is a tagged
// pointer whose low bits hold the status code and whose high bits hold
// optional iree_status_storage_t. Feeding a hrx pointer to
// iree_status_free/_ignore would mask off the low bits and dereference
// garbage (which is exactly the crash we used to hit in the kernel-
// launch pointer scan).
//
// hrx_status_code_t values are the same as iree_status_code_t so we can
// safely pass the code along; we also copy the message so iree callers
// can format the error cleanly. The incoming hrx status is consumed
// (freed) since ownership is transferred.
//===----------------------------------------------------------------------===//

static inline iree_status_t hrx_to_iree_status(hrx_status_t s) {
  if (hrx_status_is_ok(s)) {
    return iree_ok_status();
  }
  iree_status_code_t code = (iree_status_code_t)hrx_status_code(s);
  char* message_buf = NULL;
  size_t message_len = 0;
  hrx_status_t to_str_status =
      hrx_status_to_string(s, &message_buf, &message_len);
  iree_status_t iree_s;
  if (hrx_status_is_ok(to_str_status) && message_buf) {
    iree_s = iree_make_status(code, "%.*s", (int)message_len, message_buf);
  } else {
    iree_s = iree_status_from_code(code);
  }
  if (message_buf) {
    free(message_buf);
  }
  if (!hrx_status_is_ok(to_str_status)) {
    hrx_status_ignore(to_str_status);
  }
  hrx_status_ignore(s);
  return iree_s;
}

//===----------------------------------------------------------------------===//
// HRX_CALL: wrap hrx API calls for use with IREE error macros
//
// Usage:
//   IREE_RETURN_IF_ERROR(HRX_CALL(hrx_semaphore_signal(semaphore, value)));
//   IREE_RETURN_IF_ERROR(HRX_CALL(hrx_semaphore_create(dev, 0, &sem)));
//===----------------------------------------------------------------------===//

#define HRX_CALL(expr) hrx_to_iree_status(expr)

#endif  // HRX_STREAMING_BRIDGE_H_
