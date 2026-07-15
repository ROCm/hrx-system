// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Status API implementation. NULL is OK and non-NULL is an error payload.

#include <stdlib.h>
#include <string.h>

#include "hrx_internal.h"

static char hrx_status_out_of_memory_message[] = "out of memory";
static hrx_status_s hrx_status_out_of_memory_sentinel = {
    .code = HRX_STATUS_OUT_OF_MEMORY,
    .message = hrx_status_out_of_memory_message,
};

static hrx_status_t hrx_status_out_of_memory(void) {
  return &hrx_status_out_of_memory_sentinel;
}

static char* hrx_status_clone_message(const char* message) {
  const size_t message_length = strlen(message);
  if (message_length == SIZE_MAX) return NULL;
  char* clone = (char*)malloc(message_length + 1);
  if (!clone) return NULL;
  memcpy(clone, message, message_length + 1);
  return clone;
}

hrx_status_t hrx_make_status(hrx_status_code_t code, const char* message) {
  if (code == HRX_STATUS_OK) return hrx_ok_status();
  hrx_status_s* status = (hrx_status_s*)malloc(sizeof(*status));
  if (!status) return hrx_status_out_of_memory();
  status->code = code;
  status->message = message ? hrx_status_clone_message(message) : NULL;
  if (message && !status->message) {
    free(status);
    return hrx_status_out_of_memory();
  }
  return status;
}

hrx_status_code_t hrx_status_code(hrx_status_t status) {
  if (hrx_status_is_ok(status)) return HRX_STATUS_OK;
  return status->code;
}

hrx_status_t hrx_status_to_string(hrx_status_t status, char** out_message,
                                  size_t* out_length) {
  if (!out_message) {
    return hrx_make_status(HRX_STATUS_INVALID_ARGUMENT, "out_message is NULL");
  }
  *out_message = NULL;
  if (out_length) *out_length = 0;
  const char* message = "OK";
  if (!hrx_status_is_ok(status)) {
    message = status->message ? status->message : "(no message)";
  }
  char* message_clone = hrx_status_clone_message(message);
  if (!message_clone) return hrx_status_out_of_memory();
  *out_message = message_clone;
  if (out_length) *out_length = strlen(message);
  return hrx_ok_status();
}

void hrx_status_free_message(char* message) { free(message); }

void hrx_status_ignore(hrx_status_t status) {
  if (hrx_status_is_ok(status) || status == hrx_status_out_of_memory()) return;
  free(status->message);
  free(status);
}

// Convert IREE status to hrx status.
hrx_status_t hrx_status_from_iree(iree_status_t iree_status) {
  if (iree_status_is_ok(iree_status)) return hrx_ok_status();

  // Map IREE status code to hrx code.
  iree_status_code_t iree_code = iree_status_code(iree_status);
  hrx_status_code_t hrx_code;
  switch (iree_code) {
    case IREE_STATUS_INVALID_ARGUMENT:
      hrx_code = HRX_STATUS_INVALID_ARGUMENT;
      break;
    case IREE_STATUS_NOT_FOUND:
      hrx_code = HRX_STATUS_NOT_FOUND;
      break;
    case IREE_STATUS_ALREADY_EXISTS:
      hrx_code = HRX_STATUS_ALREADY_EXISTS;
      break;
    case IREE_STATUS_OUT_OF_RANGE:
      hrx_code = HRX_STATUS_OUT_OF_RANGE;
      break;
    case IREE_STATUS_UNIMPLEMENTED:
      hrx_code = HRX_STATUS_UNIMPLEMENTED;
      break;
    case IREE_STATUS_UNAVAILABLE:
      hrx_code = HRX_STATUS_UNAVAILABLE;
      break;
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      hrx_code = HRX_STATUS_OUT_OF_MEMORY;
      break;
    case IREE_STATUS_DEADLINE_EXCEEDED:
      hrx_code = HRX_STATUS_DEADLINE_EXCEEDED;
      break;
    default:
      hrx_code = HRX_STATUS_INTERNAL;
      break;
  }

  // Extract IREE error message.
  iree_allocator_t allocator = iree_allocator_system();
  char* iree_msg = NULL;
  iree_host_size_t msg_len = 0;
  if (!iree_status_to_string(iree_status, &allocator, &iree_msg, &msg_len)) {
    // Failed to get message — use generic.
    iree_status_free(iree_status);
    return hrx_make_status(hrx_code, "IREE error (could not format message)");
  }

  hrx_status_t result = hrx_make_status(hrx_code, iree_msg);
  iree_allocator_free(allocator, iree_msg);
  iree_status_free(iree_status);
  return result;
}

iree_status_t hrx_status_to_iree(hrx_status_t status) {
  if (hrx_status_is_ok(status)) return iree_ok_status();
  iree_status_code_t code = (iree_status_code_t)status->code;
  if (status->code == HRX_STATUS_OUT_OF_MEMORY) {
    code = IREE_STATUS_RESOURCE_EXHAUSTED;
  }
  iree_status_t result = iree_make_status(
      code, "%s", status->message ? status->message : "(no message)");
  hrx_status_ignore(status);
  return result;
}
