// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// Status API implementation. Follows IREE's pattern: NULL = OK,
// non-NULL = heap-allocated error with code + message.

#include "pyre_internal.h"

#include <stdlib.h>
#include <string.h>

pyre_status_t pyre_make_status(pyre_status_code_t code, const char* message) {
  if (code == PYRE_STATUS_OK) return pyre_ok_status();
  pyre_status_s* s = (pyre_status_s*)malloc(sizeof(pyre_status_s));
  if (!s) return NULL;  // OOM making error — can't do much.
  s->code = code;
  s->message = message ? strdup(message) : NULL;
  return s;
}

pyre_status_code_t pyre_status_code(pyre_status_t status) {
  if (pyre_status_is_ok(status)) return PYRE_STATUS_OK;
  return status->code;
}

pyre_status_t pyre_status_to_string(pyre_status_t status, char** out_message,
                                    size_t* out_length) {
  if (!out_message) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT,
                            "out_message is NULL");
  }
  if (pyre_status_is_ok(status)) {
    const char* ok_msg = "OK";
    *out_message = strdup(ok_msg);
    if (out_length) *out_length = strlen(ok_msg);
    return pyre_ok_status();
  }
  const char* msg = status->message ? status->message : "(no message)";
  *out_message = strdup(msg);
  if (out_length) *out_length = strlen(msg);
  return pyre_ok_status();
}

void pyre_status_free_message(char* message) { free(message); }

void pyre_status_ignore(pyre_status_t status) {
  if (pyre_status_is_ok(status)) return;
  free(status->message);
  free(status);
}

// Convert IREE status to pyre status.
pyre_status_t pyre_status_from_iree(iree_status_t iree_status) {
  if (iree_status_is_ok(iree_status)) return pyre_ok_status();

  // Map IREE status code to pyre code.
  iree_status_code_t iree_code = iree_status_code(iree_status);
  pyre_status_code_t pyre_code;
  switch (iree_code) {
    case IREE_STATUS_INVALID_ARGUMENT:
      pyre_code = PYRE_STATUS_INVALID_ARGUMENT;
      break;
    case IREE_STATUS_NOT_FOUND:
      pyre_code = PYRE_STATUS_NOT_FOUND;
      break;
    case IREE_STATUS_ALREADY_EXISTS:
      pyre_code = PYRE_STATUS_ALREADY_EXISTS;
      break;
    case IREE_STATUS_OUT_OF_RANGE:
      pyre_code = PYRE_STATUS_OUT_OF_RANGE;
      break;
    case IREE_STATUS_UNIMPLEMENTED:
      pyre_code = PYRE_STATUS_UNIMPLEMENTED;
      break;
    case IREE_STATUS_UNAVAILABLE:
      pyre_code = PYRE_STATUS_UNAVAILABLE;
      break;
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      pyre_code = PYRE_STATUS_OUT_OF_MEMORY;
      break;
    case IREE_STATUS_DEADLINE_EXCEEDED:
      pyre_code = PYRE_STATUS_DEADLINE_EXCEEDED;
      break;
    default:
      pyre_code = PYRE_STATUS_INTERNAL;
      break;
  }

  // Extract IREE error message.
  iree_allocator_t allocator = iree_allocator_system();
  char* iree_msg = NULL;
  iree_host_size_t msg_len = 0;
  if (!iree_status_to_string(iree_status, &allocator, &iree_msg, &msg_len)) {
    // Failed to get message — use generic.
    iree_status_ignore(iree_status);
    return pyre_make_status(pyre_code, "IREE error (could not format message)");
  }

  pyre_status_t result = pyre_make_status(pyre_code, iree_msg);
  iree_allocator_free(allocator, iree_msg);
  iree_status_ignore(iree_status);
  return result;
}

iree_status_t pyre_status_to_iree(pyre_status_t status) {
  if (pyre_status_is_ok(status)) return iree_ok_status();
  iree_status_code_t code = (iree_status_code_t)status->code;
  if (status->code == PYRE_STATUS_OUT_OF_MEMORY) {
    code = IREE_STATUS_RESOURCE_EXHAUSTED;
  }
  iree_status_t result = iree_make_status(code, "%s",
                                          status->message ? status->message
                                                          : "(no message)");
  pyre_status_ignore(status);
  return result;
}
