// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/status.h"

#include "iree/base/api.h"

static iree_status_t iree_status_from_loomc_status(loomc_status_t status) {
  return (iree_status_t)status;
}

static loomc_status_t loomc_status_from_iree_status(iree_status_t status) {
  return (loomc_status_t)status;
}

loomc_status_t loomc_status_allocate(loomc_status_code_t code, const char* file,
                                     uint32_t line,
                                     loomc_string_view_t message) {
  return loomc_status_from_iree_status(
      iree_status_allocate((iree_status_code_t)code, file, line,
                           iree_make_string_view(message.data, message.size)));
}

void loomc_status_free(loomc_status_t status) {
  iree_status_free(iree_status_from_loomc_status(status));
}

loomc_status_code_t loomc_status_consume_code(loomc_status_t status) {
  return (loomc_status_code_t)iree_status_consume_code(
      iree_status_from_loomc_status(status));
}

loomc_status_t loomc_status_join(loomc_status_t base_status,
                                 loomc_status_t new_status) {
  return loomc_status_from_iree_status(
      iree_status_join(iree_status_from_loomc_status(base_status),
                       iree_status_from_loomc_status(new_status)));
}

const char* loomc_status_code_string(loomc_status_code_t code) {
  return iree_status_code_string((iree_status_code_t)code);
}

loomc_string_view_t loomc_status_message(loomc_status_t status) {
  iree_string_view_t message =
      iree_status_message(iree_status_from_loomc_status(status));
  return loomc_make_string_view(message.data, message.size);
}

loomc_status_source_location_t loomc_status_source_location(
    loomc_status_t status) {
  iree_status_source_location_t source_location =
      iree_status_source_location(iree_status_from_loomc_status(status));
  return (loomc_status_source_location_t){
      .file = loomc_make_cstring_view(source_location.file),
      .line = source_location.line,
  };
}

bool loomc_status_format(loomc_status_t status,
                         loomc_host_size_t buffer_capacity, char* buffer,
                         loomc_host_size_t* out_length) {
  return iree_status_format(iree_status_from_loomc_status(status),
                            buffer_capacity, buffer, out_length);
}
