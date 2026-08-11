// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>

#include "iree/base/api.h"
#include "iree/hal/remote/protocol/commands.h"

#define FUZZ_ASSERT(condition) \
  do {                         \
    if (!(condition)) {        \
      __builtin_trap();        \
    }                          \
  } while (0)

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  iree_const_byte_span_t remaining = iree_make_const_byte_span(data, size);
  while (remaining.data_length > 0) {
    iree_hal_remote_command_view_t command;
    iree_status_t status = iree_hal_remote_command_parse(remaining, &command);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return 0;
    }

    FUZZ_ASSERT(command.bytes.data == remaining.data);
    FUZZ_ASSERT(command.bytes.data_length >=
                sizeof(iree_hal_remote_cmd_header_t));
    FUZZ_ASSERT(command.bytes.data_length <= remaining.data_length);
    FUZZ_ASSERT(command.header.length == command.bytes.data_length);

    remaining.data += command.bytes.data_length;
    remaining.data_length -= command.bytes.data_length;
  }
  return 0;
}
