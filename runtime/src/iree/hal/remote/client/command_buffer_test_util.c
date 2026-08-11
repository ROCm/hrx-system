// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/command_buffer_test_util.h"

#include "iree/hal/remote/client/command_buffer.h"

iree_byte_span_t iree_hal_remote_client_command_buffer_test_stream(
    iree_hal_command_buffer_t* command_buffer) {
  iree_const_byte_span_t stream =
      iree_hal_remote_client_command_buffer_stream(command_buffer);
  return iree_make_byte_span((void*)stream.data, stream.data_length);
}
