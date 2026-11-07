// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>

#include "iree/io/formats/parameter_index_fuzz_util.h"
#include "iree/io/formats/safetensors/safetensors_parser.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size > 1024 * 1024) {
    return 0;
  }
  iree_io_parameter_fuzz_parse(iree_make_const_byte_span(data, size),
                               iree_io_parse_safetensors_index, nullptr);
  return 0;
}
