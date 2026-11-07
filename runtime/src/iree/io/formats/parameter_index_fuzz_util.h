// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_IO_FORMATS_PARAMETER_INDEX_FUZZ_UTIL_H_
#define IREE_IO_FORMATS_PARAMETER_INDEX_FUZZ_UTIL_H_

#include "iree/base/api.h"
#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"

// Parses one memory-backed parameter file and checks every accepted index.
// Rejected input is an ordinary result when |expected_index| is NULL. When an
// expected index is supplied the parse must succeed and reproduce it exactly.
typedef iree_status_t(IREE_API_PTR* iree_io_parameter_fuzz_parse_fn_t)(
    iree_io_file_handle_t* file_handle, iree_io_parameter_index_t* index,
    iree_allocator_t host_allocator);

void iree_io_parameter_fuzz_parse(iree_const_byte_span_t contents,
                                  iree_io_parameter_fuzz_parse_fn_t parse_fn,
                                  iree_io_parameter_index_t* expected_index);

#endif  // IREE_IO_FORMATS_PARAMETER_INDEX_FUZZ_UTIL_H_
