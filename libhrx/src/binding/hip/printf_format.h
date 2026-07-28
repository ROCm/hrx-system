// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_HIP_PRINTF_FORMAT_H_
#define HRX_BINDING_HIP_PRINTF_FORMAT_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Formats one complete legacy HIP hostcall printf message into |builder|.
//
// Scalar arguments occupy eight-byte slots. Strings occupy the minimum whole
// number of eight-byte slots including their NUL terminator. %n consumes its
// device pointer slot without dereferencing it, matching the ROCm hostcall
// formatter.
//
// |builder| and |scratch_builder| are reset before formatting. On success
// |builder| contains the complete text and its size is the printf return count.
// On failure both are empty so a caller cannot accidentally publish partial
// output. Allocated storage is retained for reuse across calls.
//
// Returns INVALID_ARGUMENT for malformed device data, OUT_OF_RANGE when the
// output cannot be represented by printf's signed int result, and UNKNOWN when
// the host formatter rejects a conversion.
iree_status_t iree_hip_printf_format(iree_string_builder_t* builder,
                                     iree_string_builder_t* scratch_builder,
                                     iree_string_view_t format,
                                     const uint8_t* arguments,
                                     iree_host_size_t argument_length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_HIP_PRINTF_FORMAT_H_
