// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_JSON_H_
#define EXPERIMENTAL_ID4_PIPELINE_JSON_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Appends |value| as one quoted JSON string, including required escaping.
iree_status_t id4_pipeline_json_append_string(iree_string_builder_t* builder,
                                              iree_string_view_t value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_JSON_H_
