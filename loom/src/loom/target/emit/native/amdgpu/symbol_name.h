// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_SYMBOL_NAME_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_SYMBOL_NAME_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validates one symbol name against the AMDGPU artifact symbol grammar.
iree_status_t loom_amdgpu_symbol_name_validate(iree_string_view_t symbol);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_SYMBOL_NAME_H_
