// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_MODULE_BYTECODE_H_
#define LOOMC_MODULE_BYTECODE_H_

#include "loom/ir/module.h"
#include "loomc/module.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Serializes |internal_module| with the context and representation providers
// owned by |module| into an in-memory Loom bytecode source.
LOOMC_API_PRIVATE loomc_status_t
loomc_module_serialize_internal_bytecode_to_source(
    const loomc_module_t* module, const loom_module_t* internal_module,
    loomc_string_view_t identifier, loomc_allocator_t allocator,
    loomc_source_t** out_source);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_MODULE_BYTECODE_H_
