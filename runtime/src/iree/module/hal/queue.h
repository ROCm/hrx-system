// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULE_HAL_QUEUE_H_
#define IREE_MODULE_HAL_QUEUE_H_

#include "iree/module/hal/types.h"
#include "iree/vm/native_call.h"

#ifdef __cplusplus
extern "C" {
#endif

// Binds the generated copy ABI to its native implementation and resolved HAL
// types. The returned entry borrows |types| for its complete module lifetime.
iree_vm_native_export_t iree_hal_vm_queue_copy_export(
    const iree_hal_module_types_t* types);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_MODULE_HAL_QUEUE_H_
