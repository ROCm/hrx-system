// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_TASK_EXECUTABLE_ENVIRONMENT_H_
#define IREE_HAL_DRIVERS_TASK_EXECUTABLE_ENVIRONMENT_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/task/executable/library/abi.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_executable_environment_*_t
//===----------------------------------------------------------------------===//

// Initializes |out_environment| to the default empty environment.
// |temp_allocator| may be used for temporary allocations during initialization.
void iree_hal_executable_environment_initialize(
    iree_allocator_t temp_allocator,
    iree_hal_executable_environment_v0_t* out_environment);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_TASK_EXECUTABLE_ENVIRONMENT_H_
