// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/task/executable/environment.h"

#include "iree/base/internal/cpu.h"

//===----------------------------------------------------------------------===//
// iree_hal_executable_environment_*_t
//===----------------------------------------------------------------------===//

void iree_hal_executable_environment_initialize(
    iree_allocator_t temp_allocator,
    iree_hal_executable_environment_v0_t* out_environment) {
  IREE_ASSERT_ARGUMENT(out_environment);
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_environment, 0, sizeof(*out_environment));

  iree_cpu_data_t cpu_data;
  iree_cpu_query_data(temp_allocator, &cpu_data);
  memcpy(out_environment->processor.data, cpu_data.fields,
         iree_min(sizeof(out_environment->processor.data),
                  sizeof(cpu_data.fields)));

  IREE_TRACE_ZONE_END(z0);
}
