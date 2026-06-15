// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "loomc/base.h"

static loomc_status_t loomc_system_allocator_ctl(
    void* self, loomc_allocator_command_t command, const void* params,
    void** inout_ptr) {
  (void)self;
  iree_allocator_t allocator = iree_allocator_system();
  return (loomc_status_t)allocator.ctl(
      allocator.self, (iree_allocator_command_t)command, params, inout_ptr);
}

loomc_allocator_t loomc_allocator_system(void) {
  return (loomc_allocator_t){
      .self = NULL,
      .ctl = loomc_system_allocator_ctl,
  };
}
