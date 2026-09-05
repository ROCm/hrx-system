// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/configured/provider.h"

#include "iree/base/threading/call_once.h"

static loom_target_environment_t configured_target_environment;
static iree_once_flag configured_target_environment_once = IREE_ONCE_FLAG_INIT;

static void loom_configured_target_environment_initialize_once(void) {
  IREE_CHECK_OK(loom_target_environment_initialize(
      loom_configured_target_provider_set(), &configured_target_environment));
}

const loom_target_environment_t* loom_configured_target_environment(void) {
  iree_call_once(&configured_target_environment_once,
                 loom_configured_target_environment_initialize_once);
  return &configured_target_environment;
}
