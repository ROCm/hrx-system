// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared pass-environment requirement keys for target-aware passes.

#ifndef LOOM_TARGET_PASS_REQUIREMENTS_H_
#define LOOM_TARGET_PASS_REQUIREMENTS_H_

// Pass requirement satisfied when the target capability exposes the mutable
// owner of invocation-local concrete function versions.
#define LOOM_TARGET_PASS_REQUIREMENT_MUTABLE_FUNCTION_VERSIONS \
  "target.mutable-function-versions"

#endif  // LOOM_TARGET_PASS_REQUIREMENTS_H_
