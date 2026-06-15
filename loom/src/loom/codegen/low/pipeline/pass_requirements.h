// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared pass-environment requirement keys for target-low codegen passes.

#ifndef LOOM_CODEGEN_LOW_PIPELINE_PASS_REQUIREMENTS_H_
#define LOOM_CODEGEN_LOW_PIPELINE_PASS_REQUIREMENTS_H_

// Pass requirement satisfied when the pass environment provides a target-low
// descriptor registry for low-codegen passes.
#define LOOM_LOW_PASS_REQUIREMENT_TARGET_LOW_DESCRIPTOR_REGISTRY \
  "target.low-descriptor-registry"

// Pass requirement satisfied when the pass environment provides a
// source-to-target-low lowering policy registry.
#define LOOM_LOW_PASS_REQUIREMENT_TARGET_LOW_LOWER_POLICY_REGISTRY \
  "target.low-lower-policy-registry"

#endif  // LOOM_CODEGEN_LOW_PIPELINE_PASS_REQUIREMENTS_H_
