// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical AMD XDNA artifact provider.

#ifndef LOOM_TOOLING_TARGET_AMD_XDNA_ARTIFACT_PROVIDER_H_
#define LOOM_TOOLING_TARGET_AMD_XDNA_ARTIFACT_PROVIDER_H_

#include "loom/tooling/compile/artifact.h"

#ifdef __cplusplus
extern "C" {
#endif

// Public format name for canonical AMD XDNA artifacts.
#define LOOM_XDNA_ARTIFACT_FORMAT "xdna"

// Native AIE2P implementation of the canonical XDNA artifact format.
extern const loom_artifact_provider_t loom_xdna_artifact_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TARGET_AMD_XDNA_ARTIFACT_PROVIDER_H_
