// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_SCF_CLEANUP_PATTERNS_H_
#define LOOM_TRANSFORMS_SCF_CLEANUP_PATTERNS_H_

#include "loom/rewrite/pattern_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initial prepared canonicalizations contributed by the SCF dialect.
extern const loom_rewrite_pattern_provider_t
    loom_scf_prepared_canonicalization_pattern_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SCF_CLEANUP_PATTERNS_H_
