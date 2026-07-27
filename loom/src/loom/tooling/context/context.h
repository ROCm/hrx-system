// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared context registration policy for user-facing Loom tools.
//
// Command-line tools operate on developer-authored files, including checked-in
// .loom-test files. They therefore use a wider dialect surface than embedded
// production/JIT users: production dialects, the synthetic test dialect, and
// whichever target dialects the linked provider environment contributes.

#ifndef LOOM_TOOLING_CONTEXT_CONTEXT_H_
#define LOOM_TOOLING_CONTEXT_CONTEXT_H_

#include "iree/base/api.h"
#include "loom/ir/context.h"
#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers the common dialect surface for user-facing Loom tools.
//
// The context must be initialized and not yet finalized. This does not register
// target-provider dialects; use
// loom_tooling_context_register_tool_dialects_with_target_environment when the
// tool has a composed target environment.
iree_status_t loom_tooling_context_register_tool_dialects(
    loom_context_t* context);

// Registers the common tool dialect surface and all target dialects contributed
// by |target_environment|. A NULL target environment registers only the common
// tool dialect surface.
iree_status_t
loom_tooling_context_register_tool_dialects_with_target_environment(
    const loom_target_environment_t* target_environment,
    loom_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_CONTEXT_CONTEXT_H_
