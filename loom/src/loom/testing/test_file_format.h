// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical formatting support for .loom-test containers.

#ifndef LOOM_TESTING_TEST_FILE_FORMAT_H_
#define LOOM_TESTING_TEST_FILE_FORMAT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loomc/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Canonicalizes every ordinary input section in |source| while preserving
// directives, case separators, and expected-output sections byte-for-byte.
// ROUNDTRIP and FORMAT cases with explicit expected sections retain their
// authored input spelling because the spelling difference is the behavior
// under test.
// Parser or verifier failures are accepted only when every emitted diagnostic
// has a one-to-one match in that case's annotations and at least one matched
// diagnostic is an error.
//
// |out_source| is reset before formatting. It retains no partial output when
// formatting fails.
iree_status_t loom_test_file_format(iree_string_view_t source,
                                    iree_string_view_t filename,
                                    loomc_context_t* context,
                                    loomc_workspace_t* workspace,
                                    iree_arena_block_pool_t* block_pool,
                                    iree_allocator_t allocator,
                                    iree_string_builder_t* out_source);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TESTING_TEST_FILE_FORMAT_H_
