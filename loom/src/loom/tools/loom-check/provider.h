// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-check provider composition.
//
// Target packages expose reusable loom_target_provider_t records for compiler
// capabilities and may wrap them in loom_check_provider_t records for
// check-only emit or requirement namespaces. The provider layer assembles those
// records into the loom_check_environment_t consumed by the shared CLI runner.

#ifndef LOOM_TOOLS_LOOM_CHECK_PROVIDER_H_
#define LOOM_TOOLS_LOOM_CHECK_PROVIDER_H_

#include "iree/base/api.h"
#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_check_emit_provider_t loom_check_emit_provider_t;
typedef struct loom_check_requirement_provider_t
    loom_check_requirement_provider_t;

// Target-owned contribution linked into a loom-check environment.
typedef struct loom_check_provider_t {
  // Stable provider name used in diagnostics and help text.
  iree_string_view_t name;
  // Optional target provider contribution reused by production tools.
  const loom_target_provider_t* target_provider;
  // Optional emit provider table contributed by this provider.
  const loom_check_emit_provider_t* const* emit_providers;
  // Number of entries in |emit_providers|.
  iree_host_size_t emit_provider_count;
  // Optional requirement provider table contributed by this provider.
  const loom_check_requirement_provider_t* const* requirement_providers;
  // Number of entries in |requirement_providers|.
  iree_host_size_t requirement_provider_count;
} loom_check_provider_t;

// Static provider table linked into a loom-check binary or embedding.
typedef struct loom_check_provider_set_t {
  // Provider contribution table.
  const loom_check_provider_t* const* providers;
  // Number of entries in |providers|.
  iree_host_size_t provider_count;
} loom_check_provider_set_t;

// Runs loom-check using tool dialects plus |provider_set|'s target/check
// contributions.
int loom_check_provider_main(int argc, char** argv,
                             const loom_check_provider_set_t* provider_set);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_PROVIDER_H_
