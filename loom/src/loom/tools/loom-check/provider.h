// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-check provider composition.
//
// Target packages contribute check-only emit and requirement namespaces. The
// final binary supplies its public compiler environment separately; inspection
// handlers borrow that environment without owning compiler registration.

#ifndef LOOM_TOOLS_LOOM_CHECK_PROVIDER_H_
#define LOOM_TOOLS_LOOM_CHECK_PROVIDER_H_

#include "iree/base/api.h"
#include "loom/tooling/input/format.h"
#include "loomc/target.h"

struct loom_check_compile_provider_t;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_check_emit_provider_t loom_check_emit_provider_t;
typedef struct loom_check_requirement_provider_t
    loom_check_requirement_provider_t;

// Target-owned contribution linked into a loom-check environment.
typedef struct loom_check_provider_t {
  // Stable name identifying this check-only contribution.
  iree_string_view_t name;
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
  // Compiler environment borrowed from the final binary for the run.
  loomc_target_environment_t* target_environment;
  // Public compiler integrations owned by the final runner binary.
  const struct loom_check_compile_provider_t* compile_provider;
} loom_check_provider_set_t;

// Runs loom-check using |provider_set|'s compiler environment and check
// handlers. The caller keeps the environment alive until this returns.
int loom_check_provider_main(int argc, char** argv,
                             const loom_check_provider_set_t* provider_set,
                             loom_input_format_list_t input_formats);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_PROVIDER_H_
