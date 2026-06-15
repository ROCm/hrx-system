// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/cli/help.h"

#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"

static bool loom_tooling_cli_default_help_filter(iree_string_view_t flag_file,
                                                 iree_string_view_t flag_name,
                                                 void* user_data) {
  (void)flag_name;
  (void)user_data;
  return iree_string_view_starts_with(flag_file, IREE_SV("loom/src/loom/")) ||
         iree_string_view_find(flag_file, IREE_SV("/loom/src/loom/"), 0) !=
             IREE_STRING_VIEW_NPOS ||
         iree_string_view_ends_with(
             flag_file, IREE_SV("runtime/src/iree/base/tooling/flags.c"));
}

void loom_tooling_cli_set_default_help_filter(void) {
  iree_flags_set_help_filter(loom_tooling_cli_default_help_filter,
                             /*user_data=*/NULL);
}

bool loom_tooling_cli_is_agents_markdown_arg(const char* arg) {
  return arg && iree_string_view_equal(iree_make_cstring_view(arg),
                                       IREE_SV("--agents_md"));
}
