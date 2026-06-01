// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/base/target_platform.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"

#if defined(IREE_SANITIZER_THREAD)
// Treat any TSan report as a test failure. Without this, TSan prints WARNING
// lines to stderr but the process exits 0 and ctest marks the test PASSed,
// so a race that doesn't otherwise crash the test is silently ignored.
// User-provided TSAN_OPTIONS still wins (the env var overrides these
// program-level defaults), so developers can pass halt_on_error=0 to keep
// running after the first race.
extern "C" const char* __tsan_default_options() {
  return "halt_on_error=1:abort_on_error=1";
}
#endif  // IREE_SANITIZER_THREAD

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();

  // Pass through flags to gtest (allowing --help to fall through).
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK |
                               IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                           &argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);

  int ret = RUN_ALL_TESTS();

  IREE_TRACE_APP_EXIT(ret);
  return ret;
}
