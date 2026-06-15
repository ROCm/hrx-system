// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/output.h"

#include <stdio.h>

#include "iree/base/target_platform.h"

#if defined(IREE_PLATFORM_WINDOWS)
#include <io.h>
#else
#include <unistd.h>
#endif  // IREE_PLATFORM_WINDOWS

static bool loom_check_stderr_is_tty(void) {
#if defined(IREE_PLATFORM_WINDOWS)
  return _isatty(_fileno(stderr)) != 0;
#else
  return isatty(fileno(stderr)) != 0;
#endif  // IREE_PLATFORM_WINDOWS
}

static const char* loom_check_color_pass(void) {
  return loom_check_stderr_is_tty() ? "\033[32m" : "";
}

static const char* loom_check_color_fail(void) {
  return loom_check_stderr_is_tty() ? "\033[31m" : "";
}

static const char* loom_check_color_skip(void) {
  return loom_check_stderr_is_tty() ? "\033[33m" : "";
}

static const char* loom_check_color_reset(void) {
  return loom_check_stderr_is_tty() ? "\033[0m" : "";
}

static const char* loom_check_outcome_label(loom_check_outcome_t outcome,
                                            bool xfail) {
  if (outcome == LOOM_CHECK_SKIP) {
    return "SKIP";
  }
  if (outcome == LOOM_CHECK_PASS) {
    return xfail ? "XFAIL" : "PASS";
  }
  return xfail ? "XPASS" : "FAIL";
}

void loom_check_print_case_header(iree_string_view_t filename,
                                  iree_host_size_t case_index,
                                  const loom_check_case_t* test_case,
                                  const loom_check_result_t* result) {
  const char* color = loom_check_color_fail();
  if (result->final_outcome == LOOM_CHECK_PASS) {
    color = loom_check_color_pass();
  } else if (result->final_outcome == LOOM_CHECK_SKIP) {
    color = loom_check_color_skip();
  }
  const char* label =
      loom_check_outcome_label(result->final_outcome, test_case->xfail);

  fprintf(stderr, "%s%s%s ", color, label, loom_check_color_reset());
  fprintf(stderr, "%.*s :: case %zu", (int)filename.size, filename.data,
          case_index + 1);
  fprintf(stderr, " [%s", loom_check_mode_name(test_case->mode));
  if (test_case->mode == LOOM_CHECK_MODE_PASS ||
      test_case->mode == LOOM_CHECK_MODE_PASS_REPORT) {
    fprintf(stderr, " %.*s", (int)test_case->pipeline.size,
            test_case->pipeline.data);
  } else if (test_case->mode == LOOM_CHECK_MODE_FORMAT) {
    fprintf(stderr, " %.*s", (int)test_case->format_target.size,
            test_case->format_target.data);
  } else if (test_case->mode == LOOM_CHECK_MODE_EMIT) {
    fprintf(stderr, " %.*s", (int)test_case->emit_target.size,
            test_case->emit_target.data);
  }
  fprintf(stderr, "]\n");
}

void loom_check_print_summary(iree_host_size_t pass_count,
                              iree_host_size_t fail_count,
                              iree_host_size_t skip_count) {
  const iree_host_size_t total = pass_count + fail_count + skip_count;
  if (total == 0) {
    return;
  }

  fprintf(stderr, "\n%s%zu passed%s", loom_check_color_pass(), pass_count,
          loom_check_color_reset());
  if (fail_count > 0) {
    fprintf(stderr, ", %s%zu failed%s", loom_check_color_fail(), fail_count,
            loom_check_color_reset());
  }
  if (skip_count > 0) {
    fprintf(stderr, ", %s%zu skipped%s", loom_check_color_skip(), skip_count,
            loom_check_color_reset());
  }
  fprintf(stderr, " (%zu total)\n", total);
}
