// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/tool/amdf-info/invocation.h"

int main(int argument_count, char** arguments) {
  amdf_info_options_t options;
  const char* error = amdf_info_options_parse(
      argument_count, (const char* const*)arguments, &options);
  amdf_info_selector_t selector;
  if (!error && !options.help) {
    error = amdf_info_selector_parse(options.device, &selector);
  }
  if (error) {
    fprintf(stderr, "amdf-info: %s\n", error);
    return AMDF_INFO_EXIT_USAGE;
  }
  if (options.help) {
    amdf_info_options_print_usage(stdout);
    return fflush(stdout) == 0 && !ferror(stdout) ? AMDF_INFO_EXIT_SUCCESS
                                                  : AMDF_INFO_EXIT_FAILURE;
  }
  const amdf_info_exit_code_t result =
      amdf_info_invoke(&options, selector, amdf_query_api, stdout);
  if (ferror(stdout)) {
    fprintf(stderr, "amdf-info: failed to write the report\n");
  }
  return result;
}
