// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <string.h>

#include "loomc/loomc.h"

static const char kSourceText[] = "func.def public @entry() { func.return }";

typedef struct source_info_state_t {
  // Source whose borrowed bytes are inspected by the example.
  loomc_source_t* source;
} source_info_state_t;

static void print_status(loomc_status_t status) {
  char buffer[1024] = {0};
  loomc_host_size_t length = 0;
  loomc_status_format(status, sizeof(buffer), buffer, &length);
  fprintf(stderr, "%.*s\n", (int)length, buffer);
}

static void source_info_state_initialize(source_info_state_t* state) {
  memset(state, 0, sizeof(*state));
}

static void source_info_state_deinitialize(source_info_state_t* state) {
  loomc_source_release(state->source);
}

static loomc_status_t create_source(source_info_state_t* state) {
  loomc_source_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(options),
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view("example.loom"),
      .contents = loomc_make_byte_span(kSourceText, sizeof(kSourceText) - 1),
      .storage = LOOMC_SOURCE_STORAGE_BORROWED,
  };
  return loomc_source_create(&options, loomc_allocator_system(),
                             &state->source);
}

static void print_source_info(const source_info_state_t* state) {
  loomc_string_view_t identifier = loomc_source_identifier(state->source);
  loomc_byte_span_t contents = loomc_source_contents(state->source);
  printf("%.*s: %zu bytes\n", (int)identifier.size, identifier.data,
         contents.data_length);
}

static loomc_status_t run_source_info_example(void) {
  source_info_state_t state;
  source_info_state_initialize(&state);

  loomc_status_t status = create_source(&state);
  if (loomc_status_is_ok(status)) {
    print_source_info(&state);
  }

  source_info_state_deinitialize(&state);
  return status;
}

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  loomc_status_t status = run_source_info_example();
  if (loomc_status_is_ok(status)) {
    return 0;
  }
  print_status(status);
  loomc_status_free(status);
  return 1;
}
