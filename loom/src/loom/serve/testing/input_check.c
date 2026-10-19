// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Runs authored input checks with the serving tokenizer and native
// capabilities.
#include <stdio.h>
#include <stdlib.h>

#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/program/input.h"
#include "loom/serve/runtime/program/program.h"
#include "loom/util/stream.h"
#include "loomc/iree.h"
#include "loomc/target/vm.h"

IREE_FLAG(string, source, "",
          "Authored input check program (.loom or .loombc).");
IREE_FLAG_LIST(string, provider,
               "Additional source provider; may be repeated.");
IREE_FLAG(string, tokenizer, "", "Hugging Face tokenizer.json path.");
IREE_FLAG(string, entry, "check", "Zero-argument check entry point.");

static iree_status_t loom_serve_input_check_run(iree_allocator_t allocator) {
  if (!FLAG_source[0] || !FLAG_tokenizer[0] || !FLAG_entry[0]) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--source, --tokenizer, and --entry must be nonempty");
  }
  iree_vm_environment_t* environment = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_environment_allocate(allocator, &environment));
  iree_io_file_contents_t* contents = NULL;
  iree_status_t status = iree_io_file_contents_read(
      iree_make_cstring_view(FLAG_tokenizer), allocator, &contents);
  iree_tokenizer_t* tokenizer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_from_huggingface_json(
        iree_make_string_view((const char*)contents->const_buffer.data,
                              contents->const_buffer.data_length),
        allocator, &tokenizer);
  }
  iree_io_file_contents_free(contents);
  iree_vm_module_t* input = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_serve_input_module_create(environment, tokenizer, allocator,
                                            &input);
  }
  loomc_target_environment_t* target = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_target_environment_create_vm(
        loomc_allocator_from_iree(allocator), &target));
  }
  const iree_flag_string_list_t providers = FLAG_provider_list();
  const loom_serve_sources_t sources = {
      .base_directory = IREE_SV("."),
      .count = providers.count,
      .values = providers.values,
  };
  const iree_string_view_t entry = iree_make_cstring_view(FLAG_entry);
  loom_serve_compilation_report_t compilation = {0};
  loom_serve_program_t* program = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_serve_program_allocate(
        target, environment, &sources, iree_make_cstring_view(FLAG_source), 1,
        &entry, (iree_vm_module_span_t){&input, 1}, &compilation, allocator,
        &program);
  }
  iree_vm_function_t function = {0};
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_process_lookup_function(loom_serve_program_process(program),
                                        IREE_SV("model"), entry, &function);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(loom_serve_program_invocation(program), function,
                            iree_vm_variant_span_empty(),
                            iree_vm_variant_span_empty());
  }
  loom_serve_program_free(program);
  loomc_target_environment_release(target);
  iree_vm_module_release(input);
  iree_tokenizer_free(tokenizer);
  iree_vm_environment_free(environment);
  loom_output_stream_t output;
  loom_output_stream_for_file(stderr, &output);
  return iree_status_join(status,
                          loom_serve_compilation_write(&compilation, &output));
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  iree_status_t status = loom_serve_input_check_run(iree_allocator_system());
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
