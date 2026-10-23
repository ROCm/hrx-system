// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/testing/text_policy.h"

#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "loom/model/card.h"
#include "loom/serve/runtime/program/input.h"
#include "loom/serve/runtime/program/json.h"
#include "loomc/iree.h"
#include "loomc/target/vm.h"

struct loom_serve_test_text_policy_t {
  // Allocator owning this record and its host resources.
  iree_allocator_t host_allocator;
  // Owned reference-type environment, outliving all program values.
  iree_vm_environment_t* environment;
  // Owned tokenizer borrowed by the program's input module.
  iree_tokenizer_t* tokenizer;
  // Owned resolved providers and card configuration borrowed by the program.
  loom_serve_sources_t* sources;
  // Owned JIT program and serialized invocation.
  loom_serve_program_t* program;
  // Resolved source functions and owned deployment defaults.
  loom_serve_text_chat_policy_t chat;
};

void loom_serve_test_text_policy_free(loom_serve_test_text_policy_t* policy) {
  if (!policy) {
    return;
  }
  loom_serve_text_chat_policy_deinitialize(&policy->chat);
  loom_serve_program_free(policy->program);
  loom_serve_sources_free(policy->sources, policy->host_allocator);
  iree_tokenizer_free(policy->tokenizer);
  iree_vm_environment_free(policy->environment);
  iree_allocator_free(policy->host_allocator, policy);
}

iree_status_t loom_serve_test_text_policy_allocate(
    const loom_serve_sources_t* requested_sources,
    iree_string_view_t tokenizer_path, iree_string_view_t options,
    loom_serve_compilation_report_t* report, iree_allocator_t host_allocator,
    loom_serve_test_text_policy_t** out_policy) {
  *out_policy = NULL;
  loom_serve_test_text_policy_t* policy = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*policy), (void**)&policy));
  policy->host_allocator = host_allocator;
  iree_status_t status = loom_serve_sources_resolve(
      requested_sources, host_allocator, &policy->sources);
  if (iree_status_is_ok(status) && policy->sources->card) {
    const loom_model_selection_t* selection =
        loom_model_card_selection(policy->sources->card);
    if (!iree_string_view_equal(selection->kind, IREE_SV("text")) ||
        (selection->version && selection->version != 1)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "request checks require text service version 1 or development");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_environment_allocate(host_allocator, &policy->environment);
  }
  iree_io_file_contents_t* tokenizer_contents = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_file_contents_read(tokenizer_path, host_allocator,
                                        &tokenizer_contents);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_from_huggingface_json(
        iree_make_string_view(
            (const char*)tokenizer_contents->const_buffer.data,
            tokenizer_contents->const_buffer.data_length),
        host_allocator, &policy->tokenizer);
  }
  iree_io_file_contents_free(tokenizer_contents);
  iree_vm_module_t* libraries[3] = {0};
  if (iree_status_is_ok(status)) {
    status = loom_serve_input_module_create(
        policy->environment, policy->tokenizer, host_allocator, &libraries[0]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_json_module_create(policy->environment, host_allocator,
                                           &libraries[1]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_chat_tools_module_create(
        policy->environment, host_allocator, &libraries[2]);
  }
  const iree_string_view_t roots[] = {
      IREE_SVL("model_name"),       IREE_SVL("chat_image_limits"),
      IREE_SVL("chat_begin"),       IREE_SVL("chat_message"),
      IREE_SVL("chat_end"),         IREE_SVL("render_tool"),
      IREE_SVL("parse_tools"),      IREE_SVL("prepare_input"),
      IREE_SVL("output_next"),      IREE_SVL("complete_text"),
      IREE_SVL("request_defaults"), IREE_SVL("request_options")};
  loomc_target_environment_t* target_environment = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_target_environment_create_vm(
        loomc_allocator_from_iree(host_allocator), &target_environment));
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_program_allocate_service(
        target_environment, policy->environment, policy->sources,
        policy->sources->control, IREE_ARRAYSIZE(roots), roots,
        iree_vm_module_span_from_array(libraries), report, host_allocator,
        &policy->program);
  }
  loomc_target_environment_release(target_environment);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(libraries); ++i) {
    iree_vm_module_release(libraries[i]);
  }
  iree_vm_buffer_t* options_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_clone(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span(options.data, options.size), 1,
        host_allocator, &options_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_chat_policy_initialize(
        policy->environment, policy->program, options_buffer, &policy->chat);
  }
  iree_vm_buffer_release(options_buffer);
  if (!iree_status_is_ok(status)) {
    loom_serve_test_text_policy_free(policy);
    return status;
  }
  *out_policy = policy;
  return iree_ok_status();
}

const loom_serve_text_chat_policy_t* loom_serve_test_text_policy_chat(
    const loom_serve_test_text_policy_t* policy) {
  return &policy->chat;
}
