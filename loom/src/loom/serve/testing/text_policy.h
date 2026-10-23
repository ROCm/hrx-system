// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_TESTING_TEXT_POLICY_H_
#define LOOM_SERVE_TESTING_TEXT_POLICY_H_

#include "loom/serve/runtime/program/sources.h"
#include "loom/serve/service/text/chat.h"

#ifdef __cplusplus
extern "C" {
#endif

// Host-only source-policy owner for request tools and API tests. Uses the
// production VM, tokenizer, and input/json/tools modules without a HAL device,
// model parameters, command compilation, or an HTTP listener.
typedef struct loom_serve_test_text_policy_t loom_serve_test_text_policy_t;

// Resolves the requested control program or model card, compiles its chat
// roots, and binds the supplied tokenizer and deployment options. Owns the
// resolved sources through policy destruction. Compilation evidence survives
// failure; the tokenizer path follows the process's working directory.
iree_status_t loom_serve_test_text_policy_allocate(
    const loom_serve_sources_t* requested_sources,
    iree_string_view_t tokenizer_path, iree_string_view_t options,
    loom_serve_compilation_report_t* report, iree_allocator_t host_allocator,
    loom_serve_test_text_policy_t** out_policy);

// Every chat and escaped VM reference has retired before destruction.
void loom_serve_test_text_policy_free(loom_serve_test_text_policy_t* policy);

// Borrowed source binding. The caller serializes policy calls, just as a model
// does, and preserves this owner through every chat and returned reference.
const loom_serve_text_chat_policy_t* loom_serve_test_text_policy_chat(
    const loom_serve_test_text_policy_t* policy);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_TESTING_TEXT_POLICY_H_
