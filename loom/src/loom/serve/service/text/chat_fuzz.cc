// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "iree/base/api.h"
#include "iree/base/string_builder.h"
#include "loom/serve/runtime/program/compilation.h"
#include "loom/serve/runtime/program/sources.h"
#include "loom/serve/service/text/chat.h"
#include "loom/serve/testing/text_policy.h"
#include "loom/util/json.h"

namespace {

constexpr size_t kMaximumBodyLength = 8 * 1024 * 1024;
constexpr size_t kMaximumStructuredInputLength =
    (kMaximumBodyLength - 4096) / 6;
std::string control_path;
std::string tokenizer_path;

void CheckSourcePaths() {
  IREE_CHECK_OK(
      !control_path.empty() && !tokenizer_path.empty()
          ? iree_ok_status()
          : iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                             "chat fuzzer source paths are required"));
}

class Environment {
 public:
  Environment() {
    CheckSourcePaths();
    const loom_serve_sources_t sources = {
        .base_directory = IREE_SV("."),
        .control =
            iree_make_string_view(control_path.data(), control_path.size()),
    };
    loom_serve_compilation_report_t report = {};
    IREE_CHECK_OK(loom_serve_test_text_policy_allocate(
        &sources,
        iree_make_string_view(tokenizer_path.data(), tokenizer_path.size()),
        IREE_SV("{}"), &report, iree_allocator_system(), &owner_));
  }

  ~Environment() { loom_serve_test_text_policy_free(owner_); }

  const loom_serve_text_chat_policy_t* policy() const {
    return loom_serve_test_text_policy_chat(owner_);
  }

 private:
  // Real source program and VM environment shared across serial mutations.
  loom_serve_test_text_policy_t* owner_ = nullptr;
};

Environment& GetEnvironment() {
  static Environment environment;
  return environment;
}

std::string Quote(iree_string_view_t value) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_CHECK_OK(loom_json_write_escaped_string(&stream, value));
  const iree_string_view_t result = iree_string_builder_view(&builder);
  std::string text(result.data, result.size);
  iree_string_builder_deinitialize(&builder);
  return text;
}

std::string GenerateBody(const uint8_t* data, size_t size) {
  if (size > kMaximumStructuredInputLength) {
    return std::string(reinterpret_cast<const char*>(data), size);
  }
  const char mode = static_cast<char>(data[0]);
  const iree_string_view_t payload =
      iree_make_string_view(reinterpret_cast<const char*>(data + 1), size - 1);
  switch (mode) {
    case 'u':
      return std::string(
                 R"({"model":"portable-chat","messages":[{"role":"user","content":)") +
             Quote(payload) + "}]}";
    case 'p': {
      const iree_string_view_t first =
          iree_string_view_substr(payload, 0, payload.size / 2);
      const iree_string_view_t second = iree_string_view_substr(
          payload, payload.size / 2, IREE_HOST_SIZE_MAX);
      return std::string(
                 R"({"model":"portable-chat","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":[{"type":"text","text":)") +
             Quote(first) + R"(},{"type":"text","text":)" + Quote(second) +
             "}]}]}";
    }
    case 't': {
      const std::string arguments =
          std::string("{\"value\":") + Quote(payload) + "}";
      return std::string(
                 R"({"model":"portable-chat","tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],"messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"call","type":"function","function":{"name":"lookup","arguments":)") +
             Quote(iree_make_string_view(arguments.data(), arguments.size())) +
             R"(}}]},{"role":"tool","tool_call_id":"call","content":)" +
             Quote(payload) + "}]}";
    }
    default:
      return std::string(reinterpret_cast<const char*>(data), size);
  }
}

void Exercise(iree_string_view_t body, bool require_success) {
  const loom_serve_text_chat_policy_t* policy = GetEnvironment().policy();
  loom_serve_text_chat_t chat = {};
  iree_status_t status =
      loom_serve_text_chat_initialize(policy, policy->name, body, body.size,
                                      128, iree_allocator_system(), &chat);
  if (require_success && !iree_status_is_ok(status)) {
    IREE_CHECK_OK(status);
  }
  if (iree_status_is_ok(status)) {
    loom_serve_text_chat_deinitialize(&chat);
  }
  iree_status_free(status);
}

}  // namespace

extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv) {
  constexpr std::string_view kControlPrefix = "--loom_control=";
  constexpr std::string_view kTokenizerPrefix = "--loom_tokenizer=";
  int retained = 1;
  for (int i = 1; i < *argc; ++i) {
    const std::string_view argument((*argv)[i]);
    if (argument.compare(0, kControlPrefix.size(), kControlPrefix) == 0) {
      control_path.assign(argument.substr(kControlPrefix.size()));
    } else if (argument.compare(0, kTokenizerPrefix.size(), kTokenizerPrefix) ==
               0) {
      tokenizer_path.assign(argument.substr(kTokenizerPrefix.size()));
    } else {
      (*argv)[retained++] = (*argv)[i];
    }
  }
  *argc = retained;
  CheckSourcePaths();
  (void)GetEnvironment();
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (!size || size > kMaximumBodyLength) {
    return 0;
  }
  const bool structured = size <= kMaximumStructuredInputLength &&
                          (data[0] == 'u' || data[0] == 'p' || data[0] == 't');
  const std::string body = GenerateBody(data, size);
  Exercise(iree_make_string_view(body.data(), body.size()), structured);
  return 0;
}
