// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "iree/base/api.h"
#include "iree/base/internal/base64.h"
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
constexpr size_t kMaximumImageInputLength = 1024 * 1024 - 4096;
std::string control_path;
std::string media_control_path;
std::string tokenizer_path;

void CheckSourcePaths() {
  IREE_CHECK_OK(
      !control_path.empty() && !media_control_path.empty() &&
              !tokenizer_path.empty()
          ? iree_ok_status()
          : iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                             "chat fuzzer source paths are required"));
}

class Environment {
 public:
  explicit Environment(const std::string& source_path) {
    const loom_serve_sources_t sources = {
        .base_directory = IREE_SV("."),
        .control =
            iree_make_string_view(source_path.data(), source_path.size()),
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

class Environments {
 public:
  Environments() : chat(control_path), media(media_control_path) {}

  // Tool-capable independent source used for arbitrary text requests.
  Environment chat;
  // Image-capable source used for PAM-backed media requests.
  Environment media;
};

Environments& GetEnvironments() {
  static Environments environments;
  return environments;
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

std::string EncodePamDataUrl(iree_const_byte_span_t data) {
  std::string encoded(iree_base64_encoded_size(data.data_length), '\0');
  iree_host_size_t encoded_length = 0;
  IREE_CHECK_OK(iree_base64_encode(
      data, iree_make_mutable_string_view(encoded.data(), encoded.size()),
      &encoded_length));
  encoded.resize(encoded_length);
  return "data:image/x-portable-arbitrarymap;base64," + encoded;
}

std::string GeneratePamDataUrl(iree_const_byte_span_t payload) {
  const size_t supplied_bytes = payload.data_length ? payload.data_length : 1;
  const size_t pixel_count = (supplied_bytes + 2) / 3;
  const size_t width = std::min(pixel_count, size_t{256});
  const size_t height = (pixel_count + width - 1) / width;
  const size_t pixel_bytes = width * height * 3;
  std::string pam = "P7\nWIDTH " + std::to_string(width) + "\nHEIGHT " +
                    std::to_string(height) +
                    "\nDEPTH 3\nMAXVAL 255\nTUPLTYPE RGB\nENDHDR\n";
  const size_t header_length = pam.size();
  pam.resize(header_length + pixel_bytes, '\0');
  if (payload.data_length) {
    std::copy(payload.data, payload.data + payload.data_length,
              pam.begin() + header_length);
  }

  return EncodePamDataUrl(iree_make_const_byte_span(pam.data(), pam.size()));
}

std::string GenerateImageBody(const std::string& model,
                              const std::string& url) {
  return std::string(R"({"model":)") + model +
         R"(,"messages":[{"role":"user","content":[{"type":"text","text":"before"},{"type":"image_url","image_url":{"url":)" +
         Quote(iree_make_string_view(url.data(), url.size())) +
         R"(,"detail":"auto"}},{"type":"text","text":"after"}]}]})";
}

std::string GenerateBody(iree_string_view_t model_name, const uint8_t* data,
                         size_t size) {
  if (size > kMaximumStructuredInputLength) {
    return std::string(reinterpret_cast<const char*>(data), size);
  }
  const char mode = static_cast<char>(data[0]);
  const iree_string_view_t payload =
      iree_make_string_view(reinterpret_cast<const char*>(data + 1), size - 1);
  const std::string model = Quote(model_name);
  switch (mode) {
    case 'u':
      return std::string(R"({"model":)") + model +
             R"(,"messages":[{"role":"user","content":)" + Quote(payload) +
             "}]}";
    case 'p': {
      const iree_string_view_t first =
          iree_string_view_substr(payload, 0, payload.size / 2);
      const iree_string_view_t second = iree_string_view_substr(
          payload, payload.size / 2, IREE_HOST_SIZE_MAX);
      return std::string(R"({"model":)") + model +
             R"(,"stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":[{"type":"text","text":)" +
             Quote(first) + R"(},{"type":"text","text":)" + Quote(second) +
             "}]}]}";
    }
    case 't': {
      const std::string arguments =
          std::string("{\"value\":") + Quote(payload) + "}";
      return std::string(R"({"model":)") + model +
             R"(,"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],"messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"call","type":"function","function":{"name":"lookup","arguments":)" +
             Quote(iree_make_string_view(arguments.data(), arguments.size())) +
             R"(}}]},{"role":"tool","tool_call_id":"call","content":)" +
             Quote(payload) + "}]}";
    }
    case 'i': {
      if (size > kMaximumImageInputLength) {
        return std::string(reinterpret_cast<const char*>(data), size);
      }
      const std::string url =
          GeneratePamDataUrl(iree_make_const_byte_span(data + 1, size - 1));
      return GenerateImageBody(model, url);
    }
    case 'a': {
      if (size > kMaximumImageInputLength) {
        return std::string(reinterpret_cast<const char*>(data), size);
      }
      const std::string url =
          EncodePamDataUrl(iree_make_const_byte_span(data + 1, size - 1));
      return GenerateImageBody(model, url);
    }
    default:
      return std::string(reinterpret_cast<const char*>(data), size);
  }
}

void Exercise(const loom_serve_text_chat_policy_t* policy,
              iree_string_view_t body, bool require_success) {
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
  constexpr std::string_view kMediaControlPrefix = "--loom_media_control=";
  constexpr std::string_view kTokenizerPrefix = "--loom_tokenizer=";
  int retained = 1;
  for (int i = 1; i < *argc; ++i) {
    const std::string_view argument((*argv)[i]);
    if (argument.compare(0, kControlPrefix.size(), kControlPrefix) == 0) {
      control_path.assign(argument.substr(kControlPrefix.size()));
    } else if (argument.compare(0, kMediaControlPrefix.size(),
                                kMediaControlPrefix) == 0) {
      media_control_path.assign(argument.substr(kMediaControlPrefix.size()));
    } else if (argument.compare(0, kTokenizerPrefix.size(), kTokenizerPrefix) ==
               0) {
      tokenizer_path.assign(argument.substr(kTokenizerPrefix.size()));
    } else {
      (*argv)[retained++] = (*argv)[i];
    }
  }
  *argc = retained;
  CheckSourcePaths();
  (void)GetEnvironments();
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (!size || size > kMaximumBodyLength) {
    return 0;
  }
  Environments& environments = GetEnvironments();
  const bool media = data[0] == 'i' || data[0] == 'a';
  const loom_serve_text_chat_policy_t* policy =
      media ? environments.media.policy() : environments.chat.policy();
  const bool structured =
      size <= kMaximumStructuredInputLength &&
      (data[0] == 'u' || data[0] == 'p' || data[0] == 't' ||
       (data[0] == 'i' && size <= kMaximumImageInputLength));
  const std::string body = GenerateBody(policy->name, data, size);
  Exercise(policy, iree_make_string_view(body.data(), body.size()), structured);
  return 0;
}
