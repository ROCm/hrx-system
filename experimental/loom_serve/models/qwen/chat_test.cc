// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/models/qwen/chat.h"

#include <string>
#include <utility>

#include "experimental/loom_serve/runtime/input.h"
#include "experimental/loom_serve/runtime/json.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

IREE_FLAG(string, chat_source, "", "Production chat policy source.");

namespace {

const char kTools[] =
    R"([{"type":"function","function":{"name":"read","description":"Read text","parameters":{"type":"object","properties":{"path":{"type":"string"},"offset":{"type":"number"},"literal":{"type":"boolean"}},"required":["path"]}}}])";

std::string Request(const std::string& messages,
                    const std::string& options = "") {
  return "{\"model\":\"qwen3.8-27b\",\"stream\":true,\"messages\":" + messages +
         options + "}";
}

std::string View(const iree_string_builder_t& builder) {
  auto view = iree_string_builder_view(&builder);
  return std::string(view.data ? view.data : "", view.size);
}

class QwenChatTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_string_builder_initialize(iree_allocator_system(), &checkpoint_);
    iree_string_builder_initialize(iree_allocator_system(), &calls_);
    IREE_ASSERT_OK(
        iree_vm_environment_allocate(iree_allocator_system(), &environment_));
    IREE_ASSERT_OK(loom_serve_input_module_create(
        environment_, nullptr, &libraries_[0], iree_allocator_system()));
    IREE_ASSERT_OK(loom_serve_json_module_create(environment_, &libraries_[1],
                                                 iree_allocator_system()));
    const iree_string_view_t roots[] = {IREE_SVL("render_tool")};
    IREE_ASSERT_OK(loom_serve_program_create(
        environment_, iree_make_cstring_view(FLAG_chat_source), 1, roots,
        iree_vm_module_span_from_array(libraries_), iree_allocator_system(),
        &program_));
    IREE_ASSERT_OK(loom_serve_qwen_chat_policy_initialize(environment_,
                                                          program_, &policy_));
  }
  void TearDown() override {
    if (initialized_) {
      loom_serve_qwen_chat_deinitialize(&chat_);
    }
    iree_string_builder_deinitialize(&calls_);
    iree_string_builder_deinitialize(&checkpoint_);
    loom_serve_program_destroy(program_);
    iree_vm_module_release(libraries_[1]);
    iree_vm_module_release(libraries_[0]);
    iree_vm_environment_free(environment_);
  }
  iree_status_t Initialize(std::string body) {
    body_ = std::move(body);
    auto status = loom_serve_qwen_chat_initialize(
        &policy_, iree_make_string_view(body_.data(), body_.size()), 128,
        iree_allocator_system(), &chat_);
    initialized_ = iree_status_is_ok(status);
    return status;
  }
  iree_status_t Complete(const std::string& text) {
    iree_string_builder_reset(&checkpoint_);
    iree_string_builder_reset(&calls_);
    return loom_serve_qwen_chat_complete(
        &chat_, iree_make_string_view(text.data(), text.size()), 7,
        &checkpoint_, &calls_, &call_count_);
  }

  // Body storage borrowed by the initialized request.
  std::string body_;
  // Parsed request under test.
  loom_serve_qwen_chat_t chat_ = {};
  // Whether chat owns initialized storage.
  bool initialized_ = false;
  // Completed canonical transcript.
  iree_string_builder_t checkpoint_;
  // Serialized streaming tool-call delta.
  iree_string_builder_t calls_;
  // Number of completed tool calls.
  iree_host_size_t call_count_ = 0;
  // Environment outliving source policy and all request buffer references.
  iree_vm_environment_t* environment_ = nullptr;
  // Native JSON and input validation capabilities, with no model callbacks.
  iree_vm_module_t* libraries_[2] = {};
  // One source program shared by incoming and completed chat processing.
  loom_serve_program_t* program_ = nullptr;
  // Cold-resolved formatter in that program.
  loom_serve_qwen_chat_policy_t policy_ = {};
};

TEST_F(QwenChatTest, TextPartsAndNullHaveDistinctMeanings) {
  IREE_ASSERT_OK(Initialize(Request(
      R"([{"role":"system","content":" Be concise. "},{"role":"user","content":[{"type":"text","text":"Say "},{"type":"text","text":"\u03bb"}]},{"role":"assistant","content":null},{"role":"user","content":"null"}])",
      R"(,"max_tokens":17,"temperature":0,"stream_options":{"include_usage":true})")));
  EXPECT_EQ(chat_.max_tokens, 17u);
  EXPECT_TRUE(chat_.include_usage);
  EXPECT_EQ(View(chat_.prompt),
            "<|im_start|>system\nBe concise.<|im_end|>\n"
            "<|im_start|>user\nSay λ<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n<|im_end|>\n"
            "<|im_start|>user\nnull<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n");
  IREE_ASSERT_OK(Complete("  λ\n"));
  EXPECT_EQ(call_count_, 0u);
  EXPECT_EQ(View(calls_), "[]");
  EXPECT_EQ(View(checkpoint_), View(chat_.prompt) + "λ<|im_end|>\n");
}

TEST_F(QwenChatTest, GeneratedTypedCallMatchesPiToolHistoryCheckpoint) {
  const std::string tools = std::string(",\"tools\":") + kTools;
  IREE_ASSERT_OK(Initialize(Request(
      R"([{"role":"system","content":"Read only."},{"role":"user","content":"Read file."}])",
      tools)));
  IREE_ASSERT_OK(Complete(
      "I will read it.\n<tool_call><function=read>\n"
      "<parameter=path>\nfile λ.txt\n</parameter>"
      "<parameter=offset>7</parameter>"
      "<parameter=literal>false</parameter></function></tool_call>\n"));
  EXPECT_EQ(call_count_, 1u);
  EXPECT_EQ(
      View(calls_),
      "[{\"index\":0,\"id\":\"call_7_0\",\"type\":\"function\",\"function\":{"
      "\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\\\"file λ.txt\\\","
      "\\\"offset\\\":7,\\\"literal\\\":false}\"}}]");
  const std::string followup = Request(
      R"([{"role":"system","content":"Read only."},{"role":"user","content":"Read file."},{"role":"assistant","content":"I will read it.\n","tool_calls":[{"id":"call_7_0","type":"function","function":{"name":"read","arguments":"{\"path\":\"file λ.txt\",\"offset\":7,\"literal\":false}"}}]},{"role":"tool","tool_call_id":"call_7_0","content":"file contents"}])",
      tools);
  loom_serve_qwen_chat_t next;
  IREE_ASSERT_OK(loom_serve_qwen_chat_initialize(
      &policy_, iree_make_string_view(followup.data(), followup.size()), 128,
      iree_allocator_system(), &next));
  EXPECT_EQ(View(next.prompt),
            View(checkpoint_) +
                "<|im_start|>user\n<tool_response>\nfile contents\n"
                "</tool_response><|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n");
  loom_serve_qwen_chat_deinitialize(&next);
}

TEST_F(QwenChatTest, MultipleGeneratedCallsHaveDistinctIdsAndCanonicalOrder) {
  IREE_ASSERT_OK(
      Initialize(Request(R"([{"role":"user","content":"Read both."}])",
                         std::string(",\"tools\":") + kTools)));
  IREE_ASSERT_OK(
      Complete("<tool_call><function=read><parameter=path>a</parameter></"
               "function></tool_call>\n"
               "<tool_call><function=read><parameter=path>b</parameter></"
               "function></tool_call>"));
  EXPECT_EQ(call_count_, 2u);
  EXPECT_NE(View(calls_).find("call_7_1"), std::string::npos);
  EXPECT_NE(View(checkpoint_).find("</tool_call>\n<tool_call>"),
            std::string::npos);
}

TEST_F(QwenChatTest, HistoryFormatsEveryArgumentTypeInSource) {
  IREE_ASSERT_OK(Initialize(Request(
      R"([{"role":"user","content":"Continue."},{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"inspect_v2.run-1","arguments":"{\"s\":\"\\u03bb\\n\",\"n\":-1.25e+2,\"o\":{\"x\":true},\"a\":[false,null],\"t\":true,\"f\":false,\"z\":null,\"empty\":\"\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"done"}])")));
  EXPECT_EQ(View(chat_.prompt),
            "<|im_start|>user\nContinue.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            "<tool_call>\n<function=inspect_v2.run-1>\n"
            "<parameter=s>\nλ\n\n</parameter>\n"
            "<parameter=n>\n-1.25e+2\n</parameter>\n"
            "<parameter=o>\n{\"x\":true}\n</parameter>\n"
            "<parameter=a>\n[false,null]\n</parameter>\n"
            "<parameter=t>\ntrue\n</parameter>\n"
            "<parameter=f>\nfalse\n</parameter>\n"
            "<parameter=z>\nnull\n</parameter>\n"
            "<parameter=empty>\n\n</parameter>\n"
            "</function>\n</tool_call><|im_end|>\n"
            "<|im_start|>user\n<tool_response>\ndone\n"
            "</tool_response><|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n");
}

TEST_F(QwenChatTest, SourceRejectsInvalidIdentifiersThenAcceptsValidHistory) {
  for (const auto& name :
       {std::string(), std::string("has space"), std::string("<injection>"),
        std::string("λ"), std::string(129, 'a')}) {
    SCOPED_TRACE(name);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        Initialize(Request(
            R"([{"role":"user","content":"Continue."},{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":")" +
            name +
            R"(","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_1","content":"done"}])")));
  }
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Initialize(Request(
          R"([{"role":"user","content":"Continue."},{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read","arguments":"{\"\\u0078\":1}"}}]},{"role":"tool","tool_call_id":"call_1","content":"done"}])")));
  IREE_ASSERT_OK(Initialize(Request(
      R"([{"role":"user","content":"Continue."},{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_1","content":"done"}])")));
  EXPECT_NE(
      View(chat_.prompt)
          .find("<tool_call>\n<function=read>\n</function>\n</tool_call>"),
      std::string::npos);
}

TEST(QwenChatStreamingTest, EveryToolMarkerFragmentIsWithheld) {
  const std::string text = "Reading now.\n<tool_call>\n<function=read>";
  iree_host_size_t safe_end = 0;
  for (size_t length = 0; length <= text.size(); ++length) {
    safe_end = loom_serve_qwen_chat_text_end(
        iree_make_string_view(text.data(), length), safe_end);
    EXPECT_LE(safe_end, std::string("Reading now.\n").size());
    EXPECT_EQ(text.substr(0, safe_end).find('<'), std::string::npos);
  }
  EXPECT_EQ(safe_end, std::string("Reading now.\n").size());
  EXPECT_EQ(loom_serve_qwen_chat_text_end(IREE_SV("A < comparison"), 2), 14u);
}

TEST_F(QwenChatTest, UnsupportedOptionsFailAtTheBoundary) {
  const std::string messages = R"([{"role":"user","content":"Hello"}])";
  for (const char* options :
       {R"(,"temperature":0.7)", R"(,"stream":false)", R"(,"n":2)",
        R"(,"enable_thinking":true)", R"(,"max_tokens":0)",
        R"(,"max_tokens":1,"max_completion_tokens":2)",
        R"(,"tool_choice":"required")"}) {
    SCOPED_TRACE(options);
    const auto body = Request(messages, options);
    loom_serve_qwen_chat_t chat;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_qwen_chat_initialize(
            &policy_, iree_make_string_view(body.data(), body.size()), 128,
            iree_allocator_system(), &chat));
  }
  const auto body = Request(
      R"([{"role":"user","content":[{"type":"image_url","image_url":{}}]}])");
  loom_serve_qwen_chat_t chat;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_qwen_chat_initialize(
          &policy_, iree_make_string_view(body.data(), body.size()), 128,
          iree_allocator_system(), &chat));
}

TEST_F(QwenChatTest, InvalidGeneratedCallsNeverBecomeExecutableDeltas) {
  IREE_ASSERT_OK(Initialize(Request(R"([{"role":"user","content":"Read."}])",
                                    std::string(",\"tools\":") + kTools)));
  for (const char* text :
       {"<tool_call><function=unknown></function></tool_call>",
        "<tool_call><function=read></function></tool_call>",
        "<tool_call><function=read><parameter=path>a</"
        "parameter><parameter=path>b</parameter></function></tool_call>",
        "<tool_call><function=read><parameter=path>a</"
        "parameter><parameter=offset>oops</parameter></function></tool_call>",
        "<tool_call><function=read><parameter=path>a",
        "<tool_call><function=read><parameter=path>a</parameter></function></"
        "tool_call>suffix"}) {
    SCOPED_TRACE(text);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Complete(text));
  }
}

TEST_F(QwenChatTest, ToolHistoryRequiresOneCompleteArgumentsObject) {
  const auto body = Request(
      R"([{"role":"user","content":"Read."},{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"read","arguments":"{} trailing"}}]},{"role":"tool","tool_call_id":"call_1","content":"result"}])");
  loom_serve_qwen_chat_t chat;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_qwen_chat_initialize(
          &policy_, iree_make_string_view(body.data(), body.size()), 128,
          iree_allocator_system(), &chat));
}

TEST_F(QwenChatTest, InvalidGeneratedSyntaxHasABoundedPrintableDiagnostic) {
  IREE_ASSERT_OK(Initialize(Request(R"([{"role":"user","content":"Read."}])",
                                    std::string(",\"tools\":") + kTools)));
  for (const auto& [suffix, preview] :
       {std::pair<std::string, std::string>{"", ""},
        {"{\"name\":\"read\"}", "{\"name\":\"read\"}"},
        {"<function name=\"read\">", "<function name=\"read\">"},
        {"x\x1B[31m\ny", "x.[31m.y"},
        {std::string(100, 'x'), std::string(32, 'x') + "..."}}) {
    SCOPED_TRACE(preview);
    iree::Status status(Complete("<tool_call>" + suffix));
    EXPECT_EQ(status.code(), iree::StatusCode::kInvalidArgument);
    EXPECT_NE(
        status.ToString().find("expected '<function='; got '" + preview + "'"),
        std::string::npos);
    EXPECT_EQ(call_count_, 0u);
  }
}

TEST_F(QwenChatTest, StrictSamplingIsNotPretended) {
  const auto body = Request(
      R"([{"role":"user","content":"Read."}])",
      R"(,"tools":[{"type":"function","function":{"name":"read","strict":true,"parameters":{"type":"object","properties":{}}}}])");
  loom_serve_qwen_chat_t chat;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_qwen_chat_initialize(
          &policy_, iree_make_string_view(body.data(), body.size()), 128,
          iree_allocator_system(), &chat));
}

TEST_F(QwenChatTest, EventsCarryFinishReasonAndRetainedUsage) {
  IREE_ASSERT_OK(loom_serve_qwen_chat_event(
      7, IREE_SV("{\"content\":\"hello\"}"), IREE_SV("stop"), &calls_));
  IREE_ASSERT_OK(loom_serve_qwen_chat_usage(7, 120, 100, 4, &calls_));
  EXPECT_NE(View(calls_).find("\"finish_reason\":\"stop\""), std::string::npos);
  EXPECT_NE(
      View(calls_).find(
          "\"prompt_tokens\":120,\"completion_tokens\":4,\"total_tokens\":124"),
      std::string::npos);
  EXPECT_NE(View(calls_).find("\"cached_tokens\":100"), std::string::npos);
}

}  // namespace
