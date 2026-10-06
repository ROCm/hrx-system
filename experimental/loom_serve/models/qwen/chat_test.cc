// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/models/qwen/chat.h"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "experimental/loom_serve/runtime/input.h"
#include "experimental/loom_serve/runtime/json.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"

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
    // A real BPE tokenizer with every ASCII character, Unicode content and
    // explicit postprocessing detects accidental special-token insertion.
    std::string configuration = R"({"model":{"type":"BPE","vocab":{)";
    for (int i = 0; i < 128; ++i) {
      char entry[32];
      snprintf(entry, sizeof(entry), "%s\"\\u%04x\":%d", i ? "," : "", i, i);
      configuration += entry;
    }
    configuration += R"(,"<bos>":128,"<eos>":129,"λ":130},"merges":[]},
      "added_tokens":[
        {"id":128,"content":"<bos>","single_word":false,"lstrip":false,
         "rstrip":false,"normalized":false,"special":true},
        {"id":129,"content":"<eos>","single_word":false,"lstrip":false,
         "rstrip":false,"normalized":false,"special":true}],
      "post_processor":{"type":"TemplateProcessing","single":[
        {"SpecialToken":{"id":"<bos>","type_id":0}},
        {"Sequence":{"id":"A","type_id":0}},
        {"SpecialToken":{"id":"<eos>","type_id":0}}],"pair":[],
        "special_tokens":{
          "<bos>":{"id":"<bos>","ids":[128],"tokens":["<bos>"]},
          "<eos>":{"id":"<eos>","ids":[129],"tokens":["<eos>"]}}}})";
    IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
        iree_make_cstring_view(configuration.c_str()), iree_allocator_system(),
        &tokenizer_));
    IREE_ASSERT_OK(loom_serve_input_module_create(
        environment_, tokenizer_, &libraries_[0], iree_allocator_system()));
    IREE_ASSERT_OK(loom_serve_json_module_create(environment_, &libraries_[1],
                                                 iree_allocator_system()));
    const iree_string_view_t roots[] = {IREE_SVL("render_tool"),
                                        IREE_SVL("prepare_input")};
    IREE_ASSERT_OK(loom_serve_program_create(
        environment_, iree_make_cstring_view(FLAG_chat_source),
        IREE_ARRAYSIZE(roots), roots,
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
    iree_tokenizer_free(tokenizer_);
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
  // Actual tokenizer borrowed by the native input capability.
  iree_tokenizer_t* tokenizer_ = nullptr;
  // Native JSON and input validation capabilities, with no model callbacks.
  iree_vm_module_t* libraries_[2] = {};
  // One source program shared by incoming and completed chat processing.
  loom_serve_program_t* program_ = nullptr;
  // Cold-resolved formatter in that program.
  loom_serve_qwen_chat_policy_t policy_ = {};
};

TEST_F(QwenChatTest, CompleteInputMatchesEveryTurnFormat) {
  for (auto format :
       {LOOM_SERVE_QWEN_CHAT_INPUT_RENDERED, LOOM_SERVE_QWEN_CHAT_INPUT_USER}) {
    for (auto boundary : {LOOM_SERVE_QWEN_CHAT_BOUNDARY_FRESH,
                          LOOM_SERVE_QWEN_CHAT_BOUNDARY_OPEN,
                          LOOM_SERVE_QWEN_CHAT_BOUNDARY_TERMINATED}) {
      for (const std::string text : {"", "hello λ", "<bos>"}) {
        SCOPED_TRACE(::testing::Message() << "format=" << format << " boundary="
                                          << boundary << " text=" << text);
        std::string rendered;
        if (boundary == LOOM_SERVE_QWEN_CHAT_BOUNDARY_OPEN) {
          rendered = "<|im_end|>\n";
        } else if (boundary == LOOM_SERVE_QWEN_CHAT_BOUNDARY_TERMINATED) {
          rendered = "\n";
        }
        if (format == LOOM_SERVE_QWEN_CHAT_INPUT_USER) {
          rendered += "<|im_start|>user\n" + text +
                      "<|im_end|>\n"
                      "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        } else {
          rendered += text;
        }
        int32_t reference[512] = {};
        iree_host_size_t reference_count = 0;
        IREE_ASSERT_OK(iree_tokenizer_encode(
            tokenizer_, iree_make_cstring_view(rendered.c_str()),
            IREE_TOKENIZER_ENCODE_FLAG_NONE,
            iree_tokenizer_make_token_output(reference, nullptr, nullptr,
                                             IREE_ARRAYSIZE(reference)),
            iree_allocator_system(), &reference_count));
        for (iree_host_size_t capacity :
             {iree_host_size_t{0}, reference_count / 2, reference_count,
              reference_count + 3}) {
          SCOPED_TRACE(capacity);
          std::vector<int32_t> output(capacity + 1, -7);
          iree_host_size_t count = 99;
          iree_status_t status = loom_serve_qwen_chat_prepare_input(
              &policy_, iree_make_cstring_view(text.c_str()), format, boundary,
              capacity, output.data(), &count, iree_allocator_system());
          if (capacity < reference_count) {
            IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
            EXPECT_EQ(count, 0u);
            EXPECT_EQ(output, std::vector<int32_t>(capacity + 1, -7));
          } else {
            IREE_ASSERT_OK(status);
            EXPECT_EQ(count, reference_count);
            EXPECT_EQ(
                std::vector<int32_t>(output.begin(), output.begin() + count),
                std::vector<int32_t>(reference, reference + reference_count));
            // Neither a successful source call nor a rejected one owns the
            // caller's unpopulated token tail.
            for (iree_host_size_t i = count; i < output.size(); ++i) {
              EXPECT_EQ(output[i], -7);
            }
          }
        }
      }
    }
  }
}

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
