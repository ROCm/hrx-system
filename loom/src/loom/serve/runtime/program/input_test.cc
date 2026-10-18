// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/input.h"

#include <string>
#include <vector>

#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/program/program.h"
#include "loomc/iree.h"
#include "loomc/target/vm.h"

IREE_FLAG(string, input_source, "", "Source input capability callers.");

namespace {

class InputTest : public ::testing::TestWithParam<bool> {
 protected:
  // Copied compiler evidence retained beyond individual results.
  loom_serve_compilation_report_t compilation = {};

  void SetUp() override {
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_target_environment_create_vm(
        loomc_allocator_from_iree(host_allocator), &target_environment)));
    IREE_ASSERT_OK(iree_vm_environment_allocate(host_allocator, &environment));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &types));
    std::string configuration = R"({
          "model":{"type":"BPE","vocab":{
            "a":0,"b":1,"z":2,"zz":3,"<bos>":4,"<eos>":5,
            "▁":6,"é":7,"🦜":8,"end":10},"merges":[]},
          "decoder":{"type":"Metaspace","replacement":"▁",
            "prepend_scheme":"always"},
          "added_tokens":[
            {"id":3,"content":"zz","single_word":false,"lstrip":false,
             "rstrip":false,"normalized":false,"special":true},
            {"id":4,"content":"<bos>","single_word":false,"lstrip":false,
             "rstrip":false,"normalized":false,"special":true},
            {"id":5,"content":"<eos>","single_word":false,"lstrip":false,
             "rstrip":false,"normalized":false,"special":true}])";
    if (GetParam()) {
      configuration += R"(,
          "post_processor":{"type":"TemplateProcessing","single":[
            {"SpecialToken":{"id":"<bos>","type_id":0}},
            {"Sequence":{"id":"A","type_id":0}},
            {"SpecialToken":{"id":"<eos>","type_id":0}}],"pair":[],
            "special_tokens":{
              "<bos>":{"id":"<bos>","ids":[4],"tokens":["<bos>"]},
              "<eos>":{"id":"<eos>","ids":[5],"tokens":["<eos>"]}}})";
    }
    configuration += "}";
    LoadTokenizer(configuration);
  }
  void LoadTokenizer(const std::string& configuration) {
    iree_tokenizer_t* replacement = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_from_huggingface_json(
        iree_make_cstring_view(configuration.c_str()), host_allocator,
        &replacement));
    loom_serve_program_free(program);
    program = nullptr;
    iree_vm_module_release(module);
    module = nullptr;
    for (auto& value : retained) {
      iree_vm_variant_reset(&value);
    }
    retained.clear();
    iree_tokenizer_free(tokenizer);
    tokenizer = replacement;
    LoadProgram(tokenizer);
  }
  void LoadProgram(const iree_tokenizer_t* capability_tokenizer) {
    IREE_ASSERT_OK(loom_serve_input_module_create(
        environment, capability_tokenizer, host_allocator, &module));
    const iree_string_view_t roots[] = {
        IREE_SVL("tokens"),       IREE_SVL("find"),
        IREE_SVL("check"),        IREE_SVL("text"),
        IREE_SVL("new_encoder"),  IREE_SVL("reset_encoder"),
        IREE_SVL("feed_encoder"), IREE_SVL("finish_encoder"),
        IREE_SVL("new_decoder"),  IREE_SVL("reset_decoder"),
        IREE_SVL("feed_decoder"), IREE_SVL("finish_decoder")};
    const loom_serve_sources_t sources = {.base_directory = IREE_SV(".")};
    IREE_ASSERT_OK(loom_serve_program_allocate(
        target_environment, environment, &sources,
        iree_make_cstring_view(FLAG_input_source), IREE_ARRAYSIZE(roots), roots,
        (iree_vm_module_span_t){&module, 1}, &compilation, host_allocator,
        &program));
  }
  void TearDown() override {
    loom_serve_program_free(program);
    iree_vm_module_release(module);
    for (auto& value : retained) {
      iree_vm_variant_reset(&value);
    }
    retained.clear();
    iree_tokenizer_free(tokenizer);
    iree_vm_environment_free(environment);
    loomc_target_environment_release(target_environment);
  }
  iree_vm_variant_t Text(iree_string_view_t text) {
    iree_vm_buffer_t* buffer = nullptr;
    IREE_CHECK_OK(
        iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                             iree_make_const_byte_span(text.data, text.size), 1,
                             host_allocator, &buffer));
    return iree_vm_buffer_variant_from_ptr_move(&types, &buffer);
  }
  iree_status_t Invoke(iree_string_view_t entry,
                       iree_vm_variant_span_t arguments,
                       iree_vm_variant_span_t results) {
    iree_vm_function_t function = {};
    IREE_RETURN_IF_ERROR(
        iree_vm_process_lookup_function(loom_serve_program_process(program),
                                        IREE_SV("model"), entry, &function));
    return iree_vm_invoke(loom_serve_program_invocation(program), function,
                          arguments, results);
  }
  iree_vm_variant_t Tokens(const std::vector<int32_t>& tokens) {
    std::string bytes(tokens.size() * 4, '\0');
    for (size_t i = 0; i < tokens.size(); ++i) {
      iree_unaligned_store_le_u32(bytes.data() + i * 4, tokens[i]);
    }
    return Text(iree_make_string_view(bytes.data(), bytes.size()));
  }
  iree_vm_variant_t Keep(iree_vm_variant_t value) {
    retained.push_back(value);
    return iree_vm_variant_borrow(value);
  }
  iree_vm_variant_t Buffer(iree_host_size_t capacity) {
    iree_vm_buffer_t* buffer = nullptr;
    IREE_CHECK_OK(iree_vm_buffer_create(capacity, 1, host_allocator, &buffer));
    return Keep(iree_vm_buffer_variant_from_ptr_move(&types, &buffer));
  }
  iree_const_byte_span_t Read(iree_vm_variant_t value) {
    iree_vm_buffer_t* buffer = nullptr;
    IREE_CHECK_OK(
        iree_vm_buffer_ptr_from_variant_borrowed(&types, value, &buffer));
    iree_const_byte_span_t bytes = iree_const_byte_span_empty();
    IREE_CHECK_OK(iree_vm_buffer_map_read(
        buffer, 0, iree_vm_buffer_length(buffer), &bytes));
    return bytes;
  }
  iree_vm_variant_t Encoder(int32_t flags, int64_t token_capacity) {
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(flags),
                                     iree_vm_variant_from_i64(4096),
                                     iree_vm_variant_from_i64(token_capacity)};
    iree_vm_variant_t result = {};
    IREE_CHECK_OK(Invoke(IREE_SV("new_encoder"),
                         iree_vm_variant_span_from_array(arguments),
                         {&result, 1}));
    return Keep(result);
  }
  iree_vm_variant_t Decoder(int32_t flags, int64_t token_capacity) {
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(flags),
                                     iree_vm_variant_from_i64(token_capacity)};
    iree_vm_variant_t result = {};
    IREE_CHECK_OK(Invoke(IREE_SV("new_decoder"),
                         iree_vm_variant_span_from_array(arguments),
                         {&result, 1}));
    return Keep(result);
  }
  iree_status_t Reset(iree_string_view_t entry, iree_vm_variant_t state,
                      int32_t flags) {
    iree_vm_variant_t arguments[] = {iree_vm_variant_borrow(state),
                                     iree_vm_variant_from_i32(flags)};
    return Invoke(entry, iree_vm_variant_span_from_array(arguments),
                  iree_vm_variant_span_empty());
  }
  iree_status_t Feed(iree_string_view_t entry, iree_vm_variant_t state,
                     iree_vm_variant_t input, int64_t input_offset,
                     int64_t input_length, iree_vm_variant_t output,
                     int64_t output_offset, int64_t capacity,
                     int64_t* out_consumed, int64_t* out_written) {
    iree_vm_variant_t arguments[] = {iree_vm_variant_borrow(state),
                                     iree_vm_variant_borrow(input),
                                     iree_vm_variant_from_i64(input_offset),
                                     iree_vm_variant_from_i64(input_length),
                                     iree_vm_variant_borrow(output),
                                     iree_vm_variant_from_i64(output_offset),
                                     iree_vm_variant_from_i64(capacity)};
    iree_vm_variant_t results[2] = {};
    IREE_RETURN_IF_ERROR(Invoke(entry,
                                iree_vm_variant_span_from_array(arguments),
                                iree_vm_variant_span_from_array(results)));
    IREE_RETURN_IF_ERROR(iree_vm_i64_from_variant(results[0], out_consumed));
    return iree_vm_i64_from_variant(results[1], out_written);
  }
  iree_status_t Finish(iree_string_view_t entry, iree_vm_variant_t state,
                       iree_vm_variant_t output, int64_t output_offset,
                       int64_t capacity, int64_t* out_written,
                       bool* out_complete) {
    iree_vm_variant_t arguments[] = {iree_vm_variant_borrow(state),
                                     iree_vm_variant_borrow(output),
                                     iree_vm_variant_from_i64(output_offset),
                                     iree_vm_variant_from_i64(capacity)};
    iree_vm_variant_t results[2] = {};
    IREE_RETURN_IF_ERROR(Invoke(entry,
                                iree_vm_variant_span_from_array(arguments),
                                iree_vm_variant_span_from_array(results)));
    IREE_RETURN_IF_ERROR(iree_vm_i64_from_variant(results[0], out_written));
    int32_t complete = 0;
    IREE_RETURN_IF_ERROR(iree_vm_i32_from_variant(results[1], &complete));
    *out_complete = complete != 0;
    return iree_ok_status();
  }
  // Source-owned references retained independently of the reusable invocation.
  std::vector<iree_vm_variant_t> retained;
  // Allocator for the actual tokenizer, module and VM process.
  const iree_allocator_t host_allocator = iree_allocator_system();
  // Environment outliving every returned reference.
  iree_vm_environment_t* environment = nullptr;
  // VM-only grammar for the authored capability fixtures.
  loomc_target_environment_t* target_environment = nullptr;
  // Canonical VM buffer reference type.
  iree_vm_ref_types_t types = {};
  // Actual BPE tokenizer, borrowed by the input module.
  iree_tokenizer_t* tokenizer = nullptr;
  // Input module retained by the shared source program.
  iree_vm_module_t* module = nullptr;
  // Shared invocation used across requests and input failures.
  loom_serve_program_t* program = nullptr;
};

TEST_P(InputTest, BoundedTokensAndZeroPadding) {
  const uint32_t expected[] = {0, 1, 0, 0, 0};
  for (int64_t capacity : {0, 1, 3, 5, 9}) {
    iree_vm_variant_t arguments[] = {Text(IREE_SV("abaaa")),
                                     iree_vm_variant_from_i32(0),
                                     iree_vm_variant_from_i64(capacity)};
    iree_vm_variant_t results[2] = {};
    IREE_ASSERT_OK(Invoke(IREE_SV("tokens"),
                          iree_vm_variant_span_from_array(arguments),
                          iree_vm_variant_span_from_array(results)));
    int64_t count = 0;
    IREE_ASSERT_OK(iree_vm_i64_from_variant(results[1], &count));
    EXPECT_EQ(count, iree_min(capacity, 5));
    iree_vm_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(
        iree_vm_buffer_ptr_from_variant_borrowed(&types, results[0], &buffer));
    iree_const_byte_span_t bytes = iree_const_byte_span_empty();
    IREE_ASSERT_OK(iree_vm_buffer_map_read(
        buffer, 0, iree_vm_buffer_length(buffer), &bytes));
    ASSERT_EQ(bytes.data_length, capacity * 4);
    for (int64_t i = 0; i < capacity; ++i) {
      EXPECT_EQ(iree_unaligned_load_le_u32(bytes.data + i * 4),
                i < count ? expected[i] : 0);
    }
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  }
}

TEST_P(InputTest, BoundedModesMatchCompleteTokenizer) {
  const iree_tokenizer_encode_flags_t modes[] = {
      IREE_TOKENIZER_ENCODE_FLAG_NONE,
      IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      IREE_TOKENIZER_ENCODE_FLAG_NO_SPECIAL_TOKEN_MATCHING,
      IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS |
          IREE_TOKENIZER_ENCODE_FLAG_NO_SPECIAL_TOKEN_MATCHING};
  for (iree_tokenizer_encode_flags_t flags : modes) {
    for (const char* text : {"", "ab", "azzb"}) {
      int32_t expected[16] = {};
      iree_host_size_t expected_count = 0;
      IREE_ASSERT_OK(iree_tokenizer_encode(
          tokenizer, iree_make_cstring_view(text), flags,
          iree_tokenizer_make_token_output(expected, nullptr, nullptr,
                                           IREE_ARRAYSIZE(expected)),
          host_allocator, &expected_count));
      for (int64_t capacity : {0, 1, 3, 4, 8}) {
        SCOPED_TRACE(::testing::Message() << "flags=" << flags << " text="
                                          << text << " capacity=" << capacity);
        iree_vm_variant_t arguments[] = {Text(iree_make_cstring_view(text)),
                                         iree_vm_variant_from_i32(flags),
                                         iree_vm_variant_from_i64(capacity)};
        iree_vm_variant_t results[2] = {};
        IREE_ASSERT_OK(Invoke(IREE_SV("tokens"),
                              iree_vm_variant_span_from_array(arguments),
                              iree_vm_variant_span_from_array(results)));
        int64_t count = 0;
        IREE_ASSERT_OK(iree_vm_i64_from_variant(results[1], &count));
        EXPECT_EQ(count, iree_min(capacity, (int64_t)expected_count));
        iree_vm_buffer_t* buffer = nullptr;
        IREE_ASSERT_OK(iree_vm_buffer_ptr_from_variant_borrowed(
            &types, results[0], &buffer));
        iree_const_byte_span_t bytes = iree_const_byte_span_empty();
        IREE_ASSERT_OK(iree_vm_buffer_map_read(
            buffer, 0, iree_vm_buffer_length(buffer), &bytes));
        ASSERT_EQ(bytes.data_length, capacity * 4);
        for (int64_t i = 0; i < capacity; ++i) {
          EXPECT_EQ(iree_unaligned_load_le_u32(bytes.data + i * 4),
                    i < count ? (uint32_t)expected[i] : 0u);
        }
        iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
        iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
      }
    }
  }
}

TEST_P(InputTest, ComposedExpansionUsesOnlyRequestedPrefixCapacity) {
  const std::string configuration =
      R"({"normalizer":{"type":"Sequence","normalizers":[
        {"type":"Replace","pattern":{"String":"a"},"content":")" +
      std::string(32, 'b') + R"("},
        {"type":"Replace","pattern":{"String":"b"},"content":")" +
      std::string(32, 'c') +
      R"("}]},"model":{"type":"BPE","vocab":{"c":1},"merges":[]}})";
  LoadTokenizer(configuration);
  for (int64_t capacity : {0, 1, 31, 4096, 8192, 8193}) {
    SCOPED_TRACE(capacity);
    iree_vm_variant_t arguments[] = {Text(IREE_SV("aaaaaaaa")),
                                     iree_vm_variant_from_i32(0),
                                     iree_vm_variant_from_i64(capacity)};
    iree_vm_variant_t results[2] = {};
    IREE_ASSERT_OK(Invoke(IREE_SV("tokens"),
                          iree_vm_variant_span_from_array(arguments),
                          iree_vm_variant_span_from_array(results)));
    int64_t count = 0;
    IREE_ASSERT_OK(iree_vm_i64_from_variant(results[1], &count));
    EXPECT_EQ(count, iree_min(capacity, int64_t{8192}));
    iree_vm_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(
        iree_vm_buffer_ptr_from_variant_borrowed(&types, results[0], &buffer));
    iree_const_byte_span_t bytes = iree_const_byte_span_empty();
    IREE_ASSERT_OK(iree_vm_buffer_map_read(
        buffer, 0, iree_vm_buffer_length(buffer), &bytes));
    ASSERT_EQ(bytes.data_length, capacity * 4);
    for (int64_t i = 0; i < capacity; ++i) {
      EXPECT_EQ(iree_unaligned_load_le_u32(bytes.data + i * 4),
                i < count ? 1u : 0u);
    }
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  }
}

TEST_P(InputTest, IncompleteAndInvalidUtf8LeaveInvocationReusable) {
  LoadTokenizer(R"({"normalizer":{"type":"NFC"},
      "model":{"type":"BPE","vocab":{"a":0},"merges":[]}})");
  const iree_string_view_t invalid[] = {
      IREE_SVL("a\xc3"),
      IREE_SVL("a\xe2\x98"),
      IREE_SVL("a\xf0\x9f\xa6"),
      IREE_SVL("a\x80"),
      IREE_SVL("a\xc0\xaf"),
      IREE_SVL("a\xed\xa0\x80"),
      IREE_SVL("a\xf4\x90\x80\x80"),
  };
  for (iree_string_view_t text : invalid) {
    for (int64_t capacity : {0, 1, 16}) {
      iree_vm_variant_t arguments[] = {Text(text), iree_vm_variant_from_i32(0),
                                       iree_vm_variant_from_i64(capacity)};
      iree_vm_variant_t results[2] = {};
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_INVALID_ARGUMENT,
          Invoke(IREE_SV("tokens"), iree_vm_variant_span_from_array(arguments),
                 iree_vm_variant_span_from_array(results)));
      iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
      iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    }
    iree_vm_variant_t arguments[] = {Text(IREE_SV("a")),
                                     iree_vm_variant_from_i32(0),
                                     iree_vm_variant_from_i64(16)};
    iree_vm_variant_t results[2] = {};
    IREE_ASSERT_OK(Invoke(IREE_SV("tokens"),
                          iree_vm_variant_span_from_array(arguments),
                          iree_vm_variant_span_from_array(results)));
    int64_t count = 0;
    IREE_ASSERT_OK(iree_vm_i64_from_variant(results[1], &count));
    EXPECT_EQ(count, 1);
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  }
}

TEST_P(InputTest, VocabularyLookupDoesNotInventMissingTokens) {
  for (const auto& text : {"a", "b", "missing"}) {
    iree_vm_variant_t argument = Text(iree_make_cstring_view(text));
    iree_vm_variant_t result = {};
    IREE_ASSERT_OK(Invoke(IREE_SV("find"), {&argument, 1}, {&result, 1}));
    int32_t id = 0;
    IREE_ASSERT_OK(iree_vm_i32_from_variant(result, &id));
    EXPECT_EQ(id, std::string(text) == "a"   ? 0
                  : std::string(text) == "b" ? 1
                                             : -1);
    iree_vm_variant_reset(&argument);
    iree_vm_variant_reset(&result);
  }
}

TEST_P(InputTest, InputRejectionLeavesInvocationReusable) {
  for (int32_t condition : {0, 1, 0, 1}) {
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(condition),
                                     Text(IREE_SV("model input rejected"))};
    iree_status_t status =
        Invoke(IREE_SV("check"), iree_vm_variant_span_from_array(arguments),
               iree_vm_variant_span_empty());
    IREE_EXPECT_STATUS_IS(
        condition ? IREE_STATUS_OK : IREE_STATUS_INVALID_ARGUMENT, status);
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  }
  iree_vm_variant_t arguments[] = {Text(IREE_SV("a")),
                                   iree_vm_variant_from_i32(0),
                                   iree_vm_variant_from_i64(-1)};
  iree_vm_variant_t results[2] = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      Invoke(IREE_SV("tokens"), iree_vm_variant_span_from_array(arguments),
             iree_vm_variant_span_from_array(results)));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
}

TEST_P(InputTest, UnsupportedFlagsLeaveInvocationReusable) {
  for (int32_t flags : {1, 2, 6, -1, 0}) {
    iree_vm_variant_t arguments[] = {Text(IREE_SV("a")),
                                     iree_vm_variant_from_i32(flags),
                                     iree_vm_variant_from_i64(4)};
    iree_vm_variant_t results[2] = {};
    IREE_EXPECT_STATUS_IS(
        flags ? IREE_STATUS_INVALID_ARGUMENT : IREE_STATUS_OK,
        Invoke(IREE_SV("tokens"), iree_vm_variant_span_from_array(arguments),
               iree_vm_variant_span_from_array(results)));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  }
}

TEST_P(InputTest, CompleteDecodePreservesUnicodeAndResetsLeadingSpace) {
  struct loom_serve_test_input_case_t {
    // Complete token span passed to the source capability.
    std::vector<int32_t> tokens;
    // Tokenizer special-token behavior.
    int32_t flags;
    // Exact decoded UTF-8 bytes.
    std::string text;
  };
  const loom_serve_test_input_case_t cases[] = {
      {{}, 0, ""},
      {{0, 1}, 0, "ab"},
      {{6, 7, 6, 8}, 0, "é 🦜"},
      {{4, 6, 7, 6, 8, 5}, 1, "é 🦜"},
      {{4, 0, 3, 1, 5}, 0, "<bos>azzb<eos>"},
      {{4, 0, 3, 1, 5}, 1, "ab"},
      {{4, 3, 5}, 1, ""},
  };
  for (const auto& test : cases) {
    for (int64_t capacity = 0; capacity <= (int64_t)test.text.size() + 2;
         ++capacity) {
      SCOPED_TRACE(::testing::Message()
                   << "text=" << test.text << " capacity=" << capacity);
      iree_vm_variant_t arguments[] = {Tokens(test.tokens),
                                       iree_vm_variant_from_i32(test.flags),
                                       iree_vm_variant_from_i64(capacity)};
      iree_vm_variant_t results[2] = {};
      iree_status_t status =
          Invoke(IREE_SV("text"), iree_vm_variant_span_from_array(arguments),
                 iree_vm_variant_span_from_array(results));
      if (capacity < (int64_t)test.text.size()) {
        IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
        EXPECT_TRUE(iree_vm_variant_is_null(results[0]));
      } else {
        IREE_ASSERT_OK(status);
        int64_t length = 0;
        IREE_ASSERT_OK(iree_vm_i64_from_variant(results[1], &length));
        EXPECT_EQ(length, test.text.size());
        iree_vm_buffer_t* buffer = nullptr;
        IREE_ASSERT_OK(iree_vm_buffer_ptr_from_variant_borrowed(
            &types, results[0], &buffer));
        iree_const_byte_span_t bytes = iree_const_byte_span_empty();
        IREE_ASSERT_OK(iree_vm_buffer_map_read(
            buffer, 0, iree_vm_buffer_length(buffer), &bytes));
        EXPECT_EQ(bytes.data_length, capacity);
        EXPECT_EQ(std::string((const char*)bytes.data, length), test.text);
        for (int64_t i = length; i < capacity; ++i) {
          EXPECT_EQ(bytes.data[i], 0);
        }
      }
      iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
      iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    }
  }
}

TEST_P(InputTest, DecodeRejectsMissingIdsAndMalformedArguments) {
  for (int32_t id : {-1, 9, 11, 0}) {
    iree_vm_variant_t arguments[] = {Tokens({id}), iree_vm_variant_from_i32(0),
                                     iree_vm_variant_from_i64(4)};
    iree_vm_variant_t results[2] = {};
    IREE_EXPECT_STATUS_IS(
        id ? IREE_STATUS_INVALID_ARGUMENT : IREE_STATUS_OK,
        Invoke(IREE_SV("text"), iree_vm_variant_span_from_array(arguments),
               iree_vm_variant_span_from_array(results)));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  }
  for (int i = 0; i < 3; ++i) {
    iree_vm_variant_t arguments[] = {
        i == 0 ? Text(IREE_SV("abc")) : Tokens({0}),
        iree_vm_variant_from_i32(i == 1 ? 2 : 0),
        iree_vm_variant_from_i64(i == 2 ? -1 : 4)};
    iree_vm_variant_t results[2] = {};
    IREE_EXPECT_STATUS_IS(
        i == 2 ? IREE_STATUS_OUT_OF_RANGE : IREE_STATUS_INVALID_ARGUMENT,
        Invoke(IREE_SV("text"), iree_vm_variant_span_from_array(arguments),
               iree_vm_variant_span_from_array(results)));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  }
}

TEST_P(InputTest, DecodeWithoutTokenizerDoesNotDisableInputValidation) {
  loom_serve_program_free(program);
  program = nullptr;
  iree_vm_module_release(module);
  module = nullptr;
  LoadProgram(nullptr);
  iree_vm_variant_t arguments[] = {Tokens({0}), iree_vm_variant_from_i32(0),
                                   iree_vm_variant_from_i64(4)};
  iree_vm_variant_t results[2] = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      Invoke(IREE_SV("text"), iree_vm_variant_span_from_array(arguments),
             iree_vm_variant_span_from_array(results)));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  for (int32_t condition : {1, 0}) {
    iree_vm_variant_t check[] = {iree_vm_variant_from_i32(condition),
                                 Text(IREE_SV("cold input rejected"))};
    IREE_EXPECT_STATUS_IS(
        condition ? IREE_STATUS_OK : IREE_STATUS_INVALID_ARGUMENT,
        Invoke(IREE_SV("check"), iree_vm_variant_span_from_array(check),
               iree_vm_variant_span_empty()));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(check));
  }
}

TEST_P(InputTest, InterleavedEncodersPreserveStateAcrossInputAndOutputCredit) {
  const std::string text[] = {"ab", "azzbé🦜abazza"};
  const int32_t flags[] = {IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS, 0};
  auto output = Buffer(17);
  iree_vm_variant_t states[] = {Encoder(flags[0], 1), Encoder(flags[1], 1)};
  iree_vm_variant_t inputs[] = {
      Keep(Text(iree_make_cstring_view(text[0].c_str()))),
      Keep(Text(iree_make_cstring_view(text[1].c_str())))};
  int64_t available[2] = {};
  int64_t position[2] = {};
  bool complete[2] = {};
  std::vector<int32_t> actual[2];
  bool emitted_before_end = false;
  while (!complete[0] || !complete[1]) {
    for (int row = 0; row < 2; ++row) {
      if (complete[row]) {
        continue;
      }
      int64_t written = 0;
      if (position[row] < (int64_t)text[row].size()) {
        const bool new_input = available[row] < (int64_t)text[row].size();
        available[row] += new_input ? 1 : 0;
        int64_t consumed = 0;
        IREE_ASSERT_OK(Feed(IREE_SV("feed_encoder"), states[row], inputs[row],
                            position[row], available[row] - position[row],
                            output, 1, 0, &consumed, &written));
        const int64_t buffered = consumed;
        position[row] += consumed;
        EXPECT_EQ(written, 0);
        IREE_ASSERT_OK(Feed(IREE_SV("feed_encoder"), states[row], inputs[row],
                            position[row], available[row] - position[row],
                            output, 1, 4, &consumed, &written));
        ASSERT_TRUE(buffered || consumed || written || new_input);
        position[row] += consumed;
        emitted_before_end |=
            written && available[row] < (int64_t)text[row].size();
      } else {
        IREE_ASSERT_OK(Finish(IREE_SV("finish_encoder"), states[row], output, 1,
                              1, &written, &complete[row]));
        ASSERT_TRUE(written || complete[row]);
      }
      ASSERT_LE(written, 1);  // Creation-time staging bounds offered credit.
      if (written) {
        actual[row].push_back(
            iree_unaligned_load_le_u32(Read(output).data + 1));
      }
      EXPECT_EQ(Read(output).data[0], 0);
      EXPECT_EQ(Read(output).data[16], 0);
    }
  }
  EXPECT_TRUE(emitted_before_end);
  for (int row = 0; row < 2; ++row) {
    int32_t expected[32] = {};
    iree_host_size_t count = 0;
    IREE_ASSERT_OK(iree_tokenizer_encode(
        tokenizer, iree_make_cstring_view(text[row].c_str()), flags[row],
        iree_tokenizer_make_token_output(expected, nullptr, nullptr, 32),
        host_allocator, &count));
    EXPECT_EQ(actual[row], std::vector<int32_t>(expected, expected + count));
    int64_t written = 0;
    IREE_ASSERT_OK(Finish(IREE_SV("finish_encoder"), states[row], output, 1, 0,
                          &written, &complete[row]));
    EXPECT_EQ(written, 0);
    EXPECT_TRUE(complete[row]);
    int64_t consumed = 0;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_FAILED_PRECONDITION,
        Feed(IREE_SV("feed_encoder"), states[row], inputs[row], 0, 1, output, 1,
             1, &consumed, &written));
  }
}

TEST_P(InputTest, EncoderResetChangesFlagsAndDrainsFramingWithOneTokenCredit) {
  auto state = Encoder(0, 4);
  auto input = Keep(Text(IREE_SV("azzb")));
  auto output = Buffer(4);
  for (int32_t flags : {4, 8, 12, 0}) {
    IREE_ASSERT_OK(Reset(IREE_SV("reset_encoder"), state, flags));
    int64_t position = 0;
    bool complete = false;
    std::vector<int32_t> actual;
    while (!complete) {
      int64_t written = 0;
      if (position < 4) {
        int64_t consumed = 0;
        IREE_ASSERT_OK(Feed(IREE_SV("feed_encoder"), state, input, position,
                            4 - position, output, 0, 1, &consumed, &written));
        ASSERT_TRUE(consumed || written);
        position += consumed;
      } else {
        IREE_ASSERT_OK(Finish(IREE_SV("finish_encoder"), state, output, 0, 1,
                              &written, &complete));
        ASSERT_TRUE(written || complete);
      }
      if (written) {
        actual.push_back(iree_unaligned_load_le_u32(Read(output).data));
      }
    }
    std::vector<int32_t> expected = flags & 8 ? std::vector<int32_t>{0, 2, 2, 1}
                                              : std::vector<int32_t>{0, 3, 1};
    if ((flags & 4) && GetParam()) {
      expected.insert(expected.begin(), 4);
      expected.push_back(5);
    }
    EXPECT_EQ(actual, expected);
  }
  IREE_ASSERT_OK(Reset(IREE_SV("reset_encoder"), state, 4));
  int64_t written = 0;
  bool complete = false;
  IREE_ASSERT_OK(Finish(IREE_SV("finish_encoder"), state, output, 0, 0,
                        &written, &complete));
  EXPECT_EQ(written, 0);
  std::vector<int32_t> actual;
  while (!complete) {
    IREE_ASSERT_OK(Finish(IREE_SV("finish_encoder"), state, output, 0, 1,
                          &written, &complete));
    ASSERT_TRUE(written || complete);
    if (written) {
      actual.push_back(iree_unaligned_load_le_u32(Read(output).data));
    }
  }
  EXPECT_EQ(actual,
            GetParam() ? std::vector<int32_t>({4, 5}) : std::vector<int32_t>());
}

TEST_P(InputTest, DecoderPrefixesRemainIndependentOfResetSuffixes) {
  auto prefix = Decoder(1, 1);
  auto suffix = Decoder(0, 1);
  auto input = Keep(Tokens({4, 6, 7, 6, 8, 5}));
  auto output = Buffer(9);
  std::string actual;
  for (int64_t position = 0; position < 6; ++position) {
    int64_t consumed = 0;
    int64_t written = 0;
    IREE_ASSERT_OK(Feed(IREE_SV("feed_decoder"), prefix, input, position * 4,
                        6 - position, output, 1, 8, &consumed, &written));
    ASSERT_EQ(consumed, 1);
    actual.append((const char*)Read(output).data + 1, written);
    IREE_ASSERT_OK(Reset(IREE_SV("reset_decoder"), suffix, 0));
    IREE_ASSERT_OK(Feed(IREE_SV("feed_decoder"), suffix, input, 12, 3, output,
                        1, 8, &consumed, &written));
    EXPECT_EQ(consumed, 1);
    EXPECT_EQ(written, 0);  // A fresh suffix removes its leading metaspace.
  }
  EXPECT_EQ(actual, "é 🦜");
  int64_t written = 0;
  bool complete = false;
  IREE_ASSERT_OK(Finish(IREE_SV("finish_decoder"), prefix, output, 0, 0,
                        &written, &complete));
  EXPECT_EQ(written, 0);
  EXPECT_TRUE(complete);
  int64_t consumed = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        Feed(IREE_SV("feed_decoder"), prefix, input, 0, 1,
                             output, 0, 8, &consumed, &written));
  IREE_ASSERT_OK(Reset(IREE_SV("reset_decoder"), prefix, 0));
  IREE_ASSERT_OK(Feed(IREE_SV("feed_decoder"), prefix, input, 0, 1, output, 0,
                      8, &consumed, &written));
  EXPECT_EQ(std::string((const char*)Read(output).data, written), "<bos>");
}

TEST_P(InputTest, DecoderRetainsIncompleteUtf8UntilFinalOutputCredit) {
  LoadTokenizer(R"({"model":{"type":"BPE","vocab":{
      "<0xF0>":0,"<0x9F>":1,"<0xA6>":2,"<0x9C>":3},"merges":[]},
      "decoder":{"type":"ByteFallback"}})");
  auto state = Decoder(0, 2);
  auto input = Keep(Tokens({0, 1, 2, 3}));
  auto output = Buffer(4);
  int64_t consumed = 0;
  int64_t written = 0;
  IREE_ASSERT_OK(Feed(IREE_SV("feed_decoder"), state, input, 0, 4, output, 0, 4,
                      &consumed, &written));
  EXPECT_EQ(consumed, 2);
  EXPECT_EQ(written, 0);
  bool complete = false;
  IREE_ASSERT_OK(Finish(IREE_SV("finish_decoder"), state, output, 0, 0,
                        &written, &complete));
  EXPECT_FALSE(complete);
  IREE_ASSERT_OK(Finish(IREE_SV("finish_decoder"), state, output, 0, 2,
                        &written, &complete));
  EXPECT_EQ(written, 0);
  EXPECT_FALSE(complete);
  for (bool expected_complete : {false, true}) {
    IREE_ASSERT_OK(Finish(IREE_SV("finish_decoder"), state, output, 0, 3,
                          &written, &complete));
    EXPECT_EQ(complete, expected_complete);
    EXPECT_EQ(std::string((const char*)Read(output).data, written), "�");
  }
  IREE_ASSERT_OK(Reset(IREE_SV("reset_decoder"), state, 0));
  for (int64_t offset : {0, 8}) {
    IREE_ASSERT_OK(Feed(IREE_SV("feed_decoder"), state, input, offset, 2,
                        output, 0, 4, &consumed, &written));
    EXPECT_EQ(consumed, 2);
  }
  EXPECT_EQ(std::string((const char*)Read(output).data, written), "🦜");
}

TEST_P(InputTest, DecoderDrainsJoinedTextTransformAcrossCredit) {
  LoadTokenizer(R"({
    "model":{"type":"BPE","vocab":{
      "":0," ":1,"Good ":2,"da":3,"y ":4},"merges":[]},
    "decoder":{"type":"Sequence","decoders":[
      {"type":"Fuse"},
      {"type":"Strip","content":" ","start":2,"stop":2},
      {"type":"Replace","pattern":{"String":"day"},"content":"sunny day"}]}})");
  auto state = Decoder(0, 3);
  auto input = Keep(Tokens({0, 1, 0, 1, 2, 3, 4, 1, 1}));
  auto output = Buffer(33);
  for (int64_t credit : {4, 32}) {
    IREE_ASSERT_OK(Reset(IREE_SV("reset_decoder"), state, 0));
    std::string actual;
    int64_t position = 0;
    bool complete = false;
    while (!complete) {
      int64_t written = 0;
      if (position < 9) {
        int64_t consumed = 0;
        IREE_ASSERT_OK(Feed(IREE_SV("feed_decoder"), state, input, position * 4,
                            9 - position, output, 1, credit, &consumed,
                            &written));
        ASSERT_TRUE(consumed || written);
        position += consumed;
      } else {
        IREE_ASSERT_OK(Finish(IREE_SV("finish_decoder"), state, output, 1,
                              credit, &written, &complete));
        ASSERT_TRUE(written || complete);
      }
      actual.append((const char*)Read(output).data + 1, written);
    }
    // Strip removes two of the three trailing spaces.
    EXPECT_EQ(actual, "Good sunny day ");
  }
}

TEST_P(InputTest, CodecRangeAndFlagRejectionPreservesAcceptedState) {
  auto encoder = Encoder(0, 1);
  auto decoder = Decoder(0, 1);
  auto input = Keep(Text(IREE_SV("xa")));
  auto ids = Keep(Tokens({9, 0}));  // ID 9 is a vocabulary hole.
  auto output = Buffer(8);
  int64_t consumed = 0;
  int64_t written = 0;
  for (iree_string_view_t entry :
       {IREE_SV("reset_encoder"), IREE_SV("reset_decoder")}) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        Reset(entry,
              iree_string_view_equal(entry, IREE_SV("reset_encoder")) ? encoder
                                                                      : decoder,
              -1));
  }
  for (int64_t offset : {int64_t{-1}, int64_t{3}, INT64_MAX}) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                          Feed(IREE_SV("feed_encoder"), encoder, input, offset,
                               1, output, 0, 1, &consumed, &written));
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        Feed(IREE_SV("feed_encoder"), encoder, input, 0, 2,
                             output, 5, 1, &consumed, &written));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        Feed(IREE_SV("feed_decoder"), decoder, ids, 0,
                             INT64_MAX, output, 0, 8, &consumed, &written));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        Feed(IREE_SV("feed_encoder"), encoder, input, 0, 1, ids,
                             0, 1, &consumed, &written));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Feed(IREE_SV("feed_decoder"), decoder, ids, 0, 1,
                             output, 0, 8, &consumed, &written));
  IREE_ASSERT_OK(Feed(IREE_SV("feed_decoder"), decoder, ids, 4, 1, output, 1, 7,
                      &consumed, &written));
  EXPECT_EQ(consumed, 1);
  EXPECT_EQ(std::string((const char*)Read(output).data + 1, written), "a");
  IREE_ASSERT_OK(Feed(IREE_SV("feed_encoder"), encoder, input, 1, 1, output, 1,
                      1, &consumed, &written));
  EXPECT_EQ(consumed, 1);
  std::vector<uint32_t> actual;
  if (written) {
    actual.push_back(iree_unaligned_load_le_u32(Read(output).data + 1));
  }
  bool complete = false;
  while (!complete) {
    IREE_ASSERT_OK(Finish(IREE_SV("finish_encoder"), encoder, output, 1, 1,
                          &written, &complete));
    ASSERT_TRUE(written || complete);
    if (written) {
      actual.push_back(iree_unaligned_load_le_u32(Read(output).data + 1));
    }
  }
  EXPECT_EQ(actual, std::vector<uint32_t>({0}));
  // Source buffers can contain unaligned IDs and disjoint output ranges.
  IREE_ASSERT_OK(Feed(IREE_SV("feed_decoder"), decoder, output, 1, 1, output, 5,
                      3, &consumed, &written));
  EXPECT_EQ(consumed, 1);
  EXPECT_EQ(std::string((const char*)Read(output).data + 5, written), "a");
}

TEST_P(InputTest, CodecsOutliveCreatingProgramAndRejectInvalidCreation) {
  const int64_t invalid[][3] = {
      {-1, 4096, 1}, {0, 7, 1}, {0, 17, 1}, {0, 4096, 0}, {0, 4096, -1}};
  for (const auto& row : invalid) {
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(row[0]),
                                     iree_vm_variant_from_i64(row[1]),
                                     iree_vm_variant_from_i64(row[2])};
    iree_vm_variant_t result = {};
    IREE_EXPECT_STATUS_IS(
        row[2] < 0 ? IREE_STATUS_OUT_OF_RANGE : IREE_STATUS_INVALID_ARGUMENT,
        Invoke(IREE_SV("new_encoder"),
               iree_vm_variant_span_from_array(arguments), {&result, 1}));
    EXPECT_TRUE(iree_vm_variant_is_null(result));
  }
  for (int32_t flags : {-1, 0}) {
    iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(flags),
                                     iree_vm_variant_from_i64(0)};
    iree_vm_variant_t result = {};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        Invoke(IREE_SV("new_decoder"),
               iree_vm_variant_span_from_array(arguments), {&result, 1}));
    EXPECT_TRUE(iree_vm_variant_is_null(result));
  }
  auto encoder = Encoder(0, 1);
  auto decoder = Decoder(0, 1);
  auto input = Keep(Text(IREE_SV("a")));
  auto output = Buffer(4);
  loom_serve_program_free(program);
  program = nullptr;
  iree_vm_module_release(module);
  module = nullptr;
  // Objects borrow the model's tokenizer independently of module identity.
  LoadProgram(nullptr);
  int64_t consumed = 0;
  int64_t written = 0;
  IREE_ASSERT_OK(Feed(IREE_SV("feed_encoder"), encoder, input, 0, 1, output, 0,
                      1, &consumed, &written));
  EXPECT_EQ(consumed, 1);
  IREE_ASSERT_OK(Reset(IREE_SV("reset_decoder"), decoder, 1));
  // Fixture teardown destroys the replacement program before the last codec
  // owners, then frees the borrowed tokenizer after those owners retire.
}

INSTANTIATE_TEST_SUITE_P(FramingModes, InputTest, ::testing::Bool());

}  // namespace
