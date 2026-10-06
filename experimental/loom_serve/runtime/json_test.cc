// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/json.h"

#include <array>
#include <string>
#include <string_view>
#include <utility>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"

namespace {

class JsonTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_vm_environment_allocate(allocator_, &environment_));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment_, IREE_SV("vm")),
        &types_));
    IREE_ASSERT_OK(
        loom_serve_json_module_create(environment_, &module_, allocator_));
    IREE_ASSERT_OK(
        iree_vm_program_create({module_, {}}, allocator_, &program_));
    IREE_ASSERT_OK(iree_vm_invocation_initialize(
        iree_make_byte_span(storage_, sizeof(storage_)), &invocation_));
    IREE_ASSERT_OK(iree_vm_process_create(program_, invocation_, {}, allocator_,
                                          &process_));
  }
  void TearDown() override {
    ResetCall();
    iree_vm_process_release(process_);
    if (invocation_) {
      iree_vm_invocation_deinitialize(invocation_);
    }
    iree_vm_program_release(program_);
    iree_vm_module_release(module_);
    iree_vm_environment_free(environment_);
  }
  void ResetCall() {
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments_));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results_));
  }
  iree_vm_variant_t Text(std::string_view text) {
    iree_vm_buffer_t* buffer = nullptr;
    IREE_CHECK_OK(iree_vm_buffer_clone(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span(text.data(), text.size()), 1, allocator_,
        &buffer));
    return iree_vm_buffer_variant_from_ptr_move(&types_, &buffer);
  }
  iree_status_t Invoke(iree_string_view_t name, iree_host_size_t argument_count,
                       iree_host_size_t result_count) {
    iree_vm_function_t function = {};
    IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
        process_, IREE_SV("json"), name, &function));
    return iree_vm_invoke(invocation_, function, {arguments_, argument_count},
                          {results_, result_count});
  }
  iree_status_t Members(std::string_view document) {
    ResetCall();
    arguments_[0] = Text(document);
    return Invoke(IREE_SV("members"), 1, 3);
  }
  iree_const_byte_span_t Records(int32_t expected_type,
                                 int64_t expected_count) {
    int32_t type = 0;
    int64_t count = 0;
    IREE_CHECK_OK(iree_vm_i32_from_variant(results_[1], &type));
    IREE_CHECK_OK(iree_vm_i64_from_variant(results_[2], &count));
    EXPECT_EQ(type, expected_type);
    EXPECT_EQ(count, expected_count);
    iree_vm_buffer_t* buffer = nullptr;
    IREE_CHECK_OK(iree_vm_buffer_ptr_from_variant_borrowed(&types_, results_[0],
                                                           &buffer));
    iree_const_byte_span_t bytes = {};
    IREE_CHECK_OK(iree_vm_buffer_map_read(
        buffer, 0, iree_vm_buffer_length(buffer), &bytes));
    EXPECT_EQ(bytes.data_length, expected_count * 5 * sizeof(int64_t));
    return bytes;
  }
  iree_status_t Unescape(std::string_view text, int64_t offset, int64_t length,
                         std::string& target, int64_t target_offset,
                         iree_vm_buffer_access_flags_t access) {
    ResetCall();
    arguments_[0] = Text(text);
    arguments_[1] = iree_vm_variant_from_i64(offset);
    arguments_[2] = iree_vm_variant_from_i64(length);
    iree_vm_buffer_t* buffer = nullptr;
    IREE_RETURN_IF_ERROR(iree_vm_buffer_wrap(
        access, iree_make_byte_span(target.data(), target.size()),
        iree_vm_buffer_release_callback_null(), allocator_, &buffer));
    arguments_[3] = iree_vm_buffer_variant_from_ptr_move(&types_, &buffer);
    arguments_[4] = iree_vm_variant_from_i64(target_offset);
    return Invoke(IREE_SV("unescape"), 5, 1);
  }

  // Allocation policy for the actual native module and VM process.
  const iree_allocator_t allocator_ = iree_allocator_system();
  // Environment outliving the process and all buffer references.
  iree_vm_environment_t* environment_ = nullptr;
  // Canonical buffer reference identity.
  iree_vm_ref_types_t types_ = {};
  // Native JSON implementation under test, also retained by the program.
  iree_vm_module_t* module_ = nullptr;
  // Linked native program using the same invocation API as source callers.
  iree_vm_program_t* program_ = nullptr;
  // One process reused across successful and rejected requests.
  iree_vm_process_t* process_ = nullptr;
  // Backing for synchronous execution of the native exports.
  alignas(iree_max_align_t) uint8_t storage_[16384] = {};
  // Invocation borrowing storage_ until teardown.
  iree_vm_invocation_t* invocation_ = nullptr;
  // Owned arguments, reset after structural rejection as well as execution.
  iree_vm_variant_t arguments_[5] = {};
  // Owned results, empty on failed invocation.
  iree_vm_variant_t results_[3] = {};
};

TEST_F(JsonTest, MembersPreserveTypesOffsetsAndOrder) {
  const std::string document =
      R"( /* input */ {"text":"\u03bb\n","number":-12.5e+2,"object":{"nested":[true,null]},"array":[0,"x"],"true":true,"false":false,"null":null,"duplicate":1,"duplicate":2,"empty":"","\u0061":"a",} /* end */ )";
  const struct {
    // Raw key spelling without JSON quotes.
    std::string_view key;
    // Raw value spelling, with string quotes excluded.
    std::string_view value;
    // Public JSON module type code.
    uint64_t type;
  } expected[] = {{"text", R"(\u03bb\n)", 0},
                  {"number", "-12.5e+2", 1},
                  {"object", R"({"nested":[true,null]})", 2},
                  {"array", R"([0,"x"])", 3},
                  {"true", "true", 4},
                  {"false", "false", 5},
                  {"null", "null", 6},
                  {"duplicate", "1", 1},
                  {"duplicate", "2", 1},
                  {"empty", "", 0},
                  {R"(\u0061)", "a", 0}};
  IREE_ASSERT_OK(Members(document));
  const auto records = Records(2, IREE_ARRAYSIZE(expected));
  for (size_t i = 0; i < IREE_ARRAYSIZE(expected); ++i) {
    SCOPED_TRACE(i);
    const uint8_t* record = records.data + i * 40;
    const uint64_t key_begin = iree_unaligned_load_le_u64(record);
    const uint64_t key_length = iree_unaligned_load_le_u64(record + 8);
    const uint64_t value_begin = iree_unaligned_load_le_u64(record + 16);
    const uint64_t value_length = iree_unaligned_load_le_u64(record + 24);
    ASSERT_LE(key_begin, document.size());
    ASSERT_LE(key_length, document.size() - key_begin);
    ASSERT_LE(value_begin, document.size());
    ASSERT_LE(value_length, document.size() - value_begin);
    EXPECT_EQ(document.substr(key_begin, key_length), expected[i].key);
    EXPECT_EQ(document.substr(value_begin, value_length), expected[i].value);
    EXPECT_EQ(iree_unaligned_load_le_u64(record + 32), expected[i].type);
  }
}

TEST_F(JsonTest, ArraysAndEmptyContainersHaveTheSameRecordContract) {
  const std::string document = R"(["",7,{},[],true,false,null])";
  const char* expected[] = {"", "7", "{}", "[]", "true", "false", "null"};
  IREE_ASSERT_OK(Members(document));
  auto records = Records(3, IREE_ARRAYSIZE(expected));
  for (size_t i = 0; i < IREE_ARRAYSIZE(expected); ++i) {
    const uint8_t* record = records.data + i * 40;
    EXPECT_EQ(iree_unaligned_load_le_u64(record), 0u);
    EXPECT_EQ(iree_unaligned_load_le_u64(record + 8), 0u);
    const uint64_t begin = iree_unaligned_load_le_u64(record + 16);
    const uint64_t length = iree_unaligned_load_le_u64(record + 24);
    ASSERT_LE(begin, document.size());
    ASSERT_LE(length, document.size() - begin);
    EXPECT_EQ(document.substr(begin, length), expected[i]);
    EXPECT_EQ(iree_unaligned_load_le_u64(record + 32), i);
  }
  IREE_ASSERT_OK(Members("{}"));
  Records(2, 0);
  IREE_ASSERT_OK(Members("[]"));
  Records(3, 0);
}

TEST_F(JsonTest, MalformedInputDoesNotPoisonTheSharedInvocation) {
  for (const char* document : {"", "true", "null", "42", "\"text\"", "{", "[",
                               "{\"x\":}", "[true]x", "{\"x\":\xff}"}) {
    SCOPED_TRACE(document);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Members(document));
    IREE_ASSERT_OK(Members("{}"));
    Records(2, 0);
  }
  ResetCall();
  arguments_[0] = iree_vm_buffer_variant_from_ptr_borrowed(&types_, nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Invoke(IREE_SV("members"), 1, 3));
}

TEST_F(JsonTest, UnescapeUsesOnlyTheProvidedRanges) {
  for (const auto& entry : {std::pair<std::string, std::string>{"", ""},
                            {R"(\"\\\/\b\f\n\r\t)", "\"\\/\b\f\n\r\t"},
                            {R"(λ\u03bb\uD83D\uDE80)", "λλ🚀"}}) {
    const std::string source = "!!" + entry.first + "!!";
    std::string target(entry.first.size() + 8, '#');
    IREE_ASSERT_OK(Unescape(
        source, 2, entry.first.size(), target, 3,
        IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE));
    int64_t written = 0;
    IREE_ASSERT_OK(iree_vm_i64_from_variant(results_[0], &written));
    EXPECT_EQ(written, entry.second.size());
    EXPECT_EQ(target.substr(0, 3), "###");
    EXPECT_EQ(target.substr(3, written), entry.second);
    EXPECT_EQ(target.substr(3 + written),
              std::string(target.size() - 3 - written, '#'));
  }
}

TEST_F(JsonTest, UnescapeRejectsMalformedRangesEscapesAndReadOnlyOutput) {
  std::string target(16, '#');
  const auto writable =
      IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE;
  for (const char* source :
       {"\\", R"(\q)", R"(\u0)", R"(\uXXXX)", R"(\uD800)", R"(\uDC00)"}) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          Unescape(source, 0, std::string_view(source).size(),
                                   target, 3, writable));
  }
  for (const auto& ranges : {std::array<int64_t, 3>{-1, 1, 0},
                             {0, -1, 0},
                             {2, 2, 0},
                             {0, 1, -1},
                             {0, 1, 17}}) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        Unescape("abc", ranges[0], ranges[1], target, ranges[2], writable));
  }
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      Unescape("a", 0, 1, target, 0, IREE_VM_BUFFER_ACCESS_FLAG_READ));
  EXPECT_EQ(target, std::string(16, '#'));
  IREE_ASSERT_OK(Unescape("a", 0, 1, target, 3, writable));
  EXPECT_EQ(target, "###a############");
}

TEST_F(JsonTest, InsufficientCapacityIsNotASizeQuery) {
  const auto writable =
      IREE_VM_BUFFER_ACCESS_FLAG_READ | IREE_VM_BUFFER_ACCESS_FLAG_WRITE;
  for (size_t capacity : {0u, 1u, 2u}) {
    std::string target(capacity, '#');
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          Unescape("abc", 0, 3, target, 0, writable));
  }
  std::string empty;
  IREE_ASSERT_OK(Unescape("", 0, 0, empty, 0, writable));
  int64_t written = -1;
  IREE_ASSERT_OK(iree_vm_i64_from_variant(results_[0], &written));
  EXPECT_EQ(written, 0);
}

}  // namespace
