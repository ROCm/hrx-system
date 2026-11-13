// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/module/string/module.h"

#include <array>
#include <initializer_list>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/sync.h"

namespace {

class StringModuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_vm_module_t* module = nullptr;
    IREE_ASSERT_OK(
        iree_vm_string_module_create(iree_allocator_system(), &module));
    iree_status_t status = iree_vm_program_create(
        {module, {}}, iree_allocator_system(), &program_);
    iree_vm_module_release(module);
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(iree_vm_invocation_initialize(
        iree_make_byte_span(storage_, sizeof(storage_)), &invocation_));
    IREE_ASSERT_OK(iree_vm_process_create(program_, invocation_, {},
                                          iree_allocator_system(), &process_));
  }

  void TearDown() override {
    iree_vm_process_release(process_);
    if (invocation_) {
      iree_vm_invocation_deinitialize(invocation_);
    }
    iree_vm_program_release(program_);
  }

  void Expect(const char* name, uint32_t codepoint,
              std::initializer_list<uint32_t> expected) {
    SCOPED_TRACE(::testing::Message() << name << "(" << codepoint << ")");
    iree_vm_function_t function = {};
    IREE_ASSERT_OK(iree_vm_process_lookup_function(
        process_, IREE_SV("string"), iree_make_cstring_view(name), &function));
    iree_vm_variant_t argument =
        iree_vm_variant_from_i32(static_cast<int32_t>(codepoint));
    std::array<iree_vm_variant_t, 4> results = {};
    IREE_ASSERT_OK(iree_vm_invoke(invocation_, function, {&argument, 1},
                                  {results.data(), expected.size()}));
    size_t index = 0;
    for (uint32_t value : expected) {
      int32_t actual = 0;
      IREE_ASSERT_OK(iree_vm_i32_from_variant(results[index++], &actual));
      EXPECT_EQ(static_cast<uint32_t>(actual), value);
    }
  }

  // Program retains the native module after the creation reference is released.
  iree_vm_program_t* program_ = nullptr;
  // Process reused across calls with different expansion lengths.
  iree_vm_process_t* process_ = nullptr;
  // Backing for synchronous invocation, with no environment or ref types.
  alignas(iree_max_align_t) uint8_t storage_[16384] = {};
  // Invocation borrowing storage_ until teardown.
  iree_vm_invocation_t* invocation_ = nullptr;
};

TEST_F(StringModuleTest, FullCaseMappingsAndUnusedResults) {
  Expect("codepoint.to_upper", 0x0390, {3, 0x0399, 0x0308, 0x0301});
  Expect("codepoint.to_upper", 0x00DF, {2, 'S', 'S', 0});
  Expect("codepoint.to_upper", 'a', {1, 'A', 0, 0});
  Expect("codepoint.to_upper", 0x1F80, {2, 0x1F08, 0x0399, 0});
  Expect("codepoint.to_upper", 0x1F88, {2, 0x1F08, 0x0399, 0});
  Expect("codepoint.to_upper", 0x01C5, {1, 0x01C4, 0, 0});
  Expect("codepoint.to_upper", 0x10428, {1, 0x10400, 0, 0});
  Expect("codepoint.to_lower", 0x0130, {2, 'i', 0x0307});
  Expect("codepoint.to_lower", 'A', {1, 'a', 0});
  Expect("codepoint.to_lower", 0x01C5, {1, 0x01C6, 0});
  Expect("codepoint.to_lower", 0x10400, {1, 0x10428, 0});
}

TEST_F(StringModuleTest, NumericTypesAndWhitespacePolicy) {
  for (uint32_t codepoint : {0x0030, 0x0039, 0x0661, 0x1D7CE, 0x1D7FF}) {
    Expect("codepoint.is_decimal", codepoint, {1});
    Expect("codepoint.is_digit", codepoint, {1});
  }
  for (uint32_t codepoint : {0x00B2, 0x2460}) {
    Expect("codepoint.is_decimal", codepoint, {0});
    Expect("codepoint.is_digit", codepoint, {1});
  }
  for (uint32_t codepoint : {0x00BC, 0x2160, 0x1D800}) {
    Expect("codepoint.is_decimal", codepoint, {0});
    Expect("codepoint.is_digit", codepoint, {0});
  }
  for (uint32_t codepoint : {0x0009, 0x000D, 0x0020, 0x0085, 0x00A0, 0x3000}) {
    Expect("codepoint.is_whitespace", codepoint, {1});
  }
  for (uint32_t codepoint : {0x001C, 0x001F, 0x200B, 0xFEFF}) {
    Expect("codepoint.is_whitespace", codepoint, {0});
  }
}

TEST_F(StringModuleTest, UnmappedAndInvalidInput) {
  for (uint32_t codepoint :
       {0u, 0xD800u, 0x1F642u, 0x10FFFFu, 0x110000u, 0xFFFFFFFFu}) {
    Expect("codepoint.to_upper", codepoint, {1, codepoint, 0, 0});
    Expect("codepoint.to_lower", codepoint, {1, codepoint, 0});
    Expect("codepoint.is_decimal", codepoint, {0});
    Expect("codepoint.is_digit", codepoint, {0});
    Expect("codepoint.is_whitespace", codepoint, {0});
  }
}

}  // namespace
