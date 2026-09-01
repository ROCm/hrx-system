// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/tooling/dump.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/bytecode/module_test_data.h"
#include "iree/vm/bytecode/wire/module_format.h"

namespace iree::vm::bytecode::testing {
namespace {

struct StringSink {
  // Complete text received from successful callback invocations.
  std::string value;
  // Number of successful callback invocations permitted before failure.
  size_t write_limit = std::numeric_limits<size_t>::max();
  // Number of callback invocations observed.
  size_t write_count = 0;
};

iree_status_t AppendToString(void* user_data, iree_string_view_t text) {
  auto* sink = static_cast<StringSink*>(user_data);
  if (sink->write_count == sink->write_limit) {
    return iree_make_status(IREE_STATUS_ABORTED, "sink rejected output");
  }
  ++sink->write_count;
  sink->value.append(text.data, text.size);
  return iree_ok_status();
}

iree_status_t DumpImage(const std::vector<uint8_t>& image, StringSink* sink) {
  return iree_vm_bytecode_module_dump(
      IREE_SV("test.module"),
      iree_make_const_byte_span(image.data(), image.size()),
      {AppendToString, sink}, iree_allocator_system());
}

uint8_t* FindFirstInstruction(std::vector<uint8_t>* image) {
  return FindFunctionImage(image, 0).bytecode;
}

TEST(VMBytecodeDumpTest, DumpsCompleteOwnershipModuleDeterministically) {
  const std::vector<uint8_t> image = BuildOwnershipModuleImage();
  StringSink first_sink;
  IREE_ASSERT_OK(DumpImage(image, &first_sink));
  StringSink second_sink;
  IREE_ASSERT_OK(DumpImage(image, &second_sink));

  EXPECT_EQ(first_sink.value, second_sink.value);
  EXPECT_NE(first_sink.value.find("vm.module \"test.module\" core=0.0"),
            std::string::npos);
  EXPECT_NE(first_sink.value.find("ref<vm, buffer>"), std::string::npos);
  EXPECT_NE(first_sink.value.find("@run -> function[1]"), std::string::npos);
  EXPECT_NE(first_sink.value.find(
                "authored_type = \"(i32) -> (i32, !vm.ref<vm, buffer>)\""),
            std::string::npos);
  EXPECT_NE(first_sink.value.find(
                "documentation = \"Adds the process seed and returns image "
                "rodata.\""),
            std::string::npos);
  EXPECT_NE(first_sink.value.find("\"model.kind\" = utf8(\"ownership\")"),
            std::string::npos);
  EXPECT_NE(first_sink.value.find("\"result.note\" = utf8(\"stable\")"),
            std::string::npos);
  EXPECT_NE(first_sink.value.find("control.block"), std::string::npos);
  EXPECT_NE(first_sink.value.find("constant.s16"), std::string::npos);
  EXPECT_NE(first_sink.value.find("global.value.immutable.store"),
            std::string::npos);
  EXPECT_NE(first_sink.value.find("buffer.rodata.load"), std::string::npos);
}

TEST(VMBytecodeDumpTest, DumpsHALPageAndOptionalImportWithoutProviders) {
  const std::vector<uint8_t> image = BuildHALInspectionModuleImage();
  StringSink sink;
  IREE_ASSERT_OK(DumpImage(image, &sink));

  EXPECT_NE(sink.value.find("page=0xF0 version=0.0"), std::string::npos);
  EXPECT_NE(sink.value.find("ref<hal, device_group>"), std::string::npos);
  EXPECT_NE(sink.value.find("@runtime.support::query_device_count : (ref<hal, "
                            "device_group>) -> (i64) optional"),
            std::string::npos);
  EXPECT_NE(sink.value.find("hal.device.group.count"), std::string::npos);
  EXPECT_EQ(sink.value.find("documentation ="), std::string::npos);
  EXPECT_EQ(sink.value.find("authored_type ="), std::string::npos);
}

TEST(VMBytecodeDumpTest, DumpsFunctionLocalSwitchTargets) {
  const std::vector<uint8_t> image = BuildSwitchInspectionModuleImage();
  StringSink sink;
  IREE_ASSERT_OK(DumpImage(image, &sink));

  EXPECT_NE(sink.value.find("switch_targets=1"), std::string::npos);
  EXPECT_NE(sink.value.find("switch_target[0] = +12 (word=3)"),
            std::string::npos);
  EXPECT_NE(sink.value.find("control.switch selector_v8=0 target_count_u16=1 "
                            "target_base_u32=0"),
            std::string::npos);
}

TEST(VMBytecodeDumpTest, RejectsUnknownInstructionAfterStructuralInspection) {
  std::vector<uint8_t> image = BuildOwnershipModuleImage();
  uint8_t* instruction = FindFirstInstruction(&image);
  ASSERT_NE(instruction, nullptr);
  instruction[0] = 0;

  StringSink sink;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, DumpImage(image, &sink));
}

TEST(VMBytecodeDumpTest, RejectsNoncanonicalInstructionPadding) {
  std::vector<uint8_t> image = BuildOwnershipModuleImage();
  uint8_t* instruction = FindFirstInstruction(&image);
  ASSERT_NE(instruction, nullptr);
  instruction[1] = 1;

  StringSink sink;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, DumpImage(image, &sink));
}

TEST(VMBytecodeDumpTest, PropagatesWriteFailure) {
  const std::vector<uint8_t> image = BuildOwnershipModuleImage();
  StringSink sink;
  sink.write_limit = 2;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, DumpImage(image, &sink));
  EXPECT_EQ(sink.write_count, 2u);
}

TEST(VMBytecodeDumpTest, RequiresWriteCallback) {
  const std::vector<uint8_t> image = BuildOwnershipModuleImage();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_vm_bytecode_module_dump(
          IREE_SV("test.module"),
          iree_make_const_byte_span(image.data(), image.size()),
          {nullptr, nullptr}, iree_allocator_system()));
}

}  // namespace
}  // namespace iree::vm::bytecode::testing
