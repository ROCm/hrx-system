// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <string>

#include "iree/io/file_contents.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/target/emit/llvmir/bitcode_writer.h"
#include "loom/target/emit/llvmir/test_modules.h"
#include "loom/target/emit/llvmir/text_writer.h"
#include "loom/target/emit/llvmir/verify.h"
#include "loom/target/tool/llvm.h"
#include "loom/util/stream.h"

namespace loom {
namespace {

using ModulePtr =
    std::unique_ptr<loom_llvmir_module_t, void (*)(loom_llvmir_module_t*)>;
using StreamPtr =
    std::unique_ptr<iree_io_stream_t, void (*)(iree_io_stream_t*)>;

iree_string_view_t StringView(const std::string& value) {
  return iree_make_string_view(value.data(), value.size());
}

std::string ToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

std::string ToString(const loom_tool_output_t& output) {
  return output.data ? std::string(output.data, output.length) : std::string();
}

bool IsToolUnavailable(iree_status_code_t status_code) {
  return status_code == IREE_STATUS_NOT_FOUND ||
         status_code == IREE_STATUS_UNAVAILABLE ||
         status_code == IREE_STATUS_UNIMPLEMENTED;
}

StreamPtr CreateStream() {
  iree_io_stream_t* stream = NULL;
  IREE_CHECK_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE,
      1024, iree_allocator_system(), &stream));
  return StreamPtr(stream, iree_io_stream_release);
}

std::string StreamBytes(iree_io_stream_t* stream) {
  iree_io_stream_pos_t length = iree_io_stream_length(stream);
  IREE_ASSERT_GE(length, 0);
  std::string bytes((size_t)length, '\0');
  IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_CHECK_OK(iree_io_stream_read(stream, bytes.size(), bytes.data(), NULL));
  return bytes;
}

iree_status_t BuildBitcodeFixture(loom_llvmir_test_module_scenario_t scenario,
                                  std::string* out_bytes) {
  loom_llvmir_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_test_module_build(
      scenario, iree_allocator_system(), &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);
  IREE_RETURN_IF_ERROR(loom_llvmir_verify_module(module_ptr.get()));

  StreamPtr stream = CreateStream();
  IREE_RETURN_IF_ERROR(
      loom_llvmir_bitcode_write_module(module_ptr.get(), stream.get()));
  *out_bytes = StreamBytes(stream.get());
  return iree_ok_status();
}

iree_status_t BuildTextFixture(loom_llvmir_test_module_scenario_t scenario,
                               std::string* out_text) {
  loom_llvmir_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(loom_llvmir_test_module_build(
      scenario, iree_allocator_system(), &module));
  ModulePtr module_ptr(module, loom_llvmir_module_free);
  IREE_RETURN_IF_ERROR(loom_llvmir_verify_module(module_ptr.get()));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  iree_status_t status =
      loom_llvmir_text_write_module(module_ptr.get(), &stream);
  if (iree_status_is_ok(status)) {
    *out_text = ToString(iree_string_builder_view(&builder));
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

iree_status_t WriteTempFile(const std::string& path,
                            const std::string& contents) {
  return iree_io_file_contents_write(
      StringView(path),
      iree_make_const_byte_span(contents.data(), contents.size()),
      iree_allocator_system());
}

iree_status_t ReadTempFile(const std::string& path,
                           iree_io_file_contents_t** out_contents) {
  return iree_io_file_contents_read(StringView(path), iree_allocator_system(),
                                    out_contents);
}

loom_llvm_toolchain_t ToolchainFromEnvironment() {
  loom_llvm_toolchain_t toolchain;
  loom_llvm_toolchain_initialize_from_environment(&toolchain);
  return toolchain;
}

TEST(LlvmIrToolTest, QueriesVersion) {
  loom_llvm_toolchain_t toolchain = ToolchainFromEnvironment();
  loom_tool_output_t version_text = {};
  iree_status_t status =
      loom_llvm_tool_query_version(&toolchain, LOOM_LLVM_TOOL_LLVM_DIS,
                                   iree_allocator_system(), &version_text);
  if (IsToolUnavailable(iree_status_code(status))) {
    IREE_EXPECT_NOT_OK(status);
    GTEST_SKIP() << "llvm-dis is unavailable in this test environment";
  }
  IREE_ASSERT_OK(status);

  std::string version = ToString(version_text);
  EXPECT_NE(version.find("LLVM"), std::string::npos);
  EXPECT_EQ(version.find('\r'), std::string::npos);
  loom_tool_output_deinitialize(&version_text, iree_allocator_system());
}

TEST(LlvmIrToolTest, AssemblesTextAndVerifiesBitcode) {
  std::string text;
  IREE_ASSERT_OK(BuildTextFixture(LOOM_LLVMIR_TEST_MODULE_OBJECT_VADD4, &text));
  iree::testing::TempFilePath input_file("loom_llvm_tool_test", ".ll");
  iree::testing::TempFilePath bitcode_file("loom_llvm_tool_test", ".bc");
  IREE_ASSERT_OK(WriteTempFile(input_file.path(), text));

  loom_llvm_toolchain_t toolchain = ToolchainFromEnvironment();
  iree_status_t status = loom_llvm_tool_assemble_ir_text_file(
      &toolchain, StringView(input_file.path()),
      StringView(bitcode_file.path()), iree_allocator_system());
  if (IsToolUnavailable(iree_status_code(status))) {
    IREE_EXPECT_NOT_OK(status);
    GTEST_SKIP() << "llvm-as is unavailable in this test environment";
  }
  IREE_ASSERT_OK(status);

  status = loom_llvm_tool_verify_bitcode_file(
      &toolchain, StringView(bitcode_file.path()), iree_allocator_system());
  if (IsToolUnavailable(iree_status_code(status))) {
    IREE_EXPECT_NOT_OK(status);
    GTEST_SKIP() << "opt is unavailable in this test environment";
  }
  IREE_ASSERT_OK(status);
}

TEST(LlvmIrToolTest, DisassemblesBitcode) {
  std::string bitcode;
  IREE_ASSERT_OK(
      BuildBitcodeFixture(LOOM_LLVMIR_TEST_MODULE_OBJECT_VADD4, &bitcode));
  iree::testing::TempFilePath bitcode_file("loom_llvm_tool_test", ".bc");
  IREE_ASSERT_OK(WriteTempFile(bitcode_file.path(), bitcode));

  loom_llvm_toolchain_t toolchain = ToolchainFromEnvironment();
  loom_tool_output_t text = {};
  iree_status_t status = loom_llvm_tool_disassemble_bitcode_file(
      &toolchain, StringView(bitcode_file.path()), iree_allocator_system(),
      &text);
  if (IsToolUnavailable(iree_status_code(status))) {
    IREE_EXPECT_NOT_OK(status);
    GTEST_SKIP() << "llvm-dis is unavailable in this test environment";
  }
  IREE_ASSERT_OK(status);

  std::string disassembly = ToString(text);
  EXPECT_NE(disassembly.find("define dso_local void @vadd4_object"),
            std::string::npos);
  EXPECT_EQ(disassembly.find('\r'), std::string::npos);
  loom_tool_output_deinitialize(&text, iree_allocator_system());
}

TEST(LlvmIrToolTest, DisassemblesBitcodeBytes) {
  std::string bitcode;
  IREE_ASSERT_OK(
      BuildBitcodeFixture(LOOM_LLVMIR_TEST_MODULE_OBJECT_VADD4, &bitcode));

  loom_llvm_toolchain_t toolchain = ToolchainFromEnvironment();
  loom_tool_output_t text = {};
  iree_status_t status = loom_llvm_tool_disassemble_bitcode(
      &toolchain, iree_make_const_byte_span(bitcode.data(), bitcode.size()),
      iree_allocator_system(), &text);
  if (IsToolUnavailable(iree_status_code(status))) {
    IREE_EXPECT_NOT_OK(status);
    GTEST_SKIP() << "llvm-dis is unavailable in this test environment";
  }
  IREE_ASSERT_OK(status);

  std::string disassembly = ToString(text);
  EXPECT_NE(disassembly.find("define dso_local void @vadd4_object"),
            std::string::npos);
  EXPECT_EQ(disassembly.find('\r'), std::string::npos);
  loom_tool_output_deinitialize(&text, iree_allocator_system());
}

TEST(LlvmIrToolTest, DisassemblesAndVerifiesCastsBitcode) {
  std::string bitcode;
  IREE_ASSERT_OK(BuildBitcodeFixture(LOOM_LLVMIR_TEST_MODULE_CASTS, &bitcode));
  iree::testing::TempFilePath bitcode_file("loom_llvm_tool_test", ".bc");
  IREE_ASSERT_OK(WriteTempFile(bitcode_file.path(), bitcode));

  loom_llvm_toolchain_t toolchain = ToolchainFromEnvironment();
  loom_tool_output_t text = {};
  iree_status_t status = loom_llvm_tool_disassemble_bitcode_file(
      &toolchain, StringView(bitcode_file.path()), iree_allocator_system(),
      &text);
  if (IsToolUnavailable(iree_status_code(status))) {
    IREE_EXPECT_NOT_OK(status);
    GTEST_SKIP() << "llvm-dis is unavailable in this test environment";
  }
  IREE_ASSERT_OK(status);

  std::string disassembly = ToString(text);
  EXPECT_NE(disassembly.find("ptrtoaddr ptr %pointer to i64"),
            std::string::npos)
      << disassembly;
  loom_tool_output_deinitialize(&text, iree_allocator_system());

  status = loom_llvm_tool_verify_bitcode_file(
      &toolchain, StringView(bitcode_file.path()), iree_allocator_system());
  if (IsToolUnavailable(iree_status_code(status))) {
    IREE_EXPECT_NOT_OK(status);
    GTEST_SKIP() << "opt is unavailable in this test environment";
  }
  IREE_ASSERT_OK(status);
}

TEST(LlvmIrToolTest, CompilesX86Object) {
  std::string bitcode;
  IREE_ASSERT_OK(
      BuildBitcodeFixture(LOOM_LLVMIR_TEST_MODULE_OBJECT_VADD4, &bitcode));
  iree::testing::TempFilePath bitcode_file("loom_llvm_tool_test", ".bc");
  iree::testing::TempFilePath object_file("loom_llvm_tool_test", ".o");
  IREE_ASSERT_OK(WriteTempFile(bitcode_file.path(), bitcode));

  loom_llvm_toolchain_t toolchain = ToolchainFromEnvironment();
  iree_status_t status = loom_llvm_tool_compile_object_file(
      &toolchain, StringView(bitcode_file.path()),
      StringView(object_file.path()), NULL, 0, iree_allocator_system());
  if (IsToolUnavailable(iree_status_code(status))) {
    IREE_EXPECT_NOT_OK(status);
    GTEST_SKIP() << "llc is unavailable in this test environment";
  }
  IREE_ASSERT_OK(status);

  iree_io_file_contents_t* contents = NULL;
  IREE_ASSERT_OK(ReadTempFile(object_file.path(), &contents));
  ASSERT_GT(contents->const_buffer.data_length, 0u);
  iree_io_file_contents_free(contents);
}

TEST(LlvmIrToolTest, CompilesX86ObjectBytes) {
  std::string bitcode;
  IREE_ASSERT_OK(
      BuildBitcodeFixture(LOOM_LLVMIR_TEST_MODULE_OBJECT_VADD4, &bitcode));

  loom_llvm_toolchain_t toolchain = ToolchainFromEnvironment();
  loom_tool_output_t object = {};
  iree_status_t status = loom_llvm_tool_compile_object(
      &toolchain, iree_make_const_byte_span(bitcode.data(), bitcode.size()),
      NULL, 0, iree_allocator_system(), &object);
  if (IsToolUnavailable(iree_status_code(status))) {
    IREE_EXPECT_NOT_OK(status);
    GTEST_SKIP() << "llc is unavailable in this test environment";
  }
  IREE_ASSERT_OK(status);

  EXPECT_GT(object.length, 0u);
  loom_tool_output_deinitialize(&object, iree_allocator_system());
}

TEST(LlvmIrToolTest, DisassemblesX86ObjectBytes) {
  std::string bitcode;
  IREE_ASSERT_OK(
      BuildBitcodeFixture(LOOM_LLVMIR_TEST_MODULE_OBJECT_VADD4, &bitcode));

  loom_llvm_toolchain_t toolchain = ToolchainFromEnvironment();
  loom_tool_output_t object = {};
  iree_status_t status = loom_llvm_tool_compile_object(
      &toolchain, iree_make_const_byte_span(bitcode.data(), bitcode.size()),
      NULL, 0, iree_allocator_system(), &object);
  if (IsToolUnavailable(iree_status_code(status))) {
    IREE_EXPECT_NOT_OK(status);
    GTEST_SKIP() << "llc is unavailable in this test environment";
  }
  IREE_ASSERT_OK(status);

  const iree_string_view_t arguments[] = {IREE_SV("--disassemble")};
  loom_tool_output_t disassembly = {};
  status = loom_llvm_tool_disassemble_object(
      &toolchain, iree_make_const_byte_span(object.data, object.length),
      arguments, IREE_ARRAYSIZE(arguments), iree_allocator_system(),
      &disassembly);
  if (IsToolUnavailable(iree_status_code(status))) {
    loom_tool_output_deinitialize(&object, iree_allocator_system());
    IREE_EXPECT_NOT_OK(status);
    GTEST_SKIP() << "llvm-objdump is unavailable in this test environment";
  }
  if (!iree_status_is_ok(status)) {
    loom_tool_output_deinitialize(&object, iree_allocator_system());
  }
  IREE_ASSERT_OK(status);

  std::string disassembly_text = ToString(disassembly);
  EXPECT_NE(disassembly_text.find("vadd4_object"), std::string::npos)
      << disassembly_text;
  EXPECT_EQ(disassembly_text.find('\r'), std::string::npos);
  loom_tool_output_deinitialize(&disassembly, iree_allocator_system());
  loom_tool_output_deinitialize(&object, iree_allocator_system());
}

TEST(LlvmIrToolTest, CompilesX86Assembly) {
  std::string bitcode;
  IREE_ASSERT_OK(
      BuildBitcodeFixture(LOOM_LLVMIR_TEST_MODULE_OBJECT_VADD4, &bitcode));
  iree::testing::TempFilePath bitcode_file("loom_llvm_tool_test", ".bc");
  iree::testing::TempFilePath assembly_file("loom_llvm_tool_test", ".s");
  IREE_ASSERT_OK(WriteTempFile(bitcode_file.path(), bitcode));

  loom_llvm_toolchain_t toolchain = ToolchainFromEnvironment();
  iree_status_t status = loom_llvm_tool_compile_assembly_file(
      &toolchain, StringView(bitcode_file.path()),
      StringView(assembly_file.path()), NULL, 0, iree_allocator_system());
  if (IsToolUnavailable(iree_status_code(status))) {
    IREE_EXPECT_NOT_OK(status);
    GTEST_SKIP() << "llc is unavailable in this test environment";
  }
  IREE_ASSERT_OK(status);

  iree_io_file_contents_t* contents = NULL;
  IREE_ASSERT_OK(ReadTempFile(assembly_file.path(), &contents));
  std::string assembly((const char*)contents->const_buffer.data,
                       contents->const_buffer.data_length);
  EXPECT_NE(assembly.find("vadd4_object"), std::string::npos) << assembly;
  iree_io_file_contents_free(contents);
}

TEST(LlvmIrToolTest, CompilesX86AssemblyBytes) {
  std::string bitcode;
  IREE_ASSERT_OK(
      BuildBitcodeFixture(LOOM_LLVMIR_TEST_MODULE_OBJECT_VADD4, &bitcode));

  loom_llvm_toolchain_t toolchain = ToolchainFromEnvironment();
  loom_tool_output_t assembly = {};
  iree_status_t status = loom_llvm_tool_compile_assembly(
      &toolchain, iree_make_const_byte_span(bitcode.data(), bitcode.size()),
      NULL, 0, iree_allocator_system(), &assembly);
  if (IsToolUnavailable(iree_status_code(status))) {
    IREE_EXPECT_NOT_OK(status);
    GTEST_SKIP() << "llc is unavailable in this test environment";
  }
  IREE_ASSERT_OK(status);

  std::string assembly_text = ToString(assembly);
  EXPECT_NE(assembly_text.find("vadd4_object"), std::string::npos)
      << assembly_text;
  EXPECT_EQ(assembly_text.find('\r'), std::string::npos);
  loom_tool_output_deinitialize(&assembly, iree_allocator_system());
}

TEST(LlvmIrToolTest, CompilesAmdgpuObjectWhenTargetIsRegistered) {
  loom_llvm_toolchain_t toolchain = ToolchainFromEnvironment();
  loom_tool_output_t version_text = {};
  iree_status_t status = loom_llvm_tool_query_version(
      &toolchain, LOOM_LLVM_TOOL_LLC, iree_allocator_system(), &version_text);
  if (IsToolUnavailable(iree_status_code(status))) {
    IREE_EXPECT_NOT_OK(status);
    GTEST_SKIP() << "llc is unavailable in this test environment";
  }
  IREE_ASSERT_OK(status);
  std::string version = ToString(version_text);
  loom_tool_output_deinitialize(&version_text, iree_allocator_system());
  if (version.find("amdgcn") == std::string::npos &&
      version.find("AMDGPU") == std::string::npos) {
    GTEST_SKIP() << "installed llc does not advertise an AMDGPU target";
  }

  std::string bitcode;
  IREE_ASSERT_OK(
      BuildBitcodeFixture(LOOM_LLVMIR_TEST_MODULE_AMDGPU_INTRINSICS, &bitcode));
  iree::testing::TempFilePath bitcode_file("loom_llvm_tool_test", ".bc");
  iree::testing::TempFilePath object_file("loom_llvm_tool_test", ".o");
  IREE_ASSERT_OK(WriteTempFile(bitcode_file.path(), bitcode));

  iree_string_view_t extra_arguments[] = {
      IREE_SV("-mtriple=amdgcn-amd-amdhsa"),
      IREE_SV("-mcpu=gfx1100"),
  };
  status = loom_llvm_tool_compile_object_file(
      &toolchain, StringView(bitcode_file.path()),
      StringView(object_file.path()), extra_arguments,
      IREE_ARRAYSIZE(extra_arguments), iree_allocator_system());
  IREE_ASSERT_OK(status);

  iree_io_file_contents_t* contents = NULL;
  IREE_ASSERT_OK(ReadTempFile(object_file.path(), &contents));
  ASSERT_GT(contents->const_buffer.data_length, 0u);
  iree_io_file_contents_free(contents);
}

}  // namespace
}  // namespace loom
