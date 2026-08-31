// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/artifact_emitter.h"

#include <string>

#include "iree/base/byte_sequence.h"
#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/low_registry.h"
#include "loom/target/arch/amd/xdna/aie2p/ops/registry.h"
#include "loom/target/emit/native/elf.h"
#include "loom/target/reporting/report.h"

namespace loom {
namespace {

uint16_t LoadLeU16(iree_const_byte_span_t bytes, size_t offset) {
  return (uint16_t)bytes.data[offset] | ((uint16_t)bytes.data[offset + 1] << 8);
}

uint32_t LoadLeU32(iree_const_byte_span_t bytes, size_t offset) {
  return (uint32_t)bytes.data[offset] |
         ((uint32_t)bytes.data[offset + 1] << 8) |
         ((uint32_t)bytes.data[offset + 2] << 16) |
         ((uint32_t)bytes.data[offset + 3] << 24);
}

iree_status_t DiscardDiagnostic(void* user_data,
                                const loom_diagnostic_emission_t* emission) {
  (void)user_data;
  (void)emission;
  return iree_ok_status();
}

class Aie2pArtifactEmitterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_low_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_LOW, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_aie2p_ops_register_dialect(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_aie2p_low_descriptor_registry_initialize(&registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* Parse(const std::string& source) {
    loom_text_parse_options_t parse_options = {
        /*.diagnostic_sink=*/{loom_diagnostic_stderr_sink, nullptr},
        /*.max_errors=*/20,
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &registry_.registry, &parse_options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("artifact.loom"), &context_, &block_pool_,
                        &parse_options, &module));
    return module;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t registry_ = {};
};

TEST_F(Aie2pArtifactEmitterTest,
       EmitsAuthoredPreparedLowThroughTargetEmitterApi) {
  const std::string source =
      "aie2p.target<core> @target\n"
      "low.func.def target<amd.xdna.aie2p.core>(@target) @kernel() "
      "-> (reg<aie2p.er>) asm {\n"
      "  %value = mov.short 1\n"
      "  return %value\n"
      "}\n";
  loom_module_t* module = Parse(source);
  ASSERT_NE(module, nullptr);

  loom_low_verify_options_t verify_options = {};
  verify_options.descriptor_registry = &registry_.registry;
  verify_options.provider_list = loom_low_verify_provider_list_empty();
  verify_options.max_errors = 20;
  loom_low_verify_scratch_t verify_scratch =
      loom_low_verify_scratch_for_module(module);
  loom_low_verify_result_t verify_result = {};
  IREE_ASSERT_OK(loom_low_verify_module(module, &verify_options,
                                        &verify_scratch, &verify_result));
  ASSERT_EQ(verify_result.error_count, 0u);

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool_, &scratch_arena);
  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_ARTIFACT_SIZE |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION;
  const loom_target_emit_request_t request = {
      /*.target_environment=*/nullptr,
      /*.low_descriptor_registry=*/&registry_.registry,
      /*.module=*/module,
      /*.function_versions=*/nullptr,
      /*.option_chain=*/nullptr,
      /*.identifier=*/IREE_SV("kernel.elf"),
      /*.artifact_manifest=*/{},
      /*.compile_report=*/&report,
      /*.diagnostic_emitter=*/{DiscardDiagnostic, nullptr},
      /*.scratch_arena=*/&scratch_arena,
      /*.allocator=*/iree_allocator_system(),
  };
  loom_target_emit_artifact_t artifact = {};
  IREE_ASSERT_OK(loom_aie2p_tile_elf_emitter.emit(&request, &artifact));
  ASSERT_EQ(artifact.target_artifact_format, LOOM_TARGET_ARTIFACT_FORMAT_ELF);
  ASSERT_NE(artifact.contents, nullptr);

  iree_byte_span_t storage = iree_byte_span_empty();
  IREE_ASSERT_OK(iree_byte_sequence_clone(artifact.contents,
                                          iree_allocator_system(), &storage));
  const iree_const_byte_span_t bytes =
      iree_make_const_byte_span(storage.data, storage.data_length);
  ASSERT_GT(bytes.data_length, 52u);
  EXPECT_EQ(std::string((const char*)bytes.data, 4), std::string("\x7f"
                                                                 "ELF",
                                                                 4));
  EXPECT_EQ(bytes.data[4], 1u);
  EXPECT_EQ(LoadLeU16(bytes, 16), LOOM_NATIVE_ELF_FILE_TYPE_EXEC);
  EXPECT_EQ(LoadLeU16(bytes, 18), LOOM_NATIVE_ELF_MACHINE_AIE);
  EXPECT_EQ(LoadLeU32(bytes, 24), 0u);
  EXPECT_EQ(LoadLeU32(bytes, 36), 3u);
  EXPECT_EQ(LoadLeU16(bytes, 44), 1u);
  EXPECT_EQ(report.artifact_kind,
            LOOM_TARGET_COMPILE_ARTIFACT_KIND_TARGET_ARTIFACT);
  EXPECT_EQ(report.artifact_size, bytes.data_length);
  EXPECT_GT(report.emitted_instruction_count, 0u);
  EXPECT_GT(report.emitted_code_byte_count, 0u);

  iree_allocator_free(iree_allocator_system(), storage.data);
  loom_target_emit_artifact_release(&artifact);
  loom_target_compile_report_deinitialize(&report);
  iree_arena_deinitialize(&scratch_arena);
  loom_module_free(module);
}

TEST_F(Aie2pArtifactEmitterTest,
       EmitsPlacedFunctionStorageAndRelocatedAddress) {
  const std::string source =
      "aie2p.target<core> @target\n"
      "low.func.def target<amd.xdna.aie2p.core>(@target) "
      "@local_address() asm {\n"
      "  %padding = storage {byte_alignment = 16, byte_length = 32} : "
      "low.storage<workgroup>\n"
      "  %storage = storage {byte_alignment = 64, byte_length = 256} : "
      "low.storage<workgroup>\n"
      "  %view = storage_view %storage {offset = 64, byte_length = 128} : "
      "low.storage<workgroup> -> low.storage<workgroup>\n"
      "  %address = storage_address %view {offset = 16} : "
      "low.storage<workgroup> -> reg<aie2p.ep>\n"
      "  %value = vlda.512.i32x16 %address, 0\n"
      "  return\n"
      "}\n";
  loom_module_t* module = Parse(source);
  ASSERT_NE(module, nullptr);

  loom_low_verify_options_t verify_options = {};
  verify_options.descriptor_registry = &registry_.registry;
  verify_options.provider_list = loom_low_verify_provider_list_empty();
  verify_options.max_errors = 20;
  loom_low_verify_scratch_t verify_scratch =
      loom_low_verify_scratch_for_module(module);
  loom_low_verify_result_t verify_result = {};
  IREE_ASSERT_OK(loom_low_verify_module(module, &verify_options,
                                        &verify_scratch, &verify_result));
  ASSERT_EQ(verify_result.error_count, 0u);

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool_, &scratch_arena);
  const loom_target_emit_request_t request = {
      /*.target_environment=*/nullptr,
      /*.low_descriptor_registry=*/&registry_.registry,
      /*.module=*/module,
      /*.function_versions=*/nullptr,
      /*.option_chain=*/nullptr,
      /*.identifier=*/IREE_SV("local_address.elf"),
      /*.artifact_manifest=*/{},
      /*.compile_report=*/nullptr,
      /*.diagnostic_emitter=*/{DiscardDiagnostic, nullptr},
      /*.scratch_arena=*/&scratch_arena,
      /*.allocator=*/iree_allocator_system(),
  };
  loom_target_emit_artifact_t artifact = {};
  IREE_ASSERT_OK(loom_aie2p_tile_elf_emitter.emit(&request, &artifact));

  iree_byte_span_t storage = iree_byte_span_empty();
  IREE_ASSERT_OK(iree_byte_sequence_clone(artifact.contents,
                                          iree_allocator_system(), &storage));
  const iree_const_byte_span_t bytes =
      iree_make_const_byte_span(storage.data, storage.data_length);
  ASSERT_GT(bytes.data_length, 84u);
  EXPECT_EQ(LoadLeU16(bytes, 44), 2u);
  constexpr size_t kProgramHeaderOffset = 52;
  const uint32_t code_offset = LoadLeU32(bytes, kProgramHeaderOffset + 4);
  constexpr size_t kStorageProgramHeaderOffset = kProgramHeaderOffset + 32;
  EXPECT_EQ(LoadLeU32(bytes, kStorageProgramHeaderOffset + 8), 0x70000u);
  EXPECT_EQ(LoadLeU32(bytes, kStorageProgramHeaderOffset + 16), 0u);
  EXPECT_EQ(LoadLeU32(bytes, kStorageProgramHeaderOffset + 20), 320u);
  const uint8_t expected_movxm[] = {0x44, 0x20, 0xc1, 0x00, 0x07, 0x00};
  ASSERT_LE((iree_host_size_t)code_offset + IREE_ARRAYSIZE(expected_movxm),
            bytes.data_length);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(expected_movxm); ++i) {
    EXPECT_EQ(bytes.data[code_offset + i], expected_movxm[i]) << i;
  }

  iree_allocator_free(iree_allocator_system(), storage.data);
  loom_target_emit_artifact_release(&artifact);
  iree_arena_deinitialize(&scratch_arena);
  loom_module_free(module);
}

TEST_F(Aie2pArtifactEmitterTest, RejectsArrayProgramRepresentation) {
  const std::string source =
      "aie2p.target<array> @target\n"
      "low.func.def target<amd.xdna.aie2p.array>(@target) "
      "@array_program() asm {\n"
      "  return\n"
      "}\n";
  loom_module_t* module = Parse(source);
  ASSERT_NE(module, nullptr);

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool_, &scratch_arena);
  const loom_target_emit_request_t request = {
      /*.target_environment=*/nullptr,
      /*.low_descriptor_registry=*/&registry_.registry,
      /*.module=*/module,
      /*.function_versions=*/nullptr,
      /*.option_chain=*/nullptr,
      /*.identifier=*/IREE_SV("array.elf"),
      /*.artifact_manifest=*/{},
      /*.compile_report=*/nullptr,
      /*.diagnostic_emitter=*/{DiscardDiagnostic, nullptr},
      /*.scratch_arena=*/&scratch_arena,
      /*.allocator=*/iree_allocator_system(),
  };
  loom_target_emit_artifact_t artifact = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        loom_aie2p_tile_elf_emitter.emit(&request, &artifact));
  EXPECT_EQ(artifact.contents, nullptr);

  iree_arena_deinitialize(&scratch_arena);
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
