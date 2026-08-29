// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/artifact_emitter.h"

#include <string>

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

TEST(Aie2pArtifactEmitterTest,
     EmitsAuthoredPreparedLowThroughTargetEmitterApi) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  iree_host_size_t vtable_count = 0;
  const loom_op_vtable_t* const* vtables =
      loom_low_dialect_vtables(&vtable_count);
  IREE_ASSERT_OK(loom_context_register_dialect(
      &context, LOOM_DIALECT_LOW, vtables, (uint16_t)vtable_count));
  IREE_ASSERT_OK(loom_aie2p_ops_register_dialect(&context));
  IREE_ASSERT_OK(loom_context_finalize(&context));

  loom_target_low_descriptor_registry_t registry = {};
  loom_aie2p_low_descriptor_registry_initialize(&registry);
  const std::string source =
      "aie2p.target<core> @target\n"
      "low.func.def target<amd.xdna.aie2p.core>(@target) @kernel() "
      "-> (reg<aie2p.er>) asm {\n"
      "  %value = mov.short 1\n"
      "  return %value\n"
      "}\n";
  loom_text_parse_options_t parse_options = {
      /*.diagnostic_sink=*/{loom_diagnostic_stderr_sink, nullptr},
      /*.max_errors=*/20,
  };
  loom_low_descriptor_text_asm_environment_initialize(
      &registry.registry, &parse_options.low_asm_environment);
  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(
      loom_text_parse(iree_make_string_view(source.data(), source.size()),
                      IREE_SV("artifact.loom"), &context, &block_pool,
                      &parse_options, &module));
  ASSERT_NE(module, nullptr);

  loom_low_verify_options_t verify_options = {};
  verify_options.descriptor_registry = &registry.registry;
  verify_options.provider_list = loom_low_verify_provider_list_empty();
  verify_options.max_errors = 20;
  loom_low_verify_scratch_t verify_scratch =
      loom_low_verify_scratch_for_module(module);
  loom_low_verify_result_t verify_result = {};
  IREE_ASSERT_OK(loom_low_verify_module(module, &verify_options,
                                        &verify_scratch, &verify_result));
  ASSERT_EQ(verify_result.error_count, 0u);

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(&block_pool, &scratch_arena);
  loom_target_compile_report_t report;
  loom_target_compile_report_initialize(&report, iree_allocator_system());
  report.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_ARTIFACT_SIZE |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION |
      LOOM_TARGET_COMPILE_REPORT_DETAIL_EMISSION;
  const loom_target_emit_request_t request = {
      /*.target_environment=*/nullptr,
      /*.low_descriptor_registry=*/&registry.registry,
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
  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
