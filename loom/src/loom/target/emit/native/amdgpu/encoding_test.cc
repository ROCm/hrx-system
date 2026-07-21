// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/encoding.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/function.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/codegen/low/verify.h"
#include "loom/error/diagnostic.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/ops/ops.h"
#include "loom/target/arch/amdgpu/planning/packet_plan.h"
#include "loom/target/low_descriptor_registry.h"

namespace loom {
namespace {

using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t*);

void RegisterDialect(loom_context_t* context, uint8_t dialect_id,
                     DialectVtablesFn dialect_vtables_fn) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
  IREE_ASSERT_OK(loom_context_register_dialect(context, dialect_id, vtables,
                                               (uint16_t)count));
}

void InitializeLowKernelContext(loom_context_t* context) {
  loom_context_initialize(iree_allocator_system(), context);
  RegisterDialect(context, LOOM_DIALECT_AMDGPU, loom_amdgpu_dialect_vtables);
  RegisterDialect(context, LOOM_DIALECT_LOW, loom_low_dialect_vtables);
  IREE_ASSERT_OK(loom_context_finalize(context));
}

std::string StringViewToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

uint32_t LoadLeU32(const uint8_t* data, size_t offset) {
  return (uint32_t)data[offset] | ((uint32_t)data[offset + 1] << 8) |
         ((uint32_t)data[offset + 2] << 16) |
         ((uint32_t)data[offset + 3] << 24);
}

std::string StatusToStringAndFree(iree_status_t status) {
  iree_allocator_t allocator = iree_allocator_system();
  char* buffer = nullptr;
  iree_host_size_t buffer_length = 0;
  std::string result = iree_status_code_string(iree_status_code(status));
  if (iree_status_to_string(status, &allocator, &buffer, &buffer_length)) {
    result.assign(buffer, buffer_length);
    iree_allocator_free(allocator, buffer);
  }
  iree_status_free(status);
  return result;
}

loom_op_t* FindFirstLowFunction(loom_module_t* module) {
  loom_block_t* block = loom_module_block(module);
  loom_op_t* op = nullptr;
  loom_block_for_each_op(block, op) {
    if (loom_low_function_def_isa(op)) {
      return op;
    }
  }
  return nullptr;
}

class TestArena {
 public:
  TestArena() {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
  }

  ~TestArena() {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_allocator_t* arena() { return &arena_; }

 private:
  // Block pool backing the test arena.
  iree_arena_block_pool_t block_pool_ = {0};
  // Arena receiving transient frame and encoding storage.
  iree_arena_allocator_t arena_ = {0};
};

class AmdgpuEncodingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    InitializeLowKernelContext(&context_);
    loom_amdgpu_low_descriptor_registry_initialize(&target_registry_);
  }

  void TearDown() override {
    ResetModule();
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t ParseSource(const std::string& source) {
    ResetModule();
    loom_text_parse_options_t parse_options = {
        /*.diagnostic_sink=*/{loom_diagnostic_stderr_sink, nullptr},
        /*.max_errors=*/20,
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &target_registry_.registry, &parse_options.low_asm_environment);
    IREE_RETURN_IF_ERROR(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("amdgpu_encoding_test.loom"), &context_,
                        &block_pool_, &parse_options, &module_));
    if (module_ == nullptr) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "AMDGPU encoding test parser returned no module");
    }
    return iree_ok_status();
  }

  iree_status_t BuildFrame(const std::string& target_architecture,
                           const std::string& function_source,
                           iree_arena_allocator_t* arena,
                           loom_low_emission_frame_t* out_frame) {
    std::string source =
        "amdgpu.target<" + target_architecture + "> @gfx_target\n";
    source += function_source;
    IREE_RETURN_IF_ERROR(ParseSource(source));

    loom_op_t* low_function = FindFirstLowFunction(module_);
    if (low_function == nullptr) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "AMDGPU encoding test source has no low func");
    }

    loom_low_verify_options_t verify_options = {
        /*.descriptor_registry=*/&target_registry_.registry,
        /*.target_selection=*/{},
        /*.emitter=*/{},
        /*.provider_list=*/{},
        /*.max_errors=*/20,
    };
    loom_low_verify_result_t verify_result = {};
    loom_low_verify_scratch_t verify_scratch =
        loom_low_verify_scratch_for_module(module_);
    IREE_RETURN_IF_ERROR(loom_low_verify_module(
        module_, &verify_options, &verify_scratch, &verify_result));
    if (verify_result.error_count != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU encoding test source failed low verify");
    }

    loom_low_emission_frame_options_t frame_options = {
        /*.descriptor_registry=*/&target_registry_.registry,
        /*.target_selection=*/{},
        /*.memory_access_table=*/{},
        /*.pressure_cliffs=*/{},
        /*.schedule_pair_affinities=*/{},
        /*.schedule_structural_state_reads=*/{},
        /*.schedule_strategy=*/LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
    };
    return loom_low_emission_frame_build(module_, low_function, &frame_options,
                                         arena, out_frame);
  }

  iree_status_t EncodeFunction(
      const std::string& target_architecture,
      const std::string& function_source, iree_arena_allocator_t* arena,
      loom_amdgpu_encoded_instruction_stream_t* out_stream) {
    loom_low_emission_frame_t frame = {};
    IREE_RETURN_IF_ERROR(
        BuildFrame(target_architecture, function_source, arena, &frame));
    loom_amdgpu_packet_plan_t packet_plan = {};
    IREE_RETURN_IF_ERROR(loom_amdgpu_packet_plan_build(
        &frame.schedule, &frame.allocation, arena, &packet_plan));
    const loom_amdgpu_encode_instruction_stream_options_t options = {
        /*.packet_plan=*/&packet_plan,
    };
    return loom_amdgpu_encode_instruction_stream_result_with_options(
        &frame.schedule, &frame.allocation, &options, out_stream, arena);
  }

 private:
  void ResetModule() {
    if (module_ != nullptr) {
      loom_module_free(module_);
      module_ = nullptr;
    }
  }

  // Block pool backing parser and context allocations.
  iree_arena_block_pool_t block_pool_ = {0};
  // Context containing the low and AMDGPU dialects.
  loom_context_t context_ = {};
  // AMDGPU target-low descriptor registry.
  loom_target_low_descriptor_registry_t target_registry_ = {};
  // Parsed module owned by this test fixture.
  loom_module_t* module_ = nullptr;
};

TEST_F(AmdgpuEncodingTest, EmitsDataSymbolRel32TextFixups) {
  static const char kFunctionSource[] =
      "low.func.def target(@gfx_target) @load_feedback_config() {\n"
      "  %pc = low.op<amdgpu.s_getpc_b64>() : () -> "
      "reg<amdgpu.sgpr x2>\n"
      "  %pc_lo = low.slice %pc[0] : reg<amdgpu.sgpr x2> -> "
      "reg<amdgpu.sgpr>\n"
      "  %pc_hi = low.slice %pc[1] : reg<amdgpu.sgpr x2> -> "
      "reg<amdgpu.sgpr>\n"
      "  %addr_lo = low.op<amdgpu.s_add_u32.rhs_symbol_rel32_lo>(%pc_lo) "
      "{symbol = @iree_feedback_config, byte_offset = 16} : "
      "(reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  %addr_hi = low.op<amdgpu.s_addc_u32.rhs_symbol_rel32_hi>(%pc_hi) "
      "{symbol = @iree_feedback_config, byte_offset = 20} : "
      "(reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  low.return\n"
      "}\n";

  TestArena arena;
  loom_amdgpu_encoded_instruction_stream_t stream = {};
  IREE_ASSERT_OK(
      EncodeFunction("gfx1100", kFunctionSource, arena.arena(), &stream));

  EXPECT_EQ(stream.instruction_count, 4u);
  EXPECT_EQ(stream.native_insertion_count, 0u);
  ASSERT_EQ(stream.text_fixup_count, 2u);
  ASSERT_GE(stream.text.data_length, 4u);

  const loom_amdgpu_hsaco_text_fixup_t& lo = stream.text_fixups[0];
  EXPECT_EQ(lo.kind, LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_LO);
  EXPECT_EQ(lo.base_pc_byte_offset, 4u);
  EXPECT_EQ(lo.target_symbol_byte_offset, 16u);
  EXPECT_EQ(StringViewToString(lo.target_symbol), "iree_feedback_config");
  ASSERT_LE(lo.literal_byte_offset + sizeof(uint32_t), stream.text.data_length);
  EXPECT_EQ(LoadLeU32(stream.text.data, (size_t)lo.literal_byte_offset), 0u);

  const loom_amdgpu_hsaco_text_fixup_t& hi = stream.text_fixups[1];
  EXPECT_EQ(hi.kind, LOOM_AMDGPU_HSACO_TEXT_FIXUP_KIND_DATA_SYMBOL_REL32_HI);
  EXPECT_EQ(hi.base_pc_byte_offset, 4u);
  EXPECT_EQ(hi.target_symbol_byte_offset, 20u);
  EXPECT_EQ(StringViewToString(hi.target_symbol), "iree_feedback_config");
  ASSERT_LE(hi.literal_byte_offset + sizeof(uint32_t), stream.text.data_length);
  EXPECT_EQ(LoadLeU32(stream.text.data, (size_t)hi.literal_byte_offset), 0u);

  EXPECT_LT(lo.literal_byte_offset, hi.literal_byte_offset);
}

TEST_F(AmdgpuEncodingTest, RejectsRel32AddWithoutPcBase) {
  static const char kFunctionSource[] =
      "low.func.def target(@gfx_target) @bad_feedback_config() {\n"
      "  %zero = low.op<amdgpu.s_mov_b32>() {imm32 = 0} : () -> "
      "reg<amdgpu.sgpr>\n"
      "  %addr_lo = low.op<amdgpu.s_add_u32.rhs_symbol_rel32_lo>(%zero) "
      "{symbol = @iree_feedback_config, byte_offset = 16} : "
      "(reg<amdgpu.sgpr>) -> reg<amdgpu.sgpr>\n"
      "  low.return\n"
      "}\n";

  TestArena arena;
  loom_amdgpu_encoded_instruction_stream_t stream = {};
  iree_status_t status =
      EncodeFunction("gfx1100", kFunctionSource, arena.arena(), &stream);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_FAILED_PRECONDITION);
  const std::string message = StatusToStringAndFree(status);
  EXPECT_NE(message.find("does not hold an s_getpc_b64 component"),
            std::string::npos);
}

TEST_F(AmdgpuEncodingTest, EncodesGfx1250ClusterAsyncPacket) {
  static const char kFunctionSource[] =
      "low.func.def target(@gfx_target) @cluster("
      "%lds: reg<amdgpu.vgpr>, %addr: reg<amdgpu.vgpr>, "
      "%saddr: reg<amdgpu.sgpr x2>) asm<amdgpu.rdna4.gfx125x.core> {\n"
      "  %m0 = s_mov_b32_m0_imm 3\n"
      "  cluster_load_async_to_lds_b64 %lds, %addr, %saddr, %m0 "
      "{nv = 1, scope = 2, th = 2}\n"
      "  s_wait_asynccnt {asynccnt = 0}\n"
      "  return\n"
      "}\n";

  TestArena arena;
  loom_amdgpu_encoded_instruction_stream_t stream = {};
  IREE_ASSERT_OK(
      EncodeFunction("gfx1250", kFunctionSource, arena.arena(), &stream));

  static const uint32_t kExpectedWords[] = {
      0xbefd0083u, 0xee1b0080u, 0x00280000u,
      0x00000001u, 0xbfca0000u, 0xbfb00000u,
  };
  EXPECT_EQ(stream.instruction_count, 4u);
  EXPECT_EQ(stream.native_insertion_count, 0u);
  ASSERT_EQ(stream.text.data_length, sizeof(kExpectedWords));
  for (size_t i = 0; i < IREE_ARRAYSIZE(kExpectedWords); ++i) {
    EXPECT_EQ(LoadLeU32(stream.text.data, i * sizeof(uint32_t)),
              kExpectedWords[i])
        << "word " << i;
  }
}

TEST_F(AmdgpuEncodingTest, EncodesGfx1250PackedFp8Conversions) {
  static const char kFunctionSource[] =
      "low.func.def target(@gfx_target) @convert_packed_fp8("
      "%fp8: reg<amdgpu.vgpr>, %bf8: reg<amdgpu.vgpr>) "
      "asm<amdgpu.rdna4.gfx125x.core> {\n"
      "  %fp16 = v_cvt_pk_f16_fp8 %fp8\n"
      "  %bf16 = v_cvt_pk_f16_bf8 %bf8\n"
      "  return\n"
      "}\n";

  TestArena arena;
  loom_amdgpu_encoded_instruction_stream_t stream = {};
  IREE_ASSERT_OK(
      EncodeFunction("gfx1250", kFunctionSource, arena.arena(), &stream));

  static const uint32_t kExpectedWords[] = {
      0x7e00eb00u,
      0x7e00ed01u,
      0xbfb00000u,
  };
  EXPECT_EQ(stream.instruction_count, 3u);
  EXPECT_EQ(stream.native_insertion_count, 0u);
  ASSERT_EQ(stream.text.data_length, sizeof(kExpectedWords));
  for (size_t i = 0; i < IREE_ARRAYSIZE(kExpectedWords); ++i) {
    EXPECT_EQ(LoadLeU32(stream.text.data, i * sizeof(uint32_t)),
              kExpectedWords[i])
        << "word " << i;
  }
}

}  // namespace
}  // namespace loom
