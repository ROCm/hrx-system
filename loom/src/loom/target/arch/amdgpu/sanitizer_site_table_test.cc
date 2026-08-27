// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "iree/base/alignment.h"
#include "iree/base/internal/arena.h"
#include "iree/base/string_view.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/pass/builtin_registry.h"
#include "loom/pass/tooling.h"
#include "loom/sanitizer/site_table.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/lower/lower.h"
#include "loom/target/arch/amdgpu/ops/registry.h"
#include "loom/testing/module_ptr.h"

namespace {

using ModulePtr = ::loom::testing::ModulePtr;

uint32_t LoadLeU32(const uint8_t* data, iree_host_size_t offset) {
  return iree_unaligned_load_le_u32(data + offset);
}

uint16_t LoadLeU16(const uint8_t* data, iree_host_size_t offset) {
  return iree_unaligned_load_le_u16(data + offset);
}

class AmdgpuSanitizerSiteTableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_amdgpu_ops_register_dialect(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_amdgpu_low_descriptor_registry_initialize(&low_registry_);
    loom_amdgpu_low_lower_policy_registry_initialize(&policy_registry_);
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr Parse(iree_string_view_t source) {
    loom_text_parse_options_t parse_options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(source, IREE_SV("sanitizer_site_table.loom"),
                                  &context_, &block_pool_, &parse_options,
                                  &module));
    return ModulePtr(module);
  }

  iree_status_t RunSourceToLow(loom_module_t* module) {
    const loom_target_low_legality_provider_t* legality_providers[] = {
        loom_amdgpu_low_legality_provider(),
    };
    const loom_target_low_legality_provider_list_t legality_provider_list =
        loom_target_low_legality_provider_list_make(
            legality_providers, IREE_ARRAYSIZE(legality_providers));

    loom_low_pass_environment_storage_t environment_storage;
    loom_pass_environment_t environment =
        loom_low_pass_environment_storage_initialize(
            &low_registry_.registry, &policy_registry_, &legality_provider_list,
            /*legalizer_provider_list=*/nullptr,
            /*math_policy_registry=*/nullptr, /*compile_report=*/nullptr,
            /*target_environment=*/nullptr,
            /*function_versions=*/nullptr, &environment_storage);
    loom_pass_tool_run_options_t run_options = {
        /*.registry=*/loom_pass_builtin_registry(),
        /*.environment=*/environment,
        /*.function_versions=*/nullptr,
        /*.predicate_provider=*/{},
        /*.block_pool=*/&block_pool_,
    };
    loom_pass_run_result_t run_result = {};
    iree_status_t status = loom_pass_tool_run_flat_pipeline(
        module, IREE_SV("source-to-low"), &run_options, &run_result);
    if (iree_status_is_ok(status)) {
      if (run_result.error_count > 0) {
        status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "source-to-low pipeline emitted %u errors",
                                  run_result.error_count);
      }
    }
    return status;
  }

  const loom_op_t* FindSiteTableOp(const loom_module_t* module) const {
    const loom_string_id_t name_id = loom_module_lookup_string(
        module, IREE_SV(LOOM_SANITIZER_SITE_TABLE_SYMBOL_NAME));
    if (name_id == LOOM_STRING_ID_INVALID) {
      return nullptr;
    }
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
      return nullptr;
    }
    return module->symbols.entries[symbol_id].defining_op;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t low_registry_ = {};
  loom_low_lower_policy_registry_t policy_registry_ = {};
};

TEST_F(AmdgpuSanitizerSiteTableTest, SourceToLowAggregatesMultiKernelSites) {
  static constexpr char kSource[] = R"(
amdgpu.target<gfx11-generic> @target

kernel.def target(@target) @read_kernel() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%input: buffer) {
  %base = index.constant 0 : offset
  %input_global = buffer.assume.memory_space<global> %input : buffer
  %input_view = buffer.view %input_global[%base] : buffer -> view<1xi32>
  sanitizer.assert.access<read> %input_view[0] : view<1xi32>
  kernel.return
}

kernel.def target(@target) @write_kernel() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%output: buffer) {
  %base = index.constant 0 : offset
  %output_global = buffer.assume.memory_space<global> %output : buffer
  %output_view = buffer.view %output_global[%base] : buffer -> view<1xi32>
  sanitizer.assert.access<write> %output_view[0] : view<1xi32>
  kernel.return
}
)";
  ModulePtr module = Parse(iree_make_cstring_view(kSource));

  IREE_ASSERT_OK(RunSourceToLow(module.get()));

  const loom_op_t* site_table_op = FindSiteTableOp(module.get());
  ASSERT_NE(site_table_op, nullptr);
  ASSERT_TRUE(loom_global_rodata_def_isa(site_table_op));
  const iree_const_byte_span_t contents =
      loom_global_rodata_def_contents(site_table_op);
  ASSERT_GE(contents.data_length,
            LOOM_SANITIZER_SITE_TABLE_HEADER_LENGTH +
                2 * LOOM_SANITIZER_SITE_TABLE_RECORD_LENGTH);
  EXPECT_EQ(
      LoadLeU32(contents.data, LOOM_SANITIZER_SITE_TABLE_HEADER_MAGIC_OFFSET),
      LOOM_SANITIZER_SITE_TABLE_MAGIC);
  EXPECT_EQ(contents.data[LOOM_SANITIZER_SITE_TABLE_HEADER_VERSION_OFFSET],
            LOOM_SANITIZER_SITE_TABLE_VERSION);
  EXPECT_EQ(
      contents.data[LOOM_SANITIZER_SITE_TABLE_HEADER_HEADER_LENGTH_OFFSET],
      LOOM_SANITIZER_SITE_TABLE_HEADER_LENGTH);
  EXPECT_EQ(LoadLeU16(contents.data,
                      LOOM_SANITIZER_SITE_TABLE_HEADER_RECORD_LENGTH_OFFSET),
            LOOM_SANITIZER_SITE_TABLE_RECORD_LENGTH);
  EXPECT_EQ(LoadLeU32(contents.data,
                      LOOM_SANITIZER_SITE_TABLE_HEADER_ROW_COUNT_OFFSET),
            2u);
  const uint32_t string_table_offset =
      LoadLeU32(contents.data,
                LOOM_SANITIZER_SITE_TABLE_HEADER_STRING_TABLE_OFFSET_OFFSET);
  const uint32_t string_table_length =
      LoadLeU32(contents.data,
                LOOM_SANITIZER_SITE_TABLE_HEADER_STRING_TABLE_LENGTH_OFFSET);
  ASSERT_GT(string_table_length, 0u);
  ASSERT_LE((uint64_t)string_table_offset + string_table_length,
            contents.data_length);

  const uint8_t* record0 =
      contents.data + LOOM_SANITIZER_SITE_TABLE_HEADER_LENGTH;
  const uint8_t* record1 = record0 + LOOM_SANITIZER_SITE_TABLE_RECORD_LENGTH;
  EXPECT_EQ(LoadLeU32(record0, LOOM_SANITIZER_SITE_TABLE_RECORD_SITE_ID_OFFSET),
            0u);
  EXPECT_EQ(LoadLeU32(record0, LOOM_SANITIZER_SITE_TABLE_RECORD_OP_KIND_OFFSET),
            LOOM_OP_SANITIZER_ASSERT_ACCESS);
  EXPECT_EQ(LoadLeU32(record0, LOOM_SANITIZER_SITE_TABLE_RECORD_FLAGS_OFFSET),
            LOOM_SANITIZER_SITE_TABLE_RECORD_HAS_SOURCE_LOCATION);
  EXPECT_EQ(LoadLeU32(record1, LOOM_SANITIZER_SITE_TABLE_RECORD_SITE_ID_OFFSET),
            1u);
  EXPECT_EQ(LoadLeU32(record1, LOOM_SANITIZER_SITE_TABLE_RECORD_OP_KIND_OFFSET),
            LOOM_OP_SANITIZER_ASSERT_ACCESS);
  EXPECT_EQ(LoadLeU32(record1, LOOM_SANITIZER_SITE_TABLE_RECORD_FLAGS_OFFSET),
            LOOM_SANITIZER_SITE_TABLE_RECORD_HAS_SOURCE_LOCATION);

  const auto expect_parsed_source = [&](const uint8_t* record) {
    EXPECT_EQ(
        LoadLeU16(record, LOOM_SANITIZER_SITE_TABLE_RECORD_SOURCE_KIND_OFFSET),
        LOOM_SANITIZER_SITE_TABLE_SOURCE_KIND_FILE);
    const uint32_t source_name_offset = LoadLeU32(
        record, LOOM_SANITIZER_SITE_TABLE_RECORD_SOURCE_NAME_OFFSET_OFFSET);
    const uint32_t source_name_length = LoadLeU32(
        record, LOOM_SANITIZER_SITE_TABLE_RECORD_SOURCE_NAME_LENGTH_OFFSET);
    ASSERT_LE((uint64_t)source_name_offset + source_name_length,
              string_table_length);
    const iree_string_view_t source_name = iree_make_string_view(
        (const char*)contents.data + string_table_offset + source_name_offset,
        source_name_length);
    EXPECT_TRUE(iree_string_view_equal(source_name,
                                       IREE_SV("sanitizer_site_table.loom")));
  };
  expect_parsed_source(record0);
  expect_parsed_source(record1);

  const uint32_t record0_start_line =
      LoadLeU32(record0, LOOM_SANITIZER_SITE_TABLE_RECORD_START_LINE_OFFSET);
  const uint32_t record1_start_line =
      LoadLeU32(record1, LOOM_SANITIZER_SITE_TABLE_RECORD_START_LINE_OFFSET);
  EXPECT_NE(record0_start_line, 0u);
  EXPECT_LT(record0_start_line, record1_start_line);
}

}  // namespace
