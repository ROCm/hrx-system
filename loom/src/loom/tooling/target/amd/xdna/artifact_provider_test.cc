// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/amd/xdna/artifact_provider.h"

#include "iree/hal/drivers/amd/xdna/image/aie2p/npu2.h"
#include "iree/hal/drivers/amd/xdna/image/image.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ops/op_registry.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/low_registry.h"
#include "loom/target/arch/amd/xdna/aie2p/profile.h"
#include "loom/target/arch/amd/xdna/aie2p/provider.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ::loom::testing::ModulePtr;

constexpr char kResidentNeighborSource[] = R"(
aie2p.target<array> @array_target
aie2p.target<core> @core_target

low.func.def public retain target<amd.xdna.aie2p.array>(@array_target) abi(array_program) @resident_neighbor() asm {
  %channel_capacity = constant.u32 1 : reg<aie2p.array.scalar : index>
  %records_per_activation = constant.u32 1 : reg<aie2p.array.scalar : index>
  %first_lane = constant.u32 0 : reg<aie2p.array.scalar : index>
  %second_lane = constant.u32 1 : reg<aie2p.array.scalar : index>
  %worker_count = constant.u32 2 : reg<aie2p.array.scalar : index>
  %column = constant.u32 0 : reg<aie2p.array.scalar : index>
  %producer_row = constant.u32 2 : reg<aie2p.array.scalar : index>
  %consumer_row = constant.u32 3 : reg<aie2p.array.scalar : index>
  %group = group %worker_count
  %producer = worker %group, %first_lane, @produce_i16
  %consumer = worker %group, %second_lane, @consume_i16
  %sender = sender %producer, 0 : reg<aie2p.array.sender : tile<1xi16>>
  %receiver = receiver %consumer, 0 : reg<aie2p.array.receiver : tile<1xi16>>
  %channel = channel %sender, %receiver, %channel_capacity, %records_per_activation : reg<aie2p.array.channel : tile<1xi16>>
  constrain.location %producer, %column, %producer_row
  constrain.location %consumer, %column, %consumer_row
  return
}

low.func.def target<amd.xdna.aie2p.core>(@core_target) abi(object_function) @produce_i16() asm {
  %output = resource<native_pointer> {index = 0, source_type = buffer} : reg<aie2p.ep>
  return
}

low.func.def target<amd.xdna.aie2p.core>(@core_target) abi(object_function) @consume_i16() asm {
  %input = resource<native_pointer> {index = 0, source_type = buffer} : reg<aie2p.ep>
  return
}
)";

iree_status_t InitializeXdnaContext(loom_context_t* context) {
  loom_context_initialize(iree_allocator_system(), context);
  iree_status_t status = loom_op_registry_register_all_dialects(context);
  if (iree_status_is_ok(status)) {
    status = loom_aie2p_target_provider.register_context(context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_context_finalize(context);
  }
  if (!iree_status_is_ok(status)) {
    loom_context_deinitialize(context);
  }
  return status;
}

class XdnaArtifactProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    IREE_ASSERT_OK(InitializeXdnaContext(&context_));
    loom_aie2p_low_descriptor_registry_initialize(&low_registry_);
  }

  void TearDown() override {
    iree_hal_amd_xdna_image_destroy(image_);
    loom_xdna_artifact_provider.deinitialize_artifact(
        &loom_xdna_artifact_provider, &artifact_, iree_allocator_system());
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t ParseModule(ModulePtr* out_module) {
    loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    loom_low_descriptor_text_asm_environment_initialize(
        &low_registry_.registry, &options.low_asm_environment);
    loom_module_t* module = nullptr;
    IREE_RETURN_IF_ERROR(loom_text_parse(
        IREE_SV(kResidentNeighborSource), IREE_SV("resident_neighbor.loom"),
        &context_, &block_pool_, &options, &module));
    *out_module = ModulePtr(module);
    return iree_ok_status();
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  loom_target_low_descriptor_registry_t low_registry_ = {};
  loom_artifact_t artifact_ = {};
  iree_hal_amd_xdna_image_t* image_ = nullptr;
};

TEST_F(XdnaArtifactProviderTest, EmitsLoaderReadyControlFreeResidentProduct) {
  ModulePtr module;
  IREE_ASSERT_OK(ParseModule(&module));

  const loom_aie2p_target_profile_t* profile = nullptr;
  IREE_ASSERT_OK(loom_aie2p_target_profile_select(
      IREE_SV("amd.xdna.strix_halo.17f0_11"), &profile));
  const loom_artifact_target_t target = {
      /*.target_profile=*/&profile->base,
      /*.target_key=*/IREE_SV("amd.xdna.strix_halo.17f0_11"),
  };
  loom_compile_options_t options;
  loom_compile_options_initialize(&options);
  bool emitted = false;
  IREE_ASSERT_OK(loom_xdna_artifact_provider.emit_artifact(
      &loom_xdna_artifact_provider, module.get(), &target, &options,
      iree_allocator_system(), &emitted, &artifact_));
  ASSERT_TRUE(emitted);
  ASSERT_NE(artifact_.target_artifact_data, nullptr);

  iree_hal_amd_xdna_aie2p_target_t image_target;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
      IREE_SV("amd.xdna.strix_halo.17f0_11"), 1, &image_target));
  IREE_ASSERT_OK(iree_hal_amd_xdna_image_create(
      artifact_.target_artifact_data, &image_target, iree_allocator_system(),
      &image_));
  loom_xdna_artifact_provider.deinitialize_artifact(
      &loom_xdna_artifact_provider, &artifact_, iree_allocator_system());

  uint32_t entry_ordinal = UINT32_MAX;
  IREE_ASSERT_OK(iree_hal_amd_xdna_image_find_entry(
      image_, IREE_SV("resident_neighbor"), &entry_ordinal));
  const iree_hal_amd_xdna_image_tables_t* tables =
      iree_hal_amd_xdna_image_tables(image_);
  const iree_xdna_elf_entry_record_t entry =
      iree_hal_amd_xdna_image_tables_entry(tables, entry_ordinal);
  EXPECT_EQ(entry.binding_count, 0u);
  ASSERT_EQ(entry.invocation_count, 2u);

  const iree_xdna_elf_invocation_record_t establishing =
      iree_hal_amd_xdna_image_tables_invocation(tables, entry.first_invocation);
  const iree_xdna_elf_invocation_record_t continuing =
      iree_hal_amd_xdna_image_tables_invocation(tables,
                                                entry.first_invocation + 1);
  EXPECT_GT(establishing.byte_length, 16u);
  EXPECT_EQ(establishing.next_invocation, 1u);
  EXPECT_EQ(continuing.byte_length, 16u);
  EXPECT_EQ(continuing.next_invocation, 1u);
}

}  // namespace
}  // namespace loom
