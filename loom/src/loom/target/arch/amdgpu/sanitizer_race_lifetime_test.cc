// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/pass_environment.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/pass/builtin_registry.h"
#include "loom/pass/tooling.h"
#include "loom/target/arch/amdgpu/abi/tsan.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/lower/descriptor_ref.h"
#include "loom/target/arch/amdgpu/lower/lower.h"
#include "loom/target/arch/amdgpu/ops/registry.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/testing/module_ptr.h"

namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class AmdgpuSanitizerRaceLifetimeTest : public ::testing::Test {
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
        .diagnostic_sink = {},
        .max_errors = 20,
    };
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(
        loom_text_parse(source, IREE_SV("sanitizer_race_lifetime.loom"),
                        &context_, &block_pool_, &parse_options, &module));
    return ModulePtr(module);
  }

  std::string SourceForTarget(const char* target) {
    std::string source = "amdgpu.target<";
    source.append(target);
    source.append(R"(> @target

kernel.def target(@target) @lifetime_sync() {
  %unit = index.constant 1 : index
  %threads = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%threads, %unit, %unit) : index
} launch() {
  %base = index.constant 0 : offset
  %bytes = index.constant 2048 : offset
  %scratch = buffer.alloca<workgroup> align(16) %bytes : buffer
  %view = buffer.view %scratch[%base] : buffer -> view<2048xi8>
  sanitizer.race.access<write> %view[0] : view<2048xi8>
  sanitizer.race.sync<workgroup> scope(workgroup) ordering(acq_rel)
  kernel.return
}
)");
    return source;
  }

  iree_status_t RunSourceToLow(loom_module_t* module) {
    const loom_target_low_legality_provider_t* legality_providers[] = {
        loom_amdgpu_low_legality_provider(),
    };
    const loom_target_low_legality_provider_list_t legality_provider_list =
        loom_target_low_legality_provider_list_make(
            legality_providers, IREE_ARRAYSIZE(legality_providers));
    const loom_codegen_pass_environment_options_t environment_options = {
        .descriptor_registry = &low_registry_.registry,
        .lower_policy_registry = &policy_registry_,
        .legality_provider_list = &legality_provider_list,
        .legalizer_registry = nullptr,
        .math_policy_registry = nullptr,
        .compile_report = nullptr,
        .target_environment = nullptr,
    };
    loom_codegen_pass_environment_storage_t environment_storage;
    loom_pass_environment_t environment =
        loom_codegen_pass_environment_storage_initialize(
            &environment_options, /*function_versions=*/nullptr,
            &environment_storage);
    loom_pass_tool_run_options_t run_options = {
        .registry = loom_pass_builtin_registry(),
        .environment = environment,
        .function_versions = nullptr,
        .predicate_provider = {},
        .block_pool = &block_pool_,
    };
    loom_pass_run_result_t run_result = {};
    iree_status_t status = loom_pass_tool_run_flat_pipeline(
        module, IREE_SV("source-to-low"), &run_options, &run_result);
    if (iree_status_is_ok(status) && run_result.error_count > 0) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "source-to-low pipeline emitted %u errors",
                                run_result.error_count);
    }
    return status;
  }

  loom_op_t* FindKernel(loom_module_t* module, iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    if (name_id == LOOM_STRING_ID_INVALID) {
      return nullptr;
    }
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
      return nullptr;
    }
    return module->symbols.entries[symbol_id].defining_op;
  }

  void BindKernel(loom_module_t* module, loom_op_t* kernel) {
    module_ = module;
    body_ = loom_low_kernel_def_body(kernel);
    const loom_string_id_t descriptor_set_id =
        loom_low_kernel_def_descriptor_set(kernel);
    descriptor_set_ = loom_low_descriptor_registry_lookup(
        &low_registry_.registry,
        loom_string_table_get(&module->strings, descriptor_set_id));
  }

  iree_string_view_t String(loom_string_id_t string_id) const {
    return loom_string_table_get(&module_->strings, string_id);
  }

  const loom_low_descriptor_t* DescriptorForRef(
      loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    if (!descriptor_set_) {
      return nullptr;
    }
    const uint32_t descriptor_ordinal =
        loom_amdgpu_descriptor_ref_ordinal(descriptor_set_, descriptor_ref);
    return loom_low_descriptor_set_descriptor_at(descriptor_set_,
                                                 descriptor_ordinal);
  }

  bool LowOpHasDescriptorRef(
      const loom_op_t* op, loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    if (!loom_low_op_isa(op)) {
      return false;
    }
    const loom_low_descriptor_t* descriptor = DescriptorForRef(descriptor_ref);
    return descriptor && loom_low_op_descriptor(op) ==
                             loom_low_descriptor_set_descriptor_ordinal(
                                 descriptor_set_, descriptor);
  }

  std::vector<loom_op_t*> OpsForDescriptorRef(
      loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    std::vector<loom_op_t*> result;
    loom_block_t* block = nullptr;
    loom_region_for_each_block(body_, block) {
      loom_op_t* op = nullptr;
      loom_block_for_each_op(block, op) {
        if (LowOpHasDescriptorRef(op, descriptor_ref)) {
          result.push_back(op);
        }
      }
    }
    return result;
  }

  std::vector<loom_op_t*> OpsForDescriptorRefInBlock(
      loom_block_t* block, loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    std::vector<loom_op_t*> result;
    loom_op_t* op = nullptr;
    loom_block_for_each_op(block, op) {
      if (LowOpHasDescriptorRef(op, descriptor_ref)) {
        result.push_back(op);
      }
    }
    return result;
  }

  bool TryGetAttrI64(loom_named_attr_slice_t attrs, iree_string_view_t name,
                     int64_t* out_value) const {
    for (iree_host_size_t i = 0; i < attrs.count; ++i) {
      if (iree_string_view_equal(String(attrs.entries[i].name_id), name)) {
        *out_value = loom_attr_as_i64(attrs.entries[i].value);
        return true;
      }
    }
    return false;
  }

  int64_t LowOpOffset(const loom_op_t* op) const {
    int64_t offset = 0;
    TryGetAttrI64(loom_low_op_attrs(op), IREE_SV("offset"), &offset);
    return offset;
  }

  loom_op_t* FindDescriptorBetween(loom_amdgpu_descriptor_ref_t descriptor_ref,
                                   const loom_op_t* after,
                                   const loom_op_t* before) const {
    for (loom_op_t* op = after->next_op; op && op != before; op = op->next_op) {
      if (LowOpHasDescriptorRef(op, descriptor_ref)) {
        return op;
      }
    }
    return nullptr;
  }

  bool BlockHasDescriptor(loom_block_t* block,
                          loom_amdgpu_descriptor_ref_t descriptor_ref) const {
    return !OpsForDescriptorRefInBlock(block, descriptor_ref).empty();
  }

  loom_op_t* FindDispatchGenerationShift() const {
    for (loom_op_t* op : OpsForDescriptorRef(
             LOOM_AMDGPU_DESCRIPTOR_REF_S_LSHR_B64_RHS_INLINE)) {
      int64_t shift = 0;
      if (!TryGetAttrI64(loom_low_op_attrs(op), IREE_SV("shift"), &shift) ||
          shift != LOOM_AMDGPU_TSAN_SHADOW_GENERATION_WINDOW_SHIFT) {
        continue;
      }
      const loom_value_id_t input = loom_low_op_operands(op).values[0];
      const loom_op_t* input_op =
          loom_value_def_op(loom_module_value(module_, input));
      if (loom_low_live_in_isa(input_op) &&
          iree_string_view_equal(String(loom_low_live_in_source(input_op)),
                                 IREE_SV("amdgpu.dispatch_id"))) {
        return op;
      }
    }
    return nullptr;
  }

  void ExpectTagAcquireBlock(loom_block_t* block,
                             loom_value_id_t record_base) const {
    std::vector<loom_op_t*> loads;
    for (loom_op_t* op : OpsForDescriptorRefInBlock(
             block, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR)) {
      const loom_value_slice_t operands = loom_low_op_operands(op);
      if (operands.count >= 2 && operands.values[1] == record_base) {
        loads.push_back(op);
      }
    }
    ASSERT_EQ(loads.size(), 2u);
    EXPECT_EQ(LowOpOffset(loads[0]),
              LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_LAYOUT_TAG_OFFSET);
    EXPECT_EQ(LowOpOffset(loads[1]),
              LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_GENERATION_TAG_OFFSET);
    for (iree_host_size_t i = 0; i < loads.size(); ++i) {
      const loom_op_t* end =
          i + 1 < loads.size() ? loads[i + 1] : block->last_op;
      loom_op_t* wait = FindDescriptorBetween(
          LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT, loads[i], end);
      ASSERT_NE(wait, nullptr);
      int64_t count = -1;
      ASSERT_TRUE(
          TryGetAttrI64(loom_low_op_attrs(wait), IREE_SV("vmcnt"), &count));
      EXPECT_EQ(count, 0);
      EXPECT_NE(FindDescriptorBetween(LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL1_INV,
                                      wait, end),
                nullptr);
      EXPECT_NE(FindDescriptorBetween(LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_GL0_INV,
                                      wait, end),
                nullptr);
    }
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_low_descriptor_registry_t low_registry_ = {};
  loom_low_lower_policy_registry_t policy_registry_ = {};
  loom_module_t* module_ = nullptr;
  loom_region_t* body_ = nullptr;
  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
};

TEST_F(AmdgpuSanitizerRaceLifetimeTest,
       SourceToLowPublishesClearedGenerationBeforeUse) {
  const std::string source = SourceForTarget("gfx11-generic");
  ModulePtr module = Parse(iree_make_cstring_view(source.c_str()));
  IREE_ASSERT_OK(RunSourceToLow(module.get()));
  loom_op_t* kernel = FindKernel(module.get(), IREE_SV("lifetime_sync"));
  ASSERT_NE(kernel, nullptr);
  ASSERT_TRUE(loom_low_kernel_def_isa(kernel));
  BindKernel(module.get(), kernel);
  ASSERT_NE(descriptor_set_, nullptr);

  loom_op_t* generation_shift_op = FindDispatchGenerationShift();
  ASSERT_NE(generation_shift_op, nullptr);
  int64_t generation_shift = 0;
  ASSERT_TRUE(TryGetAttrI64(loom_low_op_attrs(generation_shift_op),
                            IREE_SV("shift"), &generation_shift));
  EXPECT_EQ(generation_shift, LOOM_AMDGPU_TSAN_SHADOW_GENERATION_WINDOW_SHIFT);
  const loom_value_id_t dispatch_id =
      loom_low_op_operands(generation_shift_op).values[0];
  const loom_op_t* dispatch_id_op =
      loom_value_def_op(loom_module_value(module_, dispatch_id));
  ASSERT_TRUE(loom_low_live_in_isa(dispatch_id_op));
  EXPECT_TRUE(
      iree_string_view_equal(String(loom_low_live_in_source(dispatch_id_op)),
                             IREE_SV("amdgpu.dispatch_id")));

  const auto b128_stores =
      OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B128_SADDR);
  ASSERT_EQ(b128_stores.size(), 3u);
  loom_op_t* publish_store = nullptr;
  for (loom_op_t* op : b128_stores) {
    if (LowOpOffset(op) ==
        LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_LAYOUT_TAG_OFFSET) {
      publish_store = op;
    }
  }
  ASSERT_NE(publish_store, nullptr);
  const loom_value_id_t record_base =
      loom_low_op_operands(publish_store).values[2];
  loom_op_t* clear_store = nullptr;
  for (loom_op_t* op : b128_stores) {
    if (loom_low_op_operands(op).values[2] == record_base &&
        LowOpOffset(op) ==
            LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_CHUNK_STATE_OFFSET) {
      clear_store = op;
    }
  }
  ASSERT_NE(clear_store, nullptr);
  EXPECT_EQ(loom_low_op_operands(clear_store).values[2], record_base);

  std::vector<loom_op_t*> tag_loads;
  for (loom_op_t* op :
       OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR)) {
    const loom_value_slice_t operands = loom_low_op_operands(op);
    if (operands.count >= 2 && operands.values[1] == record_base &&
        (LowOpOffset(op) ==
             LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_LAYOUT_TAG_OFFSET ||
         LowOpOffset(op) ==
             LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_GENERATION_TAG_OFFSET)) {
      tag_loads.push_back(op);
    }
  }
  ASSERT_EQ(tag_loads.size(), 4u);
  uint32_t layout_load_count = 0;
  uint32_t generation_load_count = 0;
  for (loom_op_t* load : tag_loads) {
    layout_load_count += LowOpOffset(load) ==
                         LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_LAYOUT_TAG_OFFSET;
    generation_load_count +=
        LowOpOffset(load) ==
        LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_GENERATION_TAG_OFFSET;
    int64_t globally_coherent = 0;
    ASSERT_TRUE(TryGetAttrI64(loom_low_op_attrs(load), IREE_SV("glc"),
                              &globally_coherent));
    EXPECT_EQ(globally_coherent, 1);
  }
  EXPECT_EQ(layout_load_count, 2u);
  EXPECT_EQ(generation_load_count, 2u);

  loom_block_t* wait_block = nullptr;
  for (loom_block_t* block :
       {tag_loads[0]->parent_block, tag_loads[1]->parent_block,
        tag_loads[2]->parent_block, tag_loads[3]->parent_block}) {
    const loom_op_t* terminator = loom_block_const_last_op(block);
    if (loom_low_cond_br_isa(terminator) &&
        loom_low_cond_br_false_dest(terminator) == block) {
      wait_block = block;
    }
  }
  ASSERT_NE(wait_block, nullptr);
  const loom_op_t* wait_terminator = loom_block_const_last_op(wait_block);
  loom_block_t* continuation_block =
      loom_low_cond_br_true_dest(wait_terminator);

  loom_block_t* initial_block = nullptr;
  for (loom_op_t* load : tag_loads) {
    if (load->parent_block != wait_block) {
      initial_block = load->parent_block;
    }
  }
  ASSERT_NE(initial_block, nullptr);
  ExpectTagAcquireBlock(initial_block, record_base);
  ExpectTagAcquireBlock(wait_block, record_base);
  const loom_op_t* initial_terminator = loom_block_const_last_op(initial_block);
  ASSERT_TRUE(loom_low_cond_br_isa(initial_terminator));
  EXPECT_EQ(loom_low_cond_br_true_dest(initial_terminator), continuation_block);
  loom_block_t* mismatch_block =
      loom_low_cond_br_false_dest(initial_terminator);
  const loom_op_t* mismatch_terminator =
      loom_block_const_last_op(mismatch_block);
  ASSERT_TRUE(loom_low_cond_br_isa(mismatch_terminator));
  EXPECT_EQ(loom_low_cond_br_false_dest(mismatch_terminator), wait_block);
  EXPECT_TRUE(BlockHasDescriptor(mismatch_block,
                                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_ULT_U32));

  loom_block_t* initializer_block =
      loom_low_cond_br_true_dest(mismatch_terminator);
  const loom_op_t* initializer_terminator =
      loom_block_const_last_op(initializer_block);
  ASSERT_TRUE(loom_low_br_isa(initializer_terminator));
  loom_block_t* clear_header_block = loom_low_br_dest(initializer_terminator);
  loom_block_t* clear_body_block = clear_store->parent_block;
  const loom_op_t* clear_header_terminator =
      loom_block_const_last_op(clear_header_block);
  ASSERT_TRUE(loom_low_cond_br_isa(clear_header_terminator));
  EXPECT_EQ(loom_low_cond_br_true_dest(clear_header_terminator),
            clear_body_block);
  EXPECT_EQ(loom_low_cond_br_false_dest(clear_header_terminator),
            publish_store->parent_block);
  const loom_op_t* clear_body_terminator =
      loom_block_const_last_op(clear_body_block);
  ASSERT_TRUE(loom_low_br_isa(clear_body_terminator));
  EXPECT_EQ(loom_low_br_dest(clear_body_terminator), clear_header_block);
  const auto exec_narrow_ops = OpsForDescriptorRefInBlock(
      clear_body_block, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64);
  const auto exec_restore_ops = OpsForDescriptorRefInBlock(
      clear_body_block, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC);
  ASSERT_EQ(exec_narrow_ops.size(), 1u);
  ASSERT_EQ(exec_restore_ops.size(), 1u);
  EXPECT_LT(exec_narrow_ops[0]->block_ordinal, clear_store->block_ordinal);
  EXPECT_LT(clear_store->block_ordinal, exec_restore_ops[0]->block_ordinal);
  uint32_t bitmap_step_count = 0;
  loom_op_t* clear_body_op = nullptr;
  loom_block_for_each_op(clear_body_block, clear_body_op) {
    int64_t immediate = 0;
    if (TryGetAttrI64(loom_low_op_attrs(clear_body_op), IREE_SV("imm32"),
                      &immediate) &&
        immediate == LOOM_AMDGPU_TSAN_SHADOW_CHUNK_STATE_BITMAP_ALIGNMENT) {
      ++bitmap_step_count;
    }
  }
  EXPECT_EQ(bitmap_step_count, 1u);

  loom_block_t* publish_block = publish_store->parent_block;
  std::vector<loom_op_t*> header_stores;
  for (loom_op_t* op : OpsForDescriptorRefInBlock(
           publish_block, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR)) {
    if (loom_low_op_operands(op).values[2] == record_base) {
      header_stores.push_back(op);
    }
  }
  ASSERT_EQ(header_stores.size(), 2u);
  EXPECT_EQ(LowOpOffset(header_stores[0]),
            LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_EPOCH_OFFSET);
  EXPECT_EQ(LowOpOffset(header_stores[1]),
            LOOM_AMDGPU_TSAN_WORKGROUP_SHADOW_RESERVED_OFFSET);
  loom_op_t* waitcnt = FindDescriptorBetween(
      LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT, header_stores[1], publish_store);
  loom_op_t* waitcnt_vscnt =
      FindDescriptorBetween(LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT_VSCNT,
                            header_stores[1], publish_store);
  ASSERT_NE(waitcnt, nullptr);
  ASSERT_NE(waitcnt_vscnt, nullptr);
  int64_t count = -1;
  ASSERT_TRUE(
      TryGetAttrI64(loom_low_op_attrs(waitcnt), IREE_SV("vmcnt"), &count));
  EXPECT_EQ(count, 0);
  ASSERT_TRUE(TryGetAttrI64(loom_low_op_attrs(waitcnt_vscnt), IREE_SV("vscnt"),
                            &count));
  EXPECT_EQ(count, 0);
  const loom_op_t* publish_terminator = loom_block_const_last_op(publish_block);
  ASSERT_TRUE(loom_low_br_isa(publish_terminator));
  EXPECT_EQ(loom_low_br_dest(publish_terminator), continuation_block);

  for (loom_block_t* block :
       {initial_block, mismatch_block, initializer_block, clear_header_block,
        clear_body_block, publish_block, wait_block}) {
    EXPECT_FALSE(
        BlockHasDescriptor(block, LOOM_AMDGPU_DESCRIPTOR_REF_S_BARRIER));
  }

  const auto claim_ops = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_OR_B32_RTN_SADDR);
  const auto ready_ops = OpsForDescriptorRef(
      LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_OR_B32_SADDR);
  ASSERT_EQ(claim_ops.size(), 1u);
  ASSERT_EQ(ready_ops.size(), 1u);
  EXPECT_EQ(loom_low_op_operands(claim_ops[0]).values[0],
            loom_low_op_operands(ready_ops[0]).values[0]);

  loom_op_t* chunk_clear_store = nullptr;
  for (loom_op_t* op : b128_stores) {
    if (loom_low_op_operands(op).values[2] != record_base) {
      chunk_clear_store = op;
    }
  }
  ASSERT_NE(chunk_clear_store, nullptr);
  EXPECT_EQ(LowOpOffset(chunk_clear_store), 0);

  loom_block_t* chunk_clear_loop_block = chunk_clear_store->parent_block;
  EXPECT_TRUE(BlockHasDescriptor(chunk_clear_loop_block,
                                 LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32));
  const loom_op_t* chunk_clear_loop_terminator =
      loom_block_const_last_op(chunk_clear_loop_block);
  ASSERT_TRUE(loom_low_cond_br_isa(chunk_clear_loop_terminator));
  loom_block_t* chunk_clear_continue_block =
      loom_low_cond_br_true_dest(chunk_clear_loop_terminator);
  const loom_op_t* chunk_clear_continue_terminator =
      loom_block_const_last_op(chunk_clear_continue_block);
  ASSERT_TRUE(loom_low_br_isa(chunk_clear_continue_terminator));
  EXPECT_EQ(loom_low_br_dest(chunk_clear_continue_terminator),
            chunk_clear_loop_block);

  loom_block_t* chunk_publish_block = ready_ops[0]->parent_block;
  EXPECT_EQ(loom_low_cond_br_false_dest(chunk_clear_loop_terminator),
            chunk_publish_block);
  EXPECT_TRUE(BlockHasDescriptor(chunk_publish_block,
                                 LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT));
  EXPECT_TRUE(BlockHasDescriptor(chunk_publish_block,
                                 LOOM_AMDGPU_DESCRIPTOR_REF_S_WAITCNT_VSCNT));

  loom_block_t* claim_block = claim_ops[0]->parent_block;
  const loom_op_t* claim_terminator = loom_block_const_last_op(claim_block);
  ASSERT_TRUE(loom_low_cond_br_isa(claim_terminator));
  loom_block_t* chunk_clear_entry_block =
      loom_low_cond_br_true_dest(claim_terminator);
  const loom_op_t* chunk_clear_entry_terminator =
      loom_block_const_last_op(chunk_clear_entry_block);
  ASSERT_TRUE(loom_low_br_isa(chunk_clear_entry_terminator));
  EXPECT_EQ(loom_low_br_dest(chunk_clear_entry_terminator),
            chunk_clear_loop_block);
  loom_block_t* retry_block = loom_low_cond_br_false_dest(claim_terminator);
  const loom_op_t* chunk_publish_terminator =
      loom_block_const_last_op(chunk_publish_block);
  ASSERT_TRUE(loom_low_br_isa(chunk_publish_terminator));
  EXPECT_EQ(loom_low_br_dest(chunk_publish_terminator), retry_block);
  const loom_op_t* retry_terminator = loom_block_const_last_op(retry_block);
  ASSERT_TRUE(loom_low_br_isa(retry_terminator));
  loom_block_t* chunk_check_block = loom_low_br_dest(retry_terminator);
  EXPECT_TRUE(BlockHasDescriptor(
      chunk_check_block, LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B32_SADDR));
  for (loom_block_t* block :
       {claim_block, chunk_clear_entry_block, chunk_clear_loop_block,
        chunk_clear_continue_block, chunk_publish_block, retry_block,
        chunk_check_block}) {
    EXPECT_FALSE(
        BlockHasDescriptor(block, LOOM_AMDGPU_DESCRIPTOR_REF_S_BARRIER));
  }
}

TEST_F(AmdgpuSanitizerRaceLifetimeTest,
       SelectsLifetimeProtocolForEveryMemoryOrderingModel) {
  // One target represents each memory-ordering model that can select the
  // AMDGPU race sanitizer.
  for (const char* target : {"gfx90a", "gfx9-4-generic", "gfx11-generic",
                             "gfx12-generic", "gfx12-5-generic"}) {
    SCOPED_TRACE(target);
    const std::string source = SourceForTarget(target);
    ModulePtr module = Parse(iree_make_cstring_view(source.c_str()));
    IREE_ASSERT_OK(RunSourceToLow(module.get()));
    loom_op_t* kernel = FindKernel(module.get(), IREE_SV("lifetime_sync"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_TRUE(loom_low_kernel_def_isa(kernel));
    BindKernel(module.get(), kernel);
    ASSERT_NE(descriptor_set_, nullptr);

    EXPECT_NE(FindDispatchGenerationShift(), nullptr);
    EXPECT_EQ(OpsForDescriptorRef(
                  LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_OR_B32_RTN_SADDR)
                  .size(),
              1u);
    EXPECT_EQ(OpsForDescriptorRef(
                  LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_ATOMIC_OR_B32_SADDR)
                  .size(),
              1u);
    const auto stores =
        OpsForDescriptorRef(LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B128_SADDR);
    ASSERT_EQ(stores.size(), 3u);
  }
}

}  // namespace
