// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/kernel_config_materializer.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/ops/target/ops.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

struct IndexDeleter {
  void operator()(loom_link_module_index_t* index) const {
    loom_link_module_index_free(index);
  }
};
using IndexPtr = std::unique_ptr<loom_link_module_index_t, IndexDeleter>;

struct PlanDeleter {
  void operator()(loom_link_plan_t* plan) const { loom_link_plan_free(plan); }
};
using PlanPtr = std::unique_ptr<loom_link_plan_t, PlanDeleter>;

class KernelConfigMaterializerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* Parse(iree_string_view_t source) {
    loom_text_parse_options_t options = {};
    options.diagnostic_sink = {loom_diagnostic_stderr_sink, nullptr};
    options.max_errors = 20;
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_text_parse(source, IREE_SV("provider.loom"), &context_,
                                  &block_pool_, &options, &module));
    IREE_ASSERT(module != nullptr);
    return module;
  }

  void Verify(const loom_module_t* module) {
    loom_verify_options_t options = {};
    options.sink.fn = loom_diagnostic_stderr_sink;
    loom_verify_result_t result = {};
    IREE_ASSERT_OK(loom_verify_module(module, &options, &result));
    ASSERT_EQ(result.error_count, 0u);
  }

  std::vector<uint8_t> Write(const loom_module_t* module) {
    iree_io_stream_t* stream = nullptr;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    IREE_CHECK_OK(loom_bytecode_write_module(
        module, stream, /*options=*/nullptr, &block_pool_));
    std::vector<uint8_t> bytes(iree_io_stream_length(stream));
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }

  IndexPtr CreateIndex() {
    loom_link_module_index_t* index = nullptr;
    IREE_CHECK_OK(loom_link_module_index_allocate(
        &context_, &block_pool_, iree_allocator_system(), &index));
    return IndexPtr(index);
  }

  PlanPtr BuildPlan(loom_link_module_index_t* index,
                    iree_string_view_t root_name) {
    const iree_string_view_t roots[] = {root_name};
    loom_link_plan_options_t options = {};
    options.mode = LOOM_LINK_PLAN_SELECTIVE;
    options.root_symbols = {IREE_ARRAYSIZE(roots), roots};
    loom_link_plan_t* plan = nullptr;
    IREE_CHECK_OK(
        loom_link_plan_build(index, &options, iree_allocator_system(), &plan));
    return PlanPtr(plan);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(KernelConfigMaterializerTest,
       IndexesKernelFacetSchemaAcrossProviderForms) {
  loom_module_t* source_module = Parse(IREE_SV(R"(
kernel.def @dispatch_rows(%element_count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%element_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%source: buffer) {
  kernel.return
}
)"));
  Verify(source_module);
  const std::vector<uint8_t> bytecode = Write(source_module);

  auto verify_index = [](const loom_link_module_index_t* index) {
    const loom_link_module_index_symbol_t* kernel =
        loom_link_module_index_lookup_name(index, IREE_SV("dispatch_rows"));
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(iree_any_bit_set(kernel->facets.schema.interfaces,
                                 LOOM_SYMBOL_INTERFACE_KERNEL));
    EXPECT_EQ(kernel->facets.schema.root_region_count, 2u);
    EXPECT_EQ(kernel->facets.schema.kernel_configuration_region_index_plus_one,
              1u);
    EXPECT_EQ(kernel->facets.schema.body_region_index_plus_one, 2u);
    EXPECT_EQ(loom_link_module_index_facet_count(index), 3u);
    EXPECT_EQ(loom_link_module_index_symbol_facet_ordinal(kernel, 0), 0u);
    EXPECT_EQ(loom_link_module_index_symbol_facet_ordinal(kernel, 1), 1u);
    EXPECT_EQ(loom_link_module_index_symbol_facet_ordinal(kernel, 2), 2u);
  };

  IndexPtr materialized_index = CreateIndex();
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      materialized_index.get(), source_module, /*options=*/nullptr,
      /*out_provider_ordinal=*/nullptr));
  verify_index(materialized_index.get());

  IndexPtr bytecode_index = CreateIndex();
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      bytecode_index.get(),
      iree_make_const_byte_span(bytecode.data(), bytecode.size()),
      IREE_SV("provider.loombc"), /*index_options=*/nullptr,
      /*options=*/nullptr, /*out_provider_ordinal=*/nullptr));
  verify_index(bytecode_index.get());

  loom_module_free(source_module);
}

TEST_F(KernelConfigMaterializerTest,
       ProjectsConfigurationWithoutReadingPoisonedImplementation) {
  loom_module_t* source_module = Parse(IREE_SV(R"(
target.generic<reference> @dispatch_target

func.def pure @implementation_only(%value: index) -> (index) {
  func.return %value : index
}

kernel.def target(@dispatch_target) @dispatch_rows(%element_count: index) {
  %one = index.constant 1 : index
  %sixty_three = index.constant 63 : index
  %subgroup_size = target.subgroup.size : index
  %rounded_count = index.add %element_count, %sixty_three : index
  %workgroup_count = index.div %rounded_count, %subgroup_size : index
  kernel.launch.config workgroups(%workgroup_count, %one, %one) workgroup_size(%subgroup_size, %one, %one) : index
} launch(%stride: index, %rows: index, %source: tensor<[%rows]xi32>) where [mul(%rows, 16), mul(%stride, 4)] {
  %unused = func.call @implementation_only(%stride) : (index) -> (index)
  kernel.return
}
)"));
  Verify(source_module);
  std::vector<uint8_t> bytecode = Write(source_module);
  loom_module_free(source_module);

  IndexPtr index = CreateIndex();
  loom_bytecode_index_options_t index_options = {};
  index_options.diagnostic_sink = {loom_diagnostic_stderr_sink, nullptr};
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      index.get(), iree_make_const_byte_span(bytecode.data(), bytecode.size()),
      IREE_SV("provider.loombc"), &index_options, /*options=*/nullptr,
      /*out_provider_ordinal=*/nullptr));
  const loom_link_module_index_symbol_t* indexed_kernel =
      loom_link_module_index_lookup_name(index.get(), IREE_SV("dispatch_rows"));
  ASSERT_NE(indexed_kernel, nullptr);
  PlanPtr plan = BuildPlan(index.get(), IREE_SV("dispatch_rows"));
  const loom_link_module_index_symbol_t* implementation_only =
      loom_link_module_index_lookup_name(index.get(),
                                         IREE_SV("implementation_only"));
  ASSERT_NE(implementation_only, nullptr);
  EXPECT_TRUE(
      loom_link_plan_contains_symbol(plan.get(), implementation_only->ordinal));

  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_symbol_provider(index.get(), indexed_kernel);
  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_symbol_module(index.get(), indexed_kernel);
  const loom_bytecode_module_metadata_t* metadata =
      &provider->bytecode.metadata
           .modules[indexed_module->provider_module_ordinal];
  const loom_bytecode_symbol_metadata_t* symbol =
      &metadata->symbols[indexed_kernel->module_symbol_ordinal];
  ASSERT_NE(symbol->kernel_workload_region_payload_ordinal_plus_one, 0u);
  ASSERT_NE(symbol->body_region_payload_ordinal_plus_one, 0u);
  const loom_bytecode_region_payload_metadata_t* config_payload =
      &metadata->region_payloads
           [symbol->first_region_payload_index +
            symbol->kernel_workload_region_payload_ordinal_plus_one - 1];
  const loom_bytecode_region_payload_metadata_t* body_payload =
      &metadata
           ->region_payloads[symbol->first_region_payload_index +
                             symbol->body_region_payload_ordinal_plus_one - 1];
  ASSERT_GT(config_payload->length, 0u);
  ASSERT_GT(body_payload->length, 0u);
  std::fill_n(bytecode.data() + body_payload->absolute_offset,
              body_payload->length, UINT8_C(0xFF));

  loom_link_plan_materialization_environment_t environment = {};
  environment.context = &context_;
  environment.block_pool = &block_pool_;
  environment.allocator = iree_allocator_system();
  loom_link_kernel_config_materialization_t materialization = {};
  IREE_ASSERT_OK(loom_link_plan_materialize_bytecode_kernel_config(
      plan.get(), indexed_kernel->ordinal, &environment,
      IREE_SV("projected.config"), &materialization));
  std::unique_ptr<loom_module_t, decltype(&loom_module_free)> projected_module(
      materialization.module, loom_module_free);
  Verify(projected_module.get());

  ASSERT_EQ(projected_module->symbols.count, 3u);
  ASSERT_TRUE(loom_symbol_ref_is_valid(materialization.kernel_declaration));
  ASSERT_TRUE(loom_symbol_ref_is_valid(materialization.configuration_function));
  loom_op_t* kernel_op =
      projected_module->symbols
          .entries[materialization.kernel_declaration.symbol_id]
          .defining_op;
  ASSERT_TRUE(loom_kernel_decl_isa(kernel_op));
  loom_func_like_t kernel =
      loom_func_like_cast(projected_module.get(), kernel_op);
  EXPECT_EQ(loom_func_like_visibility(kernel), 0u);
  const loom_symbol_ref_t target_ref = loom_func_like_target(kernel);
  ASSERT_TRUE(loom_symbol_ref_is_valid(target_ref));
  ASSERT_EQ(target_ref.module_id, 0u);
  ASSERT_LT(target_ref.symbol_id, projected_module->symbols.count);
  EXPECT_TRUE(loom_target_decl_isa(
      projected_module->symbols.entries[target_ref.symbol_id].defining_op));
  const loom_value_slice_t workloads =
      loom_kernel_workload_arg_ids(projected_module.get(), kernel_op);
  ASSERT_EQ(workloads.count, 1u);
  uint16_t kernel_argument_count = 0;
  const loom_value_id_t* kernel_arguments =
      loom_func_like_arg_ids(kernel, &kernel_argument_count);
  ASSERT_EQ(kernel_argument_count, 3u);
  const loom_type_t tensor_type =
      loom_module_value_type(projected_module.get(), kernel_arguments[2]);
  ASSERT_TRUE(loom_type_is_tensor(tensor_type));
  EXPECT_EQ(loom_type_dim_value_id_at(tensor_type, 0), kernel_arguments[1]);
  uint16_t kernel_predicate_count = 0;
  loom_func_like_predicates(kernel, &kernel_predicate_count);
  EXPECT_EQ(kernel_predicate_count, 2u);

  loom_op_t* helper_op =
      projected_module->symbols
          .entries[materialization.configuration_function.symbol_id]
          .defining_op;
  ASSERT_TRUE(loom_func_def_isa(helper_op));
  loom_func_like_t helper =
      loom_func_like_cast(projected_module.get(), helper_op);
  EXPECT_EQ(loom_func_like_visibility(helper), 0u);
  EXPECT_EQ(loom_func_like_purity(helper), LOOM_FUNC_PURITY_PURE);
  EXPECT_EQ(loom_func_like_inline_policy(helper), LOOM_INLINE_POLICY_INLINE);
  EXPECT_EQ(loom_func_like_target(helper).module_id, target_ref.module_id);
  EXPECT_EQ(loom_func_like_target(helper).symbol_id, target_ref.symbol_id);
  uint16_t helper_argument_count = 0;
  loom_func_like_arg_ids(helper, &helper_argument_count);
  ASSERT_EQ(helper_argument_count, 1u);
  uint16_t helper_predicate_count = 0;
  loom_func_like_predicates(helper, &helper_predicate_count);
  ASSERT_EQ(helper_predicate_count, 0u);
  ASSERT_EQ(helper_op->result_count, 3u);

  loom_block_t* helper_block =
      loom_region_entry_block(loom_func_like_body(helper));
  bool saw_division = false;
  bool saw_target_query = false;
  const loom_op_t* op = nullptr;
  loom_block_for_each_op(helper_block, op) {
    saw_division |= loom_index_div_isa(op);
    saw_target_query |= loom_target_subgroup_size_isa(op);
  }
  EXPECT_TRUE(saw_division);
  EXPECT_TRUE(saw_target_query);
  ASSERT_TRUE(loom_func_return_isa(helper_block->last_op));
  EXPECT_EQ(helper_block->last_op->operand_count, 3u);

  const std::vector<uint8_t> projected_bytecode = Write(projected_module.get());
  loom_bytecode_read_options_t read_options = {};
  read_options.diagnostic_sink = {loom_diagnostic_stderr_sink, nullptr};
  read_options.verify_module = true;
  read_options.verify_max_errors = 20;
  loom_bytecode_read_result_t read_result = {};
  loom_module_t* roundtrip_module = nullptr;
  IREE_ASSERT_OK(loom_bytecode_read_module(
      iree_make_const_byte_span(projected_bytecode.data(),
                                projected_bytecode.size()),
      IREE_SV("projected.loombc"), &context_, &block_pool_, &read_options,
      &read_result, &roundtrip_module, iree_allocator_system()));
  ASSERT_EQ(read_result.error_count, 0u);
  ASSERT_NE(roundtrip_module, nullptr);
  EXPECT_EQ(roundtrip_module->symbols.count, 3u);
  loom_module_free(roundtrip_module);

  iree_arena_allocator_t materialization_arena;
  iree_arena_initialize(&block_pool_, &materialization_arena);
  loom_link_plan_materialization_t full_materialization = {};
  iree_status_t full_status =
      loom_link_plan_materialize(plan.get(), &environment, IREE_SV("full"),
                                 &materialization_arena, &full_materialization);
  IREE_EXPECT_NOT_OK(full_status);
  loom_module_free(full_materialization.module);
  iree_arena_deinitialize(&materialization_arena);
}

}  // namespace
}  // namespace loom
