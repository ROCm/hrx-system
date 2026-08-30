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
#include "loom/link/plan_materializer.h"
#include "loom/ops/config/ops.h"
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

  void AddText(loom_link_module_index_t* index, iree_string_view_t source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.diagnostic_sink = {loom_diagnostic_stderr_sink, nullptr};
    parse_options.max_errors = 20;
    IREE_CHECK_OK(loom_link_module_index_add_text(
        index, source, IREE_SV("provider.loom"), &parse_options,
        /*options=*/nullptr, /*out_provider_ordinal=*/nullptr));
  }

  const loom_symbol_t* FindSymbol(const loom_module_t* module,
                                  iree_string_view_t name) {
    for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
      const loom_symbol_t* symbol = &module->symbols.entries[i];
      if (iree_string_view_equal(module->strings.entries[symbol->name_id],
                                 name)) {
        return symbol;
      }
    }
    return nullptr;
  }

  loom_symbol_ref_t TargetConfiguration(
      const loom_link_plan_materialization_t& materialization,
      iree_host_size_t source_symbol_ordinal) {
    IREE_ASSERT_LT(source_symbol_ordinal, materialization.target_symbols.count);
    const loom_symbol_ref_t target_symbol =
        materialization.target_symbols.values[source_symbol_ordinal];
    IREE_ASSERT(loom_symbol_ref_is_valid(target_symbol));
    IREE_ASSERT_EQ(target_symbol.module_id, 0u);
    IREE_ASSERT_LT(target_symbol.symbol_id,
                   materialization.target_kernel_configurations.count);
    return materialization.target_kernel_configurations
        .values[target_symbol.symbol_id];
  }

  void VerifyBytecodeRoundTrip(const loom_module_t* module) {
    const std::vector<uint8_t> bytecode = Write(module);
    loom_bytecode_read_options_t options = {};
    options.diagnostic_sink = {loom_diagnostic_stderr_sink, nullptr};
    loom_bytecode_read_result_t result = {};
    loom_module_t* roundtrip_module = nullptr;
    IREE_ASSERT_OK(loom_bytecode_read_module(
        iree_make_const_byte_span(bytecode.data(), bytecode.size()),
        IREE_SV("roundtrip.loombc"), &context_, &block_pool_, &options, &result,
        &roundtrip_module, iree_allocator_system()));
    ASSERT_EQ(result.error_count, 0u);
    ASSERT_NE(roundtrip_module, nullptr);
    Verify(roundtrip_module);
    loom_module_free(roundtrip_module);
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
    options.mode = LOOM_LINK_PLAN_LINK;
    options.root_symbols = {IREE_ARRAYSIZE(roots), roots};
    loom_link_plan_t* plan = nullptr;
    IREE_CHECK_OK(
        loom_link_plan_build(index, &options, iree_allocator_system(), &plan));
    return PlanPtr(plan);
  }

  PlanPtr BuildConfigPlan(loom_link_module_index_t* index,
                          iree_host_size_t kernel_symbol_ordinal) {
    const loom_link_plan_root_facet_t root = {
        /*.symbol_ordinal=*/kernel_symbol_ordinal,
        /*.kind=*/LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION,
    };
    loom_link_plan_options_t options = {};
    options.mode = LOOM_LINK_PLAN_LINK;
    options.root_facets = {1, &root};
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
  const iree_string_view_t source = IREE_SV(R"(
kernel.def @dispatch_rows(%element_count: index) {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%element_count, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%source: buffer) {
  kernel.return
}
)");
  loom_module_t* source_module = Parse(source);
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
    EXPECT_EQ(kernel->facets.schema.facet_count, 3u);
    EXPECT_EQ(loom_link_module_index_facet_count(index), 3u);
    EXPECT_EQ(loom_link_module_index_symbol_facet_kind_at(kernel, 0),
              LOOM_LINK_SYMBOL_FACET_KERNEL_CONTRACT);
    EXPECT_EQ(loom_link_module_index_symbol_facet_kind_at(kernel, 1),
              LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION);
    EXPECT_EQ(loom_link_module_index_symbol_facet_kind_at(kernel, 2),
              LOOM_LINK_SYMBOL_FACET_KERNEL_IMPLEMENTATION);
    EXPECT_EQ(loom_link_module_index_symbol_facet_ordinal(
                  kernel, LOOM_LINK_SYMBOL_FACET_KERNEL_CONTRACT),
              0u);
    EXPECT_EQ(loom_link_module_index_symbol_facet_ordinal(
                  kernel, LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION),
              1u);
    EXPECT_EQ(loom_link_module_index_symbol_facet_ordinal(
                  kernel, LOOM_LINK_SYMBOL_FACET_KERNEL_IMPLEMENTATION),
              2u);
  };

  IndexPtr materialized_index = CreateIndex();
  IREE_ASSERT_OK(loom_link_module_index_add_materialized(
      materialized_index.get(), source_module, /*options=*/nullptr,
      /*out_provider_ordinal=*/nullptr));
  verify_index(materialized_index.get());

  IndexPtr text_index = CreateIndex();
  AddText(text_index.get(), source);
  verify_index(text_index.get());

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
       ReconstructsCompleteKernelAcrossProviderForms) {
  const iree_string_view_t source = IREE_SV(R"(
target.generic<reference> @dispatch_target

func.def pure @implementation_only(%value: index) -> (index) {
  func.return %value : index
}

kernel.def target(@dispatch_target) @dispatch_rows(%element_count: index) {
  %one = index.constant 1 : index
  %subgroup_size = target.subgroup.size : index
  %workgroup_count = index.div %element_count, %subgroup_size : index
  kernel.launch.config workgroups(%workgroup_count, %one, %one) workgroup_size(%subgroup_size, %one, %one) : index
} launch(%stride: index) {
  %unused = func.call @implementation_only(%stride) : (index) -> (index)
  kernel.return
}
)");
  loom_module_t* source_module = Parse(source);
  Verify(source_module);
  const std::vector<uint8_t> bytecode = Write(source_module);

  enum class ProviderForm {
    kMaterialized,
    kText,
    kBytecode,
  };
  const ProviderForm provider_forms[] = {
      ProviderForm::kMaterialized,
      ProviderForm::kText,
      ProviderForm::kBytecode,
  };
  for (ProviderForm provider_form : provider_forms) {
    IndexPtr index = CreateIndex();
    switch (provider_form) {
      case ProviderForm::kMaterialized:
        IREE_ASSERT_OK(loom_link_module_index_add_materialized(
            index.get(), source_module, /*options=*/nullptr,
            /*out_provider_ordinal=*/nullptr));
        break;
      case ProviderForm::kText:
        AddText(index.get(), source);
        break;
      case ProviderForm::kBytecode:
        IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
            index.get(),
            iree_make_const_byte_span(bytecode.data(), bytecode.size()),
            IREE_SV("provider.loombc"), /*index_options=*/nullptr,
            /*options=*/nullptr, /*out_provider_ordinal=*/nullptr));
        break;
    }

    const loom_link_module_index_symbol_t* kernel =
        loom_link_module_index_lookup_name(index.get(),
                                           IREE_SV("dispatch_rows"));
    const loom_link_module_index_symbol_t* implementation_only =
        loom_link_module_index_lookup_name(index.get(),
                                           IREE_SV("implementation_only"));
    const loom_link_module_index_symbol_t* target =
        loom_link_module_index_lookup_name(index.get(),
                                           IREE_SV("dispatch_target"));
    ASSERT_NE(kernel, nullptr);
    ASSERT_NE(implementation_only, nullptr);
    ASSERT_NE(target, nullptr);

    PlanPtr config_plan = BuildConfigPlan(index.get(), kernel->ordinal);
    EXPECT_EQ(loom_link_plan_symbol_count(config_plan.get()), 2u);
    EXPECT_EQ(loom_link_plan_facet_count(config_plan.get()), 3u);
    EXPECT_TRUE(
        loom_link_plan_contains_symbol(config_plan.get(), kernel->ordinal));
    EXPECT_TRUE(
        loom_link_plan_contains_symbol(config_plan.get(), target->ordinal));
    EXPECT_FALSE(loom_link_plan_contains_symbol(config_plan.get(),
                                                implementation_only->ordinal));
    EXPECT_TRUE(
        loom_link_plan_contains_facet(config_plan.get(), kernel->ordinal,
                                      LOOM_LINK_SYMBOL_FACET_KERNEL_CONTRACT));
    EXPECT_TRUE(loom_link_plan_contains_facet(
        config_plan.get(), kernel->ordinal,
        LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION));
    EXPECT_FALSE(loom_link_plan_contains_facet(
        config_plan.get(), kernel->ordinal,
        LOOM_LINK_SYMBOL_FACET_KERNEL_IMPLEMENTATION));

    PlanPtr full_plan = BuildPlan(index.get(), IREE_SV("dispatch_rows"));
    EXPECT_EQ(loom_link_plan_symbol_count(full_plan.get()), 3u);
    EXPECT_EQ(loom_link_plan_facet_count(full_plan.get()), 5u);
    EXPECT_TRUE(
        loom_link_plan_contains_symbol(full_plan.get(), kernel->ordinal));
    EXPECT_TRUE(
        loom_link_plan_contains_symbol(full_plan.get(), target->ordinal));
    EXPECT_TRUE(loom_link_plan_contains_symbol(full_plan.get(),
                                               implementation_only->ordinal));
    EXPECT_TRUE(loom_link_plan_contains_facet(
        full_plan.get(), kernel->ordinal,
        LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION));
    EXPECT_TRUE(loom_link_plan_contains_facet(
        full_plan.get(), kernel->ordinal,
        LOOM_LINK_SYMBOL_FACET_KERNEL_IMPLEMENTATION));

    loom_link_plan_materialization_environment_t environment = {};
    environment.context = &context_;
    environment.block_pool = &block_pool_;
    environment.allocator = iree_allocator_system();

    iree_arena_allocator_t configuration_arena;
    iree_arena_initialize(&block_pool_, &configuration_arena);
    loom_link_plan_materialization_t configuration_materialization = {};
    IREE_ASSERT_OK(loom_link_plan_materialize(
        config_plan.get(), &environment, IREE_SV("configuration"),
        &configuration_arena, &configuration_materialization));
    std::unique_ptr<loom_module_t, decltype(&loom_module_free)>
        configuration_module(configuration_materialization.module,
                             loom_module_free);
    Verify(configuration_module.get());
    ASSERT_EQ(configuration_module->symbols.count, 3u);
    const loom_symbol_ref_t configuration_kernel =
        configuration_materialization.target_symbols.values[kernel->ordinal];
    const loom_symbol_ref_t configuration_function =
        TargetConfiguration(configuration_materialization, kernel->ordinal);
    ASSERT_TRUE(loom_symbol_ref_is_valid(configuration_kernel));
    ASSERT_TRUE(loom_symbol_ref_is_valid(configuration_function));
    EXPECT_TRUE(loom_kernel_decl_isa(
        configuration_module->symbols.entries[configuration_kernel.symbol_id]
            .defining_op));
    EXPECT_TRUE(loom_func_def_isa(
        configuration_module->symbols.entries[configuration_function.symbol_id]
            .defining_op));
    EXPECT_EQ(
        configuration_module->symbols.entries[configuration_function.symbol_id]
            .defining_op->result_count,
        3u);
    EXPECT_EQ(
        FindSymbol(configuration_module.get(), IREE_SV("implementation_only")),
        nullptr);
    iree_arena_deinitialize(&configuration_arena);

    iree_arena_allocator_t materialization_arena;
    iree_arena_initialize(&block_pool_, &materialization_arena);
    loom_link_plan_materialization_t materialization = {};
    IREE_ASSERT_OK(loom_link_plan_materialize(
        full_plan.get(), &environment, IREE_SV("full"), &materialization_arena,
        &materialization));
    std::unique_ptr<loom_module_t, decltype(&loom_module_free)> full_module(
        materialization.module, loom_module_free);
    Verify(full_module.get());
    ASSERT_EQ(full_module->symbols.count, 3u);

    const loom_symbol_t* full_kernel =
        FindSymbol(full_module.get(), IREE_SV("dispatch_rows"));
    const loom_symbol_t* full_helper =
        FindSymbol(full_module.get(), IREE_SV("implementation_only"));
    const loom_symbol_t* full_target =
        FindSymbol(full_module.get(), IREE_SV("dispatch_target"));
    ASSERT_NE(full_kernel, nullptr);
    ASSERT_NE(full_helper, nullptr);
    ASSERT_NE(full_target, nullptr);
    ASSERT_TRUE(loom_kernel_def_isa(full_kernel->defining_op));
    ASSERT_TRUE(loom_func_def_isa(full_helper->defining_op));
    ASSERT_TRUE(loom_target_generic_isa(full_target->defining_op));

    loom_block_t* config_block = loom_region_entry_block(
        loom_kernel_def_config(full_kernel->defining_op));
    ASSERT_NE(config_block, nullptr);
    ASSERT_TRUE(loom_kernel_launch_config_isa(config_block->last_op));
    loom_block_t* body_block =
        loom_region_entry_block(loom_kernel_def_body(full_kernel->defining_op));
    ASSERT_NE(body_block, nullptr);
    ASSERT_TRUE(loom_kernel_return_isa(body_block->last_op));
    const loom_op_t* implementation_call = nullptr;
    const loom_op_t* body_op = nullptr;
    loom_block_for_each_op(body_block, body_op) {
      if (loom_func_call_isa(body_op)) {
        implementation_call = body_op;
      }
    }
    ASSERT_NE(implementation_call, nullptr);
    const loom_symbol_ref_t implementation_ref =
        loom_func_call_callee(implementation_call);
    ASSERT_EQ(implementation_ref.module_id, 0u);
    ASSERT_LT(implementation_ref.symbol_id, full_module->symbols.count);
    EXPECT_EQ(
        full_module->symbols.entries[implementation_ref.symbol_id].defining_op,
        full_helper->defining_op);
    const loom_symbol_ref_t target_ref =
        loom_kernel_def_target(full_kernel->defining_op);
    ASSERT_EQ(target_ref.module_id, 0u);
    ASSERT_LT(target_ref.symbol_id, full_module->symbols.count);
    EXPECT_EQ(full_module->symbols.entries[target_ref.symbol_id].defining_op,
              full_target->defining_op);
    VerifyBytecodeRoundTrip(full_module.get());
    iree_arena_deinitialize(&materialization_arena);
  }

  loom_module_free(source_module);
}

TEST_F(KernelConfigMaterializerTest,
       ProjectsSharedConfigurationClosureOncePerSourceModule) {
  const iree_string_view_t source = IREE_SV(R"(
func.def public pure @dispatch_rows$config(%value: index) -> (index) {
  func.return %value : index
}

func.def pure @rows_implementation(%value: index) -> (index) {
  func.return %value : index
}

func.def pure @columns_implementation(%value: index) -> (index) {
  func.return %value : index
}

kernel.def @dispatch_rows(%count: index) {
  %one = index.constant 1 : index
  %groups = func.call pure @dispatch_rows$config(%count) : (index) -> (index)
  kernel.launch.config workgroups(%groups, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%value: index) {
  %unused = func.call @rows_implementation(%value) : (index) -> (index)
  kernel.return
}

kernel.def @dispatch_columns(%count: index) {
  %one = index.constant 1 : index
  %groups = func.call pure @dispatch_rows$config(%count) : (index) -> (index)
  kernel.launch.config workgroups(%groups, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%value: index) {
  %unused = func.call @columns_implementation(%value) : (index) -> (index)
  kernel.return
}
)");
  loom_module_t* source_module = Parse(source);
  Verify(source_module);
  const std::vector<uint8_t> bytecode = Write(source_module);

  enum class ProviderForm {
    kMaterialized,
    kBytecode,
  };
  const ProviderForm provider_forms[] = {
      ProviderForm::kMaterialized,
      ProviderForm::kBytecode,
  };
  for (ProviderForm provider_form : provider_forms) {
    IndexPtr index = CreateIndex();
    if (provider_form == ProviderForm::kMaterialized) {
      IREE_ASSERT_OK(loom_link_module_index_add_materialized(
          index.get(), source_module, /*options=*/nullptr,
          /*out_provider_ordinal=*/nullptr));
    } else {
      IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
          index.get(),
          iree_make_const_byte_span(bytecode.data(), bytecode.size()),
          IREE_SV("provider.loombc"), /*index_options=*/nullptr,
          /*options=*/nullptr, /*out_provider_ordinal=*/nullptr));
    }

    const loom_link_module_index_symbol_t* rows =
        loom_link_module_index_lookup_name(index.get(),
                                           IREE_SV("dispatch_rows"));
    const loom_link_module_index_symbol_t* columns =
        loom_link_module_index_lookup_name(index.get(),
                                           IREE_SV("dispatch_columns"));
    const loom_link_module_index_symbol_t* shared =
        loom_link_module_index_lookup_name(index.get(),
                                           IREE_SV("dispatch_rows$config"));
    const loom_link_module_index_symbol_t* rows_implementation =
        loom_link_module_index_lookup_name(index.get(),
                                           IREE_SV("rows_implementation"));
    const loom_link_module_index_symbol_t* columns_implementation =
        loom_link_module_index_lookup_name(index.get(),
                                           IREE_SV("columns_implementation"));
    ASSERT_NE(rows, nullptr);
    ASSERT_NE(columns, nullptr);
    ASSERT_NE(shared, nullptr);
    ASSERT_NE(rows_implementation, nullptr);
    ASSERT_NE(columns_implementation, nullptr);

    const loom_link_plan_root_facet_t roots[] = {
        {
            /*.symbol_ordinal=*/rows->ordinal,
            /*.kind=*/LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION,
        },
        {
            /*.symbol_ordinal=*/columns->ordinal,
            /*.kind=*/LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION,
        },
    };
    loom_link_plan_options_t plan_options = {};
    plan_options.mode = LOOM_LINK_PLAN_LINK;
    plan_options.root_facets = {IREE_ARRAYSIZE(roots), roots};
    loom_link_plan_t* plan_storage = nullptr;
    IREE_ASSERT_OK(loom_link_plan_build(
        index.get(), &plan_options, iree_allocator_system(), &plan_storage));
    PlanPtr plan(plan_storage);
    EXPECT_TRUE(loom_link_plan_contains_symbol(plan.get(), shared->ordinal));
    EXPECT_FALSE(loom_link_plan_contains_symbol(plan.get(),
                                                rows_implementation->ordinal));
    EXPECT_FALSE(loom_link_plan_contains_symbol(
        plan.get(), columns_implementation->ordinal));

    loom_link_plan_materialization_environment_t environment = {};
    environment.context = &context_;
    environment.block_pool = &block_pool_;
    environment.allocator = iree_allocator_system();
    iree_arena_allocator_t materialization_arena;
    iree_arena_initialize(&block_pool_, &materialization_arena);
    loom_link_plan_materialization_t materialization = {};
    IREE_ASSERT_OK(loom_link_plan_materialize(
        plan.get(), &environment, IREE_SV("shared.config"),
        &materialization_arena, &materialization));
    std::unique_ptr<loom_module_t, decltype(&loom_module_free)> module(
        materialization.module, loom_module_free);
    Verify(module.get());
    ASSERT_EQ(module->symbols.count, 5u);
    const loom_symbol_t* shared_symbol =
        FindSymbol(module.get(), IREE_SV("dispatch_rows$config"));
    ASSERT_NE(shared_symbol, nullptr);
    EXPECT_TRUE(loom_func_def_isa(shared_symbol->defining_op));
    EXPECT_EQ(FindSymbol(module.get(), IREE_SV("rows_implementation")),
              nullptr);
    EXPECT_EQ(FindSymbol(module.get(), IREE_SV("columns_implementation")),
              nullptr);

    const loom_symbol_ref_t rows_config =
        TargetConfiguration(materialization, rows->ordinal);
    const loom_symbol_ref_t columns_config =
        TargetConfiguration(materialization, columns->ordinal);
    ASSERT_TRUE(loom_symbol_ref_is_valid(rows_config));
    ASSERT_TRUE(loom_symbol_ref_is_valid(columns_config));
    EXPECT_NE(rows_config.symbol_id, columns_config.symbol_id);
    EXPECT_NE(rows_config.symbol_id,
              (uint16_t)(shared_symbol - module->symbols.entries));
    EXPECT_NE(columns_config.symbol_id,
              (uint16_t)(shared_symbol - module->symbols.entries));
    EXPECT_TRUE(loom_func_def_isa(
        module->symbols.entries[rows_config.symbol_id].defining_op));
    EXPECT_TRUE(loom_func_def_isa(
        module->symbols.entries[columns_config.symbol_id].defining_op));
    iree_arena_deinitialize(&materialization_arena);
  }

  loom_module_free(source_module);
}

TEST_F(KernelConfigMaterializerTest,
       ProjectsConfigurationWithoutReadingPoisonedImplementation) {
  loom_module_t* source_module = Parse(IREE_SV(R"(
target.generic<reference> @dispatch_target

config.def @configuration_bias = 63 : index

func.def public pure @configuration_only(%value: index) -> (index) {
  %bias = config.get @configuration_bias : index
  %result = index.add %value, %bias : index
  func.return %result : index
}

func.def pure @implementation_only(%value: index) -> (index) {
  func.return %value : index
}

kernel.def target(@dispatch_target) @dispatch_rows(%element_count: index) {
  %one = index.constant 1 : index
  %two = index.constant 2 : index
  %subgroup_size = target.subgroup.size : index
  %rounded_count = func.call pure @configuration_only(%element_count) : (index) -> (index)
  %workgroup_count = index.div %rounded_count, %subgroup_size : index
  kernel.launch.config workgroups(%workgroup_count, %two, %one) workgroup_size(%subgroup_size, %one, %one) cluster_size(%one, %two, %one) : index
} launch(%stride: index, %rows: index, %source: tensor<[%rows]xi32>) where [mul(%rows, 16), mul(%stride, 4)] {
  %unused = func.call @implementation_only(%stride) : (index) -> (index)
  kernel.return
}
)"));
  Verify(source_module);
  std::vector<uint8_t> bytecode = Write(source_module);

  {
    IndexPtr materialized_index = CreateIndex();
    IREE_ASSERT_OK(loom_link_module_index_add_materialized(
        materialized_index.get(), source_module, /*options=*/nullptr,
        /*out_provider_ordinal=*/nullptr));
    const loom_link_module_index_symbol_t* materialized_kernel =
        loom_link_module_index_lookup_name(materialized_index.get(),
                                           IREE_SV("dispatch_rows"));
    const loom_link_module_index_symbol_t* materialized_implementation_only =
        loom_link_module_index_lookup_name(materialized_index.get(),
                                           IREE_SV("implementation_only"));
    const loom_link_module_index_symbol_t* materialized_configuration_only =
        loom_link_module_index_lookup_name(materialized_index.get(),
                                           IREE_SV("configuration_only"));
    const loom_link_module_index_symbol_t* materialized_configuration_bias =
        loom_link_module_index_lookup_name(materialized_index.get(),
                                           IREE_SV("configuration_bias"));
    ASSERT_NE(materialized_kernel, nullptr);
    ASSERT_NE(materialized_implementation_only, nullptr);
    ASSERT_NE(materialized_configuration_only, nullptr);
    ASSERT_NE(materialized_configuration_bias, nullptr);
    PlanPtr materialized_plan =
        BuildConfigPlan(materialized_index.get(), materialized_kernel->ordinal);
    EXPECT_TRUE(loom_link_plan_contains_symbol(materialized_plan.get(),
                                               materialized_kernel->ordinal));
    EXPECT_FALSE(loom_link_plan_contains_symbol(
        materialized_plan.get(), materialized_implementation_only->ordinal));
    EXPECT_TRUE(loom_link_plan_contains_symbol(
        materialized_plan.get(), materialized_configuration_only->ordinal));
    EXPECT_TRUE(loom_link_plan_contains_symbol(
        materialized_plan.get(), materialized_configuration_bias->ordinal));
    EXPECT_TRUE(loom_link_plan_contains_facet(
        materialized_plan.get(), materialized_kernel->ordinal,
        LOOM_LINK_SYMBOL_FACET_KERNEL_CONTRACT));
    EXPECT_TRUE(loom_link_plan_contains_facet(
        materialized_plan.get(), materialized_kernel->ordinal,
        LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION));
    EXPECT_FALSE(loom_link_plan_contains_facet(
        materialized_plan.get(), materialized_kernel->ordinal,
        LOOM_LINK_SYMBOL_FACET_KERNEL_IMPLEMENTATION));

    loom_link_plan_materialization_environment_t environment = {};
    environment.context = &context_;
    environment.block_pool = &block_pool_;
    environment.allocator = iree_allocator_system();
    iree_arena_allocator_t materialization_arena;
    iree_arena_initialize(&block_pool_, &materialization_arena);
    loom_link_plan_materialization_t materialization = {};
    IREE_ASSERT_OK(loom_link_plan_materialize(
        materialized_plan.get(), &environment, IREE_SV("materialized.config"),
        &materialization_arena, &materialization));
    std::unique_ptr<loom_module_t, decltype(&loom_module_free)>
        materialized_configuration(materialization.module, loom_module_free);
    Verify(materialized_configuration.get());
    EXPECT_EQ(materialized_configuration->symbols.count, 5u);
    EXPECT_EQ(FindSymbol(materialized_configuration.get(),
                         IREE_SV("implementation_only")),
              nullptr);
    const loom_symbol_ref_t projected_kernel =
        materialization.target_symbols.values[materialized_kernel->ordinal];
    const loom_symbol_ref_t projected_config =
        TargetConfiguration(materialization, materialized_kernel->ordinal);
    ASSERT_TRUE(loom_symbol_ref_is_valid(projected_kernel));
    ASSERT_TRUE(loom_symbol_ref_is_valid(projected_config));
    EXPECT_TRUE(loom_kernel_decl_isa(
        materialized_configuration->symbols.entries[projected_kernel.symbol_id]
            .defining_op));
    EXPECT_TRUE(loom_func_def_isa(
        materialized_configuration->symbols.entries[projected_config.symbol_id]
            .defining_op));
    iree_arena_deinitialize(&materialization_arena);
  }
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
  PlanPtr plan = BuildConfigPlan(index.get(), indexed_kernel->ordinal);
  const loom_link_module_index_symbol_t* implementation_only =
      loom_link_module_index_lookup_name(index.get(),
                                         IREE_SV("implementation_only"));
  const loom_link_module_index_symbol_t* configuration_only =
      loom_link_module_index_lookup_name(index.get(),
                                         IREE_SV("configuration_only"));
  const loom_link_module_index_symbol_t* configuration_bias =
      loom_link_module_index_lookup_name(index.get(),
                                         IREE_SV("configuration_bias"));
  ASSERT_NE(implementation_only, nullptr);
  ASSERT_NE(configuration_only, nullptr);
  ASSERT_NE(configuration_bias, nullptr);
  EXPECT_FALSE(
      loom_link_plan_contains_symbol(plan.get(), implementation_only->ordinal));
  EXPECT_TRUE(
      loom_link_plan_contains_symbol(plan.get(), configuration_only->ordinal));
  EXPECT_TRUE(
      loom_link_plan_contains_symbol(plan.get(), configuration_bias->ordinal));
  EXPECT_TRUE(
      loom_link_plan_contains_facet(plan.get(), indexed_kernel->ordinal,
                                    LOOM_LINK_SYMBOL_FACET_KERNEL_CONTRACT));
  EXPECT_TRUE(loom_link_plan_contains_facet(
      plan.get(), indexed_kernel->ordinal,
      LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION));
  EXPECT_FALSE(loom_link_plan_contains_facet(
      plan.get(), indexed_kernel->ordinal,
      LOOM_LINK_SYMBOL_FACET_KERNEL_IMPLEMENTATION));

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
  iree_arena_allocator_t config_materialization_arena;
  iree_arena_initialize(&block_pool_, &config_materialization_arena);
  loom_link_plan_materialization_t materialization = {};
  IREE_ASSERT_OK(loom_link_plan_materialize(
      plan.get(), &environment, IREE_SV("projected.config"),
      &config_materialization_arena, &materialization));
  std::unique_ptr<loom_module_t, decltype(&loom_module_free)> projected_module(
      materialization.module, loom_module_free);
  Verify(projected_module.get());

  ASSERT_EQ(projected_module->symbols.count, 5u);
  ASSERT_EQ(materialization.target_symbols.count,
            loom_link_module_index_symbol_count(index.get()));
  ASSERT_EQ(materialization.target_kernel_configurations.count,
            projected_module->symbols.count);
  const loom_symbol_ref_t projected_kernel_ref =
      materialization.target_symbols.values[indexed_kernel->ordinal];
  const loom_symbol_ref_t projected_helper_ref =
      TargetConfiguration(materialization, indexed_kernel->ordinal);
  ASSERT_TRUE(loom_symbol_ref_is_valid(projected_kernel_ref));
  ASSERT_TRUE(loom_symbol_ref_is_valid(projected_helper_ref));
  loom_op_t* kernel_op =
      projected_module->symbols.entries[projected_kernel_ref.symbol_id]
          .defining_op;
  ASSERT_TRUE(loom_kernel_decl_isa(kernel_op));
  loom_func_like_t kernel =
      loom_func_like_cast(projected_module.get(), kernel_op);
  EXPECT_EQ(loom_func_like_visibility(kernel), 0u);
  const loom_symbol_ref_t target_ref = loom_func_like_target(kernel);
  ASSERT_TRUE(loom_symbol_ref_is_valid(target_ref));
  ASSERT_EQ(target_ref.module_id, 0u);
  ASSERT_LT(target_ref.symbol_id, projected_module->symbols.count);
  EXPECT_TRUE(loom_target_generic_isa(
      projected_module->symbols.entries[target_ref.symbol_id].defining_op));
  const loom_symbol_t* projected_configuration_bias =
      FindSymbol(projected_module.get(), IREE_SV("configuration_bias"));
  ASSERT_NE(projected_configuration_bias, nullptr);
  EXPECT_TRUE(loom_config_def_isa(projected_configuration_bias->defining_op));
  const loom_symbol_t* projected_configuration =
      FindSymbol(projected_module.get(), IREE_SV("configuration_only"));
  ASSERT_NE(projected_configuration, nullptr);
  EXPECT_TRUE(loom_func_def_isa(projected_configuration->defining_op));
  EXPECT_FALSE(iree_any_bit_set(projected_configuration->flags,
                                LOOM_SYMBOL_FLAG_PUBLIC));
  const loom_func_like_t configuration_function = loom_func_like_cast(
      projected_module.get(), projected_configuration->defining_op);
  EXPECT_EQ(loom_func_like_visibility(configuration_function), 0u);
  loom_block_t* configuration_block =
      loom_region_entry_block(loom_func_like_body(configuration_function));
  bool configuration_reads_bias = false;
  const loom_op_t* configuration_op = nullptr;
  loom_block_for_each_op(configuration_block, configuration_op) {
    configuration_reads_bias |= loom_config_get_isa(configuration_op);
  }
  EXPECT_TRUE(configuration_reads_bias);
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
      projected_module->symbols.entries[projected_helper_ref.symbol_id]
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
  bool saw_configuration_call = false;
  bool saw_division = false;
  bool saw_target_query = false;
  const loom_op_t* op = nullptr;
  loom_block_for_each_op(helper_block, op) {
    saw_configuration_call |= loom_func_call_isa(op);
    saw_division |= loom_index_div_isa(op);
    saw_target_query |= loom_target_subgroup_size_isa(op);
  }
  EXPECT_TRUE(saw_configuration_call);
  EXPECT_TRUE(saw_division);
  EXPECT_TRUE(saw_target_query);
  ASSERT_TRUE(loom_func_return_isa(helper_block->last_op));
  EXPECT_EQ(helper_block->last_op->operand_count, 3u);

  const std::vector<uint8_t> projected_bytecode = Write(projected_module.get());
  loom_bytecode_read_options_t read_options = {};
  read_options.diagnostic_sink = {loom_diagnostic_stderr_sink, nullptr};
  loom_bytecode_read_result_t read_result = {};
  loom_module_t* roundtrip_module = nullptr;
  IREE_ASSERT_OK(loom_bytecode_read_module(
      iree_make_const_byte_span(projected_bytecode.data(),
                                projected_bytecode.size()),
      IREE_SV("projected.loombc"), &context_, &block_pool_, &read_options,
      &read_result, &roundtrip_module, iree_allocator_system()));
  ASSERT_EQ(read_result.error_count, 0u);
  ASSERT_NE(roundtrip_module, nullptr);
  EXPECT_EQ(roundtrip_module->symbols.count, 5u);
  loom_module_free(roundtrip_module);

  iree_arena_allocator_t materialization_arena;
  iree_arena_initialize(&block_pool_, &materialization_arena);
  PlanPtr full_plan = BuildPlan(index.get(), IREE_SV("dispatch_rows"));
  EXPECT_TRUE(loom_link_plan_contains_symbol(full_plan.get(),
                                             implementation_only->ordinal));
  loom_link_plan_materialization_t full_materialization = {};
  iree_status_t full_status =
      loom_link_plan_materialize(full_plan.get(), &environment, IREE_SV("full"),
                                 &materialization_arena, &full_materialization);
  IREE_EXPECT_NOT_OK(full_status);
  loom_module_free(full_materialization.module);
  iree_arena_deinitialize(&materialization_arena);
  projected_module.reset();
  iree_arena_deinitialize(&config_materialization_arena);
}

}  // namespace
}  // namespace loom
