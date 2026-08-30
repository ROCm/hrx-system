// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/kernel_class_materializer.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/analysis/template_provider_catalog.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/template/ops.h"
#include "loom/testing/module_ptr.h"
#include "loom/tooling/context/context.h"
#include "loom/transforms/symbol/template_selection.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class KernelClassMaterializerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_tooling_context_register_tool_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseAndVerify(iree_string_view_t source,
                           iree_arena_allocator_t* analysis_arena,
                           loom_symbol_reference_table_t* out_references) {
    loom_text_parse_options_t parse_options = {};
    parse_options.diagnostic_sink.fn = loom_diagnostic_stderr_sink;
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse_with_symbol_references(
        source, IREE_SV("kernel_class_materializer_test.loom"), &context_,
        &block_pool_, &parse_options, analysis_arena, out_references, &module));
    if (module == nullptr) return ModulePtr();
    ModulePtr module_ptr(module);
    Verify(module);
    return module_ptr;
  }

  void Verify(const loom_module_t* module) {
    loom_verify_options_t verify_options = {};
    verify_options.sink.fn = loom_diagnostic_stderr_sink;
    loom_verify_result_t verify_result = {};
    IREE_EXPECT_OK(loom_verify_module(module, &verify_options, &verify_result));
    EXPECT_EQ(verify_result.error_count, 0u);
  }

  loom_symbol_id_t FindSymbol(const loom_module_t* module,
                              iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    EXPECT_NE(name_id, LOOM_STRING_ID_INVALID);
    if (name_id == LOOM_STRING_ID_INVALID) return LOOM_SYMBOL_ID_INVALID;
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    EXPECT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    return symbol_id;
  }

  void ExpectFinalSelection(loom_module_t* module,
                            iree_arena_allocator_t* query_arena) {
    loom_symbol_fact_table_t symbol_facts = {};
    loom_symbol_fact_table_initialize(&symbol_facts, query_arena);
    loom_template_provider_catalog_t providers = {};
    loom_template_provider_catalog_initialize(&providers, query_arena);
    IREE_ASSERT_OK(loom_template_provider_catalog_build_local(
        &providers, module, &symbol_facts));
    const loom_template_selection_query_options_t query_options = {
        /*.mode=*/LOOM_TEMPLATE_SELECTION_MODE_FINAL,
        /*.catalog=*/&providers,
    };
    loom_template_selection_query_result_t query_result = {};
    IREE_ASSERT_OK(loom_template_selection_query(
        module, &query_options, &block_pool_, query_arena, &query_result));
    EXPECT_EQ(query_result.unresolved_site_count, 0u);
  }

  ModulePtr RoundTripBytecode(const loom_module_t* module) {
    iree_io_stream_t* stream = nullptr;
    IREE_EXPECT_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    if (stream == nullptr) return ModulePtr();
    IREE_EXPECT_OK(loom_bytecode_write_module(
        module, stream, /*options=*/nullptr, &block_pool_));
    const iree_io_stream_pos_t bytecode_length = iree_io_stream_length(stream);
    std::vector<uint8_t> bytecode(bytecode_length);
    IREE_EXPECT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_EXPECT_OK(
        iree_io_stream_read(stream, bytecode.size(), bytecode.data(), nullptr));
    iree_io_stream_release(stream);

    const loom_bytecode_read_options_t read_options = {
        /*.diagnostic_sink=*/
        {
            /*.fn=*/loom_diagnostic_stderr_sink,
        },
    };
    loom_bytecode_read_result_t read_result = {};
    loom_module_t* round_tripped_module = nullptr;
    IREE_EXPECT_OK(loom_bytecode_read_module(
        iree_make_const_byte_span(bytecode.data(), bytecode.size()),
        IREE_SV("kernel_class.loombc"), &context_, &block_pool_, &read_options,
        &read_result, &round_tripped_module, iree_allocator_system()));
    EXPECT_EQ(read_result.error_count, 0u);
    Verify(round_tripped_module);
    return ModulePtr(round_tripped_module);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(KernelClassMaterializerTest,
       PublishesIndependentAffineClassesAsOrdinaryKernelIR) {
  constexpr iree_host_size_t kSiteCount = 1000;
  const iree_string_view_t source = IREE_SV(R"(
template.decl @boundary.family(%effective_n: index, %duplicate: index) -> (index)

template.def<@boundary.family> priority(10) @large(%effective_n: index, %duplicate: index) -> (index) where [ge(%effective_n, 128)] {
  %two = index.constant 2 : index
  template.return %two : index
}

template.def<@boundary.family> priority(1) @small(%effective_n: index, %duplicate: index) -> (index) {
  %one = index.constant 1 : index
  template.return %one : index
}

template.decl @sign.family(%value: i32)

template.def<@sign.family> priority(10) @nonnegative(%value: i32) where [ge(%value, 0)] {
  template.return
}

template.def<@sign.family> priority(1) @negative(%value: i32) {
  template.return
}

template.decl @observe.family(%value: index)

template.def<@observe.family> @observe(%value: index) {
  template.return
}

kernel.def @classified() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%n: index, %padding: index, %sign: i32, %irrelevant: index) {
  %effective_n = index.add %n, %padding : index
  %selected = template.apply<@boundary.family>(%effective_n, %effective_n) : (index, index) -> (index)
  template.apply<@sign.family>(%sign) : (i32) -> ()
  template.apply<@observe.family>(%irrelevant) : (index) -> ()
  %condition = scalar.constant true : i1
  scf.if %condition {
    template.apply<@observe.family>(%irrelevant) : (index) -> ()
    scf.yield
  }
  kernel.return
}
)");

  iree_arena_allocator_t analysis_arena;
  iree_arena_initialize(&block_pool_, &analysis_arena);
  loom_symbol_reference_table_t references = {};
  ModulePtr source_module =
      ParseAndVerify(source, &analysis_arena, &references);
  ASSERT_NE(source_module, nullptr);

  loom_symbol_fact_table_t symbol_facts = {};
  loom_symbol_fact_table_initialize(&symbol_facts, &analysis_arena);
  loom_template_provider_catalog_t providers = {};
  loom_template_provider_catalog_initialize(&providers, &analysis_arena);
  IREE_ASSERT_OK(loom_template_provider_catalog_build_local(
      &providers, source_module.get(), &symbol_facts));
  loom_template_decision_model_catalog_t decision_models = {};
  IREE_ASSERT_OK(loom_template_decision_model_catalog_build(
      source_module.get(), &symbol_facts, &references, &providers,
      &analysis_arena, &decision_models));

  const loom_symbol_id_t kernel_symbol_id =
      FindSymbol(source_module.get(), IREE_SV("classified"));
  ASSERT_NE(kernel_symbol_id, LOOM_SYMBOL_ID_INVALID);
  const loom_func_like_t kernel = loom_func_like_cast(
      source_module.get(),
      source_module->symbols.entries[kernel_symbol_id].defining_op);
  loom_value_fact_table_t kernel_facts = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(
      &kernel_facts, &analysis_arena, source_module->values.count));
  IREE_ASSERT_OK(loom_value_fact_table_compute(&kernel_facts,
                                               source_module.get(), kernel));
  loom_symbolic_expr_context_t expression_context = {};
  loom_symbolic_expr_context_initialize(source_module.get(), &kernel_facts,
                                        &analysis_arena, &expression_context);
  const loom_template_applicability_target_t kernel_target = {};
  loom_kernel_class_classifier_t classifier = {};
  IREE_ASSERT_OK(loom_kernel_class_classifier_build(
      source_module.get(), kernel_symbol_id, &references, &decision_models,
      &kernel_facts, &expression_context, &kernel_target, &analysis_arena,
      &classifier));
  ASSERT_EQ(classifier.decision_count, 4u);
  ASSERT_EQ(classifier.projection_count, 2u);
  EXPECT_EQ(classifier.projections[0].kind,
            LOOM_KERNEL_CLASS_PROJECTION_AFFINE);
  EXPECT_EQ(classifier.projections[1].kind,
            LOOM_KERNEL_CLASS_PROJECTION_AFFINE);

  std::vector<loom_value_id_t> site_argument_values(kSiteCount * 4);
  std::vector<loom_kernel_class_site_t> sites(kSiteCount);
  loom_value_fact_table_t site_facts = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&site_facts, &analysis_arena,
                                                  kSiteCount * 4));
  site_facts.count = kSiteCount * 4;
  for (iree_host_size_t i = 0; i < kSiteCount; ++i) {
    const iree_host_size_t value_base = i * 4;
    site_argument_values[value_base + 0] =
        static_cast<loom_value_id_t>(value_base + 0);
    site_argument_values[value_base + 1] =
        static_cast<loom_value_id_t>(value_base + 1);
    site_argument_values[value_base + 2] =
        static_cast<loom_value_id_t>(value_base + 2);
    site_argument_values[value_base + 3] =
        static_cast<loom_value_id_t>(value_base + 3);
    site_facts.entries[value_base + 0] =
        loom_value_facts_exact_i64(static_cast<int64_t>(i));
    site_facts.entries[value_base + 1] =
        loom_value_facts_exact_i64(static_cast<int64_t>((i * 7) % 5));
    const int64_t effective_n =
        static_cast<int64_t>(i + static_cast<iree_host_size_t>((i * 7) % 5));
    site_facts.entries[value_base + 2] =
        loom_value_facts_exact_i64(effective_n >= 128 ? 1 : -1);
    site_facts.entries[value_base + 3] =
        loom_value_facts_exact_i64(static_cast<int64_t>(1000003 * i + 17));
    sites[i] = {
        /*.facts=*/&site_facts,
        /*.argument_values=*/&site_argument_values[value_base],
    };
  }

  const loom_kernel_class_collection_options_t collection_options =
      loom_kernel_class_collection_options_default();
  loom_kernel_class_collection_t collection = {};
  IREE_ASSERT_OK(loom_kernel_class_classifier_collect(
      &classifier, sites.data(), sites.size(), &collection_options,
      &analysis_arena, &collection));
  ASSERT_EQ(collection.class_count, 2u);
  ASSERT_EQ(collection.accepted_decision_count, 3u);
  ASSERT_EQ(collection.skipped_decision_count, 1u);

  std::array<ModulePtr, 2> class_modules;
  std::array<loom_symbol_ref_t, 2> class_kernels = {};
  std::array<bool, 2> class_is_large = {};
  std::array<iree_host_size_t, 2> class_member_counts = {};
  for (loom_decision_class_ordinal_t class_ordinal = 0;
       class_ordinal < collection.class_count; ++class_ordinal) {
    loom_kernel_class_trace_id_t trace_id =
        collection.classes[class_ordinal].trace_id;
    while (collection.traces[trace_id].decision_ordinal != 0) {
      trace_id = collection.traces[trace_id].parent_trace_id;
      ASSERT_NE(trace_id, LOOM_KERNEL_CLASS_TRACE_ID_INVALID);
    }
    const uint32_t action_ordinal = collection.traces[trace_id].action_ordinal;
    const loom_template_provider_summary_t* provider =
        loom_template_decision_model_provider(classifier.decisions[0].model,
                                              action_ordinal);
    class_is_large[class_ordinal] =
        iree_string_view_equal(provider->name, IREE_SV("large"));
    class_member_counts[class_ordinal] =
        collection.classes[class_ordinal].member_count;

    loom_kernel_class_product_t product = {};
    IREE_ASSERT_OK(loom_kernel_class_materialize(
        &classifier, &collection, class_ordinal, &block_pool_,
        iree_allocator_system(), &product));
    class_kernels[class_ordinal] = product.kernel;
    class_modules[class_ordinal] = ModulePtr(product.module);
    product.module = nullptr;
    loom_kernel_class_product_deinitialize(&product);

    ASSERT_EQ(class_kernels[class_ordinal].module_id, 0u);
    ASSERT_LT(class_kernels[class_ordinal].symbol_id,
              class_modules[class_ordinal]->symbols.count);
    EXPECT_TRUE(loom_kernel_def_isa(
        class_modules[class_ordinal]
            ->symbols.entries[class_kernels[class_ordinal].symbol_id]
            .defining_op));
  }

  // Published modules own only ordinary IR and survive every transient input.
  iree_arena_deinitialize(&analysis_arena);
  source_module.reset();
  site_argument_values.clear();
  sites.clear();

  iree_arena_allocator_t query_arena;
  iree_arena_initialize(&block_pool_, &query_arena);
  for (ModulePtr& class_module : class_modules) {
    ModulePtr round_tripped = RoundTripBytecode(class_module.get());
    ASSERT_NE(round_tripped, nullptr);
    class_module = std::move(round_tripped);
  }
  iree_host_size_t large_member_count = 0;
  iree_host_size_t small_member_count = 0;
  for (loom_decision_class_ordinal_t class_ordinal = 0;
       class_ordinal < class_modules.size(); ++class_ordinal) {
    loom_module_t* class_module = class_modules[class_ordinal].get();
    ASSERT_NE(class_module, nullptr);
    Verify(class_module);
    ExpectFinalSelection(class_module, &query_arena);

    const loom_symbol_id_t class_kernel_symbol_id =
        FindSymbol(class_module, IREE_SV("classified"));
    const loom_func_like_t class_kernel = loom_func_like_cast(
        class_module,
        class_module->symbols.entries[class_kernel_symbol_id].defining_op);
    loom_block_t* body =
        loom_region_entry_block(loom_func_like_body(class_kernel));
    iree_host_size_t assume_count = 0;
    iree_host_size_t scalar_assume_count = 0;
    iree_host_size_t call_count = 0;
    iree_host_size_t zero_result_call_count = 0;
    iree_host_size_t residual_apply_count = 0;
    loom_op_t* op = nullptr;
    loom_block_for_each_op(body, op) {
      assume_count += loom_index_assume_isa(op);
      scalar_assume_count += loom_scalar_assume_isa(op);
      if (loom_scf_if_isa(op)) {
        loom_block_t* then_block =
            loom_region_entry_block(loom_scf_if_then_region(op));
        EXPECT_TRUE(loom_template_apply_isa(then_block->first_op));
        residual_apply_count += loom_template_apply_isa(then_block->first_op);
      }
      if (!loom_template_call_isa(op)) {
        EXPECT_FALSE(loom_template_apply_isa(op));
        continue;
      }
      ++call_count;
      zero_result_call_count += op->result_count == 0;
      const loom_symbol_ref_t callee = loom_template_call_callee(op);
      const iree_string_view_t callee_name =
          class_module->strings
              .entries[class_module->symbols.entries[callee.symbol_id].name_id];
      if (iree_string_view_equal(callee_name, IREE_SV("large")) ||
          iree_string_view_equal(callee_name, IREE_SV("small"))) {
        const loom_value_slice_t operands = loom_template_call_operands(op);
        ASSERT_EQ(operands.count, 2u);
        EXPECT_EQ(operands.values[0], operands.values[1]);
        EXPECT_EQ(iree_string_view_equal(callee_name, IREE_SV("large")),
                  class_is_large[class_ordinal]);
      }
    }
    EXPECT_EQ(call_count, 3u);
    EXPECT_EQ(zero_result_call_count, 2u);
    EXPECT_EQ(residual_apply_count, 1u);
    EXPECT_EQ(assume_count, class_is_large[class_ordinal] ? 1u : 0u);
    EXPECT_EQ(scalar_assume_count, class_is_large[class_ordinal] ? 1u : 0u);
    if (class_is_large[class_ordinal]) {
      large_member_count = class_member_counts[class_ordinal];
    } else {
      small_member_count = class_member_counts[class_ordinal];
    }
  }
  EXPECT_EQ(large_member_count, 874u);
  EXPECT_EQ(small_member_count, 126u);
  iree_arena_deinitialize(&query_arena);
}

TEST_F(KernelClassMaterializerTest,
       PublishesExternalProviderSelectionAsGenericRequest) {
  const iree_string_view_t source = IREE_SV(R"(
template.decl @external.family(%n: index)

template.def<@external.family> priority(1) @fallback(%n: index) {
  template.return
}

kernel.def @classified() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch(%n: index) {
  template.apply<@external.family>(%n) : (index) -> ()
  kernel.return
}
)");

  iree_arena_allocator_t analysis_arena;
  iree_arena_initialize(&block_pool_, &analysis_arena);
  loom_symbol_reference_table_t references = {};
  ModulePtr source_module =
      ParseAndVerify(source, &analysis_arena, &references);
  ASSERT_NE(source_module, nullptr);

  const loom_symbol_id_t family_symbol_id =
      FindSymbol(source_module.get(), IREE_SV("external.family"));
  ASSERT_NE(family_symbol_id, LOOM_SYMBOL_ID_INVALID);
  const loom_predicate_t external_predicate = {
      /*.kind=*/LOOM_PREDICATE_GE,
      /*.arg_count=*/2,
      /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
      /*.reserved=*/{},
      /*.args=*/{0, 128},
  };
  const loom_template_provider_contract_t external_contract = {
      /*.kind=*/LOOM_TEMPLATE_PROVIDER_KIND_DEF,
      /*.has_body=*/true,
      /*.argument_count=*/1,
      /*.result_count=*/0,
      /*.predicate_count=*/1,
      /*.target_condition_count=*/0,
      /*.name=*/IREE_SV("external"),
      /*.priority=*/10,
      /*.predicates=*/&external_predicate,
  };
  const loom_symbol_ref_t family = {
      /*.module_id=*/0,
      /*.symbol_id=*/family_symbol_id,
  };
  loom_template_provider_summary_t external_provider = {};
  IREE_ASSERT_OK(loom_template_provider_contract_bind_family(
      &external_contract, source_module.get(), family, loom_symbol_ref_null(),
      /*origin_ordinal=*/7, &analysis_arena, &external_provider));

  loom_symbol_fact_table_t symbol_facts = {};
  loom_symbol_fact_table_initialize(&symbol_facts, &analysis_arena);
  loom_template_provider_catalog_t providers = {};
  loom_template_provider_catalog_initialize(&providers, &analysis_arena);
  IREE_ASSERT_OK(loom_template_provider_catalog_build(
      &providers, source_module.get(), &symbol_facts, &external_provider,
      /*external_provider_count=*/1));
  loom_template_decision_model_catalog_t decision_models = {};
  IREE_ASSERT_OK(loom_template_decision_model_catalog_build(
      source_module.get(), &symbol_facts, &references, &providers,
      &analysis_arena, &decision_models));

  const loom_symbol_id_t kernel_symbol_id =
      FindSymbol(source_module.get(), IREE_SV("classified"));
  ASSERT_NE(kernel_symbol_id, LOOM_SYMBOL_ID_INVALID);
  const loom_func_like_t kernel = loom_func_like_cast(
      source_module.get(),
      source_module->symbols.entries[kernel_symbol_id].defining_op);
  loom_value_fact_table_t kernel_facts = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(
      &kernel_facts, &analysis_arena, source_module->values.count));
  IREE_ASSERT_OK(loom_value_fact_table_compute(&kernel_facts,
                                               source_module.get(), kernel));
  loom_symbolic_expr_context_t expression_context = {};
  loom_symbolic_expr_context_initialize(source_module.get(), &kernel_facts,
                                        &analysis_arena, &expression_context);
  const loom_template_applicability_target_t kernel_target = {};
  loom_kernel_class_classifier_t classifier = {};
  IREE_ASSERT_OK(loom_kernel_class_classifier_build(
      source_module.get(), kernel_symbol_id, &references, &decision_models,
      &kernel_facts, &expression_context, &kernel_target, &analysis_arena,
      &classifier));
  ASSERT_EQ(classifier.decision_count, 1u);

  loom_value_fact_table_t site_facts = {};
  IREE_ASSERT_OK(loom_value_fact_table_initialize(&site_facts, &analysis_arena,
                                                  /*capacity=*/1));
  site_facts.count = 1;
  site_facts.entries[0] = loom_value_facts_exact_i64(256);
  const loom_value_id_t site_argument_value = 0;
  const loom_kernel_class_site_t site = {
      /*.facts=*/&site_facts,
      /*.argument_values=*/&site_argument_value,
  };
  const loom_kernel_class_collection_options_t collection_options =
      loom_kernel_class_collection_options_default();
  loom_kernel_class_collection_t collection = {};
  IREE_ASSERT_OK(loom_kernel_class_classifier_collect(
      &classifier, &site, /*site_count=*/1, &collection_options,
      &analysis_arena, &collection));
  ASSERT_EQ(collection.class_count, 1u);
  ASSERT_EQ(collection.accepted_decision_count, 1u);
  const loom_kernel_class_trace_t* trace =
      &collection.traces[collection.classes[0].trace_id];
  const loom_template_provider_summary_t* selected_provider =
      loom_template_decision_model_provider(classifier.decisions[0].model,
                                            trace->action_ordinal);
  ASSERT_FALSE(loom_symbol_ref_is_valid(selected_provider->symbol));

  loom_kernel_class_product_t product = {};
  IREE_ASSERT_OK(loom_kernel_class_materialize(
      &classifier, &collection, /*class_ordinal=*/0, &block_pool_,
      iree_allocator_system(), &product));
  loom_module_t* class_module = product.module;
  ModulePtr class_module_ptr(class_module);
  product.module = nullptr;
  const loom_symbol_ref_t class_kernel_ref = product.kernel;
  loom_kernel_class_product_deinitialize(&product);

  // The published class owns only ordinary IR. The selected external body is
  // represented by an assumption feeding the still-generic request.
  ASSERT_EQ(class_kernel_ref.module_id, 0u);
  const loom_symbol_id_t class_kernel_symbol_id = class_kernel_ref.symbol_id;
  const loom_func_like_t class_kernel = loom_func_like_cast(
      class_module,
      class_module->symbols.entries[class_kernel_symbol_id].defining_op);
  loom_block_t* body =
      loom_region_entry_block(loom_func_like_body(class_kernel));
  loom_op_t* assume_op = nullptr;
  loom_op_t* apply_op = nullptr;
  loom_op_t* op = nullptr;
  loom_block_for_each_op(body, op) {
    if (loom_index_assume_isa(op)) assume_op = op;
    if (loom_template_apply_isa(op)) apply_op = op;
    EXPECT_FALSE(loom_template_call_isa(op));
  }
  ASSERT_NE(assume_op, nullptr);
  ASSERT_NE(apply_op, nullptr);
  const loom_value_slice_t apply_operands =
      loom_template_apply_operands(apply_op);
  ASSERT_EQ(apply_operands.count, 1u);
  const loom_value_t* refined_operand =
      loom_module_value(class_module, apply_operands.values[0]);
  ASSERT_FALSE(loom_value_is_block_arg(refined_operand));
  EXPECT_EQ(loom_value_def_op(refined_operand), assume_op);

  // The ordinary request survives serialization independently of every
  // classifier input and selects the same external provider when linked into
  // a provider universe again.
  iree_arena_deinitialize(&analysis_arena);
  source_module.reset();
  ModulePtr round_tripped = RoundTripBytecode(class_module);
  ASSERT_NE(round_tripped, nullptr);
  class_module_ptr = std::move(round_tripped);
  class_module = class_module_ptr.get();
  Verify(class_module);

  iree_arena_allocator_t query_arena;
  iree_arena_initialize(&block_pool_, &query_arena);
  const loom_symbol_ref_t class_family = {
      /*.module_id=*/0,
      /*.symbol_id=*/FindSymbol(class_module, IREE_SV("external.family")),
  };
  loom_template_provider_summary_t rebound_external_provider = {};
  IREE_ASSERT_OK(loom_template_provider_contract_bind_family(
      &external_contract, class_module, class_family, loom_symbol_ref_null(),
      /*origin_ordinal=*/7, &query_arena, &rebound_external_provider));
  loom_symbol_fact_table_t class_symbol_facts = {};
  loom_symbol_fact_table_initialize(&class_symbol_facts, &query_arena);
  loom_template_provider_catalog_t class_providers = {};
  loom_template_provider_catalog_initialize(&class_providers, &query_arena);
  IREE_ASSERT_OK(loom_template_provider_catalog_build(
      &class_providers, class_module, &class_symbol_facts,
      &rebound_external_provider, /*external_provider_count=*/1));
  const loom_template_selection_query_options_t query_options = {
      /*.mode=*/LOOM_TEMPLATE_SELECTION_MODE_FINAL,
      /*.catalog=*/&class_providers,
      /*.function_versions=*/nullptr,
      /*.origin_count=*/8,
  };
  loom_template_selection_query_result_t query_result = {};
  IREE_ASSERT_OK(loom_template_selection_query(
      class_module, &query_options, &block_pool_, &query_arena, &query_result));
  ASSERT_EQ(query_result.required_origins.count, 1u);
  EXPECT_EQ(query_result.required_origins.values[0], 7u);
  EXPECT_EQ(query_result.unresolved_site_count, 0u);
  iree_arena_deinitialize(&query_arena);
}

}  // namespace
}  // namespace loom
