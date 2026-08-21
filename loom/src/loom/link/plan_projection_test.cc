// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/plan_projection.h"

#include <memory>
#include <vector>

#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/module/ops.h"
#include "loom/ops/test/registry.h"

namespace loom {
namespace {

struct ModuleIndexDeleter {
  void operator()(loom_link_module_index_t* index) const {
    loom_link_module_index_free(index);
  }
};
using ModuleIndexPtr =
    std::unique_ptr<loom_link_module_index_t, ModuleIndexDeleter>;

struct LinkPlanDeleter {
  void operator()(loom_link_plan_t* plan) const { loom_link_plan_free(plan); }
};
using LinkPlanPtr = std::unique_ptr<loom_link_plan_t, LinkPlanDeleter>;

class LinkPlanProjectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_func_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_FUNC, vtables, (uint16_t)vtable_count));
    iree_host_size_t semantics_count = 0;
    const loom_op_semantics_t* semantics =
        loom_func_dialect_op_semantics(&semantics_count);
    IREE_ASSERT_OK(loom_context_register_dialect_semantics(
        &context_, LOOM_DIALECT_FUNC, semantics, (uint16_t)semantics_count));
    vtables = loom_module_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_MODULE, vtables, (uint16_t)vtable_count));
    semantics = loom_module_dialect_op_semantics(&semantics_count);
    IREE_ASSERT_OK(loom_context_register_dialect_semantics(
        &context_, LOOM_DIALECT_MODULE, semantics, (uint16_t)semantics_count));
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    for (loom_module_t* module : modules_) {
      loom_module_free(module);
    }
    iree_arena_deinitialize(&arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* Parse(iree_string_view_t source) {
    loom_module_t* module = nullptr;
    const loom_text_parse_options_t options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    IREE_EXPECT_OK(loom_text_parse(source, IREE_SV("projection_test.loom"),
                                   &context_, &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    if (module) {
      modules_.push_back(module);
    }
    return module;
  }

  ModuleIndexPtr CreateIndex() {
    loom_link_module_index_t* index = nullptr;
    IREE_CHECK_OK(loom_link_module_index_create(
        &context_, &block_pool_, iree_allocator_system(), &index));
    return ModuleIndexPtr(index);
  }

  void AddModule(loom_link_module_index_t* index, const loom_module_t* module,
                 iree_string_view_t provider_name,
                 loom_link_provider_role_t role) {
    const loom_link_module_index_add_options_t options = {
        /*.provider_name=*/provider_name,
        /*.role=*/role,
    };
    IREE_ASSERT_OK(loom_link_module_index_add_materialized(
        index, module, &options, /*out_provider_ordinal=*/nullptr));
  }

  std::vector<uint8_t> WriteModule(const loom_module_t* module) {
    iree_io_stream_t* stream = nullptr;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    IREE_CHECK_OK(loom_bytecode_write_module(module, stream,
                                             /*options=*/nullptr,
                                             &block_pool_));
    const iree_io_stream_pos_t length = iree_io_stream_length(stream);
    std::vector<uint8_t> bytes(length);
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }

  void AddBytecode(loom_link_module_index_t* index,
                   const std::vector<uint8_t>& bytes,
                   iree_string_view_t provider_name,
                   loom_link_provider_role_t role) {
    const loom_link_module_index_add_options_t options = {
        /*.provider_name=*/provider_name,
        /*.role=*/role,
    };
    IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
        index, iree_make_const_byte_span(bytes.data(), bytes.size()),
        provider_name, /*index_options=*/nullptr, &options,
        /*out_provider_ordinal=*/nullptr));
  }

  LinkPlanPtr BuildPlan(const loom_link_module_index_t* index,
                        const loom_link_plan_options_t* options) {
    loom_link_plan_t* plan = nullptr;
    IREE_CHECK_OK(
        loom_link_plan_build(index, options, iree_allocator_system(), &plan));
    return LinkPlanPtr(plan);
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t arena_;
  loom_context_t context_ = {};
  std::vector<loom_module_t*> modules_;
};

TEST_F(LinkPlanProjectionTest, GroupsExactOrdinalsBySourceModule) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.decl public @callee(%x: i32) -> (i32)

func.def public @entry(%x: i32) -> (i32) {
  %y = func.call @callee(%x) : (i32) -> (i32)
  func.return %y : i32
}
)"));
  loom_module_t* library = Parse(IREE_SV(R"(
func.def @callee(%x: i32) -> (i32) {
  %y = func.call @helper(%x) : (i32) -> (i32)
  func.return %y : i32
}

func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  ModuleIndexPtr index = CreateIndex();
  AddModule(index.get(), harness, IREE_SV("harness"),
            LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddModule(index.get(), library, IREE_SV("library"),
            LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  const iree_string_view_t roots[] = {IREE_SV("@entry")};
  const loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  LinkPlanPtr plan = BuildPlan(index.get(), &options);

  loom_link_plan_module_projection_t projection;
  IREE_ASSERT_OK(
      loom_link_plan_project_modules(plan.get(), &arena_, &projection));
  ASSERT_EQ(projection.modules.count, 2u);
  ASSERT_EQ(projection.symbols.count, 4u);

  for (iree_host_size_t module_index = 0;
       module_index < projection.modules.count; ++module_index) {
    const loom_link_plan_module_selection_t* module =
        &projection.modules.values[module_index];
    ASSERT_NE(module->source_module, nullptr);
    EXPECT_EQ(module->source_module->ordinal, module_index);
    ASSERT_EQ(module->symbols.count, 2u);
    EXPECT_EQ(module->symbols.values[0].source_symbol->module_symbol_ordinal,
              0u);
    EXPECT_EQ(module->symbols.values[1].source_symbol->module_symbol_ordinal,
              1u);
    for (iree_host_size_t symbol_index = 0;
         symbol_index < module->symbols.count; ++symbol_index) {
      const loom_link_plan_module_symbol_t* projected_symbol =
          &module->symbols.values[symbol_index];
      EXPECT_EQ(projected_symbol->source_symbol->module_ordinal,
                module->source_module->ordinal);
      const loom_link_plan_symbol_t* planned_symbol =
          projected_symbol->plan_symbol;
      ASSERT_NE(planned_symbol, nullptr);
      const loom_link_module_index_symbol_t* source_symbol =
          loom_link_module_index_symbol_at(index.get(),
                                           planned_symbol->symbol_ordinal);
      ASSERT_NE(source_symbol, nullptr);
      EXPECT_EQ(source_symbol, projected_symbol->source_symbol);
    }
  }
  EXPECT_EQ(projection.modules.values[0].symbols.values[0].plan_symbol->ordinal,
            1u);
  EXPECT_EQ(projection.modules.values[0].symbols.values[1].plan_symbol->ordinal,
            0u);
}

TEST_F(LinkPlanProjectionTest, EmptyArchiveHasNoProjectionStorage) {
  ModuleIndexPtr index = CreateIndex();
  LinkPlanPtr plan = BuildPlan(index.get(), /*options=*/nullptr);

  loom_link_plan_module_projection_t projection;
  IREE_ASSERT_OK(
      loom_link_plan_project_modules(plan.get(), &arena_, &projection));
  EXPECT_EQ(projection.modules.count, 0u);
  EXPECT_EQ(projection.modules.values, nullptr);
  EXPECT_EQ(projection.symbols.count, 0u);
  EXPECT_EQ(projection.symbols.values, nullptr);
  EXPECT_EQ(projection.provider_imports.count, 0u);
  EXPECT_EQ(projection.provider_imports.values, nullptr);
  EXPECT_EQ(projection.provider_import_anchors.count, 0u);
  EXPECT_EQ(projection.provider_import_anchors.values, nullptr);
}

TEST_F(LinkPlanProjectionTest, ArchiveProjectsSymbolEmptyModules) {
  loom_module_t* metadata = Parse(IREE_SV("test.module_metadata\n"));
  ModuleIndexPtr index = CreateIndex();
  AddModule(index.get(), metadata, IREE_SV("metadata"),
            LOOM_LINK_PROVIDER_ROLE_INPUT);
  LinkPlanPtr plan = BuildPlan(index.get(), /*options=*/nullptr);

  loom_link_plan_module_projection_t projection;
  IREE_ASSERT_OK(
      loom_link_plan_project_modules(plan.get(), &arena_, &projection));
  ASSERT_EQ(projection.modules.count, 1u);
  EXPECT_EQ(projection.modules.values[0].source_module,
            loom_link_module_index_module_at(index.get(), 0));
  EXPECT_EQ(projection.modules.values[0].symbols.count, 0u);
  EXPECT_EQ(projection.modules.values[0].symbols.values, nullptr);
  EXPECT_EQ(projection.symbols.count, 0u);
  EXPECT_EQ(projection.symbols.values, nullptr);
}

TEST_F(LinkPlanProjectionTest,
       RetainsOnlyLiveUnresolvedAndConcreteImportAnchors) {
  loom_module_t* harness = Parse(IREE_SV(R"(
// Alpha candidates.
module.import "alpha" [@concrete, @dead, @resolved, @unresolved]

// Beta candidates.
module.import "beta" [@dead, @resolved, @unresolved]

func.def @concrete(%x: i32) -> (i32) {
  func.return %x : i32
}

func.decl @dead(%x: i32) -> (i32)
func.decl @resolved(%x: i32) -> (i32)
func.decl @unresolved(%x: i32) -> (i32)

func.def public @entry(%x: i32) -> (i32) {
  %concrete = func.call @concrete(%x) : (i32) -> (i32)
  %resolved = func.call @resolved(%concrete) : (i32) -> (i32)
  %unresolved = func.call @unresolved(%resolved) : (i32) -> (i32)
  func.return %unresolved : i32
}
)"));
  loom_module_t* alpha = Parse(IREE_SV(R"(
func.def @resolved(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  ModuleIndexPtr index = CreateIndex();
  AddModule(index.get(), harness, IREE_SV("harness"),
            LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddModule(index.get(), alpha, IREE_SV("alpha-library"),
            LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  loom_link_provider_binding_t bindings[] = {
      {IREE_SV("alpha"), /*.provider_ordinal=*/1},
  };
  loom_link_provider_resolver_t resolver = {0};
  IREE_ASSERT_OK(loom_link_provider_resolver_prepare(
      loom_link_module_index_provider_count(index.get()), bindings,
      IREE_ARRAYSIZE(bindings), &resolver));
  const iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
      /*.include_exported_roots=*/false,
      /*.unresolved_policy=*/LOOM_LINK_PLAN_UNRESOLVED_ALLOW,
  };
  options.provider_resolver = &resolver;
  LinkPlanPtr plan = BuildPlan(index.get(), &options);

  loom_link_plan_module_projection_t projection;
  IREE_ASSERT_OK(
      loom_link_plan_project_modules(plan.get(), &arena_, &projection));
  ASSERT_EQ(projection.modules.count, 2u);
  ASSERT_EQ(projection.provider_imports.count, 2u);
  ASSERT_EQ(projection.provider_import_anchors.count, 3u);

  const loom_link_plan_module_selection_t* harness_selection =
      &projection.modules.values[0];
  ASSERT_EQ(harness_selection->source_module->ordinal, 0u);
  ASSERT_EQ(harness_selection->provider_imports.count, 2u);
  ASSERT_EQ(harness_selection->provider_import_anchors.count, 3u);

  const auto anchor_name = [&](uint32_t source_symbol_ordinal) {
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(
            index.get(),
            harness_selection->source_module->symbol_start_ordinal +
                source_symbol_ordinal);
    return std::string(symbol->name.data, symbol->name.size);
  };

  const loom_link_plan_module_provider_import_t* alpha_import =
      &harness_selection->provider_imports.values[0];
  EXPECT_EQ(alpha_import->source_import_ordinal, 0u);
  ASSERT_EQ(alpha_import->anchors.count, 2u);
  EXPECT_EQ(anchor_name(harness_selection->provider_import_anchors
                            .values[alpha_import->anchors.first]),
            "concrete");
  EXPECT_EQ(anchor_name(harness_selection->provider_import_anchors
                            .values[alpha_import->anchors.first + 1]),
            "unresolved");
  const loom_link_module_index_provider_import_t alpha_source =
      loom_link_module_index_provider_import_at(
          index.get(), harness_selection->source_module,
          alpha_import->source_import_ordinal);
  ASSERT_EQ(alpha_source.comments.count, 1u);
  EXPECT_EQ(std::string(alpha_source.comments.values[0].data,
                        alpha_source.comments.values[0].size),
            "Alpha candidates.");
  EXPECT_FALSE(alpha_source.leading_blank_line);

  const loom_link_plan_module_provider_import_t* beta_import =
      &harness_selection->provider_imports.values[1];
  EXPECT_EQ(beta_import->source_import_ordinal, 1u);
  ASSERT_EQ(beta_import->anchors.count, 1u);
  EXPECT_EQ(anchor_name(harness_selection->provider_import_anchors
                            .values[beta_import->anchors.first]),
            "unresolved");
  const loom_link_module_index_provider_import_t beta_source =
      loom_link_module_index_provider_import_at(
          index.get(), harness_selection->source_module,
          beta_import->source_import_ordinal);
  ASSERT_EQ(beta_source.comments.count, 1u);
  EXPECT_EQ(std::string(beta_source.comments.values[0].data,
                        beta_source.comments.values[0].size),
            "Beta candidates.");
  EXPECT_TRUE(beta_source.leading_blank_line);

  EXPECT_EQ(projection.modules.values[1].provider_imports.count, 0u);
  EXPECT_EQ(projection.modules.values[1].provider_imports.values, nullptr);
  EXPECT_EQ(projection.modules.values[1].provider_import_anchors.count, 0u);
  EXPECT_EQ(projection.modules.values[1].provider_import_anchors.values,
            nullptr);

  loom_link_plan_linker_import_projection_t linker_imports;
  IREE_ASSERT_OK(loom_link_plan_project_linker_imports(
      index.get(), &projection, &arena_, &linker_imports));
  ASSERT_EQ(linker_imports.modules.count, 2u);
  ASSERT_EQ(linker_imports.provider_imports.count, 2u);
  ASSERT_EQ(linker_imports.provider_import_anchors.count, 3u);
  const loom_linker_source_provider_import_list_t harness_imports =
      linker_imports.modules.values[0];
  ASSERT_EQ(harness_imports.count, 2u);
  EXPECT_EQ(std::string(harness_imports.values[0].provider.data,
                        harness_imports.values[0].provider.size),
            "alpha");
  ASSERT_EQ(harness_imports.values[0].anchors.count, 2u);
  EXPECT_EQ(harness_imports.values[0].anchors.ordinals[0],
            harness_selection->provider_import_anchors
                .values[alpha_import->anchors.first]);
  EXPECT_EQ(harness_imports.values[0].anchors.ordinals[1],
            harness_selection->provider_import_anchors
                .values[alpha_import->anchors.first + 1]);
  EXPECT_EQ(harness_imports.values[0].comments.count, 1u);
  EXPECT_FALSE(harness_imports.values[0].leading_blank_line);
  EXPECT_EQ(std::string(harness_imports.values[1].provider.data,
                        harness_imports.values[1].provider.size),
            "beta");
  EXPECT_TRUE(harness_imports.values[1].leading_blank_line);
  EXPECT_EQ(linker_imports.modules.values[1].count, 0u);
  EXPECT_EQ(linker_imports.modules.values[1].values, nullptr);
}

TEST_F(LinkPlanProjectionTest, MapsBytecodeImportAnchorsToCompactOrdinals) {
  loom_module_t* source = Parse(IREE_SV(R"(
func.decl @first(%x: i32) -> (i32)

func.def @dead(%x: i32) -> (i32) {
  func.return %x : i32
}

func.decl @third(%x: i32) -> (i32)

// Retained candidates.
module.import "missing" [@first, @third]

func.def public @entry(%x: i32) -> (i32) {
  %first = func.call @first(%x) : (i32) -> (i32)
  %third = func.call @third(%first) : (i32) -> (i32)
  func.return %third : i32
}
)"));
  const std::vector<uint8_t> bytes = WriteModule(source);
  ModuleIndexPtr index = CreateIndex();
  AddBytecode(index.get(), bytes, IREE_SV("source.loombc"),
              LOOM_LINK_PROVIDER_ROLE_INPUT);

  const iree_string_view_t roots[] = {IREE_SV("@entry")};
  const loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
      /*.include_exported_roots=*/false,
      /*.unresolved_policy=*/LOOM_LINK_PLAN_UNRESOLVED_ALLOW,
  };
  LinkPlanPtr plan = BuildPlan(index.get(), &options);
  loom_link_plan_module_projection_t module_projection;
  IREE_ASSERT_OK(
      loom_link_plan_project_modules(plan.get(), &arena_, &module_projection));
  ASSERT_EQ(module_projection.modules.count, 1u);
  const loom_link_plan_module_selection_t* module =
      &module_projection.modules.values[0];
  ASSERT_EQ(module->symbols.count, 3u);
  EXPECT_EQ(module->symbols.values[0].source_symbol->module_symbol_ordinal, 0u);
  EXPECT_EQ(module->symbols.values[1].source_symbol->module_symbol_ordinal, 2u);
  EXPECT_EQ(module->symbols.values[2].source_symbol->module_symbol_ordinal, 3u);

  loom_link_plan_linker_import_projection_t linker_projection;
  IREE_ASSERT_OK(loom_link_plan_project_linker_imports(
      index.get(), &module_projection, &arena_, &linker_projection));
  ASSERT_EQ(linker_projection.modules.count, 1u);
  ASSERT_EQ(linker_projection.modules.values[0].count, 1u);
  const loom_linker_source_provider_import_t* provider_import =
      &linker_projection.modules.values[0].values[0];
  EXPECT_EQ(std::string(provider_import->provider.data,
                        provider_import->provider.size),
            "missing");
  ASSERT_EQ(provider_import->anchors.count, 2u);
  EXPECT_EQ(provider_import->anchors.ordinals[0], 0u);
  EXPECT_EQ(provider_import->anchors.ordinals[1], 1u);
  ASSERT_EQ(provider_import->comments.count, 1u);
  EXPECT_EQ(std::string(provider_import->comments.values[0].data,
                        provider_import->comments.values[0].size),
            "Retained candidates.");
  EXPECT_TRUE(provider_import->leading_blank_line);
}

}  // namespace
}  // namespace loom
