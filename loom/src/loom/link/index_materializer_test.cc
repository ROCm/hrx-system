// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/index_materializer.h"

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/module_index.h"
#include "loom/link/provider_resolver.h"
#include "loom/ops/module/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/op_registry.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

struct IndexDeleter {
  void operator()(loom_link_module_index_t* index) const {
    loom_link_module_index_free(index);
  }
};
using IndexPtr = std::unique_ptr<loom_link_module_index_t, IndexDeleter>;

enum class ProviderForm {
  kMaterialized,
  kText,
  kBytecode,
};

class LinkIndexMaterializerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    for (loom_module_t* module : modules_) {
      loom_module_free(module);
    }
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* Parse(iree_string_view_t source, iree_string_view_t filename) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    options.diagnostic_sink.fn = loom_diagnostic_stderr_sink;
    options.max_errors = 20;
    IREE_CHECK_OK(loom_text_parse(source, filename, &context_, &block_pool_,
                                  &options, &module));
    modules_.push_back(module);
    return module;
  }

  std::vector<uint8_t> WriteModule(const loom_module_t* module) {
    iree_io_stream_t* stream = nullptr;
    IREE_CHECK_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    IREE_CHECK_OK(loom_bytecode_write_module(
        module, stream, /*options=*/nullptr, &block_pool_));
    const iree_io_stream_pos_t length = iree_io_stream_length(stream);
    std::vector<uint8_t> bytes(length);
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }

  IndexPtr CreateIndex() {
    loom_link_module_index_t* index = nullptr;
    IREE_CHECK_OK(loom_link_module_index_create(
        &context_, &block_pool_, iree_allocator_system(), &index));
    return IndexPtr(index);
  }

  iree_host_size_t AddMaterialized(loom_link_module_index_t* index,
                                   const loom_module_t* module,
                                   iree_string_view_t name,
                                   loom_link_provider_role_t role) {
    const loom_link_module_index_add_options_t options = {
        /*.provider_name=*/name,
        /*.role=*/role,
    };
    iree_host_size_t provider_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    IREE_CHECK_OK(loom_link_module_index_add_materialized(
        index, module, &options, &provider_ordinal));
    return provider_ordinal;
  }

  iree_host_size_t AddBytecode(loom_link_module_index_t* index,
                               const std::vector<uint8_t>& bytecode,
                               iree_string_view_t name,
                               loom_link_provider_role_t role) {
    const loom_link_module_index_add_options_t options = {
        /*.provider_name=*/name,
        /*.role=*/role,
    };
    iree_host_size_t provider_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    IREE_CHECK_OK(loom_link_module_index_add_bytecode(
        index, iree_make_const_byte_span(bytecode.data(), bytecode.size()),
        name, /*index_options=*/nullptr, &options, &provider_ordinal));
    return provider_ordinal;
  }

  iree_host_size_t AddText(loom_link_module_index_t* index,
                           iree_string_view_t source, iree_string_view_t name,
                           loom_link_provider_role_t role) {
    const loom_link_module_index_add_options_t options = {
        /*.provider_name=*/name,
        /*.role=*/role,
    };
    loom_text_parse_options_t parse_options = {};
    parse_options.diagnostic_sink.fn = loom_diagnostic_stderr_sink;
    parse_options.max_errors = 20;
    iree_host_size_t provider_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    IREE_CHECK_OK(loom_link_module_index_add_text(
        index, source, name, &parse_options, &options, &provider_ordinal));
    return provider_ordinal;
  }

  iree_host_size_t AddProvider(loom_link_module_index_t* index,
                               ProviderForm form, iree_string_view_t source,
                               const loom_module_t* module,
                               const std::vector<uint8_t>& bytecode,
                               iree_string_view_t name,
                               loom_link_provider_role_t role) {
    switch (form) {
      case ProviderForm::kMaterialized:
        return AddMaterialized(index, module, name, role);
      case ProviderForm::kText:
        return AddText(index, source, name, role);
      case ProviderForm::kBytecode:
        return AddBytecode(index, bytecode, name, role);
    }
    IREE_ASSERT_UNREACHABLE("unknown provider form");
    IREE_BUILTIN_UNREACHABLE();
  }

  iree_status_t TryMaterializeWithOptions(
      const loom_link_module_index_t* index,
      const loom_link_plan_options_t* plan_options,
      loom_link_index_materialization_t* out_materialization) {
    loom_link_plan_materialization_environment_t environment = {};
    environment.context = &context_;
    environment.block_pool = &block_pool_;
    environment.allocator = iree_allocator_system();
    return loom_link_index_materialize(index, plan_options, &environment,
                                       IREE_SV("linked"), out_materialization);
  }

  iree_status_t TryMaterialize(
      const loom_link_module_index_t* index, iree_string_view_t root,
      loom_link_plan_mode_t mode,
      loom_link_plan_unresolved_policy_t unresolved_policy,
      loom_link_index_materialization_t* out_materialization) {
    const iree_string_view_t roots[] = {root};
    loom_link_plan_options_t plan_options = {};
    plan_options.mode = mode;
    plan_options.root_symbols = {
        /*.count=*/IREE_ARRAYSIZE(roots),
        /*.values=*/roots,
    };
    plan_options.unresolved_policy = unresolved_policy;
    return TryMaterializeWithOptions(index, &plan_options, out_materialization);
  }

  loom_link_index_materialization_t MaterializeWithPolicy(
      const loom_link_module_index_t* index, iree_string_view_t root,
      loom_link_plan_unresolved_policy_t unresolved_policy) {
    loom_link_index_materialization_t materialization = {};
    IREE_CHECK_OK(TryMaterialize(index, root, LOOM_LINK_PLAN_SELECTIVE,
                                 unresolved_policy, &materialization));
    return materialization;
  }

  loom_link_index_materialization_t Materialize(
      const loom_link_module_index_t* index, iree_string_view_t root) {
    return MaterializeWithPolicy(index, root, LOOM_LINK_PLAN_UNRESOLVED_ERROR);
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

  const loom_link_module_index_symbol_t* FindIndexedProviderSymbol(
      const loom_link_module_index_t* index, iree_host_size_t provider_ordinal,
      iree_string_view_t name) {
    const loom_link_module_index_provider_t* provider =
        loom_link_module_index_provider_at(index, provider_ordinal);
    if (!provider) return nullptr;
    for (iree_host_size_t module_offset = 0;
         module_offset < provider->module_count; ++module_offset) {
      const loom_link_module_index_module_t* module =
          loom_link_module_index_module_at(
              index, provider->module_start_ordinal + module_offset);
      for (iree_host_size_t symbol_offset = 0;
           symbol_offset < module->symbol_count; ++symbol_offset) {
        const loom_link_module_index_symbol_t* symbol =
            loom_link_module_index_symbol_at(
                index, module->symbol_start_ordinal + symbol_offset);
        if (iree_string_view_equal(symbol->name, name)) return symbol;
      }
    }
    return nullptr;
  }

  std::string OptionalString(const loom_module_t* module,
                             loom_string_id_t string_id) {
    if (string_id == LOOM_STRING_ID_INVALID) return {};
    const iree_string_view_t value = module->strings.entries[string_id];
    return std::string(value.data, value.size);
  }

  using LinkedSymbolShape = std::tuple<std::string, loom_symbol_flags_t, bool,
                                       std::string, std::string, std::string>;

  std::vector<LinkedSymbolShape> CaptureSymbolShapes(
      const loom_module_t* module) {
    std::vector<LinkedSymbolShape> shapes;
    shapes.reserve(module->symbols.count);
    for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
      const loom_symbol_t* symbol = &module->symbols.entries[i];
      const iree_string_view_t name = module->strings.entries[symbol->name_id];
      std::string import_module;
      std::string import_symbol;
      std::string export_symbol;
      const loom_func_like_t function =
          loom_func_like_cast(module, symbol->defining_op);
      if (loom_func_like_isa(function)) {
        import_module =
            OptionalString(module, loom_func_like_import_module(function));
        import_symbol =
            OptionalString(module, loom_func_like_import_symbol(function));
        export_symbol =
            OptionalString(module, loom_func_like_export_symbol(function));
      }
      shapes.emplace_back(
          std::string(name.data, name.size), symbol->flags,
          loom_symbol_definition_is_declaration(symbol->definition),
          std::move(import_module), std::move(import_symbol),
          std::move(export_symbol));
    }
    std::sort(shapes.begin(), shapes.end());
    return shapes;
  }

  iree_host_size_t CountProviderImports(const loom_module_t* module) {
    iree_host_size_t count = 0;
    const loom_op_t* op = nullptr;
    loom_block_for_each_op(loom_region_const_entry_block(module->body), op) {
      if (loom_module_import_isa(op)) ++count;
    }
    return count;
  }

  void Verify(const loom_module_t* module) {
    loom_verify_options_t options = {};
    options.sink.fn = loom_diagnostic_stderr_sink;
    options.max_errors = 100;
    loom_verify_result_t result = {};
    IREE_ASSERT_OK(loom_verify_module(module, &options, &result));
    EXPECT_EQ(result.error_count, 0u);
  }

  iree_arena_block_pool_t block_pool_ = {};
  loom_context_t context_ = {};
  std::vector<loom_module_t*> modules_;
};

TEST_F(LinkIndexMaterializerTest,
       SelectsBytecodeProviderWithoutReadingRejectedBody) {
  loom_module_t* root = Parse(IREE_SV(R"(
template.decl @demo.choose(%x: i32) -> (i32) where [mul(%x, 16)]

func.def public @entry(%x: i32) -> (i32) where [range(%x, 32, 32)] {
  %result = template.apply<@demo.choose>(%x) : (i32) -> (i32)
  func.return %result : i32
}
)"),
                              IREE_SV("root.loom"));
  loom_module_t* library = Parse(IREE_SV(R"(
template.decl @demo.choose(%x: i32) -> (i32)

template.def<@demo.choose> priority(10) @fast(%x: i32) -> (i32) {
  template.return %x : i32
}

template.def<@demo.choose> priority(1) @slow(%x: i32) -> (i32) {
  template.return %x : i32
}
)"),
                                 IREE_SV("library.loom"));
  std::vector<uint8_t> library_bytecode = WriteModule(library);

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), root, IREE_SV("root"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  const iree_host_size_t library_provider =
      AddBytecode(index.get(), library_bytecode, IREE_SV("library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  const loom_link_module_index_provider_t* indexed_library =
      loom_link_module_index_provider_at(index.get(), library_provider);
  ASSERT_NE(indexed_library, nullptr);
  const loom_bytecode_module_metadata_t* metadata =
      &indexed_library->bytecode.metadata.modules[0];
  const loom_bytecode_symbol_metadata_t* slow_metadata = nullptr;
  for (iree_host_size_t i = 0; i < metadata->symbol_count; ++i) {
    if (iree_string_view_equal(metadata->symbols[i].name, IREE_SV("slow"))) {
      slow_metadata = &metadata->symbols[i];
      break;
    }
  }
  ASSERT_NE(slow_metadata, nullptr);
  ASSERT_GT(slow_metadata->region_payload_count, 0u);
  for (uint8_t i = 0; i < slow_metadata->region_payload_count; ++i) {
    const loom_bytecode_region_payload_metadata_t* payload =
        &metadata
             ->region_payloads[slow_metadata->first_region_payload_index + i];
    std::fill_n(library_bytecode.data() + payload->absolute_offset,
                payload->length, UINT8_C(0xFF));
  }

  loom_link_index_materialization_t materialization =
      Materialize(index.get(), IREE_SV("@entry"));
  ASSERT_NE(materialization.plan, nullptr);
  ASSERT_NE(materialization.module, nullptr);
  Verify(materialization.module);
  EXPECT_NE(FindSymbol(materialization.module, IREE_SV("fast")), nullptr);
  EXPECT_EQ(FindSymbol(materialization.module, IREE_SV("slow")), nullptr);

  const loom_link_module_index_symbol_t* fast =
      loom_link_module_index_lookup_private(
          index.get(), loom_link_module_index_module_at(index.get(), 1),
          IREE_SV("fast"));
  const loom_link_module_index_symbol_t* slow =
      loom_link_module_index_lookup_private(
          index.get(), loom_link_module_index_module_at(index.get(), 1),
          IREE_SV("slow"));
  ASSERT_NE(fast, nullptr);
  ASSERT_NE(slow, nullptr);
  EXPECT_TRUE(
      loom_link_plan_contains_symbol(materialization.plan, fast->ordinal));
  EXPECT_FALSE(
      loom_link_plan_contains_symbol(materialization.plan, slow->ordinal));
  loom_link_index_materialization_deinitialize(&materialization);
}

TEST_F(LinkIndexMaterializerTest,
       RejectedFamilyDoesNotReadBytecodeProviderBody) {
  loom_module_t* root = Parse(IREE_SV(R"(
template.decl @demo.choose(%x: i32) -> (i32) where [mul(%x, 16)]

func.def public @entry(%x: i32) -> (i32) where [range(%x, 15, 15)] {
  %result = template.apply<@demo.choose>(%x) : (i32) -> (i32)
  func.return %result : i32
}
)"),
                              IREE_SV("root.loom"));
  loom_module_t* library = Parse(IREE_SV(R"(
template.decl @demo.choose(%x: i32) -> (i32) where [mul(%x, 16)]

template.def<@demo.choose> @implementation(%x: i32) -> (i32) {
  template.return %x : i32
}
)"),
                                 IREE_SV("library.loom"));
  std::vector<uint8_t> library_bytecode = WriteModule(library);

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), root, IREE_SV("root"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  const iree_host_size_t library_provider =
      AddBytecode(index.get(), library_bytecode, IREE_SV("library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  const loom_link_module_index_provider_t* indexed_library =
      loom_link_module_index_provider_at(index.get(), library_provider);
  ASSERT_NE(indexed_library, nullptr);
  const loom_bytecode_module_metadata_t* metadata =
      &indexed_library->bytecode.metadata.modules[0];
  const loom_bytecode_symbol_metadata_t* implementation_metadata = nullptr;
  for (iree_host_size_t i = 0; i < metadata->symbol_count; ++i) {
    if (iree_string_view_equal(metadata->symbols[i].name,
                               IREE_SV("implementation"))) {
      implementation_metadata = &metadata->symbols[i];
      break;
    }
  }
  ASSERT_NE(implementation_metadata, nullptr);
  ASSERT_GT(implementation_metadata->region_payload_count, 0u);
  for (uint8_t i = 0; i < implementation_metadata->region_payload_count; ++i) {
    const loom_bytecode_region_payload_metadata_t* payload =
        &metadata->region_payloads
             [implementation_metadata->first_region_payload_index + i];
    std::fill_n(library_bytecode.data() + payload->absolute_offset,
                payload->length, UINT8_C(0xFF));
  }

  loom_link_index_materialization_t materialization = MaterializeWithPolicy(
      index.get(), IREE_SV("@entry"), LOOM_LINK_PLAN_UNRESOLVED_ALLOW);
  ASSERT_NE(materialization.plan, nullptr);
  ASSERT_NE(materialization.module, nullptr);
  Verify(materialization.module);
  EXPECT_EQ(FindSymbol(materialization.module, IREE_SV("implementation")),
            nullptr);

  const loom_link_module_index_symbol_t* implementation =
      loom_link_module_index_lookup_private(
          index.get(), loom_link_module_index_module_at(index.get(), 1),
          IREE_SV("implementation"));
  ASSERT_NE(implementation, nullptr);
  EXPECT_FALSE(loom_link_plan_contains_symbol(materialization.plan,
                                              implementation->ordinal));
  loom_link_index_materialization_deinitialize(&materialization);
}

TEST_F(LinkIndexMaterializerTest, ClosedLinkRejectsUnresolvedApplication) {
  loom_module_t* root = Parse(IREE_SV(R"(
template.decl @demo.missing(%x: i32) -> (i32)

func.def public @entry(%x: i32) -> (i32) {
  %result = template.apply<@demo.missing>(%x) : (i32) -> (i32)
  func.return %result : i32
}
)"),
                              IREE_SV("root.loom"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), root, IREE_SV("root"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);

  loom_link_index_materialization_t materialization = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      TryMaterialize(index.get(), IREE_SV("@entry"), LOOM_LINK_PLAN_SELECTIVE,
                     LOOM_LINK_PLAN_UNRESOLVED_ERROR, &materialization));
  EXPECT_EQ(materialization.plan, nullptr);
  EXPECT_EQ(materialization.module, nullptr);
}

TEST_F(LinkIndexMaterializerTest, ArchiveRetainsEveryProvider) {
  loom_module_t* root = Parse(IREE_SV(R"(
template.decl @demo.choose(%x: i32) -> (i32)

func.def public @entry(%x: i32) -> (i32) {
  %result = template.apply<@demo.choose>(%x) : (i32) -> (i32)
  func.return %result : i32
}
)"),
                              IREE_SV("root.loom"));
  loom_module_t* library = Parse(IREE_SV(R"(
template.decl @demo.choose(%x: i32) -> (i32)

template.def<@demo.choose> priority(10) @fast(%x: i32) -> (i32) {
  template.return %x : i32
}

template.def<@demo.choose> priority(1) @slow(%x: i32) -> (i32) {
  template.return %x : i32
}
)"),
                                 IREE_SV("library.loom"));
  const std::vector<uint8_t> library_bytecode = WriteModule(library);

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), root, IREE_SV("root"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddBytecode(index.get(), library_bytecode, IREE_SV("library"),
              LOOM_LINK_PROVIDER_ROLE_LIBRARY);

  loom_link_index_materialization_t materialization = {};
  IREE_ASSERT_OK(
      TryMaterialize(index.get(), IREE_SV("@entry"), LOOM_LINK_PLAN_ARCHIVE,
                     LOOM_LINK_PLAN_UNRESOLVED_ERROR, &materialization));
  Verify(materialization.module);
  EXPECT_NE(FindSymbol(materialization.module, IREE_SV("fast")), nullptr);
  EXPECT_NE(FindSymbol(materialization.module, IREE_SV("slow")), nullptr);
  loom_link_index_materialization_deinitialize(&materialization);
}

TEST_F(LinkIndexMaterializerTest,
       SelectsTransitiveDiamondOnceAcrossBytecodeLibraries) {
  loom_module_t* root = Parse(IREE_SV(R"(
template.decl @demo.left(%x: i32) -> (i32)
template.decl @demo.right(%x: i32) -> (i32)

func.def public @entry(%x: i32) -> (i32) {
  %left = template.apply<@demo.left>(%x) : (i32) -> (i32)
  %right = template.apply<@demo.right>(%left) : (i32) -> (i32)
  func.return %right : i32
}
)"),
                              IREE_SV("root.loom"));
  loom_module_t* frontier = Parse(IREE_SV(R"(
template.decl @demo.left(%x: i32) -> (i32)
template.decl @demo.right(%x: i32) -> (i32)
template.decl @demo.shared(%x: i32) -> (i32)

template.def<@demo.left> @left_impl(%x: i32) -> (i32) {
  %result = template.apply<@demo.shared>(%x) : (i32) -> (i32)
  template.return %result : i32
}

template.def<@demo.right> @right_impl(%x: i32) -> (i32) {
  %result = template.apply<@demo.shared>(%x) : (i32) -> (i32)
  template.return %result : i32
}
)"),
                                  IREE_SV("frontier.loom"));
  loom_module_t* shared = Parse(IREE_SV(R"(
template.decl @demo.shared(%x: i32) -> (i32)
template.decl @demo.leaf(%x: i32) -> (i32)

template.def<@demo.shared> @shared_impl(%x: i32) -> (i32) {
  %result = template.apply<@demo.leaf>(%x) : (i32) -> (i32)
  template.return %result : i32
}
)"),
                                IREE_SV("shared.loom"));
  loom_module_t* leaf = Parse(IREE_SV(R"(
template.decl @demo.leaf(%x: i32) -> (i32)

template.def<@demo.leaf> @leaf_impl(%x: i32) -> (i32) {
  template.return %x : i32
}
)"),
                              IREE_SV("leaf.loom"));
  std::vector<uint8_t> frontier_bytecode = WriteModule(frontier);
  std::vector<uint8_t> shared_bytecode = WriteModule(shared);
  std::vector<uint8_t> leaf_bytecode = WriteModule(leaf);

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), root, IREE_SV("root"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddBytecode(index.get(), frontier_bytecode, IREE_SV("frontier"),
              LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  AddBytecode(index.get(), shared_bytecode, IREE_SV("shared"),
              LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  AddBytecode(index.get(), leaf_bytecode, IREE_SV("leaf"),
              LOOM_LINK_PROVIDER_ROLE_LIBRARY);

  loom_link_index_materialization_t materialization =
      Materialize(index.get(), IREE_SV("@entry"));
  Verify(materialization.module);
  EXPECT_NE(FindSymbol(materialization.module, IREE_SV("left_impl")), nullptr);
  EXPECT_NE(FindSymbol(materialization.module, IREE_SV("right_impl")), nullptr);
  EXPECT_NE(FindSymbol(materialization.module, IREE_SV("shared_impl")),
            nullptr);
  EXPECT_NE(FindSymbol(materialization.module, IREE_SV("leaf_impl")), nullptr);

  const struct {
    iree_host_size_t module_ordinal;
    const char* name;
  } provider_sources[] = {
      {1, "left_impl"},
      {1, "right_impl"},
      {2, "shared_impl"},
      {3, "leaf_impl"},
  };
  for (const auto& provider_source : provider_sources) {
    const loom_link_module_index_module_t* module =
        loom_link_module_index_module_at(index.get(),
                                         provider_source.module_ordinal);
    ASSERT_NE(module, nullptr);
    const loom_link_module_index_symbol_t* provider =
        loom_link_module_index_lookup_private(
            index.get(), module, iree_make_cstring_view(provider_source.name));
    ASSERT_NE(provider, nullptr);
    EXPECT_TRUE(loom_link_plan_contains_symbol(materialization.plan,
                                               provider->ordinal));
  }
  loom_link_index_materialization_deinitialize(&materialization);
}

TEST_F(LinkIndexMaterializerTest,
       SelectiveLinkInternalizesLibraryDependencies) {
  loom_module_t* requester = Parse(IREE_SV(R"(
func.decl @helper(%x: i32) -> (i32)

func.def public export("request_entry") @entry(%x: i32) -> (i32) {
  %result = func.call @helper(%x) : (i32) -> (i32)
  func.return %result : i32
}
)"),
                                   IREE_SV("requester.loom"));
  loom_module_t* library = Parse(IREE_SV(R"(
func.def public export("library_helper") @helper(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def public export("unrelated") @unused(%x: i32) -> (i32) {
  func.return %x : i32
}
)"),
                                 IREE_SV("library.loom"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), requester, IREE_SV("requester"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), library, IREE_SV("library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);

  loom_link_index_materialization_t materialization =
      Materialize(index.get(), IREE_SV("@entry"));
  Verify(materialization.module);

  const loom_symbol_t* entry =
      FindSymbol(materialization.module, IREE_SV("entry"));
  const loom_symbol_t* helper =
      FindSymbol(materialization.module, IREE_SV("helper"));
  ASSERT_NE(entry, nullptr);
  ASSERT_NE(helper, nullptr);
  EXPECT_EQ(FindSymbol(materialization.module, IREE_SV("unused")), nullptr);
  EXPECT_TRUE(iree_all_bits_set(
      entry->flags, LOOM_SYMBOL_FLAG_PUBLIC | LOOM_SYMBOL_FLAG_RETAIN));
  EXPECT_FALSE(iree_any_bit_set(
      helper->flags, LOOM_SYMBOL_FLAG_PUBLIC | LOOM_SYMBOL_FLAG_RETAIN));

  loom_func_like_t entry_func =
      loom_func_like_cast(materialization.module, entry->defining_op);
  loom_func_like_t helper_func =
      loom_func_like_cast(materialization.module, helper->defining_op);
  ASSERT_TRUE(loom_func_like_isa(entry_func));
  ASSERT_TRUE(loom_func_like_isa(helper_func));
  const loom_string_id_t entry_export =
      loom_func_like_export_symbol(entry_func);
  ASSERT_NE(entry_export, LOOM_STRING_ID_INVALID);
  EXPECT_TRUE(iree_string_view_equal(
      materialization.module->strings.entries[entry_export],
      IREE_SV("request_entry")));
  EXPECT_EQ(loom_func_like_export_symbol(helper_func), LOOM_STRING_ID_INVALID);

  loom_link_index_materialization_deinitialize(&materialization);
}

TEST_F(LinkIndexMaterializerTest,
       RequesterDeclarationControlsResolvedRootSurface) {
  loom_module_t* requester = Parse(IREE_SV(R"(
func.decl public import("upstream", "identity") export("request_identity") @identity(%x: i32) -> (i32)
)"),
                                   IREE_SV("requester.loom"));
  loom_module_t* library = Parse(IREE_SV(R"(
func.def public export("provider_identity") @identity(%x: i32) -> (i32) {
  func.return %x : i32
}
)"),
                                 IREE_SV("library.loom"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), library, IREE_SV("library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  AddMaterialized(index.get(), requester, IREE_SV("requester"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);

  loom_link_index_materialization_t materialization =
      Materialize(index.get(), IREE_SV("@identity"));
  Verify(materialization.module);

  const loom_link_module_index_module_t* library_module =
      loom_link_module_index_module_at(index.get(), 0);
  ASSERT_NE(library_module, nullptr);
  const loom_link_module_index_symbol_t* library_identity =
      loom_link_module_index_symbol_at(index.get(),
                                       library_module->symbol_start_ordinal);
  ASSERT_NE(library_identity, nullptr);
  EXPECT_TRUE(loom_link_plan_contains_symbol(materialization.plan,
                                             library_identity->ordinal));

  const loom_symbol_t* identity =
      FindSymbol(materialization.module, IREE_SV("identity"));
  ASSERT_NE(identity, nullptr);
  EXPECT_TRUE(iree_all_bits_set(
      identity->flags, LOOM_SYMBOL_FLAG_PUBLIC | LOOM_SYMBOL_FLAG_RETAIN));
  EXPECT_FALSE(loom_symbol_definition_is_declaration(identity->definition));
  loom_func_like_t identity_func =
      loom_func_like_cast(materialization.module, identity->defining_op);
  ASSERT_TRUE(loom_func_like_isa(identity_func));
  EXPECT_EQ(loom_func_like_import_module(identity_func),
            LOOM_STRING_ID_INVALID);
  EXPECT_EQ(loom_func_like_import_symbol(identity_func),
            LOOM_STRING_ID_INVALID);
  const loom_string_id_t export_symbol =
      loom_func_like_export_symbol(identity_func);
  ASSERT_NE(export_symbol, LOOM_STRING_ID_INVALID);
  EXPECT_TRUE(iree_string_view_equal(
      materialization.module->strings.entries[export_symbol],
      IREE_SV("request_identity")));

  loom_link_index_materialization_deinitialize(&materialization);
}

TEST_F(LinkIndexMaterializerTest, UnresolvedRootPreservesImportSurface) {
  loom_module_t* requester = Parse(IREE_SV(R"(
func.decl public import("upstream", "identity") export("request_identity") @identity(%x: i32) -> (i32)
)"),
                                   IREE_SV("requester.loom"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), requester, IREE_SV("requester"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);

  loom_link_index_materialization_t materialization = MaterializeWithPolicy(
      index.get(), IREE_SV("@identity"), LOOM_LINK_PLAN_UNRESOLVED_ALLOW);
  Verify(materialization.module);

  const loom_symbol_t* identity =
      FindSymbol(materialization.module, IREE_SV("identity"));
  ASSERT_NE(identity, nullptr);
  EXPECT_TRUE(iree_all_bits_set(
      identity->flags, LOOM_SYMBOL_FLAG_PUBLIC | LOOM_SYMBOL_FLAG_RETAIN));
  EXPECT_TRUE(loom_symbol_definition_is_declaration(identity->definition));
  loom_func_like_t identity_func =
      loom_func_like_cast(materialization.module, identity->defining_op);
  ASSERT_TRUE(loom_func_like_isa(identity_func));
  EXPECT_NE(loom_func_like_import_module(identity_func),
            LOOM_STRING_ID_INVALID);
  EXPECT_NE(loom_func_like_import_symbol(identity_func),
            LOOM_STRING_ID_INVALID);

  loom_link_index_materialization_deinitialize(&materialization);
}

TEST_F(LinkIndexMaterializerTest, ExplicitPrivateRootRemainsPrivate) {
  loom_module_t* requester = Parse(IREE_SV(R"(
func.def @entry(%x: i32) -> (i32) {
  func.return %x : i32
}
)"),
                                   IREE_SV("requester.loom"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), requester, IREE_SV("requester"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);

  loom_link_index_materialization_t materialization =
      Materialize(index.get(), IREE_SV("@entry"));
  Verify(materialization.module);

  const loom_symbol_t* entry =
      FindSymbol(materialization.module, IREE_SV("entry"));
  ASSERT_NE(entry, nullptr);
  EXPECT_TRUE(iree_any_bit_set(entry->flags, LOOM_SYMBOL_FLAG_RETAIN));
  EXPECT_FALSE(iree_any_bit_set(entry->flags, LOOM_SYMBOL_FLAG_PUBLIC));

  loom_link_index_materialization_deinitialize(&materialization);
}

TEST_F(LinkIndexMaterializerTest, ArchivePreservesLibraryOutputSurface) {
  loom_module_t* requester = Parse(IREE_SV(R"(
func.def public export("request_entry") @entry(%x: i32) -> (i32) {
  func.return %x : i32
}
)"),
                                   IREE_SV("requester.loom"));
  loom_module_t* library = Parse(IREE_SV(R"(
func.def public export("library_helper") @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)"),
                                 IREE_SV("library.loom"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), requester, IREE_SV("requester"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), library, IREE_SV("library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);

  loom_link_index_materialization_t materialization = {};
  IREE_ASSERT_OK(
      TryMaterialize(index.get(), IREE_SV("@entry"), LOOM_LINK_PLAN_ARCHIVE,
                     LOOM_LINK_PLAN_UNRESOLVED_ERROR, &materialization));
  Verify(materialization.module);

  const loom_symbol_t* helper =
      FindSymbol(materialization.module, IREE_SV("helper"));
  ASSERT_NE(helper, nullptr);
  EXPECT_TRUE(iree_any_bit_set(helper->flags, LOOM_SYMBOL_FLAG_PUBLIC));
  loom_func_like_t helper_func =
      loom_func_like_cast(materialization.module, helper->defining_op);
  ASSERT_TRUE(loom_func_like_isa(helper_func));
  const loom_string_id_t export_symbol =
      loom_func_like_export_symbol(helper_func);
  ASSERT_NE(export_symbol, LOOM_STRING_ID_INVALID);
  EXPECT_TRUE(iree_string_view_equal(
      materialization.module->strings.entries[export_symbol],
      IREE_SV("library_helper")));

  loom_link_index_materialization_deinitialize(&materialization);
}

TEST_F(LinkIndexMaterializerTest,
       SelectiveLinkInternalizesGenericSymbolDependencies) {
  loom_module_t* requester = Parse(IREE_SV(R"(
check.case public @case {
  check.return
}

check.benchmark<@case> @benchmark
)"),
                                   IREE_SV("requester.loom"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), requester, IREE_SV("requester"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);

  loom_link_index_materialization_t materialization =
      Materialize(index.get(), IREE_SV("@benchmark"));
  Verify(materialization.module);

  const loom_symbol_t* check_case =
      FindSymbol(materialization.module, IREE_SV("case"));
  ASSERT_NE(check_case, nullptr);
  EXPECT_FALSE(iree_any_bit_set(check_case->flags, LOOM_SYMBOL_FLAG_PUBLIC));

  loom_link_index_materialization_deinitialize(&materialization);
}

TEST_F(LinkIndexMaterializerTest,
       SealedProductsAreEquivalentAcrossProviderRepresentations) {
  const iree_string_view_t requester_source = IREE_SV(R"(
module.import "chosen" [@api, @provided]

func.decl public export("request_api") @api(%x: i32) -> (i32)
func.decl @provided(%x: i32) -> (i32)
func.decl @partial(%x: i32) -> (i32)

func.def @local(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def @private_root(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def public export("request_entry") @entry(%x: i32) -> (i32) {
  %local = func.call @local(%x) : (i32) -> (i32)
  %provided = func.call @provided(%local) : (i32) -> (i32)
  %partial = func.call @partial(%provided) : (i32) -> (i32)
  %api = func.call @api(%partial) : (i32) -> (i32)
  func.return %api : i32
}
)");
  const iree_string_view_t chosen_source = IREE_SV(R"(
func.def public export("provider_api") @api(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def public export("provider_selected") @provided(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def public export("provider_unused") @unused(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  const iree_string_view_t wrong_source = IREE_SV(R"(
func.def public export("wrong_api") @api(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def public export("wrong_selected") @provided(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def public export("wrong_only") @wrong_only(%x: i32) -> (i32) {
  func.return %x : i32
}
)");
  const iree_string_view_t partial_source = IREE_SV(R"(
module.import "external-runtime" [@external]

func.decl @external(%x: i32) -> (i32)

func.def public export("partial_api") @partial(%x: i32) -> (i32) {
  %result = func.call @external(%x) : (i32) -> (i32)
  func.return %result : i32
}

func.def public export("partial_unused") @partial_unused(%x: i32) -> (i32) {
  func.return %x : i32
}
)");

  loom_module_t* requester = Parse(requester_source, IREE_SV("requester.loom"));
  loom_module_t* chosen = Parse(chosen_source, IREE_SV("chosen.loom"));
  loom_module_t* wrong = Parse(wrong_source, IREE_SV("wrong.loom"));
  loom_module_t* partial = Parse(partial_source, IREE_SV("partial.loom"));
  const std::vector<uint8_t> requester_bytecode = WriteModule(requester);
  const std::vector<uint8_t> chosen_bytecode = WriteModule(chosen);
  const std::vector<uint8_t> wrong_bytecode = WriteModule(wrong);

  IndexPtr partial_index = CreateIndex();
  AddMaterialized(partial_index.get(), partial, IREE_SV("partial-source"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  loom_link_index_materialization_t partial_materialization =
      MaterializeWithPolicy(partial_index.get(), IREE_SV("@partial"),
                            LOOM_LINK_PLAN_UNRESOLVED_ALLOW);
  Verify(partial_materialization.module);
  ASSERT_EQ(loom_link_plan_symbol_count(partial_materialization.plan), 2u);
  ASSERT_EQ(partial_materialization.module->symbols.count, 2u);
  ASSERT_EQ(CountProviderImports(partial_materialization.module), 1u);
  const std::vector<uint8_t> partial_bytecode =
      WriteModule(partial_materialization.module);
  loom_link_index_materialization_deinitialize(&partial_materialization);

  enum ProviderId {
    kRequester,
    kChosen,
    kWrong,
    kPartial,
  };
  struct Variant {
    const char* name;
    ProviderForm requester_form;
    ProviderForm chosen_form;
    ProviderForm wrong_form;
    std::array<ProviderId, 4> insertion_order;
  };
  const Variant variants[] = {
      {
          "materialized-requester",
          ProviderForm::kMaterialized,
          ProviderForm::kText,
          ProviderForm::kBytecode,
          {kRequester, kWrong, kChosen, kPartial},
      },
      {
          "text-requester",
          ProviderForm::kText,
          ProviderForm::kBytecode,
          ProviderForm::kMaterialized,
          {kChosen, kPartial, kRequester, kWrong},
      },
      {
          "bytecode-requester",
          ProviderForm::kBytecode,
          ProviderForm::kMaterialized,
          ProviderForm::kText,
          {kPartial, kWrong, kChosen, kRequester},
      },
  };

  std::vector<LinkedSymbolShape> reference_symbols;
  for (const Variant& variant : variants) {
    SCOPED_TRACE(variant.name);
    IndexPtr index = CreateIndex();
    iree_host_size_t requester_provider =
        LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    iree_host_size_t chosen_provider = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    iree_host_size_t wrong_provider = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    iree_host_size_t partial_provider = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    for (ProviderId provider : variant.insertion_order) {
      switch (provider) {
        case kRequester:
          requester_provider =
              AddProvider(index.get(), variant.requester_form, requester_source,
                          requester, requester_bytecode, IREE_SV("requester"),
                          LOOM_LINK_PROVIDER_ROLE_INPUT);
          break;
        case kChosen:
          chosen_provider =
              AddProvider(index.get(), variant.chosen_form, chosen_source,
                          chosen, chosen_bytecode, IREE_SV("chosen"),
                          LOOM_LINK_PROVIDER_ROLE_LIBRARY);
          break;
        case kWrong:
          wrong_provider =
              AddProvider(index.get(), variant.wrong_form, wrong_source, wrong,
                          wrong_bytecode, IREE_SV("wrong"),
                          LOOM_LINK_PROVIDER_ROLE_LIBRARY);
          break;
        case kPartial:
          partial_provider = AddBytecode(index.get(), partial_bytecode,
                                         IREE_SV("partial.loombc"),
                                         LOOM_LINK_PROVIDER_ROLE_LIBRARY);
          break;
      }
    }
    ASSERT_NE(requester_provider, LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL);
    ASSERT_NE(chosen_provider, LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL);
    ASSERT_NE(wrong_provider, LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL);
    ASSERT_NE(partial_provider, LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL);

    loom_link_provider_binding_t bindings[] = {
        {IREE_SV("chosen"), chosen_provider},
    };
    loom_link_provider_resolver_t resolver = {};
    IREE_ASSERT_OK(loom_link_provider_resolver_prepare(
        loom_link_module_index_provider_count(index.get()), bindings,
        IREE_ARRAYSIZE(bindings), &resolver));
    const iree_string_view_t explicit_roots[] = {
        IREE_SV("@private_root"),
    };
    loom_link_plan_options_t options = {
        /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
        /*.root_symbols=*/
        {
            /*.count=*/IREE_ARRAYSIZE(explicit_roots),
            /*.values=*/explicit_roots,
        },
        /*.include_input_exports=*/true,
        /*.unresolved_policy=*/LOOM_LINK_PLAN_UNRESOLVED_ALLOW,
    };
    options.provider_resolver = &resolver;
    loom_link_index_materialization_t materialization = {};
    IREE_ASSERT_OK(
        TryMaterializeWithOptions(index.get(), &options, &materialization));
    Verify(materialization.module);

    EXPECT_EQ(loom_link_plan_symbol_count(materialization.plan), 10u);
    EXPECT_EQ(materialization.module->symbols.count, 7u);
    const auto expect_selected = [&](iree_host_size_t provider,
                                     iree_string_view_t name, bool expected) {
      const loom_link_module_index_symbol_t* symbol =
          FindIndexedProviderSymbol(index.get(), provider, name);
      ASSERT_NE(symbol, nullptr);
      EXPECT_EQ(
          loom_link_plan_contains_symbol(materialization.plan, symbol->ordinal),
          expected);
    };
    expect_selected(chosen_provider, IREE_SV("api"), true);
    expect_selected(chosen_provider, IREE_SV("provided"), true);
    expect_selected(chosen_provider, IREE_SV("unused"), false);
    expect_selected(wrong_provider, IREE_SV("api"), false);
    expect_selected(wrong_provider, IREE_SV("provided"), false);
    expect_selected(wrong_provider, IREE_SV("wrong_only"), false);
    expect_selected(partial_provider, IREE_SV("partial"), true);
    expect_selected(partial_provider, IREE_SV("external"), true);
    EXPECT_EQ(FindIndexedProviderSymbol(index.get(), partial_provider,
                                        IREE_SV("partial_unused")),
              nullptr);

    const loom_symbol_t* entry =
        FindSymbol(materialization.module, IREE_SV("entry"));
    const loom_symbol_t* api =
        FindSymbol(materialization.module, IREE_SV("api"));
    const loom_symbol_t* private_root =
        FindSymbol(materialization.module, IREE_SV("private_root"));
    const loom_symbol_t* local =
        FindSymbol(materialization.module, IREE_SV("local"));
    const loom_symbol_t* provided =
        FindSymbol(materialization.module, IREE_SV("provided"));
    const loom_symbol_t* linked_partial =
        FindSymbol(materialization.module, IREE_SV("partial"));
    const loom_symbol_t* external =
        FindSymbol(materialization.module, IREE_SV("external"));
    ASSERT_NE(entry, nullptr);
    ASSERT_NE(api, nullptr);
    ASSERT_NE(private_root, nullptr);
    ASSERT_NE(local, nullptr);
    ASSERT_NE(provided, nullptr);
    ASSERT_NE(linked_partial, nullptr);
    ASSERT_NE(external, nullptr);
    EXPECT_TRUE(iree_all_bits_set(
        entry->flags, LOOM_SYMBOL_FLAG_PUBLIC | LOOM_SYMBOL_FLAG_RETAIN));
    EXPECT_TRUE(iree_all_bits_set(
        api->flags, LOOM_SYMBOL_FLAG_PUBLIC | LOOM_SYMBOL_FLAG_RETAIN));
    EXPECT_TRUE(iree_any_bit_set(private_root->flags, LOOM_SYMBOL_FLAG_RETAIN));
    EXPECT_FALSE(
        iree_any_bit_set(private_root->flags, LOOM_SYMBOL_FLAG_PUBLIC));
    EXPECT_FALSE(
        iree_any_bit_set(local->flags | provided->flags | linked_partial->flags,
                         LOOM_SYMBOL_FLAG_PUBLIC | LOOM_SYMBOL_FLAG_RETAIN));
    EXPECT_TRUE(loom_symbol_definition_is_declaration(external->definition));
    EXPECT_FALSE(iree_any_bit_set(
        external->flags, LOOM_SYMBOL_FLAG_PUBLIC | LOOM_SYMBOL_FLAG_RETAIN));
    EXPECT_EQ(FindSymbol(materialization.module, IREE_SV("unused")), nullptr);
    EXPECT_EQ(FindSymbol(materialization.module, IREE_SV("wrong_only")),
              nullptr);
    EXPECT_EQ(FindSymbol(materialization.module, IREE_SV("partial_unused")),
              nullptr);

    const loom_func_like_t entry_function =
        loom_func_like_cast(materialization.module, entry->defining_op);
    const loom_func_like_t api_function =
        loom_func_like_cast(materialization.module, api->defining_op);
    ASSERT_TRUE(loom_func_like_isa(entry_function));
    ASSERT_TRUE(loom_func_like_isa(api_function));
    EXPECT_EQ(OptionalString(materialization.module,
                             loom_func_like_export_symbol(entry_function)),
              "request_entry");
    EXPECT_EQ(OptionalString(materialization.module,
                             loom_func_like_export_symbol(api_function)),
              "request_api");

    const loom_op_t* retained_import = nullptr;
    const loom_op_t* op = nullptr;
    loom_block_for_each_op(
        loom_region_const_entry_block(materialization.module->body), op) {
      if (!loom_module_import_isa(op)) continue;
      ASSERT_EQ(retained_import, nullptr);
      retained_import = op;
    }
    ASSERT_NE(retained_import, nullptr);
    EXPECT_EQ(OptionalString(materialization.module,
                             loom_module_import_provider(retained_import)),
              "external-runtime");
    const loom_symbol_ref_array_t import_symbols =
        loom_module_import_symbols(retained_import);
    ASSERT_EQ(import_symbols.count, 1u);
    EXPECT_EQ(
        import_symbols.values[0].symbol_id,
        (loom_symbol_id_t)(external - materialization.module->symbols.entries));

    const std::vector<LinkedSymbolShape> product_symbols =
        CaptureSymbolShapes(materialization.module);
    if (!reference_symbols.empty()) {
      EXPECT_EQ(product_symbols, reference_symbols);
    } else {
      reference_symbols = product_symbols;
    }

    loom_link_index_materialization_deinitialize(&materialization);
  }
}

}  // namespace
}  // namespace loom
