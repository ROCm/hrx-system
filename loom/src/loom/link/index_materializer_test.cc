// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/index_materializer.h"

#include <algorithm>
#include <memory>
#include <string>
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
    loom_link_plan_materialization_environment_t environment = {};
    environment.context = &context_;
    environment.block_pool = &block_pool_;
    environment.allocator = iree_allocator_system();
    return loom_link_index_materialize(index, &plan_options, &environment,
                                       IREE_SV("linked"), out_materialization);
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
  ASSERT_TRUE(slow_metadata->has_body);
  std::fill_n(library_bytecode.data() + slow_metadata->body_absolute_offset,
              slow_metadata->body_length, UINT8_C(0xFF));

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
  ASSERT_TRUE(implementation_metadata->has_body);
  std::fill_n(
      library_bytecode.data() + implementation_metadata->body_absolute_offset,
      implementation_metadata->body_length, UINT8_C(0xFF));

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

}  // namespace
}  // namespace loom
