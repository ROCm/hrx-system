// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbol_dependencies.h"

#include <algorithm>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class SymbolDependenciesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CONFIG, loom_config_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_GLOBAL, loom_global_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    iree_arena_initialize(&block_pool_, &analysis_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&analysis_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("symbol_dependencies_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  ModulePtr AllocateModule() {
    loom_module_t* module = nullptr;
    IREE_CHECK_OK(loom_module_allocate(&context_, IREE_SV("test"), &block_pool_,
                                       nullptr, iree_allocator_system(),
                                       &module));
    return ModulePtr(module);
  }

  loom_symbol_id_t FindSymbol(const loom_module_t* module,
                              iree_string_view_t name) {
    loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT_NE(name_id, LOOM_STRING_ID_INVALID);
    loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT_NE(symbol_id, LOOM_SYMBOL_ID_INVALID);
    return symbol_id;
  }

  loom_symbol_ref_t AddSymbol(loom_module_t* module, iree_string_view_t name) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module, name, &name_id));
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_CHECK_OK(loom_module_add_symbol(module, name_id, &symbol_id));
    return {
        /*.module_id=*/0,
        /*.symbol_id=*/symbol_id,
    };
  }

  loom_named_attr_t MakeSymbolNamedAttr(loom_module_t* module,
                                        iree_string_view_t key,
                                        loom_symbol_ref_t ref) {
    loom_string_id_t key_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module, key, &key_id));
    return {
        /*.name_id=*/key_id,
        /*.reserved=*/{},
        /*.value=*/loom_attr_symbol(ref),
    };
  }

  loom_attribute_t MakeSymbolDict(loom_module_t* module, iree_string_view_t key,
                                  loom_symbol_ref_t ref) {
    loom_named_attr_t attr = MakeSymbolNamedAttr(module, key, ref);
    loom_attribute_t dict = {};
    IREE_CHECK_OK(loom_module_make_canonical_attr_dict(
        module, loom_make_named_attr_slice(&attr, 1), &dict));
    return dict;
  }

  loom_attribute_t MakeEmptyDict(loom_module_t* module) {
    loom_attribute_t dict = {};
    IREE_CHECK_OK(loom_module_make_canonical_attr_dict(
        module, loom_named_attr_slice_empty(), &dict));
    return dict;
  }

  loom_symbol_dependency_table_t BuildTable(const loom_module_t* module) {
    loom_symbol_dependency_table_t table = {};
    IREE_CHECK_OK(
        loom_symbol_dependency_table_build(module, &analysis_arena_, &table));
    return table;
  }

  const loom_symbol_dependency_edge_t* FindEdge(
      const loom_symbol_dependency_table_t& table,
      loom_symbol_id_t source_symbol_id, loom_symbol_id_t target_symbol_id,
      loom_symbol_dependency_edge_kind_t kind) {
    if (source_symbol_id == LOOM_SYMBOL_ID_INVALID) {
      loom_symbol_dependency_edge_id_t edge_id = table.first_module_edge_id;
      while (edge_id != LOOM_SYMBOL_DEPENDENCY_EDGE_ID_INVALID) {
        const loom_symbol_dependency_edge_t* edge = &table.edges[edge_id];
        if (edge->target_symbol_id == target_symbol_id && edge->kind == kind) {
          return edge;
        }
        edge_id = edge->next_outgoing_edge_id;
      }
      return nullptr;
    }

    loom_symbol_dependency_edge_id_t edge_id =
        table.symbols[source_symbol_id].first_outgoing_edge_id;
    while (edge_id != LOOM_SYMBOL_DEPENDENCY_EDGE_ID_INVALID) {
      const loom_symbol_dependency_edge_t* edge = &table.edges[edge_id];
      if (edge->target_symbol_id == target_symbol_id && edge->kind == kind) {
        return edge;
      }
      edge_id = edge->next_outgoing_edge_id;
    }
    return nullptr;
  }

  std::vector<iree_host_size_t> ComponentNodes(const loom_scc_t& component) {
    std::vector<iree_host_size_t> nodes(component.nodes,
                                        component.nodes + component.node_count);
    std::sort(nodes.begin(), nodes.end());
    return nodes;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  iree_arena_allocator_t analysis_arena_;
};

TEST_F(SymbolDependenciesTest, CallsAndGlobalAccessesUseGeneratedDescriptors) {
  ModulePtr module = ParseModule(R"(
global.variable @state : index

func.def @reader() -> (index) {
  %value = global.load @state : index
  func.return %value : index
}

func.def @writer(%value: index) {
  global.store %value, @state : index
  func.return
}

func.def @entry() -> (index) {
  %value = func.call @reader() : () -> (index)
  func.return %value : index
}
)");

  loom_symbol_id_t state = FindSymbol(module.get(), IREE_SV("state"));
  loom_symbol_id_t reader = FindSymbol(module.get(), IREE_SV("reader"));
  loom_symbol_id_t writer = FindSymbol(module.get(), IREE_SV("writer"));
  loom_symbol_id_t entry = FindSymbol(module.get(), IREE_SV("entry"));

  loom_symbol_dependency_table_t table = BuildTable(module.get());

  const loom_symbol_dependency_edge_t* read_edge =
      FindEdge(table, reader, state, LOOM_SYMBOL_DEPENDENCY_EDGE_GLOBAL_ACCESS);
  ASSERT_NE(read_edge, nullptr);
  ASSERT_NE(read_edge->user_op, nullptr);
  EXPECT_EQ(read_edge->user_op->kind, LOOM_OP_GLOBAL_LOAD);

  const loom_symbol_dependency_edge_t* write_edge =
      FindEdge(table, writer, state, LOOM_SYMBOL_DEPENDENCY_EDGE_GLOBAL_ACCESS);
  ASSERT_NE(write_edge, nullptr);
  ASSERT_NE(write_edge->user_op, nullptr);
  EXPECT_EQ(write_edge->user_op->kind, LOOM_OP_GLOBAL_STORE);

  const loom_symbol_dependency_edge_t* call_edge =
      FindEdge(table, entry, reader, LOOM_SYMBOL_DEPENDENCY_EDGE_CALL);
  ASSERT_NE(call_edge, nullptr);
  ASSERT_NE(call_edge->user_op, nullptr);
  EXPECT_EQ(call_edge->user_op->kind, LOOM_OP_FUNC_CALL);

  EXPECT_EQ(table.symbols[state].incoming_count, 2u);
  EXPECT_EQ(table.symbols[reader].incoming_count, 1u);
}

TEST_F(SymbolDependenciesTest, ConfigReadsUseNormalSymbolAttrEdges) {
  ModulePtr module = ParseModule(R"(
config.def @enable_mtp = true : i1

func.def @reader() -> (i1) {
  %enabled = config.get @enable_mtp : i1
  func.return %enabled : i1
}
)");

  loom_symbol_id_t enable_mtp = FindSymbol(module.get(), IREE_SV("enable_mtp"));
  loom_symbol_id_t reader = FindSymbol(module.get(), IREE_SV("reader"));

  loom_symbol_dependency_table_t table = BuildTable(module.get());

  const loom_symbol_dependency_edge_t* config_edge = FindEdge(
      table, reader, enable_mtp, LOOM_SYMBOL_DEPENDENCY_EDGE_SYMBOL_ATTR);
  ASSERT_NE(config_edge, nullptr);
  ASSERT_NE(config_edge->user_op, nullptr);
  EXPECT_EQ(config_edge->user_op->kind, LOOM_OP_CONFIG_GET);
  EXPECT_EQ(table.symbols[enable_mtp].incoming_count, 1u);
}

TEST_F(SymbolDependenciesTest, NestedDictRefsFeedSymbolSccGraph) {
  ModulePtr module = ParseModule(R"(
test.record @base
test.record @derived {depends = @base}
test.record @a {depends = @b}
test.record @b {depends = @a}
)");

  loom_symbol_id_t base = FindSymbol(module.get(), IREE_SV("base"));
  loom_symbol_id_t derived = FindSymbol(module.get(), IREE_SV("derived"));
  loom_symbol_id_t a = FindSymbol(module.get(), IREE_SV("a"));
  loom_symbol_id_t b = FindSymbol(module.get(), IREE_SV("b"));

  loom_symbol_dependency_table_t table = BuildTable(module.get());

  EXPECT_NE(
      FindEdge(table, derived, base, LOOM_SYMBOL_DEPENDENCY_EDGE_NESTED_ATTR),
      nullptr);
  EXPECT_NE(FindEdge(table, a, b, LOOM_SYMBOL_DEPENDENCY_EDGE_NESTED_ATTR),
            nullptr);
  EXPECT_NE(FindEdge(table, b, a, LOOM_SYMBOL_DEPENDENCY_EDGE_NESTED_ATTR),
            nullptr);

  loom_scc_list_t sccs = {};
  loom_scc_graph_t graph = loom_symbol_dependency_scc_graph(&table);
  IREE_ASSERT_OK(loom_scc_compute(&graph, nullptr, &analysis_arena_, &sccs));

  bool found_cycle = false;
  for (iree_host_size_t i = 0; i < sccs.count; ++i) {
    const loom_scc_t& component = sccs.values[i];
    if (!component.is_cycle) continue;
    EXPECT_EQ(ComponentNodes(component), (std::vector<iree_host_size_t>{a, b}));
    found_cycle = true;
  }
  EXPECT_TRUE(found_cycle);
}

TEST_F(SymbolDependenciesTest, RebuildsAfterAttrMutationAndErase) {
  ModulePtr module = AllocateModule();
  loom_builder_t builder = {};
  loom_builder_initialize(module.get(), &module->arena,
                          loom_module_block(module.get()), &builder);

  loom_symbol_ref_t base_ref = AddSymbol(module.get(), IREE_SV("base"));
  loom_op_t* base_op = nullptr;
  IREE_ASSERT_OK(loom_test_record_build(&builder, 0, 0, base_ref,
                                        loom_named_attr_slice_empty(),
                                        LOOM_LOCATION_UNKNOWN, &base_op));

  loom_named_attr_t depends_attr =
      MakeSymbolNamedAttr(module.get(), IREE_SV("depends"), base_ref);
  loom_symbol_ref_t derived_ref = AddSymbol(module.get(), IREE_SV("derived"));
  loom_op_t* derived_op = nullptr;
  IREE_ASSERT_OK(loom_test_record_build(
      &builder, 0, 0, derived_ref, loom_make_named_attr_slice(&depends_attr, 1),
      LOOM_LOCATION_UNKNOWN, &derived_op));

  loom_symbol_dependency_table_t table = BuildTable(module.get());
  EXPECT_NE(FindEdge(table, derived_ref.symbol_id, base_ref.symbol_id,
                     LOOM_SYMBOL_DEPENDENCY_EDGE_NESTED_ATTR),
            nullptr);

  loom_op_attrs(derived_op)[2] = MakeEmptyDict(module.get());
  table = BuildTable(module.get());
  EXPECT_EQ(FindEdge(table, derived_ref.symbol_id, base_ref.symbol_id,
                     LOOM_SYMBOL_DEPENDENCY_EDGE_NESTED_ATTR),
            nullptr);

  loom_op_attrs(derived_op)[2] =
      MakeSymbolDict(module.get(), IREE_SV("depends"), base_ref);
  table = BuildTable(module.get());
  EXPECT_NE(FindEdge(table, derived_ref.symbol_id, base_ref.symbol_id,
                     LOOM_SYMBOL_DEPENDENCY_EDGE_NESTED_ATTR),
            nullptr);

  IREE_ASSERT_OK(loom_op_erase(module.get(), derived_op));
  table = BuildTable(module.get());
  EXPECT_EQ(FindEdge(table, derived_ref.symbol_id, base_ref.symbol_id,
                     LOOM_SYMBOL_DEPENDENCY_EDGE_NESTED_ATTR),
            nullptr);
}

TEST_F(SymbolDependenciesTest, CompactionRemapsRebuiltDependencies) {
  ModulePtr module = AllocateModule();
  loom_builder_t builder = {};
  loom_builder_initialize(module.get(), &module->arena,
                          loom_module_block(module.get()), &builder);

  loom_symbol_ref_t dead_ref = AddSymbol(module.get(), IREE_SV("dead"));
  loom_symbol_ref_t base_ref = AddSymbol(module.get(), IREE_SV("base"));
  loom_op_t* base_op = nullptr;
  IREE_ASSERT_OK(loom_test_record_build(&builder, 0, 0, base_ref,
                                        loom_named_attr_slice_empty(),
                                        LOOM_LOCATION_UNKNOWN, &base_op));

  loom_named_attr_t depends_attr =
      MakeSymbolNamedAttr(module.get(), IREE_SV("depends"), base_ref);
  loom_symbol_ref_t owner_ref = AddSymbol(module.get(), IREE_SV("owner"));
  loom_op_t* owner_op = nullptr;
  IREE_ASSERT_OK(loom_test_record_build(
      &builder, 0, 0, owner_ref, loom_make_named_attr_slice(&depends_attr, 1),
      LOOM_LOCATION_UNKNOWN, &owner_op));

  EXPECT_EQ(dead_ref.symbol_id, 0u);
  EXPECT_EQ(base_ref.symbol_id, 1u);
  EXPECT_EQ(owner_ref.symbol_id, 2u);
  loom_symbol_dependency_table_t table = BuildTable(module.get());
  EXPECT_NE(FindEdge(table, owner_ref.symbol_id, base_ref.symbol_id,
                     LOOM_SYMBOL_DEPENDENCY_EDGE_NESTED_ATTR),
            nullptr);

  iree_host_size_t removed_count = 0;
  IREE_ASSERT_OK(loom_module_compact_symbols(module.get(), &analysis_arena_,
                                             &removed_count));
  EXPECT_EQ(removed_count, 1u);

  loom_symbol_id_t base = FindSymbol(module.get(), IREE_SV("base"));
  loom_symbol_id_t owner = FindSymbol(module.get(), IREE_SV("owner"));
  EXPECT_EQ(base, 0u);
  EXPECT_EQ(owner, 1u);

  table = BuildTable(module.get());
  EXPECT_EQ(table.symbol_count, 2u);
  EXPECT_NE(
      FindEdge(table, owner, base, LOOM_SYMBOL_DEPENDENCY_EDGE_NESTED_ATTR),
      nullptr);
}

TEST_F(SymbolDependenciesTest, TypeAndEncodingRefsUseOneTable) {
  ModulePtr module = AllocateModule();
  loom_builder_t builder = {};
  loom_builder_initialize(module.get(), &module->arena,
                          loom_module_block(module.get()), &builder);

  loom_symbol_ref_t meta_ref = AddSymbol(module.get(), IREE_SV("meta"));
  loom_op_t* meta_op = nullptr;
  IREE_ASSERT_OK(loom_test_record_build(&builder, 0, 0, meta_ref,
                                        loom_named_attr_slice_empty(),
                                        LOOM_LOCATION_UNKNOWN, &meta_op));

  loom_attribute_t nested_encoding_dict =
      MakeSymbolDict(module.get(), IREE_SV("ref"), meta_ref);
  loom_string_id_t encoding_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module.get(), IREE_SV("dependent"),
                                           &encoding_name_id));
  loom_string_id_t refs_key_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module.get(), IREE_SV("refs"), &refs_key_id));
  loom_named_attr_t encoding_attrs[] = {{
      /*.name_id=*/refs_key_id,
      /*.reserved=*/{},
      /*.value=*/nested_encoding_dict,
  }};
  loom_encoding_t encoding = {
      /*.name_id=*/encoding_name_id,
      /*.alias_id=*/{},
      /*.attribute_count=*/IREE_ARRAYSIZE(encoding_attrs),
      /*.reserved=*/{},
      /*.attributes=*/encoding_attrs,
  };
  uint16_t encoding_id = 0;
  IREE_ASSERT_OK(
      loom_module_add_encoding(module.get(), &encoding, &encoding_id));

  loom_type_t encoded_type =
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), encoding_id);
  loom_type_id_t encoded_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_type_id(module.get(), encoded_type, &encoded_type_id));

  loom_string_id_t type_key_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module.get(), IREE_SV("type"), &type_key_id));
  loom_string_id_t encoding_key_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module.get(), IREE_SV("encoding"),
                                           &encoding_key_id));
  loom_named_attr_t owner_attrs[] = {
      {
          /*.name_id=*/type_key_id,
          /*.reserved=*/{},
          /*.value=*/loom_attr_type(encoded_type_id),
      },
      {
          /*.name_id=*/encoding_key_id,
          /*.reserved=*/{},
          /*.value=*/loom_attr_encoding(encoding_id),
      },
  };

  loom_symbol_ref_t owner_ref = AddSymbol(module.get(), IREE_SV("owner"));
  loom_op_t* owner_op = nullptr;
  IREE_ASSERT_OK(loom_test_record_build(
      &builder, 0, 0, owner_ref,
      loom_make_named_attr_slice(owner_attrs, IREE_ARRAYSIZE(owner_attrs)),
      LOOM_LOCATION_UNKNOWN, &owner_op));

  loom_symbol_ref_t typed_func_ref =
      AddSymbol(module.get(), IREE_SV("typed_func"));
  loom_op_t* typed_func_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &builder, 0, 0, 0, typed_func_ref, nullptr, 0, &encoded_type, 1, nullptr,
      0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &typed_func_op));

  loom_symbol_dependency_table_t table = BuildTable(module.get());
  loom_symbol_id_t meta = meta_ref.symbol_id;
  loom_symbol_id_t owner = owner_ref.symbol_id;
  loom_symbol_id_t typed_func = typed_func_ref.symbol_id;

  EXPECT_NE(FindEdge(table, owner, meta, LOOM_SYMBOL_DEPENDENCY_EDGE_TYPE_ATTR),
            nullptr);
  EXPECT_NE(
      FindEdge(table, owner, meta, LOOM_SYMBOL_DEPENDENCY_EDGE_ENCODING_ATTR),
      nullptr);
  EXPECT_NE(
      FindEdge(table, typed_func, meta, LOOM_SYMBOL_DEPENDENCY_EDGE_VALUE_TYPE),
      nullptr);
  EXPECT_NE(FindEdge(table, LOOM_SYMBOL_ID_INVALID, meta,
                     LOOM_SYMBOL_DEPENDENCY_EDGE_MODULE_ENCODING),
            nullptr);
}

}  // namespace
}  // namespace loom
