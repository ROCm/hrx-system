// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbol_references.h"

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
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/template/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

class SymbolReferencesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CONFIG, loom_config_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_GLOBAL, loom_global_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_KERNEL, loom_kernel_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEMPLATE, loom_template_dialect_vtables);
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    RegisterParameterizedAttrs(LOOM_DIALECT_TARGET,
                               loom_target_dialect_parameterized_attrs);
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
  using ParameterizedAttrsFn =
      const loom_parameterized_attr_descriptor_t* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id,
                       DialectVtablesFn dialect_vtables_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  void RegisterParameterizedAttrs(uint8_t dialect_id,
                                  ParameterizedAttrsFn parameterized_attrs_fn) {
    iree_host_size_t count = 0;
    const loom_parameterized_attr_descriptor_t* descriptors =
        parameterized_attrs_fn(&count);
    IREE_ASSERT_OK(loom_context_register_parameterized_attrs(
        &context_, dialect_id, descriptors, count));
  }

  ModulePtr ParseModule(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("symbol_references_test.loom"),
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

  loom_symbol_reference_table_t BuildTable(const loom_module_t* module) {
    loom_symbol_reference_table_t table = {};
    IREE_CHECK_OK(
        loom_symbol_reference_table_build(module, &analysis_arena_, &table));
    return table;
  }

  const loom_symbol_reference_occurrence_t* FindOccurrence(
      const loom_symbol_reference_table_t& table,
      loom_symbol_id_t source_symbol_id, loom_symbol_id_t target_symbol_id,
      loom_symbol_reference_occurrence_kind_t kind) {
    if (source_symbol_id == LOOM_SYMBOL_ID_INVALID) {
      loom_symbol_reference_occurrence_id_t occurrence_id =
          table.first_module_occurrence_id;
      while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
        const loom_symbol_reference_occurrence_t* occurrence =
            &table.occurrences[occurrence_id];
        if (occurrence->target_symbol_id == target_symbol_id &&
            occurrence->kind == kind) {
          return occurrence;
        }
        occurrence_id = occurrence->next_outgoing_occurrence_id;
      }
      return nullptr;
    }

    loom_symbol_reference_occurrence_id_t occurrence_id =
        table.symbols[source_symbol_id].first_outgoing_occurrence_id;
    while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
      const loom_symbol_reference_occurrence_t* occurrence =
          &table.occurrences[occurrence_id];
      if (occurrence->target_symbol_id == target_symbol_id &&
          occurrence->kind == kind) {
        return occurrence;
      }
      occurrence_id = occurrence->next_outgoing_occurrence_id;
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

TEST_F(SymbolReferencesTest, CallsAndGlobalAccessesUseGeneratedDescriptors) {
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

  loom_symbol_reference_table_t table = BuildTable(module.get());

  const loom_symbol_reference_occurrence_t* read_occurrence = FindOccurrence(
      table, reader, state, LOOM_SYMBOL_REFERENCE_OCCURRENCE_GLOBAL_ACCESS);
  ASSERT_NE(read_occurrence, nullptr);
  ASSERT_NE(read_occurrence->user_op, nullptr);
  EXPECT_EQ(read_occurrence->user_op->kind, LOOM_OP_GLOBAL_LOAD);
  EXPECT_EQ(read_occurrence->target_interfaces,
            LOOM_SYMBOL_INTERFACE_GLOBAL | LOOM_SYMBOL_INTERFACE_RODATA);

  const loom_symbol_reference_occurrence_t* write_occurrence = FindOccurrence(
      table, writer, state, LOOM_SYMBOL_REFERENCE_OCCURRENCE_GLOBAL_ACCESS);
  ASSERT_NE(write_occurrence, nullptr);
  ASSERT_NE(write_occurrence->user_op, nullptr);
  EXPECT_EQ(write_occurrence->user_op->kind, LOOM_OP_GLOBAL_STORE);
  EXPECT_EQ(write_occurrence->target_interfaces, LOOM_SYMBOL_INTERFACE_GLOBAL);

  const loom_symbol_reference_occurrence_t* call_occurrence = FindOccurrence(
      table, entry, reader, LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL);
  ASSERT_NE(call_occurrence, nullptr);
  ASSERT_NE(call_occurrence->user_op, nullptr);
  EXPECT_EQ(call_occurrence->user_op->kind, LOOM_OP_FUNC_CALL);
  EXPECT_EQ(call_occurrence->target_interfaces, LOOM_SYMBOL_INTERFACE_CALLABLE);

  EXPECT_EQ(read_occurrence->source_root_region_index_plus_one, 1u);
  EXPECT_EQ(write_occurrence->source_root_region_index_plus_one, 1u);
  EXPECT_EQ(call_occurrence->source_root_region_index_plus_one, 1u);

  EXPECT_EQ(table.symbols[state].incoming_count, 2u);
  EXPECT_EQ(table.symbols[reader].incoming_count, 1u);
}

TEST_F(SymbolReferencesTest, KernelReferencesRetainDistinctTargetInterfaces) {
  ModulePtr module = ParseModule(R"(
kernel.decl @logical() launch()
kernel.entry.decl @configured()

func.def @entry() {
  kernel.launch @logical() : ()
  kernel.dispatch @configured[]() : []()
  func.return
}
)");

  const loom_symbol_id_t logical = FindSymbol(module.get(), IREE_SV("logical"));
  const loom_symbol_id_t configured =
      FindSymbol(module.get(), IREE_SV("configured"));
  const loom_symbol_id_t entry = FindSymbol(module.get(), IREE_SV("entry"));

  const loom_symbol_reference_table_t table = BuildTable(module.get());

  const loom_symbol_reference_occurrence_t* launch_occurrence = FindOccurrence(
      table, entry, logical, LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL);
  ASSERT_NE(launch_occurrence, nullptr);
  ASSERT_NE(launch_occurrence->user_op, nullptr);
  EXPECT_EQ(launch_occurrence->user_op->kind, LOOM_OP_KERNEL_LAUNCH);
  EXPECT_EQ(launch_occurrence->target_interfaces, LOOM_SYMBOL_INTERFACE_KERNEL);

  const loom_symbol_reference_occurrence_t* dispatch_occurrence =
      FindOccurrence(table, entry, configured,
                     LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL);
  ASSERT_NE(dispatch_occurrence, nullptr);
  ASSERT_NE(dispatch_occurrence->user_op, nullptr);
  EXPECT_EQ(dispatch_occurrence->user_op->kind, LOOM_OP_KERNEL_DISPATCH);
  EXPECT_EQ(dispatch_occurrence->target_interfaces,
            LOOM_SYMBOL_INTERFACE_KERNEL_ENTRY);
}

TEST_F(SymbolReferencesTest, OccurrencesRetainContractAndRootRegionOrigins) {
  ModulePtr module = ParseModule(R"(
target.generic<reference> @header_target
func.decl @body_dependency()

func.def target(@header_target) @single_root() {
  func.call @body_dependency() : () -> ()
  func.return
}

test.record @config_dependency
test.record @implementation_dependency
test.split_func @split_root() {
  test.symbol_array_attrs [@config_dependency] using []
  test.yield
} launch {
  test.symbol_array_attrs [@implementation_dependency] using []
  test.yield
}
)");

  const loom_symbol_id_t header_target =
      FindSymbol(module.get(), IREE_SV("header_target"));
  const loom_symbol_id_t body_dependency =
      FindSymbol(module.get(), IREE_SV("body_dependency"));
  const loom_symbol_id_t single_root =
      FindSymbol(module.get(), IREE_SV("single_root"));
  const loom_symbol_id_t config_dependency =
      FindSymbol(module.get(), IREE_SV("config_dependency"));
  const loom_symbol_id_t implementation_dependency =
      FindSymbol(module.get(), IREE_SV("implementation_dependency"));
  const loom_symbol_id_t split_root =
      FindSymbol(module.get(), IREE_SV("split_root"));

  const loom_symbol_reference_table_t table = BuildTable(module.get());

  const loom_symbol_reference_occurrence_t* header_occurrence =
      FindOccurrence(table, single_root, header_target,
                     LOOM_SYMBOL_REFERENCE_OCCURRENCE_SYMBOL_ATTR);
  ASSERT_NE(header_occurrence, nullptr);
  EXPECT_EQ(header_occurrence->source_root_region_index_plus_one, 0u);

  const loom_symbol_reference_occurrence_t* body_occurrence =
      FindOccurrence(table, single_root, body_dependency,
                     LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL);
  ASSERT_NE(body_occurrence, nullptr);
  EXPECT_EQ(body_occurrence->source_root_region_index_plus_one, 1u);

  const loom_symbol_reference_occurrence_t* config_occurrence =
      FindOccurrence(table, split_root, config_dependency,
                     LOOM_SYMBOL_REFERENCE_OCCURRENCE_SYMBOL_ATTR);
  ASSERT_NE(config_occurrence, nullptr);
  EXPECT_EQ(config_occurrence->source_root_region_index_plus_one, 1u);

  const loom_symbol_reference_occurrence_t* implementation_occurrence =
      FindOccurrence(table, split_root, implementation_dependency,
                     LOOM_SYMBOL_REFERENCE_OCCURRENCE_SYMBOL_ATTR);
  ASSERT_NE(implementation_occurrence, nullptr);
  EXPECT_EQ(implementation_occurrence->source_root_region_index_plus_one, 2u);
}

TEST_F(SymbolReferencesTest, TemplateDemandsRetainOwningSymbol) {
  ModulePtr module = ParseModule(R"(
template.decl @outer.contract(%value: i32) -> (i32)
template.decl @demo.contract(%value: i32) -> (i32)

template.def<@outer.contract> @outer_provider(%value: i32) -> (i32) {
  %result = template.apply<@demo.contract>(%value) : (i32) -> (i32)
  template.return %result : i32
}

template.def<@demo.contract> @demo_provider(%value: i32) -> (i32) {
  template.return %value : i32
}

func.def public @entry(%arg: i32) -> (i32) {
  %result = template.apply<@demo.contract>(%arg) : (i32) -> (i32)
  func.return %result : i32
}
)");

  const loom_symbol_id_t outer_provider =
      FindSymbol(module.get(), IREE_SV("outer_provider"));
  const loom_symbol_id_t entry = FindSymbol(module.get(), IREE_SV("entry"));
  const loom_symbol_id_t family_symbol_id =
      FindSymbol(module.get(), IREE_SV("demo.contract"));
  const loom_symbol_id_t undemanded_family_symbol_id =
      FindSymbol(module.get(), IREE_SV("outer.contract"));

  const loom_symbol_reference_table_t table = BuildTable(module.get());

  ASSERT_EQ(table.template_demands.count, 2u);
  EXPECT_TRUE(loom_symbol_reference_template_family_is_demanded(
      &table, family_symbol_id));
  EXPECT_FALSE(loom_symbol_reference_template_family_is_demanded(
      &table, undemanded_family_symbol_id));
  const loom_symbol_id_t source_symbol_ids[] = {outer_provider, entry};
  for (loom_symbol_id_t source_symbol_id : source_symbol_ids) {
    ASSERT_EQ(table.symbols[source_symbol_id].template_demand_count, 1u);
    loom_template_demand_id_t demand_id =
        table.symbols[source_symbol_id].first_template_demand_id;
    ASSERT_NE(demand_id, LOOM_TEMPLATE_DEMAND_ID_INVALID);
    const loom_template_demand_t& demand =
        table.template_demands.values[demand_id];
    EXPECT_EQ(demand.source_symbol_id, source_symbol_id);
    EXPECT_EQ(demand.source_root_region_index_plus_one, 1u);
    EXPECT_EQ(demand.family_symbol_id, family_symbol_id);
    ASSERT_NE(demand.apply_op, nullptr);
    EXPECT_EQ(demand.apply_op->kind, LOOM_OP_TEMPLATE_APPLY);
    EXPECT_EQ(demand.next_source_demand_id, LOOM_TEMPLATE_DEMAND_ID_INVALID);
  }

  ASSERT_EQ(table.template_providers.count, 2u);
  ASSERT_NE(table.template_providers.first_by_family_symbol_id, nullptr);
  loom_template_provider_reference_id_t provider_id =
      table.template_providers
          .first_by_family_symbol_id[undemanded_family_symbol_id];
  ASSERT_NE(provider_id, LOOM_TEMPLATE_PROVIDER_REFERENCE_ID_INVALID);
  EXPECT_EQ(table.template_providers.values[provider_id].symbol_id,
            outer_provider);
  EXPECT_EQ(
      table.template_providers.values[provider_id].next_family_provider_id,
      LOOM_TEMPLATE_PROVIDER_REFERENCE_ID_INVALID);

  provider_id =
      table.template_providers.first_by_family_symbol_id[family_symbol_id];
  ASSERT_NE(provider_id, LOOM_TEMPLATE_PROVIDER_REFERENCE_ID_INVALID);
  EXPECT_EQ(table.template_providers.values[provider_id].symbol_id,
            FindSymbol(module.get(), IREE_SV("demo_provider")));
  EXPECT_EQ(
      table.template_providers.values[provider_id].next_family_provider_id,
      LOOM_TEMPLATE_PROVIDER_REFERENCE_ID_INVALID);
}

TEST_F(SymbolReferencesTest, OpenParameterizedArraysAreNotSymbolReferences) {
  ModulePtr module = ParseModule(R"(
template.decl @demo.contract(%arg: i32) -> (i32)

template.def<@demo.contract> requires [#target.subgroup.size<64>] @conditional(%arg: i32) -> (i32) {
  template.return %arg : i32
}
)");

  loom_symbol_id_t conditional =
      FindSymbol(module.get(), IREE_SV("conditional"));
  loom_symbol_id_t family = FindSymbol(module.get(), IREE_SV("demo.contract"));
  loom_symbol_reference_table_t table = BuildTable(module.get());

  ASSERT_EQ(table.symbol_count, 2u);
  EXPECT_EQ(table.symbols[conditional].outgoing_count, 1u);
  EXPECT_EQ(table.occurrence_count, 1u);
  EXPECT_NE(FindOccurrence(table, conditional, family,
                           LOOM_SYMBOL_REFERENCE_OCCURRENCE_SYMBOL_ATTR),
            nullptr);
}

TEST_F(SymbolReferencesTest, ConfigReadsUseNormalSymbolAttrOccurrences) {
  ModulePtr module = ParseModule(R"(
config.def @enable_mtp = true : i1

func.def @reader() -> (i1) {
  %enabled = config.get @enable_mtp : i1
  func.return %enabled : i1
}
)");

  loom_symbol_id_t enable_mtp = FindSymbol(module.get(), IREE_SV("enable_mtp"));
  loom_symbol_id_t reader = FindSymbol(module.get(), IREE_SV("reader"));

  loom_symbol_reference_table_t table = BuildTable(module.get());

  const loom_symbol_reference_occurrence_t* config_occurrence = FindOccurrence(
      table, reader, enable_mtp, LOOM_SYMBOL_REFERENCE_OCCURRENCE_SYMBOL_ATTR);
  ASSERT_NE(config_occurrence, nullptr);
  ASSERT_NE(config_occurrence->user_op, nullptr);
  EXPECT_EQ(config_occurrence->user_op->kind, LOOM_OP_CONFIG_GET);
  EXPECT_EQ(table.symbols[enable_mtp].incoming_count, 1u);
}

TEST_F(SymbolReferencesTest, NestedDictRefsFeedSymbolSccGraph) {
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

  loom_symbol_reference_table_t table = BuildTable(module.get());

  EXPECT_NE(FindOccurrence(table, derived, base,
                           LOOM_SYMBOL_REFERENCE_OCCURRENCE_NESTED_ATTR),
            nullptr);
  EXPECT_NE(
      FindOccurrence(table, a, b, LOOM_SYMBOL_REFERENCE_OCCURRENCE_NESTED_ATTR),
      nullptr);
  EXPECT_NE(
      FindOccurrence(table, b, a, LOOM_SYMBOL_REFERENCE_OCCURRENCE_NESTED_ATTR),
      nullptr);

  loom_scc_list_t sccs = {};
  loom_scc_graph_t graph = loom_symbol_reference_dependency_scc_graph(&table);
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

TEST_F(SymbolReferencesTest,
       AvailabilityOccurrencesRemainIndexedWithoutCreatingDependencies) {
  ModulePtr module = ParseModule(R"(
test.record @provider {options = #test.options<mode = fast, target = @dep_one>}
test.record @dep_one {depends = @provider}
test.record @dep_two {depends = @provider}
test.record @array_target

func.def @typed(%arg: test.matrix<bf16, scope = subgroup, rows = 16, target = @dep_two>) {
  test.template_param_symbol<@provider>
  test.parameterized_attr_array [#test.options<mode = fast, target = @array_target>] using []
  func.return
}
)");

  loom_symbol_id_t provider = FindSymbol(module.get(), IREE_SV("provider"));
  loom_symbol_id_t dep_one = FindSymbol(module.get(), IREE_SV("dep_one"));
  loom_symbol_id_t dep_two = FindSymbol(module.get(), IREE_SV("dep_two"));
  loom_symbol_id_t array_target =
      FindSymbol(module.get(), IREE_SV("array_target"));
  loom_symbol_id_t typed = FindSymbol(module.get(), IREE_SV("typed"));

  loom_symbol_reference_table_t table = BuildTable(module.get());

  const loom_symbol_reference_occurrence_t* first_dependency = FindOccurrence(
      table, dep_one, provider, LOOM_SYMBOL_REFERENCE_OCCURRENCE_NESTED_ATTR);
  ASSERT_NE(first_dependency, nullptr);
  EXPECT_TRUE(loom_symbol_reference_occurrence_is_dependency(first_dependency));
  const loom_symbol_reference_occurrence_t* second_dependency = FindOccurrence(
      table, dep_two, provider, LOOM_SYMBOL_REFERENCE_OCCURRENCE_NESTED_ATTR);
  ASSERT_NE(second_dependency, nullptr);
  EXPECT_TRUE(
      loom_symbol_reference_occurrence_is_dependency(second_dependency));

  const loom_symbol_reference_occurrence_t* parameter_availability =
      FindOccurrence(table, provider, dep_one,
                     LOOM_SYMBOL_REFERENCE_OCCURRENCE_NESTED_ATTR);
  ASSERT_NE(parameter_availability, nullptr);
  EXPECT_EQ(parameter_availability->role,
            LOOM_SYMBOL_REFERENCE_ROLE_AVAILABILITY);
  const loom_symbol_reference_occurrence_t* type_availability = FindOccurrence(
      table, typed, dep_two, LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE);
  ASSERT_NE(type_availability, nullptr);
  EXPECT_EQ(type_availability->role, LOOM_SYMBOL_REFERENCE_ROLE_AVAILABILITY);
  const loom_symbol_reference_occurrence_t* array_availability = FindOccurrence(
      table, typed, array_target, LOOM_SYMBOL_REFERENCE_OCCURRENCE_NESTED_ATTR);
  ASSERT_NE(array_availability, nullptr);
  EXPECT_EQ(array_availability->role, LOOM_SYMBOL_REFERENCE_ROLE_AVAILABILITY);
  const loom_symbol_reference_occurrence_t* nested_availability =
      FindOccurrence(table, typed, provider,
                     LOOM_SYMBOL_REFERENCE_OCCURRENCE_SYMBOL_ATTR);
  ASSERT_NE(nested_availability, nullptr);
  EXPECT_EQ(nested_availability->role, LOOM_SYMBOL_REFERENCE_ROLE_AVAILABILITY);

  EXPECT_EQ(table.symbols[provider].incoming_count, 3u);
  uint32_t dependency_count = 0;
  uint32_t availability_count = 0;
  loom_symbol_reference_occurrence_id_t occurrence_id =
      table.symbols[provider].first_incoming_occurrence_id;
  while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
    const loom_symbol_reference_occurrence_t* occurrence =
        &table.occurrences[occurrence_id];
    if (loom_symbol_reference_occurrence_is_dependency(occurrence)) {
      ++dependency_count;
    } else {
      ++availability_count;
    }
    occurrence_id = occurrence->next_incoming_occurrence_id;
  }
  EXPECT_EQ(dependency_count, 2u);
  EXPECT_EQ(availability_count, 1u);

  loom_scc_list_t sccs = {};
  loom_scc_graph_t graph = loom_symbol_reference_dependency_scc_graph(&table);
  IREE_ASSERT_OK(loom_scc_compute(&graph, nullptr, &analysis_arena_, &sccs));
  for (iree_host_size_t i = 0; i < sccs.count; ++i) {
    EXPECT_FALSE(sccs.values[i].is_cycle);
  }
}

TEST_F(SymbolReferencesTest, ValueTypeOccurrencesBelongToDefinitions) {
  ModulePtr module = ParseModule(R"(
test.record @target

func.decl @decl(%arg: test.matrix<bf16, scope = subgroup, rows = 16, target = @target>)

func.def @typed(%arg: test.matrix<bf16, scope = subgroup, rows = 16, target = @target>) {
  test.use %arg : test.matrix<bf16, scope = subgroup, rows = 16, target = @target>
  test.use %arg : test.matrix<bf16, scope = subgroup, rows = 16, target = @target>
  func.return
}
)");

  const loom_symbol_id_t target = FindSymbol(module.get(), IREE_SV("target"));
  const loom_symbol_id_t decl = FindSymbol(module.get(), IREE_SV("decl"));
  const loom_symbol_id_t typed = FindSymbol(module.get(), IREE_SV("typed"));
  const loom_symbol_reference_table_t table = BuildTable(module.get());

  EXPECT_NE(FindOccurrence(table, decl, target,
                           LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE),
            nullptr);

  uint32_t occurrence_count = 0;
  loom_symbol_reference_occurrence_id_t occurrence_id =
      table.symbols[typed].first_outgoing_occurrence_id;
  while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
    const loom_symbol_reference_occurrence_t* occurrence =
        &table.occurrences[occurrence_id];
    if (occurrence->target_symbol_id == target &&
        occurrence->kind == LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE) {
      ++occurrence_count;
    }
    occurrence_id = occurrence->next_outgoing_occurrence_id;
  }
  EXPECT_EQ(occurrence_count, 1u);
}

TEST_F(SymbolReferencesTest,
       SymbolArraysIndexEveryOccurrenceWithoutAvailabilityEdges) {
  ModulePtr module = ParseModule(R"(
test.record @provider_a
test.record @provider_b {depends = @consumer}

func.def @consumer() {
  test.symbol_array_attrs [@provider_a, @provider_a] using [@provider_b, @provider_a]
  func.return
}
)");

  loom_symbol_id_t provider_a = FindSymbol(module.get(), IREE_SV("provider_a"));
  loom_symbol_id_t provider_b = FindSymbol(module.get(), IREE_SV("provider_b"));
  loom_symbol_id_t consumer = FindSymbol(module.get(), IREE_SV("consumer"));
  loom_symbol_reference_table_t table = BuildTable(module.get());

  EXPECT_EQ(table.symbols[provider_a].incoming_count, 3u);
  EXPECT_EQ(table.symbols[provider_b].incoming_count, 1u);
  EXPECT_EQ(table.symbols[consumer].outgoing_count, 4u);

  uint32_t provider_a_dependency_count = 0;
  uint32_t provider_a_availability_count = 0;
  loom_symbol_reference_occurrence_id_t occurrence_id =
      table.symbols[provider_a].first_incoming_occurrence_id;
  while (occurrence_id != LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID) {
    const loom_symbol_reference_occurrence_t* occurrence =
        &table.occurrences[occurrence_id];
    ASSERT_EQ(occurrence->source_symbol_id, consumer);
    ASSERT_EQ(occurrence->kind, LOOM_SYMBOL_REFERENCE_OCCURRENCE_SYMBOL_ATTR);
    if (loom_symbol_reference_occurrence_is_dependency(occurrence)) {
      ++provider_a_dependency_count;
    } else {
      ++provider_a_availability_count;
    }
    occurrence_id = occurrence->next_incoming_occurrence_id;
  }
  EXPECT_EQ(provider_a_dependency_count, 2u);
  EXPECT_EQ(provider_a_availability_count, 1u);

  const loom_symbol_reference_occurrence_t* provider_b_availability =
      FindOccurrence(table, consumer, provider_b,
                     LOOM_SYMBOL_REFERENCE_OCCURRENCE_SYMBOL_ATTR);
  ASSERT_NE(provider_b_availability, nullptr);
  EXPECT_EQ(provider_b_availability->role,
            LOOM_SYMBOL_REFERENCE_ROLE_AVAILABILITY);

  loom_scc_list_t sccs = {};
  loom_scc_graph_t graph = loom_symbol_reference_dependency_scc_graph(&table);
  IREE_ASSERT_OK(loom_scc_compute(&graph, nullptr, &analysis_arena_, &sccs));
  for (iree_host_size_t i = 0; i < sccs.count; ++i) {
    EXPECT_FALSE(sccs.values[i].is_cycle);
  }
}

TEST_F(SymbolReferencesTest, RebuildsAfterAttrMutationAndErase) {
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
      &builder, LOOM_TEST_RECORD_BUILD_FLAG_HAS_DICT, 0, derived_ref,
      loom_make_named_attr_slice(&depends_attr, 1), LOOM_LOCATION_UNKNOWN,
      &derived_op));

  loom_symbol_reference_table_t table = BuildTable(module.get());
  EXPECT_NE(FindOccurrence(table, derived_ref.symbol_id, base_ref.symbol_id,
                           LOOM_SYMBOL_REFERENCE_OCCURRENCE_NESTED_ATTR),
            nullptr);

  loom_op_attrs(derived_op)[2] = MakeEmptyDict(module.get());
  table = BuildTable(module.get());
  EXPECT_EQ(FindOccurrence(table, derived_ref.symbol_id, base_ref.symbol_id,
                           LOOM_SYMBOL_REFERENCE_OCCURRENCE_NESTED_ATTR),
            nullptr);

  loom_op_attrs(derived_op)[2] =
      MakeSymbolDict(module.get(), IREE_SV("depends"), base_ref);
  table = BuildTable(module.get());
  EXPECT_NE(FindOccurrence(table, derived_ref.symbol_id, base_ref.symbol_id,
                           LOOM_SYMBOL_REFERENCE_OCCURRENCE_NESTED_ATTR),
            nullptr);

  IREE_ASSERT_OK(loom_op_erase(module.get(), derived_op));
  table = BuildTable(module.get());
  EXPECT_EQ(FindOccurrence(table, derived_ref.symbol_id, base_ref.symbol_id,
                           LOOM_SYMBOL_REFERENCE_OCCURRENCE_NESTED_ATTR),
            nullptr);
}

TEST_F(SymbolReferencesTest, CompactionRemapsRebuiltReferences) {
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
      &builder, LOOM_TEST_RECORD_BUILD_FLAG_HAS_DICT, 0, owner_ref,
      loom_make_named_attr_slice(&depends_attr, 1), LOOM_LOCATION_UNKNOWN,
      &owner_op));

  EXPECT_EQ(dead_ref.symbol_id, 0u);
  EXPECT_EQ(base_ref.symbol_id, 1u);
  EXPECT_EQ(owner_ref.symbol_id, 2u);
  loom_symbol_reference_table_t table = BuildTable(module.get());
  EXPECT_NE(FindOccurrence(table, owner_ref.symbol_id, base_ref.symbol_id,
                           LOOM_SYMBOL_REFERENCE_OCCURRENCE_NESTED_ATTR),
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
  EXPECT_NE(FindOccurrence(table, owner, base,
                           LOOM_SYMBOL_REFERENCE_OCCURRENCE_NESTED_ATTR),
            nullptr);
}

TEST_F(SymbolReferencesTest, TypeAndEncodingRefsUseOneTable) {
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
      /*.family=*/{},
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
      &builder, LOOM_TEST_RECORD_BUILD_FLAG_HAS_DICT, 0, owner_ref,
      loom_make_named_attr_slice(owner_attrs, IREE_ARRAYSIZE(owner_attrs)),
      LOOM_LOCATION_UNKNOWN, &owner_op));

  loom_symbol_ref_t typed_func_ref =
      AddSymbol(module.get(), IREE_SV("typed_func"));
  loom_op_t* typed_func_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &builder, 0, 0, 0, typed_func_ref, nullptr, 0, &encoded_type, 1, nullptr,
      0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &typed_func_op));

  loom_type_t register_type = {};
  IREE_ASSERT_OK(loom_module_intern_register_type(
      module.get(), /*carrier_payload0=*/1,
      /*carrier_payload1=*/(uint64_t)1 << 16, encoded_type, &register_type));
  loom_symbol_ref_t register_func_ref =
      AddSymbol(module.get(), IREE_SV("register_func"));
  loom_op_t* register_func_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &builder, 0, 0, 0, register_func_ref, nullptr, 0, &register_type, 1,
      nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &register_func_op));

  loom_symbol_reference_table_t table = BuildTable(module.get());
  loom_symbol_id_t meta = meta_ref.symbol_id;
  loom_symbol_id_t owner = owner_ref.symbol_id;
  loom_symbol_id_t typed_func = typed_func_ref.symbol_id;
  loom_symbol_id_t register_func = register_func_ref.symbol_id;

  EXPECT_NE(FindOccurrence(table, owner, meta,
                           LOOM_SYMBOL_REFERENCE_OCCURRENCE_TYPE_ATTR),
            nullptr);
  EXPECT_NE(FindOccurrence(table, owner, meta,
                           LOOM_SYMBOL_REFERENCE_OCCURRENCE_ENCODING_ATTR),
            nullptr);
  EXPECT_NE(FindOccurrence(table, typed_func, meta,
                           LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE),
            nullptr);
  EXPECT_NE(FindOccurrence(table, register_func, meta,
                           LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE),
            nullptr);
  EXPECT_NE(FindOccurrence(table, LOOM_SYMBOL_ID_INVALID, meta,
                           LOOM_SYMBOL_REFERENCE_OCCURRENCE_MODULE_ENCODING),
            nullptr);
}

}  // namespace
}  // namespace loom
