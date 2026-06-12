// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/planner.h"

#include <memory>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/io/stream.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/test/ops.h"

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

std::string StringViewToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

class LinkPlannerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_CHECK, loom_check_dialect_vtables,
                    loom_check_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_CONFIG, loom_config_dialect_vtables,
                    loom_config_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables,
                    loom_func_dialect_op_semantics);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables,
                    loom_test_dialect_op_semantics);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    for (loom_module_t* module : modules_) {
      loom_module_free(module);
    }
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);
  using DialectSemanticsFn = const loom_op_semantics_t* (*)(iree_host_size_t*);

  void RegisterDialect(uint8_t dialect_id, DialectVtablesFn dialect_vtables_fn,
                       DialectSemanticsFn dialect_semantics_fn) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
    iree_host_size_t semantics_count = 0;
    const loom_op_semantics_t* semantics =
        dialect_semantics_fn(&semantics_count);
    IREE_ASSERT_OK(loom_context_register_dialect_semantics(
        &context_, dialect_id, semantics, (uint16_t)semantics_count));
  }

  loom_module_t* Parse(iree_string_view_t source,
                       iree_string_view_t filename = IREE_SV("test.loom")) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t parse_options = {
        /*.diagnostic_sink=*/{},
        /*.max_errors=*/20,
    };
    IREE_EXPECT_OK(loom_text_parse(source, filename, &context_, &block_pool_,
                                   &parse_options, &module));
    EXPECT_NE(module, nullptr);
    if (module) {
      modules_.push_back(module);
    }
    return module;
  }

  IndexPtr CreateIndex() {
    loom_link_module_index_t* index = nullptr;
    IREE_CHECK_OK(loom_link_module_index_create(
        &context_, &block_pool_, iree_allocator_system(), &index));
    return IndexPtr(index);
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

    iree_io_stream_pos_t length = iree_io_stream_length(stream);
    std::vector<uint8_t> bytes(length);
    IREE_CHECK_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IREE_CHECK_OK(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }

  void AddMaterialized(loom_link_module_index_t* index,
                       const loom_module_t* module, iree_string_view_t name,
                       loom_link_provider_role_t role) {
    loom_link_module_index_add_options_t options = {
        /*.provider_name=*/name,
        /*.role=*/role,
    };
    IREE_ASSERT_OK(loom_link_module_index_add_materialized(
        index, module, &options, /*out_provider_ordinal=*/nullptr));
  }

  PlanPtr BuildPlan(const loom_link_module_index_t* index,
                    const loom_link_plan_options_t* options) {
    loom_link_plan_t* plan = nullptr;
    IREE_CHECK_OK(loom_link_plan_build(index, options, &block_pool_,
                                       iree_allocator_system(), &plan));
    return PlanPtr(plan);
  }

  iree_status_t BuildPlanStatus(const loom_link_module_index_t* index,
                                const loom_link_plan_options_t* options,
                                PlanPtr* out_plan) {
    loom_link_plan_t* plan = nullptr;
    iree_status_t status = loom_link_plan_build(index, options, &block_pool_,
                                                iree_allocator_system(), &plan);
    if (iree_status_is_ok(status)) {
      *out_plan = PlanPtr(plan);
    }
    return status;
  }

  const loom_link_plan_symbol_t* FindPlannedSymbol(
      const loom_link_plan_t* plan,
      const loom_link_module_index_symbol_t* symbol) {
    if (!symbol) return nullptr;
    for (iree_host_size_t i = 0; i < loom_link_plan_symbol_count(plan); ++i) {
      const loom_link_plan_symbol_t* planned =
          loom_link_plan_symbol_at(plan, i);
      if (planned && planned->symbol_ordinal == symbol->ordinal) {
        return planned;
      }
    }
    return nullptr;
  }

  bool ContainsSymbol(const loom_link_plan_t* plan,
                      const loom_link_module_index_symbol_t* symbol) {
    return symbol && loom_link_plan_contains_symbol(plan, symbol->ordinal);
  }

  std::vector<std::string> PlannedNames(const loom_link_plan_t* plan) {
    const loom_link_module_index_t* index = loom_link_plan_index(plan);
    std::vector<std::string> names;
    for (iree_host_size_t i = 0; i < loom_link_plan_symbol_count(plan); ++i) {
      const loom_link_plan_symbol_t* planned =
          loom_link_plan_symbol_at(plan, i);
      const loom_link_module_index_symbol_t* symbol =
          loom_link_module_index_symbol_at(index, planned->symbol_ordinal);
      names.push_back(StringViewToString(symbol->name));
    }
    return names;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_ = {};
  std::vector<loom_module_t*> modules_;
};

typedef struct BytecodePlanMaterializer {
  // Materialized IR matching the used bytecode provider.
  const loom_module_t* used_module;
  // Number of times the used provider was materialized.
  int used_count;
  // Number of times an unused provider was materialized.
  int unused_count;
} BytecodePlanMaterializer;

static iree_status_t MaterializeUsedBytecodeModule(
    void* user_data, const loom_link_module_index_t* index,
    const loom_link_module_index_module_t* module,
    const loom_module_t** out_module) {
  BytecodePlanMaterializer* materializer = (BytecodePlanMaterializer*)user_data;
  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_provider_at(index, module->provider_ordinal);
  if (!provider) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "missing provider for test module");
  }
  if (iree_string_view_equal(provider->name, IREE_SV("used-lib"))) {
    ++materializer->used_count;
    *out_module = materializer->used_module;
    return iree_ok_status();
  }
  ++materializer->unused_count;
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "unused bytecode provider was materialized");
}

TEST_F(LinkPlannerTest, ArchiveSelectsAllSymbolsInStableIndexOrder) {
  loom_module_t* first = Parse(IREE_SV(R"(
func.def public @entry_a(%x: i32) -> (i32) {
  %y = func.call @helper(%x) : (i32) -> (i32)
  func.return %y : i32
}

func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* second = Parse(IREE_SV(R"(
func.def public @entry_b(%x: i32) -> (i32) {
  %y = func.call @helper(%x) : (i32) -> (i32)
  func.return %y : i32
}

func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), first, IREE_SV("first"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), second, IREE_SV("second"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);

  PlanPtr plan = BuildPlan(index.get(), /*options=*/nullptr);

  EXPECT_EQ(loom_link_plan_symbol_count(plan.get()), 4u);
  EXPECT_EQ(
      PlannedNames(plan.get()),
      (std::vector<std::string>{"entry_a", "helper", "entry_b", "helper"}));
  for (iree_host_size_t i = 0; i < loom_link_plan_symbol_count(plan.get());
       ++i) {
    const loom_link_plan_symbol_t* symbol =
        loom_link_plan_symbol_at(plan.get(), i);
    ASSERT_NE(symbol, nullptr);
    EXPECT_EQ(symbol->reason, LOOM_LINK_PLAN_LIVE_ARCHIVE);
    EXPECT_EQ(symbol->cause_ordinal, LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL);
  }
}

TEST_F(LinkPlannerTest, SelectiveRootClosureSelectsPrivateDependencyOnly) {
  loom_module_t* module = Parse(IREE_SV(R"(
func.def public @entry(%x: i32) -> (i32) {
  %y = func.call @helper(%x) : (i32) -> (i32)
  func.return %y : i32
}

func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def public @unused(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def @unused_private(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), module, IREE_SV("input"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("entry"));
  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_module_at(index.get(), 0);
  ASSERT_NE(indexed_module, nullptr);
  const loom_link_module_index_symbol_t* helper =
      loom_link_module_index_lookup_private(index.get(), indexed_module,
                                            IREE_SV("helper"));
  const loom_link_module_index_symbol_t* unused =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("unused"));
  const loom_link_module_index_symbol_t* unused_private =
      loom_link_module_index_lookup_private(index.get(), indexed_module,
                                            IREE_SV("unused_private"));

  ASSERT_TRUE(ContainsSymbol(plan.get(), entry));
  ASSERT_TRUE(ContainsSymbol(plan.get(), helper));
  EXPECT_FALSE(ContainsSymbol(plan.get(), unused));
  EXPECT_FALSE(ContainsSymbol(plan.get(), unused_private));

  const loom_link_plan_symbol_t* planned_entry =
      FindPlannedSymbol(plan.get(), entry);
  const loom_link_plan_symbol_t* planned_helper =
      FindPlannedSymbol(plan.get(), helper);
  ASSERT_NE(planned_entry, nullptr);
  ASSERT_NE(planned_helper, nullptr);
  EXPECT_EQ(planned_entry->reason, LOOM_LINK_PLAN_LIVE_ROOT);
  EXPECT_EQ(planned_helper->reason, LOOM_LINK_PLAN_LIVE_DEPENDENCY);
  EXPECT_EQ(planned_helper->cause_ordinal, planned_entry->ordinal);
}

TEST_F(LinkPlannerTest, SelectiveBytecodePlanningMaterializesSelectedModules) {
  loom_module_t* used = Parse(IREE_SV(R"(
func.def public @entry(%x: i32) -> (i32) {
  %y = func.call @helper(%x) : (i32) -> (i32)
  func.return %y : i32
}

func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* unused = Parse(IREE_SV(R"(
func.def public @unused(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  std::vector<uint8_t> used_bytes = WriteModule(used);
  std::vector<uint8_t> unused_bytes = WriteModule(unused);

  IndexPtr index = CreateIndex();
  loom_link_module_index_add_options_t used_options = {
      /*.provider_name=*/IREE_SV("used-lib"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      index.get(),
      iree_make_const_byte_span(used_bytes.data(), used_bytes.size()),
      IREE_SV("used.loombc"), /*read_options=*/nullptr, &used_options,
      /*out_provider_ordinal=*/nullptr));
  loom_link_module_index_add_options_t unused_options = {
      /*.provider_name=*/IREE_SV("unused-lib"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      index.get(),
      iree_make_const_byte_span(unused_bytes.data(), unused_bytes.size()),
      IREE_SV("unused.loombc"), /*read_options=*/nullptr, &unused_options,
      /*out_provider_ordinal=*/nullptr));

  const loom_link_module_index_module_t* used_module =
      loom_link_module_index_module_at(index.get(), 0);
  const loom_link_module_index_module_t* unused_module =
      loom_link_module_index_module_at(index.get(), 1);
  ASSERT_NE(used_module, nullptr);
  ASSERT_NE(unused_module, nullptr);
  EXPECT_EQ(used_module->materialized_module, nullptr);
  EXPECT_EQ(unused_module->materialized_module, nullptr);

  BytecodePlanMaterializer materializer = {
      /*.used_module=*/used,
  };
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
      /*.include_exported_roots=*/{},
      /*.unresolved_policy=*/{},
      /*.check_policy=*/{},
      /*.strip_symbol=*/{},
      /*.strip_symbol_user_data=*/{},
      /*.materialize_module=*/MaterializeUsedBytecodeModule,
      /*.materialize_module_user_data=*/&materializer,
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("entry"));
  const loom_link_module_index_symbol_t* helper =
      loom_link_module_index_lookup_private(index.get(), used_module,
                                            IREE_SV("helper"));
  const loom_link_module_index_symbol_t* unused_symbol =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("unused"));
  EXPECT_TRUE(ContainsSymbol(plan.get(), entry));
  EXPECT_TRUE(ContainsSymbol(plan.get(), helper));
  EXPECT_FALSE(ContainsSymbol(plan.get(), unused_symbol));
  EXPECT_EQ(materializer.used_count, 1);
  EXPECT_EQ(materializer.unused_count, 0);
}

TEST_F(LinkPlannerTest, SelectiveApplyUsesBytecodeProviderContractIndex) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.def public @entry(%x: i32) -> (i32) {
  %y = func.apply<demo.bytecode>(%x) : (i32) -> (i32)
  func.return %y : i32
}
)"));
  loom_module_t* used = Parse(IREE_SV(R"(
func.template<demo.bytecode> @bytecode_provider(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* unused = Parse(IREE_SV(R"(
func.template<demo.unused> @unused_provider(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  std::vector<uint8_t> used_bytes = WriteModule(used);
  std::vector<uint8_t> unused_bytes = WriteModule(unused);

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  loom_link_module_index_add_options_t used_options = {
      /*.provider_name=*/IREE_SV("used-lib"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      index.get(),
      iree_make_const_byte_span(used_bytes.data(), used_bytes.size()),
      IREE_SV("used.loombc"), /*read_options=*/nullptr, &used_options,
      /*out_provider_ordinal=*/nullptr));
  loom_link_module_index_add_options_t unused_options = {
      /*.provider_name=*/IREE_SV("unused-lib"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_LIBRARY,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      index.get(),
      iree_make_const_byte_span(unused_bytes.data(), unused_bytes.size()),
      IREE_SV("unused.loombc"), /*read_options=*/nullptr, &unused_options,
      /*out_provider_ordinal=*/nullptr));

  BytecodePlanMaterializer materializer = {
      /*.used_module=*/used,
  };
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/
      {
          /*.count=*/IREE_ARRAYSIZE(roots),
          /*.values=*/roots,
      },
      /*.include_exported_roots=*/{},
      /*.unresolved_policy=*/{},
      /*.check_policy=*/{},
      /*.strip_symbol=*/{},
      /*.strip_symbol_user_data=*/{},
      /*.materialize_module=*/MaterializeUsedBytecodeModule,
      /*.materialize_module_user_data=*/&materializer,
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("entry"));
  const loom_link_module_index_module_t* used_module =
      loom_link_module_index_module_at(index.get(), 1);
  const loom_link_module_index_module_t* unused_module =
      loom_link_module_index_module_at(index.get(), 2);
  ASSERT_NE(used_module, nullptr);
  ASSERT_NE(unused_module, nullptr);
  const loom_link_module_index_symbol_t* bytecode_provider =
      loom_link_module_index_lookup_private(index.get(), used_module,
                                            IREE_SV("bytecode_provider"));
  const loom_link_module_index_symbol_t* unused_provider =
      loom_link_module_index_lookup_private(index.get(), unused_module,
                                            IREE_SV("unused_provider"));

  EXPECT_TRUE(ContainsSymbol(plan.get(), entry));
  EXPECT_TRUE(ContainsSymbol(plan.get(), bytecode_provider));
  EXPECT_FALSE(ContainsSymbol(plan.get(), unused_provider));
  EXPECT_EQ(materializer.used_count, 1);
  EXPECT_EQ(materializer.unused_count, 0);
}

TEST_F(LinkPlannerTest, SelectiveRootMayNameUniquePrivateSymbol) {
  loom_module_t* module = Parse(IREE_SV(R"(
func.def @entry(%x: i32) -> (i32) {
  %y = func.call @helper(%x) : (i32) -> (i32)
  func.return %y : i32
}

func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), module, IREE_SV("input"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_module_at(index.get(), 0);
  ASSERT_NE(indexed_module, nullptr);
  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_private(index.get(), indexed_module,
                                            IREE_SV("entry"));
  const loom_link_module_index_symbol_t* helper =
      loom_link_module_index_lookup_private(index.get(), indexed_module,
                                            IREE_SV("helper"));
  EXPECT_TRUE(ContainsSymbol(plan.get(), entry));
  EXPECT_TRUE(ContainsSymbol(plan.get(), helper));
}

TEST_F(LinkPlannerTest, SelectiveDeclarationPullsConcreteProviderDefinition) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.decl public @callee(%x: i32) -> (i32)

func.def public @entry(%x: i32) -> (i32) {
  %y = func.call @callee(%x) : (i32) -> (i32)
  func.return %y : i32
}
)"));
  loom_module_t* library = Parse(IREE_SV(R"(
func.def public @callee(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def public @unused(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), library, IREE_SV("library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("entry"));
  const loom_link_module_index_symbol_t* callee_decl =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("callee"));
  ASSERT_NE(callee_decl, nullptr);
  const loom_link_module_index_symbol_t* callee_def =
      loom_link_module_index_next_global_duplicate(index.get(), callee_decl);
  ASSERT_NE(callee_def, nullptr);
  const loom_link_module_index_symbol_t* unused =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("unused"));

  EXPECT_TRUE(ContainsSymbol(plan.get(), entry));
  EXPECT_TRUE(ContainsSymbol(plan.get(), callee_decl));
  EXPECT_TRUE(ContainsSymbol(plan.get(), callee_def));
  EXPECT_FALSE(ContainsSymbol(plan.get(), unused));

  const loom_link_module_index_provider_t* def_provider =
      loom_link_module_index_symbol_provider(index.get(), callee_def);
  ASSERT_NE(def_provider, nullptr);
  EXPECT_EQ(StringViewToString(def_provider->name), "library");

  const loom_link_plan_symbol_t* planned_decl =
      FindPlannedSymbol(plan.get(), callee_decl);
  const loom_link_plan_symbol_t* planned_def =
      FindPlannedSymbol(plan.get(), callee_def);
  ASSERT_NE(planned_decl, nullptr);
  ASSERT_NE(planned_def, nullptr);
  EXPECT_EQ(planned_decl->reason, LOOM_LINK_PLAN_LIVE_DEPENDENCY);
  EXPECT_EQ(planned_def->reason, LOOM_LINK_PLAN_LIVE_DEPENDENCY);
  EXPECT_EQ(planned_def->cause_ordinal, planned_decl->ordinal);
}

TEST_F(LinkPlannerTest, SelectiveDeclarationMayPullPrivateConcreteDefinition) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.decl @callee(%x: i32) -> (i32)

func.def @entry(%x: i32) -> (i32) {
  %y = func.call @callee(%x) : (i32) -> (i32)
  func.return %y : i32
}
)"));
  loom_module_t* library = Parse(IREE_SV(R"(
func.def @callee(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), library, IREE_SV("library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_module_t* harness_module =
      loom_link_module_index_module_at(index.get(), 0);
  const loom_link_module_index_module_t* library_module =
      loom_link_module_index_module_at(index.get(), 1);
  ASSERT_NE(harness_module, nullptr);
  ASSERT_NE(library_module, nullptr);
  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_private(index.get(), harness_module,
                                            IREE_SV("entry"));
  const loom_link_module_index_symbol_t* callee_decl =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("callee"));
  const loom_link_module_index_symbol_t* callee_def =
      loom_link_module_index_lookup_private(index.get(), library_module,
                                            IREE_SV("callee"));
  EXPECT_TRUE(ContainsSymbol(plan.get(), entry));
  EXPECT_TRUE(ContainsSymbol(plan.get(), callee_decl));
  EXPECT_TRUE(ContainsSymbol(plan.get(), callee_def));
}

TEST_F(LinkPlannerTest, SelectiveApplyPullsProviderContractImplementations) {
  loom_module_t* harness = Parse(IREE_SV(R"(
test.target<low_core> @root_target

func.def public target(@root_target) @entry(%x: i32) -> (i32) {
  %y = func.apply<demo.targeted>(%x) : (i32) -> (i32)
  func.return %y : i32
}
)"));
  loom_module_t* library = Parse(IREE_SV(R"(
test.target<low_core> @gfx11
test.target<low_core> @gfx12

func.template<demo.targeted> target(@gfx11) priority(20) @gfx11_provider(%x: i32) -> (i32) {
  func.return %x : i32
}

func.template<demo.targeted> target(@gfx12) priority(20) @gfx12_provider(%x: i32) -> (i32) {
  func.return %x : i32
}

func.template<demo.targeted> priority(1) @fallback_provider(%x: i32) -> (i32) {
  func.return %x : i32
}

func.template<demo.unused> @unused_provider(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), library, IREE_SV("library"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/
      {
          /*.count=*/IREE_ARRAYSIZE(roots),
          /*.values=*/roots,
      },
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("entry"));
  const loom_link_module_index_module_t* library_module =
      loom_link_module_index_module_at(index.get(), 1);
  ASSERT_NE(library_module, nullptr);
  const loom_link_module_index_symbol_t* gfx11 =
      loom_link_module_index_lookup_private(index.get(), library_module,
                                            IREE_SV("gfx11"));
  const loom_link_module_index_symbol_t* gfx12 =
      loom_link_module_index_lookup_private(index.get(), library_module,
                                            IREE_SV("gfx12"));
  const loom_link_module_index_symbol_t* gfx11_provider =
      loom_link_module_index_lookup_private(index.get(), library_module,
                                            IREE_SV("gfx11_provider"));
  const loom_link_module_index_symbol_t* gfx12_provider =
      loom_link_module_index_lookup_private(index.get(), library_module,
                                            IREE_SV("gfx12_provider"));
  const loom_link_module_index_symbol_t* fallback_provider =
      loom_link_module_index_lookup_private(index.get(), library_module,
                                            IREE_SV("fallback_provider"));
  const loom_link_module_index_symbol_t* unused_provider =
      loom_link_module_index_lookup_private(index.get(), library_module,
                                            IREE_SV("unused_provider"));

  EXPECT_TRUE(ContainsSymbol(plan.get(), entry));
  EXPECT_TRUE(ContainsSymbol(plan.get(), gfx11_provider));
  EXPECT_TRUE(ContainsSymbol(plan.get(), gfx12_provider));
  EXPECT_TRUE(ContainsSymbol(plan.get(), fallback_provider));
  EXPECT_FALSE(ContainsSymbol(plan.get(), unused_provider));
  EXPECT_TRUE(ContainsSymbol(plan.get(), gfx11));
  EXPECT_TRUE(ContainsSymbol(plan.get(), gfx12));

  const loom_link_plan_symbol_t* planned_entry =
      FindPlannedSymbol(plan.get(), entry);
  const loom_link_plan_symbol_t* planned_gfx11_provider =
      FindPlannedSymbol(plan.get(), gfx11_provider);
  ASSERT_NE(planned_entry, nullptr);
  ASSERT_NE(planned_gfx11_provider, nullptr);
  EXPECT_EQ(planned_gfx11_provider->reason, LOOM_LINK_PLAN_LIVE_PROVIDER);
  EXPECT_EQ(planned_gfx11_provider->cause_ordinal, planned_entry->ordinal);
}

TEST_F(LinkPlannerTest, SelectiveRootIgnoresUnreachableDuplicateDefinition) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.def public @entry(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* first = Parse(IREE_SV(R"(
func.def public @unused(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* second = Parse(IREE_SV(R"(
func.def public @unused(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), first, IREE_SV("first"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  AddMaterialized(index.get(), second, IREE_SV("second"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("entry"));
  const loom_link_module_index_symbol_t* unused =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("unused"));
  ASSERT_TRUE(ContainsSymbol(plan.get(), entry));
  EXPECT_FALSE(ContainsSymbol(plan.get(), unused));
  EXPECT_EQ(loom_link_plan_symbol_count(plan.get()), 1u);
}

TEST_F(LinkPlannerTest, ArchiveRejectsDuplicatePublicConcreteDefinitions) {
  loom_module_t* first = Parse(IREE_SV(R"(
func.def public @same(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* second = Parse(IREE_SV(R"(
func.def public @same(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), first, IREE_SV("first"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), second, IREE_SV("second"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);

  PlanPtr plan;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ALREADY_EXISTS,
      BuildPlanStatus(index.get(), /*options=*/nullptr, &plan));
}

TEST_F(LinkPlannerTest, ExportedRootsRejectDuplicateDefinitionsBehindDecl) {
  loom_module_t* harness = Parse(IREE_SV(R"(
func.decl public @same(%x: i32) -> (i32)
)"));
  loom_module_t* first = Parse(IREE_SV(R"(
func.def public @same(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));
  loom_module_t* second = Parse(IREE_SV(R"(
func.def public @same(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), harness, IREE_SV("harness"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  AddMaterialized(index.get(), first, IREE_SV("first"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  AddMaterialized(index.get(), second, IREE_SV("second"),
                  LOOM_LINK_PROVIDER_ROLE_LIBRARY);
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/{},
      /*.include_exported_roots=*/true,
  };

  PlanPtr plan;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        BuildPlanStatus(index.get(), &options, &plan));
}

TEST_F(LinkPlannerTest, SelectiveReportsMissingRoot) {
  loom_module_t* module = Parse(IREE_SV(R"(
func.def public @entry(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), module, IREE_SV("input"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  iree_string_view_t roots[] = {IREE_SV("@missing")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };

  PlanPtr plan;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        BuildPlanStatus(index.get(), &options, &plan));
}

static bool StripNamedSymbol(void* user_data,
                             const loom_link_module_index_t* index,
                             const loom_link_module_index_symbol_t* symbol) {
  (void)index;
  iree_string_view_t* stripped_name = (iree_string_view_t*)user_data;
  return iree_string_view_equal(symbol->name, *stripped_name);
}

TEST_F(LinkPlannerTest, StripPolicyControlsRequiredDependencies) {
  loom_module_t* module = Parse(IREE_SV(R"(
func.def public @entry(%x: i32) -> (i32) {
  %y = func.call @helper(%x) : (i32) -> (i32)
  func.return %y : i32
}

func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), module, IREE_SV("input"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  iree_string_view_t roots[] = {IREE_SV("@entry")};
  iree_string_view_t stripped_name = IREE_SV("helper");
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
      /*.include_exported_roots=*/{},
      /*.unresolved_policy=*/{},
      /*.check_policy=*/{},
      /*.strip_symbol=*/StripNamedSymbol,
      /*.strip_symbol_user_data=*/&stripped_name,
  };

  PlanPtr plan;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        BuildPlanStatus(index.get(), &options, &plan));

  options.unresolved_policy = LOOM_LINK_PLAN_UNRESOLVED_ALLOW;
  plan = BuildPlan(index.get(), &options);
  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("entry"));
  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_module_at(index.get(), 0);
  ASSERT_NE(indexed_module, nullptr);
  const loom_link_module_index_symbol_t* helper =
      loom_link_module_index_lookup_private(index.get(), indexed_module,
                                            IREE_SV("helper"));
  EXPECT_TRUE(ContainsSymbol(plan.get(), entry));
  EXPECT_FALSE(ContainsSymbol(plan.get(), helper));
}

TEST_F(LinkPlannerTest, CheckStripPolicyRemovesBytecodeCasesAndBenchmarks) {
  loom_module_t* module = Parse(IREE_SV(R"(
func.def public @kernel(%x: i32) -> (i32) {
  func.return %x : i32
}

check.case public @kernel_case {
  %input = check.literal value(1) : i32
  %actual = func.call @kernel(%input) : (i32) -> (i32)
  check.expect.equal actual(%actual) expected(%input) : i32
  check.return
}

check.benchmark<@kernel_case> @kernel_bench
)"));
  ASSERT_NE(module, nullptr);
  std::vector<uint8_t> bytes = WriteModule(module);

  IndexPtr index = CreateIndex();
  loom_link_module_index_add_options_t provider_options = {
      /*.provider_name=*/IREE_SV("kernel-lib"),
      /*.role=*/LOOM_LINK_PROVIDER_ROLE_INPUT,
  };
  IREE_ASSERT_OK(loom_link_module_index_add_bytecode(
      index.get(), iree_make_const_byte_span(bytes.data(), bytes.size()),
      IREE_SV("kernel-lib.loombc"), /*read_options=*/nullptr, &provider_options,
      /*out_provider_ordinal=*/nullptr));
  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_module_at(index.get(), 0);
  ASSERT_NE(indexed_module, nullptr);
  EXPECT_EQ(indexed_module->materialized_module, nullptr);

  loom_link_plan_options_t strip_options = {
      /*.mode=*/LOOM_LINK_PLAN_ARCHIVE,
      /*.root_symbols=*/{},
      /*.include_exported_roots=*/{},
      /*.unresolved_policy=*/{},
      /*.check_policy=*/LOOM_LINK_PLAN_CHECK_STRIP,
  };
  PlanPtr plan = BuildPlan(index.get(), &strip_options);

  const loom_link_module_index_symbol_t* kernel =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("kernel"));
  const loom_link_module_index_symbol_t* check_case =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("kernel_case"));
  const loom_link_module_index_symbol_t* benchmark =
      loom_link_module_index_lookup_private(index.get(), indexed_module,
                                            IREE_SV("kernel_bench"));
  EXPECT_TRUE(ContainsSymbol(plan.get(), kernel));
  EXPECT_FALSE(ContainsSymbol(plan.get(), check_case));
  EXPECT_FALSE(ContainsSymbol(plan.get(), benchmark));
}

TEST_F(LinkPlannerTest, KeepCheckPolicyPreservesCaseDependencies) {
  loom_module_t* module = Parse(IREE_SV(R"(
func.def public @kernel(%x: i32) -> (i32) {
  func.return %x : i32
}

check.case public @kernel_case {
  %input = check.literal value(1) : i32
  %actual = func.call @kernel(%input) : (i32) -> (i32)
  check.expect.equal actual(%actual) expected(%input) : i32
  check.return
}
)"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), module, IREE_SV("input"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  iree_string_view_t roots[] = {IREE_SV("@kernel_case")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_symbol_t* kernel =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("kernel"));
  const loom_link_module_index_symbol_t* check_case =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("kernel_case"));
  EXPECT_TRUE(ContainsSymbol(plan.get(), kernel));
  EXPECT_TRUE(ContainsSymbol(plan.get(), check_case));
}

TEST_F(LinkPlannerTest, CheckStripPolicyRejectsStrippedRoots) {
  loom_module_t* module = Parse(IREE_SV(R"(
check.case public @kernel_case {
  check.return
}
)"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), module, IREE_SV("input"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  iree_string_view_t roots[] = {IREE_SV("@kernel_case")};
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/{/*.count=*/IREE_ARRAYSIZE(roots), /*.values=*/roots},
      /*.include_exported_roots=*/{},
      /*.unresolved_policy=*/{},
      /*.check_policy=*/LOOM_LINK_PLAN_CHECK_STRIP,
  };

  PlanPtr plan;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        BuildPlanStatus(index.get(), &options, &plan));
}

TEST_F(LinkPlannerTest, ExportedRootPolicySelectsExportsAndDependencies) {
  loom_module_t* module = Parse(IREE_SV(R"(
func.def public @entry(%x: i32) -> (i32) {
  %y = func.call @helper(%x) : (i32) -> (i32)
  func.return %y : i32
}

func.def @helper(%x: i32) -> (i32) {
  func.return %x : i32
}

func.def public @second(%x: i32) -> (i32) {
  func.return %x : i32
}
)"));

  IndexPtr index = CreateIndex();
  AddMaterialized(index.get(), module, IREE_SV("input"),
                  LOOM_LINK_PROVIDER_ROLE_INPUT);
  loom_link_plan_options_t options = {
      /*.mode=*/LOOM_LINK_PLAN_SELECTIVE,
      /*.root_symbols=*/{},
      /*.include_exported_roots=*/true,
  };
  PlanPtr plan = BuildPlan(index.get(), &options);

  const loom_link_module_index_symbol_t* entry =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("entry"));
  const loom_link_module_index_symbol_t* second =
      loom_link_module_index_lookup_global(index.get(), IREE_SV("second"));
  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_module_at(index.get(), 0);
  ASSERT_NE(indexed_module, nullptr);
  const loom_link_module_index_symbol_t* helper =
      loom_link_module_index_lookup_private(index.get(), indexed_module,
                                            IREE_SV("helper"));
  EXPECT_TRUE(ContainsSymbol(plan.get(), entry));
  EXPECT_TRUE(ContainsSymbol(plan.get(), second));
  EXPECT_TRUE(ContainsSymbol(plan.get(), helper));
}

}  // namespace
}  // namespace loom
