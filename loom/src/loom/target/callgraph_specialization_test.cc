// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/callgraph_specialization.h"

#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/pass_environment.h"
#include "loom/target/provider.h"
#include "loom/target/specialization.h"
#include "loom/target/test/target_records.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

typedef struct TestTargetProfile {
  // Generic target profile base.
  loom_target_profile_t base;

  // Exact subgroup size projected by this profile.
  uint32_t subgroup_size;
} TestTargetProfile;

static iree_status_t ProjectTestProfileFacts(
    const loom_target_profile_t* base_profile, iree_arena_allocator_t* arena,
    loom_target_facts_t* out_facts) {
  (void)arena;
  const auto* profile =
      reinterpret_cast<const TestTargetProfile*>(base_profile);
  out_facts->selector = LOOM_TEST_TARGET_KIND_LOW_CORE;
  out_facts->storage.snapshot.subgroup_size = profile->subgroup_size;
  return iree_ok_status();
}

static const loom_target_profile_type_t kTestProfileType = {
    /*.name=*/IREE_SVL("callgraph-specialization-test"),
    /*.fact_type=*/&loom_test_target_fact_type,
    /*.project_facts=*/ProjectTestProfileFacts,
};

static TestTargetProfile MakeTestProfile(uint32_t subgroup_size) {
  return TestTargetProfile{
      /*.base=*/
      {
          /*.type=*/&kTestProfileType,
          /*.target_bundle=*/
          loom_target_bundle_table_lookup(&loom_test_target_bundles,
                                          LOOM_TEST_TARGET_KIND_LOW_CORE),
      },
      /*.subgroup_size=*/subgroup_size,
  };
}

static const loom_target_provider_t kTestProvider = {
    /*.profile_type=*/&kTestProfileType,
};

static const loom_target_provider_t* const kTestProviders[] = {
    &kTestProvider,
};

static const loom_target_provider_set_t kTestProviderSet =
    loom_target_provider_set_make(kTestProviders,
                                  IREE_ARRAYSIZE(kTestProviders));

struct DiagnosticCollector {
  std::vector<const loom_error_def_t*> errors;
  std::vector<std::string> strings;
};

static iree_status_t CollectDiagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  auto* collector = static_cast<DiagnosticCollector*>(user_data);
  collector->errors.push_back(emission->error);
  for (iree_host_size_t i = 0; i < emission->param_count; ++i) {
    if (emission->params[i].kind != LOOM_PARAM_STRING) continue;
    const iree_string_view_t value = emission->params[i].string;
    collector->strings.emplace_back(value.data, value.size);
  }
  return iree_ok_status();
}

class TargetCallgraphSpecializationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TARGET, loom_target_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(
        loom_target_environment_initialize(&kTestProviderSet, &environment_));
    iree_arena_initialize(&block_pool_, &version_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&version_arena_);
    loom_target_environment_deinitialize(&environment_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn =
      const loom_op_vtable_t* const* (*)(iree_host_size_t*);

  void RegisterDialect(loom_dialect_id_t dialect_id, DialectVtablesFn fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, dialect_id, vtables, static_cast<uint16_t>(vtable_count)));
  }

  ModulePtr Parse(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_CHECK_OK(
        loom_text_parse(iree_make_cstring_view(source),
                        IREE_SV("target_callgraph_specialization_test.loom"),
                        &context_, &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_symbol_ref_t SymbolRef(const loom_module_t* module,
                              iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    return loom_symbol_ref_t{/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  }

  loom_func_like_t Function(const loom_module_t* module,
                            loom_symbol_ref_t ref) {
    IREE_ASSERT(ref.module_id == 0);
    IREE_ASSERT(ref.symbol_id < module->symbols.count);
    loom_func_like_t function = loom_func_like_cast(
        module, module->symbols.entries[ref.symbol_id].defining_op);
    IREE_ASSERT(loom_func_like_isa(function));
    return function;
  }

  loom_func_like_t Function(const loom_module_t* module,
                            iree_string_view_t name) {
    return Function(module, SymbolRef(module, name));
  }

  loom_symbol_ref_t OnlySemanticCallee(const loom_module_t* module,
                                       loom_func_like_t function) {
    loom_symbol_ref_t callee_ref = loom_symbol_ref_null();
    iree_host_size_t call_count = 0;
    loom_block_t* block =
        loom_region_entry_block(loom_func_like_body(function));
    for (iree_host_size_t i = 0; i < block->op_count; ++i) {
      loom_call_like_t call =
          loom_call_like_cast(module, loom_block_op(block, i));
      if (!loom_call_like_isa(call) ||
          loom_call_like_kind(call) != LOOM_CALL_LIKE_KIND_SEMANTIC) {
        continue;
      }
      callee_ref = loom_call_like_callee(call);
      ++call_count;
    }
    EXPECT_EQ(call_count, 1u);
    return callee_ref;
  }

  bool ContainsSubgroupQuery(loom_func_like_t function) {
    loom_block_t* block =
        loom_region_entry_block(loom_func_like_body(function));
    for (iree_host_size_t i = 0; i < block->op_count; ++i) {
      if (loom_target_subgroup_size_isa(loom_block_op(block, i))) return true;
    }
    return false;
  }

  loom_target_specialization_result_t Specialize(
      loom_module_t* module,
      const loom_target_specialization_request_t* requests,
      iree_host_size_t request_count) {
    loom_target_specialization_result_t result = {};
    IREE_CHECK_OK(loom_target_specialize_functions(&environment_, module,
                                                   {
                                                       /*.values=*/requests,
                                                       /*.count=*/request_count,
                                                   },
                                                   /*.diagnostic_emitter=*/{},
                                                   &version_arena_, &result));
    EXPECT_EQ(result.error_count, 0u);
    return result;
  }

  bool Run(loom_module_t* module, loom_function_version_owner_t* owner,
           DiagnosticCollector* collector = nullptr) {
    iree_arena_allocator_t pass_arena;
    iree_arena_initialize(&block_pool_, &pass_arena);
    const loom_target_pass_capability_t target_capability =
        loom_target_pass_capability_make_mutable(&environment_, owner);
    const loom_pass_environment_capability_t* capabilities[] = {
        &target_capability.base,
    };
    const loom_pass_environment_t pass_environment =
        loom_pass_environment_make(capabilities, IREE_ARRAYSIZE(capabilities));
    const loom_pass_info_t* pass_info =
        loom_target_callgraph_specialization_pass_info();
    std::vector<uint8_t> statistic_storage(
        pass_info->statistic_layout->storage_size, 0);
    loom_pass_t pass = {};
    pass.info = pass_info;
    pass.module_run = loom_target_callgraph_specialization_run;
    pass.instance_arena = &pass_arena;
    pass.arena = &pass_arena;
    pass.statistic_storage = statistic_storage.data();
    pass.environment = &pass_environment;
    if (collector != nullptr) {
      pass.diagnostic_emitter = {
          /*.fn=*/CollectDiagnostic,
          /*.user_data=*/collector,
      };
    }
    IREE_CHECK_OK(loom_target_callgraph_specialization_run(&pass, module));
    const bool changed = pass.changed;
    iree_arena_deinitialize(&pass_arena);
    return changed;
  }

  const loom_target_function_version_t* Version(
      const loom_function_version_owner_t& owner, loom_func_like_t function) {
    return loom_target_function_version_const_cast(
        loom_function_version_list_find(&owner.list, function));
  }

  const loom_target_function_version_t* Version(
      const loom_module_t* module, const loom_function_version_owner_t& owner,
      loom_symbol_ref_t ref) {
    return Version(owner, Function(module, ref));
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_environment_t environment_;
  iree_arena_allocator_t version_arena_;
};

TEST_F(TargetCallgraphSpecializationTest,
       SharesTargetlessHelpersWithinEachInvocationContext) {
  ModulePtr module = Parse(R"(
func.def @leaf() -> (index) {
  %size = target.subgroup.size : index
  func.return %size : index
}

func.def @middle() -> (index) {
  %size = func.call @leaf() : () -> (index)
  func.return %size : index
}

func.def public @left() -> (index) {
  %size = func.call @middle() : () -> (index)
  func.return %size : index
}

func.def public @right() -> (index) {
  %size = func.call @middle() : () -> (index)
  func.return %size : index
}

func.def public @wide() -> (index) {
  %size = func.call @middle() : () -> (index)
  func.return %size : index
}
)");
  const TestTargetProfile wave32 = MakeTestProfile(32);
  const TestTargetProfile wave64 = MakeTestProfile(64);
  const loom_target_specialization_request_t requests[] = {
      {/*.function_name=*/IREE_SV("left"),
       /*.target_profile=*/&wave32.base},
      {/*.function_name=*/IREE_SV("right"),
       /*.target_profile=*/&wave32.base},
      {/*.function_name=*/IREE_SV("wide"),
       /*.target_profile=*/&wave64.base},
  };
  loom_target_specialization_result_t specialization =
      Specialize(module.get(), requests, IREE_ARRAYSIZE(requests));
  ASSERT_EQ(specialization.function_versions.list.count, 3u);
  const iree_host_size_t source_symbol_count = module->symbols.count;

  EXPECT_TRUE(Run(module.get(), &specialization.function_versions));

  EXPECT_EQ(specialization.function_versions.list.count, 7u);
  EXPECT_EQ(module->symbols.count, source_symbol_count + 2);
  const loom_func_like_t left = Function(module.get(), IREE_SV("left"));
  const loom_func_like_t right = Function(module.get(), IREE_SV("right"));
  const loom_func_like_t wide = Function(module.get(), IREE_SV("wide"));
  const loom_symbol_ref_t middle32_ref = OnlySemanticCallee(module.get(), left);
  EXPECT_EQ(OnlySemanticCallee(module.get(), right).symbol_id,
            middle32_ref.symbol_id);
  const loom_symbol_ref_t middle64_ref = OnlySemanticCallee(module.get(), wide);
  EXPECT_NE(middle64_ref.symbol_id, middle32_ref.symbol_id);

  const loom_func_like_t middle32 = Function(module.get(), middle32_ref);
  const loom_func_like_t middle64 = Function(module.get(), middle64_ref);
  const loom_symbol_ref_t leaf32_ref =
      OnlySemanticCallee(module.get(), middle32);
  const loom_symbol_ref_t leaf64_ref =
      OnlySemanticCallee(module.get(), middle64);
  EXPECT_NE(leaf64_ref.symbol_id, leaf32_ref.symbol_id);
  EXPECT_TRUE(ContainsSubgroupQuery(Function(module.get(), leaf32_ref)));
  EXPECT_TRUE(ContainsSubgroupQuery(Function(module.get(), leaf64_ref)));

  const loom_target_function_version_t* left_version =
      Version(specialization.function_versions, left);
  const loom_target_function_version_t* right_version =
      Version(specialization.function_versions, right);
  const loom_target_function_version_t* wide_version =
      Version(specialization.function_versions, wide);
  const loom_target_function_version_t* middle32_version =
      Version(module.get(), specialization.function_versions, middle32_ref);
  const loom_target_function_version_t* middle64_version =
      Version(module.get(), specialization.function_versions, middle64_ref);
  const loom_target_function_version_t* leaf32_version =
      Version(module.get(), specialization.function_versions, leaf32_ref);
  const loom_target_function_version_t* leaf64_version =
      Version(module.get(), specialization.function_versions, leaf64_ref);
  ASSERT_NE(left_version, nullptr);
  ASSERT_NE(right_version, nullptr);
  ASSERT_NE(wide_version, nullptr);
  ASSERT_NE(middle32_version, nullptr);
  ASSERT_NE(middle64_version, nullptr);
  ASSERT_NE(leaf32_version, nullptr);
  ASSERT_NE(leaf64_version, nullptr);
  EXPECT_EQ(left_version->resolved_target.provider, &kTestProvider);
  EXPECT_EQ(right_version->resolved_target.provider, &kTestProvider);
  EXPECT_EQ(wide_version->resolved_target.provider, &kTestProvider);
  EXPECT_EQ(middle32_version->resolved_target.provider, &kTestProvider);
  EXPECT_EQ(middle64_version->resolved_target.provider, &kTestProvider);
  EXPECT_EQ(leaf32_version->resolved_target.provider, &kTestProvider);
  EXPECT_EQ(leaf64_version->resolved_target.provider, &kTestProvider);
  EXPECT_EQ(left_version->resolved_target.facts,
            right_version->resolved_target.facts);
  EXPECT_EQ(left_version->resolved_target.facts,
            middle32_version->resolved_target.facts);
  EXPECT_EQ(left_version->resolved_target.facts,
            leaf32_version->resolved_target.facts);
  EXPECT_EQ(wide_version->resolved_target.facts,
            middle64_version->resolved_target.facts);
  EXPECT_EQ(wide_version->resolved_target.facts,
            leaf64_version->resolved_target.facts);
  EXPECT_NE(left_version->resolved_target.facts,
            wide_version->resolved_target.facts);
  EXPECT_EQ(
      middle32_version->resolved_target.facts->storage.snapshot.subgroup_size,
      32u);
  EXPECT_EQ(
      middle64_version->resolved_target.facts->storage.snapshot.subgroup_size,
      64u);
  EXPECT_EQ(
      middle32_version->function_target_facts->storage.export_plan.abi_kind,
      LOOM_TARGET_ABI_UNKNOWN);
  EXPECT_EQ(
      middle64_version->function_target_facts->storage.export_plan.abi_kind,
      LOOM_TARGET_ABI_UNKNOWN);
  EXPECT_FALSE(loom_symbol_ref_is_valid(loom_func_like_target(middle32)));
  EXPECT_FALSE(loom_symbol_ref_is_valid(loom_func_like_target(middle64)));

  EXPECT_FALSE(Run(module.get(), &specialization.function_versions));
  EXPECT_EQ(specialization.function_versions.list.count, 7u);
  EXPECT_EQ(module->symbols.count, source_symbol_count + 2);
}

TEST_F(TargetCallgraphSpecializationTest,
       RetargetsRecursiveCallsToTheirConcreteVersion) {
  ModulePtr module = Parse(R"(
func.def @recursive() {
  func.call @recursive() : ()
  func.return
}

func.def public @wave32_root() {
  func.call @recursive() : ()
  func.return
}

func.def public @wave64_root() {
  func.call @recursive() : ()
  func.return
}
)");
  const TestTargetProfile wave32 = MakeTestProfile(32);
  const TestTargetProfile wave64 = MakeTestProfile(64);
  const loom_target_specialization_request_t requests[] = {
      {/*.function_name=*/IREE_SV("wave32_root"),
       /*.target_profile=*/&wave32.base},
      {/*.function_name=*/IREE_SV("wave64_root"),
       /*.target_profile=*/&wave64.base},
  };
  loom_target_specialization_result_t specialization =
      Specialize(module.get(), requests, IREE_ARRAYSIZE(requests));

  ASSERT_TRUE(Run(module.get(), &specialization.function_versions));
  ASSERT_EQ(specialization.function_versions.list.count, 4u);
  const loom_symbol_ref_t recursive32_ref = OnlySemanticCallee(
      module.get(), Function(module.get(), IREE_SV("wave32_root")));
  const loom_symbol_ref_t recursive64_ref = OnlySemanticCallee(
      module.get(), Function(module.get(), IREE_SV("wave64_root")));
  EXPECT_NE(recursive32_ref.symbol_id, recursive64_ref.symbol_id);
  const loom_symbol_ref_t recursive32_self =
      OnlySemanticCallee(module.get(), Function(module.get(), recursive32_ref));
  const loom_symbol_ref_t recursive64_self =
      OnlySemanticCallee(module.get(), Function(module.get(), recursive64_ref));
  EXPECT_EQ(recursive32_self.symbol_id, recursive32_ref.symbol_id);
  EXPECT_EQ(recursive64_self.symbol_id, recursive64_ref.symbol_id);
}

TEST_F(TargetCallgraphSpecializationTest,
       ReusesPublishedVersionWithAnAuthoredTargetRequirement) {
  ModulePtr module = Parse(R"(
test.target<low_core> @wave32 {subgroup_size = 32}

func.def target(@wave32) @helper() {
  func.return
}

func.def public @root() {
  func.call @helper() : ()
  func.return
}
)");
  const TestTargetProfile wave32 = MakeTestProfile(32);
  const loom_target_specialization_request_t request = {
      /*.function_name=*/IREE_SV("root"),
      /*.target_profile=*/&wave32.base,
  };
  loom_target_specialization_result_t specialization =
      Specialize(module.get(), &request, 1);

  ASSERT_TRUE(Run(module.get(), &specialization.function_versions));
  ASSERT_EQ(specialization.function_versions.list.count, 2u);
  const iree_host_size_t symbol_count = module->symbols.count;
  const loom_symbol_ref_t helper_ref =
      OnlySemanticCallee(module.get(), Function(module.get(), IREE_SV("root")));
  const loom_target_function_version_t* helper_version =
      Version(module.get(), specialization.function_versions, helper_ref);
  ASSERT_NE(helper_version, nullptr);
  ASSERT_NE(helper_version->target_requirement_facts, nullptr);
  EXPECT_EQ(
      helper_version->resolved_target.facts->storage.snapshot.subgroup_size,
      32u);

  EXPECT_FALSE(Run(module.get(), &specialization.function_versions));
  EXPECT_EQ(specialization.function_versions.list.count, 2u);
  EXPECT_EQ(module->symbols.count, symbol_count);
  EXPECT_EQ(
      OnlySemanticCallee(module.get(), Function(module.get(), IREE_SV("root")))
          .symbol_id,
      helper_ref.symbol_id);
}

TEST_F(TargetCallgraphSpecializationTest,
       RejectsMultipleContextsForExternallyReachableCalleeBeforeMutation) {
  ModulePtr module = Parse(R"(
func.def public @shared() {
  func.return
}

func.def public @wave32_root() {
  func.call @shared() : ()
  func.return
}

func.def public @wave64_root() {
  func.call @shared() : ()
  func.return
}
)");
  const TestTargetProfile wave32 = MakeTestProfile(32);
  const TestTargetProfile wave64 = MakeTestProfile(64);
  const loom_target_specialization_request_t requests[] = {
      {/*.function_name=*/IREE_SV("wave32_root"),
       /*.target_profile=*/&wave32.base},
      {/*.function_name=*/IREE_SV("wave64_root"),
       /*.target_profile=*/&wave64.base},
  };
  loom_target_specialization_result_t specialization =
      Specialize(module.get(), requests, IREE_ARRAYSIZE(requests));
  const iree_host_size_t symbol_count = module->symbols.count;
  const iree_host_size_t string_count = module->strings.count;
  DiagnosticCollector collector;

  EXPECT_FALSE(
      Run(module.get(), &specialization.function_versions, &collector));

  ASSERT_EQ(collector.errors.size(), 1u);
  EXPECT_EQ(collector.errors[0], LOOM_ERR_TARGET_069);
  ASSERT_EQ(collector.strings.size(), 1u);
  EXPECT_EQ(collector.strings[0], "shared");
  EXPECT_EQ(specialization.function_versions.list.count, 2u);
  EXPECT_EQ(module->symbols.count, symbol_count);
  EXPECT_EQ(module->strings.count, string_count);
  EXPECT_EQ(OnlySemanticCallee(module.get(),
                               Function(module.get(), IREE_SV("wave32_root")))
                .symbol_id,
            SymbolRef(module.get(), IREE_SV("shared")).symbol_id);
  EXPECT_EQ(OnlySemanticCallee(module.get(),
                               Function(module.get(), IREE_SV("wave64_root")))
                .symbol_id,
            SymbolRef(module.get(), IREE_SV("shared")).symbol_id);
}

TEST_F(TargetCallgraphSpecializationTest,
       RejectsIncompatibleAuthoredCalleeTargetBeforeMutation) {
  ModulePtr module = Parse(R"(
test.target<low_core> @wave64 {subgroup_size = 64}

func.def target(@wave64) @helper() {
  func.return
}

func.def public @wave32_root() {
  func.call @helper() : ()
  func.return
}
)");
  const TestTargetProfile wave32 = MakeTestProfile(32);
  const loom_target_specialization_request_t request = {
      /*.function_name=*/IREE_SV("wave32_root"),
      /*.target_profile=*/&wave32.base,
  };
  loom_target_specialization_result_t specialization =
      Specialize(module.get(), &request, 1);
  const iree_host_size_t symbol_count = module->symbols.count;
  const iree_host_size_t string_count = module->strings.count;
  DiagnosticCollector collector;

  EXPECT_FALSE(
      Run(module.get(), &specialization.function_versions, &collector));

  ASSERT_EQ(collector.errors.size(), 1u);
  EXPECT_EQ(collector.errors[0], LOOM_ERR_TARGET_052);
  ASSERT_EQ(collector.strings.size(), 3u);
  EXPECT_EQ(collector.strings[0], "helper");
  EXPECT_EQ(collector.strings[1], "wave64");
  EXPECT_EQ(specialization.function_versions.list.count, 1u);
  EXPECT_EQ(module->symbols.count, symbol_count);
  EXPECT_EQ(module->strings.count, string_count);
}

}  // namespace
}  // namespace loom
