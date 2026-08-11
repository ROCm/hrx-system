// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/specialization.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/error_catalog.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/target/provider.h"
#include "loom/target/test/target_records.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

typedef struct TestTargetProfile {
  // Generic target profile base.
  loom_target_profile_t base;

  // Test target selector projected into facts.
  loom_test_target_kind_t kind;

  // Optional counter incremented by each fact projection.
  uint32_t* projection_count;
} TestTargetProfile;

static iree_status_t ProjectTestProfileFacts(
    const loom_target_profile_t* base_profile, iree_arena_allocator_t* arena,
    loom_target_facts_t* out_facts) {
  (void)arena;
  const auto* profile =
      reinterpret_cast<const TestTargetProfile*>(base_profile);
  if (profile->projection_count != nullptr) {
    ++*profile->projection_count;
  }
  out_facts->selector = profile->kind;
  return iree_ok_status();
}

static const loom_target_profile_type_t kTestProfileType = {
    /*.name=*/IREE_SVL("specialization-test"),
    /*.fact_type=*/&loom_test_target_fact_type,
    /*.project_facts=*/ProjectTestProfileFacts,
};

static TestTargetProfile MakeTestProfile(loom_test_target_kind_t kind) {
  return TestTargetProfile{
      /*.base=*/
      {
          /*.type=*/&kTestProfileType,
          /*.target_bundle=*/
          loom_target_bundle_table_lookup(&loom_test_target_bundles, kind),
      },
      /*.kind=*/kind,
      /*.projection_count=*/nullptr,
  };
}

static const loom_target_provider_t kTestProvider = {
    /*.profile_type=*/&kTestProfileType,
    /*.materialize_definition=*/nullptr,
    /*.register_context=*/nullptr,
    /*.initialize_low_descriptor_registry=*/nullptr,
    /*.initialize_low_lower_policy_registry=*/nullptr,
    /*.initialize_math_policy_registry=*/nullptr,
    /*.low_legality_provider_list=*/{},
    /*.legalizer_provider_list=*/{},
    /*.low_packet_diagnostic_provider_list=*/{},
    /*.low_asm_diagnostic_provider_list=*/{},
    /*.low_verify_provider_list=*/{},
    /*.emitter_list=*/{},
    /*.pass_registry=*/nullptr,
    /*.contribute_pipeline=*/nullptr,
};

static const loom_target_provider_t* const kTestProviders[] = {
    &kTestProvider,
};

static const loom_target_provider_set_t kTestProviderSet =
    loom_target_provider_set_make(kTestProviders,
                                  IREE_ARRAYSIZE(kTestProviders));

struct DiagnosticCollector {
  const loom_error_def_t* error = nullptr;
  std::vector<std::string> strings;
};

static iree_status_t CollectDiagnostic(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  auto* collector = static_cast<DiagnosticCollector*>(user_data);
  collector->error = emission->error;
  collector->strings.clear();
  for (iree_host_size_t i = 0; i < emission->param_count; ++i) {
    if (emission->params[i].kind != LOOM_PARAM_STRING) {
      continue;
    }
    const iree_string_view_t value = emission->params[i].string;
    collector->strings.emplace_back(value.data, value.size);
  }
  return iree_ok_status();
}

class TargetSpecializationTest : public ::testing::Test {
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
    iree_arena_initialize(&block_pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
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
    IREE_CHECK_OK(loom_text_parse(iree_make_cstring_view(source),
                                  IREE_SV("target_specialization_test.loom"),
                                  &context_, &block_pool_, &options, &module));
    return ModulePtr(module);
  }

  loom_func_like_t Function(const loom_module_t* module,
                            iree_string_view_t name) {
    const loom_string_id_t name_id = loom_module_lookup_string(module, name);
    IREE_ASSERT(name_id != LOOM_STRING_ID_INVALID);
    const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
    IREE_ASSERT(symbol_id != LOOM_SYMBOL_ID_INVALID);
    loom_func_like_t function = loom_func_like_cast(
        module, module->symbols.entries[symbol_id].defining_op);
    IREE_ASSERT(loom_func_like_isa(function));
    return function;
  }

  loom_test_target_kind_t TargetKind(const loom_module_t* module,
                                     loom_func_like_t function) {
    const loom_symbol_ref_t target_ref = loom_func_like_target(function);
    IREE_ASSERT(loom_symbol_ref_is_valid(target_ref));
    IREE_ASSERT(target_ref.module_id == 0);
    const loom_op_t* target_op =
        module->symbols.entries[target_ref.symbol_id].defining_op;
    IREE_ASSERT(loom_test_target_isa(target_op));
    return loom_test_target_kind(target_op);
  }

  loom_target_specialization_result_t Specialize(
      loom_module_t* module,
      const loom_target_specialization_request_t* requests,
      iree_host_size_t request_count,
      DiagnosticCollector* diagnostic_collector = nullptr) {
    loom_target_specialization_result_t result = {};
    IREE_CHECK_OK(loom_target_specialize_functions(
        &environment_, module,
        {
            /*.values=*/requests,
            /*.count=*/request_count,
        },
        {
            /*.fn=*/diagnostic_collector ? CollectDiagnostic : nullptr,
            /*.user_data=*/diagnostic_collector,
        },
        &arena_, &result));
    return result;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_environment_t environment_;
  iree_arena_allocator_t arena_;
};

TEST_F(TargetSpecializationTest,
       ProducesFunctionVersionsWithoutChangingAuthoredTargets) {
  ModulePtr module = Parse(R"(
test.target<low_core> @family
test.target<quirky> @unrequested_family

func.def public target(@family) @generic() {
  func.return
}

func.def public @targetless() {
  func.return
}

func.def public target(@unrequested_family) @unrequested() {
  func.return
}
)");
  const TestTargetProfile exact_profile =
      MakeTestProfile(LOOM_TEST_TARGET_KIND_LOW_CORE);
  const loom_target_specialization_request_t requests[] = {
      {
          /*.function_name=*/IREE_SV("@generic"),
          /*.target_profile=*/&exact_profile.base,
      },
      {
          /*.function_name=*/IREE_SV("targetless"),
          /*.target_profile=*/&exact_profile.base,
      },
  };
  const loom_func_like_t generic = Function(module.get(), IREE_SV("generic"));
  const loom_func_like_t targetless =
      Function(module.get(), IREE_SV("targetless"));
  const loom_func_like_t unrequested =
      Function(module.get(), IREE_SV("unrequested"));
  const loom_symbol_ref_t generic_target = loom_func_like_target(generic);
  const loom_symbol_ref_t unrequested_target =
      loom_func_like_target(unrequested);
  const iree_host_size_t symbol_count = module->symbols.count;

  const loom_target_specialization_result_t result =
      Specialize(module.get(), requests, IREE_ARRAYSIZE(requests));
  ASSERT_EQ(result.error_count, 0u);
  ASSERT_EQ(result.function_versions.list.count, 2u);
  EXPECT_EQ(module->symbols.count, symbol_count);
  EXPECT_EQ(TargetKind(module.get(), generic), LOOM_TEST_TARGET_KIND_LOW_CORE);
  EXPECT_EQ(loom_func_like_target(generic).module_id, generic_target.module_id);
  EXPECT_EQ(loom_func_like_target(generic).symbol_id, generic_target.symbol_id);
  EXPECT_FALSE(loom_symbol_ref_is_valid(loom_func_like_target(targetless)));
  EXPECT_EQ(TargetKind(module.get(), unrequested),
            LOOM_TEST_TARGET_KIND_QUIRKY);
  EXPECT_EQ(loom_func_like_target(unrequested).module_id,
            unrequested_target.module_id);
  EXPECT_EQ(loom_func_like_target(unrequested).symbol_id,
            unrequested_target.symbol_id);

  const loom_target_function_version_t* generic_version =
      loom_target_function_version_list_find(&result.function_versions.list,
                                             generic);
  ASSERT_NE(generic_version, nullptr);
  EXPECT_EQ(generic_version->resolved_target.provider, &kTestProvider);
  ASSERT_NE(generic_version->resolved_target.facts, nullptr);
  ASSERT_NE(generic_version->target_requirement_facts, nullptr);
  EXPECT_EQ(generic_version->target_requirement_facts->selector,
            LOOM_TEST_TARGET_KIND_LOW_CORE);
  ASSERT_NE(generic_version->function_target_facts, nullptr);
  EXPECT_EQ(generic_version->function_target_facts->selector,
            LOOM_TEST_TARGET_KIND_LOW_CORE);

  const loom_target_function_version_t* targetless_version =
      loom_target_function_version_list_find(&result.function_versions.list,
                                             targetless);
  ASSERT_NE(targetless_version, nullptr);
  EXPECT_EQ(targetless_version->resolved_target.provider, &kTestProvider);
  ASSERT_NE(targetless_version->resolved_target.facts, nullptr);
  EXPECT_EQ(targetless_version->target_requirement_facts, nullptr);
  ASSERT_NE(targetless_version->function_target_facts, nullptr);
  EXPECT_EQ(targetless_version->function_target_facts->selector,
            LOOM_TEST_TARGET_KIND_LOW_CORE);
  EXPECT_EQ(loom_target_function_version_list_find(
                &result.function_versions.list, unrequested),
            nullptr);
}

TEST_F(TargetSpecializationTest, SharesProfileProjectionAndTargetlessContext) {
  ModulePtr module = Parse(R"(
func.def public @left() {
  func.return
}

func.def public @right() {
  func.return
}
)");
  uint32_t projection_count = 0;
  TestTargetProfile exact_profile =
      MakeTestProfile(LOOM_TEST_TARGET_KIND_LOW_CORE);
  exact_profile.projection_count = &projection_count;
  const loom_target_specialization_request_t requests[] = {
      {
          /*.function_name=*/IREE_SV("left"),
          /*.target_profile=*/&exact_profile.base,
      },
      {
          /*.function_name=*/IREE_SV("right"),
          /*.target_profile=*/&exact_profile.base,
      },
  };
  const loom_func_like_t left = Function(module.get(), IREE_SV("left"));
  const loom_func_like_t right = Function(module.get(), IREE_SV("right"));

  const loom_target_specialization_result_t result =
      Specialize(module.get(), requests, IREE_ARRAYSIZE(requests));
  ASSERT_EQ(result.error_count, 0u);
  ASSERT_EQ(result.function_versions.list.count, 2u);
  EXPECT_EQ(projection_count, 1u);
  const loom_target_function_version_t* left_version =
      loom_target_function_version_list_find(&result.function_versions.list,
                                             left);
  const loom_target_function_version_t* right_version =
      loom_target_function_version_list_find(&result.function_versions.list,
                                             right);
  ASSERT_NE(left_version, nullptr);
  ASSERT_NE(right_version, nullptr);
  EXPECT_EQ(left_version->resolved_target.facts,
            right_version->resolved_target.facts);
  EXPECT_NE(left_version->resolved_target.facts,
            left_version->function_target_facts);
  EXPECT_NE(right_version->resolved_target.facts,
            right_version->function_target_facts);
  EXPECT_NE(left_version->function_target_facts,
            right_version->function_target_facts);
}

TEST_F(TargetSpecializationTest, PreservesMatchingAuthoredExactTarget) {
  ModulePtr module = Parse(R"(
test.target<low_core> @exact

func.def public target(@exact) @entry() {
  func.return
}
)");
  const loom_func_like_t entry_before =
      Function(module.get(), IREE_SV("entry"));
  const loom_symbol_ref_t authored_ref = loom_func_like_target(entry_before);
  const TestTargetProfile exact_profile =
      MakeTestProfile(LOOM_TEST_TARGET_KIND_LOW_CORE);
  const loom_target_specialization_request_t request = {
      /*.function_name=*/IREE_SV("entry"),
      /*.target_profile=*/&exact_profile.base,
  };

  const loom_target_specialization_result_t result =
      Specialize(module.get(), &request, 1);
  ASSERT_EQ(result.error_count, 0u);
  ASSERT_EQ(result.function_versions.list.count, 1u);
  const loom_symbol_ref_t effective_ref =
      loom_func_like_target(Function(module.get(), IREE_SV("entry")));
  EXPECT_EQ(effective_ref.module_id, authored_ref.module_id);
  EXPECT_EQ(effective_ref.symbol_id, authored_ref.symbol_id);
  const loom_target_function_version_t* version =
      loom_target_function_version_const_cast(
          result.function_versions.list.values[0]);
  ASSERT_NE(version, nullptr);
  EXPECT_EQ(version->function_target_facts->selector,
            LOOM_TEST_TARGET_KIND_LOW_CORE);
}

TEST_F(TargetSpecializationTest,
       RefinesDistinctAuthoredFunctionFactsWithoutCreatingTargetRecords) {
  ModulePtr module = Parse(R"(
test.target<low_core> @left_requirement {
  abi = hal_kernel,
  linkage = dso_local
}
test.target<low_core> @right_requirement {
  abi = object_function,
  linkage = default
}

func.def public target(@left_requirement) export("left_kernel") @left() {
  func.return
}

func.def public target(@right_requirement) export("right_function") @right() {
  func.return
}
)");
  const TestTargetProfile exact_profile =
      MakeTestProfile(LOOM_TEST_TARGET_KIND_LOW_CORE);
  const loom_target_specialization_request_t requests[] = {
      {
          /*.function_name=*/IREE_SV("left"),
          /*.target_profile=*/&exact_profile.base,
      },
      {
          /*.function_name=*/IREE_SV("right"),
          /*.target_profile=*/&exact_profile.base,
      },
  };
  const loom_func_like_t left = Function(module.get(), IREE_SV("left"));
  const loom_func_like_t right = Function(module.get(), IREE_SV("right"));
  const loom_symbol_ref_t left_target = loom_func_like_target(left);
  const loom_symbol_ref_t right_target = loom_func_like_target(right);
  const iree_host_size_t symbol_count = module->symbols.count;

  const loom_target_specialization_result_t result =
      Specialize(module.get(), requests, IREE_ARRAYSIZE(requests));
  ASSERT_EQ(result.error_count, 0u);
  ASSERT_EQ(result.function_versions.list.count, 2u);
  EXPECT_EQ(module->symbols.count, symbol_count);
  EXPECT_EQ(loom_func_like_target(left).module_id, left_target.module_id);
  EXPECT_EQ(loom_func_like_target(left).symbol_id, left_target.symbol_id);
  EXPECT_EQ(loom_func_like_target(right).module_id, right_target.module_id);
  EXPECT_EQ(loom_func_like_target(right).symbol_id, right_target.symbol_id);
  const loom_target_function_version_t* left_version =
      loom_target_function_version_list_find(&result.function_versions.list,
                                             left);
  const loom_target_function_version_t* right_version =
      loom_target_function_version_list_find(&result.function_versions.list,
                                             right);
  ASSERT_NE(left_version, nullptr);
  ASSERT_NE(right_version, nullptr);
  const loom_target_facts_t* left_version_facts =
      left_version->function_target_facts;
  const loom_target_facts_t* right_version_facts =
      right_version->function_target_facts;
  ASSERT_NE(left_version_facts, nullptr);
  ASSERT_NE(right_version_facts, nullptr);
  EXPECT_EQ(left_version_facts->storage.export_plan.abi_kind,
            LOOM_TARGET_ABI_HAL_KERNEL);
  EXPECT_EQ(right_version_facts->storage.export_plan.abi_kind,
            LOOM_TARGET_ABI_OBJECT_FUNCTION);
  EXPECT_TRUE(iree_string_view_equal(
      left_version_facts->storage.export_plan.export_symbol,
      IREE_SV("left_kernel")));
  EXPECT_TRUE(iree_string_view_equal(
      right_version_facts->storage.export_plan.export_symbol,
      IREE_SV("right_function")));
  EXPECT_EQ(left_version_facts->storage.export_plan.linkage,
            LOOM_TARGET_LINKAGE_DSO_LOCAL);
  EXPECT_EQ(right_version_facts->storage.export_plan.linkage,
            LOOM_TARGET_LINKAGE_DEFAULT);
}

TEST_F(TargetSpecializationTest,
       ConflictLeavesEveryRequestedFunctionTargetUnchanged) {
  ModulePtr module = Parse(R"(
test.target<low_core> @exact

func.def public target(@exact) @conflict() {
  func.return
}

func.def public @otherwise_compatible() {
  func.return
}
)");
  const TestTargetProfile incompatible_profile =
      MakeTestProfile(LOOM_TEST_TARGET_KIND_QUIRKY);
  const TestTargetProfile exact_profile =
      MakeTestProfile(LOOM_TEST_TARGET_KIND_LOW_CORE);
  const loom_target_specialization_request_t requests[] = {
      {
          /*.function_name=*/IREE_SV("conflict"),
          /*.target_profile=*/&incompatible_profile.base,
      },
      {
          /*.function_name=*/IREE_SV("otherwise_compatible"),
          /*.target_profile=*/&exact_profile.base,
      },
  };
  const loom_symbol_ref_t authored_ref =
      loom_func_like_target(Function(module.get(), IREE_SV("conflict")));
  DiagnosticCollector diagnostic_collector;

  const loom_target_specialization_result_t result = Specialize(
      module.get(), requests, IREE_ARRAYSIZE(requests), &diagnostic_collector);
  ASSERT_EQ(result.error_count, 1u);
  EXPECT_EQ(result.function_versions.list.count, 0u);
  EXPECT_EQ(diagnostic_collector.error, LOOM_ERR_TARGET_052);
  ASSERT_EQ(diagnostic_collector.strings.size(), 3u);
  EXPECT_EQ(diagnostic_collector.strings[0], "conflict");
  EXPECT_EQ(diagnostic_collector.strings[1], "exact");
  EXPECT_EQ(diagnostic_collector.strings[2], "test-quirky");
  const loom_symbol_ref_t conflict_ref =
      loom_func_like_target(Function(module.get(), IREE_SV("conflict")));
  EXPECT_EQ(conflict_ref.module_id, authored_ref.module_id);
  EXPECT_EQ(conflict_ref.symbol_id, authored_ref.symbol_id);
  EXPECT_FALSE(loom_symbol_ref_is_valid(loom_func_like_target(
      Function(module.get(), IREE_SV("otherwise_compatible")))));
}

TEST_F(TargetSpecializationTest, RejectsUnlinkedProfileFamily) {
  ModulePtr module = Parse(R"(
func.def public @entry() {
  func.return
}
)");
  static const loom_target_profile_type_t kUnlinkedProfileType = {
      /*.name=*/IREE_SVL("unlinked"),
      /*.fact_type=*/&loom_test_target_fact_type,
      /*.project_facts=*/ProjectTestProfileFacts,
  };
  TestTargetProfile profile = MakeTestProfile(LOOM_TEST_TARGET_KIND_LOW_CORE);
  profile.base.type = &kUnlinkedProfileType;
  const loom_target_specialization_request_t request = {
      /*.function_name=*/IREE_SV("entry"),
      /*.target_profile=*/&profile.base,
  };
  loom_target_specialization_result_t result = {};

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_target_specialize_functions(
                            &environment_, module.get(),
                            {
                                /*.values=*/&request,
                                /*.count=*/1,
                            },
                            /*diagnostic_emitter=*/{}, &arena_, &result));
  EXPECT_EQ(result.function_versions.list.count, 0u);
}

TEST_F(TargetSpecializationTest,
       SpecializesAuthoredTargetDeclarationFromProfile) {
  ModulePtr module = Parse(R"(
target.decl @external

func.def public target(@external) @entry() {
  func.return
}
)");
  const TestTargetProfile exact_profile =
      MakeTestProfile(LOOM_TEST_TARGET_KIND_LOW_CORE);
  const loom_target_specialization_request_t request = {
      /*.function_name=*/IREE_SV("entry"),
      /*.target_profile=*/&exact_profile.base,
  };
  const loom_func_like_t entry = Function(module.get(), IREE_SV("entry"));
  const loom_symbol_ref_t authored_target_ref = loom_func_like_target(entry);
  const iree_host_size_t symbol_count = module->symbols.count;

  const loom_target_specialization_result_t result =
      Specialize(module.get(), &request, 1);
  ASSERT_EQ(result.error_count, 0u);
  ASSERT_EQ(result.function_versions.list.count, 1u);
  EXPECT_EQ(module->symbols.count, symbol_count);
  EXPECT_EQ(loom_func_like_target(entry).module_id,
            authored_target_ref.module_id);
  EXPECT_EQ(loom_func_like_target(entry).symbol_id,
            authored_target_ref.symbol_id);
  const loom_target_function_version_t* version =
      loom_target_function_version_list_find(&result.function_versions.list,
                                             entry);
  ASSERT_NE(version, nullptr);
  EXPECT_TRUE(iree_string_view_equal(version->authored_target_name,
                                     IREE_SV("external")));
  EXPECT_EQ(version->target_requirement_facts, nullptr);
  ASSERT_NE(version->function_target_facts, nullptr);
  EXPECT_EQ(version->function_target_facts->selector,
            LOOM_TEST_TARGET_KIND_LOW_CORE);
}

TEST_F(TargetSpecializationTest, RejectsMissingAndDuplicateFunctions) {
  ModulePtr module = Parse(R"(
func.def public @entry() {
  func.return
}
)");
  const TestTargetProfile exact_profile =
      MakeTestProfile(LOOM_TEST_TARGET_KIND_LOW_CORE);
  const loom_target_specialization_request_t missing_request = {
      /*.function_name=*/IREE_SV("missing"),
      /*.target_profile=*/&exact_profile.base,
  };
  loom_target_specialization_result_t result = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        loom_target_specialize_functions(
                            &environment_, module.get(),
                            {
                                /*.values=*/&missing_request,
                                /*.count=*/1,
                            },
                            /*diagnostic_emitter=*/{}, &arena_, &result));

  const loom_target_specialization_request_t duplicate_requests[] = {
      {
          /*.function_name=*/IREE_SV("entry"),
          /*.target_profile=*/&exact_profile.base,
      },
      {
          /*.function_name=*/IREE_SV("@entry"),
          /*.target_profile=*/&exact_profile.base,
      },
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_target_specialize_functions(
                            &environment_, module.get(),
                            {
                                /*.values=*/duplicate_requests,
                                /*.count=*/IREE_ARRAYSIZE(duplicate_requests),
                            },
                            /*diagnostic_emitter=*/{}, &arena_, &result));
}

}  // namespace
}  // namespace loom
