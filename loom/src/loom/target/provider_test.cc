// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/provider.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_registry.h"
#include "loom/pass/builder.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

const loom_pass_info_t* TargetAlphaPassInfo(void) {
  static const loom_pass_info_t kInfo = {
      /*.name=*/IREE_SVL("target-alpha"),
      /*.description=*/IREE_SVL("Target alpha pass."),
      /*.kind=*/LOOM_PASS_FUNCTION,
  };
  return &kInfo;
}

const loom_pass_info_t* TargetBetaPassInfo(void) {
  static const loom_pass_info_t kInfo = {
      /*.name=*/IREE_SVL("target-beta"),
      /*.description=*/IREE_SVL("Target beta pass."),
      /*.kind=*/LOOM_PASS_FUNCTION,
  };
  return &kInfo;
}

iree_status_t NoopFunctionPass(loom_pass_t* pass, loom_module_t* module,
                               loom_func_like_t function) {
  return iree_ok_status();
}

iree_status_t EmitNothing(const loom_target_emit_request_t* request,
                          bool* out_emitted,
                          loom_target_emit_artifact_t* out_artifact) {
  (void)request;
  *out_emitted = false;
  *out_artifact = {};
  return iree_ok_status();
}

static loom_pass_descriptor_t MakeFunctionPassDescriptor(
    iree_string_view_t key, loom_pass_info_fn_t info) {
  loom_pass_descriptor_t descriptor = {};
  descriptor.key = key;
  descriptor.info = info;
  descriptor.function_run = NoopFunctionPass;
  return descriptor;
}

static iree_status_t ContributeMaterialization(
    const loom_target_pipeline_contribution_t* contribution) {
  if (contribution->phase !=
      LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_MATERIALIZATION) {
    return iree_ok_status();
  }
  loom_op_t* run_op = nullptr;
  return loom_pass_ir_build_run(contribution->builder, 0,
                                IREE_SV("target-materialize"),
                                loom_named_attr_slice_empty(), &run_op);
}

static iree_status_t ContributePreparation(
    const loom_target_pipeline_contribution_t* contribution) {
  if (contribution->phase !=
      LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_PREPARATION) {
    return iree_ok_status();
  }
  loom_op_t* run_op = nullptr;
  return loom_pass_ir_build_run(contribution->builder, 0,
                                IREE_SV("target-prepare"),
                                loom_named_attr_slice_empty(), &run_op);
}

struct PipelineBuildData {
  const loom_target_environment_t* environment;
};

static iree_status_t BuildContributedPipeline(loom_builder_t* builder,
                                              void* user_data) {
  const PipelineBuildData* data =
      static_cast<const PipelineBuildData*>(user_data);
  IREE_RETURN_IF_ERROR(loom_target_environment_contribute_pipeline(
      data->environment, LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_MATERIALIZATION,
      loom_pass_environment_empty(), builder));
  loom_op_t* run_op = nullptr;
  IREE_RETURN_IF_ERROR(
      loom_pass_ir_build_run(builder, 0, IREE_SV("driver-cleanup"),
                             loom_named_attr_slice_empty(), &run_op));
  return loom_target_environment_contribute_pipeline(
      data->environment, LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_PREPARATION,
      loom_pass_environment_empty(), builder);
}

TEST(TargetProviderSetStorageTest, AppendsProvidersInExactOrder) {
  static const loom_target_provider_t first_provider = {};
  static const loom_target_provider_t second_provider = {};
  static const loom_target_provider_t* const initial_providers[] = {
      &first_provider,
      &second_provider,
  };
  const loom_target_provider_set_t initial_provider_set =
      loom_target_provider_set_make(initial_providers,
                                    IREE_ARRAYSIZE(initial_providers));
  loom_target_provider_set_storage_t storage;
  loom_target_provider_set_storage_initialize(&storage);
  const loom_target_provider_set_t* provider_set = &storage.provider_set;

  IREE_ASSERT_OK(loom_target_provider_set_storage_append_set(
      &storage, &initial_provider_set));
  IREE_ASSERT_OK(loom_target_provider_set_storage_append(&storage, nullptr));
  IREE_ASSERT_OK(
      loom_target_provider_set_storage_append(&storage, &second_provider));
  IREE_ASSERT_OK(loom_target_provider_set_storage_append_set(
      &storage, &storage.provider_set));

  EXPECT_EQ(provider_set->providers, storage.providers);
  ASSERT_EQ(provider_set->provider_count, 6u);
  EXPECT_EQ(provider_set->providers[0], &first_provider);
  EXPECT_EQ(provider_set->providers[1], &second_provider);
  EXPECT_EQ(provider_set->providers[2], &second_provider);
  EXPECT_EQ(provider_set->providers[3], &first_provider);
  EXPECT_EQ(provider_set->providers[4], &second_provider);
  EXPECT_EQ(provider_set->providers[5], &second_provider);
}

TEST(TargetProviderSetStorageTest, CapacityFailureDoesNotPartiallyAppend) {
  static const loom_target_provider_t provider = {};
  loom_target_provider_set_storage_t storage;
  loom_target_provider_set_storage_initialize(&storage);
  for (iree_host_size_t i = 0;
       i < LOOM_TARGET_PROVIDER_SET_STORAGE_CAPACITY - 1; ++i) {
    IREE_ASSERT_OK(
        loom_target_provider_set_storage_append(&storage, &provider));
  }
  static const loom_target_provider_t* const providers[] = {
      &provider,
      &provider,
  };
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_target_provider_set_storage_append_set(&storage, &provider_set));
  EXPECT_EQ(storage.provider_set.provider_count,
            LOOM_TARGET_PROVIDER_SET_STORAGE_CAPACITY - 1);

  IREE_ASSERT_OK(loom_target_provider_set_storage_append(&storage, &provider));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_target_provider_set_storage_append(&storage, &provider));
  EXPECT_EQ(storage.provider_set.provider_count,
            LOOM_TARGET_PROVIDER_SET_STORAGE_CAPACITY);
}

TEST(TargetEnvironmentTest, RejectsMalformedProviderTables) {
  loom_target_environment_t environment = {};
  const loom_target_provider_set_t missing_provider_storage =
      loom_target_provider_set_make(nullptr, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_target_environment_initialize(
                            &missing_provider_storage, &environment));

  const loom_target_provider_t* null_providers[] = {nullptr};
  const loom_target_provider_set_t null_provider_set =
      loom_target_provider_set_make(null_providers, 1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_environment_initialize(&null_provider_set, &environment));

  loom_target_provider_t provider = {};
  provider.emitter_list.count = 1;
  const loom_target_provider_t* providers[] = {&provider};
  const loom_target_provider_set_t missing_emitter_storage =
      loom_target_provider_set_make(providers, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_target_environment_initialize(
                            &missing_emitter_storage, &environment));

  const loom_target_emitter_t* null_emitters[] = {nullptr};
  provider.emitter_list = loom_target_emitter_list_make(null_emitters, 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_target_environment_initialize(
                            &missing_emitter_storage, &environment));
}

class TargetProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_status_t AllocateModule(iree_string_view_t name, ModulePtr* out_module) {
    *out_module = nullptr;
    loom_module_t* module = nullptr;
    IREE_RETURN_IF_ERROR(loom_module_allocate(&context_, name, &block_pool_,
                                              nullptr, iree_allocator_system(),
                                              &module));
    *out_module = ModulePtr(module);
    return iree_ok_status();
  }

  iree_string_view_t RunKey(loom_module_t* module, const loom_op_t* op) {
    return loom_string_table_get(&module->strings, loom_pass_run_key(op));
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(TargetProviderTest, ContributesPassIrByPhase) {
  static const loom_target_provider_t materialization_provider = {
      /*.profile_type=*/{},
      /*.materialize_definition=*/{},
      /*.register_context=*/{},
      /*.initialize_low_descriptor_registry=*/{},
      /*.initialize_low_lower_policy_registry=*/{},
      /*.initialize_math_policy_registry=*/{},
      /*.low_legality_provider_list=*/{},
      /*.legalizer_provider_list=*/{},
      /*.low_packet_diagnostic_provider_list=*/{},
      /*.low_asm_diagnostic_provider_list=*/{},
      /*.low_verify_provider_list=*/{},
      /*.emitter_list=*/{},
      /*.canonical_module_emitter=*/nullptr,
      /*.pass_registry=*/{},
      /*.contribute_pipeline=*/ContributeMaterialization,
  };
  static const loom_target_provider_t preparation_provider = {
      /*.profile_type=*/{},
      /*.materialize_definition=*/{},
      /*.register_context=*/{},
      /*.initialize_low_descriptor_registry=*/{},
      /*.initialize_low_lower_policy_registry=*/{},
      /*.initialize_math_policy_registry=*/{},
      /*.low_legality_provider_list=*/{},
      /*.legalizer_provider_list=*/{},
      /*.low_packet_diagnostic_provider_list=*/{},
      /*.low_asm_diagnostic_provider_list=*/{},
      /*.low_verify_provider_list=*/{},
      /*.emitter_list=*/{},
      /*.canonical_module_emitter=*/nullptr,
      /*.pass_registry=*/{},
      /*.contribute_pipeline=*/ContributePreparation,
  };
  static const loom_target_provider_t* const providers[] = {
      &materialization_provider,
      &preparation_provider,
  };
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {0};
  IREE_ASSERT_OK(
      loom_target_environment_initialize(&provider_set, &environment));

  ModulePtr module;
  IREE_ASSERT_OK(AllocateModule(IREE_SV("pipeline"), &module));
  PipelineBuildData build_data = {
      /*.environment=*/&environment,
  };
  loom_op_t* pipeline_op = nullptr;
  IREE_ASSERT_OK(loom_pass_ir_build_pipeline(
      module.get(), IREE_SV("compile"), LOOM_PASS_ANCHOR_MODULE,
      BuildContributedPipeline, &build_data, &pipeline_op));

  loom_block_t* pipeline_body =
      loom_region_entry_block(loom_pass_pipeline_body(pipeline_op));
  ASSERT_NE(pipeline_body, nullptr);
  ASSERT_EQ(pipeline_body->op_count, 4u);

  loom_op_t* materialize_run = pipeline_body->first_op;
  ASSERT_TRUE(loom_pass_run_isa(materialize_run));
  EXPECT_TRUE(iree_string_view_equal(RunKey(module.get(), materialize_run),
                                     IREE_SV("target-materialize")));

  loom_op_t* cleanup_run = materialize_run->next_op;
  ASSERT_TRUE(loom_pass_run_isa(cleanup_run));
  EXPECT_TRUE(iree_string_view_equal(RunKey(module.get(), cleanup_run),
                                     IREE_SV("driver-cleanup")));

  loom_op_t* prepare_run = cleanup_run->next_op;
  ASSERT_TRUE(loom_pass_run_isa(prepare_run));
  EXPECT_TRUE(iree_string_view_equal(RunKey(module.get(), prepare_run),
                                     IREE_SV("target-prepare")));
  EXPECT_TRUE(loom_pass_yield_isa(pipeline_body->last_op));

  loom_target_environment_deinitialize(&environment);
}

TEST_F(TargetProviderTest, ComposesTargetPassRegistries) {
  static const loom_pass_descriptor_t first_descriptors[] = {
      MakeFunctionPassDescriptor(IREE_SV("target-beta"), TargetBetaPassInfo),
  };
  static const loom_pass_descriptor_t second_descriptors[] = {
      MakeFunctionPassDescriptor(IREE_SV("target-alpha"), TargetAlphaPassInfo),
  };
  static const loom_pass_registry_t first_registry = {
      /*.descriptors=*/first_descriptors,
      /*.descriptor_count=*/IREE_ARRAYSIZE(first_descriptors),
  };
  static const loom_pass_registry_t second_registry = {
      /*.descriptors=*/second_descriptors,
      /*.descriptor_count=*/IREE_ARRAYSIZE(second_descriptors),
  };
  static const loom_target_provider_t first_provider = {
      /*.profile_type=*/{},
      /*.materialize_definition=*/{},
      /*.register_context=*/{},
      /*.initialize_low_descriptor_registry=*/{},
      /*.initialize_low_lower_policy_registry=*/{},
      /*.initialize_math_policy_registry=*/{},
      /*.low_legality_provider_list=*/{},
      /*.legalizer_provider_list=*/{},
      /*.low_packet_diagnostic_provider_list=*/{},
      /*.low_asm_diagnostic_provider_list=*/{},
      /*.low_verify_provider_list=*/{},
      /*.emitter_list=*/{},
      /*.canonical_module_emitter=*/nullptr,
      /*.pass_registry=*/&first_registry,
  };
  static const loom_target_provider_t second_provider = {
      /*.profile_type=*/{},
      /*.materialize_definition=*/{},
      /*.register_context=*/{},
      /*.initialize_low_descriptor_registry=*/{},
      /*.initialize_low_lower_policy_registry=*/{},
      /*.initialize_math_policy_registry=*/{},
      /*.low_legality_provider_list=*/{},
      /*.legalizer_provider_list=*/{},
      /*.low_packet_diagnostic_provider_list=*/{},
      /*.low_asm_diagnostic_provider_list=*/{},
      /*.low_verify_provider_list=*/{},
      /*.emitter_list=*/{},
      /*.canonical_module_emitter=*/nullptr,
      /*.pass_registry=*/&second_registry,
  };
  static const loom_target_provider_t* const providers[] = {
      &first_provider,
      &second_provider,
  };
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {0};
  IREE_ASSERT_OK(
      loom_target_environment_initialize(&provider_set, &environment));

  const loom_pass_registry_t* registry =
      loom_target_environment_pass_registry(&environment);
  ASSERT_EQ(registry->descriptor_count, 2u);
  EXPECT_TRUE(iree_string_view_equal(registry->descriptors[0].key,
                                     IREE_SV("target-alpha")));
  EXPECT_TRUE(iree_string_view_equal(registry->descriptors[1].key,
                                     IREE_SV("target-beta")));

  const loom_pass_descriptor_t* descriptor = nullptr;
  IREE_ASSERT_OK(
      loom_pass_registry_lookup(registry, IREE_SV("target-beta"), &descriptor));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->info, &TargetBetaPassInfo);

  loom_target_environment_deinitialize(&environment);
}

TEST_F(TargetProviderTest, LooksUpProfileProvider) {
  static const loom_target_fact_type_t kOwnedFactType = {};
  static const loom_target_fact_type_t kUnownedFactType = {};
  static const loom_target_profile_type_t kOwnedProfileType = {
      /*.name=*/IREE_SVL("owned"),
      /*.fact_type=*/&kOwnedFactType,
  };
  static const loom_target_profile_type_t kUnownedProfileType = {
      /*.name=*/IREE_SVL("unowned"),
      /*.fact_type=*/&kUnownedFactType,
  };
  static const loom_target_provider_t provider = {
      /*.profile_type=*/&kOwnedProfileType,
  };
  static const loom_target_provider_t* const providers[] = {
      &provider,
  };
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {};
  IREE_ASSERT_OK(
      loom_target_environment_initialize(&provider_set, &environment));

  EXPECT_EQ(loom_target_environment_lookup_profile_provider(&environment,
                                                            &kOwnedProfileType),
            &provider);
  EXPECT_EQ(loom_target_environment_lookup_profile_provider(
                &environment, &kUnownedProfileType),
            nullptr);
  EXPECT_EQ(loom_target_environment_lookup_fact_provider(&environment,
                                                         &kOwnedFactType),
            &provider);
  EXPECT_EQ(loom_target_environment_lookup_fact_provider(&environment,
                                                         &kUnownedFactType),
            nullptr);

  loom_target_environment_deinitialize(&environment);
}

TEST_F(TargetProviderTest, LooksUpProviderWithoutProfileByFactType) {
  static const loom_target_fact_type_t kOwnedFactType = {};
  static const loom_target_fact_type_t kUnownedFactType = {};
  loom_target_provider_t provider = {};
  provider.target_fact_type = &kOwnedFactType;
  const loom_target_provider_t* providers[] = {&provider};
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {};
  IREE_ASSERT_OK(
      loom_target_environment_initialize(&provider_set, &environment));

  EXPECT_EQ(loom_target_environment_lookup_fact_provider(&environment,
                                                         &kOwnedFactType),
            &provider);
  EXPECT_EQ(loom_target_environment_lookup_fact_provider(&environment,
                                                         &kUnownedFactType),
            nullptr);

  loom_target_environment_deinitialize(&environment);
}

TEST_F(TargetProviderTest, ComposesCanonicalModuleEmitterByFactType) {
  static const loom_target_fact_type_t kFactType = {};
  static const loom_target_emitter_t kEmitter = {
      /*.name=*/IREE_SVL("module-emitter"),
      /*.public_artifact_format=*/IREE_SVL("module-format"),
      /*.default_identifier=*/{},
      /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN,
      /*.default_pipeline_options=*/{},
      /*.emit=*/EmitNothing,
  };
  static const loom_target_emitter_t* const kEmitters[] = {&kEmitter};
  loom_target_provider_t target_provider = {};
  target_provider.target_fact_type = &kFactType;
  loom_target_provider_t emission_provider = {};
  emission_provider.emitter_list =
      loom_target_emitter_list_make(kEmitters, IREE_ARRAYSIZE(kEmitters));
  emission_provider.canonical_module_emitter = &kEmitter;
  emission_provider.canonical_module_fact_type = &kFactType;
  const loom_target_provider_t* providers[] = {
      &target_provider,
      &emission_provider,
  };
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {};
  IREE_ASSERT_OK(
      loom_target_environment_initialize(&provider_set, &environment));

  EXPECT_EQ(
      loom_target_environment_lookup_fact_provider(&environment, &kFactType),
      &target_provider);
  EXPECT_EQ(loom_target_environment_lookup_canonical_module_emitter(
                &environment, &kFactType),
            &kEmitter);
  EXPECT_EQ(loom_target_environment_lookup_emitter(&environment,
                                                   IREE_SV("module-format")),
            &kEmitter);
  EXPECT_EQ(loom_target_environment_lookup_emitter(&environment,
                                                   IREE_SV("missing-format")),
            nullptr);

  loom_target_environment_deinitialize(&environment);
}

TEST_F(TargetProviderTest, ComposesCanonicalKernelEmitterByFactType) {
  static const loom_target_fact_type_t kFactType = {};
  static const loom_target_emitter_t kEmitter = {
      /*.name=*/IREE_SVL("kernel-emitter"),
      /*.public_artifact_format=*/IREE_SVL("kernel-format"),
      /*.default_identifier=*/{},
      /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN,
      /*.default_pipeline_options=*/{},
      /*.emit=*/EmitNothing,
  };
  static const loom_target_emitter_t* const kEmitters[] = {&kEmitter};
  loom_target_provider_t target_provider = {};
  target_provider.target_fact_type = &kFactType;
  loom_target_provider_t emission_provider = {};
  emission_provider.emitter_list =
      loom_target_emitter_list_make(kEmitters, IREE_ARRAYSIZE(kEmitters));
  emission_provider.canonical_kernel_emitter = &kEmitter;
  emission_provider.canonical_kernel_fact_type = &kFactType;
  const loom_target_provider_t* providers[] = {
      &target_provider,
      &emission_provider,
  };
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {};
  IREE_ASSERT_OK(
      loom_target_environment_initialize(&provider_set, &environment));

  EXPECT_EQ(
      loom_target_environment_lookup_fact_provider(&environment, &kFactType),
      &target_provider);
  EXPECT_EQ(loom_target_environment_lookup_canonical_kernel_emitter(
                &environment, &kFactType),
            &kEmitter);

  loom_target_environment_deinitialize(&environment);
}

TEST_F(TargetProviderTest, RejectsCanonicalModuleEmitterWithoutFactType) {
  static const loom_target_emitter_t kEmitter = {
      /*.name=*/IREE_SVL("module-emitter"),
      /*.public_artifact_format=*/IREE_SVL("module-format"),
      /*.default_identifier=*/{},
      /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN,
      /*.default_pipeline_options=*/{},
      /*.emit=*/EmitNothing,
  };
  static const loom_target_emitter_t* const kEmitters[] = {&kEmitter};
  loom_target_provider_t provider = {};
  provider.emitter_list =
      loom_target_emitter_list_make(kEmitters, IREE_ARRAYSIZE(kEmitters));
  provider.canonical_module_emitter = &kEmitter;
  const loom_target_provider_t* providers[] = {&provider};
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_environment_initialize(&provider_set, &environment));
}

TEST_F(TargetProviderTest, RejectsCanonicalModuleFactTypeWithoutEmitter) {
  static const loom_target_fact_type_t kFactType = {};
  loom_target_provider_t provider = {};
  provider.canonical_module_fact_type = &kFactType;
  const loom_target_provider_t* providers[] = {&provider};
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_environment_initialize(&provider_set, &environment));
}

TEST_F(TargetProviderTest, RejectsUncontributedCanonicalModuleEmitter) {
  static const loom_target_fact_type_t kFactType = {};
  static const loom_target_emitter_t kEmitter = {
      /*.name=*/IREE_SVL("module-emitter"),
      /*.public_artifact_format=*/IREE_SVL("module-format"),
      /*.default_identifier=*/{},
      /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN,
      /*.default_pipeline_options=*/{},
      /*.emit=*/EmitNothing,
  };
  loom_target_provider_t provider = {};
  provider.canonical_module_emitter = &kEmitter;
  provider.canonical_module_fact_type = &kFactType;
  const loom_target_provider_t* providers[] = {&provider};
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_environment_initialize(&provider_set, &environment));
}

TEST_F(TargetProviderTest, RejectsDuplicateCanonicalModuleEmitter) {
  static const loom_target_fact_type_t kFactType = {};
  static const loom_target_emitter_t kFirstEmitter = {
      /*.name=*/IREE_SVL("first-module-emitter"),
      /*.public_artifact_format=*/IREE_SVL("first-module-format"),
      /*.default_identifier=*/{},
      /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN,
      /*.default_pipeline_options=*/{},
      /*.emit=*/EmitNothing,
  };
  static const loom_target_emitter_t kSecondEmitter = {
      /*.name=*/IREE_SVL("second-module-emitter"),
      /*.public_artifact_format=*/IREE_SVL("second-module-format"),
      /*.default_identifier=*/{},
      /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN,
      /*.default_pipeline_options=*/{},
      /*.emit=*/EmitNothing,
  };
  static const loom_target_emitter_t* const kFirstEmitters[] = {
      &kFirstEmitter,
  };
  static const loom_target_emitter_t* const kSecondEmitters[] = {
      &kSecondEmitter,
  };
  loom_target_provider_t first_provider = {};
  first_provider.emitter_list = loom_target_emitter_list_make(
      kFirstEmitters, IREE_ARRAYSIZE(kFirstEmitters));
  first_provider.canonical_module_emitter = &kFirstEmitter;
  first_provider.canonical_module_fact_type = &kFactType;
  loom_target_provider_t second_provider = {};
  second_provider.emitter_list = loom_target_emitter_list_make(
      kSecondEmitters, IREE_ARRAYSIZE(kSecondEmitters));
  second_provider.canonical_module_emitter = &kSecondEmitter;
  second_provider.canonical_module_fact_type = &kFactType;
  const loom_target_provider_t* providers[] = {
      &first_provider,
      &second_provider,
  };
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_environment_initialize(&provider_set, &environment));
}

TEST_F(TargetProviderTest, RejectsDuplicateCanonicalKernelEmitter) {
  static const loom_target_fact_type_t kFactType = {};
  static const loom_target_emitter_t kFirstEmitter = {
      /*.name=*/IREE_SVL("first-kernel-emitter"),
      /*.public_artifact_format=*/IREE_SVL("first-kernel-format"),
      /*.default_identifier=*/{},
      /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN,
      /*.default_pipeline_options=*/{},
      /*.emit=*/EmitNothing,
  };
  static const loom_target_emitter_t kSecondEmitter = {
      /*.name=*/IREE_SVL("second-kernel-emitter"),
      /*.public_artifact_format=*/IREE_SVL("second-kernel-format"),
      /*.default_identifier=*/{},
      /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN,
      /*.default_pipeline_options=*/{},
      /*.emit=*/EmitNothing,
  };
  static const loom_target_emitter_t* const kFirstEmitters[] = {
      &kFirstEmitter,
  };
  static const loom_target_emitter_t* const kSecondEmitters[] = {
      &kSecondEmitter,
  };
  loom_target_provider_t first_provider = {};
  first_provider.emitter_list = loom_target_emitter_list_make(
      kFirstEmitters, IREE_ARRAYSIZE(kFirstEmitters));
  first_provider.canonical_kernel_emitter = &kFirstEmitter;
  first_provider.canonical_kernel_fact_type = &kFactType;
  loom_target_provider_t second_provider = {};
  second_provider.emitter_list = loom_target_emitter_list_make(
      kSecondEmitters, IREE_ARRAYSIZE(kSecondEmitters));
  second_provider.canonical_kernel_emitter = &kSecondEmitter;
  second_provider.canonical_kernel_fact_type = &kFactType;
  const loom_target_provider_t* providers[] = {
      &first_provider,
      &second_provider,
  };
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_environment_initialize(&provider_set, &environment));
}

TEST_F(TargetProviderTest, RejectsDuplicateFactTypeProviders) {
  static const loom_target_fact_type_t kFactType = {
      /*.name=*/IREE_SVL("shared-facts"),
  };
  loom_target_provider_t first_provider = {};
  first_provider.target_fact_type = &kFactType;
  loom_target_provider_t second_provider = {};
  second_provider.target_fact_type = &kFactType;
  const loom_target_provider_t* providers[] = {
      &first_provider,
      &second_provider,
  };
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_environment_initialize(&provider_set, &environment));
}

TEST_F(TargetProviderTest, RejectsDuplicateProfileFamilyNames) {
  static const loom_target_fact_type_t kFirstFactType = {};
  static const loom_target_fact_type_t kSecondFactType = {};
  static const loom_target_profile_type_t kFirstProfileType = {
      /*.name=*/IREE_SVL("shared-family"),
      /*.fact_type=*/&kFirstFactType,
  };
  static const loom_target_profile_type_t kSecondProfileType = {
      /*.name=*/IREE_SVL("shared-family"),
      /*.fact_type=*/&kSecondFactType,
  };
  loom_target_provider_t first_provider = {};
  first_provider.profile_type = &kFirstProfileType;
  loom_target_provider_t second_provider = {};
  second_provider.profile_type = &kSecondProfileType;
  const loom_target_provider_t* providers[] = {
      &first_provider,
      &second_provider,
  };
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_environment_initialize(&provider_set, &environment));
}

TEST_F(TargetProviderTest, RejectsDuplicateEmitterFormats) {
  static const loom_target_emitter_t kFirstEmitter = {
      /*.name=*/IREE_SVL("first-emitter"),
      /*.public_artifact_format=*/IREE_SVL("shared-format"),
      /*.default_identifier=*/{},
      /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN,
      /*.default_pipeline_options=*/{},
      /*.emit=*/EmitNothing,
  };
  static const loom_target_emitter_t kSecondEmitter = {
      /*.name=*/IREE_SVL("second-emitter"),
      /*.public_artifact_format=*/IREE_SVL("shared-format"),
      /*.default_identifier=*/{},
      /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN,
      /*.default_pipeline_options=*/{},
      /*.emit=*/EmitNothing,
  };
  static const loom_target_emitter_t* const kFirstEmitters[] = {
      &kFirstEmitter,
  };
  static const loom_target_emitter_t* const kSecondEmitters[] = {
      &kSecondEmitter,
  };
  loom_target_provider_t first_provider = {};
  first_provider.emitter_list = loom_target_emitter_list_make(
      kFirstEmitters, IREE_ARRAYSIZE(kFirstEmitters));
  loom_target_provider_t second_provider = {};
  second_provider.emitter_list = loom_target_emitter_list_make(
      kSecondEmitters, IREE_ARRAYSIZE(kSecondEmitters));
  const loom_target_provider_t* providers[] = {
      &first_provider,
      &second_provider,
  };
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_environment_initialize(&provider_set, &environment));
}

TEST_F(TargetProviderTest, RejectsIncompleteEmitterDescriptors) {
  static const loom_target_emitter_t kMissingName = {
      /*.name=*/{},
      /*.public_artifact_format=*/IREE_SVL("missing-name"),
      /*.default_identifier=*/{},
      /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN,
      /*.default_pipeline_options=*/{},
      /*.emit=*/EmitNothing,
  };
  static const loom_target_emitter_t kMissingFormat = {
      /*.name=*/IREE_SVL("missing-format"),
      /*.public_artifact_format=*/{},
      /*.default_identifier=*/{},
      /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN,
      /*.default_pipeline_options=*/{},
      /*.emit=*/EmitNothing,
  };
  static const loom_target_emitter_t kMissingEmit = {
      /*.name=*/IREE_SVL("missing-emit"),
      /*.public_artifact_format=*/IREE_SVL("missing-emit"),
      /*.default_identifier=*/{},
      /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN,
      /*.default_pipeline_options=*/{},
      /*.emit=*/nullptr,
  };
  static const loom_target_emitter_t* const kInvalidEmitters[] = {
      &kMissingName,
      &kMissingFormat,
      &kMissingEmit,
  };
  for (const loom_target_emitter_t* emitter : kInvalidEmitters) {
    const loom_target_emitter_t* emitters[] = {emitter};
    loom_target_provider_t provider = {};
    provider.emitter_list = loom_target_emitter_list_make(emitters, 1);
    const loom_target_provider_t* providers[] = {&provider};
    const loom_target_provider_set_t provider_set =
        loom_target_provider_set_make(providers, 1);
    loom_target_environment_t environment = {};
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_target_environment_initialize(&provider_set, &environment));
  }
}

TEST_F(TargetProviderTest, RejectsMismatchedProfileAndProviderFactTypes) {
  static const loom_target_fact_type_t kProfileFactType = {};
  static const loom_target_fact_type_t kProviderFactType = {};
  static const loom_target_profile_type_t kProfileType = {
      /*.name=*/IREE_SVL("mismatched"),
      /*.fact_type=*/&kProfileFactType,
  };
  loom_target_provider_t provider = {};
  provider.profile_type = &kProfileType;
  provider.target_fact_type = &kProviderFactType;
  const loom_target_provider_t* providers[] = {&provider};
  const loom_target_provider_set_t provider_set =
      loom_target_provider_set_make(providers, IREE_ARRAYSIZE(providers));
  loom_target_environment_t environment = {};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_target_environment_initialize(&provider_set, &environment));
}

}  // namespace
}  // namespace loom
