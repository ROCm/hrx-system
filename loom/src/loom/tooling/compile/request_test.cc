// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/compile/request.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ops/target/ops.h"
#include "loom/testing/context.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

static const loom_target_snapshot_t kTargetSnapshot = {
    /*.name=*/IREE_SVL("Snapshot123"),
};
static const loom_target_export_plan_t kTargetExportPlan = {
    /*.name=*/IREE_SVL("ExportPlan123"),
};
static const loom_target_config_t kTargetConfig = {
    /*.name=*/IREE_SVL("TargetConfig123"),
};
static const loom_target_bundle_t kTargetBundle = {
    /*.name=*/IREE_SVL("TargetBundle123"),
    /*.snapshot=*/&kTargetSnapshot,
    /*.export_plan=*/&kTargetExportPlan,
    /*.config=*/&kTargetConfig,
};

static iree_status_t ProjectTargetFacts(const loom_target_profile_t* profile,
                                        iree_arena_allocator_t* arena,
                                        loom_target_facts_t* out_facts) {
  (void)profile;
  (void)arena;
  (void)out_facts;
  return iree_ok_status();
}

static const loom_target_profile_type_t kTargetProfileType = {
    /*.name=*/IREE_SVL("TargetFamily123"),
    /*.fact_type=*/&loom_target_generic_fact_type,
    /*.project_facts=*/ProjectTargetFacts,
};
static const loom_target_profile_t kTargetProfile = {
    /*.type=*/&kTargetProfileType,
    /*.target_bundle=*/&kTargetBundle,
};
static const loom_target_fact_type_t kOtherTargetFactType = {
    /*.name=*/IREE_SVL("OtherTargetFamily123"),
    /*.storage_size=*/sizeof(loom_target_facts_t),
};
static const loom_target_profile_type_t kOtherTargetProfileType = {
    /*.name=*/IREE_SVL("OtherTargetFamily123"),
    /*.fact_type=*/&kOtherTargetFactType,
    /*.project_facts=*/ProjectTargetFacts,
};

static iree_status_t SelectTargetProfile(
    iree_string_view_t selector, const loom_target_profile_t** out_profile) {
  *out_profile = nullptr;
  if (!iree_string_view_equal(selector, IREE_SV("Target456"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown synthetic target selector");
  }
  *out_profile = &kTargetProfile;
  return iree_ok_status();
}

static iree_status_t EmitDiagnosticFormat(
    const loom_target_emit_request_t* request,
    loom_target_emit_artifact_t* out_artifact) {
  (void)request;
  *out_artifact = {};
  return iree_ok_status();
}

static const loom_target_emitter_t kDiagnosticEmitter = {
    /*.name=*/IREE_SVL("DiagnosticEmitter123"),
    /*.public_artifact_format=*/IREE_SVL("DiagnosticFormat123"),
    /*.default_identifier=*/IREE_SVL("diagnostic.out"),
    /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN,
    /*.emit=*/EmitDiagnosticFormat,
};
static const loom_target_emitter_t* const kTargetEmitters[] = {
    &kDiagnosticEmitter,
};

static const loom_artifact_provider_t kExecutableProvider = {
    /*.name=*/IREE_SVL("Compiler123"),
    /*.public_artifact_format=*/IREE_SVL("ExecutableFormat123"),
    /*.flags=*/LOOM_ARTIFACT_PROVIDER_FLAG_CANONICAL,
    /*.target_profile_type=*/&kTargetProfileType,
};
static const loom_artifact_provider_t kAlternateExecutableProvider = {
    /*.name=*/IREE_SVL("AlternateCompiler123"),
    /*.public_artifact_format=*/IREE_SVL("AlternateExecutableFormat123"),
    /*.flags=*/0,
    /*.target_profile_type=*/&kTargetProfileType,
};
static const loom_artifact_provider_t kDuplicateCanonicalProvider = {
    /*.name=*/IREE_SVL("DuplicateCompiler123"),
    /*.public_artifact_format=*/IREE_SVL("DuplicateExecutableFormat123"),
    /*.flags=*/LOOM_ARTIFACT_PROVIDER_FLAG_CANONICAL,
    /*.target_profile_type=*/&kTargetProfileType,
};
static const loom_artifact_provider_t kOtherExecutableProvider = {
    /*.name=*/IREE_SVL("OtherCompiler123"),
    /*.public_artifact_format=*/IREE_SVL("OtherExecutableFormat123"),
    /*.flags=*/LOOM_ARTIFACT_PROVIDER_FLAG_CANONICAL,
    /*.target_profile_type=*/&kOtherTargetProfileType,
};

class CompileRequestTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_testing_context_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    target_provider_.profile_type = &kTargetProfileType;
    target_provider_.emitter_list = loom_target_emitter_list_make(
        kTargetEmitters, IREE_ARRAYSIZE(kTargetEmitters));
    target_provider_.target_fact_type = &loom_target_generic_fact_type;
    target_provider_.select_profile = SelectTargetProfile;
    target_providers_[0] = &target_provider_;
    target_provider_set_ = loom_target_provider_set_make(target_providers_, 1);
    IREE_ASSERT_OK(loom_target_environment_initialize(&target_provider_set_,
                                                      &environment_));
  }

  void TearDown() override {
    loom_target_environment_deinitialize(&environment_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr Parse(const char* source) {
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source),
                                   IREE_SV("request_test.loom"), &context_,
                                   &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return ModulePtr(module);
  }

  loom_compile_request_t Resolve(
      const loom_module_t* module, loom_compile_request_options_t options,
      const loom_artifact_provider_t* const* providers,
      iree_host_size_t provider_count,
      const loom_target_fact_type_t* inferred_target_fact_type = nullptr) {
    const loom_artifact_provider_registry_t registry = {
        /*.providers=*/providers,
        /*.provider_count=*/provider_count,
    };
    loom_compile_request_t request = {};
    IREE_EXPECT_OK(
        loom_compile_request_resolve(module, &options, &registry, &environment_,
                                     inferred_target_fact_type, &request));
    return request;
  }

  static ModulePtr ParseKernel(CompileRequestTest* test, bool with_target) {
    return with_target ? test->Parse(R"(
target.generic<reference> @Target789 {
  subgroup_size = 32
}
kernel.def target(@Target789) @Kernel123() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)")
                       : test->Parse(R"(
kernel.def @Kernel123() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)");
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_provider_t target_provider_ = {};
  const loom_target_provider_t* target_providers_[1] = {};
  loom_target_provider_set_t target_provider_set_ = {};
  loom_target_environment_t environment_;
};

TEST_F(CompileRequestTest, InfersKernelAndCanonicalFormat) {
  ModulePtr module = ParseKernel(this, true);
  const loom_artifact_provider_t* providers[] = {&kExecutableProvider};
  const loom_compile_request_t request =
      Resolve(module.get(), {}, providers, IREE_ARRAYSIZE(providers));

  EXPECT_EQ(request.product, LOOM_COMPILE_PRODUCT_KERNEL);
  EXPECT_TRUE(
      iree_string_view_equal(request.format, IREE_SV("ExecutableFormat123")));
  EXPECT_EQ(request.producer.kind, LOOM_COMPILE_PRODUCER_ARTIFACT);
  EXPECT_EQ(request.producer.value.artifact_provider, &kExecutableProvider);
  EXPECT_EQ(request.target_fact_type, &loom_target_generic_fact_type);
  EXPECT_EQ(request.explicit_target.target_profile, nullptr);
}

TEST_F(CompileRequestTest, InfersPublicCommandBeforeKernelDependencies) {
  ModulePtr module = Parse(R"(
kernel.def @Kernel123() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
command.program.def public @Command123() launch() {
  kernel.launch @Kernel123() : ()
  command.return
}
)");
  const loom_compile_request_t request = Resolve(module.get(), {}, nullptr, 0);

  EXPECT_EQ(request.product, LOOM_COMPILE_PRODUCT_COMMAND);
  EXPECT_TRUE(iree_string_view_equal(request.format, IREE_SV("loom-command")));
  EXPECT_EQ(request.producer.kind, LOOM_COMPILE_PRODUCER_COMMAND);
  EXPECT_EQ(request.target_fact_type, nullptr);
}

TEST_F(CompileRequestTest, RejectsMixedExplicitRootProducts) {
  ModulePtr module = Parse(R"(
kernel.def @Kernel123() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
command.program.def public @Command123() launch() {
  command.return
}
)");
  const iree_string_view_t roots[] = {
      IREE_SV("@Kernel123"),
      IREE_SV("@Command123"),
  };
  const loom_compile_request_options_t options = {
      /*.roots=*/
      {
          /*.count=*/IREE_ARRAYSIZE(roots),
          /*.values=*/roots,
      },
  };
  const loom_artifact_provider_registry_t registry = {};
  loom_compile_request_t request = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_compile_request_resolve(module.get(), &options, &registry,
                                   &environment_, nullptr, &request));
}

TEST_F(CompileRequestTest, ProductConstraintCannotReinterpretRoots) {
  ModulePtr module = ParseKernel(this, true);
  const loom_artifact_provider_t* providers[] = {&kExecutableProvider};
  const loom_artifact_provider_registry_t registry = {
      /*.providers=*/providers,
      /*.provider_count=*/IREE_ARRAYSIZE(providers),
  };
  const iree_string_view_t roots[] = {IREE_SV("@Kernel123")};
  const loom_compile_request_options_t options = {
      /*.roots=*/
      {
          /*.count=*/IREE_ARRAYSIZE(roots),
          /*.values=*/roots,
      },
      /*.product=*/IREE_SV("command"),
  };
  loom_compile_request_t request = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_compile_request_resolve(module.get(), &options, &registry,
                                   &environment_, nullptr, &request));
}

TEST_F(CompileRequestTest, ExplicitProductSelectsCanonicalRoots) {
  ModulePtr module = Parse(R"(
kernel.def @Kernel123() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
command.program.def public @Command123() launch() {
  kernel.launch @Kernel123() : ()
  command.return
}
)");
  const loom_artifact_provider_t* providers[] = {&kExecutableProvider};
  const loom_compile_request_options_t options = {
      /*.roots=*/{},
      /*.product=*/IREE_SV("kernel"),
      /*.format=*/{},
      /*.target=*/IREE_SV("TargetFamily123:Target456"),
  };
  const loom_compile_request_t request =
      Resolve(module.get(), options, providers, IREE_ARRAYSIZE(providers));

  EXPECT_EQ(request.product, LOOM_COMPILE_PRODUCT_KERNEL);
  EXPECT_EQ(request.producer.value.artifact_provider, &kExecutableProvider);
  EXPECT_EQ(request.explicit_target.target_profile, &kTargetProfile);
}

TEST_F(CompileRequestTest, RejectsUnknownProduct) {
  ModulePtr module = ParseKernel(this, true);
  const loom_artifact_provider_registry_t registry = {};
  const loom_compile_request_options_t options = {
      /*.roots=*/{},
      /*.product=*/IREE_SV("Product123"),
  };
  loom_compile_request_t request = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_compile_request_resolve(module.get(), &options, &registry,
                                   &environment_, nullptr, &request));
}

TEST_F(CompileRequestTest, ModuleRequiresExplicitFormat) {
  ModulePtr module = Parse(R"(
func.def public @Function123() {
  func.return
}
)");
  const loom_artifact_provider_registry_t registry = {};
  loom_compile_request_t request = {};
  loom_compile_request_options_t options = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_compile_request_resolve(module.get(), &options, &registry,
                                   &environment_, nullptr, &request));

  options.format = IREE_SV("DiagnosticFormat123");
  IREE_ASSERT_OK(loom_compile_request_resolve(
      module.get(), &options, &registry, &environment_, nullptr, &request));
  EXPECT_EQ(request.product, LOOM_COMPILE_PRODUCT_MODULE);
  EXPECT_EQ(request.producer.kind, LOOM_COMPILE_PRODUCER_TARGET_EMITTER);
  EXPECT_EQ(request.producer.value.target_emitter, &kDiagnosticEmitter);
}

TEST_F(CompileRequestTest, ExplicitTargetSpecializesUntargetedKernel) {
  ModulePtr module = ParseKernel(this, false);
  const loom_artifact_provider_t* providers[] = {&kExecutableProvider};
  const loom_compile_request_options_t options = {
      /*.roots=*/{},
      /*.product=*/IREE_SV("kernel"),
      /*.format=*/{},
      /*.target=*/IREE_SV("TargetFamily123:Target456"),
  };
  const loom_compile_request_t request =
      Resolve(module.get(), options, providers, IREE_ARRAYSIZE(providers));

  EXPECT_EQ(request.explicit_target.target_profile, &kTargetProfile);
  EXPECT_TRUE(iree_string_view_equal(request.explicit_target.target_key,
                                     IREE_SV("Target456")));
  EXPECT_EQ(request.target_fact_type, &loom_target_generic_fact_type);
  EXPECT_EQ(request.producer.value.artifact_provider, &kExecutableProvider);
}

TEST_F(CompileRequestTest, InferredTargetSelectsUntargetedKernelFormat) {
  ModulePtr module = ParseKernel(this, false);
  const loom_artifact_provider_t* providers[] = {&kExecutableProvider};
  const loom_compile_request_t request =
      Resolve(module.get(), {}, providers, IREE_ARRAYSIZE(providers),
              &loom_target_generic_fact_type);

  EXPECT_EQ(request.explicit_target.target_profile, nullptr);
  EXPECT_EQ(request.target_fact_type, &loom_target_generic_fact_type);
  EXPECT_TRUE(
      iree_string_view_equal(request.format, IREE_SV("ExecutableFormat123")));
  EXPECT_EQ(request.producer.value.artifact_provider, &kExecutableProvider);
}

TEST_F(CompileRequestTest, RejectsIncompatibleInferredTargetFamily) {
  ModulePtr module = ParseKernel(this, true);
  const loom_artifact_provider_t* providers[] = {&kExecutableProvider};
  const loom_artifact_provider_registry_t registry = {
      /*.providers=*/providers,
      /*.provider_count=*/IREE_ARRAYSIZE(providers),
  };
  const loom_compile_request_options_t options = {};
  loom_compile_request_t request = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_compile_request_resolve(
                            module.get(), &options, &registry, &environment_,
                            &kOtherTargetFactType, &request));
}

TEST_F(CompileRequestTest, ExplicitFormatMustMatchKernelTarget) {
  ModulePtr module = ParseKernel(this, true);
  const loom_artifact_provider_t* providers[] = {&kOtherExecutableProvider};
  const loom_artifact_provider_registry_t registry = {
      /*.providers=*/providers,
      /*.provider_count=*/IREE_ARRAYSIZE(providers),
  };
  const loom_compile_request_options_t options = {
      /*.roots=*/{},
      /*.product=*/{},
      /*.format=*/IREE_SV("OtherExecutableFormat123"),
  };
  loom_compile_request_t request = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_compile_request_resolve(module.get(), &options, &registry,
                                   &environment_, nullptr, &request));
}

TEST_F(CompileRequestTest, MissingOptionalKernelProviderFailsClosed) {
  ModulePtr module = ParseKernel(this, true);
  const loom_artifact_provider_registry_t registry = {};
  const loom_compile_request_options_t options = {};
  loom_compile_request_t request = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_compile_request_resolve(module.get(), &options, &registry,
                                   &environment_, nullptr, &request));
}

TEST_F(CompileRequestTest, MissingOptionalTargetFamilyFailsClosed) {
  ModulePtr module = ParseKernel(this, true);
  const loom_artifact_provider_t* providers[] = {&kExecutableProvider};
  const loom_artifact_provider_registry_t registry = {
      /*.providers=*/providers,
      /*.provider_count=*/IREE_ARRAYSIZE(providers),
  };
  const loom_target_provider_set_t empty_provider_set =
      loom_target_provider_set_make(nullptr, 0);
  loom_target_environment_t empty_environment;
  IREE_ASSERT_OK(loom_target_environment_initialize(&empty_provider_set,
                                                    &empty_environment));

  const loom_compile_request_options_t options = {};
  loom_compile_request_t request = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_compile_request_resolve(module.get(), &options, &registry,
                                   &empty_environment, nullptr, &request));

  loom_target_environment_deinitialize(&empty_environment);
}

TEST_F(CompileRequestTest, ExplicitFormatSelectsNoncanonicalAlternative) {
  ModulePtr module = ParseKernel(this, true);
  const loom_artifact_provider_t* providers[] = {
      &kExecutableProvider,
      &kAlternateExecutableProvider,
  };
  const loom_artifact_provider_registry_t registry = {
      /*.providers=*/providers,
      /*.provider_count=*/IREE_ARRAYSIZE(providers),
  };
  loom_compile_request_options_t options = {};
  loom_compile_request_t request = {};
  IREE_ASSERT_OK(loom_compile_request_resolve(
      module.get(), &options, &registry, &environment_, nullptr, &request));
  EXPECT_EQ(request.producer.value.artifact_provider, &kExecutableProvider);

  options.format = IREE_SV("AlternateExecutableFormat123");
  IREE_ASSERT_OK(loom_compile_request_resolve(
      module.get(), &options, &registry, &environment_, nullptr, &request));
  EXPECT_EQ(request.producer.value.artifact_provider,
            &kAlternateExecutableProvider);
}

TEST_F(CompileRequestTest, RejectsMultipleCanonicalFormats) {
  ModulePtr module = ParseKernel(this, true);
  const loom_artifact_provider_t* providers[] = {
      &kExecutableProvider,
      &kDuplicateCanonicalProvider,
  };
  const loom_artifact_provider_registry_t registry = {
      /*.providers=*/providers,
      /*.provider_count=*/IREE_ARRAYSIZE(providers),
  };
  const loom_compile_request_options_t options = {};
  loom_compile_request_t request = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_compile_request_resolve(module.get(), &options, &registry,
                                   &environment_, nullptr, &request));
}

TEST_F(CompileRequestTest, DiagnosticFormatCanInspectKernelProduct) {
  ModulePtr module = ParseKernel(this, true);
  const loom_artifact_provider_registry_t registry = {};
  const loom_compile_request_options_t options = {
      /*.roots=*/{},
      /*.product=*/IREE_SV("kernel"),
      /*.format=*/IREE_SV("DiagnosticFormat123"),
  };
  loom_compile_request_t request = {};
  IREE_ASSERT_OK(loom_compile_request_resolve(
      module.get(), &options, &registry, &environment_, nullptr, &request));
  EXPECT_EQ(request.product, LOOM_COMPILE_PRODUCT_KERNEL);
  EXPECT_EQ(request.producer.kind, LOOM_COMPILE_PRODUCER_TARGET_EMITTER);
}

}  // namespace
}  // namespace loom
