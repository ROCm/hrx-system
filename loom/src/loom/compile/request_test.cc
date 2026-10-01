// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/compile/request.h"

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

using ::iree::testing::status::StatusIs;
using ::testing::HasSubstr;

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
    const loom_target_emit_request_t* request, bool* out_emitted,
    loom_target_emit_artifact_t* out_artifact) {
  (void)request;
  *out_emitted = false;
  *out_artifact = {};
  return iree_ok_status();
}

static const loom_target_emitter_t kDiagnosticEmitter = {
    /*.name=*/IREE_SVL("DiagnosticEmitter123"),
    /*.public_artifact_format=*/IREE_SVL("DiagnosticFormat123"),
    /*.default_identifier=*/IREE_SVL("diagnostic.out"),
    /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN,
    /*.default_pipeline_options=*/{},
    /*.emit=*/EmitDiagnosticFormat,
};
static const loom_target_emitter_t kAlternateEmitter = {
    /*.name=*/IREE_SVL("AlternateEmitter123"),
    /*.public_artifact_format=*/IREE_SVL("AlternateFormat123"),
    /*.default_identifier=*/IREE_SVL("alternate.out"),
    /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN,
    /*.default_pipeline_options=*/{},
    /*.emit=*/EmitDiagnosticFormat,
};
static const loom_target_emitter_t* const kTargetEmitters[] = {
    &kDiagnosticEmitter,
    &kAlternateEmitter,
};

class CompileRequestTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &request_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_testing_context_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    target_provider_.profile_type = &kTargetProfileType;
    target_provider_.target_fact_type = &loom_target_generic_fact_type;
    target_provider_.select_profile = SelectTargetProfile;
    emission_provider_.emitter_list = loom_target_emitter_list_make(
        kTargetEmitters, IREE_ARRAYSIZE(kTargetEmitters));
    emission_provider_.canonical_kernel_emitter = &kDiagnosticEmitter;
    emission_provider_.canonical_kernel_fact_type =
        &loom_target_generic_fact_type;
    target_providers_[0] = &target_provider_;
    target_providers_[1] = &emission_provider_;
    target_provider_set_ = loom_target_provider_set_make(
        target_providers_, IREE_ARRAYSIZE(target_providers_));
    IREE_ASSERT_OK(loom_target_environment_initialize(&target_provider_set_,
                                                      &environment_));
  }

  void TearDown() override {
    loom_target_environment_deinitialize(&environment_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&request_arena_);
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

  loom_compile_request_t Resolve(const loom_module_t* module,
                                 loom_compile_request_options_t options) {
    loom_compile_request_t request = {};
    IREE_EXPECT_OK(loom_compile_request_resolve(module, &options, &environment_,
                                                &request_arena_, &request));
    EXPECT_EQ(loom_compile_request_is_command(&request),
              request.target_emitter == nullptr);
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
  iree_arena_allocator_t request_arena_;
  loom_context_t context_;
  loom_target_provider_t target_provider_ = {};
  loom_target_provider_t emission_provider_ = {};
  const loom_target_provider_t* target_providers_[2] = {};
  loom_target_provider_set_t target_provider_set_ = {};
  loom_target_environment_t environment_;
};

TEST_F(CompileRequestTest, InfersKernelAndCanonicalFormat) {
  ModulePtr module = ParseKernel(this, true);
  const loom_compile_request_t request = Resolve(module.get(), {});

  EXPECT_EQ(request.selection.product, LOOM_COMPILE_PRODUCT_KERNEL);
  EXPECT_TRUE(
      iree_string_view_equal(request.target_emitter->public_artifact_format,
                             IREE_SV("DiagnosticFormat123")));
  EXPECT_EQ(request.target_emitter, &kDiagnosticEmitter);
  EXPECT_EQ(request.selection.target_fact_type, &loom_target_generic_fact_type);
  EXPECT_EQ(request.explicit_target.profile, nullptr);
  ASSERT_EQ(request.selection.roots.count, 1u);
  EXPECT_TRUE(iree_string_view_equal(request.selection.roots.values[0],
                                     IREE_SV("Kernel123")));
}

TEST_F(CompileRequestTest, ResolvesArrayProgramsWithDefaultAndNamedRoots) {
  ModulePtr module = Parse(R"(
func.def public abi(array_program) @entry() {
  func.return
}
)");
  const iree_string_view_t roots[] = {IREE_SV("entry")};
  loom_compile_request_options_t options = {};
  options.target = IREE_SV("TargetFamily123:Target456");
  for (iree_host_size_t root_count : {0, 1}) {
    options.roots = {root_count, roots};
    const loom_compile_request_t request = Resolve(module.get(), options);
    EXPECT_EQ(request.selection.product, LOOM_COMPILE_PRODUCT_KERNEL);
    EXPECT_EQ(request.target_emitter, &kDiagnosticEmitter);
    ASSERT_EQ(request.selection.roots.count, 1u);
    EXPECT_TRUE(iree_string_view_equal(request.selection.roots.values[0],
                                       IREE_SV("entry")));
  }
}

TEST_F(CompileRequestTest, KeepsObjectFunctionsAsModuleProducts) {
  ModulePtr module = Parse(R"(
func.def public abi(object_function) @entry() {
  func.return
}
)");
  const iree_string_view_t roots[] = {IREE_SV("entry")};
  loom_compile_request_options_t options = {};
  options.format = IREE_SV("DiagnosticFormat123");
  for (iree_host_size_t root_count : {0, 1}) {
    options.roots = {root_count, roots};
    const loom_compile_request_t request = Resolve(module.get(), options);
    EXPECT_EQ(request.selection.product, LOOM_COMPILE_PRODUCT_MODULE);
    EXPECT_EQ(request.target_emitter, &kDiagnosticEmitter);
    EXPECT_EQ(request.selection.roots.count, root_count);
  }
}

TEST_F(CompileRequestTest, InfersModuleProductFromUnfilteredDefaultRootSet) {
  ModulePtr module = Parse(R"(
func.def public @kept() {
  func.return
}
func.def public @excluded() {
  func.return
}
func.def public @also_kept() {
  func.return
}
func.def @helper() {
  func.return
}
)");
  const iree_string_view_t excluded_roots[] = {IREE_SV("@excluded")};
  loom_compile_request_options_t options = {};
  options.format = IREE_SV("DiagnosticFormat123");
  options.excluded_roots = {IREE_ARRAYSIZE(excluded_roots), excluded_roots};
  const loom_compile_request_t request = Resolve(module.get(), options);

  EXPECT_EQ(request.selection.product, LOOM_COMPILE_PRODUCT_MODULE);
  ASSERT_EQ(request.selection.roots.count, 2u);
  EXPECT_TRUE(iree_string_view_equal(request.selection.roots.values[0],
                                     IREE_SV("kept")));
  EXPECT_TRUE(iree_string_view_equal(request.selection.roots.values[1],
                                     IREE_SV("also_kept")));
}

TEST_F(CompileRequestTest, PreservesRemainingDefaultKernelRoots) {
  ModulePtr module = Parse(R"(
kernel.def @kept() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
kernel.def @excluded() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)");
  const iree_string_view_t excluded_roots[] = {IREE_SV("excluded")};
  loom_compile_request_options_t options = {};
  options.target = IREE_SV("TargetFamily123:Target456");
  options.excluded_roots = {IREE_ARRAYSIZE(excluded_roots), excluded_roots};
  const loom_compile_request_t request = Resolve(module.get(), options);

  EXPECT_EQ(request.selection.product, LOOM_COMPILE_PRODUCT_KERNEL);
  ASSERT_EQ(request.selection.roots.count, 1u);
  EXPECT_TRUE(iree_string_view_equal(request.selection.roots.values[0],
                                     IREE_SV("kept")));
}

TEST_F(CompileRequestTest, ExcludedKernelDoesNotParticipateInTargetSelection) {
  ModulePtr module = Parse(R"(
target.generic<reference> @Target789 {
  subgroup_size = 32
}
target.decl @UnavailableTarget
kernel.def target(@Target789) @kept() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
kernel.def target(@UnavailableTarget) @excluded() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)");
  const iree_string_view_t excluded_roots[] = {IREE_SV("excluded")};
  loom_compile_request_options_t options = {};
  options.excluded_roots = {IREE_ARRAYSIZE(excluded_roots), excluded_roots};
  const loom_compile_request_t request = Resolve(module.get(), options);

  EXPECT_EQ(request.selection.product, LOOM_COMPILE_PRODUCT_KERNEL);
  EXPECT_EQ(request.selection.target_fact_type, &loom_target_generic_fact_type);
  ASSERT_EQ(request.selection.roots.count, 1u);
  EXPECT_TRUE(iree_string_view_equal(request.selection.roots.values[0],
                                     IREE_SV("kept")));
}

TEST_F(CompileRequestTest, RejectsExplicitAndExcludedRoots) {
  ModulePtr module = Parse(R"(
func.def public @entry() {
  func.return
}
func.def public @other() {
  func.return
}
)");
  const iree_string_view_t roots[] = {IREE_SV("entry")};
  const iree_string_view_t excluded_roots[] = {IREE_SV("other")};
  loom_compile_request_options_t options = {};
  options.roots = {IREE_ARRAYSIZE(roots), roots};
  options.product = IREE_SV("module");
  options.format = IREE_SV("DiagnosticFormat123");
  options.excluded_roots = {IREE_ARRAYSIZE(excluded_roots), excluded_roots};
  loom_compile_request_t request = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_compile_request_resolve(module.get(), &options, &environment_,
                                   &request_arena_, &request));
}

TEST_F(CompileRequestTest, RejectsMissingExcludedRoot) {
  ModulePtr module = Parse(R"(
func.def public @entry() {
  func.return
}
)");
  const iree_string_view_t excluded_roots[] = {IREE_SV("missing")};
  loom_compile_request_options_t options = {};
  options.product = IREE_SV("module");
  options.format = IREE_SV("DiagnosticFormat123");
  options.excluded_roots = {IREE_ARRAYSIZE(excluded_roots), excluded_roots};
  loom_compile_request_t request = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      loom_compile_request_resolve(module.get(), &options, &environment_,
                                   &request_arena_, &request));
}

TEST_F(CompileRequestTest, RejectsNonCanonicalExcludedRoot) {
  ModulePtr module = Parse(R"(
func.def public @entry() {
  func.return
}
func.def @private_helper() {
  func.return
}
)");
  const iree_string_view_t excluded_roots[] = {IREE_SV("private_helper")};
  loom_compile_request_options_t options = {};
  options.product = IREE_SV("module");
  options.format = IREE_SV("DiagnosticFormat123");
  options.excluded_roots = {IREE_ARRAYSIZE(excluded_roots), excluded_roots};
  loom_compile_request_t request = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_compile_request_resolve(module.get(), &options, &environment_,
                                   &request_arena_, &request));
}

TEST_F(CompileRequestTest, RejectsRepeatedExcludedRoots) {
  ModulePtr module = Parse(R"(
func.def public @entry() {
  func.return
}
)");
  const iree_string_view_t repeated_roots[] = {
      IREE_SV("entry"),
      IREE_SV("@entry"),
  };
  loom_compile_request_options_t options = {};
  options.product = IREE_SV("module");
  options.format = IREE_SV("DiagnosticFormat123");
  options.excluded_roots = {IREE_ARRAYSIZE(repeated_roots), repeated_roots};
  loom_compile_request_t request = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_compile_request_resolve(module.get(), &options, &environment_,
                                   &request_arena_, &request));
}

TEST_F(CompileRequestTest, RejectsEmptyDefaultRootSet) {
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
  const iree_string_view_t excluded_roots[] = {IREE_SV("Command123")};
  loom_compile_request_options_t options = {};
  options.excluded_roots = {IREE_ARRAYSIZE(excluded_roots), excluded_roots};
  loom_compile_request_t request = {};
  iree::Status status(loom_compile_request_resolve(
      module.get(), &options, &environment_, &request_arena_, &request));
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kInvalidArgument));
  EXPECT_THAT(status.ToString(),
              HasSubstr("excluded roots empty the default command root set"));
}

TEST_F(CompileRequestTest, RequiresExplicitSelectionOfPrivateArrayPrograms) {
  ModulePtr module = Parse(R"(
func.def abi(array_program) @entry() {
  func.return
}
)");
  const iree_string_view_t roots[] = {IREE_SV("entry")};
  loom_compile_request_options_t options = {};
  options.product = IREE_SV("kernel");
  options.target = IREE_SV("TargetFamily123:Target456");
  loom_compile_request_t rejected_request = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_compile_request_resolve(module.get(), &options, &environment_,
                                   &request_arena_, &rejected_request));

  options.roots = {IREE_ARRAYSIZE(roots), roots};
  options.product = iree_string_view_empty();
  const loom_compile_request_t request = Resolve(module.get(), options);
  EXPECT_EQ(request.selection.product, LOOM_COMPILE_PRODUCT_KERNEL);
  EXPECT_EQ(request.target_emitter, &kDiagnosticEmitter);
}

TEST_F(CompileRequestTest, PreservesExplicitRootOrderAndDuplicates) {
  ModulePtr module = Parse(R"(
func.def public @first() {
  func.return
}
func.def public @second() {
  func.return
}
)");
  const iree_string_view_t roots[] = {
      IREE_SV("second"),
      IREE_SV("first"),
      IREE_SV("@second"),
  };
  loom_compile_request_options_t options = {};
  options.roots = {IREE_ARRAYSIZE(roots), roots};
  options.format = IREE_SV("DiagnosticFormat123");

  const loom_compile_request_t request = Resolve(module.get(), options);

  EXPECT_EQ(request.selection.roots.values, roots);
  EXPECT_EQ(request.selection.roots.count, IREE_ARRAYSIZE(roots));
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
  const loom_compile_request_t request = Resolve(module.get(), {});

  EXPECT_EQ(request.selection.product, LOOM_COMPILE_PRODUCT_COMMAND);
  EXPECT_EQ(request.target_emitter, nullptr);
  ASSERT_EQ(request.selection.roots.count, 1u);
  module.reset();
  EXPECT_TRUE(iree_string_view_equal(request.selection.roots.values[0],
                                     IREE_SV("Command123")));
}

TEST_F(CompileRequestTest, RoutesKernelPipelineThroughProductBoundary) {
  ModulePtr module = Parse(R"(
target.generic<reference> @Target789 {
  subgroup_size = 32
}
pipeline.def<kernel> public retain target(@Target789) @KernelPipeline() launch() {
  pipeline.return
}
pipeline.def @GenericPipeline() launch() {
  pipeline.return
}
)");

  loom_compile_request_options_t options = {
      /*.roots=*/{},
      /*.product=*/IREE_SV("kernel"),
  };
  loom_compile_request_t request = Resolve(module.get(), options);
  EXPECT_EQ(request.selection.product, LOOM_COMPILE_PRODUCT_KERNEL);
  EXPECT_EQ(request.target_emitter, &kDiagnosticEmitter);
  EXPECT_EQ(request.selection.target_fact_type, &loom_target_generic_fact_type);
  ASSERT_EQ(request.selection.roots.count, 1u);
  EXPECT_TRUE(iree_string_view_equal(request.selection.roots.values[0],
                                     IREE_SV("KernelPipeline")));

  const iree_string_view_t generic_roots[] = {IREE_SV("@GenericPipeline")};
  options.roots = {
      /*.count=*/IREE_ARRAYSIZE(generic_roots),
      /*.values=*/generic_roots,
  };
  options.product = iree_string_view_empty();
  options.format = IREE_SV("DiagnosticFormat123");
  request = Resolve(module.get(), options);
  EXPECT_EQ(request.selection.product, LOOM_COMPILE_PRODUCT_MODULE);
  EXPECT_EQ(request.target_emitter, &kDiagnosticEmitter);
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
  loom_compile_request_t request = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_compile_request_resolve(module.get(), &options, &environment_,
                                   &request_arena_, &request));
}

TEST_F(CompileRequestTest, ProductConstraintCannotReinterpretRoots) {
  ModulePtr module = ParseKernel(this, true);
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
      loom_compile_request_resolve(module.get(), &options, &environment_,
                                   &request_arena_, &request));
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
  const loom_compile_request_options_t options = {
      /*.roots=*/{},
      /*.product=*/IREE_SV("kernel"),
      /*.format=*/{},
      /*.target=*/IREE_SV("TargetFamily123:Target456"),
  };
  const loom_compile_request_t request = Resolve(module.get(), options);

  EXPECT_EQ(request.selection.product, LOOM_COMPILE_PRODUCT_KERNEL);
  EXPECT_EQ(request.target_emitter, &kDiagnosticEmitter);
  EXPECT_EQ(request.explicit_target.profile, &kTargetProfile);
  ASSERT_EQ(request.selection.roots.count, 1u);
  EXPECT_TRUE(iree_string_view_equal(request.selection.roots.values[0],
                                     IREE_SV("Kernel123")));
}

TEST_F(CompileRequestTest, RejectsUnknownProduct) {
  ModulePtr module = ParseKernel(this, true);
  const loom_compile_request_options_t options = {
      /*.roots=*/{},
      /*.product=*/IREE_SV("Product123"),
  };
  loom_compile_request_t request = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_compile_request_resolve(module.get(), &options, &environment_,
                                   &request_arena_, &request));
}

TEST_F(CompileRequestTest, ModuleFormatSelection) {
  ModulePtr module = Parse(R"(
func.def public @Function123() {
  func.return
}
)");
  loom_compile_request_t request = {};
  loom_compile_request_options_t options = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_compile_request_resolve(module.get(), &options, &environment_,
                                   &request_arena_, &request));

  options.format = IREE_SV("DiagnosticFormat123");
  IREE_ASSERT_OK(loom_compile_request_resolve(
      module.get(), &options, &environment_, &request_arena_, &request));
  EXPECT_EQ(request.selection.product, LOOM_COMPILE_PRODUCT_MODULE);
  EXPECT_EQ(request.target_emitter, &kDiagnosticEmitter);

  options.format = iree_string_view_empty();
  options.target = IREE_SV("TargetFamily123:Target456");
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_compile_request_resolve(module.get(), &options, &environment_,
                                   &request_arena_, &request));

  loom_target_environment_deinitialize(&environment_);
  emission_provider_.canonical_module_emitter = &kDiagnosticEmitter;
  emission_provider_.canonical_module_fact_type =
      &loom_target_generic_fact_type;
  IREE_ASSERT_OK(
      loom_target_environment_initialize(&target_provider_set_, &environment_));
  IREE_ASSERT_OK(loom_compile_request_resolve(
      module.get(), &options, &environment_, &request_arena_, &request));
  EXPECT_EQ(request.selection.product, LOOM_COMPILE_PRODUCT_MODULE);
  EXPECT_EQ(request.target_emitter, &kDiagnosticEmitter);
  EXPECT_EQ(request.explicit_target.profile, &kTargetProfile);
  EXPECT_TRUE(
      iree_string_view_equal(request.target_emitter->public_artifact_format,
                             IREE_SV("DiagnosticFormat123")));
}

TEST_F(CompileRequestTest, ExplicitTargetSpecializesUntargetedKernel) {
  ModulePtr module = Parse(R"(
target.generic<reference> @Target789 {
  subgroup_size = 32
}
kernel.def target(@Target789) @Targeted() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
kernel.def @Untargeted() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)");
  const loom_compile_request_options_t options = {
      /*.roots=*/{},
      /*.product=*/IREE_SV("kernel"),
      /*.format=*/{},
      /*.target=*/IREE_SV("TargetFamily123:Target456"),
  };
  const loom_compile_request_t request = Resolve(module.get(), options);

  EXPECT_EQ(request.explicit_target.profile, &kTargetProfile);
  EXPECT_TRUE(iree_string_view_equal(
      request.explicit_target.specification.selector, IREE_SV("Target456")));
  EXPECT_EQ(request.target_emitter, &kDiagnosticEmitter);
  EXPECT_EQ(request.selection.target_fact_type, &loom_target_generic_fact_type);
  EXPECT_EQ(request.selection.untargeted_kernel_count, 1u);
  EXPECT_EQ(request.selection.roots.count, 2u);
}

TEST_F(CompileRequestTest, MissingCanonicalKernelEmitterFailsClosed) {
  loom_target_environment_deinitialize(&environment_);
  emission_provider_.canonical_kernel_emitter = nullptr;
  emission_provider_.canonical_kernel_fact_type = nullptr;
  IREE_ASSERT_OK(
      loom_target_environment_initialize(&target_provider_set_, &environment_));
  ModulePtr module = ParseKernel(this, true);
  const loom_compile_request_options_t options = {};
  loom_compile_request_t request = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_compile_request_resolve(module.get(), &options, &environment_,
                                   &request_arena_, &request));
}

TEST_F(CompileRequestTest, MissingOptionalTargetFamilyFailsClosed) {
  ModulePtr module = ParseKernel(this, true);
  const loom_target_provider_set_t empty_provider_set =
      loom_target_provider_set_make(nullptr, 0);
  loom_target_environment_t empty_environment;
  IREE_ASSERT_OK(loom_target_environment_initialize(&empty_provider_set,
                                                    &empty_environment));

  const loom_compile_request_options_t options = {};
  loom_compile_request_t request = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_compile_request_resolve(module.get(), &options, &empty_environment,
                                   &request_arena_, &request));

  loom_target_environment_deinitialize(&empty_environment);
}

TEST_F(CompileRequestTest, ExplicitFormatSelectsNoncanonicalAlternative) {
  ModulePtr module = ParseKernel(this, true);
  loom_compile_request_options_t options = {};
  loom_compile_request_t request = {};
  IREE_ASSERT_OK(loom_compile_request_resolve(
      module.get(), &options, &environment_, &request_arena_, &request));
  EXPECT_EQ(request.target_emitter, &kDiagnosticEmitter);

  options.format = IREE_SV("AlternateFormat123");
  IREE_ASSERT_OK(loom_compile_request_resolve(
      module.get(), &options, &environment_, &request_arena_, &request));
  EXPECT_EQ(request.target_emitter, &kAlternateEmitter);
}

TEST_F(CompileRequestTest, DiagnosticFormatCanInspectKernelProduct) {
  ModulePtr module = ParseKernel(this, true);
  const loom_compile_request_options_t options = {
      /*.roots=*/{},
      /*.product=*/IREE_SV("kernel"),
      /*.format=*/IREE_SV("DiagnosticFormat123"),
  };
  loom_compile_request_t request = {};
  IREE_ASSERT_OK(loom_compile_request_resolve(
      module.get(), &options, &environment_, &request_arena_, &request));
  EXPECT_EQ(request.selection.product, LOOM_COMPILE_PRODUCT_KERNEL);
  EXPECT_EQ(request.target_emitter, &kDiagnosticEmitter);
}

}  // namespace
}  // namespace loom
