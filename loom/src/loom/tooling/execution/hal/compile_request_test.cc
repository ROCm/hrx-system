// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/compile_request.h"

#include <memory>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ops/target/ops.h"
#include "loom/testing/context.h"
#include "loom/testing/module_ptr.h"

namespace loom {
namespace {

using ModulePtr = ::loom::testing::ModulePtr;

static const loom_target_snapshot_t kFakeTargetSnapshot = {
    /*.name=*/IREE_SVL("FakeTargetSnapshot123"),
};
static const loom_target_export_plan_t kFakeTargetExportPlan = {
    /*.name=*/IREE_SVL("FakeTargetExportPlan123"),
};
static const loom_target_config_t kFakeTargetConfig = {
    /*.name=*/IREE_SVL("FakeTargetConfig123"),
};
static const loom_target_bundle_t kFakeTargetBundle = {
    /*.name=*/IREE_SVL("FakeTargetBundle123"),
    /*.snapshot=*/&kFakeTargetSnapshot,
    /*.export_plan=*/&kFakeTargetExportPlan,
    /*.config=*/&kFakeTargetConfig,
};

static iree_status_t ProjectFakeTargetFacts(
    const loom_target_profile_t* profile, iree_arena_allocator_t* arena,
    loom_target_facts_t* out_facts) {
  (void)profile;
  (void)arena;
  (void)out_facts;
  return iree_ok_status();
}

static const loom_target_profile_type_t kFakeTargetProfileType = {
    /*.name=*/IREE_SVL("FakeTargetFamily123"),
    /*.fact_type=*/&loom_target_generic_fact_type,
    /*.project_facts=*/ProjectFakeTargetFacts,
};
static const loom_target_profile_t kFakeTargetProfile = {
    /*.type=*/&kFakeTargetProfileType,
    /*.target_bundle=*/&kFakeTargetBundle,
};

static iree_status_t SelectFakeTargetProfile(
    iree_string_view_t selector, const loom_target_profile_t** out_profile) {
  *out_profile = nullptr;
  if (!iree_string_view_equal(selector, IREE_SV("FakeGenericTarget456"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown fake target selector");
  }
  *out_profile = &kFakeTargetProfile;
  return iree_ok_status();
}

static const loom_artifact_provider_t kFakeArtifactProvider = {
    /*.name=*/IREE_SVL("FakeArtifactProvider123"),
    /*.public_artifact_format=*/IREE_SVL("FakeExecutableFormat123"),
    /*.flags=*/LOOM_ARTIFACT_PROVIDER_FLAG_CANONICAL,
    /*.target_profile_type=*/&kFakeTargetProfileType,
};

static const loom_target_facts_t kFakeTargetRequirement = {
    /*.fact_type=*/&loom_target_generic_fact_type,
};
static const loom_target_facts_t* g_selected_target_requirement = nullptr;

static iree_status_t SelectFakeCompatibleTarget(
    const loom_device_provider_t* provider,
    const loom_run_hal_runtime_t* runtime,
    const loom_target_facts_t* target_requirement,
    loom_device_target_t* out_target) {
  (void)provider;
  g_selected_target_requirement = target_requirement;
  const iree_hal_executable_target_selection_t selection = {
      /*.family=*/IREE_SV("FakeTargetFamily123"),
      /*.target_key=*/IREE_SV("FakeExactTarget789"),
  };
  const iree_hal_executable_target_selection_result_t result =
      iree_hal_device_spec_select_executable_target(
          iree_hal_device_spec(runtime->device), &selection);
  if (result.outcome != IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "fake exact target is unavailable");
  }
  *out_target = (loom_device_target_t){
      /*.executable_target=*/result.target,
  };
  loom_target_bundle_storage_initialize_from_bundle(
      &kFakeTargetBundle, &out_target->target_bundle_storage);
  return iree_ok_status();
}

static iree_status_t ProjectFakeDeviceTargetFacts(
    const loom_device_provider_t* provider,
    const loom_run_hal_runtime_t* runtime, const loom_device_target_t* target,
    iree_arena_allocator_t* arena, loom_target_facts_t* out_facts) {
  (void)provider;
  (void)runtime;
  (void)target;
  (void)arena;
  (void)out_facts;
  return iree_ok_status();
}

static const loom_device_provider_t kFakeDeviceProvider = {
    /*.artifact_provider=*/&kFakeArtifactProvider,
    /*.driver_name=*/IREE_SVL("fake"),
    /*.select_target=*/nullptr,
    /*.select_compatible_target=*/SelectFakeCompatibleTarget,
    /*.project_target_facts=*/ProjectFakeDeviceTargetFacts,
};

typedef struct FakeHalDevice {
  // HAL resource header used by device vtable dispatch.
  iree_hal_resource_t resource;
  // Immutable device facts borrowed from the test fixture.
  const iree_hal_device_spec_t* device_spec;
} FakeHalDevice;

static const iree_hal_device_spec_t* FakeHalDeviceSpec(
    iree_hal_device_t* base_device) {
  FakeHalDevice* device = reinterpret_cast<FakeHalDevice*>(base_device);
  return device->device_spec;
}

static iree_hal_device_vtable_t MakeFakeHalDeviceVtable() {
  iree_hal_device_vtable_t vtable = {};
  vtable.device_spec = FakeHalDeviceSpec;
  return vtable;
}

static const iree_hal_device_vtable_t kFakeHalDeviceVtable =
    MakeFakeHalDeviceVtable();

struct DeviceSpecDeleter {
  void operator()(iree_hal_device_spec_t* device_spec) const {
    iree_hal_device_spec_release(device_spec);
  }
};
using DeviceSpecPtr =
    std::unique_ptr<iree_hal_device_spec_t, DeviceSpecDeleter>;

class HalCompileRequestTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_testing_context_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));

    target_provider_.profile_type = &kFakeTargetProfileType;
    target_provider_.target_fact_type = &loom_target_generic_fact_type;
    target_provider_.select_profile = SelectFakeTargetProfile;
    target_providers_[0] = &target_provider_;
    target_provider_set_ = loom_target_provider_set_make(target_providers_, 1);
    IREE_ASSERT_OK(loom_target_environment_initialize(&target_provider_set_,
                                                      &target_environment_));

    const iree_hal_executable_target_t executable_targets[] = {
        {
            /*.family=*/IREE_SV("FakeTargetFamily123"),
            /*.target_key=*/IREE_SV("FakeGenericTarget456"),
            /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
            /*.priority=*/50,
            /*.physical_device_affinity=*/1,
        },
        {
            /*.family=*/IREE_SV("FakeTargetFamily123"),
            /*.target_key=*/IREE_SV("FakeExactTarget789"),
            /*.kind=*/IREE_HAL_EXECUTABLE_TARGET_KIND_EXACT,
            /*.priority=*/100,
            /*.physical_device_affinity=*/1,
        },
    };
    const iree_hal_device_executable_spec_t executables = {
        /*.target_count=*/IREE_ARRAYSIZE(executable_targets),
        /*.targets=*/executable_targets,
    };
    const iree_hal_device_spec_params_t params = {
        /*.identity=*/nullptr,
        /*.memory=*/nullptr,
        /*.virtual_memory=*/nullptr,
        /*.queues=*/nullptr,
        /*.dispatch=*/nullptr,
        /*.timing=*/nullptr,
        /*.executables=*/&executables,
        /*.sanitizer=*/nullptr,
        /*.facet_count=*/0,
        /*.facets=*/nullptr,
    };
    iree_hal_device_spec_t* device_spec = nullptr;
    IREE_ASSERT_OK(iree_hal_device_spec_create(&params, iree_allocator_system(),
                                               &device_spec));
    device_spec_.reset(device_spec);
    device_.device_spec = device_spec_.get();
    iree_hal_resource_initialize(&kFakeHalDeviceVtable, &device_.resource);
    runtime_.device = reinterpret_cast<iree_hal_device_t*>(&device_);
    g_selected_target_requirement = nullptr;
  }

  void TearDown() override {
    loom_target_environment_deinitialize(&target_environment_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  ModulePtr ParseKernel(bool with_target = true) {
    static constexpr char kTargetedSource[] = R"(
target.generic<reference> @FakeAuthoredTarget123 {
  subgroup_size = 32
}
kernel.def target(@FakeAuthoredTarget123) @FakeKernel123() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)";
    static constexpr char kTargetlessSource[] = R"(
kernel.def @FakeKernel123() {
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%one, %one, %one) workgroup_size(%one, %one, %one) : index
} launch() {
  kernel.return
}
)";
    loom_module_t* module = nullptr;
    loom_text_parse_options_t options = {};
    IREE_EXPECT_OK(loom_text_parse(
        with_target ? IREE_SV(kTargetedSource) : IREE_SV(kTargetlessSource),
        IREE_SV("compile_request_test.loom"), &context_, &block_pool_, &options,
        &module));
    EXPECT_NE(module, nullptr);
    return ModulePtr(module);
  }

  loom_run_hal_compile_resolve_options_t ResolveOptions(
      const loom_module_t* module,
      loom_compile_request_options_t compile_options) {
    return (loom_run_hal_compile_resolve_options_t){
        /*.module=*/module,
        /*.compile=*/compile_options,
        /*.target_requirement=*/&kFakeTargetRequirement,
        /*.target_environment=*/&target_environment_,
        /*.device_provider=*/&kFakeDeviceProvider,
        /*.runtime=*/&runtime_,
    };
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_target_provider_t target_provider_ = {};
  const loom_target_provider_t* target_providers_[1] = {};
  loom_target_provider_set_t target_provider_set_ = {};
  loom_target_environment_t target_environment_;
  DeviceSpecPtr device_spec_;
  FakeHalDevice device_ = {};
  loom_run_hal_runtime_t runtime_ = {};
};

TEST_F(HalCompileRequestTest, ExplicitTargetOverridesPreferredExactTarget) {
  ModulePtr module = ParseKernel();
  const iree_string_view_t roots[] = {IREE_SV("@FakeKernel123")};
  const loom_compile_request_options_t compile_options = {
      /*.roots=*/
      {
          /*.count=*/IREE_ARRAYSIZE(roots),
          /*.values=*/roots,
      },
      /*.product=*/IREE_SV("kernel"),
      /*.format=*/IREE_SV("FakeExecutableFormat123"),
      /*.target=*/IREE_SV("FakeTargetFamily123:FakeGenericTarget456"),
  };
  loom_run_hal_compile_resolve_options_t options =
      ResolveOptions(module.get(), compile_options);

  loom_run_hal_compile_request_t request = {};
  IREE_ASSERT_OK(loom_run_hal_compile_request_resolve(&options, &request));

  EXPECT_EQ(request.compile.product, LOOM_COMPILE_PRODUCT_KERNEL);
  EXPECT_EQ(request.compile.producer.value.artifact_provider,
            &kFakeArtifactProvider);
  ASSERT_NE(request.device_target.executable_target, nullptr);
  EXPECT_EQ(request.device_target.executable_target->kind,
            IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC);
  EXPECT_TRUE(
      iree_string_view_equal(loom_device_target_key(&request.device_target),
                             IREE_SV("FakeGenericTarget456")));
  EXPECT_EQ(g_selected_target_requirement, nullptr);
  const loom_target_bundle_t* target_bundle =
      loom_device_target_bundle(&request.device_target);
  ASSERT_NE(target_bundle, nullptr);
  EXPECT_EQ(target_bundle->snapshot,
            &request.device_target.target_bundle_storage.snapshot);
  EXPECT_EQ(target_bundle->export_plan,
            &request.device_target.target_bundle_storage.export_plan);
  EXPECT_EQ(target_bundle->config,
            &request.device_target.target_bundle_storage.config);

  loom_device_target_profile_t device_profile = {};
  const loom_target_profile_t* specialization_profile = nullptr;
  IREE_ASSERT_OK(loom_run_hal_compile_request_target_profile(
      &request, &kFakeDeviceProvider, &runtime_, &device_profile,
      &specialization_profile));
  EXPECT_EQ(specialization_profile, &kFakeTargetProfile);
  EXPECT_EQ(device_profile.base.type, nullptr);
}

TEST_F(HalCompileRequestTest, ImplicitTargetUsesCompatibilitySelection) {
  ModulePtr module = ParseKernel(false);
  const iree_string_view_t roots[] = {IREE_SV("@FakeKernel123")};
  const loom_compile_request_options_t compile_options = {
      /*.roots=*/
      {
          /*.count=*/IREE_ARRAYSIZE(roots),
          /*.values=*/roots,
      },
  };
  loom_run_hal_compile_resolve_options_t options =
      ResolveOptions(module.get(), compile_options);
  options.target_requirement = nullptr;

  loom_run_hal_compile_request_t request = {};
  IREE_ASSERT_OK(loom_run_hal_compile_request_resolve(&options, &request));

  EXPECT_EQ(g_selected_target_requirement, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(loom_device_target_key(&request.device_target),
                             IREE_SV("FakeExactTarget789")));
  loom_device_target_profile_t device_profile = {};
  const loom_target_profile_t* specialization_profile = nullptr;
  IREE_ASSERT_OK(loom_run_hal_compile_request_target_profile(
      &request, &kFakeDeviceProvider, &runtime_, &device_profile,
      &specialization_profile));
  EXPECT_EQ(specialization_profile, &device_profile.base);
  EXPECT_EQ(device_profile.provider, &kFakeDeviceProvider);
  EXPECT_EQ(device_profile.target, &request.device_target);
}

TEST_F(HalCompileRequestTest, RejectsTargetNotAdvertisedByDevice) {
  ModulePtr module = ParseKernel();
  const iree_string_view_t roots[] = {IREE_SV("@FakeKernel123")};
  const loom_compile_request_options_t compile_options = {
      /*.roots=*/
      {
          /*.count=*/IREE_ARRAYSIZE(roots),
          /*.values=*/roots,
      },
      /*.product=*/{},
      /*.format=*/{},
      /*.target=*/IREE_SV("FakeTargetFamily123:FakeGenericTarget456"),
  };
  loom_run_hal_compile_resolve_options_t options =
      ResolveOptions(module.get(), compile_options);

  const iree_hal_executable_target_selection_t exact_only_selection = {
      /*.family=*/IREE_SV("FakeTargetFamily123"),
      /*.target_key=*/IREE_SV("FakeExactTarget789"),
  };
  const iree_hal_executable_target_selection_result_t exact_target =
      iree_hal_device_spec_select_executable_target(device_spec_.get(),
                                                    &exact_only_selection);
  const iree_hal_device_executable_spec_t exact_only_executables = {
      /*.target_count=*/1,
      /*.targets=*/exact_target.target,
  };
  const iree_hal_device_spec_params_t params = {
      /*.identity=*/nullptr,
      /*.memory=*/nullptr,
      /*.virtual_memory=*/nullptr,
      /*.queues=*/nullptr,
      /*.dispatch=*/nullptr,
      /*.timing=*/nullptr,
      /*.executables=*/&exact_only_executables,
      /*.sanitizer=*/nullptr,
      /*.facet_count=*/0,
      /*.facets=*/nullptr,
  };
  iree_hal_device_spec_t* exact_only_spec = nullptr;
  IREE_ASSERT_OK(iree_hal_device_spec_create(&params, iree_allocator_system(),
                                             &exact_only_spec));
  device_spec_.reset(exact_only_spec);
  device_.device_spec = device_spec_.get();

  loom_run_hal_compile_request_t request = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      loom_run_hal_compile_request_resolve(&options, &request));
  EXPECT_EQ(request.device_target.executable_target, nullptr);
}

}  // namespace
}  // namespace loom
