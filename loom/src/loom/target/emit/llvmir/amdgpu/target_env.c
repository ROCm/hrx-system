// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/amdgpu/target_env.h"

#include <stdint.h>

#include "loom/target/arch/amdgpu/target_info_defs.h"

#define LOOM_LLVMIR_AMDGPU_TARGET_TRIPLE IREE_SVL("amdgcn-amd-amdhsa")
#define LOOM_LLVMIR_AMDGPU_DATA_LAYOUT                           \
  IREE_SVL(                                                      \
      "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-"  \
      "p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:" \
      "256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:"  \
      "256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-"  \
      "A5-G1-ni:7:8:9")
#define LOOM_LLVMIR_AMDGPU_BUFFER_RESOURCE_ADDRESS_SPACE 7

static const loom_llvmir_target_env_t kAmdgcnAmdAmdhsaTargetEnv = {
    .name = LOOM_LLVMIR_AMDGPU_TARGET_TRIPLE,
    .target_triple = LOOM_LLVMIR_AMDGPU_TARGET_TRIPLE,
    .data_layout = LOOM_LLVMIR_AMDGPU_DATA_LAYOUT,
    .object_format = LOOM_LLVMIR_OBJECT_FORMAT_ELF,
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 32,
    .offset_bitwidth = 64,
    .address_spaces =
        {
            .generic = 0,
            .global = 1,
            .local = 3,
            .constant = 4,
            .private_memory = 5,
            .buffer_resource = LOOM_LLVMIR_AMDGPU_BUFFER_RESOURCE_ADDRESS_SPACE,
        },
};

static const loom_target_snapshot_t kAmdgpuHalSnapshot = {
    .name = LOOM_LLVMIR_AMDGPU_TARGET_TRIPLE,
    .codegen_format = LOOM_TARGET_CODEGEN_FORMAT_LLVMIR,
    .artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .default_pointer_bitwidth = 64,
    .index_bitwidth = 32,
    .offset_bitwidth = 64,
    .max_workgroup_size =
        {
            .x = 1024,
            .y = 1024,
            .z = 1024,
        },
    .max_flat_workgroup_size = 1024,
    .max_grid_size =
        {
            .x = INT32_MAX,
            .y = UINT16_MAX,
            .z = UINT16_MAX,
        },
    .max_flat_grid_size = UINT32_MAX,
    .max_workgroup_count =
        {
            .x = INT32_MAX,
            .y = UINT16_MAX,
            .z = UINT16_MAX,
        },
    .memory_spaces =
        {
            .generic = 0,
            .global = 1,
            .workgroup = 3,
            .constant = 4,
            .private_memory = 5,
            .host = UINT32_MAX,
            .descriptor = LOOM_LLVMIR_AMDGPU_BUFFER_RESOURCE_ADDRESS_SPACE,
        },
};

static const loom_target_export_plan_t kAmdgpuHalExportPlan = {
    .name = IREE_SVL("amdgpu-hal"),
    .abi_kind = LOOM_TARGET_ABI_HAL_KERNEL,
    .linkage = LOOM_TARGET_LINKAGE_DEFAULT,
    .hal_kernel =
        {
            .required_workgroup_size = {.x = 0, .y = 0, .z = 0},
            .flat_workgroup_size_min = 0,
            .flat_workgroup_size_max = 0,
            .buffer_resource_flags = LOOM_AMDGPU_HAL_BUFFER_RESOURCE_FLAGS,
        },
};

static const loom_target_config_t kAmdgpuHalConfig = {
    .name = IREE_SVL("default"),
};

static const loom_target_bundle_t kAmdgpuHalBundle = {
    .name = IREE_SVL("amdgpu-hal"),
    .snapshot = &kAmdgpuHalSnapshot,
    .export_plan = &kAmdgpuHalExportPlan,
    .config = &kAmdgpuHalConfig,
};

// This built-in profile is a fixture/default provider convenience. Production
// lowering should prefer derived profiles from the generic target bundle above.
static const loom_llvmir_target_profile_t kAmdgpuHalProfile = {
    .name = IREE_SVL("amdgpu-hal"),
    .target_env = &kAmdgcnAmdAmdhsaTargetEnv,
    .kind = LOOM_LLVMIR_TARGET_PROFILE_KERNEL,
    .exported_linkage = LOOM_LLVMIR_LINKAGE_DEFAULT,
    .kernel =
        {
            .calling_convention = LOOM_LLVMIR_CALLING_CONVENTION_AMDGPU_KERNEL,
            .required_workgroup_size_metadata_name =
                IREE_SVL("reqd_work_group_size"),
            .required_workgroup_size = {.x = 0, .y = 0, .z = 0},
            .flat_workgroup_size_min = 1,
            .flat_workgroup_size_max = 1024,
            .binding_resource_flags = LOOM_AMDGPU_HAL_BUFFER_RESOURCE_FLAGS,
            .flat_workgroup_size_attr_name =
                IREE_SVL("amdgpu-flat-work-group-size"),
            .uniform_workgroup_size_attr_name =
                IREE_SVL("uniform-work-group-size"),
            .flags = LOOM_LLVMIR_KERNEL_PROFILE_FLAG_ALWAYSINLINE,
            .coordinate_intrinsics =
                {
                    .workitem_id_x = IREE_SVL("llvm.amdgcn.workitem.id.x"),
                    .workitem_id_y = IREE_SVL("llvm.amdgcn.workitem.id.y"),
                    .workitem_id_z = IREE_SVL("llvm.amdgcn.workitem.id.z"),
                    .workgroup_id_x = IREE_SVL("llvm.amdgcn.workgroup.id.x"),
                    .workgroup_id_y = IREE_SVL("llvm.amdgcn.workgroup.id.y"),
                    .workgroup_id_z = IREE_SVL("llvm.amdgcn.workgroup.id.z"),
                },
            .binding_parameter_attrs =
                {
                    {
                        .kind = LOOM_LLVMIR_ATTR_INREG,
                        .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
                    },
                    {
                        .kind = LOOM_LLVMIR_ATTR_NOUNDEF,
                        .type_id = LOOM_LLVMIR_TYPE_ID_INVALID,
                    },
                },
            .binding_parameter_attr_count = 2,
        },
};

static const loom_llvmir_target_profile_t* const kAmdgpuTargetProfiles[] = {
    &kAmdgpuHalProfile,
};

static bool loom_llvmir_amdgpu_project_bundle(
    const loom_llvmir_target_profile_projection_request_t* request,
    const loom_llvmir_target_profile_t** out_profile) {
  *out_profile = NULL;
  const loom_target_bundle_t* bundle = request->bundle;
  if (!iree_string_view_is_empty(request->target_triple) &&
      !iree_string_view_equal(request->target_triple,
                              kAmdgcnAmdAmdhsaTargetEnv.target_triple)) {
    return false;
  }
  if (iree_string_view_equal(bundle->name, kAmdgpuHalProfile.name)) {
    *out_profile = &kAmdgpuHalProfile;
    return true;
  }
  if (iree_string_view_is_empty(request->target_triple)) {
    return false;
  }
  if (bundle->export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return false;
  }
  *out_profile = &kAmdgpuHalProfile;
  return true;
}

static const loom_llvmir_target_profile_provider_t
    kAmdgpuTargetProfileProvider = {
        .name = IREE_SVL("amdgpu"),
        .profiles = kAmdgpuTargetProfiles,
        .profile_count = IREE_ARRAYSIZE(kAmdgpuTargetProfiles),
        .llc_target_name = IREE_SVL("amdgcn"),
        .project_bundle = loom_llvmir_amdgpu_project_bundle,
};

const loom_target_bundle_t* loom_llvmir_target_bundle_amdgpu_hal(void) {
  return &kAmdgpuHalBundle;
}

const loom_llvmir_target_env_t* loom_llvmir_target_env_amdgcn_amd_amdhsa(void) {
  return &kAmdgcnAmdAmdhsaTargetEnv;
}

const loom_llvmir_target_profile_t* loom_llvmir_target_profile_amdgpu_hal(
    void) {
  return &kAmdgpuHalProfile;
}

const loom_llvmir_target_profile_provider_t*
loom_llvmir_amdgpu_target_profile_provider(void) {
  return &kAmdgpuTargetProfileProvider;
}

iree_status_t loom_llvmir_target_profile_initialize_amdgpu_hal(
    loom_llvmir_target_profile_t* out_profile) {
  *out_profile = kAmdgpuHalProfile;
  return iree_ok_status();
}
