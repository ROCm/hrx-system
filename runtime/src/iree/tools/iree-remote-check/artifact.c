// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tools/iree-remote-check/artifact.h"

#include <string.h>

#if defined(IREE_REMOTE_CHECK_HAVE_AMDGPU_ARTIFACT)
#include "iree/tools/iree-remote-check/amdgpu_artifact.h"
#endif  // IREE_REMOTE_CHECK_HAVE_AMDGPU_ARTIFACT
#if defined(IREE_REMOTE_CHECK_HAVE_VULKAN_ARTIFACT)
#include "iree/tools/iree-remote-check/vulkan_artifact.h"
#endif  // IREE_REMOTE_CHECK_HAVE_VULKAN_ARTIFACT

#if defined(IREE_REMOTE_CHECK_HAVE_AMDGPU_ARTIFACT) && \
    defined(IREE_REMOTE_CHECK_HAVE_VULKAN_ARTIFACT)
#define IREE_REMOTE_CHECK_ARTIFACT_DESCRIPTION "AMDGPU or Vulkan artifacts"
#elif defined(IREE_REMOTE_CHECK_HAVE_AMDGPU_ARTIFACT)
#define IREE_REMOTE_CHECK_ARTIFACT_DESCRIPTION "AMDGPU artifacts"
#elif defined(IREE_REMOTE_CHECK_HAVE_VULKAN_ARTIFACT)
#define IREE_REMOTE_CHECK_ARTIFACT_DESCRIPTION "Vulkan artifacts"
#else
#define IREE_REMOTE_CHECK_ARTIFACT_DESCRIPTION "artifacts"
#endif

iree_status_t iree_remote_check_select_artifact(
    const iree_hal_device_spec_t* device_spec,
    iree_remote_check_artifact_t* out_artifact) {
  IREE_ASSERT_ARGUMENT(device_spec);
  IREE_ASSERT_ARGUMENT(out_artifact);
  memset(out_artifact, 0, sizeof(*out_artifact));

#if defined(IREE_REMOTE_CHECK_HAVE_AMDGPU_ARTIFACT)
  IREE_RETURN_IF_ERROR(
      iree_remote_check_select_amdgpu_artifact(device_spec, out_artifact));
  if (out_artifact->executable_target) return iree_ok_status();
#endif  // IREE_REMOTE_CHECK_HAVE_AMDGPU_ARTIFACT

#if defined(IREE_REMOTE_CHECK_HAVE_VULKAN_ARTIFACT)
  IREE_RETURN_IF_ERROR(
      iree_remote_check_select_vulkan_artifact(device_spec, out_artifact));
  if (out_artifact->executable_target) return iree_ok_status();

#endif  // IREE_REMOTE_CHECK_HAVE_VULKAN_ARTIFACT

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "the remote device has no executable target covered by the "
      "embedded " IREE_REMOTE_CHECK_ARTIFACT_DESCRIPTION
#if defined(IREE_REMOTE_CHECK_HAVE_AMDGPU_ARTIFACT)
      "; configure matching AMDGPU targets with "
      "--//runtime/src/iree/hal/drivers/amdgpu:targets"
#endif  // IREE_REMOTE_CHECK_HAVE_AMDGPU_ARTIFACT
  );
}
