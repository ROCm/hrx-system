// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_QWEN3_VL_TEST_UTIL_H_
#define EXPERIMENTAL_ID4_STAGES_QWEN3_VL_TEST_UTIL_H_

#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/test_util.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

namespace id4::test {

extern const char kQwen3VlConditionKernelSource[];

typedef struct Qwen3VlExecutableCache {
  // HAL resource header.
  iree_hal_resource_t resource;
  // Host allocator used for executable cache storage.
  iree_allocator_t host_allocator;
  // Number of infer-format calls observed.
  iree_host_size_t infer_count;
  // Number of can-prepare calls observed.
  iree_host_size_t can_prepare_count;
  // Number of prepare calls observed.
  iree_host_size_t prepare_count;
  // Caching mode from the latest prepare call.
  iree_hal_executable_caching_mode_t last_caching_mode;
} Qwen3VlExecutableCache;

iree_status_t CreateQwen3VlExecutableCache(iree_allocator_t host_allocator,
                                           Qwen3VlExecutableCache** out_cache);

iree_status_t CreateQwen3VlStage(iree_hal_device_group_t* device_group,
                                 Qwen3VlExecutableCache* executable_cache,
                                 id4_pipeline_kernel_cache_t* kernel_cache,
                                 iree_allocator_t host_allocator,
                                 id4_pipeline_stage_t** out_stage);

}  // namespace id4::test

#endif  // EXPERIMENTAL_ID4_STAGES_QWEN3_VL_TEST_UTIL_H_
