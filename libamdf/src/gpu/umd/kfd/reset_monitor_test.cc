// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/reset_monitor.h"

#include <cerrno>
#include <cstring>

#include "gtest/gtest.h"

namespace {

class KfdResetMonitorTest : public ::testing::Test {
 protected:
  static amdf_status_t ContextIoctl(void* user_data, int render_descriptor,
                                    union drm_amdgpu_ctx* context) {
    auto* self = static_cast<KfdResetMonitorTest*>(user_data);
    self->last_render_descriptor_ = render_descriptor;
    self->last_context_ = *context;
    switch (context->in.op) {
      case AMDGPU_CTX_OP_ALLOC_CTX:
        ++self->allocate_count_;
        if (!amdf_status_is_ok(self->allocate_status_)) {
          return self->allocate_status_;
        }
        context->out.alloc.ctx_id = self->allocated_context_identifier_;
        return AMDF_STATUS_OK;
      case AMDGPU_CTX_OP_QUERY_STATE2:
        ++self->query_count_;
        if (!amdf_status_is_ok(self->query_status_)) {
          return self->query_status_;
        }
        context->out.state.flags = self->query_flags_;
        return AMDF_STATUS_OK;
      case AMDGPU_CTX_OP_FREE_CTX:
        ++self->release_count_;
        return self->release_status_;
      default:
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }

  amdf_gpu_kfd_reset_monitor_native_api_t native_api_ = {
      .user_data = this,
      .context_ioctl = ContextIoctl,
  };
  amdf_status_t allocate_status_ = AMDF_STATUS_OK;
  amdf_status_t query_status_ = AMDF_STATUS_OK;
  amdf_status_t release_status_ = AMDF_STATUS_OK;
  uint32_t allocated_context_identifier_ = 47;
  uint64_t query_flags_ = 0;
  int allocate_count_ = 0;
  int query_count_ = 0;
  int release_count_ = 0;
  int last_render_descriptor_ = -1;
  union drm_amdgpu_ctx last_context_ = {};
};

TEST_F(KfdResetMonitorTest, InitializationPublishesOnlyAfterNativeSuccess) {
  amdf_gpu_kfd_reset_monitor_t monitor;
  std::memset(&monitor, 0xA5, sizeof(monitor));
  const amdf_gpu_kfd_reset_monitor_t original = monitor;
  allocate_status_ = amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, ENOMEM);

  EXPECT_EQ(amdf_gpu_kfd_reset_monitor_initialize(19, &native_api_, &monitor),
            allocate_status_);
  EXPECT_EQ(std::memcmp(&monitor, &original, sizeof(monitor)), 0);
  EXPECT_EQ(allocate_count_, 1);
  EXPECT_EQ(last_render_descriptor_, 19);
  EXPECT_EQ(last_context_.in.op, AMDGPU_CTX_OP_ALLOC_CTX);
  EXPECT_EQ(last_context_.in.priority, AMDGPU_CTX_PRIORITY_NORMAL);
}

TEST_F(KfdResetMonitorTest, QueryPublishesOnlyAfterNativeSuccess) {
  amdf_gpu_kfd_reset_monitor_t monitor = {};
  ASSERT_EQ(amdf_gpu_kfd_reset_monitor_initialize(19, &native_api_, &monitor),
            AMDF_STATUS_OK);
  amdf_gpu_kfd_reset_state_t state = {true, true};
  const amdf_gpu_kfd_reset_state_t original = state;
  query_status_ = amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EIO);

  EXPECT_EQ(amdf_gpu_kfd_reset_monitor_query(&monitor, &state), query_status_);
  EXPECT_EQ(std::memcmp(&state, &original, sizeof(state)), 0);
  EXPECT_EQ(query_count_, 1);
  EXPECT_EQ(last_context_.in.op, AMDGPU_CTX_OP_QUERY_STATE2);
  EXPECT_EQ(last_context_.in.ctx_id, allocated_context_identifier_);
}

TEST_F(KfdResetMonitorTest, ResetObservationLatchesAcrossQueries) {
  amdf_gpu_kfd_reset_monitor_t monitor = {};
  ASSERT_EQ(amdf_gpu_kfd_reset_monitor_initialize(19, &native_api_, &monitor),
            AMDF_STATUS_OK);

  query_flags_ = AMDGPU_CTX_QUERY2_FLAGS_RESET_IN_PROGRESS;
  amdf_gpu_kfd_reset_state_t state = {};
  ASSERT_EQ(amdf_gpu_kfd_reset_monitor_query(&monitor, &state), AMDF_STATUS_OK);
  EXPECT_FALSE(state.reset_observed);
  EXPECT_TRUE(state.reset_in_progress);

  query_flags_ =
      AMDGPU_CTX_QUERY2_FLAGS_RESET | AMDGPU_CTX_QUERY2_FLAGS_RESET_IN_PROGRESS;
  ASSERT_EQ(amdf_gpu_kfd_reset_monitor_query(&monitor, &state), AMDF_STATUS_OK);
  EXPECT_TRUE(state.reset_observed);
  EXPECT_TRUE(state.reset_in_progress);

  query_flags_ = 0;
  ASSERT_EQ(amdf_gpu_kfd_reset_monitor_query(&monitor, &state), AMDF_STATUS_OK);
  EXPECT_TRUE(state.reset_observed);
  EXPECT_FALSE(state.reset_in_progress);
}

TEST_F(KfdResetMonitorTest, ReleaseFailureRetainsContextForExactRetry) {
  amdf_gpu_kfd_reset_monitor_t monitor = {};
  ASSERT_EQ(amdf_gpu_kfd_reset_monitor_initialize(19, &native_api_, &monitor),
            AMDF_STATUS_OK);
  release_status_ = amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EBUSY);

  EXPECT_EQ(amdf_gpu_kfd_reset_monitor_deinitialize(&monitor), release_status_);
  EXPECT_TRUE(monitor.context_owned);
  EXPECT_EQ(monitor.context_identifier, allocated_context_identifier_);
  EXPECT_EQ(last_context_.in.op, AMDGPU_CTX_OP_FREE_CTX);
  EXPECT_EQ(last_context_.in.ctx_id, allocated_context_identifier_);

  release_status_ = AMDF_STATUS_OK;
  EXPECT_EQ(amdf_gpu_kfd_reset_monitor_deinitialize(&monitor), AMDF_STATUS_OK);
  EXPECT_FALSE(monitor.context_owned);
  EXPECT_EQ(monitor.context_identifier, 0u);
  EXPECT_EQ(release_count_, 2);
}

}  // namespace
