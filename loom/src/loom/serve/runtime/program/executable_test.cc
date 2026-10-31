// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/executable.h"

#include "iree/testing/status_matchers.h"
#include "loom/serve/testing/jit.h"

namespace {

using loom_serve::testing::JitTest;
using loom_serve::testing::loom_serve_test_compilation_t;

static iree_status_t CreateOwnedSequence(iree_const_byte_span_t contents,
                                         iree_allocator_t host_allocator,
                                         iree_byte_sequence_t** out_sequence) {
  iree_byte_span_t storage = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(
      iree_allocator_clone(host_allocator, contents, (void**)&storage.data));
  storage.data_length = contents.data_length;
  iree_status_t status = iree_byte_sequence_create_from_span_move(
      &storage, host_allocator, out_sequence);
  iree_allocator_free(host_allocator, storage.data);
  return status;
}

class ExecutableTest : public JitTest {
 protected:
  void CheckImageReuse(bool direct) {
    const loomc_config_binding_t bindings[] = {
        {loomc_make_cstring_view("increment.groups"),
         loomc_make_cstring_view("2")},
        {loomc_make_cstring_view("increment.groups"),
         loomc_make_cstring_view("4")},
    };
    const loomc_config_options_t configs[] = {
        {&bindings[0], 1, {}, LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED},
        {&bindings[1], 1, {}, LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED},
    };
    // The second pass also proves image eviction/recreation and that recorded
    // users outlive the catalog, not just the specialized views.
    for (int pass = 0; pass < 2; ++pass) {
      SCOPED_TRACE(pass);
      loom_serve_test_compilation_t jobs[2];
      for (size_t i = 0; i < 2; ++i) {
        jobs[i].jit = jit_;
        jobs[i].direct_dispatch = direct;
        jobs[i].root =
            direct ? IREE_SV("increment_groups") : IREE_SV("advance_groups");
        jobs[i].config = &configs[i];
        jobs[i].Start(
            loom_serve_host_executor(loom_serve_device_host(device_owner_)));
      }
      for (size_t i = 0; i < 2; ++i) {
        loom_serve_jit_stage_t* stage = nullptr;
        auto status = jobs[i].Await(&stage);
        if (iree_status_is_ok(status)) {
          if (direct) {
            dispatches_[i] = loom_serve_jit_stage_dispatch(stage);
            loom_serve_dispatch_retain(dispatches_[i]);
          } else {
            status = loom_serve_jit_stage_record(
                stage, iree_hal_queue_family(dispatch_),
                IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, nullptr, nullptr,
                &commands_[i]);
          }
        }
        loom_serve_jit_stage_free(stage);
        IREE_EXPECT_OK(status);
      }
      if (HasFailure()) {
        return;
      }
      const auto storage = loom_serve_jit_cache_storage(jit_);
      EXPECT_EQ(storage.count, 2u);
      EXPECT_EQ(storage.executable_count, 1u);
      EXPECT_EQ(compilation_.kernels, 2u * (pass + 1));
      EXPECT_EQ(compilation_.reused_kernels, 0u);
      if (direct) {
        EXPECT_NE(dispatches_[0], dispatches_[1]);
      }
      loom_serve_jit_trim(jit_);
      EXPECT_EQ(loom_serve_jit_cache_storage(jit_).count, 2u);
      EXPECT_EQ(loom_serve_jit_cache_storage(jit_).executable_count, 1u);
      if (pass) {
        loom_serve_jit_release(jit_);
        jit_ = nullptr;
      }
      if (!buffer_) {
        IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
            loom_serve_pool_storage(storage_owner_),
            iree_hal_make_pool_reservation_request(sizeof(input_), 0),
            iree_infinite_timeout(), &buffer_));
      }
      const iree_hal_transfer_operation_t upload = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
          .upload = {input_.data(), buffer_, 0, sizeof(input_)},
      };
      uint64_t completion = 0;
      IREE_ASSERT_OK(
          loom_serve_execution_transfer(execution_, 1, &upload, &completion));
      const iree_hal_buffer_binding_t binding = {buffer_, 0, sizeof(input_)};
      for (size_t i = 0; i < 2; ++i) {
        if (direct) {
          iree_hal_buffer_ref_t scratch;
          IREE_ASSERT_OK(loom_serve_dispatch_issue(dispatches_[i], execution_,
                                                   {1, &binding}, &scratch,
                                                   &completion));
        } else {
          IREE_ASSERT_OK(loom_serve_execution_execute(
              execution_, commands_[i], {1, &binding}, &completion));
        }
      }
      const iree_hal_transfer_operation_t download = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download = {buffer_, 0, output_.data(), sizeof(output_)},
      };
      IREE_ASSERT_OK(
          loom_serve_execution_feedback(execution_, 1, &download, &completion));
      IREE_ASSERT_OK(
          loom_serve_execution_feedback_wait(execution_, completion));
      for (size_t i = 0; i < output_.size(); ++i) {
        EXPECT_EQ(output_[i], i < 2 ? 114 : (i < 4 ? 107 : 100)) << i;
      }
      for (size_t i = 0; i < 2; ++i) {
        iree_hal_command_buffer_release(commands_[i]);
        commands_[i] = nullptr;
        loom_serve_dispatch_release(dispatches_[i]);
        dispatches_[i] = nullptr;
        if (jit_) {
          loom_serve_jit_trim(jit_);
          EXPECT_EQ(loom_serve_jit_cache_storage(jit_).executable_count,
                    i ? 0u : 1u);
        }
      }
      if (jit_) {
        EXPECT_EQ(loom_serve_jit_cache_storage(jit_).count, 0u);
      }
    }
  }
};

TEST_P(ExecutableTest, CommandInputsShareCodeAndPreserveLaunchGeometry) {
  CheckImageReuse(false);
}

TEST_P(ExecutableTest, DirectInputsShareCodeAndPreserveLaunchGeometry) {
  CheckImageReuse(true);
}

TEST_P(ExecutableTest, FailedImageLoadsAreNotRetainedAndCanRetireCache) {
  const auto* targets =
      iree_hal_device_spec_executables(iree_hal_device_spec(device_));
  ASSERT_GT(targets->target_count, 0u);
  struct Completion {
    // Owned cache, retired by the last completion.
    loom_serve_executable_cache_t* cache = nullptr;
    // Successful delivery, independent of the expected loader failure.
    bool completed = false;
    // Last request can destroy its cache inside the callback.
    bool retire = false;
  } completion;
  IREE_ASSERT_OK(loom_serve_executable_cache_allocate(
      iree_hal_queue_family(dispatch_), &targets->targets[0], host_allocator_,
      &completion.cache));
  for (int attempt = 0; attempt < 2; ++attempt) {
    SCOPED_TRACE(attempt);
    completion.completed = false;
    completion.retire = attempt != 0;
    const uint8_t invalid[] = {0, 1, 2, 3};
    iree_byte_sequence_t* image = nullptr;
    IREE_ASSERT_OK(
        CreateOwnedSequence(iree_make_const_byte_span(invalid, sizeof(invalid)),
                            host_allocator_, &image));
    loom_serve_executable_request_t request = {
        .image = image,
        .completion =
            [](void* user_data, iree_status_t status,
               loom_serve_executable_t* executable) {
              auto* completion = static_cast<Completion*>(user_data);
              IREE_EXPECT_NOT_OK(status);
              EXPECT_EQ(executable, nullptr);
              loom_serve_executable_release(executable);
              EXPECT_EQ(
                  loom_serve_executable_cache_storage(completion->cache).count,
                  0u);
              if (completion->retire) {
                loom_serve_executable_cache_free(completion->cache);
                completion->cache = nullptr;
              }
              completion->completed = true;
            },
        .user_data = &completion,
    };
    loom_serve_executable_cache_load(completion.cache, &request);
    EXPECT_TRUE(completion.completed);
    EXPECT_EQ(request.image, nullptr);
  }
  EXPECT_EQ(completion.cache, nullptr);
}

INSTANTIATE_TEST_SUITE_P(SharedWorkers, ExecutableTest,
                         ::testing::Values(1, 4));

}  // namespace
