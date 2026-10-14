// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/finite/component_test_fixture.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <utility>

#include "iree/testing/temp_file.h"
#include "loom/serve/testing/service_application.h"

namespace loom_serve::testing::finite {

iree_status_t ComponentOwner::Run(iree_task_executor_t* executor,
                                  std::function<iree_status_t(bool*)> advance) {
  executor_ = executor;
  advance_ = std::move(advance);
  released_ = false;
  iree_task_process_initialize(Drain, 0, 1, &process_);
  process_.user_data = this;
  process_.release_fn = Release;
  Wake(this);
  std::unique_lock<std::mutex> lock(mutex_);
  condition_.wait(lock, [&] { return released_; });
  return std::exchange(status_, iree_ok_status());
}

void ComponentOwner::Wake(void* user_data) {
  auto* owner = static_cast<ComponentOwner*>(user_data);
  iree_task_executor_schedule_process(owner->executor_, &owner->process_);
}

iree_status_t ComponentOwner::Drain(iree_task_process_t* process,
                                    const iree_task_worker_context_t* context,
                                    iree_task_process_drain_result_t* result) {
  auto* owner = static_cast<ComponentOwner*>(process->user_data);
  owner->status_ = owner->advance_(&result->completed);
  result->completed |= !iree_status_is_ok(owner->status_);
  return iree_ok_status();
}

void ComponentOwner::Release(iree_task_process_t* process) {
  auto* owner = static_cast<ComponentOwner*>(process->user_data);
  std::lock_guard<std::mutex> lock(owner->mutex_);
  owner->released_ = true;
  owner->condition_.notify_one();
}

void FiniteComponentTest::SetUp() {
  auto application_options = loom_serve_test_application_options_default();
  application_options.slab_size = 65536;
  loom_serve_test_application_t* application = nullptr;
  IREE_ASSERT_OK(loom_serve_test_application_acquire(
      &application_options, host_allocator_, &application));
  host_ = loom_serve_test_application_host(application);
  device_ = loom_serve_test_application_device_at(application, 0);
  directory_ = iree::testing::MakeTempFilePath("finite_component");
  ASSERT_TRUE(std::filesystem::create_directory(directory_));
  const char* source_path = std::getenv("LOOM_SERVE_TEST_COMPONENT_SOURCE");
  ASSERT_NE(source_path, nullptr);
  const auto source_directory =
      std::filesystem::path(source_path).parent_path();
  for (const char* file :
       {"prepare.loom", "control.loom", "kernels.loom", "policy.loom"}) {
    ASSERT_TRUE(std::filesystem::copy_file(
        source_directory / file, std::filesystem::path(directory_) / file));
  }
  std::string header =
      R"({"factor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
  header.append((8 - header.size() % 8) % 8, ' ');
  std::ofstream checkpoint(directory_ + "/weights.safetensors",
                           std::ios::binary);
  uint8_t bytes[8];
  iree_unaligned_store_le_u64(bytes, header.size());
  checkpoint.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
  checkpoint.write(header.data(), header.size());
  iree_unaligned_store_le_u32(bytes,
                              0x3f800000);  // F32 1.0, prepared to 2.0.
  checkpoint.write(reinterpret_cast<const char*>(bytes), 4);
  checkpoint.close();
  ASSERT_TRUE(checkpoint.good());
  options_ = {
      .sources =
          {
              .base_directory = iree_make_cstring_view(directory_.c_str()),
              .prepare = IREE_SV("prepare.loom"),
              .control = IREE_SV("control.loom"),
              .count = 1,
              .values = &provider_,
          },
      .asset_directory = IREE_SV("."),
      .options = IREE_SV("{}"),
  };
  IREE_ASSERT_OK(loom_serve_finite_component_allocate(
      device_, &options_, owner_.observer(), host_allocator_, &component_));
}

void FiniteComponentTest::TearDown() {
  if (component_) {
    IREE_EXPECT_OK(Retire(component_));
    loom_serve_finite_component_free(component_);
  }
}

iree_status_t FiniteComponentTest::Prepare(
    loom_serve_finite_component_t* component,
    loom_serve_finite_model_t** out_model) {
  return owner_.Run(loom_serve_host_executor(host_), [&](bool* complete) {
    iree_status_t status =
        loom_serve_finite_component_prepare(component, out_model);
    *complete = *out_model != nullptr;
    return status;
  });
}

iree_status_t FiniteComponentTest::Retire(
    loom_serve_finite_component_t* component) {
  return owner_.Run(loom_serve_host_executor(host_), [&](bool* complete) {
    return loom_serve_finite_component_retire(component, complete);
  });
}

iree_status_t FiniteComponentTest::Trim() {
  return owner_.Run(loom_serve_host_executor(host_), [&](bool* complete) {
    return loom_serve_finite_component_trim(component_, complete);
  });
}

void FiniteComponentTest::Project(loom_serve_finite_model_t* model, float first,
                                  loom_serve_finite_result_t* out_result) {
  iree_vm_buffer_t* input = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_create(16, 16, host_allocator_, &input));
  iree_byte_span_t bytes;
  IREE_ASSERT_OK(iree_vm_buffer_map_write(input, 0, 16, &bytes));
  for (uint32_t i = 0; i < 4; ++i) {
    iree_unaligned_store_le_f32(bytes.data + i * 4,
                                first + static_cast<float>(i));
  }
  loom_serve_finite_prepared_t prepared = {};
  IREE_ASSERT_OK(
      loom_serve_finite_model_prepare_request(model, input, &prepared));
  const iree_host_size_t rows = 1;
  IREE_ASSERT_OK(
      loom_serve_finite_model_prepare_batch(model, 1, &prepared, &rows));
  iree_vm_buffer_release(prepared.bytes);
  iree_vm_buffer_release(input);
  bool admitted = false;
  IREE_ASSERT_OK(loom_serve_residency_try_acquire(
      loom_serve_finite_model_residency(model), &admitted));
  ASSERT_TRUE(admitted);
  loom_serve_execution_frontier_t frontier;
  IREE_ASSERT_OK(loom_serve_finite_model_submit(model, &frontier));
  // Only this outer application thread joins device completion.
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      {2, frontier.semaphores, frontier.values}, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));
  loom_serve_finite_model_complete(model);
  loom_serve_finite_model_take_result(model, out_result);
}

void FiniteComponentTest::ExpectResult(const loom_serve_finite_result_t* result,
                                       float first) {
  const auto output = loom_serve_finite_result_row(result, 0, 0);
  ASSERT_EQ(output.data_length, 16u);
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(iree_unaligned_load_le_f32(output.data + i * 4),
              (first + static_cast<float>(i)) * 2.0f);
  }
}

void FiniteComponentTest::Invoke(loom_serve_finite_model_t* model) {
  loom_serve_finite_result_t result = {};
  ASSERT_NO_FATAL_FAILURE(Project(model, 1.0f, &result));
  ExpectResult(&result, 1.0f);
  loom_serve_finite_result_deinitialize(&result);
}

}  // namespace loom_serve::testing::finite
