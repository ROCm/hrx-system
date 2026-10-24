// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/module.h"

#include <array>
#include <fstream>
#include <string>
#include <utility>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/execution/retirement.h"
#include "loom/serve/runtime/program/program.h"
#include "loom/serve/testing/device.h"

namespace {

class ParameterModuleTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  void SetUp() override {
    loom_serve_retirement_initialize(&retirement_);
    IREE_ASSERT_OK(
        loom_serve_host_create(GetParam(), nullptr, allocator_, &host_));
    iree_hal_driver_t* driver = nullptr;
    IREE_ASSERT_OK(loom_serve_test_driver_acquire(host_, &driver));
    const loom_serve_device_options_t options = {
        .driver = driver,
        .uri = loom_serve_test_device()->device_uri,
    };
    auto status = loom_serve_device_group_allocate(host_, 1, &options,
                                                   allocator_, &group_);
    iree_hal_driver_release(driver);
    IREE_ASSERT_OK(status);
    owner_ = loom_serve_device_group_at(group_, 0);
    device_ = loom_serve_device_handle(owner_);
    queue_ = loom_serve_device_transfer_queue(owner_);
    IREE_ASSERT_OK(iree_vm_environment_allocate(allocator_, &environment_));
    const iree_vm_ref_type_table_t* table = nullptr;
    IREE_ASSERT_OK(iree_hal_module_register_types(environment_, &table));
    IREE_ASSERT_OK(iree_hal_module_types_resolve(table, &types_));
    IREE_ASSERT_OK(iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment_, IREE_SV("vm")),
        &vm_types_));
    for (size_t i = 0; i < 2; ++i) {
      gates_[i] = Semaphore();
      produced_[i] = Semaphore();
      consumed_[i] = Semaphore();
      descriptors_[i] = Buffer(24, &retirement_);
      targets_[i] = Buffer(16, &retirement_);
    }
  }

  iree_hal_buffer_t* Buffer(size_t length,
                            loom_serve_retirement_t* retirement) {
    iree_hal_buffer_t* buffer = nullptr;
    IREE_CHECK_OK(iree_hal_pool_allocate_buffer(
        loom_serve_device_storage_pool(owner_),
        iree_hal_make_pool_reservation_request(length, 256),
        iree_infinite_timeout(), &buffer));
    IREE_CHECK_OK(loom_serve_retirement_track(retirement, &buffer, allocator_));
    return buffer;
  }

  iree_hal_semaphore_t* Semaphore() {
    iree_hal_semaphore_t* result = nullptr;
    IREE_CHECK_OK(
        iree_hal_semaphore_create(device_, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                                  0, IREE_HAL_SEMAPHORE_FLAG_NONE, &result));
    return result;
  }

  static uint8_t Byte(size_t domain, size_t byte) {
    return 1 + (domain * 61 + byte * 13) % 255;
  }

  void DropModule() {
    loom_serve_program_free(std::exchange(program_, nullptr));
    iree_vm_module_release(std::exchange(module_, nullptr));
  }

  void TearDown() override {
    DropModule();
    loom_serve_parameter_sources_release(sources_);
    for (size_t i = 0; i < 2; ++i) {
      if (gates_[i]) {
        IREE_EXPECT_OK(iree_hal_semaphore_signal(gates_[i], 2, nullptr));
      }
      iree_hal_buffer_release(descriptors_[i]);
      iree_hal_buffer_release(targets_[i]);
    }
    loom_serve_retirement_await(&retirement_);
    loom_serve_retirement_deinitialize(&retirement_);
    for (size_t i = 0; i < 2; ++i) {
      iree_hal_semaphore_release(gates_[i]);
      iree_hal_semaphore_release(produced_[i]);
      iree_hal_semaphore_release(consumed_[i]);
      if (paths_[i].Exists()) {
        EXPECT_TRUE(paths_[i].Remove());
      }
    }
    iree_vm_environment_free(environment_);
    IREE_EXPECT_OK(loom_serve_device_group_free(group_));
    loom_serve_host_release(host_);
  }

  void SelectSource(size_t domain) {
    const std::string header =
        R"({"same-key":{"dtype":"U8","shape":[16],"data_offsets":[0,16]}})";
    std::ofstream file(paths_[domain].path(), std::ios::binary);
    uint8_t length[8];
    iree_unaligned_store_le_u64(length, header.size());
    file.write(reinterpret_cast<const char*>(length), sizeof(length));
    file.write(header.data(), header.size());
    for (size_t i = 0; i < 16; ++i) {
      file.put(Byte(domain, i));
    }
    file.close();
    ASSERT_TRUE(file.good());
    const loom_serve_sources_t names = {.base_directory = IREE_SV(".")};
    const loom_model_file_t path = {paths_[domain].path_view(), {}};
    const loom_model_file_list_t files = {1, &path};
    IREE_ASSERT_OK(loom_serve_parameter_sources_create(
        nullptr, &names, 1, &files, allocator_, &sources_));
    if (!module_) {
      IREE_ASSERT_OK(loom_serve_parameters_module_create(
          environment_, &types_, queue_, sources_, allocator_, &module_));
      IREE_ASSERT_OK(loom_serve_program_link(module_, {nullptr, 0}, allocator_,
                                             &program_));
      IREE_ASSERT_OK(iree_vm_process_lookup_function(
          loom_serve_program_process(program_), IREE_SV("parameters"),
          IREE_SV("gather"), &gather_));
    } else {
      loom_serve_parameters_module_bind_sources(module_, sources_);
    }
    loom_serve_parameter_sources_release(std::exchange(sources_, nullptr));
  }

  void Submit(size_t domain) {
    std::array<uint8_t, 24> record = {};
    iree_unaligned_store_le_u64(record.data() + 16, 16);
    uint64_t one = 1;
    IREE_ASSERT_OK(iree_hal_queue_update(
        queue_, {1, &gates_[domain], &one}, {1, &produced_[domain], &one},
        record.data(), 0, descriptors_[domain], 0, record.size(), nullptr,
        IREE_HAL_UPDATE_FLAG_NONE));
    iree_hal_fence_t* produced = nullptr;
    IREE_ASSERT_OK(
        iree_hal_fence_create_at(produced_[domain], 1, allocator_, &produced));
    iree_vm_buffer_t* key = nullptr;
    IREE_ASSERT_OK(iree_vm_buffer_clone(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span("same-key", 8), 1, allocator_, &key));
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32(0),
        iree_vm_buffer_variant_from_ptr_move(&vm_types_, &key),
        iree_hal_fence_variant_from_ptr_move(&types_, &produced),
        iree_hal_buffer_variant_from_ptr_borrowed(&types_,
                                                  descriptors_[domain]),
        iree_vm_variant_from_i64(0),
        iree_vm_variant_from_i64(1),
        iree_hal_buffer_variant_from_ptr_borrowed(&types_, targets_[domain])};
    iree_vm_variant_t result = {};
    auto status =
        iree_vm_invoke(loom_serve_program_invocation(program_), gather_,
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_from_ptr(&result, 1));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_hal_fence_t* ready = nullptr;
    if (iree_status_is_ok(status)) {
      status =
          iree_hal_fence_ptr_from_variant_borrowed(&types_, result, &ready);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_queue_download(
          queue_, iree_hal_fence_semaphore_list(ready),
          {1, &consumed_[domain], &one}, targets_[domain], 0,
          output_[domain].data(), output_[domain].size(), nullptr);
    }
    iree_vm_variant_reset(&result);
    IREE_ASSERT_OK(status);
    iree_hal_buffer_release(std::exchange(descriptors_[domain], nullptr));
    iree_hal_buffer_release(std::exchange(targets_[domain], nullptr));
  }

  // Host allocation policy, alive through final queue retirement.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Shared workers and proactor used by the native queue implementation.
  loom_serve_host_t* host_ = nullptr;
  // Physical owner enclosing every module and accepted operation.
  loom_serve_device_group_t* group_ = nullptr;
  // Borrowed serving storage and queue capabilities.
  loom_serve_device_t* owner_ = nullptr;
  // Borrowed physical device, alive through group teardown.
  iree_hal_device_t* device_ = nullptr;
  // Borrowed exact transfer queue used by all tested operations.
  iree_hal_queue_t* queue_ = nullptr;
  // Canonical type owner outliving the module and escaped fence references.
  iree_vm_environment_t* environment_ = nullptr;
  // Device reference identities supplied to the native capability.
  iree_hal_module_types_t types_ = {};
  // Host byte-buffer identities supplied to the native capability.
  iree_vm_ref_types_t vm_types_ = {};
  // Temporary cold catalog ownership released after every binding.
  loom_serve_parameter_sources_t* sources_ = nullptr;
  // Read capability whose binding changes while an old operation is pending.
  iree_vm_module_t* module_ = nullptr;
  // Actual VM process, not a direct call into the native implementation.
  loom_serve_program_t* program_ = nullptr;
  // Process-bound callable, valid only through DropModule.
  iree_vm_function_t gather_ = {};
  // Exact buffer captures joined even on an early test assertion.
  loom_serve_retirement_t retirement_;
  // Independent producer prerequisites, released by the outer test owner.
  std::array<iree_hal_semaphore_t*, 2> gates_ = {};
  // Descriptor uploads complete independently of file reads.
  std::array<iree_hal_semaphore_t*, 2> produced_ = {};
  // Final byte observations, never used as a storage-retirement substitute.
  std::array<iree_hal_semaphore_t*, 2> consumed_ = {};
  // Caller-owned descriptor views, transferred to queue captures on submission.
  std::array<iree_hal_buffer_t*, 2> descriptors_ = {};
  // Caller-owned destination views, transferred to queue captures on
  // submission.
  std::array<iree_hal_buffer_t*, 2> targets_ = {};
  // Host observations kept alive through actual source-buffer retirement.
  std::array<std::array<uint8_t, 16>, 2> output_ = {};
  // Distinct immutable files intentionally using the same key and source
  // ordinal.
  iree::testing::TempFilePath paths_[2] = {
      iree::testing::TempFilePath("parameter_module_a", ".safetensors"),
      iree::testing::TempFilePath("parameter_module_b", ".safetensors")};
};

TEST_P(ParameterModuleTest, RebindingAndRemovalPreserveAcceptedReads) {
  ASSERT_NO_FATAL_FAILURE(SelectSource(0));
  ASSERT_NO_FATAL_FAILURE(Submit(0));
  ASSERT_NO_FATAL_FAILURE(SelectSource(1));
  ASSERT_NO_FATAL_FAILURE(Submit(1));
  DropModule();
  EXPECT_FALSE(loom_serve_retirement_is_idle(&retirement_));
  for (size_t domain : {1u, 0u}) {
    IREE_ASSERT_OK(iree_hal_semaphore_signal(gates_[domain], 1, nullptr));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(consumed_[domain], 1,
                                           iree_infinite_timeout(), 0));
    for (size_t i = 0; i < output_[domain].size(); ++i) {
      EXPECT_EQ(output_[domain][i], Byte(domain, i));
    }
    if (domain == 1) {
      EXPECT_FALSE(loom_serve_retirement_is_idle(&retirement_));
    }
  }
}

TEST_P(ParameterModuleTest, RejectsInvalidSourcesAndCountsBeforeSubmission) {
  ASSERT_NO_FATAL_FAILURE(SelectSource(0));
  iree_vm_buffer_t* key = nullptr;
  IREE_ASSERT_OK(iree_vm_buffer_create(0, 1, allocator_, &key));
  const auto invoke = [&](int32_t source, int64_t count) {
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32(source),
        iree_vm_buffer_variant_from_ptr_borrowed(&vm_types_, key),
        iree_hal_fence_variant_from_ptr_borrowed(&types_, nullptr),
        iree_hal_buffer_variant_from_ptr_borrowed(&types_, descriptors_[0]),
        iree_vm_variant_from_i64(0),
        iree_vm_variant_from_i64(count),
        iree_hal_buffer_variant_from_ptr_borrowed(&types_, targets_[0])};
    iree_vm_variant_t result = {};
    auto status =
        iree_vm_invoke(loom_serve_program_invocation(program_), gather_,
                       iree_vm_variant_span_from_array(arguments),
                       iree_vm_variant_span_from_ptr(&result, 1));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    if (iree_status_is_ok(status)) {
      iree_hal_fence_t* ready = nullptr;
      status =
          iree_hal_fence_ptr_from_variant_borrowed(&types_, result, &ready);
      if (iree_status_is_ok(status)) {
        status = iree_hal_fence_query(ready);
      }
    }
    iree_vm_variant_reset(&result);
    return status;
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, invoke(-1, 0));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, invoke(1, 0));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, invoke(0, -1));
  IREE_EXPECT_OK(invoke(0, 0));
  loom_serve_parameters_module_bind_sources(module_, nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, invoke(0, 0));
  iree_vm_buffer_release(key);
}

INSTANTIATE_TEST_SUITE_P(Workers, ParameterModuleTest,
                         ::testing::Values(1u, 4u));

}  // namespace
