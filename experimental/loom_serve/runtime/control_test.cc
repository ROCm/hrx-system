// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <filesystem>
#include <string>

#include "experimental/loom_serve/runtime/execution.h"
#include "experimental/loom_serve/runtime/jit.h"
#include "experimental/loom_serve/runtime/module.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/drivers/init.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/sync.h"

IREE_FLAG(string, control_sources, "", "Portable control source catalog.");

namespace {

iree_hal_queue_t* SelectQueue(iree_hal_device_t* device,
                              iree_hal_queue_family_role_flags_t role) {
  const auto* queues =
      iree_hal_device_spec_queues(iree_hal_device_spec(device));
  for (iree_host_size_t i = 0; i < queues->family_count; ++i) {
    const auto& family = queues->families[i];
    if (family.provisioned_queue_count &&
        iree_all_bits_set(family.role_flags, role)) {
      return iree_hal_device_queue(device, i, 0);
    }
  }
  return nullptr;
}

struct State {
  // Prefill token count/position base/EOS, then decode position/count/EOS.
  std::array<int32_t, 3> control = {};
  // Current selected token followed by retained generated-token history.
  std::array<int32_t, 129> tokens = {};
  // Generated count, padding canaries, and EOS at element seven.
  std::array<int32_t, 8> progress = {};
};

class ControlTest
    : public ::testing::TestWithParam<iree_hal_command_buffer_mode_t> {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(CreateDevice());
    auto* dispatch =
        SelectQueue(device_, IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH);
    auto* transfer =
        SelectQueue(device_, IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER);
    ASSERT_NE(dispatch, nullptr);
    ASSERT_NE(transfer, nullptr);
    IREE_ASSERT_OK(loom_serve_execution_create(dispatch, transfer, allocator_,
                                               &execution_));
    const auto* family = iree_hal_queue_family(dispatch);
    directory_ =
        std::filesystem::path(FLAG_control_sources).parent_path().string();
    IREE_ASSERT_OK(loom_serve_jit_create(
        device_, dispatch, iree_make_cstring_view(directory_.c_str()), nullptr,
        allocator_, &jit_));
    IREE_ASSERT_OK(Prepare(family, "prefill_to_decode", &commands_[0]));
    IREE_ASSERT_OK(Prepare(family, "decode_transition", &commands_[1]));
    IREE_ASSERT_OK(Prepare(family, "device_continuation", &commands_[2]));
    for (auto& row : buffers_) {
      for (size_t i = 0; i < row.size(); ++i) {
        iree_hal_buffer_params_t params = {};
        params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
        params.usage =
            IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
        IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
            iree_hal_device_allocator(device_), params, kLengths[i], &row[i]));
      }
    }
    for (size_t i = 0; i < continuation_.buffers.size(); ++i) {
      iree_hal_buffer_params_t params = {};
      params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
      params.usage =
          IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
      if (i == 2) {
        params.usage |= IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS;
      }
      IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
          iree_hal_device_allocator(device_), params, kContinuationLengths[i],
          &continuation_.buffers[i]));
    }
    IREE_ASSERT_OK(CreateModel());
  }

  void TearDown() override {
    // Host transfer payloads are fixture members so even a failed assertion
    // leaves their storage alive until all accepted work has retired here.
    if (execution_) {
      IREE_EXPECT_OK(loom_serve_execution_drain(execution_));
    }
    iree_vm_process_release(process_);
    if (invocation_) {
      iree_vm_invocation_deinitialize(invocation_);
    }
    iree_vm_program_release(program_);
    iree_vm_module_release(native_module_);
    iree_vm_module_release(bytecode_module_);
    iree_vm_environment_free(environment_);
    for (auto& row : buffers_) {
      for (auto* buffer : row) {
        iree_hal_buffer_release(buffer);
      }
    }
    for (auto* buffer : continuation_.buffers) {
      iree_hal_buffer_release(buffer);
    }
    for (auto* command : commands_) {
      iree_hal_command_buffer_release(command);
    }
    loom_serve_jit_destroy(jit_);
    loom_serve_execution_release(execution_);
    iree_hal_device_group_release(group_);
    iree_hal_device_release(device_);
    iree_async_frontier_tracker_release(frontier_tracker_);
    iree_async_proactor_pool_release(proactor_pool_);
  }

  iree_status_t CreateDevice() {
    iree_hal_driver_registry_t* registry = nullptr;
    IREE_RETURN_IF_ERROR(
        iree_hal_driver_registry_allocate(allocator_, &registry));
    iree_status_t status = iree_hal_register_all_available_drivers(registry);
    if (iree_status_is_ok(status)) {
      status = iree_async_proactor_pool_create(
          iree_numa_node_count(), nullptr,
          iree_async_proactor_pool_options_default(), allocator_,
          &proactor_pool_);
    }
    if (iree_status_is_ok(status)) {
      auto params = iree_hal_device_create_params_default();
      params.proactor_pool = proactor_pool_;
      status = iree_hal_create_device(registry, IREE_SV("amdgpu"), &params,
                                      allocator_, &device_);
    }
    iree_hal_driver_registry_free(registry);
    if (iree_status_is_ok(status)) {
      status = iree_async_frontier_tracker_create(
          iree_async_frontier_tracker_options_default(), allocator_,
          &frontier_tracker_);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_group_create_from_device(
          device_, frontier_tracker_, allocator_, &group_);
    }
    return status;
  }

  iree_status_t Prepare(const iree_hal_queue_family_t* family, const char* root,
                        iree_hal_command_buffer_t** out_command) {
    const loomc_config_options_t config = {};
    loom_serve_jit_stage_t* stage = nullptr;
    auto status = loom_serve_jit_compile(jit_, iree_make_cstring_view(root),
                                         &config, &stage);
    if (iree_status_is_ok(status)) {
      status = loom_serve_jit_stage_record(stage, family, GetParam(), nullptr,
                                           out_command);
    }
    loom_serve_jit_stage_destroy(stage);
    return status;
  }

  iree_status_t CreateModel() {
    IREE_RETURN_IF_ERROR(
        iree_vm_environment_allocate(allocator_, &environment_));
    const iree_vm_ref_type_table_t* table = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_module_register_types(environment_, &table));
    IREE_RETURN_IF_ERROR(iree_hal_module_types_resolve(table, &types_));
    const loom_serve_stage_t stages[] = {
        {commands_[0], 3},
        {commands_[1], 3},
    };
    const iree_byte_span_t feedback[] = {
        iree_make_byte_span(outputs_[0].tokens.data(), kLengths[1]),
    };
    loom_serve_module_options_t options = {};
    options.binding_capacity = 12;
    options.stages = {IREE_ARRAYSIZE(stages), stages};
    options.feedback = {IREE_ARRAYSIZE(feedback), feedback};
    IREE_RETURN_IF_ERROR(loom_serve_module_create(&types_, execution_, options,
                                                  allocator_, &native_module_));
    const std::string path = directory_ + "/control.loom";
    iree_const_byte_span_t image;
    const iree_string_view_t roots[] = {IREE_SVL("step"), IREE_SVL("fork")};
    IREE_RETURN_IF_ERROR(loom_serve_jit_compile_vm(
        iree_make_cstring_view(path.c_str()), IREE_ARRAYSIZE(roots), roots,
        allocator_, &image));
    iree_status_t status = iree_vm_bytecode_module_create_trusted(
        environment_, IREE_SV("model"), {image, allocator_}, allocator_,
        &bytecode_module_);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(allocator_, (void*)image.data);
      return status;
    }
    iree_vm_module_t* libraries[] = {native_module_};
    IREE_RETURN_IF_ERROR(iree_vm_program_create(
        {bytecode_module_, iree_vm_module_span_from_array(libraries)},
        allocator_, &program_));
    IREE_RETURN_IF_ERROR(iree_vm_invocation_initialize(
        iree_make_byte_span(invocation_storage_.data(),
                            invocation_storage_.size()),
        &invocation_));
    IREE_RETURN_IF_ERROR(iree_vm_process_create(program_, invocation_,
                                                iree_vm_variant_span_empty(),
                                                allocator_, &process_));
    return iree_vm_process_lookup_function(process_, IREE_SV("model"),
                                           IREE_SV("step"), &step_);
  }

  iree_status_t InvokeFunction(iree_vm_function_t function,
                               iree_vm_variant_span_t arguments,
                               uint64_t* out_value) {
    iree_vm_variant_t results[1] = {};
    iree_status_t status =
        iree_vm_invoke(invocation_, function, arguments,
                       iree_vm_variant_span_from_array(results));
    iree_vm_variant_span_reset(arguments);
    int64_t value = 0;
    if (iree_status_is_ok(status)) {
      status = iree_vm_i64_from_variant(results[0], &value);
    }
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    if (iree_status_is_ok(status)) {
      *out_value = static_cast<uint64_t>(value);
    }
    return status;
  }

  iree_status_t Invoke(int initialize,
                       const std::array<iree_hal_buffer_t*, 3>& buffers,
                       uint64_t* out_value) {
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32(initialize),
        iree_hal_buffer_variant_from_ptr_borrowed(&types_, buffers[0]),
        iree_hal_buffer_variant_from_ptr_borrowed(&types_, buffers[1]),
        iree_hal_buffer_variant_from_ptr_borrowed(&types_, buffers[2]),
    };
    return InvokeFunction(step_, iree_vm_variant_span_from_array(arguments),
                          out_value);
  }

  iree_status_t InvokeFork(int next_stage, uint64_t* out_value) {
    iree_vm_function_t function = {};
    IREE_RETURN_IF_ERROR(iree_vm_process_lookup_function(
        process_, IREE_SV("model"), IREE_SV("fork"), &function));
    iree_vm_variant_t arguments[7] = {};
    arguments[0] = iree_vm_variant_from_i32(next_stage);
    for (size_t row = 0; row < 2; ++row) {
      for (size_t i = 0; i < 3; ++i) {
        arguments[1 + row * 3 + i] = iree_hal_buffer_variant_from_ptr_borrowed(
            &types_, buffers_[row][i]);
      }
    }
    return InvokeFunction(function, iree_vm_variant_span_from_array(arguments),
                          out_value);
  }

  iree_status_t Upload(size_t row, size_t first, size_t count,
                       uint64_t* out_value) {
    std::array<const void*, 3> data = {inputs_[row].control.data(),
                                       inputs_[row].tokens.data(),
                                       inputs_[row].progress.data()};
    std::array<iree_hal_transfer_operation_t, 3> operations = {};
    for (size_t i = 0; i < count; ++i) {
      const size_t slot = first + i;
      operations[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
      operations[i].upload.source = data[slot];
      operations[i].upload.target_buffer = buffers_[row][slot];
      operations[i].upload.length =
          count == 1 ? sizeof(int32_t) : kLengths[slot];
    }
    return loom_serve_execution_transfer(execution_, count, operations.data(),
                                         out_value);
  }

  iree_status_t Download(size_t row, uint64_t* out_value) {
    std::array<void*, 3> data = {outputs_[row].control.data(),
                                 outputs_[row].tokens.data(),
                                 outputs_[row].progress.data()};
    std::array<iree_hal_transfer_operation_t, 3> operations = {};
    for (size_t i = 0; i < operations.size(); ++i) {
      operations[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
      operations[i].download.source_buffer = buffers_[row][i];
      operations[i].download.target = data[i];
      operations[i].download.length = kLengths[i];
    }
    return loom_serve_execution_feedback(execution_, operations.size(),
                                         operations.data(), out_value);
  }

  // Exact byte extents of the three generation-state buffers.
  static constexpr std::array<size_t, 3> kLengths = {12, 516, 32};
  // Schedule, counters/canary, and the command product's transient count slab.
  static constexpr std::array<size_t, 3> kContinuationLengths = {96, 16, 16};
  // Host allocation policy used for all runtime objects.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Async services outliving the device and device group.
  iree_async_proactor_pool_t* proactor_pool_ = nullptr;
  // Cross-queue frontier registry retained through group destruction.
  iree_async_frontier_tracker_t* frontier_tracker_ = nullptr;
  // Owned target device.
  iree_hal_device_t* device_ = nullptr;
  // Owned domain for the device's semaphores.
  iree_hal_device_group_t* group_ = nullptr;
  // Shared ordered execution capability.
  loom_serve_execution_t* execution_ = nullptr;
  // Source directory shared by the catalog and VM program.
  std::string directory_;
  // One source index and compiler for all prepared control stages.
  loom_serve_jit_t* jit_ = nullptr;
  // Cached stage commands and the device-owned continuation loop.
  std::array<iree_hal_command_buffer_t*, 3> commands_ = {};
  // Two independently retained session states.
  std::array<std::array<iree_hal_buffer_t*, 3>, 2> buffers_ = {};
  // Persistent host upload storage, never recycled ahead of completion.
  std::array<State, 2> inputs_;
  // Persistent host readback storage, inspected only after completion.
  std::array<State, 2> outputs_;
  // Device-produced grids and all host payloads retained until the final drain.
  struct {
    // Schedule, state, and reusable transient storage in command binding order.
    std::array<iree_hal_buffer_t*, 3> buffers = {};
    // Disabled X/Y/Z/all-axis grids, each followed by one live grid.
    std::array<int32_t, 24> schedule = {0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
                                        1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1};
    // Initial producer, consumer and heartbeat counters, followed by a canary.
    std::array<int32_t, 4> input = {0, 0, 0, -12345};
    // Final counters/canary, inspected only after feedback completion.
    std::array<int32_t, 4> output = {};
  } continuation_;
  // Environment owning the type-registration namespace.
  iree_vm_environment_t* environment_ = nullptr;
  // Canonical HAL type handles.
  iree_hal_module_types_t types_ = {};
  // Owned native execution module.
  iree_vm_module_t* native_module_ = nullptr;
  // Owned compiled model-control module.
  iree_vm_module_t* bytecode_module_ = nullptr;
  // Single linked model program shared by all rows.
  iree_vm_program_t* program_ = nullptr;
  // Preallocated host call storage.
  alignas(iree_max_align_t) std::array<uint8_t, 16384> invocation_storage_ = {};
  // Invocation borrowing invocation_storage_.
  iree_vm_invocation_t* invocation_ = nullptr;
  // Single shared model process, not a process per session.
  iree_vm_process_t* process_ = nullptr;
  // Cold-resolved entry used without string lookup on every step.
  iree_vm_function_t step_ = iree_vm_function_null();
};

TEST_P(ControlTest, InterleavedRowsRetainHistory) {
  inputs_[0].control = {3, 17, 99};
  inputs_[1].control = {4, 100, 31};
  for (auto& input : inputs_) {
    input.tokens.fill(-1000);
    input.progress.fill(-2000);
  }
  auto expected = inputs_;
  std::array<int32_t, 2> positions = {20, 104};
  std::array<int32_t, 2> steps = {};
  uint64_t completion = 0;
  for (int tick = 0; tick < 8; ++tick) {
    for (size_t j = 0; j < 2; ++j) {
      const size_t row = (j + tick) % 2;
      // Row zero pauses for a tool while row one continues on shared commands.
      if (row == 0 && tick >= 2 && tick < 5) {
        continue;
      }
      const int32_t step = steps[row]++;
      const int32_t token = tick == 7 ? inputs_[row].control[2]
                                      : static_cast<int32_t>(10 * row + tick);
      inputs_[row].tokens[0] = token;
      IREE_ASSERT_OK(
          Upload(row, step == 0 ? 0 : 1, step == 0 ? 3 : 1, &completion));
      IREE_ASSERT_OK(Invoke(step == 0, buffers_[row], &completion));
      expected[row].tokens[0] = token;
      if (step == 0) {
        expected[row].tokens[1] = token;
      }
      expected[row].tokens[step + 2] = token;
      expected[row].control[0] = positions[row] + step + 1;
      expected[row].control[1] = step + 2;
      expected[row].progress[0] = step + 2;
      expected[row].progress[7] = tick == 7 ? 1 : 0;
    }
    // Both VM calls return before waiting. Device work owns the binding values
    // even though the shared process's native-call scratch has been
    // overwritten.
    IREE_ASSERT_OK(Download(0, &completion));
    IREE_ASSERT_OK(Download(1, &completion));
    IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
    for (size_t row = 0; row < 2; ++row) {
      EXPECT_EQ(outputs_[row].control, expected[row].control);
      EXPECT_EQ(outputs_[row].tokens, expected[row].tokens);
      EXPECT_EQ(outputs_[row].progress, expected[row].progress);
    }
  }
}

TEST_P(ControlTest, RejectedNativeCallLeavesTheTimelineUsable) {
  uint64_t completion = 777;
  auto invalid = buffers_[0];
  invalid[1] = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Invoke(1, invalid, &completion));
  EXPECT_EQ(completion, 777u);
  IREE_ASSERT_OK(loom_serve_execution_drain(execution_));
  inputs_[0].control = {3, 17, 99};
  inputs_[0].tokens[0] = 7;
  IREE_ASSERT_OK(Upload(0, 0, 3, &completion));
  EXPECT_EQ(completion, 1u);
  IREE_ASSERT_OK(Invoke(1, buffers_[0], &completion));
  EXPECT_EQ(completion, 2u);
  IREE_ASSERT_OK(Download(0, &completion));
  IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
  EXPECT_EQ(outputs_[0].control[0], 21);
  EXPECT_EQ(outputs_[0].control[1], 2);
  EXPECT_EQ(outputs_[0].tokens[1], 7);
  EXPECT_EQ(outputs_[0].tokens[2], 7);
}

TEST_P(ControlTest, FeedbackForkDoesNotAdvanceWork) {
  inputs_[0].control = {3, 17, 99};
  inputs_[0].tokens[0] = 7;
  inputs_[1].control = {4, 100, 31};
  inputs_[1].tokens[0] = 11;
  uint64_t work_completion = 0;
  uint64_t feedback_completion = 0;
  IREE_ASSERT_OK(Upload(0, 0, 3, &work_completion));
  IREE_ASSERT_OK(Upload(1, 0, 3, &work_completion));
  IREE_ASSERT_OK(Invoke(1, buffers_[0], &work_completion));
  EXPECT_EQ(work_completion, 3u);
  iree_hal_transfer_operation_t invalid = {};
  invalid.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
  invalid.download.source_buffer = buffers_[0][1];
  invalid.download.source_offset = kLengths[1];
  invalid.download.target = outputs_[0].tokens.data();
  invalid.download.length = sizeof(int32_t);
  feedback_completion = 777;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_serve_execution_feedback(execution_, 1, &invalid,
                                                      &feedback_completion));
  EXPECT_EQ(feedback_completion, 777u);
  IREE_ASSERT_OK(Download(0, &feedback_completion));
  EXPECT_EQ(feedback_completion, 1u);

  // Later model work consumes distinct state and may proceed independently of
  // row zero's feedback. Neither host output has been inspected or recycled.
  IREE_ASSERT_OK(Invoke(1, buffers_[1], &work_completion));
  EXPECT_EQ(work_completion, 4u);
  IREE_ASSERT_OK(Download(1, &feedback_completion));
  EXPECT_EQ(feedback_completion, 2u);
  IREE_ASSERT_OK(loom_serve_execution_drain(execution_));
  EXPECT_EQ(outputs_[0].control[0], 21);
  EXPECT_EQ(outputs_[0].control[1], 2);
  EXPECT_EQ(outputs_[0].tokens[2], 7);
  EXPECT_EQ(outputs_[1].control[0], 105);
  EXPECT_EQ(outputs_[1].control[1], 2);
  EXPECT_EQ(outputs_[1].tokens[2], 11);
}

TEST_P(ControlTest, IndexedArityFamilyRejectsBeforeSubmission) {
  for (int count = 0; count <= 12; ++count) {
    SCOPED_TRACE(count);
    const std::string name = "execute_" + std::to_string(count);
    iree_vm_function_t function = {};
    IREE_ASSERT_OK(iree_vm_process_lookup_function(
        process_, IREE_SV("runner"), iree_make_cstring_view(name.c_str()),
        &function));
    iree_vm_variant_t arguments[13] = {};
    arguments[0] = iree_vm_variant_from_i32(count == 3 ? -1 : 0);
    for (int i = 0; i < count; ++i) {
      arguments[i + 1] = iree_hal_buffer_variant_from_ptr_borrowed(
          &types_, buffers_[0][i % 3]);
    }
    uint64_t completion = 777;
    IREE_EXPECT_STATUS_IS(
        count == 3 ? IREE_STATUS_OUT_OF_RANGE : IREE_STATUS_INVALID_ARGUMENT,
        InvokeFunction(function,
                       iree_vm_variant_span_from_ptr(arguments, count + 1),
                       &completion));
    EXPECT_EQ(completion, 777u);
  }
  inputs_[0].control = {3, 17, 99};
  inputs_[0].tokens[0] = 7;
  uint64_t completion = 0;
  IREE_ASSERT_OK(Upload(0, 0, 3, &completion));
  EXPECT_EQ(completion, 1u);
  IREE_ASSERT_OK(Invoke(1, buffers_[0], &completion));
  EXPECT_EQ(completion, 2u);
}

TEST_P(ControlTest, FeedbackChecksBothExtentsAndAllowsSubranges) {
  iree_vm_function_t function = {};
  IREE_ASSERT_OK(iree_vm_process_lookup_function(
      process_, IREE_SV("runner"), IREE_SV("feedback"), &function));
  struct Range {
    // Host feedback slot supplied by source.
    int32_t slot;
    // Byte range in the device source.
    int64_t offset;
    // Requested byte count in both source and destination.
    int64_t length;
  };
  const Range invalid[] = {
      {1, 0, 4}, {0, 0, 520}, {0, 516, 4}, {0, -1, 4}, {0, 0, -1}};
  for (auto range : invalid) {
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32(range.slot),
        iree_hal_buffer_variant_from_ptr_borrowed(&types_, buffers_[0][1]),
        iree_vm_variant_from_i64(range.offset),
        iree_vm_variant_from_i64(range.length),
    };
    uint64_t completion = 777;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        InvokeFunction(function, iree_vm_variant_span_from_array(arguments),
                       &completion));
    EXPECT_EQ(completion, 777u);
  }
  iree_vm_variant_t null_arguments[] = {
      iree_vm_variant_from_i32(0),
      iree_hal_buffer_variant_from_ptr_borrowed(&types_, nullptr),
      iree_vm_variant_from_i64(0),
      iree_vm_variant_from_i64(4),
  };
  uint64_t completion = 777;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      InvokeFunction(function, iree_vm_variant_span_from_array(null_arguments),
                     &completion));
  EXPECT_EQ(completion, 777u);
  inputs_[0].tokens[1] = 41;
  outputs_[0].tokens.fill(-12345);
  IREE_ASSERT_OK(Upload(0, 0, 3, &completion));
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i32(0),
      iree_hal_buffer_variant_from_ptr_borrowed(&types_, buffers_[0][1]),
      iree_vm_variant_from_i64(4),
      iree_vm_variant_from_i64(4),
  };
  IREE_ASSERT_OK(InvokeFunction(
      function, iree_vm_variant_span_from_array(arguments), &completion));
  EXPECT_EQ(completion, 1u);
  IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
  EXPECT_EQ(outputs_[0].tokens[0], 41);
  for (size_t i = 1; i < outputs_[0].tokens.size(); ++i) {
    EXPECT_EQ(outputs_[0].tokens[i], -12345);
  }
}

TEST_P(ControlTest, SourceForkAndPartialFailureRetireBorrowedStorage) {
  inputs_[0].control = {3, 17, 99};
  inputs_[0].tokens[0] = 7;
  inputs_[1].control = {4, 100, 31};
  inputs_[1].tokens[0] = 11;
  uint64_t completion = 0;
  IREE_ASSERT_OK(Upload(0, 0, 3, &completion));
  IREE_ASSERT_OK(Upload(1, 0, 3, &completion));
  IREE_ASSERT_OK(InvokeFork(0, &completion));
  // Two uploads, then producer and independent continuation; the source's
  // feedback call did not advance the work frontier.
  EXPECT_EQ(completion, 4u);
  IREE_ASSERT_OK(loom_serve_execution_drain(execution_));
  EXPECT_EQ(outputs_[0].tokens[2], 7);
  IREE_ASSERT_OK(Download(1, &completion));
  IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
  EXPECT_EQ(outputs_[1].control[0], 105);
  EXPECT_EQ(outputs_[1].tokens[2], 11);

  // Reset the producer, then reject the final call after producer and feedback
  // have both been accepted. Storage stays alive through the explicit join.
  inputs_[0].tokens[0] = 19;
  IREE_ASSERT_OK(Upload(0, 0, 3, &completion));
  completion = 777;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, InvokeFork(99, &completion));
  EXPECT_EQ(completion, 777u);
  IREE_ASSERT_OK(loom_serve_execution_drain(execution_));
  EXPECT_EQ(outputs_[0].tokens[2], 19);
  IREE_ASSERT_OK(Invoke(0, buffers_[0], &completion));
  EXPECT_EQ(completion, 7u);
  IREE_ASSERT_OK(Download(0, &completion));
  IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
  EXPECT_EQ(outputs_[0].control[0], 22);
  EXPECT_EQ(outputs_[0].tokens[3], 19);
}

TEST_P(ControlTest, DeviceProducedCountsGateConcurrentContinuation) {
  std::array<iree_hal_transfer_operation_t, 2> uploads = {};
  const void* sources[] = {continuation_.schedule.data(),
                           continuation_.input.data()};
  for (size_t i = 0; i < uploads.size(); ++i) {
    uploads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
    uploads[i].upload.source = sources[i];
    uploads[i].upload.target_buffer = continuation_.buffers[i];
    uploads[i].upload.length = kContinuationLengths[i];
  }
  uint64_t completion = 0;
  IREE_ASSERT_OK(loom_serve_execution_transfer(execution_, uploads.size(),
                                               uploads.data(), &completion));
  std::array<iree_hal_buffer_binding_t, 3> bindings = {};
  for (size_t i = 0; i < bindings.size(); ++i) {
    bindings[i] = {continuation_.buffers[i], 0, kContinuationLengths[i]};
  }
  // Every replay contains eight producer/consumer epochs sharing one count
  // tuple. Submission dependencies order replays; no host result gates them.
  for (int i = 0; i < 16; ++i) {
    IREE_ASSERT_OK(loom_serve_execution_execute(
        execution_, commands_[2], {bindings.size(), bindings.data()},
        &completion));
  }
  iree_hal_transfer_operation_t download = {};
  download.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
  download.download.source_buffer = continuation_.buffers[1];
  download.download.target = continuation_.output.data();
  download.download.length = kContinuationLengths[1];
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(execution_, 1, &download, &completion));
  IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
  const std::array<int32_t, 4> expected = {128, 64, 128, -12345};
  EXPECT_EQ(continuation_.output, expected);
}

INSTANTIATE_TEST_SUITE_P(
    RecordingPolicy, ControlTest,
    ::testing::Values(IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
                      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA));

}  // namespace
