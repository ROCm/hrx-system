// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

#include "experimental/loom_serve/command.h"
#include "experimental/loom_serve/control_data.h"
#include "experimental/loom_serve/execution.h"
#include "experimental/loom_serve/module.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/drivers/init.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/sync.h"

IREE_FLAG(string, control_manifest, "",
          "Generated artifact-set path used to locate sibling command files.");

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

iree_const_byte_span_t Image(const char* name) {
  const auto* files = loom_serve_control_data_create();
  for (size_t i = 0; i < loom_serve_control_data_size(); ++i) {
    if (std::filesystem::path(files[i].name).filename() == name) {
      return iree_make_const_byte_span(files[i].data, files[i].size);
    }
  }
  ADD_FAILURE() << "missing embedded artifact " << name;
  return iree_const_byte_span_empty();
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
    IREE_ASSERT_OK(LoadEntry(family, "prefill_finish_control.hsaco",
                             "qwen38_prefill_finish_control", &entries_[0]));
    IREE_ASSERT_OK(LoadEntry(family, "decode_commit_token.hsaco",
                             "qwen38_decode_commit_token", &entries_[1]));
    // Program-local projections are emitted by this source's command product:
    // prefill references finish/commit, while decode references commit alone.
    IREE_ASSERT_OK(Prepare(family, 0, 2, entries_.data(), &commands_[0]));
    IREE_ASSERT_OK(Prepare(family, 1, 1, &entries_[1], &commands_[1]));
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
    for (auto* command : commands_) {
      iree_hal_command_buffer_release(command);
    }
    for (auto entry : entries_) {
      iree_hal_executable_release(entry.executable);
    }
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

  iree_status_t LoadEntry(const iree_hal_queue_family_t* family,
                          const char* image, const char* symbol,
                          loom_serve_command_entry_t* entry) {
    iree_hal_executable_target_selection_t selection = {};
    selection.family = IREE_SV("amdgpu");
    selection.kind_flags = IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_EXACT;
    selection.physical_device_affinity =
        iree_hal_queue_family_spec(family)->physical_device_affinity;
    const auto target = iree_hal_device_spec_select_executable_target(
        iree_hal_device_spec(device_), &selection);
    if (target.outcome !=
        IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "no unambiguous AMDGPU executable target");
    }
    iree_hal_executable_load_params_t params;
    iree_hal_executable_load_params_initialize(&params);
    params.executable_data = Image(image);
    IREE_RETURN_IF_ERROR(iree_hal_executable_load(family, target.target,
                                                  &params, &entry->executable));
    return iree_hal_executable_lookup_function_by_name(
        entry->executable, iree_make_cstring_view(symbol), &entry->function);
  }

  iree_status_t Prepare(const iree_hal_queue_family_t* family, int index,
                        size_t entry_count,
                        const loom_serve_command_entry_t* entries,
                        iree_hal_command_buffer_t** out_command) {
    const auto path =
        std::filesystem::path(FLAG_control_manifest).parent_path() /
        "control_commands" / ("program-" + std::to_string(index) + ".loomcmd");
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      return iree_make_status(IREE_STATUS_NOT_FOUND, "%s", path.c_str());
    }
    std::vector<uint8_t> bytes(std::istreambuf_iterator<char>(input), {});
    if (input.bad()) {
      return iree_make_status(IREE_STATUS_DATA_LOSS, "%s", path.c_str());
    }
    loom_cmd_program_t parsed;
    IREE_RETURN_IF_ERROR(loom_cmd_program_parse(
        iree_make_const_byte_span(bytes.data(), bytes.size()), &parsed));
    return loom_serve_command_create(family, GetParam(), &parsed, 0, nullptr,
                                     entry_count, entries, allocator_,
                                     out_command);
  }

  iree_status_t CreateModel() {
    IREE_RETURN_IF_ERROR(
        iree_vm_environment_allocate(allocator_, &environment_));
    const iree_vm_ref_type_table_t* table = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_module_register_types(environment_, &table));
    IREE_RETURN_IF_ERROR(iree_hal_module_types_resolve(table, &types_));
    const loom_serve_stage_t stages[] = {
        {IREE_SVL("prefill_to_decode"), commands_[0], 3},
        {IREE_SVL("decode_transition"), commands_[1], 3},
    };
    IREE_RETURN_IF_ERROR(
        loom_serve_module_create(&types_, execution_, IREE_ARRAYSIZE(stages),
                                 stages, allocator_, &native_module_));
    const auto source = Image("control_model.vm");
    uint8_t* image = nullptr;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        allocator_, source.data_length, reinterpret_cast<void**>(&image)));
    std::memcpy(image, source.data, source.data_length);
    iree_status_t status = iree_vm_bytecode_module_create(
        environment_, IREE_SV("model"),
        {iree_make_const_byte_span(image, source.data_length), allocator_},
        allocator_, &bytecode_module_);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(allocator_, image);
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

  iree_status_t Invoke(int initialize,
                       const std::array<iree_hal_buffer_t*, 3>& buffers,
                       uint64_t* out_value) {
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32(initialize),
        iree_hal_buffer_variant_from_ptr_borrowed(&types_, buffers[0]),
        iree_hal_buffer_variant_from_ptr_borrowed(&types_, buffers[1]),
        iree_hal_buffer_variant_from_ptr_borrowed(&types_, buffers[2]),
    };
    iree_vm_variant_t results[1] = {};
    iree_status_t status = iree_vm_invoke(
        invocation_, step_, iree_vm_variant_span_from_array(arguments),
        iree_vm_variant_span_from_array(results));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
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
  // One loaded implementation of each generation-state kernel.
  std::array<loom_serve_command_entry_t, 2> entries_ = {};
  // One cached command buffer per stage, shared across both rows.
  std::array<iree_hal_command_buffer_t*, 2> commands_ = {};
  // Two independently retained session states.
  std::array<std::array<iree_hal_buffer_t*, 3>, 2> buffers_ = {};
  // Persistent host upload storage, never recycled ahead of completion.
  std::array<State, 2> inputs_;
  // Persistent host readback storage, inspected only after completion.
  std::array<State, 2> outputs_;
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

INSTANTIATE_TEST_SUITE_P(
    RecordingPolicy, ControlTest,
    ::testing::Values(IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
                      IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA));

}  // namespace
