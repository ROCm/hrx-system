// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/jit.h"

#include <array>
#include <filesystem>

#include "experimental/loom_serve/runtime/device.h"
#include "experimental/loom_serve/runtime/execution.h"
#include "experimental/loom_serve/runtime/module.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/sync.h"

IREE_FLAG(string, jit_sources, "", "Portable test source catalog.");

namespace {

class JitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const loom_serve_device_options_t options = {.uri = IREE_SV("amdgpu")};
    IREE_ASSERT_OK(
        loom_serve_device_create(&options, &device_owner_, allocator_));
    device_ = loom_serve_device_handle(device_owner_);
    dispatch_ = loom_serve_device_dispatch_queue(device_owner_);
    execution_ = loom_serve_device_execution(device_owner_);
    directory_ = std::filesystem::path(FLAG_jit_sources).parent_path().string();
    IREE_ASSERT_OK(loom_serve_jit_create(
        device_, dispatch_, iree_make_cstring_view(directory_.c_str()), nullptr,
        allocator_, &jit_));
  }

  void TearDown() override {
    if (execution_) {
      IREE_EXPECT_OK(loom_serve_execution_drain(execution_));
    }
    iree_vm_process_release(process_);
    if (invocation_) {
      iree_vm_invocation_deinitialize(invocation_);
    }
    iree_vm_program_release(program_);
    iree_vm_module_release(native_);
    iree_vm_module_release(bytecode_);
    iree_vm_environment_free(environment_);
    for (auto* command : commands_) {
      iree_hal_command_buffer_release(command);
    }
    iree_hal_buffer_release(buffer_);
    loom_serve_jit_destroy(jit_);
    IREE_EXPECT_OK(loom_serve_device_destroy(device_owner_));
  }

  // Null uses the source provider; text overlays its value for this request.
  iree_status_t Compile(const char* delta, size_t index,
                        iree_string_view_t root = IREE_SV("advance"),
                        const char* temporary_bytes = nullptr) {
    loomc_config_binding_t bindings[2] = {};
    size_t count = 0;
    if (delta) {
      bindings[count++] = {loomc_make_cstring_view("increment.delta"),
                           loomc_make_cstring_view(delta)};
    }
    if (temporary_bytes) {
      bindings[count++] = {loomc_make_cstring_view("increment.temporary_bytes"),
                           loomc_make_cstring_view(temporary_bytes)};
    }
    const loomc_config_options_t config = {
        bindings, count, {}, LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED};
    loom_serve_jit_stage_t* stage = nullptr;
    auto status = loom_serve_jit_compile(jit_, root, &config, &stage);
    if (iree_status_is_ok(status)) {
      transients_[index] =
          loom_serve_jit_stage_program(stage)->requirements.transient;
      status = loom_serve_jit_stage_record(
          stage, iree_hal_queue_family(dispatch_),
          IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, nullptr, &commands_[index]);
    }
    loom_serve_jit_stage_destroy(stage);
    return status;
  }

  iree_status_t PrepareVm() {
    IREE_RETURN_IF_ERROR(
        iree_vm_environment_allocate(allocator_, &environment_));
    const iree_vm_ref_type_table_t* table = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_module_register_types(environment_, &table));
    IREE_RETURN_IF_ERROR(iree_hal_module_types_resolve(table, &types_));
    const loom_serve_stage_t stages[] = {
        {commands_[0], 1, transients_[0]},
        {commands_[1], 1, transients_[1]},
    };
    loom_serve_module_options_t options = {};
    options.binding_capacity = 1;
    options.stages = {commands_[1] ? 2u : 1u, stages};
    IREE_RETURN_IF_ERROR(loom_serve_module_create(&types_, execution_, options,
                                                  allocator_, &native_));
    const std::string path = directory_ + "/control.loom";
    iree_const_byte_span_t image;
    const iree_string_view_t roots[] = {IREE_SVL("step"),
                                        IREE_SVL("selected_step")};
    IREE_RETURN_IF_ERROR(loom_serve_jit_compile_vm(
        iree_make_cstring_view(path.c_str()), IREE_ARRAYSIZE(roots), roots,
        allocator_, &image));
    auto status = iree_vm_bytecode_module_create_trusted(
        environment_, IREE_SV("model"), {image, allocator_}, allocator_,
        &bytecode_);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(allocator_, (void*)image.data);
      return status;
    }
    iree_vm_module_t* libraries[] = {native_};
    IREE_RETURN_IF_ERROR(iree_vm_program_create(
        {bytecode_, iree_vm_module_span_from_array(libraries)}, allocator_,
        &program_));
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

  // Host allocation policy for all fixture resources.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Production device owner outliving all compiler, VM and I/O resources.
  loom_serve_device_t* device_owner_ = nullptr;
  // Borrowed live device used for profile extraction.
  iree_hal_device_t* device_ = nullptr;
  // Exact dispatch queue borrowed from device.
  iree_hal_queue_t* dispatch_ = nullptr;
  // Production submission/feedback timelines borrowed from device_owner_.
  loom_serve_execution_t* execution_ = nullptr;
  // Portable source location from runfiles.
  std::string directory_;
  // Model-resident compiler shared by both variants.
  loom_serve_jit_t* jit_ = nullptr;
  // Independently prepared variants retaining executable ownership.
  std::array<iree_hal_command_buffer_t*, 2> commands_ = {};
  // Compiler requirements retained independently of the stage/compiler.
  std::array<loom_cmd_program_transient_requirement_t, 2> transients_ = {};
  // Device state shared across the variants.
  iree_hal_buffer_t* buffer_ = nullptr;
  // Upload backing retained until teardown drains accepted work.
  std::array<int32_t, 8> input_ = {100, 100, 100, 100, 100, 100, 100, 100};
  // Feedback backing retained until teardown drains accepted work.
  std::array<int32_t, 8> output_ = {};
  // VM reference type registry.
  iree_vm_environment_t* environment_ = nullptr;
  // HAL types borrowed from the environment.
  iree_hal_module_types_t types_ = {};
  // Runner module retaining the first variant.
  iree_vm_module_t* native_ = nullptr;
  // JIT bytecode module owning its image.
  iree_vm_module_t* bytecode_ = nullptr;
  // One linked model program.
  iree_vm_program_t* program_ = nullptr;
  // Aligned invocation backing.
  alignas(iree_max_align_t) std::array<uint8_t, 16384> invocation_storage_ = {};
  // Invocation borrowing its backing.
  iree_vm_invocation_t* invocation_ = nullptr;
  // One shared model process.
  iree_vm_process_t* process_ = nullptr;
  // Resolved control entry point.
  iree_vm_function_t step_ = {};
};

TEST_F(JitTest, SourceDefaultsAndOverridesShareVmAndGpuState) {
  // A failed native task must join its accepted siblings before compiler reuse.
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Compile("3", 0, IREE_SV("invalid_native")));
  ASSERT_EQ(commands_[0], nullptr);
  IREE_ASSERT_OK(Compile(nullptr, 0));
  IREE_ASSERT_OK(Compile("7", 1));
  IREE_ASSERT_OK(PrepareVm());
  // Prepared commands must outlive all compiler/source storage.
  loom_serve_jit_destroy(jit_);
  jit_ = nullptr;
  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device_), params, sizeof(input_), &buffer_));
  iree_hal_transfer_operation_t upload = {};
  upload.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
  upload.upload.source = input_.data();
  upload.upload.target_buffer = buffer_;
  upload.upload.length = sizeof(input_);
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(execution_, 1, &upload, &completion));
  iree_vm_variant_t arguments[] = {
      iree_hal_buffer_variant_from_ptr_borrowed(&types_, buffer_)};
  iree_vm_variant_t results[1] = {};
  auto status = iree_vm_invoke(invocation_, step_,
                               iree_vm_variant_span_from_array(arguments),
                               iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  IREE_ASSERT_OK(status);
  const iree_hal_buffer_binding_t binding = {buffer_, 0, sizeof(input_)};
  IREE_ASSERT_OK(loom_serve_execution_execute(execution_, commands_[1],
                                              {1, &binding}, &completion));
  iree_hal_transfer_operation_t download = {};
  download.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
  download.download.source_buffer = buffer_;
  download.download.target = output_.data();
  download.download.length = sizeof(output_);
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(execution_, 1, &download, &completion));
  IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution_, completion));
  for (size_t i = 0; i < output_.size(); ++i) {
    EXPECT_EQ(output_[i], 112 + 2 * i) << "specialized element " << i;
  }
}

TEST_F(JitTest, InvalidConfigurationLeavesCompilerReusable) {
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Compile("99", 0));
  ASSERT_EQ(commands_[0], nullptr);
  IREE_ASSERT_OK(Compile("3", 0));
}

TEST_F(JitTest, InvalidRootLeavesCompilerReusable) {
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        Compile("3", 0, IREE_SV("absent")));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        Compile("3", 0, IREE_SV("missing")));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        Compile("3", 0, IREE_SV("ambiguous")));
  IREE_ASSERT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Compile("3", 0, IREE_SV("not_command")));
  ASSERT_EQ(commands_[0], nullptr);
  IREE_ASSERT_OK(Compile("3", 0));
}

TEST_F(JitTest, CommandWithoutNativeRequests) {
  IREE_ASSERT_OK(Compile("3", 0, IREE_SV("idle")));
}

TEST_F(JitTest, UnequalPrivateWorkspacesReusePoolAndTrimWithoutLosingOutput) {
  IREE_ASSERT_OK(Compile("5", 0, IREE_SV("temporary_advance"), "64"));
  // A non-class-boundary maximum exercises the pool's backing geometry.
  IREE_ASSERT_OK(Compile("7", 1, IREE_SV("temporary_advance"), "4160"));
  ASSERT_EQ(transients_[0].required_byte_length, 64u);
  ASSERT_EQ(transients_[1].required_byte_length, 4160u);
  IREE_ASSERT_OK(PrepareVm());
  EXPECT_EQ(
      loom_serve_execution_workspace_statistics(execution_).bytes_committed,
      0u);
  loom_serve_jit_destroy(jit_);
  jit_ = nullptr;
  iree_vm_function_t selected_step;
  IREE_ASSERT_OK(iree_vm_process_lookup_function(
      process_, IREE_SV("model"), IREE_SV("selected_step"), &selected_step));
  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device_), params, sizeof(input_), &buffer_));
  iree_hal_transfer_operation_t upload = {};
  upload.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
  upload.upload = {input_.data(), buffer_, 0, sizeof(input_)};
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(execution_, 1, &upload, &completion));
  const int32_t stages[] = {0, 1, 0, 1};
  int32_t expected = 100;
  for (int32_t stage : stages) {
    iree_vm_variant_t arguments[] = {
        iree_vm_variant_from_i32(stage),
        iree_hal_buffer_variant_from_ptr_borrowed(&types_, buffer_)};
    iree_vm_variant_t results[1] = {};
    auto status = iree_vm_invoke(invocation_, selected_step,
                                 iree_vm_variant_span_from_array(arguments),
                                 iree_vm_variant_span_from_array(results));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
    iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
    IREE_ASSERT_OK(status);
    expected += stage ? 7 : 5;
  }
  iree_hal_transfer_operation_t download = {};
  download.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
  download.download = {buffer_, 0, output_.data(), sizeof(output_)};
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(execution_, 1, &download, &completion));
  IREE_ASSERT_OK(loom_serve_execution_drain(execution_));
  const auto statistics = loom_serve_execution_workspace_statistics(execution_);
  EXPECT_EQ(statistics.reservation_count, 0u);
  EXPECT_EQ(statistics.reserve_count, 4u);
  EXPECT_EQ(statistics.release_count, 4u);
  EXPECT_GT(statistics.reuse_count, 0u);
  EXPECT_GT(statistics.bytes_committed, 0u);
  IREE_ASSERT_OK(loom_serve_execution_trim_workspace(execution_));
  EXPECT_EQ(
      loom_serve_execution_workspace_statistics(execution_).bytes_committed,
      0u);
  // Feedback owns the final output independently of private workspace reuse.
  for (int32_t value : output_) {
    EXPECT_EQ(value, expected);
  }
}

}  // namespace
