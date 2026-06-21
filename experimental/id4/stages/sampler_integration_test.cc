// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <vector>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/sampler.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct SemaphoreListStorage {
  // Semaphore carried by this single-entry list.
  iree_hal_semaphore_t* semaphore = nullptr;
  // Payload value paired with the semaphore.
  uint64_t payload_value = 0;

  iree_hal_semaphore_list_t list() {
    return iree_hal_semaphore_list_t{
        // One semaphore is carried by this stack-backed list.
        /*.count=*/1,
        // Stack-backed semaphore pointer array.
        /*.semaphores=*/&semaphore,
        // Stack-backed payload value array.
        /*.payload_values=*/&payload_value,
    };
  }
};

struct BufferBindingSet {
  // Number of bindings allocated from the plan.
  iree_host_size_t count = 0;
  // Owned HAL buffers backing each binding.
  iree_hal_buffer_t** buffers = nullptr;
  // Binding table entries in plan order.
  iree_hal_buffer_binding_t* bindings = nullptr;

  ~BufferBindingSet() { reset(); }

  void reset() {
    if (buffers) {
      for (iree_host_size_t i = 0; i < count; ++i) {
        iree_hal_buffer_release(buffers[i]);
      }
    }
    iree_allocator_free(iree_allocator_system(), buffers);
    iree_allocator_free(iree_allocator_system(), bindings);
    count = 0;
    buffers = nullptr;
    bindings = nullptr;
  }
};

static iree_status_t CreateSamplerStage(
    const id4::test::LiveStageContext& context,
    id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = nullptr;

  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = context.device_group.get();
  services.executable_cache = context.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_sampler_denoise_stage_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = context.kernel_cache.get();
  return id4_sampler_denoise_stage_create(&create_options,
                                          iree_allocator_system(), out_stage);
}

static iree_status_t AllocateBufferBindingSet(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    iree_host_size_t count,
    const id4_pipeline_tensor_layout_t* (*layout_at)(iree_host_size_t index,
                                                     const void* user_data),
    const void* user_data, BufferBindingSet* out_binding_set) {
  IREE_ASSERT_ARGUMENT(out_binding_set);
  out_binding_set->reset();
  if (count == 0) return iree_ok_status();

  iree_status_t status = iree_allocator_malloc_array(
      iree_allocator_system(), count, sizeof(out_binding_set->buffers[0]),
      reinterpret_cast<void**>(&out_binding_set->buffers));
  if (iree_status_is_ok(status)) {
    std::memset(out_binding_set->buffers, 0,
                count * sizeof(out_binding_set->buffers[0]));
    status = iree_allocator_malloc_array(
        iree_allocator_system(), count, sizeof(out_binding_set->bindings[0]),
        reinterpret_cast<void**>(&out_binding_set->bindings));
  }
  if (iree_status_is_ok(status)) {
    std::memset(out_binding_set->bindings, 0,
                count * sizeof(out_binding_set->bindings[0]));
    out_binding_set->count = count;
  }

  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const id4_pipeline_tensor_layout_t* layout = layout_at(i, user_data);
    if (!layout) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "missing tensor layout %" PRIhsz, i);
      break;
    }
    iree_hal_buffer_params_t params;
    std::memset(&params, 0, sizeof(params));
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
                   IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE;
    params.queue_affinity = queue_affinity;
    params.min_alignment = layout->alignment ? layout->alignment : 1;
    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), params, layout->byte_length,
        &out_binding_set->buffers[i]);
    if (iree_status_is_ok(status)) {
      out_binding_set->bindings[i] = iree_hal_buffer_binding_t{
          // Tensor buffer supplied in plan order.
          /*.buffer=*/out_binding_set->buffers[i],
          // Test buffers are exact standalone allocations.
          /*.offset=*/0,
          // Full planned tensor byte range.
          /*.length=*/layout->byte_length,
      };
    }
  }
  if (!iree_status_is_ok(status)) {
    out_binding_set->reset();
  }
  return status;
}

static const id4_pipeline_tensor_layout_t* BoundaryLayoutAt(
    iree_host_size_t index, const void* user_data) {
  const id4_pipeline_plan_t* plan =
      static_cast<const id4_pipeline_plan_t*>(user_data);
  const id4_pipeline_boundary_tensor_plan_t* boundary =
      id4_pipeline_plan_boundary_tensor_at(plan, index);
  return boundary ? &boundary->layout : nullptr;
}

static const id4_pipeline_tensor_layout_t* DiagnosticTapLayoutAt(
    iree_host_size_t index, const void* user_data) {
  const id4_pipeline_plan_t* plan =
      static_cast<const id4_pipeline_plan_t*>(user_data);
  const id4_pipeline_diagnostic_tap_plan_t* diagnostic_tap =
      id4_pipeline_plan_diagnostic_tap_at(plan, index);
  return diagnostic_tap ? &diagnostic_tap->layout : nullptr;
}

static iree_status_t AllocateBoundaryBindings(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, BufferBindingSet* out_binding_set) {
  return AllocateBufferBindingSet(device, queue_affinity,
                                  id4_pipeline_plan_boundary_tensor_count(plan),
                                  BoundaryLayoutAt, plan, out_binding_set);
}

static iree_status_t AllocateDiagnosticTapBindings(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, BufferBindingSet* out_binding_set) {
  return AllocateBufferBindingSet(device, queue_affinity,
                                  id4_pipeline_plan_diagnostic_tap_count(plan),
                                  DiagnosticTapLayoutAt, plan, out_binding_set);
}

static iree_status_t FindBoundaryBinding(
    const id4_pipeline_plan_t* plan, const BufferBindingSet& binding_set,
    iree_string_view_t name, iree_hal_buffer_binding_t* out_binding) {
  for (iree_host_size_t i = 0;
       i < id4_pipeline_plan_boundary_tensor_count(plan); ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (boundary && iree_string_view_equal(boundary->layout.name, name)) {
      *out_binding = binding_set.bindings[i];
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "boundary tensor `%.*s` not found",
                          static_cast<int>(name.size), name.data);
}

static iree_status_t FindDiagnosticTapBinding(
    const id4_pipeline_plan_t* plan, const BufferBindingSet& binding_set,
    iree_string_view_t name, iree_hal_buffer_binding_t* out_binding) {
  for (iree_host_size_t i = 0; i < id4_pipeline_plan_diagnostic_tap_count(plan);
       ++i) {
    const id4_pipeline_diagnostic_tap_plan_t* diagnostic_tap =
        id4_pipeline_plan_diagnostic_tap_at(plan, i);
    if (diagnostic_tap && iree_string_view_equal(diagnostic_tap->name, name)) {
      *out_binding = binding_set.bindings[i];
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "diagnostic tap `%.*s` not found",
                          static_cast<int>(name.size), name.data);
}

static iree_status_t QueueUpdateBinding(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_buffer_binding_t* binding, const void* source_data,
    iree_host_size_t source_length, iree_hal_semaphore_t* semaphore,
    uint64_t* inout_payload_value) {
  if (source_length != binding->length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source length %" PRIhsz " does not match binding length %" PRIu64,
        source_length, static_cast<uint64_t>(binding->length));
  }
  iree_hal_semaphore_list_t wait_list = iree_hal_semaphore_list_empty();
  SemaphoreListStorage wait_storage;
  wait_storage.semaphore = semaphore;
  wait_storage.payload_value = *inout_payload_value;
  if (wait_storage.payload_value != 0) {
    wait_list = wait_storage.list();
  }
  SemaphoreListStorage signal_storage;
  signal_storage.semaphore = semaphore;
  signal_storage.payload_value = wait_storage.payload_value + 1;
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_update(
      device, queue_affinity, wait_list, signal_storage.list(), source_data,
      /*source_offset=*/0, binding->buffer, binding->offset, binding->length,
      IREE_HAL_UPDATE_FLAG_NONE));
  *inout_payload_value = signal_storage.payload_value;
  return iree_ok_status();
}

static iree_status_t QueueFillBinding(iree_hal_device_t* device,
                                      iree_hal_queue_affinity_t queue_affinity,
                                      const iree_hal_buffer_binding_t* binding,
                                      const void* pattern,
                                      iree_host_size_t pattern_length,
                                      iree_hal_semaphore_t* semaphore,
                                      uint64_t* inout_payload_value) {
  iree_hal_semaphore_list_t wait_list = iree_hal_semaphore_list_empty();
  SemaphoreListStorage wait_storage;
  wait_storage.semaphore = semaphore;
  wait_storage.payload_value = *inout_payload_value;
  if (wait_storage.payload_value != 0) {
    wait_list = wait_storage.list();
  }
  SemaphoreListStorage signal_storage;
  signal_storage.semaphore = semaphore;
  signal_storage.payload_value = wait_storage.payload_value + 1;
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_fill(
      device, queue_affinity, wait_list, signal_storage.list(), binding->buffer,
      binding->offset, binding->length, pattern, pattern_length,
      IREE_HAL_FILL_FLAG_NONE));
  *inout_payload_value = signal_storage.payload_value;
  return iree_ok_status();
}

static iree_status_t ReadBindingToHost(iree_hal_device_t* device,
                                       iree_hal_queue_affinity_t queue_affinity,
                                       const iree_hal_buffer_binding_t* binding,
                                       iree_hal_semaphore_list_t wait_list,
                                       std::vector<uint8_t>* out_bytes) {
  out_bytes->assign(static_cast<size_t>(binding->length), 0);

  iree_io_file_handle_t* handle = nullptr;
  IREE_RETURN_IF_ERROR(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(out_bytes->data(), out_bytes->size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &handle));

  iree_hal_file_t* file = nullptr;
  iree_status_t status =
      iree_hal_file_import(device, queue_affinity, IREE_HAL_MEMORY_ACCESS_WRITE,
                           handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file);
  iree_io_file_handle_release(handle);

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      read_semaphore;
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(device, queue_affinity, 0,
                                       IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
                                       read_semaphore.out());
  }
  SemaphoreListStorage read_signal;
  read_signal.semaphore = read_semaphore.get();
  read_signal.payload_value = 1;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_write(
        device, queue_affinity, wait_list, read_signal.list(), binding->buffer,
        binding->offset, file, /*target_offset=*/0, binding->length,
        IREE_HAL_WRITE_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(
        read_semaphore.get(), read_signal.payload_value,
        iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
  }
  iree_hal_file_release(file);
  return status;
}

static std::vector<uint8_t> ToBytes(const std::vector<float>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(values[0]));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

static std::vector<float> ToF32Vector(const std::vector<uint8_t>& bytes) {
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

TEST(SamplerDenoiseStageIntegration, PrepareAndIssueDenoiseStep) {
  id4::test::LiveStageContext context;
  IREE_ASSERT_OK(id4::test::CreateLiveStageContextFromFlags(&context));

  id4::test::KernelLibraryRef kernel_library;
  IREE_ASSERT_OK(id4::test::CreateEmbeddedKernelLibrary(kernel_library.out()));

  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release> stage;
  IREE_ASSERT_OK(CreateSamplerStage(context, stage.out()));

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage.get(), &load_options));

  constexpr uint32_t kElementCount = 257;
  id4_sampler_denoise_stage_plan_options_t sampler_options;
  std::memset(&sampler_options, 0, sizeof(sampler_options));
  sampler_options.structure_size = sizeof(sampler_options);
  sampler_options.request.latent_shape =
      id4_pipeline_program_make_shape_rank1(kElementCount);

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &sampler_options;
  plan_options.flags = ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_ASSERT_OK(
      id4_pipeline_stage_plan(stage.get(), &plan_options, plan.out()));

  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.kernel_library = kernel_library.get();
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.diagnostics_sink = &diagnostics_sink;

  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      bundle;
  IREE_ASSERT_OK(id4_pipeline_stage_prepare(stage.get(), plan.get(),
                                            &prepare_options, bundle.out()));

  BufferBindingSet boundary_bindings;
  IREE_ASSERT_OK(AllocateBoundaryBindings(context.device.get(),
                                          IREE_HAL_QUEUE_AFFINITY_ANY,
                                          plan.get(), &boundary_bindings));
  BufferBindingSet diagnostic_tap_bindings;
  IREE_ASSERT_OK(AllocateDiagnosticTapBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      &diagnostic_tap_bindings));

  std::vector<float> cond_out(kElementCount);
  std::vector<float> uncond_out(kElementCount);
  std::vector<float> x_t(kElementCount, 10.0f);
  for (uint32_t i = 0; i < kElementCount; ++i) {
    cond_out[i] = static_cast<float>(i + 1);
    uncond_out[i] = static_cast<float>(i);
  }
  const std::vector<float> scalings = {1.0f, -1.0f, 0.0f};
  const std::vector<float> guidance = {7.0f, 0.0f, 0.0f};

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      update_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, update_semaphore.out()));
  uint64_t update_value = 0;

  iree_hal_buffer_binding_t cond_binding = {};
  IREE_ASSERT_OK(FindBoundaryBinding(
      plan.get(), boundary_bindings,
      id4_sampler_program_cond_out_boundary_name(), &cond_binding));
  std::vector<uint8_t> cond_bytes = ToBytes(cond_out);
  IREE_ASSERT_OK(QueueUpdateBinding(context.device.get(),
                                    IREE_HAL_QUEUE_AFFINITY_ANY, &cond_binding,
                                    cond_bytes.data(), cond_bytes.size(),
                                    update_semaphore.get(), &update_value));

  iree_hal_buffer_binding_t uncond_binding = {};
  IREE_ASSERT_OK(FindBoundaryBinding(
      plan.get(), boundary_bindings,
      id4_sampler_program_uncond_out_boundary_name(), &uncond_binding));
  std::vector<uint8_t> uncond_bytes = ToBytes(uncond_out);
  IREE_ASSERT_OK(QueueUpdateBinding(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &uncond_binding,
      uncond_bytes.data(), uncond_bytes.size(), update_semaphore.get(),
      &update_value));

  iree_hal_buffer_binding_t x_t_binding = {};
  IREE_ASSERT_OK(FindBoundaryBinding(plan.get(), boundary_bindings,
                                     id4_sampler_program_x_t_boundary_name(),
                                     &x_t_binding));
  std::vector<uint8_t> x_t_bytes = ToBytes(x_t);
  IREE_ASSERT_OK(QueueUpdateBinding(context.device.get(),
                                    IREE_HAL_QUEUE_AFFINITY_ANY, &x_t_binding,
                                    x_t_bytes.data(), x_t_bytes.size(),
                                    update_semaphore.get(), &update_value));

  iree_hal_buffer_binding_t scalings_binding = {};
  IREE_ASSERT_OK(FindBoundaryBinding(
      plan.get(), boundary_bindings,
      id4_sampler_program_scalings_boundary_name(), &scalings_binding));
  std::vector<uint8_t> scalings_bytes = ToBytes(scalings);
  IREE_ASSERT_OK(QueueUpdateBinding(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &scalings_binding,
      scalings_bytes.data(), scalings_bytes.size(), update_semaphore.get(),
      &update_value));

  iree_hal_buffer_binding_t guidance_binding = {};
  IREE_ASSERT_OK(FindBoundaryBinding(
      plan.get(), boundary_bindings,
      id4_sampler_program_guidance_boundary_name(), &guidance_binding));
  std::vector<uint8_t> guidance_bytes = ToBytes(guidance);
  IREE_ASSERT_OK(QueueUpdateBinding(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &guidance_binding,
      guidance_bytes.data(), guidance_bytes.size(), update_semaphore.get(),
      &update_value));

  const uint8_t sentinel = 0xA5;
  iree_hal_buffer_binding_t denoised_binding = {};
  IREE_ASSERT_OK(FindBoundaryBinding(
      plan.get(), boundary_bindings,
      id4_sampler_program_denoised_boundary_name(), &denoised_binding));
  IREE_ASSERT_OK(QueueFillBinding(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &denoised_binding,
      &sentinel, sizeof(sentinel), update_semaphore.get(), &update_value));

  iree_hal_buffer_binding_t guided_pred_binding = {};
  IREE_ASSERT_OK(FindDiagnosticTapBinding(
      plan.get(), diagnostic_tap_bindings,
      id4_sampler_program_guided_pred_tap_name(), &guided_pred_binding));
  IREE_ASSERT_OK(QueueFillBinding(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &guided_pred_binding,
      &sentinel, sizeof(sentinel), update_semaphore.get(), &update_value));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      issue_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, issue_semaphore.out()));

  SemaphoreListStorage issue_wait;
  issue_wait.semaphore = update_semaphore.get();
  issue_wait.payload_value = update_value;
  SemaphoreListStorage issue_signal;
  issue_signal.semaphore = issue_semaphore.get();
  issue_signal.payload_value = 1;

  id4_pipeline_stage_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.boundary_binding_count = boundary_bindings.count;
  issue_options.boundary_bindings = boundary_bindings.bindings;
  issue_options.diagnostic_tap_binding_count = diagnostic_tap_bindings.count;
  issue_options.diagnostic_tap_bindings = diagnostic_tap_bindings.bindings;
  issue_options.wait_semaphore_list = issue_wait.list();
  issue_options.signal_semaphore_list = issue_signal.list();
  issue_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(
      id4_pipeline_stage_issue(stage.get(), bundle.get(), &issue_options));

  SemaphoreListStorage read_wait;
  read_wait.semaphore = issue_semaphore.get();
  read_wait.payload_value = issue_signal.payload_value;
  std::vector<uint8_t> denoised_bytes;
  IREE_ASSERT_OK(
      ReadBindingToHost(context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
                        &denoised_binding, read_wait.list(), &denoised_bytes));
  std::vector<uint8_t> guided_pred_bytes;
  IREE_ASSERT_OK(ReadBindingToHost(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &guided_pred_binding,
      read_wait.list(), &guided_pred_bytes));

  std::vector<float> denoised = ToF32Vector(denoised_bytes);
  std::vector<float> guided_pred = ToF32Vector(guided_pred_bytes);
  for (uint32_t i = 0; i < kElementCount; ++i) {
    const float expected_guided_pred = static_cast<float>(i + 7);
    const float expected_denoised = static_cast<float>(3 - (int32_t)i);
    EXPECT_FLOAT_EQ(guided_pred[i], expected_guided_pred);
    EXPECT_FLOAT_EQ(denoised[i], expected_denoised);
  }
}

}  // namespace
