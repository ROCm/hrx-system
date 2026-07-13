// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "experimental/id4/ideogram4/request.h"
#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/ideogram4_decode.h"
#include "experimental/id4/stages/ideogram4_dit.h"
#include "experimental/id4/stages/ideogram4_dit_parameters.h"
#include "experimental/id4/stages/sampler.h"
#include "experimental/id4/tooling/capture.h"
#include "experimental/id4/tooling/image.h"
#include "experimental/id4/tooling/runtime.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

IREE_FLAG(string, id4_fixture_dir, "",
          "Directory containing a full one-step Ideogram4 fixture manifest.");
IREE_FLAG(string, id4_output_image, "",
          "Optional binary PPM path receiving the decoded image tensor.");
IREE_FLAG(string, id4_capture_dir, "",
          "Optional directory receiving captured stage boundary and diagnostic "
          "tap tensors.");
IREE_FLAG(string, id4_plan_output_dir, "",
          "Optional directory receiving planned stage JSON files.");
IREE_FLAG(int32_t, id4_step_count, 1,
          "Number of leading denoise steps from the selected sampler to run.");
IREE_FLAG(int32_t, id4_start_step, 1,
          "One-based chronological denoise step to execute first.");
IREE_FLAG(string, id4_sampler, "V4_DEFAULT_20",
          "Advertised sampler preset supplying the denoise schedule.");
IREE_FLAG(string, dit_parameter_format, "fp8_e4m3",
          "DiT parameter format: bf16 or fp8_e4m3.");
IREE_FLAG(string, dit_conditioned_fp8_scope, "dit_cond_fp8",
          "Conditioned DiT FP8 e4m3 source parameter scope.");
IREE_FLAG(string, dit_unconditioned_fp8_scope, "dit_uncond_fp8",
          "Unconditioned DiT FP8 e4m3 source parameter scope.");

namespace {

constexpr uint8_t kOutputSentinel = 0xA5;

using ParameterProviderRef =
    id4::test::OwningRef<iree_io_parameter_provider_t,
                         iree_io_parameter_provider_release>;

class RetainedHalFiles {
 public:
  RetainedHalFiles() = default;
  RetainedHalFiles(const RetainedHalFiles&) = delete;
  RetainedHalFiles& operator=(const RetainedHalFiles&) = delete;

  ~RetainedHalFiles() {
    for (iree_hal_file_t* file : files_) {
      iree_hal_file_release(file);
    }
  }

  void push(iree_hal_file_t* file) { files_.push_back(file); }

 private:
  // Imported HAL files retained while queued reads may be in flight.
  std::vector<iree_hal_file_t*> files_;
};

struct ScopedDenoiseSchedule {
  ~ScopedDenoiseSchedule() {
    id4_ideogram4_denoise_schedule_deinitialize(&value,
                                                iree_allocator_system());
  }

  id4_ideogram4_denoise_schedule_t value = {};
};

static iree_status_t FindFixtureTensor(
    const id4::test::FixtureTensorSet& fixture_tensors, iree_string_view_t role,
    iree_string_view_t stage, iree_string_view_t name,
    const id4::test::FixtureTensor** out_tensor) {
  *out_tensor = fixture_tensors.FindTensor(role, stage, name);
  if (*out_tensor) return iree_ok_status();
  return iree_make_status(
      IREE_STATUS_NOT_FOUND, "fixture tensor `%.*s/%.*s/%.*s` not found",
      static_cast<int>(role.size), role.data, static_cast<int>(stage.size),
      stage.data, static_cast<int>(name.size), name.data);
}

static iree_status_t MakeProgramShape(
    id4_pipeline_tensor_shape_t tensor_shape,
    id4_pipeline_program_shape_t* out_program_shape) {
  if (tensor_shape.rank > IREE_ARRAYSIZE(out_program_shape->dims)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "fixture tensor rank %u exceeds program max rank",
                            tensor_shape.rank);
  }
  std::memset(out_program_shape, 0, sizeof(*out_program_shape));
  out_program_shape->rank = tensor_shape.rank;
  for (uint32_t i = 0; i < tensor_shape.rank; ++i) {
    out_program_shape->dims[i] = tensor_shape.dims[i];
  }
  return iree_ok_status();
}

static std::string JoinPath(iree_string_view_t directory,
                            iree_string_view_t file_name) {
  std::string path(directory.data, directory.size);
  if (!path.empty() && path.back() != '/') path.push_back('/');
  path.append(file_name.data, file_name.size);
  return path;
}

static std::string StepCaptureName(iree_string_view_t prefix,
                                   uint32_t step_ordinal) {
  std::string name(prefix.data, prefix.size);
  name.append("_step_");
  name.append(std::to_string(step_ordinal + 1));
  return name;
}

static iree_status_t WritePlanJsonIfRequested(const id4_pipeline_plan_t* plan,
                                              iree_string_view_t file_name) {
  iree_string_view_t output_dir =
      iree_make_cstring_view(FLAG_id4_plan_output_dir);
  if (iree_string_view_is_empty(output_dir)) return iree_ok_status();

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status = id4_pipeline_plan_format_json(plan, &builder);
  if (iree_status_is_ok(status)) {
    std::string path = JoinPath(output_dir, file_name);
    iree_string_view_t json = iree_string_builder_view(&builder);
    status = iree_io_file_contents_write(
        iree_make_string_view(path.data(), path.size()),
        iree_make_const_byte_span(json.data, json.size),
        iree_allocator_system());
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t ParseDitParameterFormat(
    id4_ideogram4_dit_parameter_format_t* out_format) {
  iree_status_t status = id4_ideogram4_dit_parameter_format_parse(
      iree_make_cstring_view(FLAG_dit_parameter_format), out_format);
  if (iree_status_is_ok(status)) return status;
  return iree_status_annotate(status, IREE_SV("--dit_parameter_format"));
}

static iree_string_view_t Fp8ParameterScopeForRequest(
    id4_ideogram4_dit_conditioning_mode_t conditioning_mode) {
  switch (conditioning_mode) {
    case ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED:
      return iree_make_cstring_view(FLAG_dit_conditioned_fp8_scope);
    case ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED:
      return iree_make_cstring_view(FLAG_dit_unconditioned_fp8_scope);
    default:
      return iree_string_view_empty();
  }
}

static bool TensorShapeEquals(id4_pipeline_tensor_shape_t lhs,
                              id4_pipeline_tensor_shape_t rhs) {
  if (lhs.rank != rhs.rank) return false;
  for (uint32_t i = 0; i < lhs.rank; ++i) {
    if (lhs.dims[i] != rhs.dims[i]) return false;
  }
  return true;
}

static iree_status_t FindBoundaryBindingIndex(const id4_pipeline_plan_t* plan,
                                              iree_string_view_t name,
                                              iree_host_size_t* out_index) {
  for (iree_host_size_t i = 0;
       i < id4_pipeline_plan_boundary_tensor_count(plan); ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(plan, i);
    if (boundary && iree_string_view_equal(boundary->layout.name, name)) {
      *out_index = i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "boundary tensor `%.*s` not found",
                          static_cast<int>(name.size), name.data);
}

static iree_status_t FindFixtureInputTensor(
    const id4::test::FixtureTensorSet& fixture_tensors,
    iree_string_view_t stage, iree_string_view_t name,
    const id4::test::FixtureTensor** out_tensor) {
  return FindFixtureTensor(fixture_tensors, IREE_SV("input"), stage, name,
                           out_tensor);
}

static iree_status_t QueueReadBindingFromHostAllocation(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_buffer_binding_t* binding, const void* source_data,
    iree_host_size_t source_length, iree_hal_semaphore_t* semaphore,
    uint64_t* inout_payload_value, RetainedHalFiles* retained_files) {
  if (source_length != binding->length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source length %" PRIhsz " does not match binding length %" PRIu64,
        source_length, static_cast<uint64_t>(binding->length));
  }

  const uint8_t* source_bytes = static_cast<const uint8_t*>(source_data);
  iree_io_file_handle_t* handle = nullptr;
  IREE_RETURN_IF_ERROR(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ,
      iree_make_byte_span(const_cast<uint8_t*>(source_bytes), source_length),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &handle));

  iree_hal_file_t* file = nullptr;
  iree_status_t status =
      iree_hal_file_import(device, queue_affinity, IREE_HAL_MEMORY_ACCESS_READ,
                           handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file);
  iree_io_file_handle_release(handle);

  iree_hal_semaphore_list_t wait_list = iree_hal_semaphore_list_empty();
  id4::test::SemaphoreListStorage wait_storage;
  wait_storage.semaphore = semaphore;
  wait_storage.payload_value = *inout_payload_value;
  if (wait_storage.payload_value != 0) {
    wait_list = wait_storage.list();
  }
  id4::test::SemaphoreListStorage signal_storage;
  signal_storage.semaphore = semaphore;
  signal_storage.payload_value = wait_storage.payload_value + 1;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_read(
        device, queue_affinity, wait_list, signal_storage.list(), file,
        /*source_offset=*/0, binding->buffer, binding->offset, binding->length,
        IREE_HAL_READ_FLAG_NONE);
  }
  if (iree_status_is_ok(status)) {
    retained_files->push(file);
    *inout_payload_value = signal_storage.payload_value;
  } else {
    iree_hal_file_release(file);
  }
  return status;
}

static iree_status_t QueueUpdateBoundary(
    iree_hal_device_t* device, const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& binding_set,
    iree_string_view_t boundary_name, const void* source_data,
    iree_host_size_t source_length, iree_hal_semaphore_t* update_semaphore,
    uint64_t* inout_update_value) {
  iree_host_size_t boundary_index = 0;
  IREE_RETURN_IF_ERROR(
      FindBoundaryBindingIndex(plan, boundary_name, &boundary_index));
  return id4::test::QueueUpdateBinding(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                       &binding_set.bindings[boundary_index],
                                       source_data, source_length,
                                       update_semaphore, inout_update_value);
}

static iree_status_t UpdateBoundaryFromFixtureTensor(
    iree_hal_device_t* device, const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& binding_set,
    iree_string_view_t boundary_name,
    const id4::test::FixtureTensor& fixture_tensor,
    iree_hal_semaphore_t* update_semaphore, uint64_t* inout_update_value,
    RetainedHalFiles* retained_files) {
  iree_host_size_t boundary_index = 0;
  IREE_RETURN_IF_ERROR(
      FindBoundaryBindingIndex(plan, boundary_name, &boundary_index));
  const id4_pipeline_boundary_tensor_plan_t* boundary =
      id4_pipeline_plan_boundary_tensor_at(plan, boundary_index);
  if (fixture_tensor.dtype != boundary->layout.dtype) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fixture tensor `%s/%s` dtype does not match boundary `%.*s` dtype",
        fixture_tensor.stage.c_str(), fixture_tensor.name.c_str(),
        static_cast<int>(boundary_name.size), boundary_name.data);
  }
  if (!TensorShapeEquals(fixture_tensor.shape, boundary->layout.shape)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fixture tensor `%s/%s` shape does not match boundary `%.*s` shape",
        fixture_tensor.stage.c_str(), fixture_tensor.name.c_str(),
        static_cast<int>(boundary_name.size), boundary_name.data);
  }
  return QueueReadBindingFromHostAllocation(
      device, IREE_HAL_QUEUE_AFFINITY_ANY,
      &binding_set.bindings[boundary_index], fixture_tensor.payload.data(),
      fixture_tensor.payload.size(), update_semaphore, inout_update_value,
      retained_files);
}

static iree_status_t UpdateBoundaryFromFixture(
    iree_hal_device_t* device, const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& binding_set,
    iree_string_view_t boundary_name,
    const id4::test::FixtureTensorSet& fixture_tensors,
    iree_string_view_t fixture_stage, iree_string_view_t fixture_name,
    iree_hal_semaphore_t* update_semaphore, uint64_t* inout_update_value,
    RetainedHalFiles* retained_files) {
  const id4::test::FixtureTensor* fixture_tensor = nullptr;
  IREE_RETURN_IF_ERROR(FindFixtureTensor(fixture_tensors, IREE_SV("input"),
                                         fixture_stage, fixture_name,
                                         &fixture_tensor));
  return UpdateBoundaryFromFixtureTensor(
      device, plan, binding_set, boundary_name, *fixture_tensor,
      update_semaphore, inout_update_value, retained_files);
}

static iree_status_t ReplaceBoundaryBinding(
    const id4_pipeline_plan_t* plan, id4::test::BufferBindingSet* binding_set,
    iree_string_view_t boundary_name,
    iree_hal_buffer_binding_t replacement_binding) {
  iree_host_size_t boundary_index = 0;
  IREE_RETURN_IF_ERROR(
      FindBoundaryBindingIndex(plan, boundary_name, &boundary_index));
  binding_set->bindings[boundary_index] = replacement_binding;
  return iree_ok_status();
}

static iree_status_t ConfigureDitRequestFromFixture(
    const id4::test::FixtureTensorSet& fixture_tensors,
    iree_string_view_t fixture_stage,
    id4_ideogram4_dit_conditioning_mode_t conditioning_mode,
    id4_ideogram4_dit_request_config_t* out_request) {
  std::memset(out_request, 0, sizeof(*out_request));

  const id4::test::FixtureTensor* latent = nullptr;
  IREE_RETURN_IF_ERROR(FindFixtureTensor(fixture_tensors, IREE_SV("input"),
                                         fixture_stage, IREE_SV("x"), &latent));
  IREE_RETURN_IF_ERROR(
      MakeProgramShape(latent->shape, &out_request->latent_shape));
  out_request->conditioning_mode = conditioning_mode;

  if (conditioning_mode == ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED) {
    const id4::test::FixtureTensor* condition = nullptr;
    IREE_RETURN_IF_ERROR(FindFixtureTensor(fixture_tensors, IREE_SV("input"),
                                           fixture_stage, IREE_SV("context"),
                                           &condition));
    if (condition->shape.rank != 2 ||
        condition->shape.dims[1] >
            static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "condition fixture tensor must be rank-2 with uint32 token count");
    }
    out_request->text_token_count =
        static_cast<uint32_t>(condition->shape.dims[1]);
  }
  return iree_ok_status();
}

static iree_status_t CreateDitStage(const id4::test::LiveStageContext& context,
                                    iree_string_view_t parameter_scope,
                                    iree_string_view_t fp8_parameter_scope,
                                    id4_ideogram4_dit_parameter_format_t format,
                                    id4_pipeline_stage_t** out_stage) {
  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = context.device_group.get();
  services.executable_cache = context.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_ideogram4_dit_parameter_source_rule_list_t source_rules;
  IREE_RETURN_IF_ERROR(id4_ideogram4_dit_parameter_source_rule_list_initialize(
      format, *id4_ideogram4_dit_program_ideogram4_model_config(),
      fp8_parameter_scope, iree_allocator_system(), &source_rules));

  id4_ideogram4_dit_stage_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.services = services;
  options.kernel_cache = context.kernel_cache.get();
  options.parameter_scope = parameter_scope;
  options.model = *id4_ideogram4_dit_program_ideogram4_model_config();
  options.parameter_source_rule_count = source_rules.count;
  options.parameter_source_rules = source_rules.values;
  iree_status_t status = id4_ideogram4_dit_stage_create(
      &options, iree_allocator_system(), out_stage);
  id4_ideogram4_dit_parameter_source_rule_list_deinitialize(
      &source_rules, iree_allocator_system());
  return status;
}

static iree_status_t CreateSamplerStage(
    const id4::test::LiveStageContext& context,
    id4_pipeline_stage_t** out_stage) {
  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = context.device_group.get();
  services.executable_cache = context.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_sampler_denoise_stage_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.services = services;
  options.kernel_cache = context.kernel_cache.get();
  return id4_sampler_denoise_stage_create(&options, iree_allocator_system(),
                                          out_stage);
}

static iree_status_t CreateDecodeStage(
    const id4::test::LiveStageContext& context,
    iree_string_view_t parameter_scope, id4_pipeline_stage_t** out_stage) {
  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = context.device_group.get();
  services.executable_cache = context.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_ideogram4_decode_stage_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.services = services;
  options.kernel_cache = context.kernel_cache.get();
  options.parameter_scope = parameter_scope;
  options.model = *id4_ideogram4_decode_program_ideogram4_model_config();
  options.vae_activation_format = ID4_VAE_ACTIVATION_FORMAT_BF16_CONV_INPUT;
  return id4_ideogram4_decode_stage_create(&options, iree_allocator_system(),
                                           out_stage);
}

static iree_status_t LoadStage(id4_pipeline_stage_t* stage,
                               id4_pipeline_diagnostics_sink_t* sink) {
  id4_pipeline_stage_load_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.diagnostics_sink = sink;
  return id4_pipeline_stage_load(stage, &options);
}

static iree_status_t CreateDitParameterProviderFromFlags(
    iree_string_view_t parameter_scope, iree_string_view_t fp8_parameter_scope,
    id4_ideogram4_dit_parameter_format_t format,
    iree_io_parameter_provider_t** out_provider) {
  if (format == ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16) {
    return id4::test::CreateParameterProviderFromFlags(parameter_scope,
                                                       out_provider);
  }

  ParameterProviderRef bf16_provider;
  ParameterProviderRef fp8_provider;
  id4_tooling_parameter_provider_request_t requests[] = {
      {
          // BF16 parameter scope.
          .scope = parameter_scope,
          // BF16 provider output.
          .out_provider = bf16_provider.out(),
      },
      {
          // FP8 e4m3 source parameter scope.
          .scope = fp8_parameter_scope,
          // FP8 e4m3 source provider output.
          .out_provider = fp8_provider.out(),
      },
  };
  IREE_RETURN_IF_ERROR(id4_tooling_create_parameter_providers_from_flags(
      IREE_ARRAYSIZE(requests), requests, iree_allocator_system()));

  const id4_tooling_parameter_provider_set_entry_t entries[] = {
      {
          // BF16 parameter scope.
          .scope = parameter_scope,
          // BF16 provider.
          .provider = bf16_provider.get(),
      },
      {
          // FP8 e4m3 source parameter scope.
          .scope = fp8_parameter_scope,
          // FP8 e4m3 source provider.
          .provider = fp8_provider.get(),
      },
  };
  return id4_tooling_create_parameter_provider_set(
      IREE_ARRAYSIZE(entries), entries, iree_allocator_system(), out_provider);
}

static iree_status_t PlanDitStage(
    id4_pipeline_stage_t* stage, id4_ideogram4_dit_request_config_t request,
    id4_ideogram4_dit_weight_execution_format_t weight_execution_format,
    iree_string_view_list_t diagnostic_tap_names,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_pipeline_plan_t** out_plan) {
  id4_ideogram4_dit_stage_plan_options_t dit_options;
  std::memset(&dit_options, 0, sizeof(dit_options));
  dit_options.structure_size = sizeof(dit_options);
  dit_options.request = request;
  dit_options.activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  dit_options.weight_execution_format = weight_execution_format;
  dit_options.attention_implementation =
      ID4_IDEOGRAM4_DIT_ATTENTION_IMPLEMENTATION_ONLINE_WMMA;
  dit_options.feed_forward_implementation =
      ID4_IDEOGRAM4_DIT_FEED_FORWARD_IMPLEMENTATION_PYTORCH_PARITY;

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &dit_options;
  if (diagnostic_tap_names.count != 0) {
    plan_options.flags = ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
    plan_options.diagnostic_tap_names = diagnostic_tap_names;
  }
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = diagnostics_sink;
  return id4_pipeline_stage_plan(stage, &plan_options, out_plan);
}

static iree_status_t PlanSamplerStage(
    id4_pipeline_stage_t* stage, id4_pipeline_program_shape_t latent_shape,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_pipeline_plan_t** out_plan) {
  id4_sampler_denoise_stage_plan_options_t sampler_options;
  std::memset(&sampler_options, 0, sizeof(sampler_options));
  sampler_options.structure_size = sizeof(sampler_options);
  sampler_options.request.latent_shape = latent_shape;

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &sampler_options;
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = diagnostics_sink;
  return id4_pipeline_stage_plan(stage, &plan_options, out_plan);
}

static iree_status_t PlanDecodeStage(
    id4_pipeline_stage_t* stage,
    id4_pipeline_program_shape_t diffusion_latent_shape,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_pipeline_plan_t** out_plan) {
  id4_ideogram4_decode_stage_plan_options_t decode_options;
  std::memset(&decode_options, 0, sizeof(decode_options));
  decode_options.structure_size = sizeof(decode_options);
  decode_options.request.diffusion_latent_shape = diffusion_latent_shape;
  decode_options.request.vae_tiling.mode = ID4_VAE_TILING_MODE_DISABLED;
  decode_options.request.vae_attention_implementation =
      ID4_VAE_ATTENTION_IMPLEMENTATION_MATERIALIZED;

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &decode_options;
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = diagnostics_sink;
  return id4_pipeline_stage_plan(stage, &plan_options, out_plan);
}

static iree_status_t PrepareStage(
    id4_pipeline_stage_t* stage, const id4_pipeline_plan_t* plan,
    id4_pipeline_stage_parameter_policy_t parameter_policy,
    id4_pipeline_kernel_library_t* kernel_library,
    iree_hal_semaphore_list_t wait_list, iree_hal_semaphore_list_t signal_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink,
    id4_pipeline_bundle_t** out_bundle) {
  id4_pipeline_stage_prepare_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.parameter_policy = parameter_policy;
  options.kernel_library = kernel_library;
  options.wait_semaphore_list = wait_list;
  options.signal_semaphore_list = signal_list;
  options.diagnostics_sink = diagnostics_sink;
  return id4_pipeline_stage_prepare(stage, plan, &options, out_bundle);
}

static iree_status_t IssueStage(
    id4_pipeline_stage_t* stage, id4_pipeline_bundle_t* bundle,
    const id4::test::BufferBindingSet& boundary_bindings,
    const id4::test::BufferBindingSet& diagnostic_tap_bindings,
    iree_hal_semaphore_list_t wait_list, iree_hal_semaphore_list_t signal_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  id4_pipeline_stage_issue_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.execution_segment_submission_window = 1;
  options.boundary_binding_count = boundary_bindings.count;
  options.boundary_bindings = boundary_bindings.bindings;
  options.diagnostic_tap_binding_count = diagnostic_tap_bindings.count;
  options.diagnostic_tap_bindings = diagnostic_tap_bindings.bindings;
  options.wait_semaphore_list = wait_list;
  options.signal_semaphore_list = signal_list;
  options.diagnostics_sink = diagnostics_sink;
  return id4_pipeline_stage_issue(stage, bundle, &options);
}

static iree_status_t CaptureStageIfRequested(
    iree_string_view_t capture_directory, iree_string_view_t stage_directory,
    iree_string_view_t run_id, iree_hal_device_t* device,
    const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& boundary_bindings,
    const id4::test::BufferBindingSet& diagnostic_tap_bindings,
    iree_hal_semaphore_list_t wait_list) {
  if (iree_string_view_is_empty(capture_directory)) return iree_ok_status();

  std::string output_directory = JoinPath(capture_directory, stage_directory);
  id4_tooling_capture_execution_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.run_id = run_id;
  options.output_directory =
      iree_make_string_view(output_directory.data(), output_directory.size());
  options.plan = plan;
  options.device = device;
  options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  options.boundary_binding_count = boundary_bindings.count;
  options.boundary_bindings = boundary_bindings.bindings;
  options.diagnostic_tap_binding_count = diagnostic_tap_bindings.count;
  options.diagnostic_tap_bindings = diagnostic_tap_bindings.bindings;
  options.wait_semaphore_list = wait_list;
  options.host_allocator = iree_allocator_system();
  return id4_tooling_capture_execution(&options);
}

static std::vector<float> ToF32Vector(const std::vector<uint8_t>& bytes) {
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

static iree_status_t VerifyDecodedImageContents(
    const std::vector<uint8_t>& bytes) {
  if (bytes.empty() || bytes.size() % sizeof(float) != 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "decoded image payload has invalid byte length");
  }
  const std::vector<float> values = ToF32Vector(bytes);
  bool saw_non_sentinel = false;
  bool saw_finite = false;
  bool saw_nonzero = false;
  for (float value : values) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((uint8_t)(bits & 0xFFu) != kOutputSentinel ||
        (uint8_t)((bits >> 8) & 0xFFu) != kOutputSentinel ||
        (uint8_t)((bits >> 16) & 0xFFu) != kOutputSentinel ||
        (uint8_t)((bits >> 24) & 0xFFu) != kOutputSentinel) {
      saw_non_sentinel = true;
    }
    if (std::isfinite(value)) saw_finite = true;
    if (value != 0.0f) saw_nonzero = true;
  }
  if (!saw_non_sentinel) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "decoded image remained filled with sentinel");
  }
  if (!saw_finite) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "decoded image contains no finite values");
  }
  if (!saw_nonzero) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "decoded image contains only zero values");
  }
  return iree_ok_status();
}

static iree_status_t MaybeWriteDecodedImage(
    iree_string_view_t output_image_path,
    const id4_pipeline_tensor_shape_t& image_shape,
    const std::vector<uint8_t>& decoded_bytes) {
  if (iree_string_view_is_empty(output_image_path)) return iree_ok_status();

  id4_tooling_write_f32_rgb_ppm_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.path = output_image_path;
  options.shape = image_shape;
  options.pixels =
      iree_make_const_byte_span(decoded_bytes.data(), decoded_bytes.size());
  options.normalization = ID4_TOOLING_IMAGE_NORMALIZATION_MINUS_ONE_TO_ONE;
  options.host_allocator = iree_allocator_system();
  return id4_tooling_write_f32_rgb_ppm(&options);
}

static iree_status_t LoadFixtureBytes(
    const id4::test::FixtureTensorSet& fixture_tensors,
    iree_string_view_t stage, iree_string_view_t name,
    std::vector<uint8_t>* out_bytes) {
  const id4::test::FixtureTensor* tensor = nullptr;
  IREE_RETURN_IF_ERROR(
      FindFixtureInputTensor(fixture_tensors, stage, name, &tensor));
  *out_bytes = tensor->payload;
  return iree_ok_status();
}

static iree_status_t CompareBoundaryWithExpectedFixture(
    iree_hal_device_t* device, const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& boundary_bindings,
    iree_string_view_t boundary_name,
    const id4::test::FixtureTensorSet& fixture_tensors,
    iree_string_view_t fixture_stage, iree_string_view_t fixture_name,
    iree_hal_semaphore_list_t wait_list) {
  const id4::test::FixtureTensor* expected_tensor = nullptr;
  IREE_RETURN_IF_ERROR(FindFixtureTensor(fixture_tensors, IREE_SV("expected"),
                                         fixture_stage, fixture_name,
                                         &expected_tensor));
  iree_hal_buffer_binding_t binding = {};
  IREE_RETURN_IF_ERROR(id4::test::FindBoundaryBinding(plan, boundary_bindings,
                                                      boundary_name, &binding));
  return id4::test::CompareF32BindingWithFixtureTensor(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, &binding, wait_list,
      *expected_tensor);
}

TEST(Ideogram4OneStepTraceIntegration,
     RunsReferenceDiTSamplerAndDecodeOnDevice) {
  const iree_string_view_t fixture_directory =
      iree_make_cstring_view(FLAG_id4_fixture_dir);
  ASSERT_FALSE(iree_string_view_is_empty(fixture_directory))
      << "--id4_fixture_dir is required";
  const iree_string_view_t capture_directory =
      iree_make_cstring_view(FLAG_id4_capture_dir);
  const int32_t step_count = FLAG_id4_step_count;
  ASSERT_GT(step_count, 0) << "--id4_step_count must be positive";
  const uint32_t denoise_step_count = static_cast<uint32_t>(step_count);
  const int32_t start_step = FLAG_id4_start_step;
  ASSERT_GT(start_step, 0) << "--id4_start_step must be positive";
  const uint32_t start_step_ordinal = static_cast<uint32_t>(start_step - 1);
  id4_ideogram4_sampler_preset_t sampler_preset = {};
  IREE_ASSERT_OK(id4_ideogram4_sampler_preset_parse(
      iree_make_cstring_view(FLAG_id4_sampler), &sampler_preset));
  const uint32_t sampler_step_count =
      id4_ideogram4_sampler_preset_step_count(sampler_preset);
  ASSERT_LE(static_cast<uint64_t>(start_step_ordinal) + denoise_step_count,
            sampler_step_count);
  id4_ideogram4_dit_parameter_format_t dit_parameter_format =
      ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_INVALID;
  IREE_ASSERT_OK(ParseDitParameterFormat(&dit_parameter_format));
  id4_ideogram4_dit_weight_execution_format_t dit_weight_execution_format =
      ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_INVALID;
  switch (dit_parameter_format) {
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_BF16:
      dit_weight_execution_format =
          ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_BF16_RESIDENT;
      break;
    case ID4_IDEOGRAM4_DIT_PARAMETER_FORMAT_FP8_E4M3:
      dit_weight_execution_format =
          ID4_IDEOGRAM4_DIT_WEIGHT_EXECUTION_FORMAT_FP8_COMPACT_RHS;
      break;
    default:
      FAIL() << "unsupported DiT parameter format "
             << static_cast<uint32_t>(dit_parameter_format);
  }
  const bool capture_one_step_trace = step_count == 1;
  const bool capture_requested = !iree_string_view_is_empty(capture_directory);

  id4::test::FixtureTensorSet fixture_tensors;
  IREE_ASSERT_OK(
      id4::test::LoadFixtureTensors(fixture_directory, &fixture_tensors));

  const iree_string_view_t cond_input_stage = IREE_SV("ideogram4.cond.input");
  const iree_string_view_t uncond_input_stage =
      IREE_SV("ideogram4.uncond.input");
  id4_ideogram4_dit_request_config_t cond_request;
  IREE_ASSERT_OK(ConfigureDitRequestFromFixture(
      fixture_tensors, cond_input_stage,
      ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_CONDITIONED, &cond_request));
  id4_ideogram4_dit_request_config_t uncond_request;
  IREE_ASSERT_OK(ConfigureDitRequestFromFixture(
      fixture_tensors, uncond_input_stage,
      ID4_IDEOGRAM4_DIT_CONDITIONING_MODE_UNCONDITIONED, &uncond_request));
  ASSERT_EQ(cond_request.latent_shape.rank, uncond_request.latent_shape.rank);
  for (uint32_t i = 0; i < cond_request.latent_shape.rank; ++i) {
    ASSERT_EQ(cond_request.latent_shape.dims[i],
              uncond_request.latent_shape.dims[i]);
  }

  id4::test::LiveStageContext context;
  IREE_ASSERT_OK(id4::test::CreateLiveStageContextFromFlags(&context));
  id4::test::KernelLibraryRef kernel_library;
  IREE_ASSERT_OK(id4::test::CreateEmbeddedKernelLibrary(kernel_library.out()));

  const iree_string_view_t cond_fp8_parameter_scope =
      Fp8ParameterScopeForRequest(cond_request.conditioning_mode);
  const iree_string_view_t uncond_fp8_parameter_scope =
      Fp8ParameterScopeForRequest(uncond_request.conditioning_mode);

  ParameterProviderRef cond_provider;
  IREE_ASSERT_OK(CreateDitParameterProviderFromFlags(
      IREE_SV("dit_cond"), cond_fp8_parameter_scope, dit_parameter_format,
      cond_provider.out()));
  ParameterProviderRef uncond_provider;
  IREE_ASSERT_OK(CreateDitParameterProviderFromFlags(
      IREE_SV("dit_uncond"), uncond_fp8_parameter_scope, dit_parameter_format,
      uncond_provider.out()));
  ParameterProviderRef vae_provider;
  IREE_ASSERT_OK(id4::test::CreateParameterProviderFromFlags(
      IREE_SV("vae"), vae_provider.out()));

  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release>
      cond_stage;
  IREE_ASSERT_OK(CreateDitStage(context, IREE_SV("dit_cond"),
                                cond_fp8_parameter_scope, dit_parameter_format,
                                cond_stage.out()));
  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release>
      uncond_stage;
  IREE_ASSERT_OK(CreateDitStage(context, IREE_SV("dit_uncond"),
                                uncond_fp8_parameter_scope,
                                dit_parameter_format, uncond_stage.out()));
  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release>
      sampler_stage;
  IREE_ASSERT_OK(CreateSamplerStage(context, sampler_stage.out()));
  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release>
      decode_stage;
  IREE_ASSERT_OK(
      CreateDecodeStage(context, IREE_SV("vae"), decode_stage.out()));

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);
  IREE_ASSERT_OK(LoadStage(cond_stage.get(), &diagnostics_sink));
  IREE_ASSERT_OK(LoadStage(uncond_stage.get(), &diagnostics_sink));
  IREE_ASSERT_OK(LoadStage(sampler_stage.get(), &diagnostics_sink));
  IREE_ASSERT_OK(LoadStage(decode_stage.get(), &diagnostics_sink));

  const iree_string_view_t cond_dit_capture_tap_names[] = {
      IREE_SV("ideogram4.cond.prelude.llm_cond_norm"),
      IREE_SV("ideogram4.cond.prelude.llm_cond_proj"),
      IREE_SV("ideogram4.cond.prelude.hidden"),
      IREE_SV("ideogram4.cond.prelude.timestep_embedding"),
      IREE_SV("ideogram4.cond.prelude.t_embedding.mlp_in"),
      IREE_SV("ideogram4.cond.prelude.t_embedding.mlp_out"),
      IREE_SV("ideogram4.cond.prelude.adaln_input"),
      IREE_SV("ideogram4.cond.layers.0.adaln_modulation"),
      IREE_SV("ideogram4.cond.layers.0.attention_input"),
      IREE_SV("ideogram4.cond.layers.0.attention.qkv_projection.output"),
      IREE_SV("ideogram4.cond.layers.0.attention.qkv"),
      IREE_SV("ideogram4.cond.layers.0.attention.query_rotary"),
      IREE_SV("ideogram4.cond.layers.0.attention.key_rotary"),
      IREE_SV("ideogram4.cond.layers.0.attention.context"),
      IREE_SV("ideogram4.cond.layers.0.attention.output"),
      IREE_SV("ideogram4.cond.layers.0.post_attention_hidden"),
      IREE_SV("ideogram4.cond.layers.0.ffn.input"),
      IREE_SV("ideogram4.cond.layers.0.ffn.w1_projection.output"),
      IREE_SV("ideogram4.cond.layers.0.ffn.w3_projection.output"),
      IREE_SV("ideogram4.cond.layers.0.ffn.hidden"),
      IREE_SV("ideogram4.cond.layers.0.ffn.output"),
      IREE_SV("ideogram4.cond.layers.0.hidden"),
      IREE_SV("ideogram4.cond.layers.1.hidden"),
      IREE_SV("ideogram4.cond.layers.2.hidden"),
      IREE_SV("ideogram4.cond.layers.16.hidden"),
      IREE_SV("ideogram4.cond.layers.33.hidden"),
      IREE_SV("ideogram4.cond.final.scale"),
      IREE_SV("ideogram4.cond.final.normalized"),
      IREE_SV("ideogram4.cond.final.projected"),
      IREE_SV("ideogram4.cond.output.velocity"),
  };
  iree_string_view_list_t cond_dit_capture_taps = {};
  if (capture_one_step_trace && capture_requested) {
    cond_dit_capture_taps = (iree_string_view_list_t){
        IREE_ARRAYSIZE(cond_dit_capture_tap_names),
        cond_dit_capture_tap_names,
    };
  }

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release>
      cond_plan;
  IREE_ASSERT_OK(
      PlanDitStage(cond_stage.get(), cond_request, dit_weight_execution_format,
                   cond_dit_capture_taps, &diagnostics_sink, cond_plan.out()));
  IREE_ASSERT_OK(
      WritePlanJsonIfRequested(cond_plan.get(), IREE_SV("cond.json")));
  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release>
      uncond_plan;
  IREE_ASSERT_OK(PlanDitStage(
      uncond_stage.get(), uncond_request, dit_weight_execution_format,
      iree_string_view_list_empty(), &diagnostics_sink, uncond_plan.out()));
  IREE_ASSERT_OK(
      WritePlanJsonIfRequested(uncond_plan.get(), IREE_SV("uncond.json")));
  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release>
      sampler_plan;
  IREE_ASSERT_OK(PlanSamplerStage(sampler_stage.get(),
                                  cond_request.latent_shape, &diagnostics_sink,
                                  sampler_plan.out()));
  IREE_ASSERT_OK(
      WritePlanJsonIfRequested(sampler_plan.get(), IREE_SV("sampler.json")));
  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release>
      decode_plan;
  IREE_ASSERT_OK(PlanDecodeStage(decode_stage.get(), cond_request.latent_shape,
                                 &diagnostics_sink, decode_plan.out()));
  IREE_ASSERT_OK(
      WritePlanJsonIfRequested(decode_plan.get(), IREE_SV("decode.json")));

  iree_host_size_t decoded_boundary_index = 0;
  IREE_ASSERT_OK(FindBoundaryBindingIndex(decode_plan.get(),
                                          IREE_SV("media.image.decoded"),
                                          &decoded_boundary_index));
  const id4_pipeline_boundary_tensor_plan_t* decoded_boundary_plan =
      id4_pipeline_plan_boundary_tensor_at(decode_plan.get(),
                                           decoded_boundary_index);
  ASSERT_NE(decoded_boundary_plan, nullptr);
  ASSERT_GE(decoded_boundary_plan->layout.shape.rank, 2u);
  ASSERT_LE(decoded_boundary_plan->layout.shape.dims[0], UINT32_MAX);
  ASSERT_LE(decoded_boundary_plan->layout.shape.dims[1], UINT32_MAX);
  ScopedDenoiseSchedule denoise_schedule;
  IREE_ASSERT_OK(id4_ideogram4_sampler_preset_lower_schedule(
      sampler_preset,
      static_cast<uint32_t>(decoded_boundary_plan->layout.shape.dims[0]),
      static_cast<uint32_t>(decoded_boundary_plan->layout.shape.dims[1]),
      iree_allocator_system(), &denoise_schedule.value));

  id4::test::BufferBindingSet cond_boundaries;
  IREE_ASSERT_OK(id4::test::AllocateBoundaryBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, cond_plan.get(),
      &cond_boundaries));
  id4::test::BufferBindingSet cond_taps;
  IREE_ASSERT_OK(id4::test::AllocateDiagnosticTapBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, cond_plan.get(),
      &cond_taps));
  id4::test::BufferBindingSet uncond_boundaries;
  IREE_ASSERT_OK(id4::test::AllocateBoundaryBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, uncond_plan.get(),
      &uncond_boundaries));
  id4::test::BufferBindingSet uncond_taps;
  IREE_ASSERT_OK(id4::test::AllocateDiagnosticTapBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, uncond_plan.get(),
      &uncond_taps));
  id4::test::BufferBindingSet sampler_boundaries;
  IREE_ASSERT_OK(id4::test::AllocateBoundaryBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, sampler_plan.get(),
      &sampler_boundaries));
  id4::test::BufferBindingSet sampler_taps;
  IREE_ASSERT_OK(id4::test::AllocateDiagnosticTapBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, sampler_plan.get(),
      &sampler_taps));
  id4::test::BufferBindingSet decode_boundaries;
  IREE_ASSERT_OK(id4::test::AllocateBoundaryBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, decode_plan.get(),
      &decode_boundaries));
  id4::test::BufferBindingSet decode_taps;
  IREE_ASSERT_OK(id4::test::AllocateDiagnosticTapBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, decode_plan.get(),
      &decode_taps));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      update_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, update_semaphore.out()));
  uint64_t update_value = 0;
  RetainedHalFiles upload_files;
  IREE_ASSERT_OK(UpdateBoundaryFromFixture(
      context.device.get(), cond_plan.get(), cond_boundaries, IREE_SV("x"),
      fixture_tensors, cond_input_stage, IREE_SV("x"), update_semaphore.get(),
      &update_value, &upload_files));
  IREE_ASSERT_OK(UpdateBoundaryFromFixture(
      context.device.get(), cond_plan.get(), cond_boundaries,
      IREE_SV("timestep"), fixture_tensors, cond_input_stage,
      IREE_SV("timestep"), update_semaphore.get(), &update_value,
      &upload_files));
  IREE_ASSERT_OK(UpdateBoundaryFromFixture(
      context.device.get(), cond_plan.get(), cond_boundaries,
      IREE_SV("image_indicator"), fixture_tensors, cond_input_stage,
      IREE_SV("image_indicator"), update_semaphore.get(), &update_value,
      &upload_files));
  IREE_ASSERT_OK(UpdateBoundaryFromFixture(
      context.device.get(), cond_plan.get(), cond_boundaries,
      IREE_SV("position_embedding"), fixture_tensors, cond_input_stage,
      IREE_SV("position_embedding"), update_semaphore.get(), &update_value,
      &upload_files));
  IREE_ASSERT_OK(UpdateBoundaryFromFixture(
      context.device.get(), cond_plan.get(), cond_boundaries,
      IREE_SV("condition"), fixture_tensors, cond_input_stage,
      IREE_SV("context"), update_semaphore.get(), &update_value,
      &upload_files));
  IREE_ASSERT_OK(UpdateBoundaryFromFixture(
      context.device.get(), uncond_plan.get(), uncond_boundaries, IREE_SV("x"),
      fixture_tensors, uncond_input_stage, IREE_SV("x"), update_semaphore.get(),
      &update_value, &upload_files));
  IREE_ASSERT_OK(UpdateBoundaryFromFixture(
      context.device.get(), uncond_plan.get(), uncond_boundaries,
      IREE_SV("timestep"), fixture_tensors, uncond_input_stage,
      IREE_SV("timestep"), update_semaphore.get(), &update_value,
      &upload_files));
  IREE_ASSERT_OK(UpdateBoundaryFromFixture(
      context.device.get(), uncond_plan.get(), uncond_boundaries,
      IREE_SV("image_indicator"), fixture_tensors, uncond_input_stage,
      IREE_SV("image_indicator"), update_semaphore.get(), &update_value,
      &upload_files));
  IREE_ASSERT_OK(UpdateBoundaryFromFixture(
      context.device.get(), uncond_plan.get(), uncond_boundaries,
      IREE_SV("position_embedding"), fixture_tensors, uncond_input_stage,
      IREE_SV("position_embedding"), update_semaphore.get(), &update_value,
      &upload_files));
  IREE_ASSERT_OK(UpdateBoundaryFromFixture(
      context.device.get(), sampler_plan.get(), sampler_boundaries,
      IREE_SV("x_t"), fixture_tensors, IREE_SV("sampler.step.1"),
      IREE_SV("x_t"), update_semaphore.get(), &update_value, &upload_files));
  std::vector<uint8_t> current_x_bytes;
  IREE_ASSERT_OK(LoadFixtureBytes(fixture_tensors, cond_input_stage,
                                  IREE_SV("x"), &current_x_bytes));
  IREE_ASSERT_OK(id4::test::QueueFillBoundaryTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, cond_plan.get(),
      cond_boundaries, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED,
      &kOutputSentinel, sizeof(kOutputSentinel), update_semaphore.get(),
      &update_value));
  IREE_ASSERT_OK(id4::test::QueueFillBoundaryTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, uncond_plan.get(),
      uncond_boundaries, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED,
      &kOutputSentinel, sizeof(kOutputSentinel), update_semaphore.get(),
      &update_value));
  IREE_ASSERT_OK(id4::test::QueueFillBoundaryTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, sampler_plan.get(),
      sampler_boundaries, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED,
      &kOutputSentinel, sizeof(kOutputSentinel), update_semaphore.get(),
      &update_value));
  IREE_ASSERT_OK(id4::test::QueueFillBoundaryTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, decode_plan.get(),
      decode_boundaries, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_EXPORTED,
      &kOutputSentinel, sizeof(kOutputSentinel), update_semaphore.get(),
      &update_value));
  IREE_ASSERT_OK(id4::test::QueueFillDiagnosticTapTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, sampler_taps,
      &kOutputSentinel, sizeof(kOutputSentinel), update_semaphore.get(),
      &update_value));

  iree_hal_buffer_binding_t cond_velocity = {};
  IREE_ASSERT_OK(id4::test::FindBoundaryBinding(
      cond_plan.get(), cond_boundaries, IREE_SV("velocity"), &cond_velocity));
  iree_hal_buffer_binding_t uncond_velocity = {};
  IREE_ASSERT_OK(
      id4::test::FindBoundaryBinding(uncond_plan.get(), uncond_boundaries,
                                     IREE_SV("velocity"), &uncond_velocity));
  IREE_ASSERT_OK(ReplaceBoundaryBinding(sampler_plan.get(), &sampler_boundaries,
                                        IREE_SV("cond_out"), cond_velocity));
  IREE_ASSERT_OK(ReplaceBoundaryBinding(sampler_plan.get(), &sampler_boundaries,
                                        IREE_SV("uncond_out"),
                                        uncond_velocity));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      cond_prepare_done;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, cond_prepare_done.out()));
  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      uncond_prepare_done;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, uncond_prepare_done.out()));
  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      decode_prepare_done;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, decode_prepare_done.out()));
  id4::test::SemaphoreListStorage cond_prepare_signal;
  cond_prepare_signal.semaphore = cond_prepare_done.get();
  cond_prepare_signal.payload_value = 1;
  id4::test::SemaphoreListStorage uncond_prepare_signal;
  uncond_prepare_signal.semaphore = uncond_prepare_done.get();
  uncond_prepare_signal.payload_value = 1;
  id4::test::SemaphoreListStorage decode_prepare_signal;
  decode_prepare_signal.semaphore = decode_prepare_done.get();
  decode_prepare_signal.payload_value = 1;

  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      cond_bundle;
  IREE_ASSERT_OK(PrepareStage(
      cond_stage.get(), cond_plan.get(),
      id4_pipeline_stage_parameters(
          id4_pipeline_checkpoint_parameter_source(cond_provider.get()),
          ID4_PIPELINE_STAGE_PARAMETER_RESIDENCY_RESIDENT,
          /*maximum_parameter_window_byte_length=*/0),
      kernel_library.get(), iree_hal_semaphore_list_empty(),
      cond_prepare_signal.list(), &diagnostics_sink, cond_bundle.out()));
  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      uncond_bundle;
  IREE_ASSERT_OK(PrepareStage(
      uncond_stage.get(), uncond_plan.get(),
      id4_pipeline_stage_parameters(
          id4_pipeline_checkpoint_parameter_source(uncond_provider.get()),
          ID4_PIPELINE_STAGE_PARAMETER_RESIDENCY_RESIDENT,
          /*maximum_parameter_window_byte_length=*/0),
      kernel_library.get(), iree_hal_semaphore_list_empty(),
      uncond_prepare_signal.list(), &diagnostics_sink, uncond_bundle.out()));
  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      sampler_bundle;
  IREE_ASSERT_OK(PrepareStage(
      sampler_stage.get(), sampler_plan.get(),
      id4_pipeline_stage_no_parameters(), kernel_library.get(),
      iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(),
      &diagnostics_sink, sampler_bundle.out()));
  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      decode_bundle;
  IREE_ASSERT_OK(PrepareStage(
      decode_stage.get(), decode_plan.get(),
      id4_pipeline_stage_parameters(
          id4_pipeline_checkpoint_parameter_source(vae_provider.get()),
          ID4_PIPELINE_STAGE_PARAMETER_RESIDENCY_RESIDENT,
          /*maximum_parameter_window_byte_length=*/0),
      kernel_library.get(), iree_hal_semaphore_list_empty(),
      decode_prepare_signal.list(), &diagnostics_sink, decode_bundle.out()));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      cond_done;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, cond_done.out()));
  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      uncond_done;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, uncond_done.out()));
  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      sampler_done;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, sampler_done.out()));
  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      decode_done;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, decode_done.out()));

  id4::test::SemaphoreListStorage cond_signal;
  cond_signal.semaphore = cond_done.get();
  id4::test::SemaphoreListStorage uncond_signal;
  uncond_signal.semaphore = uncond_done.get();
  id4::test::SemaphoreListStorage sampler_signal;
  sampler_signal.semaphore = sampler_done.get();

  iree_hal_buffer_binding_t x_next = {};
  IREE_ASSERT_OK(id4::test::FindBoundaryBinding(
      sampler_plan.get(), sampler_boundaries, IREE_SV("x_next"), &x_next));

  for (uint32_t step_ordinal = 0; step_ordinal < denoise_step_count;
       ++step_ordinal) {
    const uint32_t schedule_step_ordinal = start_step_ordinal + step_ordinal;
    const id4_ideogram4_denoise_step_t& step =
        denoise_schedule.value.steps[schedule_step_ordinal];
    fprintf(stderr,
            "[ ID4       ] denoise step %u/%u flow_time=%g "
            "next_flow_time=%g guidance=%g\n",
            schedule_step_ordinal + 1, sampler_step_count,
            (double)step.flow_time, (double)step.next_flow_time,
            (double)step.guidance_scale);

    if (step_ordinal > 0) {
      IREE_ASSERT_OK(QueueUpdateBoundary(
          context.device.get(), cond_plan.get(), cond_boundaries, IREE_SV("x"),
          current_x_bytes.data(), current_x_bytes.size(),
          update_semaphore.get(), &update_value));
      IREE_ASSERT_OK(QueueUpdateBoundary(
          context.device.get(), uncond_plan.get(), uncond_boundaries,
          IREE_SV("x"), current_x_bytes.data(), current_x_bytes.size(),
          update_semaphore.get(), &update_value));
      IREE_ASSERT_OK(QueueUpdateBoundary(
          context.device.get(), sampler_plan.get(), sampler_boundaries,
          IREE_SV("x_t"), current_x_bytes.data(), current_x_bytes.size(),
          update_semaphore.get(), &update_value));
    }
    IREE_ASSERT_OK(QueueUpdateBoundary(context.device.get(), cond_plan.get(),
                                       cond_boundaries, IREE_SV("timestep"),
                                       &step.flow_time, sizeof(step.flow_time),
                                       update_semaphore.get(), &update_value));
    IREE_ASSERT_OK(QueueUpdateBoundary(context.device.get(), uncond_plan.get(),
                                       uncond_boundaries, IREE_SV("timestep"),
                                       &step.flow_time, sizeof(step.flow_time),
                                       update_semaphore.get(), &update_value));
    const float step_values[] = {
        step.flow_time,
        step.next_flow_time,
        step.guidance_scale,
    };
    IREE_ASSERT_OK(QueueUpdateBoundary(context.device.get(), sampler_plan.get(),
                                       sampler_boundaries, IREE_SV("step"),
                                       step_values, sizeof(step_values),
                                       update_semaphore.get(), &update_value));

    id4::test::FixedSemaphoreListStorage cond_wait;
    IREE_ASSERT_OK(cond_wait.push(update_semaphore.get(), update_value));
    IREE_ASSERT_OK(cond_wait.push(cond_prepare_done.get(), 1));
    cond_signal.payload_value = static_cast<uint64_t>(step_ordinal + 1);
    IREE_ASSERT_OK(IssueStage(cond_stage.get(), cond_bundle.get(),
                              cond_boundaries, cond_taps, cond_wait.list(),
                              cond_signal.list(), &diagnostics_sink));
    if (capture_requested) {
      const std::string capture_name =
          StepCaptureName(IREE_SV("cond"), schedule_step_ordinal);
      const iree_string_view_t capture_name_view = iree_make_string_view(
          capture_name.data(),
          static_cast<iree_host_size_t>(capture_name.size()));
      IREE_ASSERT_OK(CaptureStageIfRequested(
          capture_directory, capture_name_view, capture_name_view,
          context.device.get(), cond_plan.get(), cond_boundaries, cond_taps,
          cond_signal.list()));
    }
    if (step_ordinal == 0) {
      IREE_ASSERT_OK(CompareBoundaryWithExpectedFixture(
          context.device.get(), cond_plan.get(), cond_boundaries,
          IREE_SV("velocity"), fixture_tensors, IREE_SV("graph.debug"),
          IREE_SV("ideogram4.cond.output.velocity"), cond_signal.list()));
    }

    id4::test::FixedSemaphoreListStorage uncond_wait;
    IREE_ASSERT_OK(uncond_wait.push(update_semaphore.get(), update_value));
    IREE_ASSERT_OK(uncond_wait.push(uncond_prepare_done.get(), 1));
    uncond_signal.payload_value = static_cast<uint64_t>(step_ordinal + 1);
    IREE_ASSERT_OK(IssueStage(
        uncond_stage.get(), uncond_bundle.get(), uncond_boundaries, uncond_taps,
        uncond_wait.list(), uncond_signal.list(), &diagnostics_sink));
    if (capture_requested) {
      const std::string capture_name =
          StepCaptureName(IREE_SV("uncond"), schedule_step_ordinal);
      const iree_string_view_t capture_name_view = iree_make_string_view(
          capture_name.data(),
          static_cast<iree_host_size_t>(capture_name.size()));
      IREE_ASSERT_OK(CaptureStageIfRequested(
          capture_directory, capture_name_view, capture_name_view,
          context.device.get(), uncond_plan.get(), uncond_boundaries,
          uncond_taps, uncond_signal.list()));
    }
    if (step_ordinal == 0) {
      IREE_ASSERT_OK(CompareBoundaryWithExpectedFixture(
          context.device.get(), uncond_plan.get(), uncond_boundaries,
          IREE_SV("velocity"), fixture_tensors, IREE_SV("graph.debug"),
          IREE_SV("ideogram4.uncond.output.velocity"), uncond_signal.list()));
    }

    id4::test::FixedSemaphoreListStorage sampler_wait;
    IREE_ASSERT_OK(sampler_wait.push(update_semaphore.get(), update_value));
    IREE_ASSERT_OK(sampler_wait.push(cond_done.get(),
                                     static_cast<uint64_t>(step_ordinal + 1)));
    IREE_ASSERT_OK(sampler_wait.push(uncond_done.get(),
                                     static_cast<uint64_t>(step_ordinal + 1)));
    sampler_signal.payload_value = static_cast<uint64_t>(step_ordinal + 1);
    IREE_ASSERT_OK(IssueStage(sampler_stage.get(), sampler_bundle.get(),
                              sampler_boundaries, sampler_taps,
                              sampler_wait.list(), sampler_signal.list(),
                              &diagnostics_sink));
    if (capture_requested) {
      const std::string capture_name =
          StepCaptureName(IREE_SV("sampler"), schedule_step_ordinal);
      const iree_string_view_t capture_name_view = iree_make_string_view(
          capture_name.data(),
          static_cast<iree_host_size_t>(capture_name.size()));
      IREE_ASSERT_OK(CaptureStageIfRequested(
          capture_directory, capture_name_view, capture_name_view,
          context.device.get(), sampler_plan.get(), sampler_boundaries,
          sampler_taps, sampler_signal.list()));
    }

    id4::test::SemaphoreListStorage sampler_read_wait;
    sampler_read_wait.semaphore = sampler_done.get();
    sampler_read_wait.payload_value = static_cast<uint64_t>(step_ordinal + 1);
    if (step_ordinal == 0) {
      IREE_ASSERT_OK(CompareBoundaryWithExpectedFixture(
          context.device.get(), sampler_plan.get(), sampler_boundaries,
          IREE_SV("x_next"), fixture_tensors, IREE_SV("sampler.step.1"),
          IREE_SV("x_next"), sampler_read_wait.list()));
    }
    if (step_ordinal + 1 < step_count) {
      std::vector<uint8_t> next_x_bytes;
      IREE_ASSERT_OK(id4::test::ReadBindingToHost(
          context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &x_next,
          sampler_read_wait.list(), &next_x_bytes));
      current_x_bytes = std::move(next_x_bytes);
    }
  }

  IREE_ASSERT_OK(ReplaceBoundaryBinding(decode_plan.get(), &decode_boundaries,
                                        IREE_SV("media.latent.diffusion"),
                                        x_next));

  id4::test::FixedSemaphoreListStorage decode_wait;
  IREE_ASSERT_OK(decode_wait.push(update_semaphore.get(), update_value));
  IREE_ASSERT_OK(decode_wait.push(sampler_done.get(), step_count));
  IREE_ASSERT_OK(decode_wait.push(decode_prepare_done.get(), 1));
  id4::test::SemaphoreListStorage decode_signal;
  decode_signal.semaphore = decode_done.get();
  decode_signal.payload_value = 1;
  IREE_ASSERT_OK(IssueStage(decode_stage.get(), decode_bundle.get(),
                            decode_boundaries, decode_taps, decode_wait.list(),
                            decode_signal.list(), &diagnostics_sink));

  iree_hal_buffer_binding_t decoded = {};
  IREE_ASSERT_OK(
      id4::test::FindBoundaryBinding(decode_plan.get(), decode_boundaries,
                                     IREE_SV("media.image.decoded"), &decoded));
  id4::test::SemaphoreListStorage decode_read_wait;
  decode_read_wait.semaphore = decode_done.get();
  decode_read_wait.payload_value = 1;
  std::vector<uint8_t> decoded_bytes;
  IREE_ASSERT_OK(id4::test::ReadBindingToHost(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &decoded,
      decode_read_wait.list(), &decoded_bytes));
  IREE_ASSERT_OK(VerifyDecodedImageContents(decoded_bytes));

  const id4_pipeline_boundary_tensor_plan_t* decoded_boundary = nullptr;
  for (iree_host_size_t i = 0;
       i < id4_pipeline_plan_boundary_tensor_count(decode_plan.get()); ++i) {
    const id4_pipeline_boundary_tensor_plan_t* boundary =
        id4_pipeline_plan_boundary_tensor_at(decode_plan.get(), i);
    if (boundary && iree_string_view_equal(boundary->layout.name,
                                           IREE_SV("media.image.decoded"))) {
      decoded_boundary = boundary;
      break;
    }
  }
  ASSERT_NE(decoded_boundary, nullptr);
  IREE_ASSERT_OK(
      MaybeWriteDecodedImage(iree_make_cstring_view(FLAG_id4_output_image),
                             decoded_boundary->layout.shape, decoded_bytes));
}

}  // namespace
