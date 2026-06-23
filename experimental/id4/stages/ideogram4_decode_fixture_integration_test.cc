// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cmath>
#include <cstring>
#include <vector>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/ideogram4_decode.h"
#include "experimental/id4/tooling/image.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

IREE_FLAG(string, id4_fixture_dir, "",
          "Directory containing an ID4 final-latent fixture manifest and "
          "tensor payloads used to initialize the decode stage input.");
IREE_FLAG(string, id4_output_image, "",
          "Optional binary PPM path receiving the decoded image tensor.");

namespace {

constexpr uint8_t kOutputSentinel = 0xA5;

static iree_status_t FindFixtureTensor(
    const id4::test::FixtureTensorSet& fixture_tensors, iree_string_view_t role,
    iree_string_view_t name, const id4::test::FixtureTensor** out_tensor) {
  *out_tensor = fixture_tensors.FindTensor(role, name);
  if (*out_tensor) return iree_ok_status();
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "fixture tensor `%.*s` with role `%.*s` not found",
                          static_cast<int>(name.size), name.data,
                          static_cast<int>(role.size), role.data);
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

static iree_status_t CreateDecodeStage(
    const id4::test::LiveStageContext& context,
    id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = nullptr;

  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = context.device_group.get();
  services.executable_cache = context.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_ideogram4_decode_stage_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = context.kernel_cache.get();
  create_options.model = *id4_ideogram4_decode_program_ideogram4_model_config();
  return id4_ideogram4_decode_stage_create(&create_options,
                                           iree_allocator_system(), out_stage);
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

TEST(Ideogram4DecodeStageFixtureIntegration,
     PrepareAndIssueFinalLatentFixture) {
  iree_string_view_t fixture_directory =
      iree_make_cstring_view(FLAG_id4_fixture_dir);
  ASSERT_FALSE(iree_string_view_is_empty(fixture_directory))
      << "--id4_fixture_dir is required for decode fixture integration";

  id4::test::FixtureTensorSet fixture_tensors;
  IREE_ASSERT_OK(
      id4::test::LoadFixtureTensors(fixture_directory, &fixture_tensors));

  const id4::test::FixtureTensor* latent = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("input"),
                                   IREE_SV("latent"), &latent));
  ASSERT_EQ(latent->dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  const id4::test::FixtureTensor* internal_latent = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("tap"),
                                   IREE_SV("vae.flux2.internal_latent"),
                                   &internal_latent));
  ASSERT_EQ(internal_latent->dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  const id4::test::FixtureTensor* post_quant = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("tap"),
                                   IREE_SV("vae.flux2.post_quant_conv"),
                                   &post_quant));
  ASSERT_EQ(post_quant->dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  const id4::test::FixtureTensor* decoder_conv_in = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("tap"),
                                   IREE_SV("vae.decoder.conv_in"),
                                   &decoder_conv_in));
  ASSERT_EQ(decoder_conv_in->dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  const id4::test::FixtureTensor* mid_block_1_norm1_silu = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(
      fixture_tensors, IREE_SV("tap"),
      IREE_SV("vae.decoder.mid.block_1.norm1_silu"), &mid_block_1_norm1_silu));
  ASSERT_EQ(mid_block_1_norm1_silu->dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  const id4::test::FixtureTensor* mid_block_1_conv1 = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("tap"),
                                   IREE_SV("vae.decoder.mid.block_1.conv1"),
                                   &mid_block_1_conv1));
  ASSERT_EQ(mid_block_1_conv1->dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  const id4::test::FixtureTensor* mid_block_1_norm2_silu = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(
      fixture_tensors, IREE_SV("tap"),
      IREE_SV("vae.decoder.mid.block_1.norm2_silu"), &mid_block_1_norm2_silu));
  ASSERT_EQ(mid_block_1_norm2_silu->dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  const id4::test::FixtureTensor* mid_block_1_conv2 = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("tap"),
                                   IREE_SV("vae.decoder.mid.block_1.conv2"),
                                   &mid_block_1_conv2));
  ASSERT_EQ(mid_block_1_conv2->dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  const id4::test::FixtureTensor* mid_block_1_output = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("tap"),
                                   IREE_SV("vae.decoder.mid.block_1.output"),
                                   &mid_block_1_output));
  ASSERT_EQ(mid_block_1_output->dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  const id4::test::FixtureTensor* mid_attention_norm = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("tap"),
                                   IREE_SV("vae.decoder.mid.attn_1.norm"),
                                   &mid_attention_norm));
  ASSERT_EQ(mid_attention_norm->dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  const id4::test::FixtureTensor* mid_attention_q = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("tap"),
                                   IREE_SV("vae.decoder.mid.attn_1.q"),
                                   &mid_attention_q));
  ASSERT_EQ(mid_attention_q->dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  const id4::test::FixtureTensor* mid_attention_k = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("tap"),
                                   IREE_SV("vae.decoder.mid.attn_1.k"),
                                   &mid_attention_k));
  ASSERT_EQ(mid_attention_k->dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  const id4::test::FixtureTensor* mid_attention_v = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("tap"),
                                   IREE_SV("vae.decoder.mid.attn_1.v"),
                                   &mid_attention_v));
  ASSERT_EQ(mid_attention_v->dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  const id4::test::FixtureTensor* mid_attention = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("tap"),
                                   IREE_SV("vae.decoder.mid.attn_1.attention"),
                                   &mid_attention));
  ASSERT_EQ(mid_attention->dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  const id4::test::FixtureTensor* mid_attention_proj = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("tap"),
                                   IREE_SV("vae.decoder.mid.attn_1.proj_out"),
                                   &mid_attention_proj));
  ASSERT_EQ(mid_attention_proj->dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  const id4::test::FixtureTensor* mid_attention_output = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("tap"),
                                   IREE_SV("vae.decoder.mid.attn_1.output"),
                                   &mid_attention_output));
  ASSERT_EQ(mid_attention_output->dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  const id4::test::FixtureTensor* mid_block_2_output = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("tap"),
                                   IREE_SV("vae.decoder.mid.block_2.output"),
                                   &mid_block_2_output));
  ASSERT_EQ(mid_block_2_output->dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);
  const id4::test::FixtureTensor* expected_decoded_image = nullptr;
  IREE_ASSERT_OK(FindFixtureTensor(fixture_tensors, IREE_SV("expected"),
                                   IREE_SV("media.image.decoded"),
                                   &expected_decoded_image));
  ASSERT_EQ(expected_decoded_image->dtype, ID4_PIPELINE_TENSOR_DTYPE_F32);

  id4_pipeline_program_shape_t diffusion_latent_shape;
  IREE_ASSERT_OK(MakeProgramShape(latent->shape, &diffusion_latent_shape));

  id4::test::LiveStageContext context;
  IREE_ASSERT_OK(id4::test::CreateLiveStageContextFromFlags(&context));

  id4::test::KernelLibraryRef kernel_library;
  IREE_ASSERT_OK(id4::test::CreateEmbeddedKernelLibrary(kernel_library.out()));

  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release> stage;
  IREE_ASSERT_OK(CreateDecodeStage(context, stage.out()));
  id4::test::OwningRef<iree_io_parameter_provider_t,
                       iree_io_parameter_provider_release>
      parameter_provider;
  IREE_ASSERT_OK(id4::test::CreateParameterProviderFromFlags(
      iree_string_view_empty(), parameter_provider.out()));

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_ASSERT_OK(id4_pipeline_stage_load(stage.get(), &load_options));

  id4_ideogram4_decode_stage_plan_options_t decode_options;
  std::memset(&decode_options, 0, sizeof(decode_options));
  decode_options.structure_size = sizeof(decode_options);
  decode_options.request.diffusion_latent_shape = diffusion_latent_shape;
  decode_options.request.vae_tiling.mode = ID4_VAE_TILING_MODE_DISABLED;

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &decode_options;
  const iree_string_view_t diagnostic_tap_names[] = {
      IREE_SV("vae.flux2.internal_latent"),
      IREE_SV("vae.flux2.post_quant_conv"),
      IREE_SV("vae.decoder.conv_in"),
      IREE_SV("vae.decoder.mid.block_1.norm1_silu"),
      IREE_SV("vae.decoder.mid.block_1.conv1"),
      IREE_SV("vae.decoder.mid.block_1.norm2_silu"),
      IREE_SV("vae.decoder.mid.block_1.conv2"),
      IREE_SV("vae.decoder.mid.block_1.output"),
      IREE_SV("vae.decoder.mid.attn_1.norm"),
      IREE_SV("vae.decoder.mid.attn_1.q"),
      IREE_SV("vae.decoder.mid.attn_1.k"),
      IREE_SV("vae.decoder.mid.attn_1.v"),
      IREE_SV("vae.decoder.mid.attn_1.attention"),
      IREE_SV("vae.decoder.mid.attn_1.proj_out"),
      IREE_SV("vae.decoder.mid.attn_1.output"),
      IREE_SV("vae.decoder.mid.block_2.output"),
  };
  plan_options.flags = ID4_PIPELINE_STAGE_PLAN_FLAG_CAPTURE_DIAGNOSTIC_TAPS;
  plan_options.diagnostic_tap_names = (iree_string_view_list_t){
      IREE_ARRAYSIZE(diagnostic_tap_names),
      diagnostic_tap_names,
  };
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &diagnostics_sink;

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_ASSERT_OK(
      id4_pipeline_stage_plan(stage.get(), &plan_options, plan.out()));

  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_provider = parameter_provider.get();
  prepare_options.kernel_library = kernel_library.get();
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.diagnostics_sink = &diagnostics_sink;
  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      prepare_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));
  id4::test::SemaphoreListStorage prepare_signal;
  prepare_signal.semaphore = prepare_semaphore.get();
  prepare_signal.payload_value = 1;
  prepare_options.signal_semaphore_list = prepare_signal.list();

  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      bundle;
  IREE_ASSERT_OK(id4_pipeline_stage_prepare(stage.get(), plan.get(),
                                            &prepare_options, bundle.out()));

  id4::test::BufferBindingSet boundary_bindings;
  IREE_ASSERT_OK(id4::test::AllocateBoundaryBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      &boundary_bindings));
  id4::test::BufferBindingSet diagnostic_tap_bindings;
  IREE_ASSERT_OK(id4::test::AllocateDiagnosticTapBindings(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      &diagnostic_tap_bindings));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      update_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, update_semaphore.out()));
  uint64_t update_value = 0;

  iree_hal_buffer_binding_t diffusion_binding = {};
  IREE_ASSERT_OK(id4::test::FindBoundaryBinding(
      plan.get(), boundary_bindings, IREE_SV("media.latent.diffusion"),
      &diffusion_binding));
  IREE_ASSERT_OK(id4::test::QueueUpdateBinding(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &diffusion_binding,
      latent->payload.data(), latent->payload.size(), update_semaphore.get(),
      &update_value));

  iree_hal_buffer_binding_t decoded_binding = {};
  IREE_ASSERT_OK(id4::test::FindBoundaryBinding(plan.get(), boundary_bindings,
                                                IREE_SV("media.image.decoded"),
                                                &decoded_binding));
  IREE_ASSERT_OK(id4::test::QueueFillBinding(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &decoded_binding,
      &kOutputSentinel, sizeof(kOutputSentinel), update_semaphore.get(),
      &update_value));
  IREE_ASSERT_OK(id4::test::QueueFillDiagnosticTapTensors(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      diagnostic_tap_bindings, &kOutputSentinel, sizeof(kOutputSentinel),
      update_semaphore.get(), &update_value));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      issue_semaphore;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, issue_semaphore.out()));

  id4::test::SemaphoreListStorage issue_wait;
  issue_wait.semaphore = update_semaphore.get();
  issue_wait.payload_value = update_value;
  id4::test::SemaphoreListStorage issue_signal;
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

  id4::test::SemaphoreListStorage read_wait;
  read_wait.semaphore = issue_semaphore.get();
  read_wait.payload_value = issue_signal.payload_value;
  std::vector<uint8_t> decoded_bytes;
  IREE_ASSERT_OK(id4::test::ReadBindingToHost(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &decoded_binding,
      read_wait.list(), &decoded_bytes));
  IREE_ASSERT_OK(VerifyDecodedImageContents(decoded_bytes));
  iree_string_view_t output_image_path =
      iree_make_cstring_view(FLAG_id4_output_image);
  if (!iree_string_view_is_empty(output_image_path)) {
    id4_tooling_write_f32_rgb_ppm_options_t image_options;
    std::memset(&image_options, 0, sizeof(image_options));
    image_options.structure_size = sizeof(image_options);
    image_options.path = output_image_path;
    image_options.shape = expected_decoded_image->shape;
    image_options.pixels =
        iree_make_const_byte_span(decoded_bytes.data(), decoded_bytes.size());
    image_options.normalization =
        ID4_TOOLING_IMAGE_NORMALIZATION_MINUS_ONE_TO_ONE;
    image_options.host_allocator = iree_allocator_system();
    IREE_ASSERT_OK(id4_tooling_write_f32_rgb_ppm(&image_options));
  }
  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &decoded_binding,
      read_wait.list(), *expected_decoded_image));

  iree_hal_buffer_binding_t internal_latent_binding = {};
  IREE_ASSERT_OK(id4::test::FindDiagnosticTapBinding(
      plan.get(), diagnostic_tap_bindings, IREE_SV("vae.flux2.internal_latent"),
      &internal_latent_binding));
  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      &internal_latent_binding, read_wait.list(), *internal_latent));

  iree_hal_buffer_binding_t post_quant_binding = {};
  IREE_ASSERT_OK(id4::test::FindDiagnosticTapBinding(
      plan.get(), diagnostic_tap_bindings, IREE_SV("vae.flux2.post_quant_conv"),
      &post_quant_binding));
  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &post_quant_binding,
      read_wait.list(), *post_quant));

  iree_hal_buffer_binding_t decoder_conv_in_binding = {};
  IREE_ASSERT_OK(id4::test::FindDiagnosticTapBinding(
      plan.get(), diagnostic_tap_bindings, IREE_SV("vae.decoder.conv_in"),
      &decoder_conv_in_binding));
  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      &decoder_conv_in_binding, read_wait.list(), *decoder_conv_in));

  iree_hal_buffer_binding_t mid_block_1_norm1_silu_binding = {};
  IREE_ASSERT_OK(id4::test::FindDiagnosticTapBinding(
      plan.get(), diagnostic_tap_bindings,
      IREE_SV("vae.decoder.mid.block_1.norm1_silu"),
      &mid_block_1_norm1_silu_binding));
  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      &mid_block_1_norm1_silu_binding, read_wait.list(),
      *mid_block_1_norm1_silu));

  iree_hal_buffer_binding_t mid_block_1_conv1_binding = {};
  IREE_ASSERT_OK(id4::test::FindDiagnosticTapBinding(
      plan.get(), diagnostic_tap_bindings,
      IREE_SV("vae.decoder.mid.block_1.conv1"), &mid_block_1_conv1_binding));
  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      &mid_block_1_conv1_binding, read_wait.list(), *mid_block_1_conv1));

  iree_hal_buffer_binding_t mid_block_1_norm2_silu_binding = {};
  IREE_ASSERT_OK(id4::test::FindDiagnosticTapBinding(
      plan.get(), diagnostic_tap_bindings,
      IREE_SV("vae.decoder.mid.block_1.norm2_silu"),
      &mid_block_1_norm2_silu_binding));
  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      &mid_block_1_norm2_silu_binding, read_wait.list(),
      *mid_block_1_norm2_silu));

  iree_hal_buffer_binding_t mid_block_1_conv2_binding = {};
  IREE_ASSERT_OK(id4::test::FindDiagnosticTapBinding(
      plan.get(), diagnostic_tap_bindings,
      IREE_SV("vae.decoder.mid.block_1.conv2"), &mid_block_1_conv2_binding));
  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      &mid_block_1_conv2_binding, read_wait.list(), *mid_block_1_conv2));

  iree_hal_buffer_binding_t mid_block_1_output_binding = {};
  IREE_ASSERT_OK(id4::test::FindDiagnosticTapBinding(
      plan.get(), diagnostic_tap_bindings,
      IREE_SV("vae.decoder.mid.block_1.output"), &mid_block_1_output_binding));
  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      &mid_block_1_output_binding, read_wait.list(), *mid_block_1_output));

  iree_hal_buffer_binding_t mid_attention_norm_binding = {};
  IREE_ASSERT_OK(id4::test::FindDiagnosticTapBinding(
      plan.get(), diagnostic_tap_bindings,
      IREE_SV("vae.decoder.mid.attn_1.norm"), &mid_attention_norm_binding));
  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      &mid_attention_norm_binding, read_wait.list(), *mid_attention_norm));

  iree_hal_buffer_binding_t mid_attention_q_binding = {};
  IREE_ASSERT_OK(id4::test::FindDiagnosticTapBinding(
      plan.get(), diagnostic_tap_bindings, IREE_SV("vae.decoder.mid.attn_1.q"),
      &mid_attention_q_binding));
  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      &mid_attention_q_binding, read_wait.list(), *mid_attention_q));

  iree_hal_buffer_binding_t mid_attention_k_binding = {};
  IREE_ASSERT_OK(id4::test::FindDiagnosticTapBinding(
      plan.get(), diagnostic_tap_bindings, IREE_SV("vae.decoder.mid.attn_1.k"),
      &mid_attention_k_binding));
  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      &mid_attention_k_binding, read_wait.list(), *mid_attention_k));

  iree_hal_buffer_binding_t mid_attention_v_binding = {};
  IREE_ASSERT_OK(id4::test::FindDiagnosticTapBinding(
      plan.get(), diagnostic_tap_bindings, IREE_SV("vae.decoder.mid.attn_1.v"),
      &mid_attention_v_binding));
  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      &mid_attention_v_binding, read_wait.list(), *mid_attention_v));

  iree_hal_buffer_binding_t mid_attention_binding = {};
  IREE_ASSERT_OK(id4::test::FindDiagnosticTapBinding(
      plan.get(), diagnostic_tap_bindings,
      IREE_SV("vae.decoder.mid.attn_1.attention"), &mid_attention_binding));
  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, &mid_attention_binding,
      read_wait.list(), *mid_attention));

  iree_hal_buffer_binding_t mid_attention_proj_binding = {};
  IREE_ASSERT_OK(id4::test::FindDiagnosticTapBinding(
      plan.get(), diagnostic_tap_bindings,
      IREE_SV("vae.decoder.mid.attn_1.proj_out"), &mid_attention_proj_binding));
  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      &mid_attention_proj_binding, read_wait.list(), *mid_attention_proj));

  iree_hal_buffer_binding_t mid_attention_output_binding = {};
  IREE_ASSERT_OK(id4::test::FindDiagnosticTapBinding(
      plan.get(), diagnostic_tap_bindings,
      IREE_SV("vae.decoder.mid.attn_1.output"), &mid_attention_output_binding));
  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      &mid_attention_output_binding, read_wait.list(), *mid_attention_output));

  iree_hal_buffer_binding_t mid_block_2_output_binding = {};
  IREE_ASSERT_OK(id4::test::FindDiagnosticTapBinding(
      plan.get(), diagnostic_tap_bindings,
      IREE_SV("vae.decoder.mid.block_2.output"), &mid_block_2_output_binding));
  IREE_ASSERT_OK(id4::test::CompareF32BindingWithFixtureTensor(
      context.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY,
      &mid_block_2_output_binding, read_wait.list(), *mid_block_2_output));
}

}  // namespace
