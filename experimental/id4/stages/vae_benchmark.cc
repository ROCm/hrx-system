// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/licenses/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/pipeline/stage.h"
#include "experimental/id4/stages/hal_integration_util.h"
#include "experimental/id4/stages/vae.h"
#include "experimental/id4/tooling/filesystem.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/io/file_contents.h"
#include "iree/io/parameter_provider.h"
#include "iree/testing/benchmark.h"
#include "iree/tooling/device_util.h"

IREE_FLAG(string, id4_plan_output_dir, "",
          "Optional directory receiving benchmark VAE stage plan JSON files.");
IREE_FLAG(string, id4_fixture_dir, "",
          "Directory containing an ID4 VAE latent fixture used by the "
          "fixture issue benchmark.");

namespace {

static constexpr uint32_t kFlux2LatentWidth = 64;
static constexpr uint32_t kFlux2LatentHeight = 64;
static constexpr uint32_t kFlux2LatentChannelCount = 128;
static constexpr uint32_t kFlux2LatentBatchCount = 1;
static constexpr uint64_t kFlux2DecodedElementCount = 1024ull * 1024ull * 3ull;
static constexpr uint32_t kFlux2SmallLatentWidth = 8;
static constexpr uint32_t kFlux2SmallLatentHeight = 8;
static constexpr uint64_t kFlux2SmallDecodedElementCount =
    128ull * 128ull * 3ull;
static constexpr uint32_t kFlux2TileLatentWidth = 32;
static constexpr uint32_t kFlux2TileLatentHeight = 32;
static constexpr uint64_t kFlux2TileDecodedElementCount =
    512ull * 512ull * 3ull;

struct VaeBenchmarkShape {
  // Latent tensor width used to plan the decode stage.
  uint32_t latent_width;
  // Latent tensor height used to plan the decode stage.
  uint32_t latent_height;
  // Latent tensor batch count used to plan the decode stage.
  uint32_t latent_batch_count;
  // Decoded image element count used for throughput reporting.
  uint64_t decoded_element_count;
  // VAE tiling policy used to plan the decode stage.
  id4_vae_tiling_config_t tiling;
  // Activation and parameter-storage route configured on the stage.
  id4_vae_activation_format_t activation_format;
  // File name used when --id4_plan_output_dir requests a plan dump.
  const char* plan_file_name;
};

static constexpr id4_vae_tiling_config_t DisabledTiling() {
  return {
      // Tiling policy selected for this request.
      ID4_VAE_TILING_MODE_DISABLED,
      // Requested latent tile width for explicit-size policy.
      0,
      // Requested latent tile height for explicit-size policy.
      0,
      // Relative latent width factor or tile-count hint.
      0.0f,
      // Relative latent height factor or tile-count hint.
      0.0f,
      // Requested fractional tile overlap.
      0.0f,
      // Maximum transient bytes for memory-budget policy.
      0,
  };
}

static constexpr id4_vae_tiling_config_t ExplicitTileSizeTiling(
    uint32_t tile_size_x, uint32_t tile_size_y, float overlap) {
  return {
      // Tiling policy selected for this request.
      ID4_VAE_TILING_MODE_EXPLICIT_TILE_SIZE,
      // Requested latent tile width for explicit-size policy.
      tile_size_x,
      // Requested latent tile height for explicit-size policy.
      tile_size_y,
      // Relative latent width factor or tile-count hint.
      0.0f,
      // Relative latent height factor or tile-count hint.
      0.0f,
      // Requested fractional tile overlap.
      overlap,
      // Maximum transient bytes for memory-budget policy.
      0,
  };
}

static constexpr VaeBenchmarkShape kFlux2FullImageShape = {
    // Full 1024x1024 image latent width.
    /*.latent_width=*/kFlux2LatentWidth,
    // Full 1024x1024 image latent height.
    /*.latent_height=*/kFlux2LatentHeight,
    // Single-image decode batch.
    /*.latent_batch_count=*/kFlux2LatentBatchCount,
    // Full 1024x1024 RGB output element count.
    /*.decoded_element_count=*/kFlux2DecodedElementCount,
    // Full-frame VAE decode with no spatial tiling.
    /*.tiling=*/DisabledTiling(),
    // Canonical F32 activation storage.
    /*.activation_format=*/ID4_VAE_ACTIVATION_FORMAT_F32_CANONICAL,
    // Plan dump file name.
    /*.plan_file_name=*/"decode_full_frame.json",
};

static constexpr VaeBenchmarkShape kFlux2TiledFullImageShape = {
    // Full 1024x1024 image latent width.
    /*.latent_width=*/kFlux2LatentWidth,
    // Full 1024x1024 image latent height.
    /*.latent_height=*/kFlux2LatentHeight,
    // Single-image decode batch.
    /*.latent_batch_count=*/kFlux2LatentBatchCount,
    // Full 1024x1024 RGB output element count.
    /*.decoded_element_count=*/kFlux2DecodedElementCount,
    // Stable-style 32x32 latent tiles with 50% overlap.
    /*.tiling=*/
    ExplicitTileSizeTiling(kFlux2TileLatentWidth, kFlux2TileLatentHeight, 0.5f),
    // Canonical F32 activation storage.
    /*.activation_format=*/ID4_VAE_ACTIVATION_FORMAT_F32_CANONICAL,
    // Plan dump file name.
    /*.plan_file_name=*/"decode_tiled.json",
};

static constexpr VaeBenchmarkShape kFlux2TileLocalShape = {
    // Stable-style tile latent width.
    /*.latent_width=*/kFlux2TileLatentWidth,
    // Stable-style tile latent height.
    /*.latent_height=*/kFlux2TileLatentHeight,
    // Single-image decode batch.
    /*.latent_batch_count=*/kFlux2LatentBatchCount,
    // Tile-local 512x512 RGB output element count.
    /*.decoded_element_count=*/kFlux2TileDecodedElementCount,
    // Single-tile local VAE decode with no spatial tiling.
    /*.tiling=*/DisabledTiling(),
    // Canonical F32 activation storage.
    /*.activation_format=*/ID4_VAE_ACTIVATION_FORMAT_F32_CANONICAL,
    // Plan dump file name.
    /*.plan_file_name=*/"decode_tile_local.json",
};

static constexpr VaeBenchmarkShape kFlux2Bf16FullImageShape = {
    // Full 1024x1024 image latent width.
    /*.latent_width=*/kFlux2LatentWidth,
    // Full 1024x1024 image latent height.
    /*.latent_height=*/kFlux2LatentHeight,
    // Single-image decode batch.
    /*.latent_batch_count=*/kFlux2LatentBatchCount,
    // Full 1024x1024 RGB output element count.
    /*.decoded_element_count=*/kFlux2DecodedElementCount,
    // Full-frame VAE decode with no spatial tiling.
    /*.tiling=*/DisabledTiling(),
    // BF16 prelude activation and post-quant storage.
    /*.activation_format=*/ID4_VAE_ACTIVATION_FORMAT_BF16_CONV_INPUT,
    // Plan dump file name.
    /*.plan_file_name=*/"decode_bf16_full_frame.json",
};

static constexpr VaeBenchmarkShape kFlux2Bf16SmallImageShape = {
    // Small bringup image latent width.
    /*.latent_width=*/kFlux2SmallLatentWidth,
    // Small bringup image latent height.
    /*.latent_height=*/kFlux2SmallLatentHeight,
    // Single-image decode batch.
    /*.latent_batch_count=*/kFlux2LatentBatchCount,
    // Small 128x128 RGB output element count.
    /*.decoded_element_count=*/kFlux2SmallDecodedElementCount,
    // Full-frame VAE decode with no spatial tiling.
    /*.tiling=*/DisabledTiling(),
    // BF16 prelude activation and post-quant storage.
    /*.activation_format=*/ID4_VAE_ACTIVATION_FORMAT_BF16_CONV_INPUT,
    // Plan dump file name.
    /*.plan_file_name=*/"decode_bf16_small_frame.json",
};

static constexpr VaeBenchmarkShape kFlux2Bf16TiledFullImageShape = {
    // Full 1024x1024 image latent width.
    /*.latent_width=*/kFlux2LatentWidth,
    // Full 1024x1024 image latent height.
    /*.latent_height=*/kFlux2LatentHeight,
    // Single-image decode batch.
    /*.latent_batch_count=*/kFlux2LatentBatchCount,
    // Full 1024x1024 RGB output element count.
    /*.decoded_element_count=*/kFlux2DecodedElementCount,
    // Stable-style 32x32 latent tiles with 50% overlap.
    /*.tiling=*/
    ExplicitTileSizeTiling(kFlux2TileLatentWidth, kFlux2TileLatentHeight, 0.5f),
    // BF16 prelude activation and post-quant storage.
    /*.activation_format=*/ID4_VAE_ACTIVATION_FORMAT_BF16_CONV_INPUT,
    // Plan dump file name.
    /*.plan_file_name=*/"decode_bf16_tiled.json",
};

static constexpr VaeBenchmarkShape kFlux2Bf16TileLocalShape = {
    // Stable-style tile latent width.
    /*.latent_width=*/kFlux2TileLatentWidth,
    // Stable-style tile latent height.
    /*.latent_height=*/kFlux2TileLatentHeight,
    // Single-image decode batch.
    /*.latent_batch_count=*/kFlux2LatentBatchCount,
    // Tile-local 512x512 RGB output element count.
    /*.decoded_element_count=*/kFlux2TileDecodedElementCount,
    // Single-tile local VAE decode with no spatial tiling.
    /*.tiling=*/DisabledTiling(),
    // BF16 prelude activation and post-quant storage.
    /*.activation_format=*/ID4_VAE_ACTIVATION_FORMAT_BF16_CONV_INPUT,
    // Plan dump file name.
    /*.plan_file_name=*/"decode_bf16_tile_local.json",
};

struct VaeBenchmarkContext {
  // Live HAL, executable cache, and kernel-cache context selected by flags.
  id4::test::LiveStageContext live;
  // Embedded Loom source library used during stage preparation.
  id4::test::KernelLibraryRef kernel_library;
  // Parameter provider created from standard --parameters= flags.
  id4::test::OwningRef<iree_io_parameter_provider_t,
                       iree_io_parameter_provider_release>
      parameter_provider;
  // Loaded VAE stage under benchmark.
  id4::test::OwningRef<id4_pipeline_stage_t, id4_pipeline_stage_release> stage;
  // Diagnostic event counters collected by lifecycle calls.
  id4::test::StageDiagnostics diagnostics = {};
  // Diagnostics sink passed to stage lifecycle calls.
  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  // Reference tensors loaded for the fixture issue benchmark.
  id4::test::FixtureTensorSet fixture_tensors;
};

static iree_status_t CreateVaeStage(
    const id4::test::LiveStageContext& live,
    id4_vae_activation_format_t activation_format,
    id4_pipeline_stage_t** out_stage) {
  IREE_ASSERT_ARGUMENT(out_stage);
  *out_stage = nullptr;

  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = live.device_group.get();
  services.executable_cache = live.executable_cache.get();
  services.host_allocator = iree_allocator_system();

  id4_vae_stage_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = live.kernel_cache.get();
  create_options.model = *id4_vae_program_flux2_model_config();
  create_options.activation_format = activation_format;
  return id4_vae_stage_create(&create_options, iree_allocator_system(),
                              out_stage);
}

static iree_status_t CreateLoadedVaeStageContext(
    VaeBenchmarkContext* out_context,
    id4_vae_activation_format_t activation_format) {
  IREE_ASSERT_ARGUMENT(out_context);
  out_context->diagnostics_sink =
      id4::test::DiagnosticsSink(&out_context->diagnostics);
  IREE_RETURN_IF_ERROR(
      id4::test::CreateLiveStageContextFromFlags(&out_context->live));
  IREE_RETURN_IF_ERROR(CreateVaeStage(out_context->live, activation_format,
                                      out_context->stage.out()));

  id4_pipeline_stage_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &out_context->diagnostics_sink;
  return id4_pipeline_stage_load(out_context->stage.get(), &load_options);
}

static iree_status_t CreateLoadedVaeBenchmarkContext(
    VaeBenchmarkContext* out_context, VaeBenchmarkShape shape) {
  IREE_RETURN_IF_ERROR(
      CreateLoadedVaeStageContext(out_context, shape.activation_format));
  IREE_RETURN_IF_ERROR(id4::test::CreateEmbeddedKernelLibrary(
      out_context->kernel_library.out()));
  return id4::test::CreateParameterProviderFromFlags(
      iree_string_view_empty(), out_context->parameter_provider.out());
}

static iree_status_t LoadFixtureVaeBenchmarkShape(
    VaeBenchmarkContext* context, VaeBenchmarkShape* out_shape) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_shape);
  *out_shape = {};

  const iree_string_view_t fixture_directory =
      iree_make_cstring_view(FLAG_id4_fixture_dir);
  if (iree_string_view_is_empty(fixture_directory)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--id4_fixture_dir is required for the VAE fixture benchmark");
  }
  IREE_RETURN_IF_ERROR(id4::test::LoadFixtureTensors(
      fixture_directory, &context->fixture_tensors));
  const id4::test::FixtureTensor* latent =
      context->fixture_tensors.FindTensor(IREE_SV("input"), IREE_SV("latent"));
  if (!latent) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "fixture input tensor `latent` not found");
  }
  const id4_vae_model_config_t* model = id4_vae_program_flux2_model_config();
  if (latent->dtype != ID4_PIPELINE_TENSOR_DTYPE_F32 ||
      latent->shape.rank != 4 || latent->shape.dims[0] == 0 ||
      latent->shape.dims[1] == 0 ||
      latent->shape.dims[2] != model->latent_channel_count ||
      latent->shape.dims[3] == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fixture `latent` must be non-empty F32 "
                            "[width, height, %" PRIu32 ", batch]",
                            model->latent_channel_count);
  }

  iree_device_size_t decoded_element_count = latent->shape.dims[0];
  if (!iree_device_size_checked_mul(decoded_element_count, model->scale_x,
                                    &decoded_element_count) ||
      !iree_device_size_checked_mul(decoded_element_count,
                                    latent->shape.dims[1],
                                    &decoded_element_count) ||
      !iree_device_size_checked_mul(decoded_element_count, model->scale_y,
                                    &decoded_element_count) ||
      !iree_device_size_checked_mul(decoded_element_count,
                                    model->decoded_channel_count,
                                    &decoded_element_count) ||
      !iree_device_size_checked_mul(decoded_element_count,
                                    latent->shape.dims[3],
                                    &decoded_element_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "fixture decoded element count overflows");
  }

  out_shape->latent_width = latent->shape.dims[0];
  out_shape->latent_height = latent->shape.dims[1];
  out_shape->latent_batch_count = latent->shape.dims[3];
  out_shape->decoded_element_count = decoded_element_count;
  out_shape->tiling = DisabledTiling();
  out_shape->activation_format = ID4_VAE_ACTIVATION_FORMAT_BF16_CONV_INPUT;
  out_shape->plan_file_name = "decode_bf16_fixture.json";
  return iree_ok_status();
}

static iree_status_t CreateVaePlan(VaeBenchmarkContext* context,
                                   VaeBenchmarkShape shape,
                                   id4_pipeline_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = nullptr;

  id4_vae_stage_plan_options_t vae_options;
  std::memset(&vae_options, 0, sizeof(vae_options));
  vae_options.structure_size = sizeof(vae_options);
  vae_options.request.latent_shape = id4_pipeline_program_make_shape_rank4(
      shape.latent_width, shape.latent_height, kFlux2LatentChannelCount,
      shape.latent_batch_count);
  vae_options.request.tiling = shape.tiling;

  id4_pipeline_stage_plan_options_t plan_options;
  std::memset(&plan_options, 0, sizeof(plan_options));
  plan_options.structure_size = sizeof(plan_options);
  plan_options.next = &vae_options;
  plan_options.device_index = 0;
  plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  plan_options.diagnostics_sink = &context->diagnostics_sink;
  return id4_pipeline_stage_plan(context->stage.get(), &plan_options, out_plan);
}

static iree_status_t WritePlanJsonIfRequested(VaeBenchmarkShape shape,
                                              const id4_pipeline_plan_t* plan) {
  iree_string_view_t output_dir =
      iree_make_cstring_view(FLAG_id4_plan_output_dir);
  if (iree_string_view_is_empty(output_dir)) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      id4_tooling_ensure_directory(output_dir, iree_allocator_system()));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status = id4_pipeline_plan_format_json(plan, &builder);
  iree_string_view_t path = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    status = id4_tooling_format_child_path(
        output_dir, iree_make_cstring_view(shape.plan_file_name),
        iree_allocator_system(), &path);
  }
  if (iree_status_is_ok(status)) {
    iree_string_view_t json = iree_string_builder_view(&builder);
    status = iree_io_file_contents_write(
        path, iree_make_const_byte_span(json.data, json.size),
        iree_allocator_system());
  }
  id4_tooling_free_path(&path, iree_allocator_system());
  iree_string_builder_deinitialize(&builder);
  return status;
}

static iree_status_t WritePlanJsonIfRequested(VaeBenchmarkContext* context,
                                              VaeBenchmarkShape shape) {
  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_RETURN_IF_ERROR(CreateVaePlan(context, shape, plan.out()));
  return WritePlanJsonIfRequested(shape, plan.get());
}

static iree_status_t PrepareVaeBundle(VaeBenchmarkContext* context,
                                      const id4_pipeline_plan_t* plan,
                                      iree_hal_semaphore_t* prepare_semaphore,
                                      uint64_t signal_value,
                                      id4_pipeline_bundle_t** out_bundle) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(out_bundle);
  *out_bundle = nullptr;

  id4::test::SemaphoreListStorage signal;
  signal.semaphore = prepare_semaphore;
  signal.payload_value = signal_value;

  id4_pipeline_stage_prepare_options_t prepare_options;
  std::memset(&prepare_options, 0, sizeof(prepare_options));
  prepare_options.structure_size = sizeof(prepare_options);
  prepare_options.parameter_source = id4_pipeline_stage_checkpoint_parameters(
      context->parameter_provider.get(),
      ID4_PIPELINE_STAGE_PARAMETER_RESIDENCY_RESIDENT);
  prepare_options.kernel_library = context->kernel_library.get();
  prepare_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
  prepare_options.signal_semaphore_list = signal.list();
  prepare_options.command_buffer_mode = context->live.command_buffer_mode;
  prepare_options.diagnostics_sink = &context->diagnostics_sink;
  return id4_pipeline_stage_prepare(context->stage.get(), plan,
                                    &prepare_options, out_bundle);
}

static iree_status_t IssueVaeBundle(
    VaeBenchmarkContext* context, id4_pipeline_bundle_t* bundle,
    const id4::test::BufferBindingSet& boundary_bindings,
    iree_hal_semaphore_t* semaphore, uint64_t wait_value,
    uint64_t signal_value) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(bundle);

  id4::test::SemaphoreListStorage wait;
  wait.semaphore = semaphore;
  wait.payload_value = wait_value;
  id4::test::SemaphoreListStorage signal;
  signal.semaphore = semaphore;
  signal.payload_value = signal_value;

  id4_pipeline_stage_issue_options_t issue_options;
  std::memset(&issue_options, 0, sizeof(issue_options));
  issue_options.structure_size = sizeof(issue_options);
  issue_options.region_submission_window = 1;
  issue_options.boundary_binding_count = boundary_bindings.count;
  issue_options.boundary_bindings = boundary_bindings.bindings;
  issue_options.wait_semaphore_list = wait.list();
  issue_options.signal_semaphore_list = signal.list();
  issue_options.diagnostics_sink = &context->diagnostics_sink;
  return id4_pipeline_stage_issue(context->stage.get(), bundle, &issue_options);
}

static iree_status_t WaitForSemaphore(iree_hal_semaphore_t* semaphore,
                                      uint64_t payload_value) {
  return iree_hal_semaphore_wait(semaphore, payload_value,
                                 iree_infinite_timeout(),
                                 IREE_ASYNC_WAIT_FLAG_NONE);
}

static const iree_benchmark_def_t* RegisterVaeBenchmark(
    iree_string_view_t name, iree_benchmark_fn_t run,
    iree_benchmark_unit_t time_unit) {
  iree_benchmark_def_t* benchmark = iree_make_function_benchmark(run);
  benchmark->flags = IREE_BENCHMARK_FLAG_USE_REAL_TIME;
  benchmark->time_unit = time_unit;
  return iree_benchmark_register(name, benchmark);
}

#define ID4_VAE_BENCHMARK_REGISTER(name, time_unit)                 \
  static const iree_benchmark_def_t* name##_registration            \
      IREE_ATTRIBUTE_UNUSED =                                       \
          RegisterVaeBenchmark(iree_make_cstring_view(#name), name, \
                               IREE_BENCHMARK_UNIT_##time_unit)

IREE_BENCHMARK_FN(BM_VaeStagePlanFlux2Decode) {
  VaeBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedVaeStageContext(
      &context, kFlux2FullImageShape.activation_format));
  IREE_RETURN_IF_ERROR(
      WritePlanJsonIfRequested(&context, kFlux2FullImageShape));

  uint64_t iteration_count = 0;
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    id4_pipeline_plan_t* plan = nullptr;
    IREE_RETURN_IF_ERROR(CreateVaePlan(&context, kFlux2FullImageShape, &plan));
    iree_optimization_barrier(plan);
    id4_pipeline_plan_release(plan);
    ++iteration_count;
  }
  iree_benchmark_set_items_processed(benchmark_state,
                                     static_cast<int64_t>(iteration_count));
  return iree_ok_status();
}
ID4_VAE_BENCHMARK_REGISTER(BM_VaeStagePlanFlux2Decode, MICROSECOND);

IREE_BENCHMARK_FN(BM_VaeStagePlanFlux2TiledDecode) {
  VaeBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedVaeStageContext(
      &context, kFlux2TiledFullImageShape.activation_format));
  IREE_RETURN_IF_ERROR(
      WritePlanJsonIfRequested(&context, kFlux2TiledFullImageShape));

  uint64_t iteration_count = 0;
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    id4_pipeline_plan_t* plan = nullptr;
    IREE_RETURN_IF_ERROR(
        CreateVaePlan(&context, kFlux2TiledFullImageShape, &plan));
    iree_optimization_barrier(plan);
    id4_pipeline_plan_release(plan);
    ++iteration_count;
  }
  iree_benchmark_set_items_processed(benchmark_state,
                                     static_cast<int64_t>(iteration_count));
  return iree_ok_status();
}
ID4_VAE_BENCHMARK_REGISTER(BM_VaeStagePlanFlux2TiledDecode, MICROSECOND);

IREE_BENCHMARK_FN(BM_VaeStagePlanFlux2Bf16Decode) {
  VaeBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedVaeStageContext(
      &context, kFlux2Bf16FullImageShape.activation_format));
  IREE_RETURN_IF_ERROR(
      WritePlanJsonIfRequested(&context, kFlux2Bf16FullImageShape));

  uint64_t iteration_count = 0;
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    id4_pipeline_plan_t* plan = nullptr;
    IREE_RETURN_IF_ERROR(
        CreateVaePlan(&context, kFlux2Bf16FullImageShape, &plan));
    iree_optimization_barrier(plan);
    id4_pipeline_plan_release(plan);
    ++iteration_count;
  }
  iree_benchmark_set_items_processed(benchmark_state,
                                     static_cast<int64_t>(iteration_count));
  return iree_ok_status();
}
ID4_VAE_BENCHMARK_REGISTER(BM_VaeStagePlanFlux2Bf16Decode, MICROSECOND);

static iree_status_t RunVaeStagePrepareCachedKernels(
    iree_benchmark_state_t* benchmark_state, VaeBenchmarkShape shape) {
  VaeBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedVaeBenchmarkContext(&context, shape));

  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_RETURN_IF_ERROR(CreateVaePlan(&context, shape, plan.out()));
  IREE_RETURN_IF_ERROR(WritePlanJsonIfRequested(shape, plan.get()));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      prepare_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context.live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));

  uint64_t prepare_value = 1;
  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      warm_bundle;
  IREE_RETURN_IF_ERROR(PrepareVaeBundle(&context, plan.get(),
                                        prepare_semaphore.get(), prepare_value,
                                        warm_bundle.out()));
  IREE_RETURN_IF_ERROR(
      WaitForSemaphore(prepare_semaphore.get(), prepare_value));
  warm_bundle.reset();

  uint64_t iteration_count = 0;
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    ++prepare_value;
    id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
        bundle;
    IREE_RETURN_IF_ERROR(PrepareVaeBundle(&context, plan.get(),
                                          prepare_semaphore.get(),
                                          prepare_value, bundle.out()));
    IREE_RETURN_IF_ERROR(
        WaitForSemaphore(prepare_semaphore.get(), prepare_value));
    iree_optimization_barrier(bundle.get());
    ++iteration_count;
  }
  iree_benchmark_set_items_processed(benchmark_state,
                                     static_cast<int64_t>(iteration_count));
  return iree_ok_status();
}

IREE_BENCHMARK_FN(BM_VaeStagePrepareCachedKernels) {
  return RunVaeStagePrepareCachedKernels(benchmark_state, kFlux2FullImageShape);
}
ID4_VAE_BENCHMARK_REGISTER(BM_VaeStagePrepareCachedKernels, MICROSECOND);

IREE_BENCHMARK_FN(BM_VaeStagePrepareBf16CachedKernels) {
  return RunVaeStagePrepareCachedKernels(benchmark_state,
                                         kFlux2Bf16FullImageShape);
}
ID4_VAE_BENCHMARK_REGISTER(BM_VaeStagePrepareBf16CachedKernels, MICROSECOND);

typedef iree_status_t (*VaeQueueInputsFn)(
    VaeBenchmarkContext* context, const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& boundary_bindings,
    iree_hal_semaphore_t* semaphore, uint64_t* inout_payload_value);

static iree_status_t QueueSyntheticVaeInput(
    VaeBenchmarkContext* context, const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& boundary_bindings,
    iree_hal_semaphore_t* semaphore, uint64_t* inout_payload_value) {
  const float latent_value = 0.0f;
  return id4::test::QueueFillBoundaryTensors(
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan,
      boundary_bindings, ID4_PIPELINE_BOUNDARY_TENSOR_FLAG_INITIALIZED,
      &latent_value, sizeof(latent_value), semaphore, inout_payload_value);
}

static iree_status_t QueueFixtureVaeInput(
    VaeBenchmarkContext* context, const id4_pipeline_plan_t* plan,
    const id4::test::BufferBindingSet& boundary_bindings,
    iree_hal_semaphore_t* semaphore, uint64_t* inout_payload_value) {
  return id4::test::QueueUpdateBoundaryTensorFromFixture(
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan,
      boundary_bindings, IREE_SV("media.latent.input"),
      context->fixture_tensors, IREE_SV("latent"), semaphore,
      inout_payload_value);
}

static uint64_t CeilMiB(iree_device_size_t byte_length) {
  static constexpr iree_device_size_t kMiB = 1024ull * 1024ull;
  return (uint64_t)((byte_length + kMiB - 1) / kMiB);
}

static void SetVaeBenchmarkLabel(
    iree_benchmark_state_t* benchmark_state, VaeBenchmarkShape shape,
    iree_string_view_t input_kind,
    const id4_pipeline_plan_statistics_t& statistics) {
  const id4_vae_model_config_t* model = id4_vae_program_flux2_model_config();
  const uint64_t decoded_width =
      static_cast<uint64_t>(shape.latent_width) * model->scale_x;
  const uint64_t decoded_height =
      static_cast<uint64_t>(shape.latent_height) * model->scale_y;
  const iree_string_view_t activation_format =
      shape.activation_format == ID4_VAE_ACTIVATION_FORMAT_BF16_CONV_INPUT
          ? IREE_SV("bf16_conv_input")
          : IREE_SV("f32_canonical");
  iree_string_builder_t label_builder;
  iree_string_builder_initialize(iree_allocator_system(), &label_builder);
  IREE_CHECK_OK(iree_string_builder_append_format(
      &label_builder,
      "input=%.*s latent=%" PRIu32 "x%" PRIu32 "x%" PRIu32 " image=%" PRIu64
      "x%" PRIu64
      " activation=%.*s"
      " param_total=%" PRIu64 "MiB param_source=%" PRIu64
      "MiB local_hw=%" PRIu64 "MiB boundary=%" PRIu64 "MiB kernels=%" PRIhsz
      " dispatches=%" PRIhsz " regions=%" PRIhsz,
      static_cast<int>(input_kind.size), input_kind.data, shape.latent_width,
      shape.latent_height, shape.latent_batch_count, decoded_width,
      decoded_height, static_cast<int>(activation_format.size),
      activation_format.data, CeilMiB(statistics.parameter_slab_byte_length),
      CeilMiB(statistics.parameter_source_byte_length),
      CeilMiB(statistics.memory_slab_high_water_mark),
      CeilMiB(statistics.boundary_tensor_byte_length), statistics.kernel_count,
      statistics.dispatch_count, statistics.region_count));
  iree_benchmark_set_label(benchmark_state,
                           iree_string_builder_buffer(&label_builder));
  iree_string_builder_deinitialize(&label_builder);
}

static iree_status_t RunVaeStageIssueEndToEnd(
    iree_benchmark_state_t* benchmark_state, VaeBenchmarkContext* context,
    VaeBenchmarkShape shape, iree_string_view_t input_kind,
    VaeQueueInputsFn queue_inputs) {
  id4::test::OwningRef<id4_pipeline_plan_t, id4_pipeline_plan_release> plan;
  IREE_RETURN_IF_ERROR(CreateVaePlan(context, shape, plan.out()));
  IREE_RETURN_IF_ERROR(WritePlanJsonIfRequested(shape, plan.get()));
  const id4_pipeline_plan_statistics_t statistics =
      id4_pipeline_plan_statistics(plan.get());

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      prepare_semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, prepare_semaphore.out()));

  id4::test::OwningRef<id4_pipeline_bundle_t, id4_pipeline_bundle_release>
      bundle;
  IREE_RETURN_IF_ERROR(PrepareVaeBundle(
      context, plan.get(), prepare_semaphore.get(), 1, bundle.out()));
  IREE_RETURN_IF_ERROR(WaitForSemaphore(prepare_semaphore.get(), 1));

  id4::test::BufferBindingSet boundary_bindings;
  IREE_RETURN_IF_ERROR(id4::test::AllocateBoundaryBindings(
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, plan.get(),
      &boundary_bindings));

  id4::test::OwningRef<iree_hal_semaphore_t, iree_hal_semaphore_release>
      semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      context->live.device.get(), IREE_HAL_QUEUE_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, semaphore.out()));

  uint64_t payload_value = 0;
  IREE_RETURN_IF_ERROR(queue_inputs(context, plan.get(), boundary_bindings,
                                    semaphore.get(), &payload_value));
  IREE_RETURN_IF_ERROR(WaitForSemaphore(semaphore.get(), payload_value));

  const uint64_t warm_signal_value = payload_value + 1;
  IREE_RETURN_IF_ERROR(IssueVaeBundle(context, bundle.get(), boundary_bindings,
                                      semaphore.get(), payload_value,
                                      warm_signal_value));
  IREE_RETURN_IF_ERROR(WaitForSemaphore(semaphore.get(), warm_signal_value));
  payload_value = warm_signal_value;
  context->diagnostics = {};

  uint64_t iteration_count = 0;
  iree_hal_profiling_from_flags_t* profiling = nullptr;
  iree_status_t status = iree_hal_begin_device_group_profiling_from_flags(
      context->live.device_group.get(), iree_allocator_system(), &profiling);
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    const uint64_t wait_value = payload_value;
    const uint64_t signal_value = payload_value + 1;
    status = IssueVaeBundle(context, bundle.get(), boundary_bindings,
                            semaphore.get(), wait_value, signal_value);
    if (iree_status_is_ok(status)) {
      status = WaitForSemaphore(semaphore.get(), signal_value);
    }
    if (iree_status_is_ok(status)) {
      payload_value = signal_value;
      ++iteration_count;
    }
  }
  status =
      iree_status_join(status, iree_hal_end_profiling_from_flags(profiling));
  IREE_RETURN_IF_ERROR(status);
  iree_benchmark_set_items_processed(
      benchmark_state,
      static_cast<int64_t>(iteration_count * shape.decoded_element_count));
  SetVaeBenchmarkLabel(benchmark_state, shape, input_kind, statistics);
  return iree_ok_status();
}

static iree_status_t RunSyntheticVaeStageIssueEndToEnd(
    iree_benchmark_state_t* benchmark_state, VaeBenchmarkShape shape) {
  VaeBenchmarkContext context;
  IREE_RETURN_IF_ERROR(CreateLoadedVaeBenchmarkContext(&context, shape));
  return RunVaeStageIssueEndToEnd(benchmark_state, &context, shape,
                                  IREE_SV("synthetic_zero"),
                                  QueueSyntheticVaeInput);
}

IREE_BENCHMARK_FN(BM_VaeStageIssueEndToEnd) {
  return RunSyntheticVaeStageIssueEndToEnd(benchmark_state,
                                           kFlux2FullImageShape);
}
ID4_VAE_BENCHMARK_REGISTER(BM_VaeStageIssueEndToEnd, MILLISECOND);

IREE_BENCHMARK_FN(BM_VaeStageIssueTiledEndToEnd) {
  return RunSyntheticVaeStageIssueEndToEnd(benchmark_state,
                                           kFlux2TiledFullImageShape);
}
ID4_VAE_BENCHMARK_REGISTER(BM_VaeStageIssueTiledEndToEnd, MILLISECOND);

IREE_BENCHMARK_FN(BM_VaeStageIssueTileLocalEndToEnd) {
  return RunSyntheticVaeStageIssueEndToEnd(benchmark_state,
                                           kFlux2TileLocalShape);
}
ID4_VAE_BENCHMARK_REGISTER(BM_VaeStageIssueTileLocalEndToEnd, MILLISECOND);

IREE_BENCHMARK_FN(BM_VaeStageIssueBf16EndToEnd) {
  return RunSyntheticVaeStageIssueEndToEnd(benchmark_state,
                                           kFlux2Bf16FullImageShape);
}
ID4_VAE_BENCHMARK_REGISTER(BM_VaeStageIssueBf16EndToEnd, MILLISECOND);

IREE_BENCHMARK_FN(BM_VaeStageIssueBf16SmallImageEndToEnd) {
  return RunSyntheticVaeStageIssueEndToEnd(benchmark_state,
                                           kFlux2Bf16SmallImageShape);
}
ID4_VAE_BENCHMARK_REGISTER(BM_VaeStageIssueBf16SmallImageEndToEnd, MILLISECOND);

IREE_BENCHMARK_FN(BM_VaeStageIssueBf16FixtureEndToEnd) {
  VaeBenchmarkContext context;
  VaeBenchmarkShape shape;
  IREE_RETURN_IF_ERROR(LoadFixtureVaeBenchmarkShape(&context, &shape));
  IREE_RETURN_IF_ERROR(CreateLoadedVaeBenchmarkContext(&context, shape));
  return RunVaeStageIssueEndToEnd(benchmark_state, &context, shape,
                                  IREE_SV("fixture"), QueueFixtureVaeInput);
}
ID4_VAE_BENCHMARK_REGISTER(BM_VaeStageIssueBf16FixtureEndToEnd, MILLISECOND);

IREE_BENCHMARK_FN(BM_VaeStageIssueBf16TiledEndToEnd) {
  return RunSyntheticVaeStageIssueEndToEnd(benchmark_state,
                                           kFlux2Bf16TiledFullImageShape);
}
ID4_VAE_BENCHMARK_REGISTER(BM_VaeStageIssueBf16TiledEndToEnd, MILLISECOND);

IREE_BENCHMARK_FN(BM_VaeStageIssueBf16TileLocalEndToEnd) {
  return RunSyntheticVaeStageIssueEndToEnd(benchmark_state,
                                           kFlux2Bf16TileLocalShape);
}
ID4_VAE_BENCHMARK_REGISTER(BM_VaeStageIssueBf16TileLocalEndToEnd, MILLISECOND);

}  // namespace
