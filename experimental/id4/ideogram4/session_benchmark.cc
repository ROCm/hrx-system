// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "experimental/id4/ideogram4/session.h"
#include "experimental/id4/stages/test_util.h"
#include "iree/testing/gbenchmark_harness.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/testdata/streaming_testdata.h"

namespace {

static iree_string_view_t GetEmbeddedTokenizerJson() {
  const iree_file_toc_t* toc = iree_tokenizer_streaming_testdata_create();
  for (size_t i = 0; i < iree_tokenizer_streaming_testdata_size(); ++i) {
    if (std::strcmp(toc[i].name, "bpe_bytelevel_minimal.json") == 0) {
      return iree_make_string_view(toc[i].data, toc[i].size);
    }
  }
  return iree_string_view_empty();
}

static iree_tokenizer_t* LoadTokenizer() {
  iree_tokenizer_t* tokenizer = nullptr;
  IREE_CHECK_OK(iree_tokenizer_from_huggingface_json(
      GetEmbeddedTokenizerJson(), iree_allocator_system(), &tokenizer));
  return tokenizer;
}

static id4_ideogram4_generation_config_t MakeGenerationConfig() {
  id4_ideogram4_generation_config_t config;
  std::memset(&config, 0, sizeof(config));
  config.structure_size = sizeof(config);
  config.diffusion_latent_shape =
      id4_pipeline_program_make_shape_rank4(16, 16, 128, 1);
  config.denoise_step_count = 20;
  config.dit_activation_format =
      ID4_IDEOGRAM4_DIT_ACTIVATION_FORMAT_BF16_LINEAR_INPUT;
  config.vae_tiling.mode = ID4_VAE_TILING_MODE_DISABLED;
  return config;
}

static id4_ideogram4_session_t* CreateLoadedSession(
    iree_hal_device_group_t* device_group,
    iree_hal_executable_cache_t* executable_cache,
    id4_pipeline_kernel_cache_t* kernel_cache) {
  id4_pipeline_stage_services_t services;
  std::memset(&services, 0, sizeof(services));
  services.device_group = device_group;
  services.executable_cache = executable_cache;
  services.host_allocator = iree_allocator_system();

  id4_ideogram4_session_create_options_t create_options;
  std::memset(&create_options, 0, sizeof(create_options));
  create_options.structure_size = sizeof(create_options);
  create_options.services = services;
  create_options.kernel_cache = kernel_cache;
  create_options.vae_activation_format =
      ID4_VAE_ACTIVATION_FORMAT_BF16_CONV_INPUT;

  id4_ideogram4_session_t* session = nullptr;
  IREE_CHECK_OK(id4_ideogram4_session_create(
      &create_options, iree_allocator_system(), &session));

  id4::test::StageDiagnostics diagnostics = {};
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      id4::test::DiagnosticsSink(&diagnostics);
  id4_ideogram4_session_load_options_t load_options;
  std::memset(&load_options, 0, sizeof(load_options));
  load_options.structure_size = sizeof(load_options);
  load_options.diagnostics_sink = &diagnostics_sink;
  IREE_CHECK_OK(id4_ideogram4_session_load(session, &load_options));
  return session;
}

static id4_pipeline_kernel_cache_t* CreateKernelCache() {
  id4_pipeline_kernel_cache_create_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.target_processor =
      id4_pipeline_kernel_cache_default_target_processor();

  id4_pipeline_kernel_cache_t* kernel_cache = nullptr;
  IREE_CHECK_OK(id4_pipeline_kernel_cache_create(
      &options, iree_allocator_system(), &kernel_cache));
  return kernel_cache;
}

static iree_hal_executable_cache_t* CreateExecutableCache(
    iree_hal_device_group_t* device_group) {
  iree_hal_device_t* device =
      iree_hal_device_group_device_at(device_group, /*index=*/0);
  iree_hal_executable_cache_t* executable_cache = nullptr;
  IREE_CHECK_OK(iree_hal_executable_cache_create(
      device, IREE_SV("id4_session_benchmark"), &executable_cache));
  return executable_cache;
}

static void RunPlanGenerationBenchmark(benchmark::State& state,
                                       iree_string_view_t prompt_json) {
  iree_hal_device_group_t* device_group =
      id4::test::CreateLocalSyncDeviceGroup();
  iree_hal_executable_cache_t* executable_cache =
      CreateExecutableCache(device_group);
  id4_pipeline_kernel_cache_t* kernel_cache = CreateKernelCache();
  id4_ideogram4_session_t* session =
      CreateLoadedSession(device_group, executable_cache, kernel_cache);
  iree_tokenizer_t* tokenizer = LoadTokenizer();

  id4_ideogram4_request_t request;
  std::memset(&request, 0, sizeof(request));
  IREE_CHECK_OK(id4_ideogram4_request_parse_json(
      prompt_json, iree_allocator_system(), &request));

  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  id4_pipeline_diagnostics_sink_initialize_ignore(&diagnostics_sink);

  uint64_t iteration_count = 0;
  uint32_t token_count = 0;
  for (auto _ : state) {
    id4_ideogram4_generation_plan_options_t plan_options;
    std::memset(&plan_options, 0, sizeof(plan_options));
    plan_options.structure_size = sizeof(plan_options);
    plan_options.request = &request;
    plan_options.tokenizer = tokenizer;
    plan_options.tokenizer_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
    plan_options.generation = MakeGenerationConfig();
    plan_options.device_index = 0;
    plan_options.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    plan_options.diagnostics_sink = &diagnostics_sink;

    id4_ideogram4_generation_plan_t* plan = nullptr;
    IREE_CHECK_OK(
        id4_ideogram4_session_plan_generation(session, &plan_options, &plan));
    id4_ideogram4_generation_plan_summary_t summary;
    IREE_CHECK_OK(id4_ideogram4_generation_plan_summary(plan, &summary));
    token_count = summary.qwen_token_count;
    benchmark::DoNotOptimize(summary.qwen_token_count);
    id4_ideogram4_generation_plan_release(plan);
    ++iteration_count;
  }
  state.SetItemsProcessed(static_cast<int64_t>(iteration_count * token_count));

  id4_ideogram4_request_deinitialize(&request, iree_allocator_system());
  iree_tokenizer_free(tokenizer);
  id4_ideogram4_session_release(session);
  id4_pipeline_kernel_cache_release(kernel_cache);
  iree_hal_executable_cache_release(executable_cache);
  iree_hal_device_group_release(device_group);
}

static void BM_Ideogram4SessionPlanGenerationShort(benchmark::State& state) {
  RunPlanGenerationBenchmark(state, IREE_SV("{\"prompt\":\"a city\"}"));
}
BENCHMARK(BM_Ideogram4SessionPlanGenerationShort);

static void BM_Ideogram4SessionPlanGenerationMedium(benchmark::State& state) {
  RunPlanGenerationBenchmark(
      state,
      IREE_SV("{\"prompt\":\"three people walking through a reflective city "
              "street with umbrellas and neon signs\"}"));
}
BENCHMARK(BM_Ideogram4SessionPlanGenerationMedium);

static void BM_Ideogram4SessionPlanGenerationStructured(
    benchmark::State& state) {
  RunPlanGenerationBenchmark(
      state,
      IREE_SV("{\"high_level_description\":\"three people walking through a "
              "rainy reflective city street\","
              "\"style_description\":{\"medium\":\"cinematic photo\","
              "\"lighting\":\"neon signs and warm window reflections\"},"
              "\"negative_prompt\":\"blurred faces, malformed hands, text\"}"));
}
BENCHMARK(BM_Ideogram4SessionPlanGenerationStructured);

}  // namespace
