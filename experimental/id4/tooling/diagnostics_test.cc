// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/tooling/diagnostics.h"

#include <fstream>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace {

static std::string ReadTextFile(const std::string& path) {
  std::ifstream file(path);
  return std::string(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());
}

static void ExpectFinds(const std::string& value, const char* needle) {
  EXPECT_NE(value.find(needle), std::string::npos) << "expected: " << needle;
}

TEST(DiagnosticsTest, WritesJsonLinesEvents) {
  iree::testing::TempFilePath directory("id4_tooling_diagnostics");
  id4_tooling_diagnostics_file_sink_t file_sink;
  id4_pipeline_diagnostics_sink_t diagnostics_sink;
  IREE_ASSERT_OK(id4_tooling_diagnostics_file_sink_initialize(
      directory.path_view(), iree_allocator_system(), &file_sink,
      &diagnostics_sink));

  const id4_pipeline_parameter_slab_diagnostic_t parameter_slab = {
      /*.slab_index=*/3,
      /*.request_index=*/7,
      /*.scope=*/IREE_SV("dit"),
      /*.parameter_key=*/IREE_SV("layer.\"q\".weight"),
      /*.parameter_offset=*/11,
      /*.buffer_offset=*/13,
      /*.length=*/17,
      /*.placement_id=*/1,
      /*.device_index=*/0,
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
      /*.memory_type=*/IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      /*.memory_access=*/IREE_HAL_MEMORY_ACCESS_ALL,
      /*.buffer_usage=*/IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
      /*.slab_byte_length=*/1024,
      /*.slab_alignment=*/16,
      /*.request_count=*/9,
  };
  id4_pipeline_diagnostic_event_t parameter_event = {
      /*.kind=*/ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_PARAMETER_SLAB,
      /*.stage_name=*/IREE_SV("dit\nstage"),
      /*.key=*/IREE_SV("parameter.slab"),
      /*.message=*/IREE_SV("copied\tweight"),
      /*.parameter_slab=*/&parameter_slab,
  };
  IREE_ASSERT_OK(
      id4_pipeline_diagnostics_emit(&diagnostics_sink, &parameter_event));

  const id4_pipeline_parameter_load_diagnostic_t parameter_load = {
      /*.slab_index=*/3,
      /*.load_group_index=*/4,
      /*.load_group_kind=*/IREE_SV("encode"),
      /*.first_consumer_region_id=*/9,
      /*.submit_region_id=*/10,
      /*.load_step_offset=*/5,
      /*.load_step_count=*/2,
      /*.first_load_step_name=*/IREE_SV("layer0.q_proj"),
      /*.staging_slot_count=*/2,
      /*.staging_slot_byte_length=*/4096,
      /*.staging_total_byte_length=*/8192,
      /*.staging_chunk_count=*/1,
      /*.logical_source_count=*/4,
      /*.source_gather_batch_count=*/2,
      /*.source_byte_length=*/1024,
      /*.target_byte_length=*/2048,
      /*.encoder_dispatch_count=*/2,
  };
  id4_pipeline_diagnostic_event_t parameter_load_event = {
      /*.kind=*/ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_PARAMETER_SLAB,
      /*.stage_name=*/IREE_SV("dit.stage"),
      /*.key=*/IREE_SV("parameter.slab.encode_window"),
      /*.message=*/IREE_SV("encoded parameter window"),
      /*.parameter_slab=*/&parameter_slab,
      /*.parameter_load=*/&parameter_load,
  };
  IREE_ASSERT_OK(
      id4_pipeline_diagnostics_emit(&diagnostics_sink, &parameter_load_event));

  const id4_pipeline_kernel_diagnostic_t kernel = {
      /*.phase=*/IREE_SV("prepare"),
      /*.source_identifier=*/IREE_SV("qwen3_vl/rmsnorm"),
      /*.module_path=*/IREE_SV("qwen3_vl/rmsnorm"),
      /*.target_processor=*/IREE_SV("gfx1100"),
      /*.loom_artifact_format=*/IREE_SV("amdgpu.hsaco"),
      /*.hal_executable_format=*/IREE_SV("amdgpu-elf"),
      /*.config_binding_count=*/4,
      /*.artifact_byte_length=*/4096,
      /*.inferred_executable_byte_length=*/2048,
      /*.diagnostic_index=*/IREE_HOST_SIZE_MAX,
      /*.diagnostic_severity=*/-1,
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
      /*.caching_mode=*/IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA,
  };
  id4_pipeline_diagnostic_event_t kernel_event = {
      /*.kind=*/ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_KERNEL,
      /*.stage_name=*/IREE_SV("kernel.stage"),
      /*.key=*/IREE_SV("kernel.prepare"),
      /*.message=*/IREE_SV("prepared"),
      /*.parameter_slab=*/NULL,
      /*.parameter_load=*/NULL,
      /*.kernel=*/&kernel,
  };
  IREE_ASSERT_OK(
      id4_pipeline_diagnostics_emit(&diagnostics_sink, &kernel_event));

  const id4_pipeline_timing_diagnostic_t timing = {
      /*.start_time_ns=*/100,
      /*.end_time_ns=*/175,
      /*.duration_ns=*/75,
  };
  id4_pipeline_diagnostic_event_t timing_event = {
      /*.kind=*/ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_TIMING,
      /*.stage_name=*/IREE_SV("id4.cli"),
      /*.key=*/IREE_SV("cli.phase"),
      /*.message=*/IREE_SV("completed phase"),
      /*.parameter_slab=*/NULL,
      /*.parameter_load=*/NULL,
      /*.kernel=*/NULL,
      /*.timing=*/&timing,
  };
  IREE_ASSERT_OK(
      id4_pipeline_diagnostics_emit(&diagnostics_sink, &timing_event));
  IREE_ASSERT_OK(id4_tooling_diagnostics_file_sink_deinitialize(&file_sink));

  const std::string event_log =
      ReadTextFile(directory.path() + "/events.jsonl");
  ExpectFinds(event_log, "\"kind\":\"parameter_slab\"");
  ExpectFinds(event_log, "\"stage\":\"dit\\nstage\"");
  ExpectFinds(event_log, "\"message\":\"copied\\tweight\"");
  ExpectFinds(event_log, "\"parameter_key\":\"layer.\\\"q\\\".weight\"");
  ExpectFinds(event_log, "\"slab_index\":3");
  ExpectFinds(event_log, "\"key\":\"parameter.slab.encode_window\"");
  ExpectFinds(event_log, "\"parameter_load\"");
  ExpectFinds(event_log, "\"load_group_index\":4");
  ExpectFinds(event_log, "\"load_group_kind\":\"encode\"");
  ExpectFinds(event_log, "\"first_consumer_region_id\":9");
  ExpectFinds(event_log, "\"submit_region_id\":10");
  ExpectFinds(event_log, "\"staging_slot_count\":2");
  ExpectFinds(event_log, "\"source_gather_batch_count\":2");
  ExpectFinds(event_log, "\"encoder_dispatch_count\":2");
  ExpectFinds(event_log, "\"memory_type\":");
  ExpectFinds(event_log, "\"memory_access\":");
  ExpectFinds(event_log, "\"buffer_usage\":");
  ExpectFinds(event_log, "\"first_load_step_name\":\"layer0.q_proj\"");
  ExpectFinds(event_log, "\"kind\":\"kernel\"");
  ExpectFinds(event_log, "\"module_path\":\"qwen3_vl/rmsnorm\"");
  ExpectFinds(event_log, "\"artifact_byte_length\":4096");
  ExpectFinds(event_log, "\"kind\":\"timing\"");
  ExpectFinds(event_log, "\"stage\":\"id4.cli\"");
  ExpectFinds(event_log, "\"duration_ns\":75");
}

}  // namespace
