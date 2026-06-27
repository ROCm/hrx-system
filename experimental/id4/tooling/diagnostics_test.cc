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
  ExpectFinds(event_log, "\"kind\":\"kernel\"");
  ExpectFinds(event_log, "\"module_path\":\"qwen3_vl/rmsnorm\"");
  ExpectFinds(event_log, "\"artifact_byte_length\":4096");
  ExpectFinds(event_log, "\"kind\":\"timing\"");
  ExpectFinds(event_log, "\"stage\":\"id4.cli\"");
  ExpectFinds(event_log, "\"duration_ns\":75");
}

}  // namespace
