// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"

amdf_status_t AqlDispatchTest::MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                                bool* out_matches) {
  amdf_gpu_endpoint_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO;
  info.structure_size = sizeof(info);
  const amdf_status_t status = gpu_api_->endpoint_query_info(endpoint, &info);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (info.gfx_ip.major != 9 && Pm4CommandProfile::Find(info) == nullptr) {
    *out_matches = false;
    return AMDF_STATUS_OK;
  }
  return AqlQueueTest::MatchGpuEndpoint(endpoint, out_matches);
}

void AqlDispatchTest::CreateFixedScratchQueue(
    uint32_t maximum_private_segment_byte_length, GpuUserQueue** out_queue) {
  const uint64_t wave_byte_length =
      (uint64_t{maximum_private_segment_byte_length} * 64 + 1023) &
      ~UINT64_C(1023);
  amdf_gpu_endpoint_info_t endpoint_info = {};
  endpoint_info.type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO;
  endpoint_info.structure_size = sizeof(endpoint_info);
  ASSERT_EQ(gpu_api_->endpoint_query_info(endpoint_, &endpoint_info),
            AMDF_STATUS_OK);
  const auto& compute = endpoint_info.compute;
  const auto& topology = endpoint_info.topology;
  ASSERT_TRUE(compute.wavefront_size == 32 || compute.wavefront_size == 64);
  ASSERT_GT(compute.compute_unit_count, 0u);
  ASSERT_GT(compute.maximum_scratch_wave_count_per_compute_unit, 0u);
  ASSERT_GT(topology.xcc_count, 0u);
  ASSERT_GT(topology.shader_engine_count_per_xcc, 0u);
  ASSERT_EQ(compute.compute_unit_count % topology.xcc_count, 0u);
  const uint64_t shader_engine_count =
      uint64_t{topology.xcc_count} * topology.shader_engine_count_per_xcc;
  const uint64_t wave_count =
      ((uint64_t{compute.compute_unit_count} + shader_engine_count - 1) /
       shader_engine_count) *
      shader_engine_count * compute.maximum_scratch_wave_count_per_compute_unit;
  ASSERT_LE(wave_count, UINT32_MAX);
  const uint64_t waves_per_xcc = wave_count / topology.xcc_count;
  ASSERT_EQ(waves_per_xcc % topology.shader_engine_count_per_xcc, 0u);
  const uint64_t bytes_per_xcc = waves_per_xcc * wave_byte_length;
  ASSERT_LE(bytes_per_xcc, UINT32_MAX);
  // Fixed AQL scratch covers every physical slot; a small grid does not
  // establish which compute units, XCCs or physical slots execute its waves.
  const uint64_t scratch_byte_length = wave_count * wave_byte_length;
  RecordProperty("aql_scratch_compute_unit_count", compute.compute_unit_count);
  RecordProperty("aql_scratch_slots_per_compute_unit",
                 compute.maximum_scratch_wave_count_per_compute_unit);
  RecordProperty("aql_scratch_shader_engines_per_xcc",
                 topology.shader_engine_count_per_xcc);
  RecordProperty("aql_scratch_physical_wave_capacity",
                 std::to_string(wave_count));
  RecordProperty("aql_scratch_waves_per_xcc", std::to_string(waves_per_xcc));
  RecordProperty("aql_scratch_bytes_per_wave",
                 std::to_string(wave_byte_length));
  RecordProperty("aql_scratch_bytes_per_xcc", std::to_string(bytes_per_xcc));
  RecordProperty("aql_scratch_byte_length",
                 std::to_string(scratch_byte_length));

  GpuMemory* scratch_memory = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   scratch_byte_length, &scratch_memory));
  ASSERT_EQ(scratch_memory->device_address % 4096, 0u);
  ASSERT_LT(scratch_memory->device_address, UINT64_C(1) << 48);
  ASSERT_LE(scratch_byte_length,
            (UINT64_C(1) << 48) - scratch_memory->device_address);

  // Scratch stays untouched by the CPU. Each workitem initializes the private
  // words it reads; the queue borrows the backing until destruction succeeds.
  amdf_gpu_queue_scratch_t scratch = {};
  scratch.memory = scratch_memory->memory;
  scratch.byte_length = scratch_byte_length;
  scratch.maximum_private_segment_byte_length =
      maximum_private_segment_byte_length;
  scratch.maximum_wave_count = static_cast<uint32_t>(wave_count);
  ASSERT_NO_FATAL_FAILURE(
      CreateQueue(out_queue, AMDF_QUEUE_PRODUCER_MODE_SINGLE, scratch));
}

void AqlDispatchTest::PublishKernel(GpuUserQueue& queue,
                                    const kernels::Kernel& kernel,
                                    const char* property_prefix,
                                    uint64_t* next_packet_index,
                                    uint64_t* out_descriptor_address) {
  auto& executable = executables_.emplace_back();
  ASSERT_NO_FATAL_FAILURE(
      executable.Initialize(api_, system_scope_, device_, gpu_endpoint_info_,
                            kernel, property_prefix, queue, next_packet_index));
  *out_descriptor_address = executable.descriptor_address();
}

void AqlDispatchTest::TearDown() {
  // A failed native queue removal leaves every executable backing retained.
  ASSERT_NO_FATAL_FAILURE(GpuCommandTest::TearDown());
  for (auto& executable : executables_) {
    ASSERT_TRUE(executable.Release(api_));
  }
}
