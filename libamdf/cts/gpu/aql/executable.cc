// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/aql/executable.h"

#include <cstring>
#include <string>

#include "libamdf/cts/gpu/aql/publication.h"
#include "libamdf/cts/gpu/gpu_device_fixture.h"

namespace aql {
namespace {

void CreateStorage(const amdf_api_t* api, amdf_memory_scope_t* scope,
                   amdf_device_t* device, amdf_memory_access_t access,
                   uint64_t byte_length, GpuMemory* memory) {
  const amdf_memory_device_access_t attachment = {
      device,
      {.access = access,
       .flags =
           AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
  amdf_memory_create_info_t create = {};
  create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
  create.structure_size = sizeof(create);
  create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  create.memory_profile_ordinal = FindGpuMemoryProfileOrdinal(
      api, scope, device,
      AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
      create.required_flags, attachment.requirements);
  ASSERT_NE(create.memory_profile_ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
  create.access_count = 1;
  create.accesses = &attachment;
  create.byte_length = byte_length;
  create.minimum_alignment = 4096;
  ASSERT_NO_FATAL_FAILURE(memory->Initialize(api, scope, create));
}

}  // namespace

uint64_t ExecutableByteLength(const amdf_gpu_endpoint_info_t& endpoint,
                              const kernels::Kernel& kernel) {
  const auto& image = kernel.executable;
  const auto* profile = Pm4CommandProfile::Find(endpoint);
  return profile ? profile->CodeByteLength(image.byte_length,
                                           kernel.entry_byte_offset,
                                           kernel.program.resource3)
                 : ((uint64_t{image.byte_length} + 63u) & ~UINT64_C(63)) + 192u;
}

void Executable::Initialize(const amdf_api_t* api,
                            amdf_memory_scope_t* system_scope,
                            amdf_device_t* device,
                            const amdf_gpu_endpoint_info_t& endpoint,
                            const kernels::Kernel& kernel,
                            const char* property_prefix, GpuUserQueue& queue,
                            uint64_t* next_packet_index) {
  const auto& image = kernel.executable;
  const uint64_t code_byte_length = ExecutableByteLength(endpoint, kernel);
  ASSERT_NO_FATAL_FAILURE(
      CreateStorage(api, system_scope, device,
                    AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
                    (code_byte_length + 4095u) & ~UINT64_C(4095), &code_));
  ASSERT_NO_FATAL_FAILURE(CreateStorage(
      api, system_scope, device,
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE, 4096, &commands_));
  ASSERT_NO_FATAL_FAILURE(CreateStorage(
      api, system_scope, device,
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion_));
  ASSERT_EQ(code_.device_address % 256, 0u);
  ASSERT_EQ(commands_.device_address % 4, 0u);
  ASSERT_LT(commands_.device_address, UINT64_C(1) << 48);
  ASSERT_LE(commands_.info.byte_length,
            (UINT64_C(1) << 48) - commands_.device_address);
  ASSERT_NE(completion_.device_address, 0u);
  ASSERT_EQ(completion_.device_address % alignof(Signal), 0u);
  std::memset(code_.host.pointer, 0, code_.info.byte_length);
  std::memcpy(code_.host.pointer, image.words, image.byte_length);
  const auto publication =
      CodeCacheInvalidate(endpoint, code_.device_address, image.byte_length);
  std::memset(commands_.host.pointer, 0, commands_.info.byte_length);
  std::memcpy(commands_.host.pointer, publication.data(),
              publication.size() * sizeof(uint32_t));
  std::memset(completion_.host.pointer, 0, completion_.info.byte_length);
  auto& signal = *static_cast<Signal*>(completion_.host.pointer);
  signal.kind = 1;
  signal.value = 1;
  const std::string prefix = property_prefix;
  ::testing::Test::RecordProperty(prefix + "_image_sha256", image.sha256);
  ::testing::Test::RecordProperty(prefix + "_image_byte_length",
                                  image.byte_length);
  ::testing::Test::RecordProperty(prefix + "_descriptor_byte_offset",
                                  image.descriptor_byte_offset);

  const uint64_t index = (*next_packet_index)++;
  GpuStoreRelease(queue.host.write_index_address, *next_packet_index);
  Publish(queue, index,
          IndirectBuffer(HeaderBarrier::kDisabled, commands_.device_address,
                         publication.size(), completion_.device_address,
                         {FenceScope::kNone, FenceScope::kNone}));
  // Execution completion protects publication storage; consumption is a
  // separate setup join. Workload payload dependencies belong to the case.
  GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signal.value), 0);
  ASSERT_NO_FATAL_FAILURE(queue.WaitConsumed(api, *next_packet_index));
  descriptor_address_ = code_.device_address + image.descriptor_byte_offset;
}

bool Executable::Release(const amdf_api_t* api) {
  return commands_.Release(api) && completion_.Release(api) &&
         code_.Release(api);
}

}  // namespace aql
