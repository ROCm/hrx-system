// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "memory_benchmark.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "amdf/amdf.h"
#include "benchmark/benchmark.h"
#include "util/device_cache.h"
#include "util/provider.h"

namespace {

// A native failure terminates the experiment without retries or later samples.
// No execution is submitted: process exit reclaims remaining native memory.
void CheckStatus(amdf_status_t status, const char* operation) {
  if (amdf_status_is_ok(status)) return;
  std::fprintf(stderr, "%s failed: domain=%u code=%u\n", operation,
               amdf_status_domain(status), amdf_status_code(status));
  std::exit(EXIT_FAILURE);
}

void Check(bool condition, const char* message) {
  if (condition) return;
  std::fprintf(stderr, "%s\n", message);
  std::exit(EXIT_FAILURE);
}

// All benchmark repetitions borrow one shared ordinary device. Scope/profile
// discovery and activation are outside every measured region.
class MemoryBenchmark {
 public:
  void Initialize(amdf_engine_kind_t engine_kind) {
    address_kind_ = engine_kind == AMDF_ENGINE_KIND_GPU
                        ? AMDF_MEMORY_ADDRESS_GPU
                        : AMDF_MEMORY_ADDRESS_XDNA_DMA;
    CheckStatus(amdf_cts_provider_query_api()(AMDF_ABI_VERSION_1,
                                              AMDF_ABI_VERSION_LATEST, &api_),
                "query_api");
    amdf_instance_t* instance = nullptr;
    CheckStatus(GetCtsDeviceCache().GetInstance(&instance), "instance_create");
    uint32_t count = 0;
    CheckStatus(api_->endpoint_enumerate(instance, 0, nullptr, &count),
                "endpoint_count");
    std::vector<amdf_endpoint_summary_t> summaries(count);
    CheckStatus(
        api_->endpoint_enumerate(instance, count, summaries.data(), &count),
        "endpoint_enumerate");
    amdf_endpoint_t* endpoint = nullptr;
    for (const auto& summary : summaries) {
      if (summary.engine_kind != engine_kind) continue;
      CheckStatus(GetCtsDeviceCache().OpenEndpoint(summary.id, &endpoint),
                  "endpoint_open");
      break;
    }
    if (!endpoint) return;
    const amdf_status_t status =
        engine_kind == AMDF_ENGINE_KIND_GPU
            ? GetCtsDeviceCache().GetGpuDevice(endpoint, &access_.device)
            : GetCtsDeviceCache().GetXdnaDevice(endpoint, &access_.device);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) return;
    CheckStatus(status, "device_create");
    access_.requirements.access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    access_.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    access_.requirements.address_kinds = uint64_t{1} << address_kind_;

    count = 0;
    Check(
        amdf_status_code(api_->instance_enumerate_memory_scopes(
            instance, 0, nullptr, &count)) == AMDF_STATUS_CODE_BUFFER_TOO_SMALL,
        "no system memory scopes");
    std::vector<amdf_memory_scope_t*> scopes(count);
    CheckStatus(api_->instance_enumerate_memory_scopes(instance, count,
                                                       scopes.data(), &count),
                "memory_scopes");
    for (auto* scope : scopes) {
      amdf_memory_scope_info_t info = {};
      info.type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO;
      info.structure_size = sizeof(info);
      CheckStatus(api_->memory_scope_query_info(scope, &info), "scope_info");
      if (info.kind != AMDF_MEMORY_SCOPE_KIND_SYSTEM) continue;
      for (uint32_t ordinal = 0; ordinal < info.memory_profile_count;
           ++ordinal) {
        amdf_memory_profile_t profile = {};
        profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
        profile.structure_size = sizeof(profile);
        amdf_memory_access_capabilities_t capabilities = {};
        capabilities.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
        capabilities.structure_size = sizeof(capabilities);
        const amdf_status_t profile_status =
            api_->memory_scope_query_device_profile(scope, ordinal, 1, &access_,
                                                    &profile, &capabilities);
        if (profile_status ==
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
          continue;
        }
        CheckStatus(profile_status, "memory_profile");
        constexpr auto roles =
            AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
        if ((profile.roles & roles) != roles ||
            !(profile.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE)) {
          continue;
        }
        scope_ = scope;
        create_info_.memory_profile_ordinal = ordinal;
        break;
      }
      if (scope_) break;
    }
    if (!scope_) return;
    create_info_.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    create_info_.structure_size = sizeof(create_info_);
    create_info_.access_count = 1;
    create_info_.accesses = &access_;
    create_info_.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  }

  // Measures acquisition, mapping/address lookup, and complete release without
  // touching the mapping. Native allocation may itself initialize the backing.
  void Allocation(benchmark::State& state) {
    if (!CheckAvailable(state)) return;
    const size_t byte_length = static_cast<size_t>(state.range(0));
    CheckMemory(byte_length);
    for (auto iteration : state) {
      (void)iteration;
      Acquire(byte_length);
      Release();
    }
    state.SetBytesProcessed(state.iterations() * byte_length);
  }

  // Measures the allocation lifecycle including the first full host write,
  // without an explicit publication operation.
  void Initialization(benchmark::State& state) {
    if (!CheckAvailable(state)) return;
    const size_t byte_length = static_cast<size_t>(state.range(0));
    CheckMemory(byte_length);
    uint8_t value = 0;
    for (auto iteration : state) {
      (void)iteration;
      Acquire(byte_length);
      InitializeMemory(byte_length, value++);
      Release();
    }
    state.SetBytesProcessed(state.iterations() * byte_length);
  }

  // Measures acquisition, mapping/address lookup, initialization, publication,
  // and complete release, with no hidden device creation or live allocation
  // pool.
  void Lifecycle(benchmark::State& state) {
    if (!CheckAvailable(state)) return;
    const size_t byte_length = static_cast<size_t>(state.range(0));
    CheckMemory(byte_length);
    uint8_t value = 0;
    for (auto iteration : state) {
      (void)iteration;
      Acquire(byte_length);
      Publish(byte_length, value++);
      Release();
    }
    state.SetBytesProcessed(state.iterations() * byte_length);
  }

  // Measures only CPU initialization and explicit publication of retained
  // resident backing. No command submission or device completion is implied.
  void Publication(benchmark::State& state) {
    if (!CheckAvailable(state)) return;
    const size_t byte_length = static_cast<size_t>(state.range(0));
    Acquire(byte_length);
    Publish(byte_length, 0xA5);
    Verify(byte_length, 0xA5);
    uint8_t value = 0;
    for (auto iteration : state) {
      (void)iteration;
      Publish(byte_length, value++);
    }
    if (state.iterations())
      Verify(byte_length, static_cast<uint8_t>(value - 1));
    Release();
    state.SetBytesProcessed(state.iterations() * byte_length);
  }

 private:
  bool CheckAvailable(benchmark::State& state) const {
    if (scope_) return true;
    state.SkipWithMessage("native host-visible allocation unavailable");
    return false;
  }

  void CheckMemory(size_t byte_length) {
    Acquire(byte_length);
    InitializeMemory(byte_length, 0x5A);
    Verify(byte_length, 0x5A);
    Publish(byte_length, 0xA5);
    Verify(byte_length, 0xA5);
    Release();
  }

  void Acquire(size_t byte_length) {
    create_info_.byte_length = byte_length;
    CheckStatus(api_->memory_create(scope_, &create_info_, &memory_),
                "memory_create");
    amdf_memory_map_info_t map = {};
    map.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
    map.structure_size = sizeof(map);
    map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    map.byte_length = byte_length;
    CheckStatus(api_->memory_map(memory_, &map, &mapping_), "memory_map");
    amdf_host_mapping_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
    info.structure_size = sizeof(info);
    CheckStatus(api_->host_mapping_query_info(mapping_, &info), "mapping_info");
    pointer_ = static_cast<uint8_t*>(info.pointer);
    uint64_t address = 0;
    CheckStatus(api_->memory_query_address(memory_, 0, address_kind_, &address),
                "memory_address");
    benchmark::DoNotOptimize(address);
  }

  void InitializeMemory(size_t byte_length, uint8_t value) {
    std::memset(pointer_, value, byte_length);
    benchmark::ClobberMemory();
  }

  void Publish(size_t byte_length, uint8_t value) {
    InitializeMemory(byte_length, value);
    CheckStatus(api_->host_mapping_cache_control(
                    mapping_, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, byte_length),
                "memory_publication");
  }

  void Verify(size_t byte_length, uint8_t value) {
    for (size_t i = 0; i < byte_length; ++i) {
      Check(pointer_[i] == value, "mapped memory byte mismatch");
    }
  }

  void Release() {
    CheckStatus(api_->host_mapping_destroy(mapping_), "host_mapping_destroy");
    mapping_ = nullptr;
    pointer_ = nullptr;
    CheckStatus(api_->memory_destroy(memory_), "memory_destroy");
    memory_ = nullptr;
  }

  // Core public API borrowed from the linked provider.
  const amdf_api_t* api_ = nullptr;
  // Instance-owned system scope, borrowed through the shared test device cache.
  amdf_memory_scope_t* scope_ = nullptr;
  // Explicit live consumer and its required device address contract.
  amdf_memory_device_access_t access_ = {};
  // Native address kind selected by the benchmark's device family.
  amdf_memory_address_kind_t address_kind_ = AMDF_MEMORY_ADDRESS_GPU;
  // Qualified allocation request; only byte length changes between cases.
  amdf_memory_create_info_t create_info_ = {};
  // One case-owned resident allocation, never shared between repetitions.
  amdf_memory_t* memory_ = nullptr;
  // Explicit CPU view borrowing the live allocation.
  amdf_host_mapping_t* mapping_ = nullptr;
  // First byte borrowed from the host view.
  uint8_t* pointer_ = nullptr;
};

}  // namespace

int RunMemoryBenchmarks(amdf_engine_kind_t engine_kind, int argument_count,
                        char** argument_values) {
  if (!amdf_cts_provider_initialize(&argument_count, &argument_values)) {
    return EXIT_FAILURE;
  }
  benchmark::Initialize(&argument_count, argument_values);
  if (benchmark::ReportUnrecognizedArguments(argument_count, argument_values)) {
    return EXIT_FAILURE;
  }
  MemoryBenchmark fixture;
  fixture.Initialize(engine_kind);
  const std::string prefix =
      engine_kind == AMDF_ENGINE_KIND_GPU ? "GpuMemory/" : "XdnaMemory/";
  benchmark::RegisterBenchmark(
      (prefix + "Allocation").c_str(),
      [&fixture](benchmark::State& state) { fixture.Allocation(state); })
      ->Arg(4096)
      ->Arg(1048576)
      ->UseRealTime();
  benchmark::RegisterBenchmark(
      (prefix + "Initialization").c_str(),
      [&fixture](benchmark::State& state) { fixture.Initialization(state); })
      ->Arg(4096)
      ->Arg(1048576)
      ->UseRealTime();
  benchmark::RegisterBenchmark(
      (prefix + "Lifecycle").c_str(),
      [&fixture](benchmark::State& state) { fixture.Lifecycle(state); })
      ->Arg(4096)
      ->Arg(1048576)
      ->UseRealTime();
  benchmark::RegisterBenchmark(
      (prefix + "Publication").c_str(),
      [&fixture](benchmark::State& state) { fixture.Publication(state); })
      ->Arg(4096)
      ->Arg(1048576)
      ->UseRealTime();
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  CheckStatus(GetCtsDeviceCache().Deinitialize(), "device_cleanup");
  return amdf_cts_provider_deinitialize() ? EXIT_SUCCESS : EXIT_FAILURE;
}
