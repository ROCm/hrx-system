// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/sparse_mapping_test.h"

#include <ntstatus.h>

#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <span>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/endpoint_profile.h"
#include "libamdf/src/gpu/umd/kernel_queue.h"
#include "libamdf/src/gpu/umd/memory.h"
#include "libamdf/src/gpu/umd/memory_reservation.h"
#include "libamdf/src/gpu/umd/wddm/device.h"
#include "libamdf/src/gpu/umd/wddm/sparse_mapping_queue_test.h"
#include "libamdf/src/platform/endpoint.h"
#include "libamdf/src/platform/instance.h"
#include "libamdf/src/platform/windows/endpoint.h"
#include "libamdf/src/wait.h"

namespace amdf::testing {
namespace {

// A failed native transition may retain accepted work. Stop the test process
// instead of guessing what can be released. The successful path checks final
// detach and every release while the original execution domain remains alive.
void Require(bool condition, const char* operation) {
  if (condition) {
    return;
  }
  ADD_FAILURE() << operation << ": Win32=" << GetLastError();
  std::abort();
}

void RequireStatus(amdf_status_t status, const char* operation) {
  if (amdf_status_is_ok(status)) {
    return;
  }
  ADD_FAILURE() << operation << ": domain=" << amdf_status_domain(status)
                << " code=" << amdf_status_code(status);
  std::abort();
}

// One retained native allocation and its independent fixed GPU alias.
struct NativeAllocation {
  // Ordinary provider backing with eagerly prepared mapping-source access.
  amdf_gpu_umd_memory_t* memory = nullptr;
  // Committed CPU pages for system storage, absent for device-local storage.
  void* host_pointer = nullptr;
  // Base of the independent GPU address reservation.
  uint64_t device_address = 0;
  // Native backing and fixed-alias length, rounded to reservation granularity.
  uint64_t byte_length = 0;
};

class WddmSparseMappingBackend final : public SparseMappingBackend {
 public:
  explicit WddmSparseMappingBackend(amdf_native_lifetime_t lifetime)
      : lifetime_(lifetime) {}

  // Hardware scheduling uses the implemented provider queue. The native-only
  // software-scheduled witness owns its KMT context independently and does not
  // alter the provider's advertised queue families.
  bool SelectEndpoint() {
    const char* helper_path =
        std::getenv("AMDF_SPARSE_MAPPING_QUEUE_TEST_PATH");
    Require(helper_path != nullptr, "locate native queue test helper");
    // Bazel supplies explicit runtime paths. CMake colocates private DLLs
    // with amdf, separate from this statically linked test executable.
    if (std::getenv("AMDF_WKMI_BRIDGE_PATH") == nullptr) {
      auto bridge_path = std::filesystem::path(helper_path);
      bridge_path.replace_filename("amdf_wkmi_bridge.dll");
      Require(SetEnvironmentVariableW(L"AMDF_WKMI_BRIDGE_PATH",
                                      bridge_path.c_str()) != 0,
              "select colocated WKMI bridge");
    }
    helper_module_ = LoadLibraryA(helper_path);
    Require(helper_module_ != nullptr, "load native queue test helper");
    using QueryApi = const amdf_sparse_mapping_test_queue_api_t* (*)();
    const auto query_api = reinterpret_cast<QueryApi>(
        GetProcAddress(helper_module_, "amdf_sparse_mapping_test_queue_api"));
    Require(query_api != nullptr, "resolve native queue test API");
    software_api_ = query_api();
    RequireStatus(
        amdf_platform_instance_create(amdf_allocator_system(), &platform_),
        "create platform");
    uint32_t count = 0;
    RequireStatus(
        amdf_platform_endpoint_enumerate(platform_, 0, nullptr, &count),
        "count endpoints");
    std::vector<amdf_endpoint_summary_t> endpoints(count);
    RequireStatus(amdf_platform_endpoint_enumerate(platform_, count,
                                                   endpoints.data(), &count),
                  "enumerate endpoints");
    for (const auto& summary : endpoints) {
      if (summary.engine_kind != AMDF_ENGINE_KIND_GPU) {
        continue;
      }
      amdf_endpoint_info_t info = {};
      RequireStatus(amdf_platform_endpoint_open(platform_, &summary.id,
                                                &endpoint_, &info),
                    "open GPU endpoint");
      amdf_gpu_endpoint_profile_t* profile = nullptr;
      const amdf_status_t status = amdf_gpu_umd_create_endpoint_profile(
          endpoint_, amdf_allocator_system(), &profile);
      if (status != amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        RequireStatus(status, "query GPU profile");
        amdf_windows_gpu_memory_capabilities_t memory = {};
        RequireStatus(
            amdf_windows_gpu_query_memory_capabilities(endpoint_, &memory),
            "query GPU memory capabilities");
        for (uint32_t i = 0; i < profile->queue_family_count; ++i) {
          const auto& candidate = profile->queue_families[i];
          if (candidate.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA &&
              (candidate.publication_modes &
               AMDF_QUEUE_PUBLICATION_MODE_KERNEL) &&
              (candidate.roles & AMDF_QUEUE_ROLE_TRANSFER) &&
              memory.cache_coherent_memory_supported) {
            format_features = candidate.format_features;
            amdf_free(amdf_allocator_system(), profile);
            std::fprintf(stderr, "endpoint=%s lifetime=%u scheduling=HWS\n",
                         summary.name, lifetime_);
            ::testing::Test::RecordProperty("endpoint", summary.name);
            ::testing::Test::RecordProperty("native_lifetime", lifetime_);
            return true;
          }
        }
        amdf_free(amdf_allocator_system(), profile);
        if (memory.cache_coherent_memory_supported) {
          const NTSTATUS selection_status =
              software_api_->select(endpoint_->adapter, &software_scheduler_);
          if (selection_status != STATUS_NOT_SUPPORTED) {
            RequireStatus(amdf_kmt_make_status(selection_status),
                          "query software-scheduled SDMA engine");
            software_scheduled_ = true;
            format_features = software_scheduler_.format_features;
            std::fprintf(stderr, "endpoint=%s lifetime=%u scheduling=SWS\n",
                         summary.name, lifetime_);
            ::testing::Test::RecordProperty("endpoint", summary.name);
            ::testing::Test::RecordProperty("native_lifetime", lifetime_);
            return true;
          }
        }
      }
      RequireStatus(amdf_platform_endpoint_close(endpoint_), "close endpoint");
      endpoint_ = nullptr;
    }
    RequireStatus(amdf_platform_instance_destroy(platform_),
                  "destroy platform without an admitted queue");
    platform_ = nullptr;
    Require(FreeLibrary(helper_module_) != 0,
            "unload native queue test helper");
    return false;
  }

  void Initialize(uint32_t backing_count, uint64_t reservation_byte_length,
                  uint64_t payload_byte_length) override {
    amdf_platform_instance_lock_native(platform_);
    RequireStatus(amdf_gpu_umd_instance_prepare(&instance_, lifetime_,
                                                amdf_allocator_system()),
                  "prepare native instance");
    amdf_gpu_umd_device_result_t result = {};
    RequireStatus(amdf_gpu_umd_device_create(instance_, endpoint_,
                                             amdf_allocator_system(), lifetime_,
                                             &device_, &result),
                  "create native device");
    amdf_platform_instance_unlock_native(platform_);
    Require(amdf_kmt_api_supports_gpu_memory(device_->kmt),
            "native GPU memory operations available");
    reservation_byte_length_ = reservation_byte_length;
    for (size_t i = 0; i < reservations_.size(); ++i) {
      RequireStatus(amdf_gpu_umd_memory_reservation_create(
                        amdf_allocator_system(), 1, &device_,
                        reservation_byte_length_, 0, &reservations_[i]),
                    "create provider GPU reservation");
      uint32_t owner = 0;
      uint64_t granularity = 0;
      amdf_memory_access_t required_access = 0;
      amdf_gpu_umd_memory_reservation_query_access(
          reservations_[i], 0, &reservation_addresses[i], &owner, &granularity,
          &required_access);
      Require(kSparseSlabByteLength % granularity == 0,
              "source slabs meet queried mapping geometry");
    }

    backing_.reserve(backing_count);
    backing_addresses.reserve(backing_count);
    const auto slab_domain =
        device_->properties.is_discrete
            ? AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_LOCAL
            : AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_SYSTEM;
    for (uint32_t i = 0; i < backing_count; ++i) {
      backing_.push_back(
          Allocate(kSparseSlabByteLength, slab_domain,
                   AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE));
      backing_addresses.push_back(backing_.back().device_address);
    }

    const uint64_t page_size = AMDF_WINDOWS_GPU_PAGE_SIZE;
    const uint64_t stride =
        (payload_byte_length + page_size - 1) & ~(page_size - 1);
    transport_ = Allocate(3 * stride + 2 * page_size,
                          AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_SYSTEM,
                          AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                              AMDF_MEMORY_ACCESS_EXECUTE);
    auto* host = static_cast<uint8_t*>(transport_.host_pointer);
    upload = reinterpret_cast<uint32_t*>(host);
    readback = reinterpret_cast<uint32_t*>(host + stride);
    capture = reinterpret_cast<uint32_t*>(host + 2 * stride);
    upload_address = transport_.device_address;
    readback_address = transport_.device_address + stride;
    capture_address = transport_.device_address + 2 * stride;
    completion_address = transport_.device_address + 3 * stride;
    completion_ = reinterpret_cast<volatile uint32_t*>(host + 3 * stride);
    *completion_ = 0;
    command_pointer_ = host + 3 * stride + page_size;
    command_address_ = completion_address + page_size;
    // A bounded test-owned IB page, not an invented hardware maximum. The
    // native submit path independently enforces the queried queue's limits.
    command_byte_capacity = page_size;
    if (software_scheduled_) {
      RequireStatus(amdf_kmt_make_status(software_api_->create(
                        device_->device, device_->physical_adapter_index,
                        &software_scheduler_, &software_queue_)),
                    "create persistent native SDMA context");
    } else {
      RequireStatus(amdf_gpu_umd_kernel_queue_create(
                        device_, AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA, 0, &queue_),
                    "create persistent SDMA queue");
    }
    std::fprintf(stderr,
                 "prepared: reservations=2 reservation_bytes=%" PRIu64
                 " slabs=%u slab_bytes=%" PRIu64 " payload_bytes=%" PRIu64
                 " placement=%s\n",
                 reservation_byte_length_, backing_count, kSparseSlabByteLength,
                 payload_byte_length,
                 device_->properties.is_discrete ? "LOCAL" : "SYSTEM");
  }

  void Attach(std::span<const SparseMapping> mappings) override {
    Update(mappings, AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
  }

  void Detach(std::span<const SparseMapping> mappings) override {
    Update(mappings, 0);
  }

  void Execute(std::span<const uint32_t> words,
               uint32_t completion_value) override {
    Require(words.size_bytes() <= command_byte_capacity,
            "command stream fits prepared storage");
    std::memcpy(command_pointer_, words.data(), words.size_bytes());
    MemoryBarrier();
    if (software_scheduled_) {
      RequireStatus(
          amdf_kmt_make_status(software_api_->execute(
              software_queue_, command_address_,
              static_cast<uint32_t>(words.size_bytes()), ++progress_)),
          "execute prepared software-scheduled SDMA stream");
    } else {
      uint32_t accepted_wait_count = 0;
      RequireStatus(amdf_gpu_umd_kernel_queue_submit(
                        queue_, command_address_, words.size_bytes(),
                        ++progress_, 0, nullptr, &accepted_wait_count),
                    "submit prepared SDMA stream");
      const amdf_wait_deadline_t deadline = {UINT64_MAX, 0};
      RequireStatus(
          amdf_gpu_umd_kernel_queue_wait(queue_, progress_, &deadline),
          "retire native SDMA command storage");
    }
    // Native progress and the explicit last SDMA write are separate evidence.
    // Reset/error progress never authorizes reading or releasing backing.
    MemoryBarrier();
    Require(*completion_ == completion_value,
            "SDMA execution completed all transfers");
  }

  void DetachReservations() {
    for (size_t i = 0; i < reservations_.size(); ++i) {
      RequireStatus(amdf_gpu_umd_memory_reservation_detach(reservations_[i], 0),
                    "complete full-group detach without releasing VA");
      uint64_t address = 0;
      uint32_t owner = 0;
      uint64_t granularity = 0;
      amdf_memory_access_t required_access = 0;
      amdf_gpu_umd_memory_reservation_query_access(
          reservations_[i], 0, &address, &owner, &granularity,
          &required_access);
      Require(address == reservation_addresses[i],
              "detach preserves the owned address interval");
    }
  }

  void Release() override {
    // Backing release precedes both address and queue destruction. Neither
    // destruction can supply the checked translation-removal guarantee.
    DetachReservations();
    for (const auto& allocation : backing_) {
      ReleaseAllocation(allocation);
    }
    for (auto* reservation : reservations_) {
      RequireStatus(amdf_gpu_umd_memory_reservation_destroy(reservation),
                    "release detached address reservation");
    }
    if (software_scheduled_) {
      RequireStatus(
          amdf_kmt_make_status(software_api_->destroy(software_queue_)),
          "destroy persistent native SDMA context");
    } else {
      RequireStatus(amdf_gpu_umd_kernel_queue_destroy(queue_),
                    "destroy persistent SDMA queue");
    }
    ReleaseAllocation(transport_);
    RequireStatus(amdf_gpu_umd_device_destroy(device_),
                  "destroy native device");
    RequireStatus(amdf_gpu_umd_instance_destroy(instance_),
                  "destroy native instance");
    RequireStatus(amdf_platform_endpoint_close(endpoint_), "close endpoint");
    RequireStatus(amdf_platform_instance_destroy(platform_),
                  "destroy platform");
    Require(FreeLibrary(helper_module_) != 0,
            "unload native queue test helper");
    std::fprintf(stderr, "checked final detach and native release complete\n");
  }

 private:
  void Update(std::span<const SparseMapping> mappings,
              amdf_memory_access_t access) {
    for (size_t i = 0; i < reservations_.size(); ++i) {
      std::vector<amdf_gpu_umd_mapping_update_t> updates;
      for (const auto& mapping : mappings) {
        if (mapping.reservation != i) {
          continue;
        }
        updates.push_back({
            .target_byte_offset = mapping.byte_offset,
            .byte_length = kSparseSlabByteLength,
            .memory = access != 0 ? backing_[mapping.backing].memory : nullptr,
            .access = access,
        });
      }
      if (updates.empty()) {
        continue;
      }
      bool mutated = false;
      RequireStatus(
          amdf_gpu_umd_memory_reservation_update(
              reservations_[i], 0, static_cast<uint32_t>(updates.size()),
              updates.data(), &mutated),
          "complete provider mapping update");
      Require(mutated, "nonempty mapping update reached native publication");
    }
  }

  NativeAllocation Allocate(uint64_t byte_length,
                            amdf_wkmi_bridge_gpu_allocation_domain_t domain,
                            amdf_memory_access_t access) {
    NativeAllocation allocation;
    allocation.byte_length =
        (byte_length + AMDF_WINDOWS_GPU_RESERVATION_GRANULARITY - 1) &
        ~(AMDF_WINDOWS_GPU_RESERVATION_GRANULARITY - 1);
    const bool system = domain == AMDF_WKMI_BRIDGE_GPU_ALLOCATION_DOMAIN_SYSTEM;
    amdf_memory_native_profile_t profile = {};
    RequireStatus(amdf_gpu_umd_device_query_memory_profile(
                      device_, system ? 0 : 1, &profile),
                  "query ordinary native backing profile");
    amdf_memory_native_create_info_t create = {};
    create.byte_length = allocation.byte_length;
    create.device_access = access;
    create.required_flags =
        AMDF_MEMORY_FLAG_MAPPING_SOURCE |
        (system ? AMDF_MEMORY_FLAG_HOST_COHERENT : UINT64_C(0));
    amdf_gpu_umd_memory_result_t result = {};
    RequireStatus(
        amdf_gpu_umd_memory_prepare(device_, 0, nullptr, &profile, &create,
                                    &allocation.memory, &result),
        "prepare ordinary backing and eager mapping access");
    allocation.device_address = result.device_address;
    if (system) {
      const amdf_memory_map_info_t map = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
          .structure_size = sizeof(map),
          .byte_length = allocation.byte_length,
          .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
      };
      amdf_gpu_umd_host_mapping_t* view = nullptr;
      amdf_gpu_umd_host_mapping_result_t mapped = {};
      RequireStatus(
          amdf_gpu_umd_memory_map(allocation.memory, &profile.host_mapping,
                                  &map, &view, &mapped),
          "obtain native backing host view");
      allocation.host_pointer = mapped.pointer;
      amdf_gpu_umd_host_mapping_destroy(view);
    }
    return allocation;
  }

  void ReleaseAllocation(const NativeAllocation& allocation) {
    RequireStatus(amdf_gpu_umd_memory_destroy(allocation.memory),
                  "release detached ordinary backing");
  }

  // Native lifetime policy exercised by this test instance.
  amdf_native_lifetime_t lifetime_;
  // Platform discovery and KMT connection owner.
  amdf_platform_instance_t* platform_ = nullptr;
  // Open endpoint outliving the device.
  amdf_platform_endpoint_t* endpoint_ = nullptr;
  // Native instance state prepared through the common provider contract.
  amdf_gpu_umd_instance_t* instance_ = nullptr;
  // Native device and paging owner outliving every allocation and queue.
  amdf_gpu_umd_device_t* device_ = nullptr;
  // Persistent SDMA queue retained across all mapping generations.
  amdf_gpu_umd_kernel_queue_t* queue_ = nullptr;
  // Test-only DLL retaining WKMI's binary C++ and release CRT ABI.
  HMODULE helper_module_ = nullptr;
  // Immutable native test operations, borrowed from helper_module_.
  const amdf_sparse_mapping_test_queue_api_t* software_api_ = nullptr;
  // True when the selected native engine uses software scheduling.
  bool software_scheduled_ = false;
  // Native engine facts selected once during endpoint discovery.
  amdf_sparse_mapping_test_scheduler_t software_scheduler_ = {};
  // Test-owned software context, absent when a provider HWS queue is used.
  amdf_sparse_mapping_test_queue_t* software_queue_ = nullptr;
  // Last native command submission value, excluding the reset sentinel.
  uint64_t progress_ = 0;
  // Byte length of each independent sparse reservation.
  uint64_t reservation_byte_length_ = 0;
  // Provider-owned address groups, cleared before retained backing is released.
  std::array<amdf_gpu_umd_memory_reservation_t*, 2> reservations_ = {};
  // Retained slabs whose identities do not change during replacement.
  std::vector<NativeAllocation> backing_;
  // Coherent upload, readback, capture, completion and command storage.
  NativeAllocation transport_;
  // Coherent last-write execution marker in the transport allocation.
  volatile uint32_t* completion_ = nullptr;
  // Reusable command-storage CPU view, written only after prior retirement.
  void* command_pointer_ = nullptr;
  // Stable GPU address of that same command storage.
  uint64_t command_address_ = 0;
};

class WddmSparseMappingTest
    : public ::testing::TestWithParam<amdf_native_lifetime_t> {};

TEST_P(WddmSparseMappingTest, ReplacesRetainedBackingBeforeImmediateAccess) {
  WddmSparseMappingBackend backend(GetParam());
  if (!backend.SelectEndpoint()) {
    GTEST_SKIP() << "No native SDMA engine with coherent host "
                    "storage; mapping replacement has not been exercised";
  }
  ExerciseSparseMappingReplacement(backend);
}

TEST_P(WddmSparseMappingTest, DetachPreservesAddressesForBackingReuse) {
  WddmSparseMappingBackend backend(GetParam());
  if (!backend.SelectEndpoint()) {
    GTEST_SKIP() << "No native SDMA engine with coherent host storage";
  }
  constexpr uint64_t kOffset = (UINT64_C(1) << 32) + kSparseSlabByteLength;
  constexpr uint64_t kPayloadByteLength = 4096;
  constexpr size_t kWordCount = kPayloadByteLength / sizeof(uint32_t);
  const std::array<SparseRange, 2> ranges = {{
      {0, kOffset + 512, kPayloadByteLength / 2},
      {1, 512, kPayloadByteLength / 2},
  }};
  std::array<SparseMapping, 2> mappings = {{
      {0, kOffset, 0},
      {1, 0, 1},
  }};
  backend.Initialize(2, kOffset + kSparseSlabByteLength, kPayloadByteLength);
  for (uint32_t generation = 1; generation <= 2; ++generation) {
    // Prepare every command before mapping; the first subsequent GPU access
    // must see the new backing without an intervening allocation or dispatch.
    const auto upload = SparseTransferCommands(
        backend, ranges, mappings, SparseTransfer::kUpload, 3 * generation);
    const auto readback =
        SparseTransferCommands(backend, ranges, mappings,
                               SparseTransfer::kReadback, 3 * generation + 1);
    const auto physical = SparseTransferCommands(
        backend, ranges, mappings, SparseTransfer::kPhysicalReadback,
        3 * generation + 2);
    for (size_t i = 0; i < kWordCount; ++i) {
      backend.upload[i] = SparsePattern(generation, static_cast<uint32_t>(i));
    }
    backend.Attach(mappings);
    ExecuteSparseTransfer(backend, upload);
    ExecuteSparseTransfer(backend, readback);
    bool correct = CheckSparseWords("detached address reuse", backend.readback,
                                    kWordCount, generation);
    ExecuteSparseTransfer(backend, physical);
    correct &= CheckSparseWords("independent reused backing", backend.readback,
                                kWordCount, generation);
    backend.DetachReservations();
    if (!correct) {
      break;
    }
    std::swap(mappings[0].backing, mappings[1].backing);
  }
  backend.Release();
}

INSTANTIATE_TEST_SUITE_P(NativeLifetime, WddmSparseMappingTest,
                         ::testing::Values(AMDF_NATIVE_LIFETIME_PROCESS,
                                           AMDF_NATIVE_LIFETIME_INSTANCE));

}  // namespace
}  // namespace amdf::testing
