// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif  // WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX
#include <d3d12.h>
#include <dxgi1_4.h>
#include <windows.h>
#include <wrl/client.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "libamdf/cts/gpu/gpu_device_fixture.h"
#include "libamdf/cts/gpu/pm4/encoding/commands.h"

using Microsoft::WRL::ComPtr;

#include "util/wait.h"

namespace {
constexpr uint64_t kByteLength = 65536;
constexpr uint64_t kSourceOffset = 4096;
constexpr uint64_t kDestinationOffset = 8192;
constexpr uint32_t kWordCount = 128;

class D3D12MemoryInteropTest
    : public GpuDeviceFixture,
      public ::testing::WithParamInterface<amdf_external_memory_type_t> {
 protected:
  void SetUp() override {
    GpuDeviceFixture::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    ASSERT_NE(local_scope_, nullptr);
    import_info_.type = AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO;
    import_info_.structure_size = sizeof(import_info_);
    import_info_.access_count = 1;
    import_info_.accesses = &memory_access_;
    import_info_.required_flags = AMDF_MEMORY_FLAG_SHAREABLE;
    import_info_.memory_profile_ordinal = FindMemoryProfileOrdinal(
        local_scope_,
        AMDF_MEMORY_PROFILE_ROLE_IMPORT | AMDF_MEMORY_PROFILE_ROLE_EXPORT,
        import_info_.required_flags, memory_access_.requirements);
    ASSERT_NE(import_info_.memory_profile_ordinal,
              AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    ASSERT_TRUE(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory_))));
    amdf_endpoint_info_t endpoint = {};
    endpoint.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
    endpoint.structure_size = sizeof(endpoint);
    ASSERT_EQ(api_->endpoint_query_info(endpoint_, &endpoint), AMDF_STATUS_OK);
    ASSERT_EQ(endpoint.native_identity.type,
              AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_WINDOWS_ADAPTER);
    const auto& identity = endpoint.native_identity.value.windows_adapter;
    ASSERT_EQ(identity.physical_adapter_index, 0u);
    const LUID luid = {
        static_cast<DWORD>(identity.luid),
        static_cast<LONG>(static_cast<uint32_t>(identity.luid >> 32)),
    };
    ASSERT_TRUE(
        SUCCEEDED(factory_->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter_))));
    ASSERT_TRUE(SUCCEEDED(D3D12CreateDevice(
        adapter_.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d_))));
    ASSERT_EQ(d3d_->GetNodeCount(), 1u);
    ASSERT_NO_FATAL_FAILURE(CreateSharedBuffer());
  }

  amdf_external_memory_t Transport() const {
    amdf_external_memory_t value = {};
    value.type = GetParam();
    value.payload.native_handle = shared_;
    value.source_byte_offset = kSourceOffset;
    value.byte_length = kDestinationOffset + kWordCount * 4 - kSourceOffset;
    return value;
  }

  static void AMDF_CALL ReleaseSource(void* user_data,
                                      amdf_external_memory_type_t type,
                                      amdf_external_memory_payload_t payload) {
    EXPECT_TRUE(type == AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE ||
                type == AMDF_EXTERNAL_MEMORY_TYPE_D3D12_HEAP);
    EXPECT_TRUE(CloseHandle(payload.native_handle));
    ++*static_cast<uint32_t*>(user_data);
  }

  void DestroyImportedStorage() {
    if (imported_memory_ != nullptr) {
      ASSERT_EQ(api_->memory_destroy(std::exchange(imported_memory_, nullptr),
                                     nullptr),
                AMDF_STATUS_OK);
    }
  }

  void TearDown() override {
    if (reservation_cleanup_failed_) {
      return;
    }
    if (pending_submission_ != 0) {
      ADD_FAILURE() << "Native work has not retired; retaining its backing.";
      return;
    }
    if (queue_ != nullptr) {
      const auto status = api_->kernel_queue_destroy(queue_);
      if (status != amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
        queue_ = nullptr;
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
    }
    if (reservation_ != nullptr) {
      ASSERT_NO_FATAL_FAILURE(ReleaseReservation());
    }
    if (command_mapping_ != nullptr) {
      ASSERT_EQ(api_->host_mapping_destroy(command_mapping_), AMDF_STATUS_OK);
      command_mapping_ = nullptr;
    }
    if (command_memory_ != nullptr) {
      ASSERT_EQ(api_->memory_destroy(std::exchange(command_memory_, nullptr),
                                     nullptr),
                AMDF_STATUS_OK);
    }
    ASSERT_NO_FATAL_FAILURE(DestroyImportedStorage());
    if (shared_ != nullptr) {
      ASSERT_TRUE(CloseHandle(shared_));
      shared_ = nullptr;
    }
    buffer_.Reset();
    heap_.Reset();
    d3d_.Reset();
    adapter_.Reset();
    factory_.Reset();
    GpuDeviceFixture::TearDown();
  }

  static D3D12_RESOURCE_DESC BufferDescription() {
    D3D12_RESOURCE_DESC buffer = {};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = kByteLength;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return buffer;
  }

  void CreatePlacedBuffer() {
    const auto buffer = BufferDescription();
    ASSERT_TRUE(SUCCEEDED(d3d_->CreatePlacedResource(
        heap_.Get(), 0, &buffer, D3D12_RESOURCE_STATE_COMMON, nullptr,
        IID_PPV_ARGS(&buffer_))));
  }

  void CreateSharedBuffer() {
    if (shared_ != nullptr) {
      ASSERT_TRUE(CloseHandle(shared_));
      shared_ = nullptr;
    }
    buffer_.Reset();
    heap_.Reset();
    D3D12_HEAP_PROPERTIES properties = {};
    properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    properties.CreationNodeMask = 1;
    properties.VisibleNodeMask = 1;
    if (GetParam() == AMDF_EXTERNAL_MEMORY_TYPE_D3D12_HEAP) {
      D3D12_HEAP_DESC heap = {};
      heap.SizeInBytes = kByteLength;
      heap.Properties = properties;
      heap.Flags = D3D12_HEAP_FLAG_SHARED;
      ASSERT_TRUE(SUCCEEDED(d3d_->CreateHeap(&heap, IID_PPV_ARGS(&heap_))));
      ASSERT_NO_FATAL_FAILURE(CreatePlacedBuffer());
    } else {
      const auto buffer = BufferDescription();
      ASSERT_TRUE(SUCCEEDED(d3d_->CreateCommittedResource(
          &properties, D3D12_HEAP_FLAG_SHARED, &buffer,
          D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&buffer_))));
    }
    ID3D12DeviceChild* owner =
        heap_ ? static_cast<ID3D12DeviceChild*>(heap_.Get()) : buffer_.Get();
    ASSERT_TRUE(SUCCEEDED(d3d_->CreateSharedHandle(owner, nullptr, GENERIC_ALL,
                                                   nullptr, &shared_)));
  }

  void OpenSharedBuffer(uint64_t source_byte_offset = kSourceOffset,
                        uint64_t byte_length = kDestinationOffset +
                                               kWordCount * 4 - kSourceOffset) {
    amdf_external_memory_t transport = Transport();
    transport.source_byte_offset = source_byte_offset;
    transport.byte_length = byte_length;
    transport.release = ReleaseSource;
    transport.release_user_data = &release_count_;
    const uint32_t releases_before = release_count_;
    const amdf_status_t status = api_->memory_import(
        local_scope_, &import_info_, &transport, &imported_memory_, nullptr);
    ASSERT_EQ(status, AMDF_STATUS_OK)
        << "domain=" << amdf_status_domain(status) << " code=" << std::hex
        << amdf_status_code(status);
    shared_ = nullptr;
    ASSERT_EQ(release_count_, releases_before + 1);
    const amdf_external_memory_t empty = {};
    ASSERT_EQ(std::memcmp(&transport, &empty, sizeof(transport)), 0);
    amdf_memory_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->memory_query_info(imported_memory_, &info), AMDF_STATUS_OK);
    ASSERT_EQ(info.source_byte_offset, source_byte_offset);
    ASSERT_EQ(info.byte_length, byte_length);
    ASSERT_GE(info.native_allocation_byte_length, kByteLength);
    ASSERT_EQ(api_->memory_query_address(imported_memory_, 0,
                                         AMDF_MEMORY_ADDRESS_GPU, &address_),
              AMDF_STATUS_OK);
    address_ -= source_byte_offset;
  }

  void CopyD3D12(ID3D12Resource* source, ID3D12Resource* destination) {
    // Command lists do not retain referenced resources. These independent
    // owners can survive caller and fixture teardown if retirement is unknown.
    ComPtr<ID3D12Device> device = d3d_;
    ComPtr<ID3D12Resource> source_owner = source;
    ComPtr<ID3D12Resource> destination_owner = destination;
    ComPtr<ID3D12Heap> heap_owner = heap_;
    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC queue_info = {};
    queue_info.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ASSERT_TRUE(
        SUCCEEDED(d3d_->CreateCommandQueue(&queue_info, IID_PPV_ARGS(&queue))));
    ComPtr<ID3D12CommandAllocator> allocator;
    ASSERT_TRUE(SUCCEEDED(d3d_->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))));
    ComPtr<ID3D12GraphicsCommandList> commands;
    ASSERT_TRUE(SUCCEEDED(d3d_->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
        IID_PPV_ARGS(&commands))));
    if (heap_) {
      D3D12_RESOURCE_BARRIER activation = {};
      activation.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
      activation.Aliasing.pResourceAfter = buffer_.Get();
      commands->ResourceBarrier(1, &activation);
    }
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = buffer_.Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = source == buffer_.Get()
                                        ? D3D12_RESOURCE_STATE_COPY_SOURCE
                                        : D3D12_RESOURCE_STATE_COPY_DEST;
    commands->ResourceBarrier(1, &barrier);
    commands->CopyBufferRegion(destination, 0, source, 0, kByteLength);
    std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    commands->ResourceBarrier(1, &barrier);
    ASSERT_TRUE(SUCCEEDED(commands->Close()));
    ComPtr<ID3D12Fence> fence;
    ASSERT_TRUE(SUCCEEDED(
        d3d_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))));
    ID3D12CommandList* lists[] = {commands.Get()};
    queue->ExecuteCommandLists(1, lists);

    // No fatal assertion may unwind submitted owners before classification.
    // A rejected signal does not promise a fence value that can be waited on.
    const char* operation = "queue signal";
    HRESULT status = queue->Signal(fence.Get(), 1);
    if (SUCCEEDED(status)) {
      operation = "fence wait";
      status = fence->SetEventOnCompletion(1, nullptr);
    }
    const uint64_t completed_value = fence->GetCompletedValue();
    const HRESULT removed_reason = device->GetDeviceRemovedReason();
    if (completed_value != 1 && completed_value != UINT64_MAX &&
        SUCCEEDED(removed_reason)) {
      // Preserve every submitted dependency until process exit. An HRESULT
      // failure alone proves neither completion nor device removal.
      queue.Detach();
      commands.Detach();
      allocator.Detach();
      fence.Detach();
      source_owner.Detach();
      destination_owner.Detach();
      heap_owner.Detach();
      device.Detach();
      FAIL() << "D3D12 retirement unproven; retaining submitted owners: "
             << operation << " returned " << std::hex << status;
    }
    ASSERT_TRUE(SUCCEEDED(status)) << operation << ": " << std::hex << status;
    // Removal permits cleanup, but its UINT64_MAX fence is not successful work.
    ASSERT_EQ(completed_value, 1u);
    ASSERT_EQ(removed_reason, S_OK) << std::hex << removed_reason;
  }

  void CreateStaging(D3D12_HEAP_TYPE type, ID3D12Resource** out_resource) {
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = type;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;
    const auto description = buffer_->GetDesc();
    const auto state = type == D3D12_HEAP_TYPE_UPLOAD
                           ? D3D12_RESOURCE_STATE_GENERIC_READ
                           : D3D12_RESOURCE_STATE_COPY_DEST;
    ASSERT_TRUE(SUCCEEDED(d3d_->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &description, state, nullptr,
        IID_PPV_ARGS(out_resource))));
  }

  void PrepareNativeCopy(uint32_t family_ordinal, uint64_t source_address,
                         uint64_t destination_address) {
    amdf_gpu_kernel_queue_create_info_t queue = {};
    queue.type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO;
    queue.structure_size = sizeof(queue);
    queue.queue_family_ordinal = family_ordinal;
    ASSERT_NO_FATAL_FAILURE(CheckVisibility(family_ordinal));
    ASSERT_EQ(gpu_api_->kernel_queue_create(device_, &queue, &queue_),
              AMDF_STATUS_OK);
    const amdf_memory_device_access_t access = {
        device_,
        {
            .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                      AMDF_MEMORY_ACCESS_EXECUTE,
            .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
        }};
    amdf_memory_create_info_t create = {};
    create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    create.structure_size = sizeof(create);
    create.access_count = 1;
    create.accesses = &access;
    create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    create.byte_length = 65536;
    create.memory_profile_ordinal = FindGpuMemoryProfileOrdinal(
        api_, system_scope_, device_,
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
        create.required_flags, access.requirements);
    ASSERT_NE(create.memory_profile_ordinal,
              AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    ASSERT_EQ(
        api_->memory_create(system_scope_, &create, &command_memory_, nullptr),
        AMDF_STATUS_OK);
    amdf_memory_map_info_t map = {};
    map.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
    map.structure_size = sizeof(map);
    map.byte_length = create.byte_length;
    map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    ASSERT_EQ(api_->memory_map(command_memory_, &map, &command_mapping_),
              AMDF_STATUS_OK);
    amdf_host_mapping_info_t mapped = {};
    mapped.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
    mapped.structure_size = sizeof(mapped);
    ASSERT_EQ(api_->host_mapping_query_info(command_mapping_, &mapped),
              AMDF_STATUS_OK);
    Pm4CommandWriter commands(static_cast<uint32_t*>(mapped.pointer),
                              *pm4_profile_);
    commands.SystemBarrier();
    for (uint32_t i = 0; i < kWordCount; ++i) {
      commands.CopyData32(source_address + i * sizeof(uint32_t),
                          destination_address + i * sizeof(uint32_t));
    }
    commands.SystemBarrier();
    commands.PadToEightWords();
    command_byte_length_ = commands.word_count() * sizeof(uint32_t);
    ASSERT_EQ(api_->host_mapping_cache_control(command_mapping_,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, create.byte_length),
              AMDF_STATUS_OK);
  }

  void CheckVisibility(uint32_t family_ordinal) {
    amdf_memory_profile_pair_query_t query = {};
    query.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY;
    query.structure_size = sizeof(query);
    query.memory_profile_ordinal = import_info_.memory_profile_ordinal;
    query.access_count = import_info_.access_count;
    query.accesses = import_info_.accesses;
    query.required_flags = import_info_.required_flags;
    query.external_memory_type = GetParam();
    query.producer.kind = AMDF_MEMORY_SITE_KIND_DEVICE;
    query.producer.value.device.queue_family_ordinal = family_ordinal;
    query.consumer = query.producer;
    amdf_memory_pair_info_t prospective = {};
    prospective.type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
    prospective.structure_size = sizeof(prospective);
    ASSERT_EQ(
        api_->memory_scope_query_pair_info(local_scope_, &query, &prospective),
        AMDF_STATUS_OK);
    amdf_memory_site_t site = {};
    site.type = AMDF_STRUCTURE_TYPE_MEMORY_SITE;
    site.structure_size = sizeof(site);
    site.kind = AMDF_MEMORY_SITE_KIND_DEVICE;
    site.value.device.memory = imported_memory_;
    site.value.device.queue_family_ordinal = family_ordinal;
    amdf_memory_pair_info_t actual = {};
    actual.type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
    actual.structure_size = sizeof(actual);
    ASSERT_EQ(api_->memory_query_pair_info(&site, &site, &actual),
              AMDF_STATUS_OK);
    // Either query must authorize this fixed recipe before queue creation.
    // Prospective and concrete records need not be bytewise identical.
    for (const auto* pair : {&prospective, &actual}) {
      ASSERT_NE(pair->flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE,
                0u);
      ASSERT_EQ(pair->release.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
      ASSERT_EQ(pair->release.executor, AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
      ASSERT_EQ(pair->release.operation,
                AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM);
      ASSERT_EQ(pair->acquire.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
      ASSERT_EQ(pair->acquire.executor, AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
      ASSERT_EQ(pair->acquire.operation,
                AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM);
    }
  }

  void ExpectRejected(amdf_external_memory_t source) {
    source.release = ReleaseSource;
    source.release_user_data = &release_count_;
    const amdf_external_memory_t original = source;
    const uint32_t releases_before = release_count_;
    auto* const sentinel = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
    amdf_memory_t* output = sentinel;
    const amdf_status_t status = api_->memory_import(
        local_scope_, &import_info_, &source, &output, nullptr);
    ASSERT_FALSE(amdf_status_is_ok(status));
    EXPECT_EQ(output, sentinel);
    EXPECT_EQ(std::memcmp(&source, &original, sizeof(source)), 0);
    EXPECT_EQ(release_count_, releases_before);
    DWORD flags = 0;
    EXPECT_TRUE(GetHandleInformation(source.payload.native_handle, &flags));
  }

  void ExecuteNativeCopy() {
    amdf_gpu_kernel_command_t command = {};
    command.memory = command_memory_;
    command.byte_length = command_byte_length_;
    amdf_gpu_kernel_queue_submission_info_t submit = {};
    submit.type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO;
    submit.structure_size = sizeof(submit);
    submit.command_count = 1;
    submit.commands = &command;
    ASSERT_EQ(
        gpu_api_->kernel_queue_submit(queue_, &submit, &pending_submission_),
        AMDF_STATUS_OK);
    ASSERT_EQ(api_->kernel_queue_wait(queue_, pending_submission_,
                                      amdf::cts::kWaitTimeoutNs,
                                      amdf::cts::kWaitPollIntervalNs),
              AMDF_STATUS_OK);
    pending_submission_ = 0;
  }

  void ReexportAndDetach() {
    // Drop the original API resource before producing the return transport.
    buffer_.Reset();
    heap_.Reset();
    amdf_memory_export_info_t export_info = {};
    export_info.type = AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO;
    export_info.structure_size = sizeof(export_info);
    export_info.external_memory_type = GetParam();
    export_info.byte_length =
        kDestinationOffset + kWordCount * 4 - kSourceOffset;
    amdf_external_memory_t exported = {};
    ASSERT_EQ(api_->memory_export(imported_memory_, &export_info, &exported),
              AMDF_STATUS_OK);
    ASSERT_EQ(exported.source_byte_offset, kSourceOffset);
    ASSERT_EQ(exported.byte_length, export_info.byte_length);
    ASSERT_NO_FATAL_FAILURE(DestroyImportedStorage());
    const HRESULT opened =
        GetParam() == AMDF_EXTERNAL_MEMORY_TYPE_D3D12_HEAP
            ? d3d_->OpenSharedHandle(exported.payload.native_handle,
                                     IID_PPV_ARGS(&heap_))
            : d3d_->OpenSharedHandle(exported.payload.native_handle,
                                     IID_PPV_ARGS(&buffer_));
    api_->external_memory_release(&exported);
    ASSERT_TRUE(SUCCEEDED(opened)) << std::hex << opened;
    if (heap_) {
      ASSERT_EQ(heap_->GetDesc().SizeInBytes, kByteLength);
      ASSERT_NO_FATAL_FAILURE(CreatePlacedBuffer());
    }
    const auto description = buffer_->GetDesc();
    ASSERT_EQ(description.Dimension, D3D12_RESOURCE_DIMENSION_BUFFER);
    ASSERT_EQ(description.Width, kByteLength);
  }

  void RunRoundTrip(uint32_t family_ordinal) {
    ASSERT_NO_FATAL_FAILURE(CreateSharedBuffer());
    ASSERT_NO_FATAL_FAILURE(OpenSharedBuffer());
    ASSERT_NO_FATAL_FAILURE(PrepareNativeCopy(family_ordinal,
                                              address_ + kSourceOffset,
                                              address_ + kDestinationOffset));
    ComPtr<ID3D12Resource> upload;
    ComPtr<ID3D12Resource> readback;
    ASSERT_NO_FATAL_FAILURE(CreateStaging(D3D12_HEAP_TYPE_UPLOAD, &upload));
    ASSERT_NO_FATAL_FAILURE(CreateStaging(D3D12_HEAP_TYPE_READBACK, &readback));
    std::vector<uint32_t> expected(kByteLength / 4);
    for (uint32_t generation = 1; generation <= 8; ++generation) {
      for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = (generation * 0x13579bdu) ^
                      (static_cast<uint32_t>(i) * 0x9e3779b9u);
      }
      void* pointer = nullptr;
      const D3D12_RANGE no_read = {0, 0};
      ASSERT_TRUE(SUCCEEDED(upload->Map(0, &no_read, &pointer)));
      std::memcpy(pointer, expected.data(), kByteLength);
      const D3D12_RANGE written = {0, kByteLength};
      upload->Unmap(0, &written);
      ASSERT_NO_FATAL_FAILURE(CopyD3D12(upload.Get(), buffer_.Get()));
      ASSERT_NO_FATAL_FAILURE(ExecuteNativeCopy());
      for (uint32_t i = 0; i < kWordCount; ++i) {
        const uint32_t source = expected[kSourceOffset / 4 + i];
        expected[kDestinationOffset / 4 + i] = source;
      }
      ASSERT_NO_FATAL_FAILURE(CopyD3D12(buffer_.Get(), readback.Get()));
      ASSERT_TRUE(SUCCEEDED(readback->Map(0, &written, &pointer)));
      const auto* actual = static_cast<const uint32_t*>(pointer);
      size_t mismatches = 0;
      for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
          if (mismatches < 4) {
            std::printf(
                "mismatch: generation=%u offset=%zu expected=%08x "
                "actual=%08x\n",
                generation, i * 4, expected[i], actual[i]);
          }
          ++mismatches;
        }
      }
      readback->Unmap(0, &no_read);
      ASSERT_EQ(mismatches, 0u);
    }
    ASSERT_NO_FATAL_FAILURE(ReexportAndDetach());
    ASSERT_NO_FATAL_FAILURE(CopyD3D12(buffer_.Get(), readback.Get()));
    void* pointer = nullptr;
    const D3D12_RANGE read = {0, kByteLength};
    ASSERT_TRUE(SUCCEEDED(readback->Map(0, &read, &pointer)));
    const int difference = std::memcmp(pointer, expected.data(), kByteLength);
    const D3D12_RANGE no_write = {0, 0};
    readback->Unmap(0, &no_write);
    ASSERT_EQ(difference, 0);
  }

  void ReleaseReservation() {
    const amdf_status_t status =
        api_->memory_reservation_destroy(std::exchange(reservation_, nullptr));
    reservation_cleanup_failed_ = !amdf_status_is_ok(status);
    ASSERT_EQ(status, AMDF_STATUS_OK);
  }

  enum class MappingPublication { kHostCompleted, kQueueOrdered };

  void RunMappingReplacement(uint32_t family_ordinal,
                             MappingPublication publication) {
    amdf_memory_reservation_capabilities_t capabilities = {};
    capabilities.type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES;
    capabilities.structure_size = sizeof(capabilities);
    const amdf_status_t status =
        api_->device_query_memory_reservation_capabilities(device_, 0,
                                                           &capabilities);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      GTEST_SKIP() << "host-completed GPU mapping updates are unavailable";
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    memory_access_.requirements.flags |= AMDF_MEMORY_FLAG_MAPPING_SOURCE;
    memory_access_.requirements.access |= capabilities.required_access;
    import_info_.memory_profile_ordinal = FindMemoryProfileOrdinal(
        local_scope_,
        AMDF_MEMORY_PROFILE_ROLE_IMPORT | AMDF_MEMORY_PROFILE_ROLE_EXPORT,
        import_info_.required_flags, memory_access_.requirements);
    ASSERT_NE(import_info_.memory_profile_ordinal,
              AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    const uint64_t granularity = capabilities.mapping_granularity;
    ASSERT_GE(granularity, kWordCount * 4u);
    ASSERT_LE(granularity, kByteLength / 5);
    // An unaligned logical prefix combined with a nonzero logical source
    // offset must select the native page once, not omit or repeat the prefix.
    const uint64_t prefix = granularity / 2;
    ASSERT_NO_FATAL_FAILURE(OpenSharedBuffer(prefix, kByteLength - prefix));
    const amdf_memory_reservation_access_t access = {
        device_, 0, memory_access_.requirements.access};
    const amdf_memory_reservation_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
        .structure_size = sizeof(create),
        .byte_length = capabilities.reservation_granularity,
        .access_count = 1,
        .accesses = &access,
    };
    ASSERT_GE(create.byte_length, 2 * granularity);
    ASSERT_EQ(
        api_->memory_reservation_create(instance_, &create, &reservation_),
        AMDF_STATUS_OK);
    amdf_memory_reservation_access_info_t projection = {};
    projection.type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO;
    projection.structure_size = sizeof(projection);
    ASSERT_EQ(api_->memory_reservation_query_access_info(reservation_, 0,
                                                         &projection),
              AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(PrepareNativeCopy(
        family_ordinal, projection.address, projection.address + granularity));
    if (publication == MappingPublication::kQueueOrdered) {
      amdf_kernel_queue_info_t queue_info = {};
      queue_info.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO;
      queue_info.structure_size = sizeof(queue_info);
      ASSERT_EQ(api_->kernel_queue_query_info(queue_, &queue_info),
                AMDF_STATUS_OK);
      if ((queue_info.capabilities &
           AMDF_KERNEL_QUEUE_CAPABILITY_VIRTUAL_MEMORY) == 0) {
        GTEST_SKIP() << "native queue-ordered mappings are not advertised";
      }
    }
    ComPtr<ID3D12Resource> upload;
    ComPtr<ID3D12Resource> readback;
    ASSERT_NO_FATAL_FAILURE(CreateStaging(D3D12_HEAP_TYPE_UPLOAD, &upload));
    ASSERT_NO_FATAL_FAILURE(CreateStaging(D3D12_HEAP_TYPE_READBACK, &readback));
    std::vector<uint32_t> expected(kByteLength / 4);
    for (uint32_t i = 0; i < expected.size(); ++i) {
      expected[i] = 0x5a13b87du ^ (i * 0x9e3779b9u);
    }
    void* pointer = nullptr;
    const D3D12_RANGE no_read = {0, 0};
    ASSERT_TRUE(SUCCEEDED(upload->Map(0, &no_read, &pointer)));
    std::memcpy(pointer, expected.data(), kByteLength);
    const D3D12_RANGE full_range = {0, kByteLength};
    upload->Unmap(0, &full_range);
    ASSERT_NO_FATAL_FAILURE(CopyD3D12(upload.Get(), buffer_.Get()));

    // Both generations use the same command storage and queue. Foreign API
    // allocations and submissions never intervene between mapping and access.
    for (uint32_t generation = 0; generation < 2; ++generation) {
      const uint64_t source_offset = (generation == 0 ? 1 : 4) * granularity;
      const uint64_t destination_offset =
          (generation == 0 ? 2 : 3) * granularity;
      const std::array<amdf_memory_mapping_update_t, 2> updates = {{
          {
              .target_byte_offset = 0,
              .byte_length = granularity,
              .memory = imported_memory_,
              .access = access.access,
              .source_byte_offset = source_offset - prefix,
          },
          {
              .target_byte_offset = granularity,
              .byte_length = granularity,
              .memory = imported_memory_,
              .access = access.access,
              .source_byte_offset = destination_offset - prefix,
          },
      }};
      if (publication == MappingPublication::kQueueOrdered) {
        ASSERT_EQ(api_->kernel_queue_update_memory(
                      queue_, reservation_, 0, updates.size(), updates.data(),
                      &pending_submission_),
                  AMDF_STATUS_OK);
      } else {
        ASSERT_EQ(api_->memory_reservation_update(
                      reservation_, 0, updates.size(), updates.data()),
                  AMDF_STATUS_OK);
      }
      ASSERT_NO_FATAL_FAILURE(ExecuteNativeCopy());
      for (uint32_t i = 0; i < kWordCount; ++i) {
        expected[destination_offset / 4 + i] = expected[source_offset / 4 + i];
      }
      ASSERT_NO_FATAL_FAILURE(CopyD3D12(buffer_.Get(), readback.Get()));
      ASSERT_TRUE(SUCCEEDED(readback->Map(0, &full_range, &pointer)));
      const int difference = std::memcmp(pointer, expected.data(), kByteLength);
      readback->Unmap(0, &no_read);
      ASSERT_EQ(difference, 0) << "generation=" << generation;
    }
    ASSERT_NO_FATAL_FAILURE(ReleaseReservation());
    ASSERT_NO_FATAL_FAILURE(DestroyImportedStorage());
    // The foreign owner remains usable after checked removal and native close.
    ASSERT_NO_FATAL_FAILURE(CopyD3D12(buffer_.Get(), readback.Get()));
    ASSERT_TRUE(SUCCEEDED(readback->Map(0, &full_range, &pointer)));
    const int difference = std::memcmp(pointer, expected.data(), kByteLength);
    readback->Unmap(0, &no_read);
    ASSERT_EQ(difference, 0);
  }

  // Typed import request using the fixture's complete consumer set.
  amdf_memory_import_info_t import_info_ = {};
  // Public memory that owns native access to the foreign resource.
  amdf_memory_t* imported_memory_ = nullptr;
  // Stable alias range borrowing the imported storage until checked removal.
  amdf_memory_reservation_t* reservation_ = nullptr;
  // Consuming removal failure preserves the import and native dependencies.
  bool reservation_cleanup_failed_ = false;
  // Number of source transports consumed by successful imports.
  uint32_t release_count_ = 0;
  // Factory used to locate the matching foreign API adapter.
  ComPtr<IDXGIFactory4> factory_;
  // Static command recipe selected by the PM4-consuming subclass.
  const Pm4CommandProfile* pm4_profile_ = nullptr;
  // Foreign adapter independently admitted against the libamdf device.
  ComPtr<IDXGIAdapter1> adapter_;
  // Foreign device owning the resource and verification queues.
  ComPtr<ID3D12Device> d3d_;
  // Shared committed or placed buffer used for generation and verification.
  ComPtr<ID3D12Resource> buffer_;
  // Explicit backing owner for the placed-buffer transport.
  ComPtr<ID3D12Heap> heap_;
  // Source sharing handle until consumed by import or fixture teardown.
  HANDLE shared_ = nullptr;
  // Prepared native address translated to resource-relative byte zero.
  uint64_t address_ = 0;
  // Native queue owning the copy stream's publication.
  amdf_kernel_queue_t* queue_ = nullptr;
  // Native command backing retained until retirement.
  amdf_memory_t* command_memory_ = nullptr;
  // CPU view used only to construct commands, never shared data.
  amdf_host_mapping_t* command_mapping_ = nullptr;
  // Actual encoded stream length in bytes.
  uint64_t command_byte_length_ = 0;
  // Accepted submission requiring retirement before teardown.
  uint64_t pending_submission_ = 0;
};

// Only the native round-trip consumes the fixed PM4 cache recipe. Handle-only
// interop cases retain the broader endpoint selection of the base fixture.
class D3D12Pm4MemoryInteropTest : public D3D12MemoryInteropTest {
 protected:
  amdf_status_t MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                 bool* out_matches) override {
    amdf_gpu_endpoint_info_t gpu_info = {};
    gpu_info.type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO;
    gpu_info.structure_size = sizeof(gpu_info);
    amdf_status_t status = gpu_api_->endpoint_query_info(endpoint, &gpu_info);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    pm4_profile_ = Pm4CommandProfile::Find(gpu_info);
    if (!pm4_profile_) {
      *out_matches = false;
      return AMDF_STATUS_OK;
    }
    amdf_endpoint_info_t endpoint_info = {};
    endpoint_info.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
    endpoint_info.structure_size = sizeof(endpoint_info);
    status = api_->endpoint_query_info(endpoint, &endpoint_info);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    constexpr amdf_queue_roles_t kRoles =
        AMDF_QUEUE_ROLE_TRANSFER | AMDF_QUEUE_ROLE_CACHE_CONTROL;
    constexpr amdf_cache_operations_t kOperations =
        AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
        AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM;
    for (uint32_t ordinal = 0; ordinal < endpoint_info.queue_family_count;
         ++ordinal) {
      amdf_queue_family_info_t family = {};
      family.type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO;
      family.structure_size = sizeof(family);
      status =
          api_->endpoint_query_queue_family_info(endpoint, ordinal, &family);
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 &&
          family.format_version == AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1 &&
          (family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_KERNEL) &&
          (family.roles & kRoles) == kRoles &&
          (family.format_features &
           AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR) &&
          (family.cache_operations & kOperations) == kOperations &&
          (family.cache_transition_kinds &
           AMDF_CACHE_TRANSITION_KINDS_GLOBAL)) {
        family_ordinal_ = family.ordinal;
        *out_matches = true;
        return AMDF_STATUS_OK;
      }
    }
    *out_matches = false;
    return AMDF_STATUS_OK;
  }

  // Exact kernel-publication family selected before native device activation.
  uint32_t family_ordinal_ = UINT32_MAX;
};

TEST_P(D3D12Pm4MemoryInteropTest, ForeignNativeForeignRoundTrip) {
  RunRoundTrip(family_ordinal_);
}

TEST_P(D3D12Pm4MemoryInteropTest, ReplacesImportedSlicesOnAnExistingQueue) {
  RunMappingReplacement(family_ordinal_, MappingPublication::kHostCompleted);
}

TEST_P(D3D12Pm4MemoryInteropTest, QueueOrdersImportedSliceReplacement) {
  RunMappingReplacement(family_ordinal_, MappingPublication::kQueueOrdered);
}

TEST_P(D3D12MemoryInteropTest, ImportOwnsBackingAfterSourceRelease) {
  ASSERT_NO_FATAL_FAILURE(CreateSharedBuffer());
  ASSERT_NO_FATAL_FAILURE(OpenSharedBuffer());
  buffer_.Reset();
  heap_.Reset();
}

TEST_P(D3D12MemoryInteropTest, RejectsPhysicalPaddingOutsideLogicalBuffer) {
  amdf_external_memory_t source = Transport();
  source.source_byte_offset = kByteLength - 1;
  source.byte_length = 2;
  ASSERT_NO_FATAL_FAILURE(ExpectRejected(source));
  ASSERT_NO_FATAL_FAILURE(OpenSharedBuffer());
}

TEST_P(D3D12MemoryInteropTest, RejectsUnsupportedHostVisibility) {
  import_info_.required_flags |= AMDF_MEMORY_FLAG_HOST_VISIBLE;
  ASSERT_NO_FATAL_FAILURE(ExpectRejected(Transport()));
  import_info_.required_flags &= ~AMDF_MEMORY_FLAG_HOST_VISIBLE;
  ASSERT_NO_FATAL_FAILURE(OpenSharedBuffer());
}

TEST_P(D3D12MemoryInteropTest, RejectsIncompatiblePayloads) {
  amdf_external_memory_t source = Transport();
  HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  ASSERT_NE(event, nullptr);
  source.payload.native_handle = event;
  ExpectRejected(source);
  EXPECT_TRUE(CloseHandle(event));

  source = Transport();
  source.type = GetParam() == AMDF_EXTERNAL_MEMORY_TYPE_D3D12_HEAP
                    ? AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE
                    : AMDF_EXTERNAL_MEMORY_TYPE_D3D12_HEAP;
  ASSERT_NO_FATAL_FAILURE(ExpectRejected(source));
  source = Transport();

  D3D12_HEAP_PROPERTIES properties = {};
  properties.Type = D3D12_HEAP_TYPE_DEFAULT;
  properties.CreationNodeMask = 1;
  properties.VisibleNodeMask = 1;
  D3D12_RESOURCE_DESC description = {};
  description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  description.Width = 32;
  description.Height = 32;
  description.DepthOrArraySize = 1;
  description.MipLevels = 1;
  description.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  description.SampleDesc.Count = 1;
  description.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  ComPtr<ID3D12Resource> texture;
  ASSERT_TRUE(SUCCEEDED(d3d_->CreateCommittedResource(
      &properties, D3D12_HEAP_FLAG_SHARED, &description,
      D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&texture))));
  HANDLE shared_texture = nullptr;
  ASSERT_TRUE(SUCCEEDED(d3d_->CreateSharedHandle(
      texture.Get(), nullptr, GENERIC_ALL, nullptr, &shared_texture)));
  source.payload.native_handle = shared_texture;
  ExpectRejected(source);
  EXPECT_TRUE(CloseHandle(shared_texture));
  ASSERT_NO_FATAL_FAILURE(OpenSharedBuffer());
}

TEST_P(D3D12MemoryInteropTest, PreservesUnalignedLogicalRangeOnReexport) {
  amdf_external_memory_t source = Transport();
  source.source_byte_offset = 13;
  source.byte_length = 117;
  ASSERT_EQ(api_->memory_import(local_scope_, &import_info_, &source,
                                &imported_memory_, nullptr),
            AMDF_STATUS_OK);
  amdf_memory_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
  info.structure_size = sizeof(info);
  ASSERT_EQ(api_->memory_query_info(imported_memory_, &info), AMDF_STATUS_OK);
  EXPECT_EQ(info.source_byte_offset, 13u);
  EXPECT_EQ(info.byte_length, 117u);
  EXPECT_EQ(info.alignment, 1u);
  amdf_memory_export_info_t export_info = {};
  export_info.type = AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO;
  export_info.structure_size = sizeof(export_info);
  export_info.external_memory_type =
      GetParam() == AMDF_EXTERNAL_MEMORY_TYPE_D3D12_HEAP
          ? AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE
          : AMDF_EXTERNAL_MEMORY_TYPE_D3D12_HEAP;
  export_info.byte_offset = 7;
  export_info.byte_length = 19;
  amdf_external_memory_t exported = {};
  EXPECT_EQ(api_->memory_export(imported_memory_, &export_info, &exported),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  const amdf_external_memory_t empty = {};
  EXPECT_EQ(std::memcmp(&exported, &empty, sizeof(exported)), 0);
  export_info.external_memory_type = GetParam();
  ASSERT_EQ(api_->memory_export(imported_memory_, &export_info, &exported),
            AMDF_STATUS_OK);
  EXPECT_EQ(exported.source_byte_offset, 20u);
  EXPECT_EQ(exported.byte_length, 19u);
  api_->external_memory_release(&exported);
}

const char* TransportName(
    const ::testing::TestParamInfo<amdf_external_memory_type_t>& info) {
  return info.param == AMDF_EXTERNAL_MEMORY_TYPE_D3D12_HEAP ? "SharedHeap"
                                                            : "CommittedBuffer";
}

INSTANTIATE_TEST_SUITE_P(
    Transport, D3D12MemoryInteropTest,
    ::testing::Values(AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE,
                      AMDF_EXTERNAL_MEMORY_TYPE_D3D12_HEAP),
    TransportName);
INSTANTIATE_TEST_SUITE_P(
    Transport, D3D12Pm4MemoryInteropTest,
    ::testing::Values(AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE,
                      AMDF_EXTERNAL_MEMORY_TYPE_D3D12_HEAP),
    TransportName);

}  // namespace
