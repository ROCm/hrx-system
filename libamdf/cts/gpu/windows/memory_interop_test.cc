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

#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "libamdf/cts/gpu/gpu_device_fixture.h"

using Microsoft::WRL::ComPtr;

namespace {
constexpr uint64_t kByteLength = 65536;
constexpr uint64_t kSourceOffset = 4096;
constexpr uint64_t kDestinationOffset = 8192;
constexpr uint32_t kWordCount = 128;

uint32_t MakePm4Header(uint32_t opcode, uint32_t count) {
  return (3u << 30) | (opcode << 8) | ((count - 2) << 16);
}

void AppendSystemBarrier(std::vector<uint32_t>& words) {
  enum : uint32_t {
    kEventWriteOpcode = 0x46,
    kAcquireMemoryOpcode = 0x58,
    kComputeShaderPartialFlush = 7 | (4 << 8),
    kConservativeGcrControl = (3 << 0) | (1 << 4) | (1 << 5) | (1 << 7) |
                              (1 << 8) | (1 << 9) | (1 << 14) | (1 << 15),
  };
  words.insert(words.end(),
               {MakePm4Header(kEventWriteOpcode, 2), kComputeShaderPartialFlush,
                MakePm4Header(kAcquireMemoryOpcode, 8), 0, UINT32_MAX, 0xff, 0,
                0, 0x0a, kConservativeGcrControl});
}

class D3D12MemoryInteropTest : public GpuDeviceFixture {
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
    value.type = AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE;
    value.payload.native_handle = shared_;
    value.source_byte_offset = kSourceOffset;
    value.byte_length = kDestinationOffset + kWordCount * 4 - kSourceOffset;
    return value;
  }

  static void AMDF_CALL ReleaseSource(void* user_data,
                                      amdf_external_memory_type_t type,
                                      amdf_external_memory_payload_t payload) {
    EXPECT_EQ(type, AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE);
    EXPECT_TRUE(CloseHandle(payload.native_handle));
    ++*static_cast<uint32_t*>(user_data);
  }

  void DestroyImportedStorage() {
    if (imported_memory_ != nullptr) {
      ASSERT_EQ(api_->memory_destroy(std::exchange(imported_memory_, nullptr)),
                AMDF_STATUS_OK);
    }
  }

  void TearDown() override {
    if (pending_submission_ != 0) {
      ADD_FAILURE() << "Native work has not retired; retaining its backing.";
      return;
    }
    if (queue_ != nullptr) {
      ASSERT_EQ(api_->kernel_queue_destroy(queue_), AMDF_STATUS_OK);
      queue_ = nullptr;
    }
    if (command_mapping_ != nullptr) {
      ASSERT_EQ(api_->host_mapping_destroy(command_mapping_), AMDF_STATUS_OK);
      command_mapping_ = nullptr;
    }
    if (command_memory_ != nullptr) {
      ASSERT_EQ(api_->memory_destroy(std::exchange(command_memory_, nullptr)),
                AMDF_STATUS_OK);
    }
    ASSERT_NO_FATAL_FAILURE(DestroyImportedStorage());
    if (shared_ != nullptr) {
      ASSERT_TRUE(CloseHandle(shared_));
      shared_ = nullptr;
    }
    buffer_.Reset();
    d3d_.Reset();
    adapter_.Reset();
    factory_.Reset();
    GpuDeviceFixture::TearDown();
  }

  void CreateSharedBuffer() {
    if (shared_ != nullptr) {
      ASSERT_TRUE(CloseHandle(shared_));
      shared_ = nullptr;
    }
    buffer_.Reset();
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC buffer = {};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = kByteLength;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ASSERT_TRUE(SUCCEEDED(d3d_->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_SHARED, &buffer, D3D12_RESOURCE_STATE_COMMON,
        nullptr, IID_PPV_ARGS(&buffer_))));
    ASSERT_TRUE(SUCCEEDED(d3d_->CreateSharedHandle(
        buffer_.Get(), nullptr, GENERIC_ALL, nullptr, &shared_)));
  }

  void OpenSharedBuffer() {
    amdf_external_memory_t transport = Transport();
    transport.release = ReleaseSource;
    transport.release_user_data = &release_count_;
    const uint32_t releases_before = release_count_;
    const amdf_status_t status = api_->memory_import(
        local_scope_, &import_info_, &transport, &imported_memory_);
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
    ASSERT_EQ(info.source_byte_offset, kSourceOffset);
    ASSERT_EQ(info.byte_length,
              kDestinationOffset + kWordCount * 4 - kSourceOffset);
    ASSERT_GE(info.native_allocation_byte_length, kByteLength);
    ASSERT_EQ(api_->memory_query_address(imported_memory_, 0,
                                         AMDF_MEMORY_ADDRESS_GPU, &address_),
              AMDF_STATUS_OK);
    address_ -= kSourceOffset;
  }

  void CopyD3D12(ID3D12Resource* source, ID3D12Resource* destination) {
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
    ID3D12CommandList* lists[] = {commands.Get()};
    queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> fence;
    ASSERT_TRUE(SUCCEEDED(
        d3d_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))));
    ASSERT_TRUE(SUCCEEDED(queue->Signal(fence.Get(), 1)));
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ASSERT_NE(event, nullptr);
    const HRESULT status = fence->SetEventOnCompletion(1, event);
    if (FAILED(status)) {
      EXPECT_TRUE(CloseHandle(event));
      FAIL() << "D3D12 fence registration: " << std::hex << status;
    }
    const DWORD wait = WaitForSingleObject(event, INFINITE);
    EXPECT_TRUE(CloseHandle(event));
    ASSERT_EQ(wait, WAIT_OBJECT_0);
    ASSERT_EQ(fence->GetCompletedValue(), 1u);
    ASSERT_TRUE(SUCCEEDED(d3d_->GetDeviceRemovedReason()));
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

  void PrepareNativeCopy() {
    amdf_endpoint_info_t endpoint = {};
    endpoint.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
    endpoint.structure_size = sizeof(endpoint);
    ASSERT_EQ(api_->endpoint_query_info(endpoint_, &endpoint), AMDF_STATUS_OK);
    uint32_t family_ordinal = UINT32_MAX;
    for (uint32_t i = 0; i < endpoint.queue_family_count; ++i) {
      amdf_queue_family_info_t family = {};
      family.type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO;
      family.structure_size = sizeof(family);
      ASSERT_EQ(api_->endpoint_query_queue_family_info(endpoint_, i, &family),
                AMDF_STATUS_OK);
      if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 &&
          family.format_version == AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1 &&
          (family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_KERNEL) &&
          (family.format_features &
           AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR)) {
        family_ordinal = i;
        break;
      }
    }
    ASSERT_NE(family_ordinal, UINT32_MAX);
    amdf_gpu_kernel_queue_create_info_t queue = {};
    queue.type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO;
    queue.structure_size = sizeof(queue);
    queue.queue_family_ordinal = family_ordinal;
    ASSERT_NO_FATAL_FAILURE(CheckVisibility(family_ordinal));
    ASSERT_EQ(gpu_api_->kernel_queue_create(device_, &queue, &queue_),
              AMDF_STATUS_OK);
    const amdf_memory_device_access_t access = {
        device_,
        {.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                   AMDF_MEMORY_ACCESS_EXECUTE,
         .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
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
    ASSERT_EQ(api_->memory_create(system_scope_, &create, &command_memory_),
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
    std::vector<uint32_t> words;
    AppendSystemBarrier(words);
    enum : uint32_t {
      kCopyDataOpcode = 0x40,
      kSourceTcL2 = 2 << 0,
      kTargetTcL2 = 2 << 8,
      kWaitForConfirmation = 1 << 20,
    };
    for (uint32_t i = 0; i < kWordCount; ++i) {
      const uint64_t source = address_ + kSourceOffset + i * 4;
      const uint64_t destination = address_ + kDestinationOffset + i * 4;
      words.insert(
          words.end(),
          {MakePm4Header(kCopyDataOpcode, 6),
           kSourceTcL2 | kTargetTcL2 | kWaitForConfirmation,
           static_cast<uint32_t>(source), static_cast<uint32_t>(source >> 32),
           static_cast<uint32_t>(destination),
           static_cast<uint32_t>(destination >> 32)});
    }
    AppendSystemBarrier(words);
    size_t padding = 8 - words.size() % 8;
    if (padding == 1) {
      padding += 8;
    }
    words.push_back(MakePm4Header(0x10, static_cast<uint32_t>(padding)));
    words.resize(words.size() + padding - 1);
    command_byte_length_ = words.size() * sizeof(uint32_t);
    std::memcpy(mapped.pointer, words.data(), command_byte_length_);
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
    query.external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE;
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
    EXPECT_EQ(std::memcmp(&prospective, &actual, sizeof(actual)), 0);
    EXPECT_EQ(actual.release.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    EXPECT_EQ(actual.release.operation, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM);
    EXPECT_EQ(actual.acquire.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    EXPECT_EQ(actual.acquire.operation,
              AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM);
  }

  void ExpectRejected(amdf_external_memory_t source) {
    source.release = ReleaseSource;
    source.release_user_data = &release_count_;
    const amdf_external_memory_t original = source;
    const uint32_t releases_before = release_count_;
    auto* const sentinel = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
    amdf_memory_t* output = sentinel;
    const amdf_status_t status =
        api_->memory_import(local_scope_, &import_info_, &source, &output);
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
                                      AMDF_TIMEOUT_INFINITE, 0),
              AMDF_STATUS_OK);
    pending_submission_ = 0;
  }

  void ReexportAndDetach() {
    // Drop the original API resource before producing the return transport.
    buffer_.Reset();
    amdf_memory_export_info_t export_info = {};
    export_info.type = AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO;
    export_info.structure_size = sizeof(export_info);
    export_info.external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE;
    export_info.byte_length =
        kDestinationOffset + kWordCount * 4 - kSourceOffset;
    amdf_external_memory_t exported = {};
    ASSERT_EQ(api_->memory_export(imported_memory_, &export_info, &exported),
              AMDF_STATUS_OK);
    ASSERT_EQ(exported.source_byte_offset, kSourceOffset);
    ASSERT_EQ(exported.byte_length, export_info.byte_length);
    ASSERT_NO_FATAL_FAILURE(DestroyImportedStorage());
    const HRESULT opened = d3d_->OpenSharedHandle(
        exported.payload.native_handle, IID_PPV_ARGS(&buffer_));
    api_->external_memory_release(&exported);
    ASSERT_TRUE(SUCCEEDED(opened)) << std::hex << opened;
    const auto description = buffer_->GetDesc();
    ASSERT_EQ(description.Dimension, D3D12_RESOURCE_DIMENSION_BUFFER);
    ASSERT_EQ(description.Width, kByteLength);
  }

  void RunRoundTrip() {
    ASSERT_NO_FATAL_FAILURE(CreateSharedBuffer());
    ASSERT_NO_FATAL_FAILURE(OpenSharedBuffer());
    ASSERT_NO_FATAL_FAILURE(PrepareNativeCopy());
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
  // Typed import request using the fixture's complete consumer set.
  amdf_memory_import_info_t import_info_ = {};
  // Public memory that owns native access to the foreign resource.
  amdf_memory_t* imported_memory_ = nullptr;
  // Number of source transports consumed by successful imports.
  uint32_t release_count_ = 0;
  // Factory used to locate the matching foreign API adapter.
  ComPtr<IDXGIFactory4> factory_;
  // Foreign adapter independently admitted against the libamdf device.
  ComPtr<IDXGIAdapter1> adapter_;
  // Foreign device owning the resource and verification queues.
  ComPtr<ID3D12Device> d3d_;
  // Shared committed buffer used for generation and verification.
  ComPtr<ID3D12Resource> buffer_;
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

TEST_F(D3D12MemoryInteropTest, ImportOwnsBackingAfterSourceRelease) {
  ASSERT_NO_FATAL_FAILURE(CreateSharedBuffer());
  ASSERT_NO_FATAL_FAILURE(OpenSharedBuffer());
  buffer_.Reset();
}

TEST_F(D3D12MemoryInteropTest, ForeignNativeForeignRoundTrip) {
  RunRoundTrip();
}

TEST_F(D3D12MemoryInteropTest, RejectsPhysicalPaddingOutsideLogicalBuffer) {
  amdf_external_memory_t source = Transport();
  source.source_byte_offset = kByteLength - 1;
  source.byte_length = 2;
  ASSERT_NO_FATAL_FAILURE(ExpectRejected(source));
  ASSERT_NO_FATAL_FAILURE(OpenSharedBuffer());
}

TEST_F(D3D12MemoryInteropTest, RejectsUnsupportedHostVisibility) {
  import_info_.required_flags |= AMDF_MEMORY_FLAG_HOST_VISIBLE;
  ASSERT_NO_FATAL_FAILURE(ExpectRejected(Transport()));
  import_info_.required_flags &= ~AMDF_MEMORY_FLAG_HOST_VISIBLE;
  ASSERT_NO_FATAL_FAILURE(OpenSharedBuffer());
}

TEST_F(D3D12MemoryInteropTest, RejectsIncompatiblePayloads) {
  amdf_external_memory_t source = Transport();
  HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  ASSERT_NE(event, nullptr);
  source.payload.native_handle = event;
  ExpectRejected(source);
  EXPECT_TRUE(CloseHandle(event));

  D3D12_HEAP_DESC heap_info = {};
  heap_info.SizeInBytes = kByteLength;
  heap_info.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
  heap_info.Properties.CreationNodeMask = 1;
  heap_info.Properties.VisibleNodeMask = 1;
  heap_info.Flags = D3D12_HEAP_FLAG_SHARED;
  ComPtr<ID3D12Heap> heap;
  ASSERT_TRUE(SUCCEEDED(d3d_->CreateHeap(&heap_info, IID_PPV_ARGS(&heap))));
  HANDLE shared_heap = nullptr;
  ASSERT_TRUE(SUCCEEDED(d3d_->CreateSharedHandle(
      heap.Get(), nullptr, GENERIC_ALL, nullptr, &shared_heap)));
  source.payload.native_handle = shared_heap;
  ExpectRejected(source);
  EXPECT_TRUE(CloseHandle(shared_heap));

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
      &heap_info.Properties, D3D12_HEAP_FLAG_SHARED, &description,
      D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&texture))));
  HANDLE shared_texture = nullptr;
  ASSERT_TRUE(SUCCEEDED(d3d_->CreateSharedHandle(
      texture.Get(), nullptr, GENERIC_ALL, nullptr, &shared_texture)));
  source.payload.native_handle = shared_texture;
  ExpectRejected(source);
  EXPECT_TRUE(CloseHandle(shared_texture));
  ASSERT_NO_FATAL_FAILURE(OpenSharedBuffer());
}

TEST_F(D3D12MemoryInteropTest, PreservesUnalignedLogicalRangeOnReexport) {
  amdf_external_memory_t source = Transport();
  source.source_byte_offset = 13;
  source.byte_length = 117;
  ASSERT_EQ(api_->memory_import(local_scope_, &import_info_, &source,
                                &imported_memory_),
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
  export_info.external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE;
  export_info.byte_offset = 7;
  export_info.byte_length = 19;
  amdf_external_memory_t exported = {};
  ASSERT_EQ(api_->memory_export(imported_memory_, &export_info, &exported),
            AMDF_STATUS_OK);
  EXPECT_EQ(exported.source_byte_offset, 20u);
  EXPECT_EQ(exported.byte_length, 19u);
  api_->external_memory_release(&exported);
}

}  // namespace
