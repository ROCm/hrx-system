// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/sparse_mapping_test.h"

#include <drm/amdgpu_drm.h>
#include <drm/drm.h>
#include <emmintrin.h>
#include <fcntl.h>
#include <linux/kfd_ioctl.h>
#include <linux/sync_file.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <span>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/endpoint_profile.h"
#include "libamdf/src/gpu/umd/kfd/buffer.h"
#include "libamdf/src/gpu/umd/kfd/device.h"
#include "libamdf/src/gpu/umd/user_queue.h"
#include "libamdf/src/platform/endpoint.h"
#include "libamdf/src/platform/instance.h"
#include "libamdf/src/wait.h"

namespace amdf::testing {
namespace {

// A failed native transition may retain accepted GPU work. Stop this test
// process instead of guessing which backing is safe to release. Success takes
// the complete, checked release path below; process exit is not its oracle.
void Require(bool condition, const char* operation) {
  if (condition) {
    return;
  }
  ADD_FAILURE() << operation << ": errno=" << errno;
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

class KfdSparseMappingBackend final : public SparseMappingBackend {
 public:
  explicit KfdSparseMappingBackend(amdf_native_lifetime_t lifetime)
      : lifetime_(lifetime) {}

  bool Prepare() {
    const amdf_status_t status = amdf_gpu_umd_instance_prepare(
        &instance_, lifetime_, amdf_allocator_system());
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      RequireStatus(amdf_gpu_umd_instance_destroy(instance_),
                    "release unsupported KFD instance");
      instance_ = nullptr;
      return false;
    }
    RequireStatus(status, "prepare KFD instance");
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
      RequireStatus(amdf_gpu_umd_create_endpoint_profile(
                        endpoint_, amdf_allocator_system(), &profile),
                    "query GPU profile");
      for (uint32_t i = 0; i < profile->queue_family_count; ++i) {
        const auto& candidate = profile->queue_families[i];
        if (candidate.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA &&
            (candidate.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_USER) &&
            (candidate.roles & AMDF_QUEUE_ROLE_TRANSFER)) {
          queue_family_ = candidate;
          break;
        }
      }
      amdf_free(amdf_allocator_system(), profile);
      if (queue_family_.command_type != 0) {
        std::fprintf(stderr, "endpoint=%s lifetime=%u\n", summary.name,
                     lifetime_);
        break;
      }
      RequireStatus(amdf_platform_endpoint_close(endpoint_), "close endpoint");
      endpoint_ = nullptr;
    }
    if (!endpoint_) {
      RequireStatus(amdf_platform_instance_destroy(platform_),
                    "release platform without a native SDMA user queue");
      platform_ = nullptr;
      RequireStatus(amdf_gpu_umd_instance_destroy(instance_),
                    "release unused KFD instance");
      instance_ = nullptr;
      return false;
    }
    return true;
  }

  void Initialize(uint32_t backing_count, uint64_t reservation_byte_length,
                  uint64_t payload_byte_length) override {
    struct utsname kernel = {};
    Require(uname(&kernel) == 0, "query kernel identity");
    ::testing::Test::RecordProperty("kernel_release", kernel.release);
    ::testing::Test::RecordProperty("native_lifetime", lifetime_);

    amdf_platform_instance_lock_native(platform_);
    amdf_gpu_umd_device_result_t result = {};
    RequireStatus(amdf_gpu_umd_device_create(instance_, endpoint_,
                                             amdf_allocator_system(), lifetime_,
                                             &device_, &result),
                  "create KFD device");
    amdf_platform_instance_unlock_native(platform_);
    std::fprintf(
        stderr, "native_gpu=%u gc=%u.%u.%u sdma=%u.%u.%u kernel=%s\n",
        device_->topology.gpu_id, device_->topology.gc_ip.major,
        device_->topology.gc_ip.minor, device_->topology.gc_ip.revision,
        device_->topology.sdma.ip.major, device_->topology.sdma.ip.minor,
        device_->topology.sdma.ip.revision, kernel.release);
    format_features = queue_family_.format_features;
    reservation_byte_length_ = reservation_byte_length;
    for (auto& address : reservation_addresses) {
      void* pointer = mmap(nullptr, reservation_byte_length_, PROT_NONE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
      Require(pointer != MAP_FAILED, "reserve sparse VA");
      address = reinterpret_cast<uintptr_t>(pointer);
      Require(address >= device_->topology.virtual_address.begin &&
                  address + reservation_byte_length_ <=
                      device_->topology.virtual_address.end,
              "reservation within native GPU address envelope");
    }

    const bool local = (device_->topology.memory_features &
                        AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY) != 0;
    backing_.resize(backing_count);
    backing_addresses.resize(backing_count);
    for (uint32_t i = 0; i < backing_count; ++i) {
      const auto buffer = Allocate(
          kSparseSlabByteLength,
          local ? KFD_IOC_ALLOC_MEM_FLAGS_VRAM : KFD_IOC_ALLOC_MEM_FLAGS_GTT,
          AMDF_GPU_KFD_BUFFER_HOST_ACCESS_NONE, &backing_[i]);
      backing_addresses[i] = buffer.device_address;
    }
    const uint64_t stride = (payload_byte_length + device_->page_size - 1) &
                            ~(uint64_t{device_->page_size} - 1);
    const uint64_t transport_byte_length = 3 * stride + 2 * device_->page_size;
    const auto transport =
        Allocate(transport_byte_length,
                 KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT,
                 AMDF_GPU_KFD_BUFFER_HOST_ACCESS_MAPPED, &transport_);
    upload = static_cast<uint32_t*>(transport.host_pointer);
    readback = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(transport.host_pointer) + stride);
    capture = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(transport.host_pointer) + 2 * stride);
    upload_address = transport.device_address;
    readback_address = transport.device_address + stride;
    capture_address = transport.device_address + 2 * stride;
    completion_address = transport.device_address + 3 * stride;
    completion_ = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(transport.host_pointer) + 3 * stride);
    *completion_ = 0;
    nop_address_ = completion_address + device_->page_size;
    std::memset(static_cast<uint8_t*>(transport.host_pointer) + 3 * stride +
                    device_->page_size,
                0, device_->page_size);

    // One import per retained native allocation. KFD continues owning each
    // original allocation and its eviction/restore participation throughout.
    bo_entries_.reserve(backing_.size() + 1);
    for (auto* buffer : backing_) {
      bo_entries_.push_back({Import(buffer), 0});
    }
    bo_entries_.push_back({Import(transport_), 0});

    drm_amdgpu_info query = {};
    query.return_pointer = reinterpret_cast<uintptr_t>(&ip_);
    query.return_size = sizeof(ip_);
    query.query = AMDGPU_INFO_HW_IP_INFO;
    query.query_hw_ip.type = AMDGPU_HW_IP_DMA;
    Ioctl(DRM_IOCTL_AMDGPU_INFO, &query, "query native DMA IB limits");
    Require(ip_.available_rings && ip_.ib_size_alignment &&
                ip_.ib_size_alignment <= device_->page_size &&
                ip_.ib_size_alignment % 4 == 0,
            "native NOP IB format");
    ring_ = static_cast<uint32_t>(__builtin_ctz(ip_.available_rings));
    drm_amdgpu_ctx context = {};
    context.in.op = AMDGPU_CTX_OP_ALLOC_CTX;
    context.in.priority = AMDGPU_CTX_PRIORITY_NORMAL;
    Ioctl(DRM_IOCTL_AMDGPU_CTX, &context, "create mapping-publication context");
    context_ = context.out.alloc.ctx_id;

    amdf_gpu_umd_user_queue_create_info_t create = {
        .command_type = queue_family_.command_type,
        .format_version = queue_family_.format_version,
        .priority = AMDF_QUEUE_PRIORITY_NORMAL,
        .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
        .required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
        .roles = queue_family_.roles,
        .ring_byte_length = queue_family_.minimum_ring_byte_length,
    };
    amdf_gpu_umd_user_queue_result_t queue_result = {};
    RequireStatus(amdf_gpu_umd_user_queue_create(device_, &create, &queue_,
                                                 &queue_result),
                  "create persistent SDMA queue");
    RequireStatus(amdf_gpu_umd_user_queue_map(queue_, nullptr, &queue_mapping_,
                                              &queue_view_),
                  "map persistent SDMA queue");
    ring_byte_length_ = queue_result.ring_byte_length;
    command_byte_capacity = ring_byte_length_ - 32;
    Require(queue_view_.index_bits == 64 && queue_view_.doorbell_bits == 64,
            "native SDMA producer word widths");
    std::fprintf(stderr,
                 "prepared: reservations=2 reservation_bytes=%" PRIu64
                 " slabs=%u slab_bytes=%" PRIu64 " payload_bytes=%" PRIu64
                 " placement=%s\n",
                 reservation_byte_length_, backing_count, kSparseSlabByteLength,
                 payload_byte_length, local ? "VRAM" : "GTT");
  }

  void Attach(std::span<const SparseMapping> mappings) override {
    std::vector<drm_amdgpu_bo_list_entry> sources;
    sources.reserve(mappings.size() + 1);
    sources.push_back(bo_entries_.back());
    for (const auto& mapping : mappings) {
      sources.push_back(bo_entries_[mapping.backing]);
    }
    amdf_gpu_kfd_device_lock_memory(device_, AMDF_GPU_KFD_MEMORY_PRIORITY_MAP);
    for (const auto& mapping : mappings) {
      Update(AMDGPU_VA_OP_REPLACE, bo_entries_[mapping.backing].bo_handle,
             reservation_addresses[mapping.reservation] + mapping.byte_offset,
             kSparseSlabByteLength);
    }
    PublishMappings(sources);
    amdf_gpu_kfd_device_unlock_memory(device_);
  }

  void Detach(std::span<const SparseMapping> mappings) override {
    amdf_gpu_kfd_device_lock_memory(device_, AMDF_GPU_KFD_MEMORY_PRIORITY_MAP);
    for (const auto& mapping : mappings) {
      Update(AMDGPU_VA_OP_CLEAR, 0,
             reservation_addresses[mapping.reservation] + mapping.byte_offset,
             kSparseSlabByteLength);
    }
    // Removal needs no old source handles or reconstruction of the installed
    // mapping set. Only the publication command's own transport is submitted.
    PublishMappings(std::span(&bo_entries_.back(), 1));
    amdf_gpu_kfd_device_unlock_memory(device_);
  }

  void Execute(std::span<const uint32_t> words,
               uint32_t completion_value) override {
    Require(words.size_bytes() < ring_byte_length_, "command stream fits ring");
    auto* ring = reinterpret_cast<uint8_t*>(queue_view_.ring_address);
    const uint64_t position = frontier_ & (ring_byte_length_ - 1);
    const uint64_t tail = ring_byte_length_ - position;
    const uint64_t first = std::min<uint64_t>(tail, words.size_bytes());
    std::memcpy(ring + position, words.data(), first);
    if (first != words.size_bytes()) {
      std::memcpy(ring, reinterpret_cast<const uint8_t*>(words.data()) + first,
                  words.size_bytes() - first);
    }
    frontier_ += words.size_bytes();
    __atomic_store_n(
        reinterpret_cast<uint64_t*>(queue_view_.write_index_address), frontier_,
        __ATOMIC_RELEASE);
    // Publish command and index stores before the write-only MMIO notification.
    _mm_sfence();
    __atomic_store_n(reinterpret_cast<uint64_t*>(queue_view_.doorbell_address),
                     frontier_, __ATOMIC_RELEASE);
    const auto completion_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (__atomic_load_n(completion_, __ATOMIC_ACQUIRE) != completion_value) {
      Require(std::chrono::steady_clock::now() < completion_deadline,
              "timed out observing sparse mapping copy completion");
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    amdf_wait_deadline_t deadline = {};
    RequireStatus(
        amdf_wait_deadline_initialize(10'000'000'000, 50'000, &deadline),
        "initialize queue retirement deadline");
    RequireStatus(
        amdf_gpu_umd_user_queue_wait_consumed(queue_, frontier_, &deadline),
        "retire submitted SDMA ring storage");
  }

  void Release() override {
    // Clear the complete reservations, including holes, without enumerating
    // their installed mappings. Completion precedes backing release while the
    // same queue is alive; queue teardown cannot supply this guarantee.
    amdf_gpu_kfd_device_lock_memory(device_, AMDF_GPU_KFD_MEMORY_PRIORITY_MAP);
    for (uint64_t address : reservation_addresses) {
      Update(AMDGPU_VA_OP_CLEAR, 0, address, reservation_byte_length_);
    }
    PublishMappings(std::span(&bo_entries_.back(), 1));
    amdf_gpu_kfd_device_unlock_memory(device_);
    for (size_t i = 0; i < backing_.size(); ++i) {
      drm_gem_close release = {
          .handle = bo_entries_[i].bo_handle,
      };
      Ioctl(DRM_IOCTL_GEM_CLOSE, &release, "close retained slab GEM handle");
      RequireStatus(amdf_gpu_kfd_buffer_destroy(backing_[i]),
                    "release retained slab");
    }
    RequireStatus(amdf_gpu_umd_user_queue_mapping_destroy(queue_mapping_),
                  "release queue mapping");
    RequireStatus(amdf_gpu_umd_user_queue_destroy(queue_),
                  "release persistent queue");
    drm_amdgpu_ctx context = {};
    context.in.op = AMDGPU_CTX_OP_FREE_CTX;
    context.in.ctx_id = context_;
    Ioctl(DRM_IOCTL_AMDGPU_CTX, &context,
          "release mapping-publication context");
    drm_gem_close release = {
        .handle = bo_entries_.back().bo_handle,
    };
    Ioctl(DRM_IOCTL_GEM_CLOSE, &release, "close transport GEM handle");
    RequireStatus(amdf_gpu_kfd_buffer_destroy(transport_), "release transport");
    for (uint64_t address : reservation_addresses) {
      Require(munmap(reinterpret_cast<void*>(address),
                     reservation_byte_length_) == 0,
              "release sparse reservation");
    }
    RequireStatus(amdf_gpu_umd_device_destroy(device_),
                  "destroy native device");
    RequireStatus(amdf_gpu_umd_instance_destroy(instance_),
                  "destroy native instance");
    RequireStatus(amdf_platform_endpoint_close(endpoint_), "close endpoint");
    RequireStatus(amdf_platform_instance_destroy(platform_),
                  "destroy platform");
    std::fprintf(stderr, "checked final detach and native release complete\n");
  }

 private:
  amdf_gpu_kfd_buffer_result_t Allocate(
      uint64_t byte_length, uint32_t flags,
      amdf_gpu_kfd_buffer_host_access_t host_access,
      amdf_gpu_kfd_buffer_t** out_buffer) {
    amdf_gpu_kfd_buffer_create_info_t create = {
        .native_flags = flags | AMDF_GPU_KFD_ALLOC_MEM_FLAGS_WRITABLE,
        .byte_length = byte_length,
        .alignment = device_->page_size,
        .host_access = host_access,
    };
    amdf_gpu_kfd_buffer_result_t result = {};
    RequireStatus(
        amdf_gpu_kfd_buffer_create(device_, &create, out_buffer, &result),
        "allocate retained native backing");
    return result;
  }

  uint32_t Import(amdf_gpu_kfd_buffer_t* buffer) {
    kfd_ioctl_export_dmabuf_args export_args = {};
    export_args.handle = amdf_gpu_kfd_buffer_handle(buffer);
    export_args.flags = O_CLOEXEC;
    Require(
        ioctl(device_->descriptor, AMDKFD_IOC_EXPORT_DMABUF, &export_args) == 0,
        "export KFD backing");
    drm_prime_handle import_args = {
        .fd = static_cast<int>(export_args.dmabuf_fd),
    };
    Ioctl(DRM_IOCTL_PRIME_FD_TO_HANDLE, &import_args,
          "import backing into exact VM");
    Require(close(import_args.fd) == 0, "close temporary DMA-BUF transport");
    return import_args.handle;
  }

  void Ioctl(unsigned long request, void* arguments, const char* operation) {
    Require(ioctl(device_->render_descriptor, request, arguments) == 0,
            operation);
  }

  void Update(uint32_t operation, uint32_t handle, uint64_t address,
              uint64_t byte_length) {
    drm_amdgpu_gem_va update = {
        .handle = handle,
        .operation = operation,
        .flags = AMDGPU_VM_DELAY_UPDATE | AMDGPU_VM_PAGE_READABLE |
                 AMDGPU_VM_PAGE_WRITEABLE,
    };
    update.va_address = address;
    update.map_size = byte_length;
    Ioctl(DRM_IOCTL_AMDGPU_GEM_VA, &update, "change sparse mapping");
  }

  void PublishMappings(std::span<const drm_amdgpu_bo_list_entry> sources) {
    // Realize the new sources' delayed PTEs and the VM's freed mappings/PDEs,
    // then inspect execution status before KFD invalidation below. This list
    // contains the current update's sources and the fixed publication IB,
    // never a walk of old mappings or resources referenced by UMQ commands.
    drm_amdgpu_cs_chunk_ib ib = {};
    ib.va_start = nop_address_;
    ib.ib_bytes = ip_.ib_size_alignment;
    ib.ip_type = AMDGPU_HW_IP_DMA;
    ib.ring = ring_;
    drm_amdgpu_bo_list_in list = {
        .operation = AMDGPU_BO_LIST_OP_CREATE,
        .bo_number = static_cast<uint32_t>(sources.size()),
        .bo_info_size = sizeof(sources[0]),
    };
    list.bo_info_ptr = reinterpret_cast<uintptr_t>(sources.data());
    std::array<drm_amdgpu_cs_chunk, 2> chunks = {};
    chunks[0].chunk_id = AMDGPU_CHUNK_ID_IB;
    chunks[0].length_dw = sizeof(ib) / 4;
    chunks[0].chunk_data = reinterpret_cast<uintptr_t>(&ib);
    chunks[1].chunk_id = AMDGPU_CHUNK_ID_BO_HANDLES;
    chunks[1].length_dw = sizeof(list) / 4;
    chunks[1].chunk_data = reinterpret_cast<uintptr_t>(&list);
    const uint64_t pointers[] = {reinterpret_cast<uintptr_t>(&chunks[0]),
                                 reinterpret_cast<uintptr_t>(&chunks[1])};
    drm_amdgpu_cs submit = {};
    submit.in.ctx_id = context_;
    submit.in.num_chunks = 2;
    submit.in.chunks = reinterpret_cast<uintptr_t>(pointers);
    _mm_mfence();
    Ioctl(DRM_IOCTL_AMDGPU_CS, &submit, "submit mapping-publication NOP");
    drm_amdgpu_fence_to_handle export_fence = {};
    export_fence.in.fence.ctx_id = context_;
    export_fence.in.fence.ip_type = AMDGPU_HW_IP_DMA;
    export_fence.in.fence.ring = ring_;
    export_fence.in.fence.seq_no = submit.out.handle;
    export_fence.in.what = AMDGPU_FENCE_TO_HANDLE_GET_SYNC_FILE_FD;
    Ioctl(DRM_IOCTL_AMDGPU_FENCE_TO_HANDLE, &export_fence,
          "export publication fence");
    const int descriptor = static_cast<int>(export_fence.out.handle);
    drm_amdgpu_wait_cs wait = {};
    wait.in.handle = submit.out.handle;
    wait.in.timeout = UINT64_MAX;
    wait.in.ip_type = AMDGPU_HW_IP_DMA;
    wait.in.ctx_id = context_;
    wait.in.ring = ring_;
    Ioctl(DRM_IOCTL_AMDGPU_WAIT_CS, &wait, "wait for mapping publication");
    Require(wait.out.status == 0, "mapping-publication fence signaled");
    sync_file_info info = {};
    Require(ioctl(descriptor, SYNC_IOC_FILE_INFO, &info) == 0,
            "query mapping-publication execution status");
    Require(info.status == 1, "mapping-publication execution succeeded");
    Require(close(descriptor) == 0, "close publication fence");

    // CPU page-table updates can advance tlb_seq without creating a compute
    // PASID flush fence. A DRM scheduling fence alone does not invalidate an
    // already-running KFD queue's translations. Reaffirming this existing KFD
    // mapping runs the native synchronization/PASID-flush path: already-mapped
    // attachments are skipped, without adding another attachment or backing.
    // KFD does not propagate the flush result; this healthy-path regression
    // cannot establish error-bearing invalidation or detach semantics.
    const uint32_t gpu_id = device_->topology.gpu_id;
    kfd_ioctl_map_memory_to_gpu_args ready = {};
    ready.handle = amdf_gpu_kfd_buffer_handle(transport_);
    ready.device_ids_array_ptr = reinterpret_cast<uintptr_t>(&gpu_id);
    ready.n_devices = 1;
    Require(
        ioctl(device_->descriptor, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &ready) == 0,
        "complete KFD mapping readiness");
    Require(ready.n_success == 1, "KFD mapping readiness completed for device");
  }

  // Native lifetime policy exercised by this process-local test instance.
  amdf_native_lifetime_t lifetime_;
  // Cold platform discovery/connection owner.
  amdf_platform_instance_t* platform_ = nullptr;
  // Open immutable native endpoint, outliving the device.
  amdf_platform_endpoint_t* endpoint_ = nullptr;
  // Discovered native SDMA user queue contract for the selected endpoint.
  amdf_queue_family_info_t queue_family_ = {};
  // Shared KFD connection and per-device VM owner.
  amdf_gpu_umd_instance_t* instance_ = nullptr;
  // Native execution domain borrowed by buffers and queues.
  amdf_gpu_umd_device_t* device_ = nullptr;
  // Fixed native queue retained across every mapping transition.
  amdf_gpu_umd_user_queue_t* queue_ = nullptr;
  // Borrowed CPU view of queue-owned producer storage.
  amdf_gpu_umd_user_queue_mapping_t* queue_mapping_ = nullptr;
  // Direct producer addresses and word widths.
  amdf_gpu_umd_user_queue_mapping_result_t queue_view_ = {};
  // Actual queue ring capacity in bytes.
  uint64_t ring_byte_length_ = 0;
  // Next monotonic SDMA ring publication position in bytes.
  uint64_t frontier_ = 0;
  // Exact byte length reserved at each sparse root.
  uint64_t reservation_byte_length_ = 0;
  // Retained physical slabs with unchanged original aliases.
  std::vector<amdf_gpu_kfd_buffer_t*> backing_;
  // Coherent upload/readback/capture/completion/NOP storage.
  amdf_gpu_kfd_buffer_t* transport_ = nullptr;
  // Coherent CPU completion word written after all SDMA transfers.
  uint32_t* completion_ = nullptr;
  // Stable all-zero IB used only for mapping publication.
  uint64_t nop_address_ = 0;
  // Imported handles used for cold mapping publication, never UMQ submission.
  std::vector<drm_amdgpu_bo_list_entry> bo_entries_;
  // Native DRM scheduling context for mapping-publication IBs.
  uint32_t context_ = 0;
  // Queried native DMA ring ordinal.
  uint32_t ring_ = 0;
  // Queried alignment requirements for native DMA IBs.
  drm_amdgpu_info_hw_ip ip_ = {};
};

class KfdSparseMappingTest
    : public ::testing::TestWithParam<amdf_native_lifetime_t> {};

TEST_P(KfdSparseMappingTest, ReplacesRetainedBackingBeforeImmediateAccess) {
  KfdSparseMappingBackend backend(GetParam());
  if (!backend.Prepare()) {
    GTEST_SKIP() << "No KFD native SDMA user queue supports lifetime "
                 << GetParam()
                 << "; mapping replacement has not been exercised";
  }
  ExerciseSparseMappingReplacement(backend);
}

INSTANTIATE_TEST_SUITE_P(NativeLifetime, KfdSparseMappingTest,
                         ::testing::Values(AMDF_NATIVE_LIFETIME_PROCESS,
                                           AMDF_NATIVE_LIFETIME_INSTANCE));

}  // namespace
}  // namespace amdf::testing
