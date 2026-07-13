// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_AMDXDNA_SHIM_WINDOWS_MCDM_KMT_API_H_
#define IREE_AMD_AIE_DRIVER_AMDXDNA_SHIM_WINDOWS_MCDM_KMT_API_H_

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
// clang-format off
// Order-sensitive Windows SDK headers: windows.h must precede the D3DKMT
// headers (d3dkmthk.h pulls d3dukmdt.h, which needs windows.h types). Do not
// let clang-format sort these.
#include <windows.h>
#include <winternl.h>

#include <d3dkmthk.h>
// clang-format on

#include <cstddef>
#include <cstdint>

namespace iree::hal::amdxdna::mcdm {

constexpr size_t kMaxRetainedAdapterHandles = 256;

struct Error {
  char message[512] = {};
};

const char* ErrorMessage(const Error* error);

enum class BufferKind {
  host_only,
  cacheable,
  execbuf,
};

struct BufferKindInfo {
  const char* name;
  uint32_t private_type;
  uint32_t xcl_flags;
};

BufferKindInfo GetBufferKindInfo(BufferKind kind);

struct KmtApi {
  PFND3DKMT_ENUMADAPTERS3 enum_adapters3 = nullptr;
  PFND3DKMT_QUERYADAPTERINFO query_adapter_info = nullptr;
  PFND3DKMT_OPENADAPTERFROMLUID open_adapter_from_luid = nullptr;
  PFND3DKMT_CLOSEADAPTER close_adapter = nullptr;
  PFND3DKMT_CREATEDEVICE create_device = nullptr;
  PFND3DKMT_DESTROYDEVICE destroy_device = nullptr;
  PFND3DKMT_CREATEPAGINGQUEUE create_paging_queue = nullptr;
  PFND3DKMT_DESTROYPAGINGQUEUE destroy_paging_queue = nullptr;
  PFND3DKMT_CREATEALLOCATION2 create_allocation2 = nullptr;
  PFND3DKMT_DESTROYALLOCATION2 destroy_allocation2 = nullptr;
  PFND3DKMT_MAPGPUVIRTUALADDRESS map_gpu_virtual_address = nullptr;
  PFND3DKMT_FREEGPUVIRTUALADDRESS free_gpu_virtual_address = nullptr;
  PFND3DKMT_MAKERESIDENT make_resident = nullptr;
  PFND3DKMT_LOCK2 lock2 = nullptr;
  PFND3DKMT_UNLOCK2 unlock2 = nullptr;
  PFND3DKMT_INVALIDATECACHE invalidate_cache = nullptr;
  PFND3DKMT_CREATECONTEXTVIRTUAL create_context_virtual = nullptr;
  PFND3DKMT_DESTROYCONTEXT destroy_context = nullptr;
  PFND3DKMT_CREATEHWQUEUE create_hw_queue = nullptr;
  PFND3DKMT_DESTROYHWQUEUE destroy_hw_queue = nullptr;
  PFND3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU wait_from_gpu = nullptr;
  PFND3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait_from_cpu = nullptr;
  PFND3DKMT_SUBMITCOMMANDTOHWQUEUE submit_command_to_hw_queue = nullptr;

  bool Load(Error* out_error);
};

struct Adapter {
  D3DKMT_HANDLE handle = 0;
  LUID luid = {};
  D3DKMT_HANDLE retained_handles[kMaxRetainedAdapterHandles] = {};
  size_t retained_handle_count = 0;
};

struct Device {
  D3DKMT_HANDLE adapter = 0;
  D3DKMT_HANDLE retained_adapter_handles[kMaxRetainedAdapterHandles] = {};
  size_t retained_adapter_handle_count = 0;
  D3DKMT_HANDLE device = 0;
  D3DKMT_HANDLE paging_queue = 0;
  D3DKMT_HANDLE paging_sync_object = 0;
  void* paging_fence_cpu = nullptr;
};

struct Buffer {
  BufferKind kind = BufferKind::host_only;
  // Logical BO size exposed to the HAL/runtime.
  uint64_t size = 0;
  // Actual GPU VA reservation size. XRT often requests a non-page-sized BO
  // and maps the page-rounded allocation returned by the driver.
  uint64_t mapped_size = 0;
  D3DKMT_HANDLE allocation = 0;
  D3DKMT_HANDLE resource = 0;
  D3DGPU_VIRTUAL_ADDRESS gpu_va = 0;
  void* cpu_ptr = nullptr;
  UINT64 paging_fence_value = 0;
};

struct Context {
  D3DKMT_HANDLE context = 0;
  D3DKMT_HANDLE hw_queue = 0;
  D3DKMT_HANDLE progress_fence = 0;
  void* progress_fence_cpu = nullptr;
  D3DGPU_VIRTUAL_ADDRESS progress_fence_gpu = 0;
  uint64_t next_fence_id = 1;
  // Driver writeback at the primary context private blob +0x40. XRT folds this
  // cookie into the 64 MiB command-aperture allocation private flags.
  uint32_t command_aperture_cookie = 0;
  // Path B (hwqueue_aie4-style per-dispatch submit) state. The completion ring
  // is the firmware "status_bo": a 0x332b allocation (NOT a plain host BO) of
  // 8-byte slots that the firmware writes command completion state into. It is
  // allocated lazily on first Path B submit. Mirrors xrt_core's ring at
  // hwqueue+0xa0 (FUN_1800261f0: 0x332b status_bo, CreateResource|CreateShared,
  // Lock2). The 0x268/0x68 command carries this ring's ALLOCATION HANDLE at
  // +0x28 and the CPU slot address at +0x38.
  Buffer completion_ring;
  D3DKMT_HANDLE completion_ring_resource = 0;
  bool completion_ring_ready = false;
  uint32_t completion_ring_offset = 0;
  uint64_t next_command_id = 1;
};

struct CommandAperture {
  uint64_t allocation_size = 0;
  uint64_t gpu_va_size = 0;
  // Small CPU-locked command BO mapped over the 64 MiB command window at the
  // VA expected by the context blob.
  D3DKMT_HANDLE allocation = 0;
  D3DKMT_HANDLE gpu_allocation = 0;
  D3DKMT_HANDLE cleanup_allocation = 0;
  D3DKMT_HANDLE resource = 0;
  D3DKMT_HANDLE gpu_resource = 0;
  D3DGPU_VIRTUAL_ADDRESS gpu_va = 0;
  void* cpu_ptr = nullptr;
  void* gpu_cpu_ptr = nullptr;
  uint64_t cpu_ptr_size = 0;
  // Control-code view inside the 64 MiB aperture BO at gpu_va + 0x80000. The
  // working XRT path does not create a separate code allocation; it copies the
  // transaction binary into the aperture allocation at this offset.
  D3DKMT_HANDLE code_allocation = 0;
  D3DKMT_HANDLE code_resource = 0;
  D3DGPU_VIRTUAL_ADDRESS code_gpu_va = 0;
  void* code_cpu_ptr = nullptr;
  uint64_t code_size = 0;
};

struct PathBChainSubmitInfo {
  D3DGPU_VIRTUAL_ADDRESS descriptor_gpu_va = 0;
  uint32_t descriptor_bytes = 0;
  uint32_t command_count = 0;
  uint32_t first_child_opcode = 0;
};

struct PathBPendingSubmit {
  uint64_t fence_id = 0;
  uint8_t* slot_cpu = nullptr;
  uint32_t slot_offset = 0;
  volatile uint32_t* packet_header = nullptr;
  Buffer exec_buffer;
  Buffer ring;
};

bool FindNpuAdapter(const KmtApi& api, Adapter* out_adapter, Error* out_error);

bool CreateDevice(const KmtApi& api, const Adapter& adapter, Device* out_device,
                  Error* out_error);

void DestroyDevice(const KmtApi& api, Device* device);

bool CreateBuffer(const KmtApi& api, const Device& device, BufferKind kind,
                  uint64_t size, Buffer* out_buffer, Error* out_error);

bool SyncBuffer(const KmtApi& api, const Device& device, const Buffer& buffer,
                uint64_t offset, uint64_t length, Error* out_error);

bool SyncCommandApertureCode(const KmtApi& api, const Device& device,
                             const CommandAperture& aperture, uint64_t offset,
                             uint64_t length, Error* out_error);

bool RefreshCommandApertureGpuMapping(const KmtApi& api, const Device& device,
                                      CommandAperture* aperture,
                                      Error* out_error);

bool EnsureCommandApertureGpuMapping(const KmtApi& api, const Device& device,
                                     CommandAperture* aperture,
                                     Error* out_error);

bool ReleaseCommandApertureGpuMapping(const KmtApi& api, const Device& device,
                                      CommandAperture* aperture,
                                      Error* out_error);

bool RefreshBufferCpuMapping(const KmtApi& api, const Device& device,
                             Buffer* buffer, Error* out_error);

bool WaitForBufferResidency(const KmtApi& api, const Device& device,
                             const Context& context, const Buffer& buffer,
                            const char* label, Error* out_error);

void DestroyBuffer(const KmtApi& api, const Device& device, Buffer* buffer);

bool CreateContext(const KmtApi& api, const Device& device,
                   const uint8_t* private_data, size_t private_data_size,
                   Context* out_context, Error* out_error);

void DestroyContext(const KmtApi& api, const Device& device, Context* context);

bool CreateCommandAperture(const KmtApi& api, const Device& device,
                           const Context& context,
                           CommandAperture* out_aperture, Error* out_error);

bool SubmitAndWaitCommandAperture(const KmtApi& api, const Device& device,
                                  Context* context, CommandAperture* aperture,
                                  Error* out_error);

bool SubmitAndWaitPathBSetup(const KmtApi& api, const Device& device,
                             Context* context, CommandAperture* aperture,
                             const void* aperture_payload,
                             size_t aperture_payload_size, Error* out_error);

bool SubmitPathBApertureSync(const KmtApi& api, const Device& device,
                             Context* context, const CommandAperture& aperture,
                             uint64_t offset, bool wait_for_cpu,
                             Error* out_error);

// Path B: per-dispatch hwqueue_aie4-style submit. Reproduces the xrt_core
// submission recovered from disassembly: reserve an 8-byte completion-ring
// slot, build the 0x268 private packet (state, completion slot, ERT packet
// copied inline at +0x68), and submit the exec BO via SubmitCommandToHwQueue.
// `ert_packet`/`ert_bytes` are the command BO's ERT packet; `packet_header` is
// updated with the firmware completion state read back from the ring slot.
bool SubmitAndWaitPathB(const KmtApi& api, const Device& device,
                        Context* context, const Buffer& exec_buffer,
                        const void* ert_packet, uint32_t ert_bytes,
                        uint32_t command_state, uint32_t* packet_header,
                        Error* out_error);

// Path B parent ERT_CMD_CHAIN submit. This is the same completion protocol as
// SubmitAndWaitPathB, but uses the recovered xrt_core opcode-6 private
// envelope: +0x48 points at the aperture-resident child descriptor block, +0x50
// carries descriptor byte count, +0x54 child count, and +0x58 the first child
// ERT opcode.
bool SubmitAndWaitPathBChain(const KmtApi& api, const Device& device,
                             Context* context, const Buffer& exec_buffer,
                             const void* ert_packet, uint32_t ert_bytes,
                             const PathBChainSubmitInfo& chain_info,
                             uint32_t* packet_header, Error* out_error);

bool SubmitPathBChain(const KmtApi& api, const Device& device, Context* context,
                      const Buffer& exec_buffer, const void* ert_packet,
                      uint32_t ert_bytes,
                      const PathBChainSubmitInfo& chain_info,
                      uint32_t* packet_header, PathBPendingSubmit* out_pending,
                      Error* out_error);

// Single-dispatch path-B issue (no wait); the async counterpart of
// SubmitAndWaitPathB. Returns the in-flight fence token in `out_pending`; wait
// for it with WaitForPathBSubmits.
bool SubmitPathB(const KmtApi& api, const Device& device, Context* context,
                 const Buffer& exec_buffer, const void* ert_packet,
                 uint32_t ert_bytes, uint32_t command_state,
                 uint32_t* packet_header, PathBPendingSubmit* out_pending,
                 Error* out_error);

bool WaitForPathBSubmits(const KmtApi& api, const Device& device,
                         Context* context, PathBPendingSubmit* pending,
                         size_t pending_count, Error* out_error);

// Non-blocking completion poll for an issued (but not yet waited) path-B
// submit: true once the HW progress fence has reached pending.fence_id.
bool IsPathBSubmitComplete(const Context& context,
                           const PathBPendingSubmit& pending);

// Stale probe path retained for diagnostics only. The working XRT IREE matmul
// capture uses opcode 2/5/9 setup packets, not opcode 10.
void DestroyCommandAperture(const KmtApi& api, const Device& device,
                            CommandAperture* aperture);

}  // namespace iree::hal::amdxdna::mcdm

#endif  // IREE_AMD_AIE_DRIVER_AMDXDNA_SHIM_WINDOWS_MCDM_KMT_API_H_
