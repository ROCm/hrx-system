// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "kmt_api.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <limits>

namespace iree::hal::amdxdna::mcdm {
namespace {

constexpr NTSTATUS kStatusPending = static_cast<NTSTATUS>(0x00000103);
constexpr uint64_t kPageSize = 4096;
constexpr uint64_t kCommandApertureAllocationSize = 0x1000;
constexpr D3DGPU_VIRTUAL_ADDRESS kCommandApertureGpuVaBase = 0x04000000;
constexpr uint64_t kCommandApertureGpuVaSize = 0x04000000;
constexpr uint32_t kCommandAperturePrivateType = 0x332b;
constexpr uint32_t kQhdlSubmitPrivateSize = 0x268;
constexpr uint32_t kQhdlSubmitPacketOffset = 0x68;
constexpr uint32_t kQhdlCompletionSlotSize = 8;
constexpr uint32_t kSubmitPrivateQwords = 13;  // 104 bytes on current driver.
constexpr size_t kContextCommandApertureCookieOffset = 0x40;
constexpr UINT kMaxComputeAdapters = 256;
constexpr UINT kMaxDriverStorePathWarmupBytes = 4096;

template <typename Fn>
Fn ResolveKmtProc(const char* name) {
  HMODULE modules[] = {
      LoadLibraryW(L"win32u.dll"),
      LoadLibraryW(L"gdi32.dll"),
      LoadLibraryW(L"dxcore.dll"),
  };
  for (HMODULE module : modules) {
    if (!module) continue;
    FARPROC proc = GetProcAddress(module, name);
    if (proc) return reinterpret_cast<Fn>(proc);
  }
  return nullptr;
}

void SetError(Error* out_error, const char* message) {
  if (!out_error) return;
  std::snprintf(out_error->message, sizeof(out_error->message), "%s",
                message ? message : "unknown MCDM error");
}

void SetErrorFormat(Error* out_error, const char* fmt, ...) {
  if (!out_error) return;
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(out_error->message, sizeof(out_error->message), fmt, args);
  va_end(args);
  out_error->message[sizeof(out_error->message) - 1] = 0;
}

const char* NtStatusSuffix(NTSTATUS status) {
  return status == kStatusPending ? " (STATUS_PENDING)" : "";
}

bool CheckStatus(const char* call_name, NTSTATUS status, Error* out_error) {
  if (status == 0) return true;
  SetErrorFormat(out_error, "%s failed with 0x%08x%s", call_name,
                 static_cast<uint32_t>(status), NtStatusSuffix(status));
  return false;
}

bool CheckStatusOrPending(const char* call_name, NTSTATUS status,
                          Error* out_error) {
  if (status == 0 || status == kStatusPending) return true;
  return CheckStatus(call_name, status, out_error);
}

uint32_t Flags32(const D3DKMT_CREATEALLOCATIONFLAGS& flags) {
  uint32_t value = 0;
  static_assert(sizeof(value) <= sizeof(flags), "flag storage mismatch");
  std::memcpy(&value, &flags, sizeof(value));
  return value;
}

bool TrustFenceCompletionEnabled() { return true; }

bool CcccCompletionSlotEnabled() { return false; }

bool XrtLockTouchEnabled() {
  // TODO(hrx-mcdm): Revisit once the exact XRT rationale is documented. A
  // disabled probe passed numerics and the full benchmark on 2026-06-12, but
  // XRT touches the exec BO around path-B submit, so keep parity for now.
  return true;
}

void InitializeCompletionSlot(uint8_t* slot_cpu) {
  if (CcccCompletionSlotEnabled()) {
    std::memset(slot_cpu, 0xCC, kQhdlCompletionSlotSize);
    return;
  }
  std::memset(slot_cpu, 0, kQhdlCompletionSlotSize);
}

void WriteU32(uint8_t* data, size_t offset, uint32_t value) {
  std::memcpy(data + offset, &value, sizeof(value));
}

void WriteU64(uint8_t* data, size_t offset, uint64_t value) {
  std::memcpy(data + offset, &value, sizeof(value));
}

struct AllocPrivate {
  uint64_t reserved0 = 0;
  uint64_t requested_size = 0;
  uint64_t aligned_size = 0;
  uint32_t reserved1 = 0;
  uint32_t private_type = 0;
  uint32_t policy = 2;
  uint32_t reserved2 = 0;
  uint32_t xcl_flags = 0;
  uint32_t reserved3 = 0;
  uint64_t reserved4 = 0;
};

static_assert(sizeof(AllocPrivate) == 56,
              "Windows MCDM BO private packet must remain 56 bytes");

uint64_t AlignUpToPage(uint64_t value) {
  return (value + 4095u) & ~uint64_t{4095u};
}

void CloseAdapterHandles(const KmtApi& api, const D3DKMT_ADAPTERINFO* adapters,
                         UINT adapter_count, D3DKMT_HANDLE keep = 0) {
  for (UINT i = 0; i < adapter_count; ++i) {
    const D3DKMT_ADAPTERINFO& adapter = adapters[i];
    if (adapter.hAdapter == keep) continue;
    D3DKMT_CLOSEADAPTER close = {};
    close.hAdapter = adapter.hAdapter;
    api.close_adapter(&close);
  }
}

bool AppendRetainedAdapterHandle(Adapter* adapter, D3DKMT_HANDLE handle,
                                 Error* out_error) {
  if (adapter->retained_handle_count >= kMaxRetainedAdapterHandles) {
    SetErrorFormat(out_error,
                   "too many retained adapter handles (capacity=%zu)",
                   kMaxRetainedAdapterHandles);
    return false;
  }
  adapter->retained_handles[adapter->retained_handle_count++] = handle;
  return true;
}

bool SelectNpuAdapterFromOpenHandles(const KmtApi& api,
                                     const D3DKMT_ADAPTERINFO* adapters,
                                     UINT adapter_count, Adapter* out_adapter,
                                     bool stop_after_match = false) {
  Adapter exact;
  Adapter fallback;
  Adapter loose;
  for (UINT i = 0; i < adapter_count; ++i) {
    const D3DKMT_ADAPTERINFO& adapter = adapters[i];
    D3DKMT_DRIVER_DESCRIPTION description = {};
    D3DKMT_QUERYADAPTERINFO query = {};
    query.hAdapter = adapter.hAdapter;
    query.Type = KMTQAITYPE_DRIVER_DESCRIPTION;
    query.pPrivateDriverData = &description;
    query.PrivateDriverDataSize = sizeof(description);
    NTSTATUS status = api.query_adapter_info(&query);
    if (status != 0) continue;

    const wchar_t* text = description.DriverDescription;
    if (std::wcscmp(text, L"AMD XDNA(TM) NPU") == 0) {
      exact.handle = adapter.hAdapter;
      exact.luid = adapter.AdapterLuid;
      break;
    }
    if (!fallback.handle &&
        std::wcscmp(text, L"NPU Compute Accelerator Device") == 0) {
      fallback.handle = adapter.hAdapter;
      fallback.luid = adapter.AdapterLuid;
      if (stop_after_match) break;
    }
    if (!loose.handle && std::wcsstr(text, L"NPU") != nullptr) {
      loose.handle = adapter.hAdapter;
      loose.luid = adapter.AdapterLuid;
      if (stop_after_match) break;
    }
  }

  Adapter selected =
      exact.handle ? exact : (fallback.handle ? fallback : loose);
  if (!selected.handle) return false;
  *out_adapter = selected;
  return true;
}

bool EnumerateComputeAdapters(const KmtApi& api,
                              D3DKMT_ADAPTERINFO* out_adapters,
                              UINT adapter_capacity, UINT* out_adapter_count,
                              Error* out_error) {
  if (!out_adapters || !out_adapter_count || adapter_capacity == 0) {
    SetError(out_error, "EnumerateComputeAdapters called with invalid output");
    return false;
  }
  *out_adapter_count = 0;
  D3DKMT_ENUMADAPTERS3 enum_args = {};
  enum_args.Filter.IncludeComputeOnly = 1;
  NTSTATUS status = api.enum_adapters3(&enum_args);
  if (!CheckStatus("D3DKMTEnumAdapters3(count)", status, out_error)) {
    return false;
  }
  if (enum_args.NumAdapters == 0) {
    SetError(out_error, "D3DKMTEnumAdapters3 returned no adapters");
    return false;
  }
  if (enum_args.NumAdapters > adapter_capacity) {
    SetErrorFormat(out_error,
                   "D3DKMTEnumAdapters3 returned too many adapters: %u "
                   "(capacity=%u)",
                   enum_args.NumAdapters, adapter_capacity);
    return false;
  }

  UINT requested_adapter_count = enum_args.NumAdapters;
  enum_args.pAdapters = out_adapters;
  enum_args.NumAdapters = requested_adapter_count;
  status = api.enum_adapters3(&enum_args);
  if (!CheckStatus("D3DKMTEnumAdapters3(list)", status, out_error)) {
    return false;
  }
  if (enum_args.NumAdapters > adapter_capacity) {
    CloseAdapterHandles(api, out_adapters, adapter_capacity);
    SetErrorFormat(out_error,
                   "D3DKMTEnumAdapters3 list grew past capacity: %u "
                   "(capacity=%u)",
                   enum_args.NumAdapters, adapter_capacity);
    return false;
  }
  *out_adapter_count = enum_args.NumAdapters;
  return true;
}

void QueryDriverStorePathForWarmup(const KmtApi& api, D3DKMT_HANDLE adapter) {
  D3DDDI_QUERYREGISTRY_INFO query_info = {};
  query_info.QueryType = D3DDDI_QUERYREGISTRY_DRIVERSTOREPATH;

  D3DKMT_QUERYADAPTERINFO query = {};
  query.hAdapter = adapter;
  query.Type = KMTQAITYPE_QUERYREGISTRY;
  query.pPrivateDriverData = &query_info;
  query.PrivateDriverDataSize = sizeof(query_info);
  NTSTATUS status = api.query_adapter_info(&query);
  if (status != 0 ||
      query_info.Status != D3DDDI_QUERYREGISTRY_STATUS_BUFFER_OVERFLOW ||
      query_info.OutputValueSize == 0) {
    return;
  }

  if (query_info.OutputValueSize > kMaxDriverStorePathWarmupBytes) {
    return;
  }
  alignas(D3DDDI_QUERYREGISTRY_INFO)
      uint8_t buffer[sizeof(D3DDDI_QUERYREGISTRY_INFO) +
                     kMaxDriverStorePathWarmupBytes] = {};
  auto* expanded = reinterpret_cast<D3DDDI_QUERYREGISTRY_INFO*>(buffer);
  expanded->QueryType = D3DDDI_QUERYREGISTRY_DRIVERSTOREPATH;
  query.pPrivateDriverData = expanded;
  query.PrivateDriverDataSize = static_cast<UINT>(
      sizeof(D3DDDI_QUERYREGISTRY_INFO) + query_info.OutputValueSize);
  api.query_adapter_info(&query);
}

void QueryUmdriverPrivateWarmup(const KmtApi& api, D3DKMT_HANDLE adapter) {
  uint64_t private_data = 0;
  D3DKMT_QUERYADAPTERINFO query = {};
  query.hAdapter = adapter;
  query.Type = KMTQAITYPE_UMDRIVERPRIVATE;
  query.pPrivateDriverData = &private_data;
  query.PrivateDriverDataSize = sizeof(private_data);
  api.query_adapter_info(&query);
}

}  // namespace

BufferKindInfo GetBufferKindInfo(BufferKind kind) {
  switch (kind) {
    case BufferKind::host_only:
      return {"host_only", 0x3329, 0x20000000};
    case BufferKind::cacheable:
      return {"cacheable", 0x3323, 0x01000000};
    case BufferKind::execbuf:
      return {"execbuf", 0x3328, 0x80000000};
  }
  return {"host_only", 0x3329, 0x20000000};
}

const char* ErrorMessage(const Error* error) {
  return error && error->message[0] ? error->message : "unknown MCDM error";
}

bool KmtApi::Load(Error* out_error) {
  enum_adapters3 =
      ResolveKmtProc<PFND3DKMT_ENUMADAPTERS3>("D3DKMTEnumAdapters3");
  query_adapter_info =
      ResolveKmtProc<PFND3DKMT_QUERYADAPTERINFO>("D3DKMTQueryAdapterInfo");
  open_adapter_from_luid = ResolveKmtProc<PFND3DKMT_OPENADAPTERFROMLUID>(
      "D3DKMTOpenAdapterFromLuid");
  close_adapter = ResolveKmtProc<PFND3DKMT_CLOSEADAPTER>("D3DKMTCloseAdapter");
  create_device = ResolveKmtProc<PFND3DKMT_CREATEDEVICE>("D3DKMTCreateDevice");
  destroy_device =
      ResolveKmtProc<PFND3DKMT_DESTROYDEVICE>("D3DKMTDestroyDevice");
  create_paging_queue =
      ResolveKmtProc<PFND3DKMT_CREATEPAGINGQUEUE>("D3DKMTCreatePagingQueue");
  destroy_paging_queue =
      ResolveKmtProc<PFND3DKMT_DESTROYPAGINGQUEUE>("D3DKMTDestroyPagingQueue");
  create_allocation2 =
      ResolveKmtProc<PFND3DKMT_CREATEALLOCATION2>("D3DKMTCreateAllocation2");
  destroy_allocation2 =
      ResolveKmtProc<PFND3DKMT_DESTROYALLOCATION2>("D3DKMTDestroyAllocation2");
  map_gpu_virtual_address = ResolveKmtProc<PFND3DKMT_MAPGPUVIRTUALADDRESS>(
      "D3DKMTMapGpuVirtualAddress");
  free_gpu_virtual_address = ResolveKmtProc<PFND3DKMT_FREEGPUVIRTUALADDRESS>(
      "D3DKMTFreeGpuVirtualAddress");
  make_resident = ResolveKmtProc<PFND3DKMT_MAKERESIDENT>("D3DKMTMakeResident");
  lock2 = ResolveKmtProc<PFND3DKMT_LOCK2>("D3DKMTLock2");
  unlock2 = ResolveKmtProc<PFND3DKMT_UNLOCK2>("D3DKMTUnlock2");
  invalidate_cache =
      ResolveKmtProc<PFND3DKMT_INVALIDATECACHE>("D3DKMTInvalidateCache");
  create_context_virtual = ResolveKmtProc<PFND3DKMT_CREATECONTEXTVIRTUAL>(
      "D3DKMTCreateContextVirtual");
  destroy_context =
      ResolveKmtProc<PFND3DKMT_DESTROYCONTEXT>("D3DKMTDestroyContext");
  create_hw_queue =
      ResolveKmtProc<PFND3DKMT_CREATEHWQUEUE>("D3DKMTCreateHwQueue");
  destroy_hw_queue =
      ResolveKmtProc<PFND3DKMT_DESTROYHWQUEUE>("D3DKMTDestroyHwQueue");
  wait_from_gpu = ResolveKmtProc<PFND3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU>(
      "D3DKMTWaitForSynchronizationObjectFromGpu");
  wait_from_cpu = ResolveKmtProc<PFND3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU>(
      "D3DKMTWaitForSynchronizationObjectFromCpu");
  submit_command_to_hw_queue = ResolveKmtProc<PFND3DKMT_SUBMITCOMMANDTOHWQUEUE>(
      "D3DKMTSubmitCommandToHwQueue");

  if (enum_adapters3 && query_adapter_info && open_adapter_from_luid &&
      close_adapter && create_device && destroy_device && create_paging_queue &&
      destroy_paging_queue && create_allocation2 && destroy_allocation2 &&
      map_gpu_virtual_address && free_gpu_virtual_address && make_resident &&
      lock2 && unlock2 && invalidate_cache && create_context_virtual &&
      destroy_context && create_hw_queue && destroy_hw_queue && wait_from_gpu &&
      wait_from_cpu && submit_command_to_hw_queue) {
    return true;
  }

  SetError(out_error,
           "failed to resolve one or more required KMT entry points");
  return false;
}

bool FindNpuAdapter(const KmtApi& api, Adapter* out_adapter, Error* out_error) {
  D3DKMT_ADAPTERINFO adapters[kMaxComputeAdapters] = {};
  UINT adapter_count = 0;
  if (!EnumerateComputeAdapters(api, adapters, kMaxComputeAdapters,
                                &adapter_count, out_error)) {
    return false;
  }

  Adapter discovery_selection;
  if (!SelectNpuAdapterFromOpenHandles(api, adapters, adapter_count,
                                       &discovery_selection,
                                       /*stop_after_match=*/true)) {
    CloseAdapterHandles(api, adapters, adapter_count);
    SetError(out_error, "no NPU adapter was found");
    return false;
  }
  QueryDriverStorePathForWarmup(api, discovery_selection.handle);
  CloseAdapterHandles(api, adapters, adapter_count);

  // XRT does one discovery pass (including the DriverStore registry query),
  // closes those adapter handles, then enumerates again and keeps the fresh NPU
  // handle for D3DKMTCreateDevice.
  std::memset(adapters, 0, sizeof(adapters));
  adapter_count = 0;
  if (!EnumerateComputeAdapters(api, adapters, kMaxComputeAdapters,
                                &adapter_count, out_error)) {
    return false;
  }

  Adapter selected;
  if (!SelectNpuAdapterFromOpenHandles(api, adapters, adapter_count,
                                       &selected)) {
    CloseAdapterHandles(api, adapters, adapter_count);
    SetError(out_error, "no NPU adapter was found after warmup");
    return false;
  }
  bool found_selected = false;
  for (UINT i = 0; i < adapter_count; ++i) {
    const D3DKMT_ADAPTERINFO& adapter = adapters[i];
    if (adapter.hAdapter == selected.handle) {
      found_selected = true;
      continue;
    }
    if (!found_selected) {
      if (!AppendRetainedAdapterHandle(&selected, adapter.hAdapter,
                                       out_error)) {
        CloseAdapterHandles(api, adapters, adapter_count);
        return false;
      }
      continue;
    }
    D3DKMT_CLOSEADAPTER close = {};
    close.hAdapter = adapter.hAdapter;
    api.close_adapter(&close);
  }

  *out_adapter = selected;
  return true;
}

bool CreateDevice(const KmtApi& api, const Adapter& adapter, Device* out_device,
                  Error* out_error) {
  Device device = {};
  device.adapter = adapter.handle;
  device.retained_adapter_handle_count = adapter.retained_handle_count;
  std::memcpy(device.retained_adapter_handles, adapter.retained_handles,
              adapter.retained_handle_count * sizeof(D3DKMT_HANDLE));

  D3DKMT_CREATEDEVICE create_device = {};
  create_device.hAdapter = adapter.handle;
  NTSTATUS status = api.create_device(&create_device);
  if (!CheckStatus("D3DKMTCreateDevice", status, out_error)) {
    for (size_t i = 0; i < device.retained_adapter_handle_count; ++i) {
      D3DKMT_CLOSEADAPTER close = {};
      close.hAdapter = device.retained_adapter_handles[i];
      api.close_adapter(&close);
    }
    return false;
  }
  device.device = create_device.hDevice;

  D3DKMT_CREATEPAGINGQUEUE paging = {};
  paging.hDevice = device.device;
  paging.Priority = D3DDDI_PAGINGQUEUE_PRIORITY_NORMAL;
  paging.PhysicalAdapterIndex = 0;
  status = api.create_paging_queue(&paging);
  if (!CheckStatus("D3DKMTCreatePagingQueue", status, out_error)) {
    D3DKMT_DESTROYDEVICE destroy_device = {};
    destroy_device.hDevice = device.device;
    api.destroy_device(&destroy_device);
    for (size_t i = 0; i < device.retained_adapter_handle_count; ++i) {
      D3DKMT_CLOSEADAPTER close = {};
      close.hAdapter = device.retained_adapter_handles[i];
      api.close_adapter(&close);
    }
    return false;
  }

  device.paging_queue = paging.hPagingQueue;
  device.paging_sync_object = paging.hSyncObject;
  device.paging_fence_cpu = paging.FenceValueCPUVirtualAddress;
  QueryUmdriverPrivateWarmup(api, device.adapter);
  *out_device = device;
  return true;
}

void DestroyDevice(const KmtApi& api, Device* device) {
  if (!device) return;
  if (device->paging_queue) {
    D3DDDI_DESTROYPAGINGQUEUE destroy_paging = {};
    destroy_paging.hPagingQueue = device->paging_queue;
    api.destroy_paging_queue(&destroy_paging);
    device->paging_queue = 0;
  }
  if (device->device) {
    D3DKMT_DESTROYDEVICE destroy_device = {};
    destroy_device.hDevice = device->device;
    api.destroy_device(&destroy_device);
    device->device = 0;
  }
  if (device->adapter) {
    D3DKMT_CLOSEADAPTER close = {};
    close.hAdapter = device->adapter;
    api.close_adapter(&close);
    device->adapter = 0;
  }
  for (size_t i = 0; i < device->retained_adapter_handle_count; ++i) {
    D3DKMT_CLOSEADAPTER close = {};
    close.hAdapter = device->retained_adapter_handles[i];
    api.close_adapter(&close);
  }
  std::memset(device->retained_adapter_handles, 0,
              sizeof(device->retained_adapter_handles));
  device->retained_adapter_handle_count = 0;
}

// Block the CPU until a paging operation (Map/MakeResident) on the paging queue
// has actually completed. The async D3DKMT paging calls return STATUS_PENDING;
// without this wait a subsequent SubmitCommandToHwQueue can be rejected with
// 0xc01e0200 (STATUS_GRAPHICS_ALLOCATION_BUSY) because the allocation is still
// being paged in. XRT's xrt::bo residency is synchronous; this mirrors it.
bool WaitForPagingFenceCpu(const KmtApi& api, const Device& device,
                           UINT64 fence_value) {
  if (fence_value == 0) return true;
  D3DKMT_HANDLE objects[1] = {device.paging_sync_object};
  UINT64 values[1] = {fence_value};
  D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait = {};
  wait.hDevice = device.device;
  wait.ObjectCount = 1;
  wait.ObjectHandleArray = objects;
  wait.FenceValueArray = values;
  wait.hAsyncEvent = nullptr;
  NTSTATUS status = api.wait_from_cpu(&wait);
  return status == 0;
}

bool CreateBuffer(const KmtApi& api, const Device& device, BufferKind kind,
                  uint64_t size, Buffer* out_buffer, Error* out_error) {
  BufferKindInfo kind_info = GetBufferKindInfo(kind);
  // XRT's exec BO submit path asks the driver for the 4 KiB command BO plus
  // the 0x68-byte private prefix used by SubmitCommandToHwQueue. The HAL still
  // sees only the 4 KiB command capacity.
  constexpr uint64_t kPathBSubmitPrivatePrefixSize = 0x68;
  uint64_t requested_size = std::max<uint64_t>(size, 1);
  if (kind == BufferKind::execbuf) {
    // Match XRT exactly: it allocates a full 4 KiB command page + the 0x68
    // prefix (0x1068, 8 KiB aligned) for every exec BO, so each runlist BO
    // lands on its own page pair. We were allocating only the exact runlist
    // bytes (~0x148, single 4 KiB page), packing multiple parents' exec BOs
    // into one coherence granule and racing the firmware on multi-parent
    // re-runs. The HAL still sees only the logical command capacity.
    requested_size = std::max<uint64_t>(requested_size, 0x1000) +
                     kPathBSubmitPrivatePrefixSize;
  }
  uint64_t aligned_size = AlignUpToPage(requested_size);
  uint64_t size_pages = aligned_size / 4096;

  AllocPrivate alloc_private = {};
  alloc_private.requested_size = requested_size;
  alloc_private.aligned_size = aligned_size;
  // The working XRT matmul capture uses normal XRT BO private types for
  // dispatch buffers: 0x3329 host-only data BOs and 0x3328 exec BOs.
  alloc_private.private_type = kind_info.private_type;
  alloc_private.xcl_flags = kind_info.xcl_flags;

  D3DDDI_ALLOCATIONINFO2 alloc_info = {};
  alloc_info.pPrivateDriverData = &alloc_private;
  alloc_info.PrivateDriverDataSize = sizeof(alloc_private);

  D3DKMT_CREATEALLOCATION create = {};
  create.hDevice = device.device;
  create.NumAllocations = 1;
  create.pAllocationInfo2 = &alloc_info;

  NTSTATUS status = api.create_allocation2(&create);
  if (!CheckStatus("D3DKMTCreateAllocation2", status, out_error)) {
    return false;
  }

  Buffer buffer = {};
  buffer.kind = kind;
  buffer.size = size;
  buffer.mapped_size = aligned_size;
  buffer.allocation = alloc_info.hAllocation;
  buffer.resource = create.hResource;

  D3DDDI_MAPGPUVIRTUALADDRESS map = {};
  map.hPagingQueue = device.paging_queue;
  map.hAllocation = buffer.allocation;
  map.SizeInPages = size_pages;
  map.Protection.Write = 1;
  status = api.map_gpu_virtual_address(&map);
  if (!CheckStatusOrPending("D3DKMTMapGpuVirtualAddress", status, out_error)) {
    DestroyBuffer(api, device, &buffer);
    return false;
  }
  buffer.gpu_va = map.VirtualAddress;

  D3DKMT_HANDLE resident_allocs[1] = {buffer.allocation};
  D3DDDI_MAKERESIDENT resident = {};
  resident.hPagingQueue = device.paging_queue;
  resident.NumAllocations = 1;
  resident.AllocationList = resident_allocs;
  resident.Flags.CantTrimFurther = 1;
  resident.Flags.MustSucceed = 1;
  status = api.make_resident(&resident);
  if (!CheckStatusOrPending("D3DKMTMakeResident", status, out_error)) {
    DestroyBuffer(api, device, &buffer);
    return false;
  }
  // XRT records the paging fence returned from MakeResident but does not block
  // the CPU for every ordinary data/command BO before submit. Keep residency
  // asynchronous here as well; the BO is submitted only after the KMD has
  // accepted the make-resident request with MustSucceed|CantTrimFurther.
  buffer.paging_fence_value = 0;

  D3DKMT_LOCK2 lock = {};
  lock.hDevice = device.device;
  lock.hAllocation = buffer.allocation;
  status = api.lock2(&lock);
  if (!CheckStatus("D3DKMTLock2", status, out_error)) {
    DestroyBuffer(api, device, &buffer);
    return false;
  }
  buffer.cpu_ptr = lock.pData;

  // Exercise a minimal write to prove the mapping is CPU-accessible. The probe
  // syncs the range before teardown.
  if (buffer.cpu_ptr && buffer.size >= sizeof(uint32_t)) {
    uint32_t pattern = 0xa11e0000u | kind_info.private_type;
    std::memcpy(buffer.cpu_ptr, &pattern, sizeof(pattern));
  }

  *out_buffer = buffer;
  return true;
}

bool SyncBuffer(const KmtApi& api, const Device& device, const Buffer& buffer,
                uint64_t offset, uint64_t length, Error* out_error) {
  D3DKMT_INVALIDATECACHE invalidate = {};
  invalidate.hDevice = device.device;
  invalidate.hAllocation = buffer.allocation;
  invalidate.Offset = offset;
  invalidate.Length = length;
  NTSTATUS status = api.invalidate_cache(&invalidate);
  return CheckStatus("D3DKMTInvalidateCache", status, out_error);
}

bool LockCommandApertureGpuAfterBootstrap(const KmtApi& api,
                                          const Device& device,
                                          CommandAperture* aperture,
                                          Error* out_error);

bool SyncCommandApertureCode(const KmtApi& api, const Device& device,
                             const CommandAperture& aperture, uint64_t offset,
                             uint64_t length, Error* out_error) {
  if (!aperture.gpu_allocation) {
    SetError(out_error, "SyncCommandApertureCode called before aperture setup");
    return false;
  }
  D3DKMT_INVALIDATECACHE invalidate = {};
  invalidate.hDevice = device.device;
  invalidate.hAllocation = aperture.gpu_allocation;
  invalidate.Offset = offset;
  invalidate.Length = length;
  NTSTATUS status = api.invalidate_cache(&invalidate);
  return CheckStatus("D3DKMTInvalidateCache(command aperture code)", status,
                     out_error);
}

bool RefreshCommandApertureGpuMapping(const KmtApi& api, const Device& device,
                                      CommandAperture* aperture,
                                      Error* out_error) {
  if (!aperture || !aperture->gpu_allocation) {
    SetError(out_error,
             "RefreshCommandApertureGpuMapping called before aperture setup");
    return false;
  }
  // The path-B code BO is written through the host mapping after bootstrap.
  // Pair the cache invalidate with a lock transition so the MCDM miniport sees
  // the freshly staged instruction bytes before the opcode-3 submit consumes
  // the aperture.
  if (aperture->gpu_cpu_ptr) {
    D3DKMT_UNLOCK2 unlock = {};
    unlock.hDevice = device.device;
    unlock.hAllocation = aperture->gpu_allocation;
    NTSTATUS status = api.unlock2(&unlock);
    if (!CheckStatus("D3DKMTUnlock2(command aperture gpu refresh)", status,
                     out_error)) {
      return false;
    }
    aperture->gpu_cpu_ptr = nullptr;
    aperture->code_cpu_ptr = nullptr;
  }
  return LockCommandApertureGpuAfterBootstrap(api, device, aperture, out_error);
}

bool RefreshBufferCpuMapping(const KmtApi& api, const Device& device,
                             Buffer* buffer, Error* out_error) {
  if (!buffer || !buffer->allocation || !buffer->cpu_ptr) return true;

  D3DKMT_UNLOCK2 unlock = {};
  unlock.hDevice = device.device;
  unlock.hAllocation = buffer->allocation;
  NTSTATUS status = api.unlock2(&unlock);
  if (!CheckStatus("D3DKMTUnlock2(refresh buffer)", status, out_error)) {
    return false;
  }
  buffer->cpu_ptr = nullptr;

  D3DKMT_LOCK2 lock = {};
  lock.hDevice = device.device;
  lock.hAllocation = buffer->allocation;
  status = api.lock2(&lock);
  if (!CheckStatus("D3DKMTLock2(refresh buffer)", status, out_error)) {
    Error relock_error = out_error ? *out_error : Error{};
    D3DKMT_LOCK2 restore_lock = {};
    restore_lock.hDevice = device.device;
    restore_lock.hAllocation = buffer->allocation;
    NTSTATUS restore_status = api.lock2(&restore_lock);
    Error restore_error;
    if (CheckStatus("D3DKMTLock2(refresh buffer restore)", restore_status,
                    &restore_error)) {
      buffer->cpu_ptr = restore_lock.pData;
    } else {
      SetErrorFormat(out_error, "%s; restore failed: %s",
                     ErrorMessage(&relock_error), ErrorMessage(&restore_error));
    }
    return false;
  }
  buffer->cpu_ptr = lock.pData;
  return true;
}

bool TouchBufferCpuMapping(const KmtApi& api, const Device& device,
                           const Buffer& buffer, const char* label,
                           Error* out_error) {
  if (!buffer.allocation) return true;

  D3DKMT_LOCK2 lock = {};
  lock.hDevice = device.device;
  lock.hAllocation = buffer.allocation;
  NTSTATUS status = api.lock2(&lock);
  char call_name[128] = "D3DKMTLock2(lock-touch)";
  if (label && label[0]) {
    std::snprintf(call_name, sizeof(call_name), "D3DKMTLock2(lock-touch:%s)",
                  label);
  }
  return CheckStatus(call_name, status, out_error);
}

bool WaitForBufferResidency(const KmtApi& api, const Device& device,
                            const Context& context, const Buffer& buffer,
                            const char* label, Error* out_error) {
  if (buffer.paging_fence_value == 0) return true;

  D3DKMT_HANDLE wait_objects[1] = {device.paging_sync_object};
  UINT64 wait_values[1] = {buffer.paging_fence_value};
  D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU wait = {};
  wait.hContext = context.context;
  wait.ObjectCount = 1;
  wait.ObjectHandleArray = wait_objects;
  wait.MonitoredFenceValueArray = wait_values;
  NTSTATUS status = api.wait_from_gpu(&wait);
  char call_name[160] = "D3DKMTWaitForSynchronizationObjectFromGpu";
  if (label && label[0]) {
    std::snprintf(call_name, sizeof(call_name),
                  "D3DKMTWaitForSynchronizationObjectFromGpu(%s)", label);
  }
  return CheckStatus(call_name, status, out_error);
}

void DestroyBuffer(const KmtApi& api, const Device& device, Buffer* buffer) {
  if (!buffer || !buffer->allocation) return;
  if (buffer->cpu_ptr) {
    D3DKMT_UNLOCK2 unlock = {};
    unlock.hDevice = device.device;
    unlock.hAllocation = buffer->allocation;
    api.unlock2(&unlock);
    buffer->cpu_ptr = nullptr;
  }
  if (buffer->gpu_va) {
    D3DKMT_FREEGPUVIRTUALADDRESS free_va = {};
    free_va.hAdapter = device.adapter;
    free_va.BaseAddress = buffer->gpu_va;
    free_va.Size = buffer->mapped_size ? buffer->mapped_size : buffer->size;
    api.free_gpu_virtual_address(&free_va);
    buffer->gpu_va = 0;
  }
  D3DKMT_DESTROYALLOCATION2 destroy = {};
  destroy.hDevice = device.device;
  destroy.Flags.AssumeNotInUse = 1;
  D3DKMT_HANDLE allocs[1] = {buffer->allocation};
  if (buffer->resource) {
    // CreateShared buffers are owned by their resource; destroy that.
    destroy.hResource = buffer->resource;
  } else {
    destroy.AllocationCount = 1;
    destroy.phAllocationList = allocs;
  }
  api.destroy_allocation2(&destroy);
  buffer->allocation = 0;
  buffer->resource = 0;
  buffer->size = 0;
  buffer->mapped_size = 0;
}

void DestroyStatusRing(const KmtApi& api, const Device& device,
                       Context* context) {
  if (!context) return;
  DestroyBuffer(api, device, &context->completion_ring);
  context->completion_ring_resource = 0;
  context->completion_ring_ready = false;
  context->completion_ring_offset = 0;
}

bool CreateContext(const KmtApi& api, const Device& device,
                   const uint8_t* private_data, size_t private_data_size,
                   Context* out_context, Error* out_error) {
  if (!out_context) {
    SetError(out_error, "CreateContext called with null output");
    return false;
  }
  if (!private_data || private_data_size > std::numeric_limits<UINT>::max()) {
    SetError(out_error, "CreateContext called with invalid private data");
    return false;
  }

  Context context = {};
  D3DKMT_CREATECONTEXTVIRTUAL create_context = {};
  create_context.hDevice = device.device;
  create_context.NodeOrdinal = 0;
  create_context.EngineAffinity = 1;
  create_context.Flags.HwQueueSupported = 1;
  create_context.pPrivateDriverData = const_cast<uint8_t*>(private_data);
  create_context.PrivateDriverDataSize = static_cast<UINT>(private_data_size);
  create_context.ClientHint = D3DKMT_CLIENTHINT_VITIS;
  NTSTATUS status = api.create_context_virtual(&create_context);
  if (!CheckStatus("D3DKMTCreateContextVirtual", status, out_error)) {
    return false;
  }
  context.context = create_context.hContext;
  if (private_data_size >= kContextCommandApertureCookieOffset +
                               sizeof(context.command_aperture_cookie)) {
    std::memcpy(&context.command_aperture_cookie,
                private_data + kContextCommandApertureCookieOffset,
                sizeof(context.command_aperture_cookie));
  }

  D3DKMT_CREATEHWQUEUE create_queue = {};
  create_queue.hHwContext = context.context;
  status = api.create_hw_queue(&create_queue);
  if (!CheckStatus("D3DKMTCreateHwQueue", status, out_error)) {
    DestroyContext(api, device, &context);
    return false;
  }
  context.hw_queue = create_queue.hHwQueue;
  context.progress_fence = create_queue.hHwQueueProgressFence;
  context.progress_fence_cpu =
      create_queue.HwQueueProgressFenceCPUVirtualAddress;
  context.progress_fence_gpu =
      create_queue.HwQueueProgressFenceGPUVirtualAddress;

  *out_context = context;
  return true;
}

void DestroyContext(const KmtApi& api, const Device& device, Context* context) {
  if (!context) return;
  // The status ring is a context-owned KMT allocation created lazily by the
  // path-B submit path. Tear it down before destroying the HW queue/context
  // handles it was bound to.
  DestroyStatusRing(api, device, context);
  if (context->hw_queue) {
    D3DKMT_DESTROYHWQUEUE destroy_queue = {};
    destroy_queue.hHwQueue = context->hw_queue;
    api.destroy_hw_queue(&destroy_queue);
    context->hw_queue = 0;
  }
  if (context->context) {
    D3DKMT_DESTROYCONTEXT destroy_context = {};
    destroy_context.hContext = context->context;
    api.destroy_context(&destroy_context);
    context->context = 0;
  }
  context->progress_fence = 0;
  context->progress_fence_cpu = nullptr;
  context->progress_fence_gpu = 0;
  context->next_fence_id = 1;
}

bool CreateCommandAperture(const KmtApi& api, const Device& device,
                           const Context& context,
                           CommandAperture* out_aperture, Error* out_error) {
  if (!out_aperture) {
    SetError(out_error, "CreateCommandAperture called with null output");
    return false;
  }

  AllocPrivate command_private = {};
  command_private.requested_size = kCommandApertureAllocationSize;
  command_private.aligned_size = kCommandApertureAllocationSize;
  command_private.private_type = kCommandAperturePrivateType;
  command_private.policy = 0;

  D3DDDI_ALLOCATIONINFO2 command_info = {};
  command_info.pPrivateDriverData = &command_private;
  command_info.PrivateDriverDataSize = sizeof(command_private);

  D3DKMT_CREATEALLOCATION create_command = {};
  create_command.hDevice = device.device;
  create_command.Flags.CreateResource = 1;
  create_command.Flags.CreateShared = 1;
  create_command.NumAllocations = 1;
  create_command.pAllocationInfo2 = &command_info;

  NTSTATUS status = api.create_allocation2(&create_command);
  if (!CheckStatus("D3DKMTCreateAllocation2(command aperture)", status,
                   out_error)) {
    return false;
  }

  CommandAperture aperture = {};
  aperture.allocation_size = kCommandApertureAllocationSize;
  aperture.gpu_va_size = kCommandApertureGpuVaSize;
  aperture.allocation = command_info.hAllocation;
  aperture.resource = create_command.hResource;

  D3DKMT_LOCK2 lock = {};
  lock.hDevice = device.device;
  lock.hAllocation = aperture.allocation;
  status = api.lock2(&lock);
  if (status != 0) {
    SetErrorFormat(out_error,
                   "D3DKMTLock2(command aperture) failed with 0x%08x%s "
                   "allocation=0x%08x resource=0x%08x",
                   static_cast<uint32_t>(status), NtStatusSuffix(status),
                   static_cast<unsigned>(aperture.allocation),
                   static_cast<unsigned>(aperture.resource));
    DestroyCommandAperture(api, device, &aperture);
    return false;
  }
  aperture.cpu_ptr = lock.pData;
  aperture.cpu_ptr_size = kCommandApertureAllocationSize;

  // XRT creates the 64 MiB command aperture as a separate cacheable allocation
  // and folds the driver context writeback cookie into xcl_flags:
  //   xcl_flags = 0x01000001 | (context_blob[0x40] << 16)
  // The transaction instruction stream lives inside this same BO at +0x8000.
  AllocPrivate gpu_private = {};
  gpu_private.requested_size = kCommandApertureGpuVaSize;
  gpu_private.aligned_size = kCommandApertureGpuVaSize;
  gpu_private.reserved1 = 1;
  gpu_private.private_type =
      GetBufferKindInfo(BufferKind::cacheable).private_type;
  gpu_private.policy = 2;
  gpu_private.xcl_flags = 0x01000001u | (context.command_aperture_cookie << 16);

  D3DDDI_ALLOCATIONINFO2 gpu_info = {};
  gpu_info.pPrivateDriverData = &gpu_private;
  gpu_info.PrivateDriverDataSize = sizeof(gpu_private);

  D3DKMT_CREATEALLOCATION create_gpu = {};
  create_gpu.hDevice = device.device;
  create_gpu.NumAllocations = 1;
  create_gpu.pAllocationInfo2 = &gpu_info;
  status = api.create_allocation2(&create_gpu);
  if (!CheckStatus("D3DKMTCreateAllocation2(command aperture gpu)", status,
                   out_error)) {
    DestroyCommandAperture(api, device, &aperture);
    return false;
  }
  aperture.gpu_allocation = gpu_info.hAllocation;

  D3DDDI_MAPGPUVIRTUALADDRESS map = {};
  map.hPagingQueue = device.paging_queue;
  map.hAllocation = aperture.gpu_allocation;
  map.SizeInPages = kCommandApertureGpuVaSize / kPageSize;
  map.Protection.Write = 1;
  status = api.map_gpu_virtual_address(&map);
  if (status != 0 && status != kStatusPending) {
    SetErrorFormat(out_error,
                   "D3DKMTMapGpuVirtualAddress(command aperture) failed with "
                   "0x%08x%s allocation=0x%08x gpu_allocation=0x%08x "
                   "resource=0x%08x pages=0x%llx",
                   static_cast<uint32_t>(status), NtStatusSuffix(status),
                   static_cast<unsigned>(aperture.allocation),
                   static_cast<unsigned>(aperture.gpu_allocation),
                   static_cast<unsigned>(aperture.resource),
                   static_cast<unsigned long long>(map.SizeInPages));
    DestroyCommandAperture(api, device, &aperture);
    return false;
  }
  aperture.gpu_va = map.VirtualAddress;
  if (aperture.gpu_va != kCommandApertureGpuVaBase) {
    SetErrorFormat(out_error,
                   "D3DKMTMapGpuVirtualAddress(command aperture) returned "
                   "VA 0x%llx, expected 0x%llx",
                   static_cast<unsigned long long>(aperture.gpu_va),
                   static_cast<unsigned long long>(kCommandApertureGpuVaBase));
    DestroyCommandAperture(api, device, &aperture);
    return false;
  }

  D3DKMT_HANDLE resident_allocs[1] = {aperture.gpu_allocation};
  D3DDDI_MAKERESIDENT resident = {};
  resident.hPagingQueue = device.paging_queue;
  resident.NumAllocations = 1;
  resident.AllocationList = resident_allocs;
  resident.Flags.CantTrimFurther = 1;
  resident.Flags.MustSucceed = 1;
  status = api.make_resident(&resident);
  if (!CheckStatusOrPending("D3DKMTMakeResident(command aperture)", status,
                            out_error)) {
    DestroyCommandAperture(api, device, &aperture);
    return false;
  }
  D3DKMT_HANDLE wait_objects[1] = {device.paging_sync_object};
  UINT64 wait_values[1] = {resident.PagingFenceValue};
  D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU wait = {};
  wait.hContext = context.context;
  wait.ObjectCount = 1;
  wait.ObjectHandleArray = wait_objects;
  wait.MonitoredFenceValueArray = wait_values;
  if (context.context) {
    status = api.wait_from_gpu(&wait);
    if (!CheckStatus(
            "D3DKMTWaitForSynchronizationObjectFromGpu(command aperture)",
            status, out_error)) {
      DestroyCommandAperture(api, device, &aperture);
      return false;
    }
  } else if (!WaitForPagingFenceCpu(api, device, resident.PagingFenceValue)) {
    SetError(out_error,
             "D3DKMTWaitForSynchronizationObjectFromCpu(command aperture "
             "precreate) failed");
    DestroyCommandAperture(api, device, &aperture);
    return false;
  }

  constexpr uint64_t kCodeOffset = 0x8000;
  aperture.code_allocation = aperture.gpu_allocation;
  aperture.code_gpu_va = aperture.gpu_va + kCodeOffset;
  aperture.code_size = aperture.gpu_va_size - kCodeOffset;

  *out_aperture = aperture;
  return true;
}

bool LockCommandApertureGpuAfterBootstrap(const KmtApi& api,
                                          const Device& device,
                                          CommandAperture* aperture,
                                          Error* out_error) {
  if (!aperture || !aperture->gpu_allocation) {
    SetError(out_error,
             "LockCommandApertureGpuAfterBootstrap called before aperture "
             "setup");
    return false;
  }
  if (aperture->gpu_cpu_ptr) return true;

  D3DKMT_LOCK2 gpu_lock = {};
  gpu_lock.hDevice = device.device;
  gpu_lock.hAllocation = aperture->gpu_allocation;
  NTSTATUS status = api.lock2(&gpu_lock);
  if (!CheckStatus("D3DKMTLock2(command aperture gpu)", status, out_error)) {
    return false;
  }
  aperture->gpu_cpu_ptr = gpu_lock.pData;
  if (aperture->gpu_cpu_ptr) {
    constexpr uint64_t kCodeOffset = 0x8000;
    aperture->code_cpu_ptr =
        static_cast<uint8_t*>(aperture->gpu_cpu_ptr) + kCodeOffset;
  }

  D3DKMT_INVALIDATECACHE invalidate = {};
  invalidate.hDevice = device.device;
  invalidate.hAllocation = aperture->gpu_allocation;
  invalidate.Offset = 0;
  invalidate.Length = aperture->gpu_va_size;
  status = api.invalidate_cache(&invalidate);
  return CheckStatus("D3DKMTInvalidateCache(command aperture)", status,
                     out_error);
}

bool SubmitAndWaitCommandAperture(const KmtApi& api, const Device& device,
                                  Context* context, CommandAperture* aperture,
                                  Error* out_error) {
  if (!context || !context->hw_queue) {
    SetError(out_error, "SubmitAndWait called without an HW queue");
    return false;
  }
  if (!aperture || !aperture->gpu_allocation) {
    SetError(out_error, "SubmitAndWait called before aperture setup");
    return false;
  }

  uint64_t submit_private[kSubmitPrivateQwords] = {};
  submit_private[0] = 2;
  submit_private[1] = aperture->gpu_allocation;
  submit_private[2] = aperture->gpu_va;

  uint64_t fence_id = context->next_fence_id++;
  D3DKMT_SUBMITCOMMANDTOHWQUEUE submit = {};
  submit.hHwQueue = context->hw_queue;
  submit.HwQueueProgressFenceId = fence_id;
  submit.CommandBuffer = aperture->gpu_va;
  submit.CommandLength = static_cast<UINT>(aperture->gpu_va_size);
  submit.PrivateDriverDataSize = sizeof(submit_private);
  submit.pPrivateDriverData = submit_private;
  NTSTATUS status = api.submit_command_to_hw_queue(&submit);
  if (!CheckStatus("D3DKMTSubmitCommandToHwQueue", status, out_error)) {
    return false;
  }

  // XRT locks and invalidates the 64 MiB aperture only after the opcode-2
  // bootstrap submit. Do the same so later CPU writes to aperture+0x8000 use
  // the same MCDM object state as the reference runtime.
  return LockCommandApertureGpuAfterBootstrap(api, device, aperture, out_error);
}

bool WaitForHwQueueFenceCpu(const KmtApi& api, const Device& device,
                            const Context& context, uint64_t fence_id,
                            const char* label, Error* out_error) {
  D3DKMT_HANDLE wait_objects[1] = {context.progress_fence};
  UINT64 wait_values[1] = {fence_id};
  D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait = {};
  wait.hDevice = device.device;
  wait.ObjectCount = 1;
  wait.ObjectHandleArray = wait_objects;
  wait.FenceValueArray = wait_values;
  // XRT's qhdl wait path calls D3DKMTWaitForSynchronizationObjectFromCpu with
  // hAsyncEvent=0 and blocks in KMT. Match that call shape instead of using an
  // asynchronous event plus a host-side wait wrapper.
  wait.hAsyncEvent = nullptr;
  NTSTATUS status = api.wait_from_cpu(&wait);
  return CheckStatus(label, status, out_error);
}

bool SubmitAndWaitPathBSetup(const KmtApi& api, const Device& device,
                             Context* context, CommandAperture* aperture,
                             const void* aperture_payload,
                             size_t aperture_payload_size, Error* out_error) {
  if (!context || !context->hw_queue || !aperture ||
      !aperture->gpu_allocation || !aperture->allocation ||
      !aperture->cpu_ptr) {
    SetError(out_error, "SubmitAndWaitPathBSetup called before aperture setup");
    return false;
  }

  uint64_t bootstrap_private[kSubmitPrivateQwords] = {};
  bootstrap_private[0] = 2;
  bootstrap_private[1] = aperture->gpu_allocation;
  bootstrap_private[2] = aperture->gpu_va;

  uint64_t fence_id = context->next_fence_id++;
  D3DKMT_SUBMITCOMMANDTOHWQUEUE submit = {};
  submit.hHwQueue = context->hw_queue;
  submit.HwQueueProgressFenceId = fence_id;
  submit.CommandBuffer = aperture->gpu_va;
  submit.CommandLength = static_cast<UINT>(aperture->gpu_va_size);
  submit.PrivateDriverDataSize = sizeof(bootstrap_private);
  submit.pPrivateDriverData = bootstrap_private;
  NTSTATUS status = api.submit_command_to_hw_queue(&submit);
  if (!CheckStatus("D3DKMTSubmitCommandToHwQueue(pathb bootstrap)", status,
                   out_error)) {
    return false;
  }
  if (!LockCommandApertureGpuAfterBootstrap(api, device, aperture, out_error)) {
    return false;
  }
  if (aperture_payload_size != 0) {
    if (!aperture_payload) {
      SetError(out_error,
               "SubmitAndWaitPathBSetup called with null aperture payload");
      return false;
    }
    if (!aperture->gpu_cpu_ptr ||
        aperture_payload_size > aperture->gpu_va_size) {
      SetError(out_error,
               "SubmitAndWaitPathBSetup aperture payload does not fit mapped "
               "command aperture");
      return false;
    }
    std::memcpy(aperture->gpu_cpu_ptr, aperture_payload, aperture_payload_size);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    FlushProcessWriteBuffers();
  }

  uint8_t setup_private[0x270] = {};
  WriteU32(setup_private, 0x00, 5);
  WriteU64(setup_private, 0x28, aperture->allocation);
  WriteU32(setup_private, 0x30, 0);
  WriteU32(setup_private, 0x34, kQhdlCompletionSlotSize);
  WriteU64(setup_private, 0x38, reinterpret_cast<uint64_t>(aperture->cpu_ptr));
  WriteU32(setup_private, 0x68, 1);
  WriteU64(setup_private, 0x70, aperture->gpu_va);

  fence_id = context->next_fence_id++;
  submit = {};
  submit.hHwQueue = context->hw_queue;
  submit.HwQueueProgressFenceId = fence_id;
  submit.CommandBuffer = 0;
  submit.CommandLength = 0;
  submit.PrivateDriverDataSize = sizeof(setup_private);
  submit.pPrivateDriverData = setup_private;
  status = api.submit_command_to_hw_queue(&submit);
  if (!CheckStatus("D3DKMTSubmitCommandToHwQueue(pathb setup5)", status,
                   out_error)) {
    return false;
  }
  return WaitForHwQueueFenceCpu(
      api, device, *context, fence_id,
      "D3DKMTWaitForSynchronizationObjectFromCpu(pathb setup5)", out_error);
}

bool SubmitPathBApertureSync(const KmtApi& api, const Device& device,
                             Context* context, const CommandAperture& aperture,
                             uint64_t offset, bool wait_for_cpu,
                             Error* out_error) {
  if (!context || !context->hw_queue || !aperture.gpu_allocation) {
    SetError(out_error, "SubmitPathBApertureSync called before aperture setup");
    return false;
  }
  uint64_t sync_private[kSubmitPrivateQwords] = {};
  sync_private[0] = 9;
  sync_private[1] = aperture.gpu_allocation;
  sync_private[2] = offset;

  uint64_t fence_id = context->next_fence_id++;
  D3DKMT_SUBMITCOMMANDTOHWQUEUE submit = {};
  submit.hHwQueue = context->hw_queue;
  submit.HwQueueProgressFenceId = fence_id;
  submit.CommandBuffer = 0;
  submit.CommandLength = 0;
  submit.PrivateDriverDataSize = sizeof(sync_private);
  submit.pPrivateDriverData = sync_private;
  NTSTATUS status = api.submit_command_to_hw_queue(&submit);
  if (!CheckStatus("D3DKMTSubmitCommandToHwQueue(pathb sync9)", status,
                   out_error)) {
    return false;
  }
  if (!wait_for_cpu) return true;
  return WaitForHwQueueFenceCpu(
      api, device, *context, fence_id,
      "D3DKMTWaitForSynchronizationObjectFromCpu(pathb sync9)", out_error);
}

// Allocate the firmware completion ring as a 0x332b status_bo (NOT a plain host
// BO), mirroring xrt_core FUN_1800261f0: CreateAllocation2 with private type
// 0x332b + CreateResource|CreateShared, then Lock2 for the CPU slot base. The
// firmware writes completion state into this allocation; an ordinary host_only
// BO is never touched, which is why earlier slots stayed 0.
bool EnsureStatusRing(const KmtApi& api, const Device& device, Context* context,
                      Error* out_error) {
  if (context->completion_ring_ready) return true;
  constexpr uint32_t kRingSize = 4096;
  AllocPrivate ring_private = {};
  ring_private.requested_size = kRingSize;
  ring_private.aligned_size = kRingSize;
  ring_private.private_type = kCommandAperturePrivateType;  // 0x332b status_bo
  ring_private.policy = 0;

  D3DDDI_ALLOCATIONINFO2 ring_info = {};
  ring_info.pPrivateDriverData = &ring_private;
  ring_info.PrivateDriverDataSize = sizeof(ring_private);

  D3DKMT_CREATEALLOCATION create_ring = {};
  create_ring.hDevice = device.device;
  create_ring.Flags.CreateResource = 1;
  create_ring.Flags.CreateShared = 1;
  create_ring.NumAllocations = 1;
  create_ring.pAllocationInfo2 = &ring_info;
  NTSTATUS status = api.create_allocation2(&create_ring);
  if (!CheckStatus("D3DKMTCreateAllocation2(status ring)", status, out_error)) {
    return false;
  }

  Buffer ring = {};
  ring.kind = BufferKind::cacheable;
  ring.size = kRingSize;
  ring.mapped_size = kRingSize;
  ring.allocation = ring_info.hAllocation;
  ring.resource = create_ring.hResource;

  D3DKMT_LOCK2 lock = {};
  lock.hDevice = device.device;
  lock.hAllocation = ring.allocation;
  status = api.lock2(&lock);
  if (!CheckStatus("D3DKMTLock2(status ring)", status, out_error)) {
    DestroyBuffer(api, device, &ring);
    return false;
  }
  ring.cpu_ptr = lock.pData;
  if (ring.cpu_ptr) std::memset(ring.cpu_ptr, 0, kRingSize);

  // Map to a GPU VA and make resident so the firmware/NPU can DMA completion
  // state into the ring. Without this the BO is only host-side (Lock2) and the
  // firmware can never write the completion -> command never completes.
  D3DDDI_MAPGPUVIRTUALADDRESS map = {};
  map.hPagingQueue = device.paging_queue;
  map.hAllocation = ring.allocation;
  map.SizeInPages = kRingSize / 4096;
  map.Protection.Write = 1;
  status = api.map_gpu_virtual_address(&map);
  if (status != 0 && status != kStatusPending) {
    CheckStatus("D3DKMTMapGpuVirtualAddress(status ring)", status, out_error);
    DestroyBuffer(api, device, &ring);
    return false;
  }
  ring.gpu_va = map.VirtualAddress;

  D3DKMT_HANDLE resident_allocs[1] = {ring.allocation};
  D3DDDI_MAKERESIDENT resident = {};
  resident.hPagingQueue = device.paging_queue;
  resident.NumAllocations = 1;
  resident.AllocationList = resident_allocs;
  resident.Flags.CantTrimFurther = 1;
  resident.Flags.MustSucceed = 1;
  status = api.make_resident(&resident);
  if (status != 0 && status != kStatusPending) {
    CheckStatus("D3DKMTMakeResident(status ring)", status, out_error);
    DestroyBuffer(api, device, &ring);
    return false;
  }
  D3DKMT_HANDLE wait_objects[1] = {device.paging_sync_object};
  UINT64 wait_values[1] = {resident.PagingFenceValue};
  D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU wait = {};
  wait.hContext = context->context;
  wait.ObjectCount = 1;
  wait.ObjectHandleArray = wait_objects;
  wait.MonitoredFenceValueArray = wait_values;
  status = api.wait_from_gpu(&wait);
  if (!CheckStatus("D3DKMTWaitForSynchronizationObjectFromGpu(status ring)",
                   status, out_error)) {
    DestroyBuffer(api, device, &ring);
    return false;
  }

  context->completion_ring = ring;
  context->completion_ring_resource = ring.resource;
  context->completion_ring_ready = true;
  context->completion_ring_offset = kQhdlCompletionSlotSize;
  return true;
}

bool SubmitAndWaitPathBImpl(const KmtApi& api, const Device& device,
                            Context* context, const Buffer& exec_buffer,
                            const void* ert_packet, uint32_t ert_bytes,
                            uint32_t command_state,
                            const PathBChainSubmitInfo* chain_info,
                            uint32_t* packet_header, Error* out_error) {
  constexpr uint32_t kCompletionRingSize = 4096;
  if (!context || !context->hw_queue) {
    SetError(out_error, "SubmitAndWaitPathB called without an HW queue");
    return false;
  }
  if (!exec_buffer.allocation || !exec_buffer.gpu_va || exec_buffer.size == 0) {
    SetError(out_error,
             "SubmitAndWaitPathB called with an invalid exec buffer");
    return false;
  }
  if (ert_bytes == 0 ||
      ert_bytes > kQhdlSubmitPrivateSize - kQhdlSubmitPacketOffset) {
    SetErrorFormat(out_error, "SubmitAndWaitPathB invalid ert_bytes=%u",
                   ert_bytes);
    return false;
  }
  if (chain_info) {
    if (!chain_info->descriptor_gpu_va || !chain_info->descriptor_bytes ||
        !chain_info->command_count) {
      SetError(out_error,
               "SubmitAndWaitPathBChain called with incomplete chain metadata");
      return false;
    }
  }

  // Lazily allocate the completion ring (device-visible, 8-byte slots). The
  // firmware writes per-command completion state here; slot 0 is reserved.
  if (!EnsureStatusRing(api, device, context, out_error)) {
    return false;
  }
  Buffer& ring = context->completion_ring;

  // Reserve an 8-byte completion slot (mirrors hwqueue_aie4 reserve).
  uint32_t slot_offset = context->completion_ring_offset;
  if (slot_offset + kQhdlCompletionSlotSize > ring.size) {
    slot_offset = kQhdlCompletionSlotSize;
  }
  context->completion_ring_offset =
      (slot_offset + 2u * kQhdlCompletionSlotSize > ring.size)
          ? kQhdlCompletionSlotSize
          : slot_offset + kQhdlCompletionSlotSize;
  uint8_t* slot_cpu = static_cast<uint8_t*>(ring.cpu_ptr) + slot_offset;
  // XRT leaves the 8-byte firmware completion slot zeroed before submit.
  // Match that exactly: the driver treats this buffer as part of the private
  // queue protocol, not just host-side scratch space.
  InitializeCompletionSlot(slot_cpu);

  // Build the 0x268 private packet (layout recovered from xrt_core
  // hwqueue_aie4::submit_command / FUN_1800304c0).
  const uint32_t effective_command_state =
      chain_info ? 6u : (command_state ? command_state : 3u);
  std::array<uint8_t, kQhdlSubmitPrivateSize> priv = {};
  WriteU32(priv.data(), 0x00, effective_command_state);
  WriteU64(priv.data(), 0x08, exec_buffer.allocation);
  WriteU64(priv.data(), 0x10, ert_bytes);
  WriteU64(priv.data(), 0x28, ring.allocation);  // status_bo allocation handle
  WriteU32(priv.data(), 0x30, slot_offset);
  WriteU32(priv.data(), 0x34, kQhdlCompletionSlotSize);
  // +0x38 is the slot CPU pointer (XRT reads completion via *(uint*)slot); the
  // firmware locates the slot from the ring device VA (+0x28) + offset (+0x30).
  WriteU64(priv.data(), 0x38, reinterpret_cast<uint64_t>(slot_cpu));
  if (chain_info) {
    WriteU64(priv.data(), 0x48, chain_info->descriptor_gpu_va);
    WriteU32(priv.data(), 0x50, chain_info->descriptor_bytes);
    WriteU32(priv.data(), 0x54, chain_info->command_count);
    WriteU32(priv.data(), 0x58, chain_info->first_child_opcode);
  }
  std::memcpy(priv.data() + kQhdlSubmitPacketOffset, ert_packet, ert_bytes);

  if (!WaitForBufferResidency(api, device, *context, exec_buffer, "pathb-exec",
                              out_error)) {
    return false;
  }
  if (!WaitForBufferResidency(api, device, *context, ring, "pathb-ring",
                              out_error)) {
    return false;
  }

  const bool xrt_lock_touch = XrtLockTouchEnabled();
  if (xrt_lock_touch && !TouchBufferCpuMapping(api, device, exec_buffer,
                                               "pre-submit", out_error)) {
    return false;
  }

  uint64_t fence_id = context->next_fence_id++;
  D3DKMT_SUBMITCOMMANDTOHWQUEUE submit = {};
  submit.hHwQueue = context->hw_queue;
  submit.HwQueueProgressFenceId = fence_id;
  submit.CommandBuffer = exec_buffer.gpu_va;
  submit.CommandLength =
      static_cast<UINT>(exec_buffer.size + kQhdlSubmitPacketOffset);
  submit.PrivateDriverDataSize = static_cast<UINT>(priv.size());
  submit.pPrivateDriverData = priv.data();
  NTSTATUS status = api.submit_command_to_hw_queue(&submit);
  if (!CheckStatus("D3DKMTSubmitCommandToHwQueue(pathb)", status, out_error)) {
    return false;
  }
  if (xrt_lock_touch && !TouchBufferCpuMapping(api, device, exec_buffer,
                                               "post-submit", out_error)) {
    return false;
  }

  if (!WaitForHwQueueFenceCpu(
          api, device, *context, fence_id,
          "D3DKMTWaitForSynchronizationObjectFromCpu(pathb)", out_error)) {
    return false;
  }

  // XRT's qhdl wait path blocks in KMT and then mirrors the low ERT state
  // nibble from the completion slot into the packet header. Do one
  // cache-visible read from the explicit protocol locations here.
  volatile uint32_t* const volatile_packet_header = packet_header;
  uint32_t slot_state = 0;
  uint32_t packet_state = volatile_packet_header ? *volatile_packet_header : 0;
  const bool trust_fence_completion = TrustFenceCompletionEnabled();
  auto read_completion_once = [&]() -> bool {
    if (trust_fence_completion) {
      // WaitForHwQueueFenceCpu above is the GPU->CPU ordering point. The
      // volatile read prevents the compiler from reusing a stale packet header
      // value while the fast path trusts the post-fence host mapping.
      std::atomic_thread_fence(std::memory_order_seq_cst);
      packet_state = volatile_packet_header ? *volatile_packet_header : 0;
      std::memcpy(&slot_state, slot_cpu, sizeof(slot_state));
      return true;
    }
    Error command_sync_err;
    if (!SyncBuffer(api, device, exec_buffer, 0, exec_buffer.size,
                    &command_sync_err)) {
      SetErrorFormat(out_error, "pathb command buffer invalidate failed: %s",
                     ErrorMessage(&command_sync_err));
      return false;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    packet_state = volatile_packet_header ? *volatile_packet_header : 0;

    Error ring_sync_err;
    if (!SyncBuffer(api, device, ring, 0, ring.size, &ring_sync_err)) {
      SetErrorFormat(out_error, "pathb completion ring invalidate failed: %s",
                     ErrorMessage(&ring_sync_err));
      return false;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    std::memcpy(&slot_state, slot_cpu, sizeof(slot_state));
    return true;
  };
  if (!read_completion_once()) return false;
  if (volatile_packet_header) {
    const uint32_t completion_state =
        ((packet_state & 0xFu) >= 4) ? packet_state : slot_state;
    uint32_t delta = (*volatile_packet_header ^ completion_state) & 0xFu;
    *volatile_packet_header ^= delta;
  }
  const uint32_t final_state =
      volatile_packet_header ? *volatile_packet_header : slot_state;
  if ((final_state & 0xFu) < 4) {
    SetErrorFormat(out_error,
                   "pathb command did not complete after fence wait: "
                   "packet_state=0x%08x slot_state=0x%08x slot_offset=0x%x",
                   packet_state, slot_state, slot_offset);
    return false;
  }
  return true;
}

bool SubmitPathBImplNoWait(const KmtApi& api, const Device& device,
                           Context* context, const Buffer& exec_buffer,
                           const void* ert_packet, uint32_t ert_bytes,
                           uint32_t command_state,
                           const PathBChainSubmitInfo* chain_info,
                           uint32_t* packet_header,
                           PathBPendingSubmit* out_pending, Error* out_error) {
  if (!out_pending) {
    SetError(out_error, "SubmitPathB called without pending storage");
    return false;
  }
  *out_pending = {};
  if (!context || !context->hw_queue) {
    SetError(out_error, "SubmitPathB called without an HW queue");
    return false;
  }
  if (!exec_buffer.allocation || !exec_buffer.gpu_va || exec_buffer.size == 0) {
    SetError(out_error, "SubmitPathB called with an invalid exec buffer");
    return false;
  }
  if (ert_bytes == 0 ||
      ert_bytes > kQhdlSubmitPrivateSize - kQhdlSubmitPacketOffset) {
    SetErrorFormat(out_error, "SubmitPathB invalid ert_bytes=%u", ert_bytes);
    return false;
  }
  if (chain_info) {
    if (!chain_info->descriptor_gpu_va || !chain_info->descriptor_bytes ||
        !chain_info->command_count) {
      SetError(out_error,
               "SubmitPathBChain called with incomplete chain metadata");
      return false;
    }
  }

  if (!EnsureStatusRing(api, device, context, out_error)) return false;
  Buffer& ring = context->completion_ring;

  uint32_t slot_offset = context->completion_ring_offset;
  if (slot_offset + kQhdlCompletionSlotSize > ring.size) {
    slot_offset = kQhdlCompletionSlotSize;
  }
  context->completion_ring_offset =
      (slot_offset + 2u * kQhdlCompletionSlotSize > ring.size)
          ? kQhdlCompletionSlotSize
          : slot_offset + kQhdlCompletionSlotSize;
  uint8_t* slot_cpu = static_cast<uint8_t*>(ring.cpu_ptr) + slot_offset;
  InitializeCompletionSlot(slot_cpu);

  const uint32_t effective_command_state =
      chain_info ? 6u : (command_state ? command_state : 3u);
  std::array<uint8_t, kQhdlSubmitPrivateSize> priv = {};
  WriteU32(priv.data(), 0x00, effective_command_state);
  WriteU64(priv.data(), 0x08, exec_buffer.allocation);
  WriteU64(priv.data(), 0x10, ert_bytes);
  WriteU64(priv.data(), 0x28, ring.allocation);
  WriteU32(priv.data(), 0x30, slot_offset);
  WriteU32(priv.data(), 0x34, kQhdlCompletionSlotSize);
  WriteU64(priv.data(), 0x38, reinterpret_cast<uint64_t>(slot_cpu));
  if (chain_info) {
    WriteU64(priv.data(), 0x48, chain_info->descriptor_gpu_va);
    WriteU32(priv.data(), 0x50, chain_info->descriptor_bytes);
    WriteU32(priv.data(), 0x54, chain_info->command_count);
    WriteU32(priv.data(), 0x58, chain_info->first_child_opcode);
  }
  std::memcpy(priv.data() + kQhdlSubmitPacketOffset, ert_packet, ert_bytes);

  if (!WaitForBufferResidency(api, device, *context, exec_buffer, "pathb-exec",
                              out_error)) {
    return false;
  }
  if (!WaitForBufferResidency(api, device, *context, ring, "pathb-ring",
                              out_error)) {
    return false;
  }

  const bool xrt_lock_touch = XrtLockTouchEnabled();
  if (xrt_lock_touch && !TouchBufferCpuMapping(api, device, exec_buffer,
                                               "pre-submit", out_error)) {
    return false;
  }

  uint64_t fence_id = context->next_fence_id++;
  D3DKMT_SUBMITCOMMANDTOHWQUEUE submit = {};
  submit.hHwQueue = context->hw_queue;
  submit.HwQueueProgressFenceId = fence_id;
  submit.CommandBuffer = exec_buffer.gpu_va;
  submit.CommandLength =
      static_cast<UINT>(exec_buffer.size + kQhdlSubmitPacketOffset);
  submit.PrivateDriverDataSize = static_cast<UINT>(priv.size());
  submit.pPrivateDriverData = priv.data();
  NTSTATUS status = api.submit_command_to_hw_queue(&submit);
  if (!CheckStatus("D3DKMTSubmitCommandToHwQueue(pathb nowait)", status,
                   out_error)) {
    return false;
  }
  if (xrt_lock_touch && !TouchBufferCpuMapping(api, device, exec_buffer,
                                               "post-submit", out_error)) {
    return false;
  }

  out_pending->fence_id = fence_id;
  out_pending->slot_cpu = slot_cpu;
  out_pending->slot_offset = slot_offset;
  out_pending->packet_header = packet_header;
  out_pending->exec_buffer = exec_buffer;
  out_pending->ring = ring;
  return true;
}

bool IsPathBSubmitComplete(const Context& context,
                           const PathBPendingSubmit& pending) {
  // Non-blocking poll of the CPU-monitored HW progress fence: the dispatch's
  // fence_id is reached once the firmware retires it. (The output buffers are
  // only host-coherent after WaitForPathBSubmits, which performs the syncs.)
  if (!context.progress_fence_cpu) return false;
  std::atomic_thread_fence(std::memory_order_seq_cst);
  const uint64_t current =
      *reinterpret_cast<const volatile uint64_t*>(context.progress_fence_cpu);
  return current >= pending.fence_id;
}

bool WaitForPathBSubmits(const KmtApi& api, const Device& device,
                         Context* context, PathBPendingSubmit* pending,
                         size_t pending_count, Error* out_error) {
  if (!pending_count) return true;
  if (!context || !context->hw_queue) {
    SetError(out_error, "WaitForPathBSubmits called without an HW queue");
    return false;
  }
  if (!pending) {
    SetError(out_error, "WaitForPathBSubmits called without commands");
    return false;
  }
  // Match XRT runlist semantics: wait for the final parent chunk. In-order HWQ
  // execution means earlier parent chunks have retired when the last fence is
  // reached. Completion state for each parent is still checked below.
  PathBPendingSubmit& last = pending[pending_count - 1];
  if (!WaitForHwQueueFenceCpu(
          api, device, *context, last.fence_id,
          "D3DKMTWaitForSynchronizationObjectFromCpu(pathb batch)",
          out_error)) {
    return false;
  }

  const bool trust_fence_completion = TrustFenceCompletionEnabled();
  for (size_t i = 0; i < pending_count; ++i) {
    PathBPendingSubmit& p = pending[i];
    volatile uint32_t* const packet_header = p.packet_header;
    uint32_t slot_state = 0;
    uint32_t packet_state = packet_header ? *packet_header : 0;
    if (trust_fence_completion) {
      // WaitForHwQueueFenceCpu above is the GPU->CPU ordering point for all
      // parent chunks. Keep command-packet header reads volatile while the fast
      // path trusts the post-fence host mapping.
      std::atomic_thread_fence(std::memory_order_seq_cst);
      packet_state = packet_header ? *packet_header : 0;
      std::memcpy(&slot_state, p.slot_cpu, sizeof(slot_state));
    } else {
      Error command_sync_err;
      if (!SyncBuffer(api, device, p.exec_buffer, 0, p.exec_buffer.size,
                      &command_sync_err)) {
        SetErrorFormat(out_error,
                       "pathb batch command buffer invalidate failed: %s",
                       ErrorMessage(&command_sync_err));
        return false;
      }
      std::atomic_thread_fence(std::memory_order_seq_cst);
      packet_state = packet_header ? *packet_header : 0;

      Error ring_sync_err;
      if (!SyncBuffer(api, device, p.ring, 0, p.ring.size, &ring_sync_err)) {
        SetErrorFormat(out_error,
                       "pathb batch completion ring invalidate failed: %s",
                       ErrorMessage(&ring_sync_err));
        return false;
      }
      std::atomic_thread_fence(std::memory_order_seq_cst);
      std::memcpy(&slot_state, p.slot_cpu, sizeof(slot_state));
    }

    if (packet_header) {
      const uint32_t completion_state =
          ((packet_state & 0xFu) >= 4) ? packet_state : slot_state;
      uint32_t delta = (*packet_header ^ completion_state) & 0xFu;
      *packet_header ^= delta;
    }
    const uint32_t final_state = packet_header ? *packet_header : slot_state;
    if ((final_state & 0xFu) < 4) {
      SetErrorFormat(out_error,
                     "pathb batch command %zu did not complete after final "
                     "fence wait: packet_state=0x%08x slot_state=0x%08x "
                     "slot_offset=0x%x",
                     i, packet_state, slot_state, p.slot_offset);
      return false;
    }
  }
  return true;
}

bool SubmitAndWaitPathB(const KmtApi& api, const Device& device,
                        Context* context, const Buffer& exec_buffer,
                        const void* ert_packet, uint32_t ert_bytes,
                        uint32_t command_state, uint32_t* packet_header,
                        Error* out_error) {
  return SubmitAndWaitPathBImpl(api, device, context, exec_buffer, ert_packet,
                                ert_bytes, command_state, nullptr,
                                packet_header, out_error);
}

bool SubmitAndWaitPathBChain(const KmtApi& api, const Device& device,
                             Context* context, const Buffer& exec_buffer,
                             const void* ert_packet, uint32_t ert_bytes,
                             const PathBChainSubmitInfo& chain_info,
                             uint32_t* packet_header, Error* out_error) {
  return SubmitAndWaitPathBImpl(api, device, context, exec_buffer, ert_packet,
                                ert_bytes, 6, &chain_info, packet_header,
                                out_error);
}

bool SubmitPathBChain(const KmtApi& api, const Device& device, Context* context,
                      const Buffer& exec_buffer, const void* ert_packet,
                      uint32_t ert_bytes,
                      const PathBChainSubmitInfo& chain_info,
                      uint32_t* packet_header, PathBPendingSubmit* out_pending,
                      Error* out_error) {
  return SubmitPathBImplNoWait(api, device, context, exec_buffer, ert_packet,
                               ert_bytes, 6, &chain_info, packet_header,
                               out_pending, out_error);
}

bool SubmitPathB(const KmtApi& api, const Device& device, Context* context,
                 const Buffer& exec_buffer, const void* ert_packet,
                 uint32_t ert_bytes, uint32_t command_state,
                 uint32_t* packet_header, PathBPendingSubmit* out_pending,
                 Error* out_error) {
  return SubmitPathBImplNoWait(api, device, context, exec_buffer, ert_packet,
                               ert_bytes, command_state, /*chain_info=*/nullptr,
                               packet_header, out_pending, out_error);
}

void DestroyCommandAperture(const KmtApi& api, const Device& device,
                            CommandAperture* aperture) {
  if (!aperture || (!aperture->allocation && !aperture->resource &&
                    !aperture->gpu_allocation && !aperture->gpu_resource &&
                    !aperture->cleanup_allocation)) {
    return;
  }
  if (aperture->gpu_va) {
    D3DKMT_FREEGPUVIRTUALADDRESS free_va = {};
    free_va.hAdapter = device.adapter;
    free_va.BaseAddress = aperture->gpu_va;
    free_va.Size = aperture->gpu_va_size;
    api.free_gpu_virtual_address(&free_va);
    aperture->gpu_va = 0;
  }
  if (aperture->gpu_cpu_ptr && aperture->gpu_allocation) {
    D3DKMT_UNLOCK2 unlock = {};
    unlock.hDevice = device.device;
    unlock.hAllocation = aperture->gpu_allocation;
    api.unlock2(&unlock);
    aperture->gpu_cpu_ptr = nullptr;
    aperture->code_cpu_ptr = nullptr;
  }
  if (aperture->cpu_ptr && aperture->allocation) {
    D3DKMT_UNLOCK2 unlock = {};
    unlock.hDevice = device.device;
    unlock.hAllocation = aperture->cleanup_allocation
                             ? aperture->cleanup_allocation
                             : aperture->allocation;
    api.unlock2(&unlock);
    aperture->cpu_ptr = nullptr;
  }

  bool owns_separate_gpu_allocation =
      aperture->gpu_allocation &&
      aperture->gpu_allocation != aperture->allocation;
  if (owns_separate_gpu_allocation) {
    D3DKMT_DESTROYALLOCATION2 destroy_gpu = {};
    D3DKMT_HANDLE gpu_allocs[1] = {aperture->gpu_allocation};
    destroy_gpu.hDevice = device.device;
    destroy_gpu.Flags.AssumeNotInUse = 1;
    destroy_gpu.Flags.SynchronousDestroy = 1;
    if (aperture->gpu_resource) {
      destroy_gpu.hResource = aperture->gpu_resource;
    } else if (aperture->gpu_allocation) {
      destroy_gpu.AllocationCount = 1;
      destroy_gpu.phAllocationList = gpu_allocs;
    }
    api.destroy_allocation2(&destroy_gpu);
    aperture->gpu_allocation = 0;
    aperture->gpu_resource = 0;
  }
  aperture->code_allocation = 0;
  aperture->code_resource = 0;

  if (aperture->resource || aperture->cleanup_allocation ||
      aperture->allocation) {
    D3DKMT_DESTROYALLOCATION2 destroy_command = {};
    D3DKMT_HANDLE command_allocs[1] = {aperture->cleanup_allocation
                                           ? aperture->cleanup_allocation
                                           : aperture->allocation};
    destroy_command.hDevice = device.device;
    destroy_command.Flags.SynchronousDestroy = 1;
    if (aperture->cleanup_allocation) {
      destroy_command.Flags.AssumeNotInUse = 1;
      destroy_command.AllocationCount = 1;
      destroy_command.phAllocationList = command_allocs;
    } else if (aperture->resource) {
      destroy_command.hResource = aperture->resource;
    } else if (aperture->allocation) {
      destroy_command.Flags.AssumeNotInUse = 1;
      destroy_command.AllocationCount = 1;
      destroy_command.phAllocationList = command_allocs;
    }
    api.destroy_allocation2(&destroy_command);
  }
  *aperture = {};
}

}  // namespace iree::hal::amdxdna::mcdm
