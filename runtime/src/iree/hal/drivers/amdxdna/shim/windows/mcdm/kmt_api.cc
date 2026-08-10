// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "kmt_api.h"

#include <intrin.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <limits>
#include <string>
#include <vector>

#ifndef IREE_HAL_AMDXDNA_MCDM_ENABLE_KMT_TRACE
#define IREE_HAL_AMDXDNA_MCDM_ENABLE_KMT_TRACE 0
#endif

namespace iree::hal::amdxdna::mcdm {
namespace {

constexpr NTSTATUS kStatusPending = static_cast<NTSTATUS>(0x00000103);
constexpr NTSTATUS kStatusBufferTooSmall =
    static_cast<NTSTATUS>(0xC0000023);
// Windows SDK 10.0.26100 names client hint 25 as VITIS, but older SDK
// headers stop before that enumerator. The D3DKMT ABI field is unchanged.
static_assert(sizeof(D3DKMT_CLIENTHINT) == sizeof(uint32_t));
constexpr D3DKMT_CLIENTHINT kClientHintVitis =
    static_cast<D3DKMT_CLIENTHINT>(25);
constexpr uint64_t kPageSize = 4096;
constexpr uint64_t kCommandApertureAllocationSize = 0x1000;
constexpr D3DGPU_VIRTUAL_ADDRESS kCommandApertureGpuVaBase = 0x04000000;
constexpr uint64_t kCommandApertureGpuVaSize = 0x04000000;
constexpr uint32_t kQhdlCompletionSlotSize = 8;
constexpr uint32_t kMaxSubmitPrivatePrefixSize = 0x78;
constexpr size_t kLegacyContextCommandApertureCookieOffset = 0x40;
constexpr size_t kLegacyV2ContextCommandApertureCookieOffset = 0x3c;
constexpr size_t kCompactContextCommandApertureCookieOffset = 0x44;
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

bool UsesLegacyCpuBufferHandles(McdmAbi abi) {
  return abi == McdmAbi::legacy || abi == McdmAbi::legacy_v0 ||
         abi == McdmAbi::legacy_v2;
}

const char* McdmAbiName(McdmAbi abi) {
  switch (abi) {
    case McdmAbi::legacy_v0:
      return "legacy_v0";
    case McdmAbi::legacy_v2:
      return "legacy_v2";
    case McdmAbi::legacy:
      return "legacy";
    case McdmAbi::compact:
      return "compact";
  }
  return "unknown";
}

size_t ContextCommandApertureCookieOffset(McdmAbi abi) {
  if (abi == McdmAbi::legacy_v0) return size_t{0x30};
  if (abi == McdmAbi::legacy_v2) {
    return kLegacyV2ContextCommandApertureCookieOffset;
  }
  if (abi == McdmAbi::compact) {
    return kCompactContextCommandApertureCookieOffset;
  }
  return kLegacyContextCommandApertureCookieOffset;
}

#if IREE_HAL_AMDXDNA_MCDM_ENABLE_KMT_TRACE
void TraceImpl(const char* fmt, ...) {
  std::fprintf(stderr, "[hrx][kmt] ");
  va_list args;
  va_start(args, fmt);
  std::vfprintf(stderr, fmt, args);
  va_end(args);
  std::fprintf(stderr, "\n");
  std::fflush(stderr);
}

uint64_t HashBytes(const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  uint64_t hash = 1469598103934665603ull;
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

uint32_t ReadTraceU32(const void* data, size_t size, size_t offset) {
  if (!data || offset > size || sizeof(uint32_t) > size - offset) return 0;
  uint32_t value = 0;
  std::memcpy(&value, static_cast<const uint8_t*>(data) + offset,
              sizeof(value));
  return value;
}

void TraceBlobImpl(const char* label, const void* data, size_t size) {
  if (!data || size == 0) return;
  TraceImpl("%s size=%zu hash=0x%016llx u32[0..3]=%08x,%08x,%08x,%08x",
            label, size, static_cast<unsigned long long>(HashBytes(data, size)),
            ReadTraceU32(data, size, 0), ReadTraceU32(data, size, 4),
            ReadTraceU32(data, size, 8), ReadTraceU32(data, size, 12));
}

std::atomic<uint64_t> g_live_kmt_bytes{0};
std::atomic<uint64_t> g_peak_kmt_bytes{0};
std::atomic<uint64_t> g_live_kmt_count{0};

void UpdatePeakKmtBytes(uint64_t live_bytes) {
  uint64_t peak = g_peak_kmt_bytes.load(std::memory_order_relaxed);
  while (live_bytes > peak &&
         !g_peak_kmt_bytes.compare_exchange_weak(
             peak, live_bytes, std::memory_order_relaxed)) {
  }
}

void StatsAllocImpl(const char* label, uint64_t bytes,
                    D3DKMT_HANDLE allocation) {
  const uint64_t live =
      g_live_kmt_bytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
  const uint64_t count =
      g_live_kmt_count.fetch_add(1, std::memory_order_relaxed) + 1;
  UpdatePeakKmtBytes(live);
  std::fprintf(stderr,
               "[hrx][kmt-stats] alloc %s bytes=%llu live=%llu peak=%llu "
               "count=%llu alloc=0x%08x\n",
               label ? label : "",
               static_cast<unsigned long long>(bytes),
               static_cast<unsigned long long>(live),
               static_cast<unsigned long long>(
                   g_peak_kmt_bytes.load(std::memory_order_relaxed)),
               static_cast<unsigned long long>(count),
               static_cast<unsigned>(allocation));
  std::fflush(stderr);
}

void StatsFreeImpl(const char* label, uint64_t bytes,
                   D3DKMT_HANDLE allocation) {
  uint64_t before = g_live_kmt_bytes.load(std::memory_order_relaxed);
  uint64_t live = 0;
  while (true) {
    live = before > bytes ? before - bytes : 0;
    if (g_live_kmt_bytes.compare_exchange_weak(
            before, live, std::memory_order_relaxed)) {
      break;
    }
  }
  uint64_t count = g_live_kmt_count.load(std::memory_order_relaxed);
  while (count > 0 &&
         !g_live_kmt_count.compare_exchange_weak(
             count, count - 1, std::memory_order_relaxed)) {
  }
  std::fprintf(stderr,
               "[hrx][kmt-stats] free %s bytes=%llu live=%llu peak=%llu "
               "count=%llu alloc=0x%08x\n",
               label ? label : "",
               static_cast<unsigned long long>(bytes),
               static_cast<unsigned long long>(live),
               static_cast<unsigned long long>(
                   g_peak_kmt_bytes.load(std::memory_order_relaxed)),
               static_cast<unsigned long long>(
                   g_live_kmt_count.load(std::memory_order_relaxed)),
               static_cast<unsigned>(allocation));
  std::fflush(stderr);
}
#define Trace(...) TraceImpl(__VA_ARGS__)
#define TraceBlob(...) TraceBlobImpl(__VA_ARGS__)
#define StatsAlloc(...) StatsAllocImpl(__VA_ARGS__)
#define StatsFree(...) StatsFreeImpl(__VA_ARGS__)
#else
#define Trace(...) ((void)0)
#define TraceBlob(...) ((void)0)
#define StatsAlloc(...) ((void)0)
#define StatsFree(...) ((void)0)
#endif

void ConfigureMakeResidentFlags(D3DDDI_MAKERESIDENT* resident) {
  if (!resident) return;
  resident->Flags.CantTrimFurther = 1;
  resident->Flags.MustSucceed = 1;
}

uint64_t BytesToMiB(uint64_t bytes) { return bytes / (1024ull * 1024ull); }

NTSTATUS QueryVideoMemoryInfo(const KmtApi& api, const Device& device,
                              D3DKMT_MEMORY_SEGMENT_GROUP group,
                              D3DKMT_QUERYVIDEOMEMORYINFO* out_info) {
  if (!out_info) return static_cast<NTSTATUS>(0xC000000D);
  std::memset(out_info, 0, sizeof(*out_info));
  if (!api.query_video_memory_info) return static_cast<NTSTATUS>(0xC00000BB);
  out_info->hProcess = nullptr;
  out_info->hAdapter = device.adapter;
  out_info->MemorySegmentGroup = group;
  out_info->PhysicalAdapterIndex = 0;
  return api.query_video_memory_info(out_info);
}

void SetMakeResidentError(const KmtApi& api, const Device& device,
                          const char* call_name, NTSTATUS status,
                          const D3DDDI_MAKERESIDENT& resident,
                          Error* out_error) {
  D3DKMT_QUERYVIDEOMEMORYINFO local_info = {};
  D3DKMT_QUERYVIDEOMEMORYINFO nonlocal_info = {};
  const NTSTATUS local_status = QueryVideoMemoryInfo(
      api, device, D3DKMT_MEMORY_SEGMENT_GROUP_LOCAL, &local_info);
  const NTSTATUS nonlocal_status = QueryVideoMemoryInfo(
      api, device, D3DKMT_MEMORY_SEGMENT_GROUP_NON_LOCAL, &nonlocal_info);

  if (local_status == 0 && nonlocal_status == 0) {
    SetErrorFormat(
        out_error,
        "%s failed with 0x%08x%s; resident flags=0x%08x "
        "allocation_count=%u trim=%llu MiB; "
        "local budget/use/reserve/avail=%llu/%llu/%llu/%llu MiB; "
        "nonlocal budget/use/reserve/avail=%llu/%llu/%llu/%llu MiB",
        call_name, static_cast<uint32_t>(status), NtStatusSuffix(status),
        resident.Flags.Value, resident.NumAllocations,
        static_cast<unsigned long long>(BytesToMiB(resident.NumBytesToTrim)),
        static_cast<unsigned long long>(BytesToMiB(local_info.Budget)),
        static_cast<unsigned long long>(BytesToMiB(local_info.CurrentUsage)),
        static_cast<unsigned long long>(
            BytesToMiB(local_info.CurrentReservation)),
        static_cast<unsigned long long>(
            BytesToMiB(local_info.AvailableForReservation)),
        static_cast<unsigned long long>(BytesToMiB(nonlocal_info.Budget)),
        static_cast<unsigned long long>(BytesToMiB(nonlocal_info.CurrentUsage)),
        static_cast<unsigned long long>(
            BytesToMiB(nonlocal_info.CurrentReservation)),
        static_cast<unsigned long long>(
            BytesToMiB(nonlocal_info.AvailableForReservation)));
    return;
  }

  SetErrorFormat(
      out_error,
      "%s failed with 0x%08x%s; resident flags=0x%08x "
      "allocation_count=%u trim=%llu MiB; "
      "D3DKMTQueryVideoMemoryInfo local=0x%08x%s nonlocal=0x%08x%s",
      call_name, static_cast<uint32_t>(status), NtStatusSuffix(status),
      resident.Flags.Value, resident.NumAllocations,
      static_cast<unsigned long long>(BytesToMiB(resident.NumBytesToTrim)),
      static_cast<uint32_t>(local_status), NtStatusSuffix(local_status),
      static_cast<uint32_t>(nonlocal_status), NtStatusSuffix(nonlocal_status));
}

bool CheckMakeResidentStatusOrPending(const KmtApi& api, const Device& device,
                                      const char* call_name, NTSTATUS status,
                                      const D3DDDI_MAKERESIDENT& resident,
                                      Error* out_error) {
  if (status == 0) return true;
  if (status == kStatusPending) {
    Trace("%s -> 0x%08x%s", call_name, static_cast<uint32_t>(status),
          NtStatusSuffix(status));
    return true;
  }
  Trace("%s -> 0x%08x%s trim=%llu flags=0x%08x allocation_count=%u",
        call_name, static_cast<uint32_t>(status), NtStatusSuffix(status),
        static_cast<unsigned long long>(resident.NumBytesToTrim),
        resident.Flags.Value, resident.NumAllocations);
  SetMakeResidentError(api, device, call_name, status, resident, out_error);
  return false;
}

bool CheckStatus(const char* call_name, NTSTATUS status, Error* out_error) {
  if (status == 0) return true;
  Trace("%s -> 0x%08x%s", call_name, static_cast<uint32_t>(status),
        NtStatusSuffix(status));
  SetErrorFormat(out_error, "%s failed with 0x%08x%s", call_name,
                 static_cast<uint32_t>(status), NtStatusSuffix(status));
  return false;
}

bool CheckStatusOrPending(const char* call_name, NTSTATUS status,
                          Error* out_error) {
  if (status == 0) return true;
  if (status == kStatusPending) {
    Trace("%s -> 0x%08x%s", call_name, static_cast<uint32_t>(status),
          NtStatusSuffix(status));
    return true;
  }
  return CheckStatus(call_name, status, out_error);
}

bool PublishCpuWriteRange(void* mapping, uint64_t mapping_size,
                          uint64_t offset, uint64_t length,
                          uint64_t granularity, Error* out_error) {
  if (length == 0) return true;
  if (!mapping || granularity == 0 ||
      (granularity & (granularity - 1)) != 0) {
    SetError(out_error, "invalid CPU write publication mapping");
    return false;
  }
  if (offset > mapping_size || length > mapping_size - offset) {
    SetError(out_error, "CPU write publication range is out of bounds");
    return false;
  }

  const uint64_t begin_offset = offset & ~(granularity - 1);
  const uint64_t write_end = offset + length;
  if (write_end >
      std::numeric_limits<uint64_t>::max() - (granularity - 1)) {
    SetError(out_error, "CPU write publication range overflows");
    return false;
  }
  const uint64_t end_offset =
      (write_end + granularity - 1) & ~(granularity - 1);
  if (end_offset > mapping_size) {
    SetError(out_error, "CPU write publication granule is out of bounds");
    return false;
  }

  constexpr uintptr_t kCpuCacheLineSize = 64;
  const uintptr_t mapping_address = reinterpret_cast<uintptr_t>(mapping);
  if (end_offset > std::numeric_limits<uintptr_t>::max() - mapping_address) {
    SetError(out_error, "CPU write publication address overflows");
    return false;
  }
  uintptr_t line =
      (mapping_address + static_cast<uintptr_t>(begin_offset)) &
      ~(kCpuCacheLineSize - 1);
  const uintptr_t end = mapping_address + static_cast<uintptr_t>(end_offset);
  // Publish only writes synchronized into this calling thread. Buffer ownership
  // is responsible for excluding concurrent writers; a process-wide write
  // buffer flush would hide violations of that contract and is unnecessary.
  std::atomic_thread_fence(std::memory_order_seq_cst);
  while (line < end) {
    _mm_clflush(reinterpret_cast<void const*>(line));
    line += kCpuCacheLineSize;
  }
  // Wait for every cache-line writeback before the caller publishes a packet or
  // aperture descriptor to the hardware queue.
  _mm_mfence();
  return true;
}

uint32_t Flags32(const D3DKMT_CREATEALLOCATIONFLAGS& flags) {
  uint32_t value = 0;
  static_assert(sizeof(value) <= sizeof(flags), "flag storage mismatch");
  std::memcpy(&value, &flags, sizeof(value));
  return value;
}

void InitializeCompletionSlot(uint8_t* slot_cpu) {
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

bool QueryDriverStorePath(const KmtApi& api, D3DKMT_HANDLE adapter,
                          std::wstring* out_path) {
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
    return false;
  }

  if (query_info.OutputValueSize > kMaxDriverStorePathWarmupBytes) {
    return false;
  }
  alignas(D3DDDI_QUERYREGISTRY_INFO)
      uint8_t buffer[sizeof(D3DDDI_QUERYREGISTRY_INFO) +
                     kMaxDriverStorePathWarmupBytes] = {};
  auto* expanded = reinterpret_cast<D3DDDI_QUERYREGISTRY_INFO*>(buffer);
  expanded->QueryType = D3DDDI_QUERYREGISTRY_DRIVERSTOREPATH;
  query.pPrivateDriverData = expanded;
  query.PrivateDriverDataSize = static_cast<UINT>(
      sizeof(D3DDDI_QUERYREGISTRY_INFO) + query_info.OutputValueSize);
  status = api.query_adapter_info(&query);
  if (status != 0 ||
      expanded->Status != D3DDDI_QUERYREGISTRY_STATUS_SUCCESS ||
      expanded->OutputValueSize == 0) {
    return false;
  }
  if (out_path) {
    size_t wchar_count = expanded->OutputValueSize / sizeof(WCHAR);
    while (wchar_count > 0 && expanded->OutputString[wchar_count - 1] == 0) {
      --wchar_count;
    }
    out_path->assign(expanded->OutputString,
                     expanded->OutputString + wchar_count);
  }
  return true;
}

void QueryDriverStorePathForWarmup(const KmtApi& api, D3DKMT_HANDLE adapter) {
  QueryDriverStorePath(api, adapter, nullptr);
}

bool ReadFileToBytes(const wchar_t* path, std::vector<uint8_t>* out_bytes) {
  if (!path || !out_bytes) return false;
  HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) return false;
  LARGE_INTEGER size = {};
  if (!GetFileSizeEx(file, &size) || size.QuadPart <= 0 ||
      size.QuadPart > 1024 * 1024) {
    CloseHandle(file);
    return false;
  }
  out_bytes->resize(static_cast<size_t>(size.QuadPart));
  DWORD read = 0;
  BOOL ok =
      ReadFile(file, out_bytes->data(), static_cast<DWORD>(out_bytes->size()),
               &read, nullptr);
  CloseHandle(file);
  if (!ok || read != out_bytes->size()) {
    out_bytes->clear();
    return false;
  }
  return true;
}

std::wstring DecodeInfText(const std::vector<uint8_t>& bytes) {
  if (bytes.size() >= 2 &&
      ((bytes[0] == 0xFF && bytes[1] == 0xFE) ||
       (bytes[0] != 0 && bytes[1] == 0))) {
    size_t offset = bytes[0] == 0xFF && bytes[1] == 0xFE ? 2 : 0;
    size_t wchar_count = (bytes.size() - offset) / sizeof(wchar_t);
    return std::wstring(
        reinterpret_cast<const wchar_t*>(bytes.data() + offset),
        reinterpret_cast<const wchar_t*>(bytes.data() + offset) + wchar_count);
  }
  int wchar_count = MultiByteToWideChar(
      CP_UTF8, 0, reinterpret_cast<const char*>(bytes.data()),
      static_cast<int>(bytes.size()), nullptr, 0);
  UINT code_page = CP_UTF8;
  if (wchar_count == 0) {
    code_page = CP_ACP;
    wchar_count = MultiByteToWideChar(
        code_page, 0, reinterpret_cast<const char*>(bytes.data()),
        static_cast<int>(bytes.size()), nullptr, 0);
  }
  std::wstring text(static_cast<size_t>(wchar_count), L'\0');
  if (wchar_count > 0) {
    MultiByteToWideChar(code_page, 0,
                        reinterpret_cast<const char*>(bytes.data()),
                        static_cast<int>(bytes.size()), text.data(),
                        wchar_count);
  }
  return text;
}

std::string NarrowForTrace(const std::wstring& text) {
  if (text.empty()) return std::string();
  int size = WideCharToMultiByte(CP_UTF8, 0, text.data(),
                                 static_cast<int>(text.size()), nullptr, 0,
                                 nullptr, nullptr);
  if (size <= 0) return std::string("<unprintable>");
  std::string result(static_cast<size_t>(size), '\0');
  WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()),
                      result.data(), size, nullptr, nullptr);
  return result;
}

void NormalizeSystemRootPath(std::wstring* path) {
  if (!path) return;
  constexpr wchar_t kSystemRootPrefix[] = L"\\SystemRoot";
  constexpr size_t kSystemRootPrefixLength =
      sizeof(kSystemRootPrefix) / sizeof(kSystemRootPrefix[0]) - 1;
  if (path->size() < kSystemRootPrefixLength ||
      _wcsnicmp(path->c_str(), kSystemRootPrefix, kSystemRootPrefixLength) !=
          0) {
    return;
  }
  wchar_t windows_directory[MAX_PATH] = {};
  UINT length = GetWindowsDirectoryW(windows_directory, MAX_PATH);
  if (length == 0 || length >= MAX_PATH) return;
  path->replace(0, kSystemRootPrefixLength, windows_directory);
}

bool ParseDriverVersionText(const std::wstring& text, DriverVersion* out) {
  if (!out) return false;
  const std::wstring key = L"DriverVer";
  size_t pos = text.find(key);
  if (pos == std::wstring::npos) return false;
  pos = text.find(L',', pos + key.size());
  if (pos == std::wstring::npos) return false;
  ++pos;
  while (pos < text.size() && (text[pos] == L' ' || text[pos] == L'\t')) {
    ++pos;
  }

  uint32_t values[4] = {};
  for (size_t i = 0; i < 4; ++i) {
    if (pos >= text.size() || text[pos] < L'0' || text[pos] > L'9') {
      return false;
    }
    uint32_t value = 0;
    while (pos < text.size() && text[pos] >= L'0' && text[pos] <= L'9') {
      value = value * 10 + static_cast<uint32_t>(text[pos] - L'0');
      ++pos;
    }
    values[i] = value;
    if (i != 3) {
      if (pos >= text.size() || text[pos] != L'.') return false;
      ++pos;
    }
  }
  out->major = values[0];
  out->minor = values[1];
  out->build = values[2];
  out->revision = values[3];
  return true;
}

bool QueryDriverVersion(const KmtApi& api, D3DKMT_HANDLE adapter,
                        DriverVersion* out_version) {
  if (!out_version) return false;
  std::wstring driver_store_path;
  if (!QueryDriverStorePath(api, adapter, &driver_store_path) ||
      driver_store_path.empty()) {
    Trace("driver-version query failed: no DriverStore path");
    return false;
  }
  NormalizeSystemRootPath(&driver_store_path);
  const std::string original_driver_store_path =
      NarrowForTrace(driver_store_path);
  if (driver_store_path.back() != L'\\' && driver_store_path.back() != L'/') {
    driver_store_path.push_back(L'\\');
  }
  driver_store_path.append(L"kipudrv.inf");
  const std::string inf_path = NarrowForTrace(driver_store_path);

  std::vector<uint8_t> bytes;
  if (!ReadFileToBytes(driver_store_path.c_str(), &bytes)) {
    Trace("driver-version query failed: cannot read INF store_path='%s' "
          "inf_path='%s'",
          original_driver_store_path.c_str(), inf_path.c_str());
    return false;
  }
  if (!ParseDriverVersionText(DecodeInfText(bytes), out_version)) {
    Trace("driver-version query failed: cannot parse INF path='%s'",
          inf_path.c_str());
    return false;
  }
  Trace("driver-version query path='%s' version=%u.%u.%u.%u",
        inf_path.c_str(), out_version->major, out_version->minor,
        out_version->build, out_version->revision);
  return true;
}

bool UsesLegacyV2ContextLayout(const DriverVersion& version) {
  return version.major == 32 && version.minor == 0 && version.build == 203 &&
         version.revision < 314;
}

}  // namespace

McdmAbi SelectMcdmAbiForDriverVersion(McdmAbi probed_abi,
                                       bool has_driver_version,
                                       const DriverVersion& driver_version) {
  if (probed_abi == McdmAbi::legacy_v2 &&
      (!has_driver_version || !UsesLegacyV2ContextLayout(driver_version))) {
    return McdmAbi::legacy;
  }
  return probed_abi;
}

bool QueryMcdmAbiDiagnostics(const KmtApi& api, D3DKMT_HANDLE adapter,
                             McdmAbiDiagnostics* out_diagnostics,
                             Error* out_error) {
  if (!out_diagnostics) {
    SetError(out_error, "missing MCDM ABI diagnostics output");
    return false;
  }
  McdmAbiDiagnostics diagnostics = {};
  // Match XRT's legacy-driver negotiation: query the two-DWORD identity first.
  // Compact drivers reject that shape with STATUS_BUFFER_TOO_SMALL, in which
  // case we retry with the three-DWORD compact identity. Unknown identities fail
  // closed.
  constexpr uint32_t kLegacyV0Identity[2] = {0, 0};
  constexpr uint32_t kLegacyV2Identity[2] = {0, 2};
  constexpr uint32_t kLegacyV3Identity[2] = {0, 3};
  constexpr uint32_t kLegacyV4Identity[2] = {0, 4};
  constexpr uint32_t kCompactIdentity[3] = {0, 2, 0};
  constexpr uint32_t kCompactV3Identity[3] = {0, 3, 0};
  constexpr uint32_t kCompactV4Identity[3] = {0, 4, 0};
  uint32_t legacy_data[2] = {};
  D3DKMT_QUERYADAPTERINFO query = {};
  query.hAdapter = adapter;
  query.Type = KMTQAITYPE_UMDRIVERPRIVATE;
  query.pPrivateDriverData = legacy_data;
  query.PrivateDriverDataSize = sizeof(legacy_data);
  diagnostics.legacy_query_attempted = true;
  NTSTATUS legacy_status = api.query_adapter_info(&query);
  diagnostics.legacy_query_status = legacy_status;
  if (legacy_status != 0 && legacy_status != kStatusBufferTooSmall) {
    return CheckStatus("D3DKMTQueryAdapterInfo(UMDRIVERPRIVATE legacy)",
                       legacy_status, out_error);
  }

  if (legacy_status == 0) {
    const bool is_legacy_v0 =
        std::memcmp(legacy_data, kLegacyV0Identity,
                    sizeof(kLegacyV0Identity)) == 0;
    const bool is_legacy_v2 =
        std::memcmp(legacy_data, kLegacyV2Identity,
                    sizeof(kLegacyV2Identity)) == 0;
    const bool is_legacy_v3 =
        std::memcmp(legacy_data, kLegacyV3Identity,
                    sizeof(kLegacyV3Identity)) == 0;
    const bool is_legacy_v4 =
        std::memcmp(legacy_data, kLegacyV4Identity,
                    sizeof(kLegacyV4Identity)) == 0;
    if (!is_legacy_v0 && !is_legacy_v2 && !is_legacy_v3 && !is_legacy_v4) {
      SetErrorFormat(out_error,
                     "unsupported two-dword MCDM identity {%u, %u}",
                     legacy_data[0], legacy_data[1]);
      return false;
    }
    diagnostics.probed_abi = is_legacy_v0   ? McdmAbi::legacy_v0
                             : is_legacy_v2 ? McdmAbi::legacy_v2
                                            : McdmAbi::legacy;
    diagnostics.selected_abi = diagnostics.probed_abi;
    diagnostics.source = McdmAbiSource::identity_query;
    diagnostics.identity_words[0] = legacy_data[0];
    diagnostics.identity_words[1] = legacy_data[1];
    diagnostics.identity_word_count = 2;
    diagnostics.accepted_identity_count = 1;
    diagnostics.identity_accepted = true;
    diagnostics.driver_version_disambiguation_required = is_legacy_v2;
    *out_diagnostics = diagnostics;
    return true;
  }

  uint32_t compact_data[3] = {};
  query.pPrivateDriverData = compact_data;
  query.PrivateDriverDataSize = sizeof(compact_data);
  diagnostics.compact_query_attempted = true;
  NTSTATUS compact_status = api.query_adapter_info(&query);
  diagnostics.compact_query_status = compact_status;
  if (compact_status == kStatusBufferTooSmall) {
    SetError(out_error,
             "MCDM driver rejected both three-dword and two-dword identity "
             "queries");
    return false;
  }
  if (compact_status != 0) {
    return CheckStatus("D3DKMTQueryAdapterInfo(UMDRIVERPRIVATE compact)",
                       compact_status, out_error);
  }
  const bool is_compact_v2 =
      std::memcmp(compact_data, kCompactIdentity,
                  sizeof(kCompactIdentity)) == 0;
  const bool is_compact_v3 =
      std::memcmp(compact_data, kCompactV3Identity,
                  sizeof(kCompactV3Identity)) == 0;
  const bool is_compact_v4 =
      std::memcmp(compact_data, kCompactV4Identity,
                  sizeof(kCompactV4Identity)) == 0;
  if (!is_compact_v2 && !is_compact_v3 && !is_compact_v4) {
    SetErrorFormat(out_error,
                   "unsupported three-dword MCDM identity {%u, %u, %u}",
                   compact_data[0], compact_data[1], compact_data[2]);
    return false;
  }
  diagnostics.selected_abi = McdmAbi::compact;
  diagnostics.probed_abi = McdmAbi::compact;
  diagnostics.source = McdmAbiSource::identity_query;
  diagnostics.identity_words[0] = compact_data[0];
  diagnostics.identity_words[1] = compact_data[1];
  diagnostics.identity_words[2] = compact_data[2];
  diagnostics.identity_word_count = 3;
  diagnostics.accepted_identity_count = 1;
  diagnostics.identity_accepted = true;
  *out_diagnostics = diagnostics;
  return true;
}

bool QueryProbedMcdmAbi(const KmtApi& api, D3DKMT_HANDLE adapter,
                        McdmAbi* out_abi, Error* out_error) {
  McdmAbiDiagnostics diagnostics = {};
  if (!QueryMcdmAbiDiagnostics(api, adapter, &diagnostics, out_error)) {
    return false;
  }
  *out_abi = diagnostics.selected_abi;
  return true;
}

McdmAbiInfo GetMcdmAbiInfo(McdmAbi abi) {
  if (abi == McdmAbi::compact) {
    return {/*status_private_type=*/0x332c,
            /*status_policy=*/2,
            /*status_xcl_flags=*/0x02000000,
            /*submit_private_prefix_size=*/0x78,
            /*setup_private_size=*/0x280,
            /*pathb_private_size=*/0x278,
            /*pathb_packet_offset=*/0x78,
            /*chain_metadata_offset=*/0x58,
            /*pathb_bo_table_entry_count=*/6,
            /*status_has_gpu_va=*/true,
            /*sync_has_allocation_handle=*/false,
            /*command_aperture_code_slot_size=*/0x8000,
            /*command_aperture_write_publish_mode=*/
                CommandApertureWritePublishMode::cpu_cache_flush,
            /*command_aperture_code_publish_granularity=*/0x8000,
            /*command_aperture_residency_after_bootstrap=*/true,
            /*command_aperture_remap_after_write=*/false,
            // XRT 2.21 destroys both compact-ABI shared resources only after
            // they are no longer in use and waits for destruction to finish.
            /*shared_resource_destroy_flags=*/0x3,
            // Compact XRT leaves mapped VA ownership with the allocation.
            /*explicit_gpu_va_free_on_destroy=*/false};
  }
  if (abi == McdmAbi::legacy_v0) {
    return {/*status_private_type=*/0x332b,
            /*status_policy=*/0,
            /*status_xcl_flags=*/0,
            /*submit_private_prefix_size=*/0x58,
            /*setup_private_size=*/0x260,
            /*pathb_private_size=*/0x258,
            /*pathb_packet_offset=*/0x58,
            /*chain_metadata_offset=*/0x40,
            /*pathb_bo_table_entry_count=*/5,
            /*status_has_gpu_va=*/false,
            /*sync_has_allocation_handle=*/true,
            /*command_aperture_code_slot_size=*/0x8000,
            /*command_aperture_write_publish_mode=*/
            CommandApertureWritePublishMode::kmt_invalidate,
            /*command_aperture_code_publish_granularity=*/0,
            /*command_aperture_residency_after_bootstrap=*/false,
            /*command_aperture_remap_after_write=*/true,
            /*shared_resource_destroy_flags=*/0,
            /*explicit_gpu_va_free_on_destroy=*/true};
  }
  if (abi == McdmAbi::legacy_v2) {
    return {/*status_private_type=*/0x332b,
            /*status_policy=*/0,
            /*status_xcl_flags=*/0,
            /*submit_private_prefix_size=*/0x60,
            /*setup_private_size=*/0x268,
            /*pathb_private_size=*/0x260,
            /*pathb_packet_offset=*/0x60,
            /*chain_metadata_offset=*/0x48,
            /*pathb_bo_table_entry_count=*/5,
            /*status_has_gpu_va=*/false,
            /*sync_has_allocation_handle=*/true,
            /*command_aperture_code_slot_size=*/0x8000,
            /*command_aperture_write_publish_mode=*/
            CommandApertureWritePublishMode::kmt_invalidate,
            /*command_aperture_code_publish_granularity=*/0,
            /*command_aperture_residency_after_bootstrap=*/false,
            /*command_aperture_remap_after_write=*/true,
            /*shared_resource_destroy_flags=*/0,
            /*explicit_gpu_va_free_on_destroy=*/true};
  }
  return {/*status_private_type=*/0x332b,
          /*status_policy=*/0,
          /*status_xcl_flags=*/0,
          /*submit_private_prefix_size=*/0x68,
          /*setup_private_size=*/0x270,
          /*pathb_private_size=*/0x268,
          /*pathb_packet_offset=*/0x68,
          /*chain_metadata_offset=*/0x48,
          /*pathb_bo_table_entry_count=*/5,
          /*status_has_gpu_va=*/false,
          /*sync_has_allocation_handle=*/true,
          /*command_aperture_code_slot_size=*/0x8000,
          /*command_aperture_write_publish_mode=*/
              CommandApertureWritePublishMode::kmt_invalidate,
          /*command_aperture_code_publish_granularity=*/0,
          /*command_aperture_residency_after_bootstrap=*/false,
          /*command_aperture_remap_after_write=*/true,
          /*shared_resource_destroy_flags=*/0,
          /*explicit_gpu_va_free_on_destroy=*/true};
}

McdmPrivateData BuildPathBSetupPrivateData(
    McdmAbi mcdm_abi, const CommandAperture& aperture) {
  const McdmAbiInfo abi = GetMcdmAbiInfo(mcdm_abi);
  McdmPrivateData private_data = {};
  private_data.size = abi.setup_private_size;
  WriteU32(private_data.data, 0x00, 5);
  WriteU64(private_data.data, 0x28, aperture.allocation);
  if (abi.status_has_gpu_va) {
    WriteU64(private_data.data, 0x30, aperture.status_gpu_va);
    WriteU32(private_data.data, 0x38, 0);
    WriteU32(private_data.data, 0x3c, kQhdlCompletionSlotSize);
    WriteU64(private_data.data, 0x40,
             reinterpret_cast<uint64_t>(aperture.cpu_ptr));
  } else {
    WriteU32(private_data.data, 0x30, 0);
    WriteU32(private_data.data, 0x34, kQhdlCompletionSlotSize);
    WriteU64(private_data.data, 0x38,
             reinterpret_cast<uint64_t>(aperture.cpu_ptr));
  }
  WriteU32(private_data.data, abi.submit_private_prefix_size, 1);
  WriteU64(private_data.data, abi.submit_private_prefix_size + 8,
           aperture.gpu_va);
  return private_data;
}

McdmPrivateData BuildPathBSyncPrivateData(McdmAbi mcdm_abi,
                                          const CommandAperture& aperture,
                                          uint64_t offset) {
  const McdmAbiInfo abi = GetMcdmAbiInfo(mcdm_abi);
  McdmPrivateData private_data = {};
  private_data.size = abi.submit_private_prefix_size;
  WriteU32(private_data.data, 0x00, 9);
  if (abi.sync_has_allocation_handle) {
    WriteU64(private_data.data, 0x08, aperture.gpu_allocation);
  }
  WriteU64(private_data.data, 0x10, offset);
  return private_data;
}

McdmPrivateData BuildPathBSubmitPrivateData(
    McdmAbi mcdm_abi, const Buffer& exec_buffer,
    const Buffer& completion_ring, uint32_t completion_slot_offset,
    const void* completion_slot_cpu, const void* ert_packet,
    uint32_t ert_bytes, uint32_t command_state,
    const PathBChainSubmitInfo* chain_info) {
  const McdmAbiInfo abi = GetMcdmAbiInfo(mcdm_abi);
  McdmPrivateData private_data = {};
  if (!ert_packet || ert_bytes == 0 ||
      ert_bytes > abi.pathb_private_size - abi.pathb_packet_offset) {
    return private_data;
  }

  private_data.size = abi.pathb_private_size;
  const uint32_t effective_command_state =
      chain_info ? 6u : (command_state ? command_state : 3u);
  WriteU32(private_data.data, 0x00, effective_command_state);
  WriteU64(private_data.data, 0x08, exec_buffer.allocation);
  WriteU64(private_data.data, 0x10, ert_bytes);
  WriteU64(private_data.data, 0x28, completion_ring.allocation);
  if (abi.status_has_gpu_va) {
    WriteU64(private_data.data, 0x30, completion_ring.gpu_va);
    WriteU32(private_data.data, 0x38, completion_slot_offset);
    WriteU32(private_data.data, 0x3c, kQhdlCompletionSlotSize);
    WriteU64(private_data.data, 0x40,
             reinterpret_cast<uint64_t>(completion_slot_cpu));
  } else {
    WriteU32(private_data.data, 0x30, completion_slot_offset);
    WriteU32(private_data.data, 0x34, kQhdlCompletionSlotSize);
    WriteU64(private_data.data, 0x38,
             reinterpret_cast<uint64_t>(completion_slot_cpu));
  }
  if (chain_info) {
    WriteU64(private_data.data, abi.chain_metadata_offset,
             chain_info->descriptor_gpu_va);
    WriteU32(private_data.data, abi.chain_metadata_offset + 8,
             chain_info->descriptor_bytes);
    WriteU32(private_data.data, abi.chain_metadata_offset + 12,
             chain_info->command_count);
    WriteU32(private_data.data, abi.chain_metadata_offset + 16,
             mcdm_abi == McdmAbi::legacy_v0
                 ? 0
                 : chain_info->first_child_opcode);
  }
  std::memcpy(private_data.data + abi.pathb_packet_offset, ert_packet,
              ert_bytes);
  return private_data;
}

BufferKindInfo GetBufferKindInfo(BufferKind kind) {
  switch (kind) {
    case BufferKind::host_only:
      return {"host_only", 0x3329, 0x20000000};
    case BufferKind::cacheable:
      return {"cacheable", 0x3323, 0x01000000};
    case BufferKind::execbuf:
      return {"execbuf", 0x3328, 0x80000000};
    case BufferKind::context_private:
      return {"context_private", 0x332c, 0x02000000};
  }
  return {"host_only", 0x3329, 0x20000000};
}

uint64_t GetPathBChainChildHandle(const Device& device, Buffer* buffer) {
  if (!buffer) return 0;
  if (UsesLegacyCpuBufferHandles(device.mcdm_abi)) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(buffer->cpu_ptr));
  }

  // The compact miniport resolves a runlist child through this observed DDI
  // record. Do not expose the XRT C++ object whose stable prefix established
  // the contract: vtables, device pointers, and synchronization members are
  // deliberately absent here.
  CompactPathBChainHandleV1& handle = buffer->compact_chain_handle;
  handle = {};
  handle.requested_size = buffer->requested_size;
  handle.xcl_flags = GetBufferKindInfo(buffer->kind).xcl_flags;
  handle.allocation = buffer->allocation;
  handle.page_count = buffer->mapped_size / kPageSize;
  handle.gpu_va = buffer->gpu_va;
  handle.cpu_ptr =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(buffer->cpu_ptr));
  handle.observed_state_80 = 2;
  handle.observed_state_c8 = 0xffffffffu;
  handle.observed_state_d0 = 1;
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&handle));
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
  query_video_memory_info = ResolveKmtProc<PFND3DKMT_QUERYVIDEOMEMORYINFO>(
      "D3DKMTQueryVideoMemoryInfo");
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
  Trace("D3DKMTCreateDevice adapter=0x%08x status=0x%08x%s device=0x%08x",
        static_cast<unsigned>(adapter.handle), static_cast<uint32_t>(status),
        NtStatusSuffix(status), static_cast<unsigned>(create_device.hDevice));
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
  if (!QueryMcdmAbiDiagnostics(api, device.adapter,
                               &device.mcdm_abi_diagnostics, out_error)) {
    DestroyDevice(api, &device);
    return false;
  }
  device.has_driver_version =
      QueryDriverVersion(api, device.adapter, &device.driver_version);
  const McdmAbi probed_mcdm_abi = device.mcdm_abi_diagnostics.probed_abi;
  device.mcdm_abi = SelectMcdmAbiForDriverVersion(
      probed_mcdm_abi, device.has_driver_version,
      device.driver_version);
  device.mcdm_abi_diagnostics.selected_abi = device.mcdm_abi;
  device.mcdm_abi_diagnostics.driver_version_disambiguation_available =
      device.mcdm_abi_diagnostics.driver_version_disambiguation_required &&
      device.has_driver_version;
  Trace("selected MCDM ABI probed=%s selected=%s driver_version=%u.%u.%u.%u "
        "has_driver_version=%u",
        McdmAbiName(probed_mcdm_abi), McdmAbiName(device.mcdm_abi),
        device.driver_version.major, device.driver_version.minor,
        device.driver_version.build, device.driver_version.revision,
        device.has_driver_version ? 1u : 0u);
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
// being paged in. XRT tracks the maximum pending paging fence per device and
// drains it at lifecycle boundaries; this is the corresponding primitive.
bool WaitForPagingFenceCpu(const KmtApi& api, const Device& device,
                           UINT64 fence_value) {
  if (fence_value == 0) return true;
  Trace("wait-paging-cpu fence=0x%llx",
        static_cast<unsigned long long>(fence_value));
  D3DKMT_HANDLE objects[1] = {device.paging_sync_object};
  UINT64 values[1] = {fence_value};
  D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait = {};
  wait.hDevice = device.device;
  wait.ObjectCount = 1;
  wait.ObjectHandleArray = objects;
  wait.FenceValueArray = values;
  wait.hAsyncEvent = nullptr;
  NTSTATUS status = api.wait_from_cpu(&wait);
  Trace("wait-paging-cpu-ret fence=0x%llx status=0x%08x%s",
        static_cast<unsigned long long>(fence_value),
        static_cast<uint32_t>(status), NtStatusSuffix(status));
  return status == 0;
}

void RecordPendingPagingFence(const Device& device, UINT64 fence_value) {
  if (fence_value == 0) return;
  auto* const watermark = &device.pending_paging_fence_value;
  LONG64 observed = InterlockedCompareExchange64(watermark, 0, 0);
  const LONG64 requested = static_cast<LONG64>(fence_value);
  while (observed < requested) {
    const LONG64 prior =
        InterlockedCompareExchange64(watermark, requested, observed);
    if (prior == observed) break;
    observed = prior;
  }
}

bool WaitForPendingPagingBeforeSubmit(const KmtApi& api,
                                      const Device& device,
                                      Error* out_error) {
  const UINT64 fence_value = static_cast<UINT64>(InterlockedCompareExchange64(
      &device.pending_paging_fence_value, 0, 0));
  if (fence_value == 0) return true;
  Trace("pending-paging-before-submit fence=0x%llx",
        static_cast<unsigned long long>(fence_value));

  // Match XRT's submit wrapper: read the paging queue's shared completion
  // value first and enter KMT only while the device-wide watermark is pending.
  if (device.paging_fence_cpu) {
    const auto* const completed =
        static_cast<const volatile UINT64*>(device.paging_fence_cpu);
    if (*completed >= fence_value) {
      Trace("pending-paging-complete fence=0x%llx completed=0x%llx",
            static_cast<unsigned long long>(fence_value),
            static_cast<unsigned long long>(*completed));
      return true;
    }
    Trace("pending-paging-wait fence=0x%llx completed=0x%llx",
          static_cast<unsigned long long>(fence_value),
          static_cast<unsigned long long>(*completed));
  }
  if (WaitForPagingFenceCpu(api, device, fence_value)) return true;
  SetErrorFormat(
      out_error,
      "D3DKMTWaitForSynchronizationObjectFromCpu(pending paging before "
      "submit) failed for fence 0x%llx",
      static_cast<unsigned long long>(fence_value));
  return false;
}

bool SubmitCommandToHwQueueAfterPaging(
    const KmtApi& api, const Device& device,
    D3DKMT_SUBMITCOMMANDTOHWQUEUE* submit, const char* label,
    Error* out_error) {
  if (!WaitForPendingPagingBeforeSubmit(api, device, out_error)) return false;
  Trace("submit %s hwq=0x%08x fence=0x%llx cmd=0x%llx len=%u priv=%u",
        label, static_cast<unsigned>(submit->hHwQueue),
        static_cast<unsigned long long>(submit->HwQueueProgressFenceId),
        static_cast<unsigned long long>(submit->CommandBuffer),
        submit->CommandLength, submit->PrivateDriverDataSize);
  TraceBlob("submit-private", submit->pPrivateDriverData,
            submit->PrivateDriverDataSize);
  const NTSTATUS status = api.submit_command_to_hw_queue(submit);
  Trace("submit-ret %s status=0x%08x%s", label,
        static_cast<uint32_t>(status), NtStatusSuffix(status));
  return CheckStatus(label, status, out_error);
}

bool CreateBuffer(const KmtApi& api, const Device& device, BufferKind kind,
                  uint64_t size, Buffer* out_buffer, Error* out_error) {
  BufferKindInfo kind_info = GetBufferKindInfo(kind);
  const McdmAbiInfo abi = GetMcdmAbiInfo(device.mcdm_abi);
  // XRT's exec BO submit path asks the driver for the 4 KiB command BO plus
  // the negotiated private prefix used by SubmitCommandToHwQueue. The HAL
  // still sees only the 4 KiB command capacity.
  uint64_t requested_size = std::max<uint64_t>(size, 1);
  if (kind == BufferKind::execbuf) {
    // Match XRT exactly: it allocates a full 4 KiB command page plus the ABI
    // prefix for every exec BO, so each runlist BO
    // lands on its own page pair. We were allocating only the exact runlist
    // bytes (~0x148, single 4 KiB page), packing multiple parents' exec BOs
    // into one coherence granule and racing the firmware on multi-parent
    // re-runs. The HAL still sees only the logical command capacity.
    requested_size = std::max<uint64_t>(requested_size, 0x1000) +
                      abi.submit_private_prefix_size;
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
  if (kind == BufferKind::context_private) {
    create.Flags.CreateResource = 1;
    create.Flags.CreateShared = 1;
  }
  create.NumAllocations = 1;
  create.pAllocationInfo2 = &alloc_info;

  NTSTATUS status = api.create_allocation2(&create);
  Trace("create-buffer kind=%s size=%llu requested=%llu aligned=%llu pages=%llu "
        "private=0x%08x flags=0x%08x status=0x%08x%s alloc=0x%08x "
        "resource=0x%08x",
        kind_info.name, static_cast<unsigned long long>(size),
        static_cast<unsigned long long>(requested_size),
        static_cast<unsigned long long>(aligned_size),
        static_cast<unsigned long long>(size_pages), kind_info.private_type,
        kind_info.xcl_flags, static_cast<uint32_t>(status),
        NtStatusSuffix(status), static_cast<unsigned>(alloc_info.hAllocation),
        static_cast<unsigned>(create.hResource));
  if (!CheckStatus("D3DKMTCreateAllocation2", status, out_error)) {
    return false;
  }

  Buffer buffer = {};
  buffer.kind = kind;
  buffer.size = size;
  buffer.requested_size = requested_size;
  buffer.mapped_size = aligned_size;
  buffer.allocation = alloc_info.hAllocation;
  buffer.resource = create.hResource;

  D3DDDI_MAPGPUVIRTUALADDRESS map = {};
  map.hPagingQueue = device.paging_queue;
  map.hAllocation = buffer.allocation;
  map.SizeInPages = size_pages;
  map.Protection.Write = 1;
  status = api.map_gpu_virtual_address(&map);
  Trace("map-buffer kind=%s alloc=0x%08x pages=%llu status=0x%08x%s "
        "gpu_va=0x%llx fence=0x%llx",
        kind_info.name, static_cast<unsigned>(buffer.allocation),
        static_cast<unsigned long long>(map.SizeInPages),
        static_cast<uint32_t>(status), NtStatusSuffix(status),
        static_cast<unsigned long long>(map.VirtualAddress),
        static_cast<unsigned long long>(map.PagingFenceValue));
  if (!CheckStatusOrPending("D3DKMTMapGpuVirtualAddress", status, out_error)) {
    DestroyBuffer(api, device, &buffer);
    return false;
  }
  RecordPendingPagingFence(device, map.PagingFenceValue);
  buffer.gpu_va = map.VirtualAddress;

  D3DKMT_HANDLE resident_allocs[1] = {buffer.allocation};
  D3DDDI_MAKERESIDENT resident = {};
  resident.hPagingQueue = device.paging_queue;
  resident.NumAllocations = 1;
  resident.AllocationList = resident_allocs;
  ConfigureMakeResidentFlags(&resident);
  status = api.make_resident(&resident);
  Trace("resident-buffer kind=%s alloc=0x%08x status=0x%08x%s fence=0x%llx",
        kind_info.name, static_cast<unsigned>(buffer.allocation),
        static_cast<uint32_t>(status), NtStatusSuffix(status),
        static_cast<unsigned long long>(resident.PagingFenceValue));
  if (!CheckMakeResidentStatusOrPending(api, device, "D3DKMTMakeResident",
                                        status, resident, out_error)) {
    DestroyBuffer(api, device, &buffer);
    return false;
  }
  RecordPendingPagingFence(device, resident.PagingFenceValue);
  // Keep residency asynchronous, but preserve the paging fence so the target
  // GPU context can depend on it before consuming this BO. Dropping the fence
  // makes the first dispatch race STATUS_PENDING Map/MakeResident operations.
  buffer.paging_fence_value = resident.PagingFenceValue;

  D3DKMT_LOCK2 lock = {};
  lock.hDevice = device.device;
  lock.hAllocation = buffer.allocation;
  status = api.lock2(&lock);
  Trace("lock-buffer kind=%s alloc=0x%08x status=0x%08x%s cpu=%p",
        kind_info.name, static_cast<unsigned>(buffer.allocation),
        static_cast<uint32_t>(status), NtStatusSuffix(status), lock.pData);
  if (!CheckStatus("D3DKMTLock2", status, out_error)) {
    DestroyBuffer(api, device, &buffer);
    return false;
  }
  buffer.cpu_ptr = lock.pData;

  StatsAlloc(kind_info.name, buffer.mapped_size, buffer.allocation);
  *out_buffer = buffer;
  return true;
}

bool PublishBufferCpuWrites(const Buffer& buffer, uint64_t offset,
                            uint64_t length, Error* out_error) {
  return PublishCpuWriteRange(buffer.cpu_ptr, buffer.size, offset, length,
                              /*granularity=*/1, out_error);
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

bool PopulatePathBBoTable(
    const Device& device, void* command_bo, size_t command_bo_size,
    const D3DGPU_VIRTUAL_ADDRESS* real_bo_gpu_vas,
    size_t real_bo_entry_count, Error* out_error) {
  constexpr size_t kBoTableWordOffset = 11;
  constexpr size_t kBoTableWords = 2 * kMaxPathBBoTableEntries;
  constexpr size_t kBoTableEndBytes =
      (kBoTableWordOffset + kBoTableWords) * sizeof(uint32_t);
  if (!command_bo || command_bo_size < kBoTableEndBytes ||
      (real_bo_entry_count && !real_bo_gpu_vas)) {
    SetError(out_error, "PopulatePathBBoTable called with invalid storage");
    return false;
  }
  const size_t table_entry_count =
      GetMcdmAbiInfo(device.mcdm_abi).pathb_bo_table_entry_count;
  if (real_bo_entry_count > table_entry_count) {
    SetErrorFormat(out_error,
                   "path-B BO table has %zu real entries but ABI permits %zu",
                   real_bo_entry_count, table_entry_count);
    return false;
  }
  auto* words = static_cast<uint32_t*>(command_bo);
  std::fill(words + kBoTableWordOffset,
            words + kBoTableWordOffset + kBoTableWords, 0);
  auto write_entry = [&](size_t index, D3DGPU_VIRTUAL_ADDRESS gpu_va) {
    words[kBoTableWordOffset + 2 * index] = static_cast<uint32_t>(gpu_va);
    words[kBoTableWordOffset + 2 * index + 1] =
        static_cast<uint32_t>(gpu_va >> 32);
  };
  for (size_t i = 0; i < real_bo_entry_count; ++i) {
    write_entry(i, real_bo_gpu_vas[i]);
  }
  return true;
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

bool WaitForBufferResidency(const KmtApi& api, const Device& device,
                            const Context& context, const Buffer& buffer,
                            const char* label, Error* out_error) {
  if (buffer.paging_fence_value == 0) return true;
  if (device.mcdm_abi == McdmAbi::compact) {
    // Compact XRT tracks all Map/MakeResident fences in one device-wide
    // watermark and drains it in the HW-queue submit wrapper. Adding per-BO
    // GPU waits changes the first-submit lifecycle and is not part of that ABI.
    Trace("wait-buffer-residency-skip label=%s alloc=0x%08x fence=0x%llx",
          label ? label : "", static_cast<unsigned>(buffer.allocation),
          static_cast<unsigned long long>(buffer.paging_fence_value));
    return true;
  }

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
  Trace("wait-buffer-residency label=%s alloc=0x%08x fence=0x%llx "
        "status=0x%08x%s",
        label ? label : "", static_cast<unsigned>(buffer.allocation),
        static_cast<unsigned long long>(buffer.paging_fence_value),
        static_cast<uint32_t>(status), NtStatusSuffix(status));
  return CheckStatus(call_name, status, out_error);
}

void DestroyBuffer(const KmtApi& api, const Device& device, Buffer* buffer) {
  if (!buffer || !buffer->allocation) return;
  const BufferKindInfo kind_info = GetBufferKindInfo(buffer->kind);
  StatsFree(kind_info.name,
            buffer->mapped_size ? buffer->mapped_size : buffer->size,
            buffer->allocation);
  if (buffer->cpu_ptr) {
    D3DKMT_UNLOCK2 unlock = {};
    unlock.hDevice = device.device;
    unlock.hAllocation = buffer->allocation;
    api.unlock2(&unlock);
    buffer->cpu_ptr = nullptr;
  }
  const McdmAbiInfo abi = GetMcdmAbiInfo(device.mcdm_abi);
  if (buffer->gpu_va && abi.explicit_gpu_va_free_on_destroy) {
    D3DKMT_FREEGPUVIRTUALADDRESS free_va = {};
    free_va.hAdapter = device.adapter;
    free_va.BaseAddress = buffer->gpu_va;
    free_va.Size = buffer->mapped_size ? buffer->mapped_size : buffer->size;
    api.free_gpu_virtual_address(&free_va);
  }
  buffer->gpu_va = 0;
  D3DKMT_DESTROYALLOCATION2 destroy = {};
  destroy.hDevice = device.device;
  destroy.Flags.AssumeNotInUse = 1;
  D3DKMT_HANDLE allocs[1] = {buffer->allocation};
  if (buffer->resource) {
    // CreateShared buffers are owned by their resource; destroy that.
    destroy.hResource = buffer->resource;
    const uint32_t resource_flags = abi.shared_resource_destroy_flags;
    if (resource_flags) destroy.Flags.Value = resource_flags;
  } else {
    destroy.AllocationCount = 1;
    destroy.phAllocationList = allocs;
  }
  api.destroy_allocation2(&destroy);
  buffer->allocation = 0;
  buffer->resource = 0;
  buffer->size = 0;
  buffer->requested_size = 0;
  buffer->mapped_size = 0;
  buffer->compact_chain_handle = {};
}

void DestroyStatusRing(const KmtApi& api, const Device& device,
                       Context* context) {
  if (!context) return;
  if (context->completion_ring_owned) {
    DestroyBuffer(api, device, &context->completion_ring);
  } else {
    context->completion_ring = {};
  }
  context->completion_ring_resource = 0;
  context->completion_ring_ready = false;
  context->completion_ring_owned = false;
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
  create_context.ClientHint = kClientHintVitis;
  NTSTATUS status = api.create_context_virtual(&create_context);
  Trace("D3DKMTCreateContextVirtual device=0x%08x status=0x%08x%s "
        "context=0x%08x private_size=%u",
        static_cast<unsigned>(device.device), static_cast<uint32_t>(status),
        NtStatusSuffix(status), static_cast<unsigned>(create_context.hContext),
        create_context.PrivateDriverDataSize);
  TraceBlob("create-context-private", create_context.pPrivateDriverData,
            create_context.PrivateDriverDataSize);
  if (!CheckStatus("D3DKMTCreateContextVirtual", status, out_error)) {
    return false;
  }
  context.context = create_context.hContext;
  const size_t cookie_offset =
      ContextCommandApertureCookieOffset(device.mcdm_abi);
  if (private_data_size >=
      cookie_offset + sizeof(context.command_aperture_cookie)) {
    std::memcpy(&context.command_aperture_cookie,
                private_data + cookie_offset,
                sizeof(context.command_aperture_cookie));
  }

  D3DKMT_CREATEHWQUEUE create_queue = {};
  create_queue.hHwContext = context.context;
  status = api.create_hw_queue(&create_queue);
  Trace("D3DKMTCreateHwQueue context=0x%08x status=0x%08x%s hwq=0x%08x "
        "progress_fence=0x%08x",
        static_cast<unsigned>(context.context), static_cast<uint32_t>(status),
        NtStatusSuffix(status), static_cast<unsigned>(create_queue.hHwQueue),
        static_cast<unsigned>(create_queue.hHwQueueProgressFence));
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

static void DestroyContextHwQueue(const KmtApi& api, Context* context) {
  if (context->hw_queue) {
    D3DKMT_DESTROYHWQUEUE destroy_queue = {};
    destroy_queue.hHwQueue = context->hw_queue;
    api.destroy_hw_queue(&destroy_queue);
    context->hw_queue = 0;
  }
}

static void DestroyContextAfterHwQueue(const KmtApi& api,
                                       const Device& device,
                                       Context* context) {
  // The HW queue can retain the status allocation until queue destruction has
  // completed. Release a context-owned ring only after its final queue user.
  DestroyStatusRing(api, device, context);
  DestroyBuffer(api, device, &context->context_private_buffer);
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

void DestroyContext(const KmtApi& api, const Device& device, Context* context) {
  if (!context) return;
  DestroyContextHwQueue(api, context);
  DestroyContextAfterHwQueue(api, device, context);
}

bool CreateCommandAperture(const KmtApi& api, const Device& device,
                           const Context& context,
                           CommandAperture* out_aperture, Error* out_error) {
  if (!out_aperture) {
    SetError(out_error, "CreateCommandAperture called with null output");
    return false;
  }

  const McdmAbiInfo abi = GetMcdmAbiInfo(device.mcdm_abi);
  AllocPrivate command_private = {};
  command_private.requested_size = kCommandApertureAllocationSize;
  command_private.aligned_size = kCommandApertureAllocationSize;
  command_private.private_type = abi.status_private_type;
  command_private.policy = abi.status_policy;
  command_private.xcl_flags = abi.status_xcl_flags;

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
  Trace("create-command-aperture status-object requested=%llu aligned=%llu "
        "private=0x%x flags=0x%x status=0x%08x%s alloc=0x%08x resource=0x%08x",
        static_cast<unsigned long long>(command_private.requested_size),
        static_cast<unsigned long long>(command_private.aligned_size),
        static_cast<unsigned>(command_private.private_type),
        static_cast<unsigned>(command_private.xcl_flags),
        static_cast<uint32_t>(status), NtStatusSuffix(status),
        static_cast<unsigned>(command_info.hAllocation),
        static_cast<unsigned>(create_command.hResource));
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
  Trace("lock-command-aperture status-object alloc=0x%08x status=0x%08x%s "
        "cpu=%p",
        static_cast<unsigned>(aperture.allocation),
        static_cast<uint32_t>(status), NtStatusSuffix(status), lock.pData);
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

  // The compact ABI exposes the status object to firmware by both allocation
  // handle and GPU VA. The legacy ABI uses only the allocation handle.
  if (abi.status_has_gpu_va) {
    D3DDDI_MAPGPUVIRTUALADDRESS status_map = {};
    status_map.hPagingQueue = device.paging_queue;
    status_map.hAllocation = aperture.allocation;
    status_map.SizeInPages = kCommandApertureAllocationSize / kPageSize;
    status_map.Protection.Write = 1;
    status = api.map_gpu_virtual_address(&status_map);
    Trace("map-command-aperture status-object alloc=0x%08x pages=%llu "
          "status=0x%08x%s gpu_va=0x%llx fence=0x%llx",
          static_cast<unsigned>(aperture.allocation),
          static_cast<unsigned long long>(status_map.SizeInPages),
          static_cast<uint32_t>(status), NtStatusSuffix(status),
          static_cast<unsigned long long>(status_map.VirtualAddress),
          static_cast<unsigned long long>(status_map.PagingFenceValue));
    if (!CheckStatusOrPending("D3DKMTMapGpuVirtualAddress(status object)",
                              status, out_error)) {
      DestroyCommandAperture(api, device, &aperture);
      return false;
    }
    RecordPendingPagingFence(device, status_map.PagingFenceValue);
    aperture.status_gpu_va = status_map.VirtualAddress;

    D3DKMT_HANDLE status_allocations[1] = {aperture.allocation};
    D3DDDI_MAKERESIDENT status_resident = {};
    status_resident.hPagingQueue = device.paging_queue;
    status_resident.NumAllocations = 1;
    status_resident.AllocationList = status_allocations;
    ConfigureMakeResidentFlags(&status_resident);
    status = api.make_resident(&status_resident);
    Trace("resident-command-aperture status-object alloc=0x%08x "
          "status=0x%08x%s fence=0x%llx",
          static_cast<unsigned>(aperture.allocation),
          static_cast<uint32_t>(status), NtStatusSuffix(status),
          static_cast<unsigned long long>(status_resident.PagingFenceValue));
    if (!CheckMakeResidentStatusOrPending(
            api, device, "D3DKMTMakeResident(status object)", status,
            status_resident, out_error)) {
      DestroyCommandAperture(api, device, &aperture);
      return false;
    }
    RecordPendingPagingFence(device, status_resident.PagingFenceValue);
  }

  // XRT creates the 64 MiB command aperture as a separate cacheable allocation
  // and folds the driver context writeback cookie into xcl_flags:
  //   xcl_flags = 0x01000001 | (context_blob[0x40] << 16)
  // The transaction instruction stream lives inside this same BO at the
  // offset selected by the negotiated ABI.
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
  Trace("create-command-aperture gpu requested=%llu aligned=%llu private=0x%x "
        "policy=0x%x flags=0x%x status=0x%08x%s alloc=0x%08x",
        static_cast<unsigned long long>(gpu_private.requested_size),
        static_cast<unsigned long long>(gpu_private.aligned_size),
        static_cast<unsigned>(gpu_private.private_type),
        static_cast<unsigned>(gpu_private.policy),
        static_cast<unsigned>(gpu_private.xcl_flags),
        static_cast<uint32_t>(status), NtStatusSuffix(status),
        static_cast<unsigned>(gpu_info.hAllocation));
  if (!CheckStatus("D3DKMTCreateAllocation2(command aperture gpu)", status,
                   out_error)) {
    DestroyCommandAperture(api, device, &aperture);
    return false;
  }
  aperture.gpu_allocation = gpu_info.hAllocation;
  StatsAlloc("command-aperture-status", kCommandApertureAllocationSize,
             aperture.allocation);
  StatsAlloc("command-aperture-gpu", kCommandApertureGpuVaSize,
             aperture.gpu_allocation);

  D3DDDI_MAPGPUVIRTUALADDRESS map = {};
  map.hPagingQueue = device.paging_queue;
  map.hAllocation = aperture.gpu_allocation;
  map.SizeInPages = kCommandApertureGpuVaSize / kPageSize;
  map.Protection.Write = 1;
  status = api.map_gpu_virtual_address(&map);
  Trace("map-command-aperture gpu alloc=0x%08x pages=%llu status=0x%08x%s "
        "gpu_va=0x%llx fence=0x%llx",
        static_cast<unsigned>(aperture.gpu_allocation),
        static_cast<unsigned long long>(map.SizeInPages),
        static_cast<uint32_t>(status), NtStatusSuffix(status),
        static_cast<unsigned long long>(map.VirtualAddress),
        static_cast<unsigned long long>(map.PagingFenceValue));
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
  RecordPendingPagingFence(device, map.PagingFenceValue);

  aperture.gpu_va = map.VirtualAddress;
  if (aperture.gpu_va != kCommandApertureGpuVaBase) {
    SetErrorFormat(
        out_error,
        "command aperture mapped at unexpected allocator-selected VA "
        "0x%llx; MCDM protocol requires 0x%llx",
        static_cast<unsigned long long>(aperture.gpu_va),
        static_cast<unsigned long long>(kCommandApertureGpuVaBase));
    DestroyCommandAperture(api, device, &aperture);
    return false;
  }

  if (!abi.command_aperture_residency_after_bootstrap) {
    D3DKMT_HANDLE resident_allocs[1] = {aperture.gpu_allocation};
    D3DDDI_MAKERESIDENT resident = {};
    resident.hPagingQueue = device.paging_queue;
    resident.NumAllocations = 1;
    resident.AllocationList = resident_allocs;
    ConfigureMakeResidentFlags(&resident);
    status = api.make_resident(&resident);
    Trace("resident-command-aperture gpu alloc=0x%08x status=0x%08x%s "
          "fence=0x%llx",
          static_cast<unsigned>(aperture.gpu_allocation),
          static_cast<uint32_t>(status), NtStatusSuffix(status),
          static_cast<unsigned long long>(resident.PagingFenceValue));
    if (!CheckMakeResidentStatusOrPending(
            api, device, "D3DKMTMakeResident(command aperture)", status,
            resident, out_error)) {
      DestroyCommandAperture(api, device, &aperture);
      return false;
    }
    RecordPendingPagingFence(device, resident.PagingFenceValue);
    if (!WaitForPagingFenceCpu(api, device, resident.PagingFenceValue)) {
      SetError(out_error,
               "D3DKMTWaitForSynchronizationObjectFromCpu(command aperture "
               "resident) failed");
      DestroyCommandAperture(api, device, &aperture);
      return false;
    }
  }

  aperture.code_allocation = aperture.gpu_allocation;
  // The final code range is selected from the context setup payload after the
  // aperture bootstrap. Until then, expose the entire mapping.
  aperture.code_offset = 0;
  aperture.code_gpu_va = aperture.gpu_va + aperture.code_offset;
  aperture.code_size = aperture.gpu_va_size - aperture.code_offset;

  *out_aperture = aperture;
  return true;
}

bool ConfigurePathBCodeRangeForSetupPayload(
    McdmAbi mcdm_abi, size_t aperture_payload_size,
    CommandAperture* aperture, Error* out_error) {
  if (!aperture) {
    SetError(out_error,
             "ConfigurePathBCodeRangeForSetupPayload called with null aperture");
    return false;
  }

  const McdmAbiInfo abi = GetMcdmAbiInfo(mcdm_abi);
  const uint64_t slot_size = abi.command_aperture_code_slot_size;
  if (!slot_size || (slot_size & (slot_size - 1)) != 0) {
    SetError(out_error, "invalid command-aperture slot size");
    return false;
  }
  const uint64_t payload_size = static_cast<uint64_t>(aperture_payload_size);
  if (payload_size >
      std::numeric_limits<uint64_t>::max() - (slot_size - 1)) {
    SetError(out_error, "command-aperture setup payload size overflows");
    return false;
  }
  const uint64_t code_offset =
      (payload_size + slot_size - 1) & ~(slot_size - 1);
  if (code_offset >= aperture->gpu_va_size) {
    SetError(out_error,
             "command-aperture setup payload leaves no valid code range");
    return false;
  }
  if (aperture->gpu_va >
      std::numeric_limits<uint64_t>::max() - code_offset) {
    SetError(out_error, "command-aperture code GPU address overflows");
    return false;
  }

  aperture->code_allocation = aperture->gpu_allocation;
  aperture->code_offset = code_offset;
  aperture->code_gpu_va = aperture->gpu_va + code_offset;
  aperture->code_cpu_ptr =
      aperture->gpu_cpu_ptr
          ? static_cast<uint8_t*>(aperture->gpu_cpu_ptr) + code_offset
          : nullptr;
  aperture->code_size = aperture->gpu_va_size - code_offset;
  return true;
}

bool ReleaseCommandApertureGpuMapping(const KmtApi& api, const Device& device,
                                      CommandAperture* aperture,
                                      Error* out_error) {
  if (!aperture || !aperture->gpu_va) return true;
  D3DKMT_FREEGPUVIRTUALADDRESS free_va = {};
  free_va.hAdapter = device.adapter;
  free_va.BaseAddress = aperture->gpu_va;
  free_va.Size = aperture->gpu_va_size;
  NTSTATUS status = api.free_gpu_virtual_address(&free_va);
  if (!CheckStatus("D3DKMTFreeGpuVirtualAddress(command aperture)", status,
                   out_error)) {
    return false;
  }
  aperture->gpu_va = 0;
  aperture->code_gpu_va = 0;
  return true;
}

bool EnsureCommandApertureGpuMapping(const KmtApi& api, const Device& device,
                                     CommandAperture* aperture,
                                     Error* out_error) {
  if (!aperture || !aperture->gpu_allocation) {
    SetError(out_error,
             "EnsureCommandApertureGpuMapping called before aperture setup");
    return false;
  }
  if (aperture->gpu_va) {
    if (aperture->gpu_va != kCommandApertureGpuVaBase) {
      SetErrorFormat(out_error,
                     "command aperture mapped at unexpected VA 0x%llx; expected 0x%llx",
                     static_cast<unsigned long long>(aperture->gpu_va),
                     static_cast<unsigned long long>(kCommandApertureGpuVaBase));
      return false;
    }
    aperture->code_allocation = aperture->gpu_allocation;
    aperture->code_gpu_va = aperture->gpu_va + aperture->code_offset;
    aperture->code_size = aperture->gpu_va_size - aperture->code_offset;
    return true;
  }

  D3DDDI_MAPGPUVIRTUALADDRESS map = {};
  map.hPagingQueue = device.paging_queue;
  map.hAllocation = aperture->gpu_allocation;
  map.SizeInPages = kCommandApertureGpuVaSize / kPageSize;
  map.Protection.Write = 1;
  NTSTATUS status = api.map_gpu_virtual_address(&map);
  Trace("map-command-aperture remap alloc=0x%08x pages=%llu status=0x%08x%s "
        "gpu_va=0x%llx fence=0x%llx",
        static_cast<unsigned>(aperture->gpu_allocation),
        static_cast<unsigned long long>(map.SizeInPages),
        static_cast<uint32_t>(status), NtStatusSuffix(status),
        static_cast<unsigned long long>(map.VirtualAddress),
        static_cast<unsigned long long>(map.PagingFenceValue));
  if (status != 0 && status != kStatusPending) {
    SetErrorFormat(out_error,
                   "D3DKMTMapGpuVirtualAddress(command aperture remap) failed with 0x%08x%s allocation=0x%08x pages=0x%llx",
                   static_cast<uint32_t>(status), NtStatusSuffix(status),
                   static_cast<unsigned>(aperture->gpu_allocation),
                   static_cast<unsigned long long>(map.SizeInPages));
    return false;
  }
  RecordPendingPagingFence(device, map.PagingFenceValue);
  if (!WaitForPagingFenceCpu(api, device, map.PagingFenceValue)) {
    SetError(out_error,
             "D3DKMTWaitForSynchronizationObjectFromCpu(command aperture remap) failed");
    return false;
  }
  aperture->gpu_va = map.VirtualAddress;
  if (aperture->gpu_va != kCommandApertureGpuVaBase) {
    SetErrorFormat(out_error,
                   "command aperture remapped at unexpected VA 0x%llx; expected 0x%llx",
                   static_cast<unsigned long long>(aperture->gpu_va),
                   static_cast<unsigned long long>(kCommandApertureGpuVaBase));
    ReleaseCommandApertureGpuMapping(api, device, aperture, out_error);
    return false;
  }

  D3DKMT_HANDLE resident_allocs[1] = {aperture->gpu_allocation};
  D3DDDI_MAKERESIDENT resident = {};
  resident.hPagingQueue = device.paging_queue;
  resident.NumAllocations = 1;
  resident.AllocationList = resident_allocs;
  ConfigureMakeResidentFlags(&resident);
  status = api.make_resident(&resident);
  Trace("resident-command-aperture remap alloc=0x%08x status=0x%08x%s "
        "fence=0x%llx",
        static_cast<unsigned>(aperture->gpu_allocation),
        static_cast<uint32_t>(status), NtStatusSuffix(status),
        static_cast<unsigned long long>(resident.PagingFenceValue));
  if (!CheckMakeResidentStatusOrPending(
          api, device, "D3DKMTMakeResident(command aperture remap)", status,
          resident, out_error)) {
    ReleaseCommandApertureGpuMapping(api, device, aperture, out_error);
    return false;
  }
  RecordPendingPagingFence(device, resident.PagingFenceValue);
  if (!WaitForPagingFenceCpu(api, device, resident.PagingFenceValue)) {
    SetError(out_error,
             "D3DKMTWaitForSynchronizationObjectFromCpu(command aperture remap resident) failed");
    ReleaseCommandApertureGpuMapping(api, device, aperture, out_error);
    return false;
  }

  aperture->code_allocation = aperture->gpu_allocation;
  aperture->code_gpu_va = aperture->gpu_va + aperture->code_offset;
  aperture->code_size = aperture->gpu_va_size - aperture->code_offset;
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
  Trace("lock-command-aperture gpu alloc=0x%08x status=0x%08x%s cpu=%p",
        static_cast<unsigned>(aperture->gpu_allocation),
        static_cast<uint32_t>(status), NtStatusSuffix(status),
        gpu_lock.pData);
  if (!CheckStatus("D3DKMTLock2(command aperture gpu)", status, out_error)) {
    return false;
  }
  aperture->gpu_cpu_ptr = gpu_lock.pData;
  if (aperture->gpu_cpu_ptr) {
    aperture->code_cpu_ptr =
        static_cast<uint8_t*>(aperture->gpu_cpu_ptr) +
        aperture->code_offset;
  }

  D3DKMT_INVALIDATECACHE invalidate = {};
  invalidate.hDevice = device.device;
  invalidate.hAllocation = aperture->gpu_allocation;
  invalidate.Offset = 0;
  invalidate.Length = aperture->gpu_va_size;
  status = api.invalidate_cache(&invalidate);
  Trace("invalidate-command-aperture gpu alloc=0x%08x bytes=%llu "
        "status=0x%08x%s",
        static_cast<unsigned>(aperture->gpu_allocation),
        static_cast<unsigned long long>(aperture->gpu_va_size),
        static_cast<uint32_t>(status), NtStatusSuffix(status));
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

  const McdmAbiInfo abi = GetMcdmAbiInfo(device.mcdm_abi);
  std::array<uint8_t, kMaxSubmitPrivatePrefixSize> submit_private = {};
  WriteU32(submit_private.data(), 0x00, 2);
  WriteU64(submit_private.data(), 0x08, aperture->gpu_allocation);
  WriteU64(submit_private.data(), 0x10, aperture->gpu_va);

  uint64_t fence_id = context->next_fence_id++;
  D3DKMT_SUBMITCOMMANDTOHWQUEUE submit = {};
  submit.hHwQueue = context->hw_queue;
  submit.HwQueueProgressFenceId = fence_id;
  submit.CommandBuffer = aperture->gpu_va;
  submit.CommandLength = static_cast<UINT>(aperture->gpu_va_size);
  submit.PrivateDriverDataSize = abi.submit_private_prefix_size;
  submit.pPrivateDriverData = submit_private.data();
  if (!SubmitCommandToHwQueueAfterPaging(
          api, device, &submit, "D3DKMTSubmitCommandToHwQueue", out_error)) {
    return false;
  }

  // XRT locks and invalidates the 64 MiB aperture only after the opcode-2
  // bootstrap submit. Do the same so later CPU writes to the negotiated code
  // view use the same MCDM object state as the reference runtime.
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
  Trace("wait-hwq-cpu %s fence=0x%llx",
        label ? label : "",
        static_cast<unsigned long long>(fence_id));
  NTSTATUS status = api.wait_from_cpu(&wait);
  Trace("wait-hwq-cpu-ret %s fence=0x%llx status=0x%08x%s",
        label ? label : "",
        static_cast<unsigned long long>(fence_id),
        static_cast<uint32_t>(status), NtStatusSuffix(status));
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

  const McdmAbiInfo abi = GetMcdmAbiInfo(device.mcdm_abi);
  std::array<uint8_t, kMaxSubmitPrivatePrefixSize> bootstrap_private = {};
  WriteU32(bootstrap_private.data(), 0x00, 2);
  WriteU64(bootstrap_private.data(), 0x08, aperture->gpu_allocation);
  WriteU64(bootstrap_private.data(), 0x10, aperture->gpu_va);

  uint64_t fence_id = context->next_fence_id++;
  D3DKMT_SUBMITCOMMANDTOHWQUEUE submit = {};
  submit.hHwQueue = context->hw_queue;
  submit.HwQueueProgressFenceId = fence_id;
  submit.CommandBuffer = aperture->gpu_va;
  submit.CommandLength = static_cast<UINT>(aperture->gpu_va_size);
  submit.PrivateDriverDataSize = abi.submit_private_prefix_size;
  submit.pPrivateDriverData = bootstrap_private.data();
  if (!SubmitCommandToHwQueueAfterPaging(
          api, device, &submit,
          "D3DKMTSubmitCommandToHwQueue(pathb bootstrap)", out_error)) {
    return false;
  }
  NTSTATUS status = 0;
  D3DKMT_HANDLE resident_allocs[1] = {};
  D3DDDI_MAKERESIDENT resident = {};
  if (abi.command_aperture_residency_after_bootstrap) {
    resident_allocs[0] = aperture->gpu_allocation;
    resident.hPagingQueue = device.paging_queue;
    resident.NumAllocations = 1;
    resident.AllocationList = resident_allocs;
    ConfigureMakeResidentFlags(&resident);
    status = api.make_resident(&resident);
    Trace("resident-command-aperture after-bootstrap alloc=0x%08x "
          "status=0x%08x%s fence=0x%llx",
          static_cast<unsigned>(aperture->gpu_allocation),
          static_cast<uint32_t>(status), NtStatusSuffix(status),
          static_cast<unsigned long long>(resident.PagingFenceValue));
    if (!CheckMakeResidentStatusOrPending(
            api, device, "D3DKMTMakeResident(command aperture after bootstrap)",
            status, resident, out_error)) {
      return false;
    }
    RecordPendingPagingFence(device, resident.PagingFenceValue);
  }
  if (!LockCommandApertureGpuAfterBootstrap(api, device, aperture, out_error)) {
    return false;
  }
  if (!ConfigurePathBCodeRangeForSetupPayload(
          device.mcdm_abi, aperture_payload_size, aperture, out_error)) {
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
    // Publish the setup PDI after the final CPU write using the negotiated
    // aperture policy. A process write barrier orders stores but does not make
    // dirty lines in this cacheable Lock2 mapping visible to the NPU.
    if (!CommitPathBCodeWrite(api, device, *aperture, /*offset=*/0,
                              aperture_payload_size, out_error)) {
      return false;
    }
  }

  McdmPrivateData setup_private =
      BuildPathBSetupPrivateData(device.mcdm_abi, *aperture);

  // The setup command owns status slot 0. XRT initializes that qhdl record
  // before submit and consumes it after the queue fence is reached; doing the
  // same prevents a later command from observing an incompletely retired
  // context setup.
  uint8_t* setup_slot = static_cast<uint8_t*>(aperture->cpu_ptr);
  InitializeCompletionSlot(setup_slot);

  fence_id = context->next_fence_id++;
  submit = {};
  submit.hHwQueue = context->hw_queue;
  submit.HwQueueProgressFenceId = fence_id;
  submit.CommandBuffer = 0;
  submit.CommandLength = 0;
  submit.PrivateDriverDataSize = setup_private.size;
  submit.pPrivateDriverData = setup_private.data;
  if (!SubmitCommandToHwQueueAfterPaging(
          api, device, &submit,
          "D3DKMTSubmitCommandToHwQueue(pathb setup5)", out_error)) {
    return false;
  }
  if (!WaitForHwQueueFenceCpu(
          api, device, *context, fence_id,
          "D3DKMTWaitForSynchronizationObjectFromCpu(pathb setup5)",
          out_error)) {
    return false;
  }
  std::atomic_thread_fence(std::memory_order_seq_cst);
  uint32_t setup_state = 0;
  std::memcpy(&setup_state, setup_slot, sizeof(setup_state));
  if (device.mcdm_abi == McdmAbi::compact && setup_state != 1) {
    SetErrorFormat(out_error,
                   "pathb setup did not complete after fence wait: "
                   "expected compact setup record 1, got 0x%08x",
                   setup_state);
    return false;
  }
  return true;
}

bool SubmitPathBApertureSync(const KmtApi& api, const Device& device,
                             Context* context, const CommandAperture& aperture,
                             uint64_t offset, bool wait_for_cpu,
                             Error* out_error) {
  if (!context || !context->hw_queue || !aperture.gpu_allocation) {
    SetError(out_error, "SubmitPathBApertureSync called before aperture setup");
    return false;
  }
  McdmPrivateData sync_private =
      BuildPathBSyncPrivateData(device.mcdm_abi, aperture, offset);

  uint64_t fence_id = context->next_fence_id++;
  D3DKMT_SUBMITCOMMANDTOHWQUEUE submit = {};
  submit.hHwQueue = context->hw_queue;
  submit.HwQueueProgressFenceId = fence_id;
  submit.CommandBuffer = 0;
  submit.CommandLength = 0;
  submit.PrivateDriverDataSize = sync_private.size;
  submit.pPrivateDriverData = sync_private.data;
  Trace("pathb-sync9 offset=0x%llx wait=%u alloc=0x%08x aperture_va=0x%llx "
        "code_offset=0x%llx code_size=0x%llx private_size=%u",
        static_cast<unsigned long long>(offset), wait_for_cpu ? 1u : 0u,
        static_cast<unsigned>(aperture.gpu_allocation),
        static_cast<unsigned long long>(aperture.gpu_va),
        static_cast<unsigned long long>(aperture.code_offset),
        static_cast<unsigned long long>(aperture.code_size),
        static_cast<unsigned>(sync_private.size));
  Error submit_error = {};
  if (!SubmitCommandToHwQueueAfterPaging(
          api, device, &submit,
          "D3DKMTSubmitCommandToHwQueue(pathb sync9)", &submit_error)) {
    SetErrorFormat(out_error,
                   "D3DKMTSubmitCommandToHwQueue(pathb sync9) failed at "
                   "offset=0x%llx wait=%u alloc=0x%08x aperture_va=0x%llx "
                   "code_offset=0x%llx code_size=0x%llx: %s",
                   static_cast<unsigned long long>(offset),
                   wait_for_cpu ? 1u : 0u,
                   static_cast<unsigned>(aperture.gpu_allocation),
                   static_cast<unsigned long long>(aperture.gpu_va),
                   static_cast<unsigned long long>(aperture.code_offset),
                   static_cast<unsigned long long>(aperture.code_size),
                   ErrorMessage(&submit_error));
    return false;
  }
  if (!wait_for_cpu) return true;
  return WaitForHwQueueFenceCpu(
      api, device, *context, fence_id,
      "D3DKMTWaitForSynchronizationObjectFromCpu(pathb sync9)", out_error);
}

bool ValidatePathBCodeRange(const McdmAbiInfo& abi,
                            const CommandAperture& aperture, uint64_t offset,
                            uint64_t length, Error* out_error) {
  if (!length || !abi.command_aperture_code_slot_size ||
      (abi.command_aperture_code_slot_size &
       (abi.command_aperture_code_slot_size - 1)) != 0 ||
      offset < aperture.code_offset || offset > aperture.gpu_va_size ||
      length > aperture.gpu_va_size - offset) {
    SetError(out_error, "invalid command-aperture code range");
    return false;
  }
  return true;
}

bool SubmitPathBCodeRangeEndMarkers(
    const KmtApi& api, const Device& device, Context* context,
    const CommandAperture& aperture, uint64_t offset, uint64_t length,
    Error* out_error) {
  const McdmAbiInfo abi = GetMcdmAbiInfo(device.mcdm_abi);
  if (!ValidatePathBCodeRange(abi, aperture, offset, length, out_error)) {
    return false;
  }
  const uint64_t slot_size = abi.command_aperture_code_slot_size;
  const uint64_t range_end = offset + length;
  uint64_t boundary = (offset + slot_size) & ~(slot_size - 1);
  const uint64_t final_boundary =
      (range_end + slot_size - 1) & ~(slot_size - 1);
  for (; boundary <= final_boundary; boundary += slot_size) {
    if (!SubmitPathBApertureSync(api, device, context, aperture, boundary,
                                 /*wait_for_cpu=*/false, out_error)) {
      return false;
    }
  }
  return true;
}

bool AcquirePathBCodeRange(const KmtApi& api, const Device& device,
                           Context* context,
                           const CommandAperture& aperture, uint64_t offset,
                           uint64_t length, Error* out_error) {
  const McdmAbiInfo abi = GetMcdmAbiInfo(device.mcdm_abi);
  if (!ValidatePathBCodeRange(abi, aperture, offset, length, out_error)) {
    return false;
  }
  // Acquiring a host-visible code range is a validation-only operation. The
  // mapped bytes must be written and made device-visible before opcode 9
  // publishes the touched aperture slots to the HW queue.
  return true;
}

bool CommitPathBCodeWrite(const KmtApi& api, const Device& device,
                          const CommandAperture& aperture, uint64_t offset,
                          uint64_t length, Error* out_error) {
  const McdmAbiInfo abi = GetMcdmAbiInfo(device.mcdm_abi);
  switch (abi.command_aperture_write_publish_mode) {
    case CommandApertureWritePublishMode::cpu_cache_flush:
      return PublishCpuWriteRange(
          aperture.gpu_cpu_ptr, aperture.gpu_va_size, offset, length,
          abi.command_aperture_code_publish_granularity, out_error);
    case CommandApertureWritePublishMode::kmt_invalidate:
      return SyncCommandApertureCode(api, device, aperture, offset, length,
                                     out_error);
  }
  SetError(out_error, "unknown command-aperture write publication mode");
  return false;
}

bool RefreshPathBSingleCodeMappingAfterWrite(
    const KmtApi& api, const Device& device, CommandAperture* aperture,
    Error* out_error) {
  const McdmAbiInfo abi = GetMcdmAbiInfo(device.mcdm_abi);
  if (!abi.command_aperture_remap_after_write) return true;
  return RefreshCommandApertureGpuMapping(api, device, aperture, out_error);
}

bool PublishPathBCodeWrite(const KmtApi& api, const Device& device,
                           Context* context,
                           const CommandAperture& aperture, uint64_t offset,
                           uint64_t length, Error* out_error) {
  // Both MCDM profiles publish device-visible code with opcode-9 end markers.
  // Compact writes use CPU cache-line flushes while legacy writes use KMT
  // invalidation; in either case the marker is submitted after publication so
  // the later state-3 command observes the completed image in queue order.
  return SubmitPathBCodeRangeEndMarkers(api, device, context, aperture, offset,
                                        length, out_error);
}

bool ReleasePathBCodeRange(const KmtApi& api, const Device& device,
                           Context* context,
                           const CommandAperture& aperture, uint64_t offset,
                           uint64_t length, Error* out_error) {
  const McdmAbiInfo abi = GetMcdmAbiInfo(device.mcdm_abi);
  if (!ValidatePathBCodeRange(abi, aperture, offset, length, out_error)) {
    return false;
  }
  if (UsesLegacyCpuBufferHandles(device.mcdm_abi)) {
    return SubmitPathBApertureSync(api, device, context, aperture, offset,
                                   /*wait_for_cpu=*/true, out_error);
  }

  const uint64_t slot_size = abi.command_aperture_code_slot_size;
  const uint64_t relative_begin = offset - aperture.code_offset;
  const uint64_t first_slot =
      aperture.code_offset + (relative_begin & ~(slot_size - 1));
  uint64_t slot_start = aperture.code_offset +
                        ((relative_begin + length - 1) & ~(slot_size - 1));
  for (;;) {
    if (!SubmitPathBApertureSync(api, device, context, aperture, slot_start,
                                 /*wait_for_cpu=*/slot_start <= first_slot,
                                 out_error)) {
      return false;
    }
    if (slot_start <= first_slot) break;
    slot_start -= slot_size;
  }
  return true;
}

// Allocate the firmware completion ring using the status object selected by
// the negotiated MCDM ABI. The firmware writes completion state into this
// allocation; an ordinary host-only BO is not part of this queue protocol.
bool EnsureStatusRing(const KmtApi& api, const Device& device, Context* context,
                      Error* out_error) {
  if (context->completion_ring_ready) return true;
  constexpr uint32_t kRingSize = 4096;
  const McdmAbiInfo abi = GetMcdmAbiInfo(device.mcdm_abi);
  AllocPrivate ring_private = {};
  ring_private.requested_size = kRingSize;
  ring_private.aligned_size = kRingSize;
  ring_private.private_type = abi.status_private_type;
  ring_private.policy = abi.status_policy;
  ring_private.xcl_flags = abi.status_xcl_flags;

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
  Trace("create-status-ring requested=%llu aligned=%llu private=0x%x flags=0x%x "
        "status=0x%08x%s alloc=0x%08x resource=0x%08x",
        static_cast<unsigned long long>(ring_private.requested_size),
        static_cast<unsigned long long>(ring_private.aligned_size),
        static_cast<unsigned>(ring_private.private_type),
        static_cast<unsigned>(ring_private.xcl_flags),
        static_cast<uint32_t>(status), NtStatusSuffix(status),
        static_cast<unsigned>(ring_info.hAllocation),
        static_cast<unsigned>(create_ring.hResource));
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
  Trace("lock-status-ring alloc=0x%08x status=0x%08x%s cpu=%p",
        static_cast<unsigned>(ring.allocation), static_cast<uint32_t>(status),
        NtStatusSuffix(status), lock.pData);
  if (!CheckStatus("D3DKMTLock2(status ring)", status, out_error)) {
    DestroyBuffer(api, device, &ring);
    return false;
  }
  ring.cpu_ptr = lock.pData;
  if (ring.cpu_ptr) std::memset(ring.cpu_ptr, 0, kRingSize);

  if (abi.status_has_gpu_va) {
    D3DDDI_MAPGPUVIRTUALADDRESS map = {};
    map.hPagingQueue = device.paging_queue;
    map.hAllocation = ring.allocation;
    map.SizeInPages = kRingSize / 4096;
    map.Protection.Write = 1;
    status = api.map_gpu_virtual_address(&map);
    Trace("map-status-ring alloc=0x%08x pages=%llu status=0x%08x%s "
          "gpu_va=0x%llx fence=0x%llx",
          static_cast<unsigned>(ring.allocation),
          static_cast<unsigned long long>(map.SizeInPages),
          static_cast<uint32_t>(status), NtStatusSuffix(status),
          static_cast<unsigned long long>(map.VirtualAddress),
          static_cast<unsigned long long>(map.PagingFenceValue));
    if (!CheckStatusOrPending("D3DKMTMapGpuVirtualAddress(status ring)",
                              status, out_error)) {
      DestroyBuffer(api, device, &ring);
      return false;
    }
    RecordPendingPagingFence(device, map.PagingFenceValue);
    ring.gpu_va = map.VirtualAddress;

    D3DKMT_HANDLE resident_allocs[1] = {ring.allocation};
    D3DDDI_MAKERESIDENT resident = {};
    resident.hPagingQueue = device.paging_queue;
    resident.NumAllocations = 1;
    resident.AllocationList = resident_allocs;
    ConfigureMakeResidentFlags(&resident);
    status = api.make_resident(&resident);
    Trace("resident-status-ring alloc=0x%08x status=0x%08x%s fence=0x%llx",
          static_cast<unsigned>(ring.allocation),
          static_cast<uint32_t>(status), NtStatusSuffix(status),
          static_cast<unsigned long long>(resident.PagingFenceValue));
    if (!CheckMakeResidentStatusOrPending(
            api, device, "D3DKMTMakeResident(status ring)", status, resident,
            out_error)) {
      DestroyBuffer(api, device, &ring);
      return false;
    }
    RecordPendingPagingFence(device, resident.PagingFenceValue);
    D3DKMT_HANDLE wait_objects[1] = {device.paging_sync_object};
    UINT64 wait_values[1] = {resident.PagingFenceValue};
    D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU wait = {};
    wait.hContext = context->context;
    wait.ObjectCount = 1;
    wait.ObjectHandleArray = wait_objects;
    wait.MonitoredFenceValueArray = wait_values;
    status = api.wait_from_gpu(&wait);
    Trace("wait-gpu-status-ring fence=0x%llx status=0x%08x%s",
          static_cast<unsigned long long>(resident.PagingFenceValue),
          static_cast<uint32_t>(status), NtStatusSuffix(status));
    if (!CheckStatus(
            "D3DKMTWaitForSynchronizationObjectFromGpu(status ring)", status,
            out_error)) {
      DestroyBuffer(api, device, &ring);
      return false;
    }
  }

  context->completion_ring = ring;
  context->completion_ring_resource = ring.resource;
  context->completion_ring_ready = true;
  context->completion_ring_owned = true;
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
  const McdmAbiInfo abi = GetMcdmAbiInfo(device.mcdm_abi);
  if (ert_bytes == 0 ||
      ert_bytes > abi.pathb_private_size - abi.pathb_packet_offset) {
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

  McdmPrivateData private_data = BuildPathBSubmitPrivateData(
      device.mcdm_abi, exec_buffer, ring, slot_offset, slot_cpu, ert_packet,
      ert_bytes, command_state, chain_info);

  if (!WaitForBufferResidency(api, device, *context, exec_buffer, "pathb-exec",
                              out_error)) {
    return false;
  }
  if (!WaitForBufferResidency(api, device, *context, ring, "pathb-ring",
                              out_error)) {
    return false;
  }

  uint64_t fence_id = context->next_fence_id++;
  D3DKMT_SUBMITCOMMANDTOHWQUEUE submit = {};
  submit.hHwQueue = context->hw_queue;
  submit.HwQueueProgressFenceId = fence_id;
  submit.CommandBuffer = exec_buffer.gpu_va;
  submit.CommandLength =
      static_cast<UINT>(exec_buffer.size + abi.pathb_packet_offset);
  submit.PrivateDriverDataSize = private_data.size;
  submit.pPrivateDriverData = private_data.data;
  if (!SubmitCommandToHwQueueAfterPaging(
          api, device, &submit, "D3DKMTSubmitCommandToHwQueue(pathb)",
          out_error)) {
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
  auto read_completion_once = [&]() -> bool {
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
  const McdmAbiInfo abi = GetMcdmAbiInfo(device.mcdm_abi);
  if (ert_bytes == 0 ||
      ert_bytes > abi.pathb_private_size - abi.pathb_packet_offset) {
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

  McdmPrivateData private_data = BuildPathBSubmitPrivateData(
      device.mcdm_abi, exec_buffer, ring, slot_offset, slot_cpu, ert_packet,
      ert_bytes, command_state, chain_info);

  if (!WaitForBufferResidency(api, device, *context, exec_buffer, "pathb-exec",
                              out_error)) {
    return false;
  }
  if (!WaitForBufferResidency(api, device, *context, ring, "pathb-ring",
                              out_error)) {
    return false;
  }

  uint64_t fence_id = context->next_fence_id++;
  D3DKMT_SUBMITCOMMANDTOHWQUEUE submit = {};
  submit.hHwQueue = context->hw_queue;
  submit.HwQueueProgressFenceId = fence_id;
  submit.CommandBuffer = exec_buffer.gpu_va;
  submit.CommandLength =
      static_cast<UINT>(exec_buffer.size + abi.pathb_packet_offset);
  submit.PrivateDriverDataSize = private_data.size;
  submit.pPrivateDriverData = private_data.data;
  if (!SubmitCommandToHwQueueAfterPaging(
          api, device, &submit,
          "D3DKMTSubmitCommandToHwQueue(pathb nowait)", out_error)) {
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
  for (size_t i = 0; i < pending_count; ++i) {
    PathBPendingSubmit& p = pending[i];
    volatile uint32_t* const packet_header = p.packet_header;
    uint32_t slot_state = 0;
    uint32_t packet_state = packet_header ? *packet_header : 0;
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

static void BeginDestroyCommandAperture(const KmtApi& api,
                                        const Device& device,
                                        CommandAperture* aperture) {
  if (!aperture) return;
  const bool owns_separate_gpu_allocation =
      aperture->gpu_allocation &&
      aperture->gpu_allocation != aperture->allocation;
  if (!owns_separate_gpu_allocation) {
    // Legacy MCDM has one aperture allocation and retains its established
    // one-phase teardown before the context is destroyed.
    DestroyCommandAperture(api, device, aperture);
    return;
  }

  // Compact MCDM has separate instruction and status allocations. XRT
  // destroys instruction storage synchronously before the HW queue, but keeps
  // the status allocation alive until after queue destruction.
  const McdmAbiInfo abi = GetMcdmAbiInfo(device.mcdm_abi);
  if (aperture->gpu_va && abi.explicit_gpu_va_free_on_destroy) {
    D3DKMT_FREEGPUVIRTUALADDRESS free_va = {};
    free_va.hAdapter = device.adapter;
    free_va.BaseAddress = aperture->gpu_va;
    free_va.Size = aperture->gpu_va_size;
    api.free_gpu_virtual_address(&free_va);
  }
  if (aperture->gpu_cpu_ptr) {
    D3DKMT_UNLOCK2 unlock = {};
    unlock.hDevice = device.device;
    unlock.hAllocation = aperture->gpu_allocation;
    api.unlock2(&unlock);
  }

  StatsFree("command-aperture-gpu", aperture->gpu_va_size,
            aperture->gpu_allocation);
  D3DKMT_DESTROYALLOCATION2 destroy_gpu = {};
  D3DKMT_HANDLE gpu_allocs[1] = {aperture->gpu_allocation};
  destroy_gpu.hDevice = device.device;
  destroy_gpu.Flags.AssumeNotInUse = 1;
  destroy_gpu.Flags.SynchronousDestroy = 1;
  if (aperture->gpu_resource) {
    destroy_gpu.hResource = aperture->gpu_resource;
  } else {
    destroy_gpu.AllocationCount = 1;
    destroy_gpu.phAllocationList = gpu_allocs;
  }
  api.destroy_allocation2(&destroy_gpu);

  aperture->gpu_allocation = 0;
  aperture->gpu_resource = 0;
  aperture->gpu_va = 0;
  aperture->gpu_va_size = 0;
  aperture->gpu_cpu_ptr = nullptr;
  aperture->code_allocation = 0;
  aperture->code_resource = 0;
  aperture->code_gpu_va = 0;
  aperture->code_cpu_ptr = nullptr;
  aperture->code_offset = 0;
  aperture->code_size = 0;
}

void DestroyContextWithCommandAperture(const KmtApi& api,
                                       const Device& device,
                                       Context* context,
                                       CommandAperture* aperture) {
  if (!context) {
    DestroyCommandAperture(api, device, aperture);
    return;
  }

  // XRT's compact-MCDM lifecycle releases instruction storage before the HW
  // queue, then releases the queue-owned status resource before context-private
  // storage and the context handle. Keep that protocol ordering wholly inside
  // the Windows DDI layer.
  BeginDestroyCommandAperture(api, device, aperture);
  DestroyContextHwQueue(api, context);
  DestroyCommandAperture(api, device, aperture);
  DestroyContextAfterHwQueue(api, device, context);
}

void DestroyCommandAperture(const KmtApi& api, const Device& device,
                            CommandAperture* aperture) {
  if (!aperture || (!aperture->allocation && !aperture->resource &&
                    !aperture->gpu_allocation && !aperture->gpu_resource &&
                    !aperture->cleanup_allocation)) {
    return;
  }
  const McdmAbiInfo abi = GetMcdmAbiInfo(device.mcdm_abi);
  if (aperture->gpu_va && abi.explicit_gpu_va_free_on_destroy) {
    D3DKMT_FREEGPUVIRTUALADDRESS free_va = {};
    free_va.hAdapter = device.adapter;
    free_va.BaseAddress = aperture->gpu_va;
    free_va.Size = aperture->gpu_va_size;
    api.free_gpu_virtual_address(&free_va);
    aperture->gpu_va = 0;
  }
  if (aperture->status_gpu_va && abi.explicit_gpu_va_free_on_destroy) {
    D3DKMT_FREEGPUVIRTUALADDRESS free_va = {};
    free_va.hAdapter = device.adapter;
    free_va.BaseAddress = aperture->status_gpu_va;
    free_va.Size = kCommandApertureAllocationSize;
    api.free_gpu_virtual_address(&free_va);
    aperture->status_gpu_va = 0;
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
    StatsFree("command-aperture-gpu", aperture->gpu_va_size,
              aperture->gpu_allocation);
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
    StatsFree("command-aperture-status", kCommandApertureAllocationSize,
              aperture->cleanup_allocation ? aperture->cleanup_allocation
                                           : aperture->allocation);
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
      const uint32_t resource_flags = abi.shared_resource_destroy_flags;
      if (resource_flags) destroy_command.Flags.Value = resource_flags;
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
