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
constexpr UINT kMaxDriverStorePathWarmupBytes = 4096;
// STX2 two-DWORD adapter info spans two incompatible context layouts.
// Captured .240 and .280 packages use the older layout; .314 and .329 use
// the modern one.
constexpr uint32_t kFirstModernLegacyContextLayoutRevision = 314;
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
    return true;
  }
  SetMakeResidentError(api, device, call_name, status, resident, out_error);
  return false;
}

bool CheckStatus(const char* call_name, NTSTATUS status, Error* out_error) {
  if (status == 0) return true;
  SetErrorFormat(out_error, "%s failed with 0x%08x%s", call_name,
                 static_cast<uint32_t>(status), NtStatusSuffix(status));
  return false;
}

bool CheckStatusOrPending(const char* call_name, NTSTATUS status,
                          Error* out_error) {
  if (status == 0) return true;
  if (status == kStatusPending) {
    return true;
  }
  return CheckStatus(call_name, status, out_error);
}

uint64_t LastLevelCacheSize() {
  static const uint64_t cache_size = []() -> uint64_t {
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationCache, nullptr, &bytes);
    if (!bytes) return 0;
    std::vector<uint8_t> storage(bytes);
    if (!GetLogicalProcessorInformationEx(
            RelationCache,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                storage.data()),
            &bytes)) {
      return 0;
    }
    uint64_t largest_l3 = 0;
    for (DWORD offset = 0; offset < bytes;) {
      const auto* info =
          reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
              storage.data() + offset);
      if (!info->Size || info->Size > bytes - offset) return 0;
      if (info->Relationship == RelationCache && info->Cache.Level == 3) {
        largest_l3 = std::max<uint64_t>(largest_l3, info->Cache.CacheSize);
      }
      offset += info->Size;
    }
    return largest_l3;
  }();
  return cache_size;
}

bool SyncSmallBufferCpuCache(const Buffer& buffer, uint64_t offset,
                             uint64_t length, Error* out_error) {
  if (length == 0) return true;
  if (!buffer.cpu_ptr || offset > buffer.size || length > buffer.size - offset) {
    SetError(out_error, "CPU BO sync range is out of bounds");
    return false;
  }
  constexpr uintptr_t kCacheLineSize = 64;
  uintptr_t line = reinterpret_cast<uintptr_t>(buffer.cpu_ptr) + offset;
  const uintptr_t end = line + length;
  _mm_lfence();
  if (line & (kCacheLineSize - 1)) {
    _mm_clflush(reinterpret_cast<void const*>(line));
    line = (line + kCacheLineSize - 1) & ~(kCacheLineSize - 1);
  }
  while (line < end) {
    _mm_clflush(reinterpret_cast<void const*>(line));
    line += kCacheLineSize;
  }
  return true;
}

enum class CpuCacheOperation {
  writeback,
  invalidate,
};

#if defined(__clang__)
__attribute__((target("clflushopt")))
#endif
void FlushCpuCacheLineOptimized(void* address) {
  _mm_clflushopt(address);
}

#if defined(__clang__)
__attribute__((target("clwb")))
#endif
void WriteBackCpuCacheLine(void* address) {
  _mm_clwb(address);
}

bool ApplyCpuCacheOperation(void* mapping, uint64_t mapping_size,
                            const CpuWriteRange* ranges, size_t range_count,
                            uint64_t granularity, CpuCacheOperation operation,
                            Error* out_error) {
  if (range_count == 0) return true;
  if (!ranges) {
    SetError(out_error, "invalid CPU cache operation ranges");
    return false;
  }
  bool has_nonempty_range = false;
  for (size_t i = 0; i < range_count; ++i) {
    has_nonempty_range |= ranges[i].length != 0;
  }
  if (!has_nonempty_range) return true;
  if (!mapping || granularity == 0 ||
      (granularity & (granularity - 1)) != 0) {
    SetError(out_error, "invalid CPU write publication mapping");
    return false;
  }

  constexpr uintptr_t kCpuCacheLineSize = 64;
  const uintptr_t mapping_address = reinterpret_cast<uintptr_t>(mapping);
  // Publish only writes synchronized into this calling thread. Buffer ownership
  // is responsible for excluding concurrent writers; a process-wide write
  // buffer flush would hide violations of that contract and is unnecessary.
  std::atomic_thread_fence(std::memory_order_release);
  struct CacheCapabilities {
    bool clflushopt = false;
    bool clwb = false;
  };
  static const CacheCapabilities cache_capabilities = []() {
    CacheCapabilities capabilities;
    int registers[4] = {};
    __cpuid(registers, 0);
    if (registers[0] < 7) return capabilities;
    __cpuidex(registers, 7, 0);
    capabilities.clflushopt = (registers[1] & (1 << 23)) != 0;
    capabilities.clwb = (registers[1] & (1 << 24)) != 0;
    return capabilities;
  }();
  for (size_t i = 0; i < range_count; ++i) {
    const uint64_t offset = ranges[i].offset;
    const uint64_t length = ranges[i].length;
    if (length == 0) continue;
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
    if (end_offset > mapping_size ||
        end_offset > std::numeric_limits<uintptr_t>::max() - mapping_address) {
      SetError(out_error, "CPU write publication granule is out of bounds");
      return false;
    }
    uintptr_t line =
        (mapping_address + static_cast<uintptr_t>(begin_offset)) &
        ~(kCpuCacheLineSize - 1);
    const uintptr_t end = mapping_address + static_cast<uintptr_t>(end_offset);
    // Keep iteration and the ordering fence in this function. The targeted
    // helpers isolate optional instructions only; they do not define a
    // publication boundary by themselves.
    while (line < end) {
      if (operation == CpuCacheOperation::writeback &&
          cache_capabilities.clwb) {
        WriteBackCpuCacheLine(reinterpret_cast<void*>(line));
      } else if (cache_capabilities.clflushopt) {
        FlushCpuCacheLineOptimized(reinterpret_cast<void*>(line));
      } else {
        _mm_clflush(reinterpret_cast<void const*>(line));
      }
      line += kCpuCacheLineSize;
    }
  }
  const bool uses_weakly_ordered_cache_operation =
      (operation == CpuCacheOperation::writeback &&
       cache_capabilities.clwb) ||
      cache_capabilities.clflushopt;
  if (uses_weakly_ordered_cache_operation) {
    _mm_sfence();
  } else {
    _mm_mfence();
  }
  return true;
}

bool PublishCpuWriteRanges(void* mapping, uint64_t mapping_size,
                           const CpuWriteRange* ranges, size_t range_count,
                           uint64_t granularity, Error* out_error) {
  return ApplyCpuCacheOperation(mapping, mapping_size, ranges, range_count,
                                granularity, CpuCacheOperation::writeback,
                                out_error);
}

bool PublishCpuWriteRange(void* mapping, uint64_t mapping_size,
                          uint64_t offset, uint64_t length,
                          uint64_t granularity, Error* out_error) {
  const CpuWriteRange range = {offset, length};
  return PublishCpuWriteRanges(mapping, mapping_size, &range, 1, granularity,
                               out_error);
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
  if (adapter->retained_handle_count >= kMaxComputeAdapterHandles) {
    SetErrorFormat(out_error,
                   "too many retained adapter handles (capacity=%zu)",
                   kMaxComputeAdapterHandles);
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
    return false;
  }
  NormalizeSystemRootPath(&driver_store_path);
  if (driver_store_path.back() != L'\\' && driver_store_path.back() != L'/') {
    driver_store_path.push_back(L'\\');
  }
  driver_store_path.append(L"kipudrv.inf");

  std::vector<uint8_t> bytes;
  if (!ReadFileToBytes(driver_store_path.c_str(), &bytes)) {
    return false;
  }
  if (!ParseDriverVersionText(DecodeInfText(bytes), out_version)) {
    return false;
  }
  return true;
}

bool UsesLegacyV2ContextLayout(const DriverVersion& version) {
  return version.major == 32 && version.minor == 0 && version.build == 203 &&
         version.revision < kFirstModernLegacyContextLayoutRevision;
}

constexpr uint32_t kExpectedKmdVersion = 0;
constexpr uint32_t kCompactAdapterInfoExtra = 0;

bool SelectAbiFromAdapterInfo(uint32_t kmd_version, uint32_t hw_type,
                              bool compact_prefix, uint32_t compact_extra,
                              McdmAbi* out_abi,
                              bool* out_stx2_disambiguation,
                              Error* out_error) {
  if (kmd_version != kExpectedKmdVersion) {
    SetErrorFormat(out_error,
                   "unsupported UMDRIVERPRIVATE adapter info "
                   "kmd_version=%u hw_type=%s (%u)",
                   kmd_version, HardwareTypeName(hw_type), hw_type);
    return false;
  }
  if (compact_prefix && compact_extra != kCompactAdapterInfoExtra) {
    SetErrorFormat(out_error,
                   "unsupported UMDRIVERPRIVATE compact adapter info "
                   "kmd_version=%u hw_type=%s (%u) extra=%u",
                   kmd_version, HardwareTypeName(hw_type), hw_type,
                   compact_extra);
    return false;
  }

  const auto type = static_cast<HardwareType>(hw_type);
  if (compact_prefix) {
    switch (type) {
      case HardwareType::stx2:
      case HardwareType::stxh:
      case HardwareType::krk1:
        *out_abi = McdmAbi::compact;
        *out_stx2_disambiguation = false;
        return true;
      case HardwareType::phx:
      case HardwareType::stx:
        break;
    }
    SetErrorFormat(out_error,
                   "unsupported UMDRIVERPRIVATE compact adapter info "
                   "kmd_version=%u hw_type=%s (%u)",
                   kmd_version, HardwareTypeName(hw_type), hw_type);
    return false;
  }

  switch (type) {
    case HardwareType::phx:
      *out_abi = McdmAbi::legacy_v0;
      *out_stx2_disambiguation = false;
      return true;
    case HardwareType::stx2:
      *out_abi = McdmAbi::legacy_v2;
      *out_stx2_disambiguation = true;
      return true;
    case HardwareType::stxh:
    case HardwareType::krk1:
      *out_abi = McdmAbi::legacy;
      *out_stx2_disambiguation = false;
      return true;
    case HardwareType::stx:
      break;
  }
  SetErrorFormat(out_error,
                 "unsupported UMDRIVERPRIVATE adapter info "
                 "kmd_version=%u hw_type=%s (%u)",
                 kmd_version, HardwareTypeName(hw_type), hw_type);
  return false;
}

void RecordAdapterInfo(McdmAbiDiagnostics* diagnostics, uint32_t kmd_version,
                       uint32_t hw_type, bool compact_prefix,
                       const uint32_t* words, uint32_t word_count) {
  diagnostics->kmd_version = kmd_version;
  diagnostics->hw_type = hw_type;
  diagnostics->compact_adapter_info = compact_prefix;
  diagnostics->source = McdmAbiSource::identity_query;
  diagnostics->identity_word_count = word_count;
  diagnostics->accepted_identity_count = 1;
  diagnostics->identity_accepted = true;
  for (uint32_t i = 0; i < word_count && i < 3; ++i) {
    diagnostics->identity_words[i] = words[i];
  }
}

}  // namespace

const char* HardwareTypeName(uint32_t hw_type) {
  switch (static_cast<HardwareType>(hw_type)) {
    case HardwareType::phx:
      return "PHX";
    case HardwareType::stx:
      return "STX";
    case HardwareType::stx2:
      return "STX2";
    case HardwareType::stxh:
      return "STXH";
    case HardwareType::krk1:
      return "KRK1";
  }
  return "UNKNOWN";
}

bool SelectMcdmAbiForDriverVersion(McdmAbi probed_abi,
                                   bool has_driver_version,
                                   const DriverVersion& driver_version,
                                   McdmAbi* out_abi, Error* out_error) {
  if (!out_abi) {
    SetError(out_error, "missing selected MCDM ABI output");
    return false;
  }
  if (probed_abi != McdmAbi::legacy_v2) {
    *out_abi = probed_abi;
    return true;
  }
  if (!has_driver_version) {
    SetError(out_error,
             "ambiguous STX2 adapter info: driver package version is "
             "unavailable");
    return false;
  }
  if (driver_version.major != 32 || driver_version.minor != 0 ||
      driver_version.build != 203) {
    SetErrorFormat(out_error,
                   "ambiguous STX2 adapter info: unsupported driver package "
                   "version %u.%u.%u.%u",
                   driver_version.major, driver_version.minor,
                   driver_version.build, driver_version.revision);
    return false;
  }
  *out_abi = UsesLegacyV2ContextLayout(driver_version) ? McdmAbi::legacy_v2
                                                       : McdmAbi::legacy;
  return true;
}

McdmSubmissionPolicy GetMcdmSubmissionPolicy(McdmAbi abi) {
  switch (abi) {
    case McdmAbi::legacy_v0:
    case McdmAbi::legacy_v2:
      return {/*supports_command_chaining=*/false,
              /*uses_shared_command_code_view=*/true,
              /*submit_completion_is_deferred=*/false};
    case McdmAbi::legacy:
    case McdmAbi::compact:
      return {/*supports_command_chaining=*/true,
              /*uses_shared_command_code_view=*/false,
              /*submit_completion_is_deferred=*/true};
  }
  return {};
}

bool SupportsHostBufferReuse(McdmAbi abi) {
  // Legacy v0/v2 stacks are certified only with immediate native-allocation
  // destruction. Later contracts support retaining device-scoped allocations
  // across HAL wrapper lifetimes.
  return abi == McdmAbi::legacy || abi == McdmAbi::compact;
}

bool QueryMcdmAbiDiagnostics(const KmtApi& api, D3DKMT_HANDLE adapter,
                             McdmAbiDiagnostics* out_diagnostics,
                             Error* out_error) {
  if (!out_diagnostics) {
    SetError(out_error, "missing MCDM ABI diagnostics output");
    return false;
  }
  McdmAbiDiagnostics diagnostics = {};
  // Probe the historical two-DWORD XRT_UMD_ADAPTER_INFO prefix first
  // {kmd_version, hw_type}. Compact miniports reject that size with
  // STATUS_BUFFER_TOO_SMALL; retry the three-DWORD compact prefix
  // {kmd_version, hw_type, 0}. Unknown hardware types fail closed.
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
    McdmAbi probed_abi = McdmAbi::legacy;
    bool stx2_disambiguation = false;
    if (!SelectAbiFromAdapterInfo(legacy_data[0], legacy_data[1],
                                  /*compact_prefix=*/false, 0, &probed_abi,
                                  &stx2_disambiguation, out_error)) {
      return false;
    }
    diagnostics.probed_abi = probed_abi;
    diagnostics.selected_abi = probed_abi;
    diagnostics.driver_version_disambiguation_required = stx2_disambiguation;
    RecordAdapterInfo(&diagnostics, legacy_data[0], legacy_data[1],
                      /*compact_prefix=*/false, legacy_data, 2);
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
             "MCDM driver rejected both three-dword and two-dword "
             "UMDRIVERPRIVATE adapter-info queries");
    return false;
  }
  if (compact_status != 0) {
    return CheckStatus("D3DKMTQueryAdapterInfo(UMDRIVERPRIVATE compact)",
                       compact_status, out_error);
  }
  McdmAbi probed_abi = McdmAbi::compact;
  bool stx2_disambiguation = false;
  if (!SelectAbiFromAdapterInfo(compact_data[0], compact_data[1],
                                /*compact_prefix=*/true, compact_data[2],
                                &probed_abi, &stx2_disambiguation,
                                out_error)) {
    return false;
  }
  diagnostics.selected_abi = probed_abi;
  diagnostics.probed_abi = probed_abi;
  diagnostics.driver_version_disambiguation_required = stx2_disambiguation;
  RecordAdapterInfo(&diagnostics, compact_data[0], compact_data[1],
                    /*compact_prefix=*/true, compact_data, 3);
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
  McdmAbiInfo info = {};
  info.command_aperture_code_slot_size = 0x8000;
  if (abi == McdmAbi::compact) {
    info.status_private_type = 0x332c;
    info.status_policy = 2;
    info.status_xcl_flags = 0x02000000;
    info.submit_private_prefix_size = 0x78;
    info.setup_private_size = 0x280;
    info.pathb_private_size = 0x278;
    info.pathb_packet_offset = 0x78;
    info.chain_metadata_offset = 0x58;
    info.pathb_bo_table_entry_count = 6;
    info.status_has_gpu_va = true;
    info.sync_has_allocation_handle = false;
    info.command_aperture_write_publish_mode =
        CommandApertureWritePublishMode::cpu_cache_flush;
    info.command_aperture_code_publish_granularity = 0x8000;
    info.command_aperture_residency_after_bootstrap = true;
    // XRT 2.21 waits for compact shared-resource destruction to finish and
    // leaves mapped VA ownership with the allocation.
    info.shared_resource_destroy_flags = 0x3;
    return info;
  }
  info.status_private_type = 0x332b;
  info.sync_has_allocation_handle = true;
  info.command_aperture_write_publish_mode =
      CommandApertureWritePublishMode::kmt_invalidate;
  info.command_aperture_remap_after_write = true;
  info.explicit_gpu_va_free_on_destroy = true;
  if (abi == McdmAbi::legacy_v0) {
    info.submit_private_prefix_size = 0x58;
    info.setup_private_size = 0x260;
    info.pathb_private_size = 0x258;
    info.pathb_packet_offset = 0x58;
    info.chain_metadata_offset = 0x40;
    info.pathb_bo_table_entry_count = 5;
    return info;
  }
  if (abi == McdmAbi::legacy_v2) {
    info.submit_private_prefix_size = 0x60;
    info.setup_private_size = 0x268;
    info.pathb_private_size = 0x260;
    info.pathb_packet_offset = 0x60;
    info.chain_metadata_offset = 0x48;
    info.pathb_bo_table_entry_count = 5;
    return info;
  }
  info.submit_private_prefix_size = 0x68;
  info.setup_private_size = 0x270;
  info.pathb_private_size = 0x268;
  info.pathb_packet_offset = 0x68;
  info.chain_metadata_offset = 0x48;
  info.pathb_bo_table_entry_count = 5;
  return info;
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
           aperture.protocol_gpu_va);
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
  D3DKMT_ADAPTERINFO adapters[kMaxComputeAdapterHandles] = {};
  UINT adapter_count = 0;
  if (!EnumerateComputeAdapters(api, adapters, kMaxComputeAdapterHandles,
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
  if (!EnumerateComputeAdapters(api, adapters, kMaxComputeAdapterHandles,
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
  if (!QueryMcdmAbiDiagnostics(api, device.adapter,
                               &device.mcdm_abi_diagnostics, out_error)) {
    DestroyDevice(api, &device);
    return false;
  }
  device.has_driver_version =
      QueryDriverVersion(api, device.adapter, &device.driver_version);
  const McdmAbi probed_mcdm_abi = device.mcdm_abi_diagnostics.probed_abi;
  if (!SelectMcdmAbiForDriverVersion(
          probed_mcdm_abi, device.has_driver_version, device.driver_version,
          &device.mcdm_abi, out_error)) {
    DestroyDevice(api, &device);
    return false;
  }
  device.mcdm_abi_diagnostics.selected_abi = device.mcdm_abi;
  device.mcdm_abi_diagnostics.driver_version_disambiguation_available =
      device.mcdm_abi_diagnostics.driver_version_disambiguation_required &&
      device.has_driver_version;
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
  if (device.paging_fence_cpu) {
    const auto* const completed =
        static_cast<const volatile UINT64*>(device.paging_fence_cpu);
    if (*completed >= fence_value) return true;
  }
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
  const NTSTATUS status = api.submit_command_to_hw_queue(submit);
  return CheckStatus(label, status, out_error);
}

bool CreateBuffer(const KmtApi& api, const Device& device, BufferKind kind,
                  uint64_t size, Buffer* out_buffer, Error* out_error) {
  BufferKindInfo kind_info = GetBufferKindInfo(kind);
  const McdmAbiInfo abi = GetMcdmAbiInfo(device.mcdm_abi);
  // XRT's exec BO submit path asks the driver for the logical command capacity
  // plus the negotiated private prefix used by SubmitCommandToHwQueue. A
  // state-3 command therefore requests 4096 + prefix bytes, while a compact
  // runlist parent requests only its 224-byte packet + prefix. The HAL still
  // sees only the logical command capacity.
  uint64_t requested_size = std::max<uint64_t>(size, 1);
  if (kind == BufferKind::execbuf) {
    requested_size += abi.submit_private_prefix_size;
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
  if (!CheckStatus("D3DKMTLock2", status, out_error)) {
    DestroyBuffer(api, device, &buffer);
    return false;
  }
  buffer.cpu_ptr = lock.pData;

  *out_buffer = buffer;
  return true;
}

bool PublishBufferCpuWrites(const Buffer& buffer, uint64_t offset,
                            uint64_t length, Error* out_error) {
  return PublishCpuWriteRange(buffer.cpu_ptr, buffer.size, offset, length,
                              /*granularity=*/1, out_error);
}

bool InvalidateBufferCpuReads(const Buffer& buffer, uint64_t offset,
                              uint64_t length, Error* out_error) {
  const CpuWriteRange range = {offset, length};
  return ApplyCpuCacheOperation(buffer.cpu_ptr, buffer.size, &range, 1,
                                /*granularity=*/1,
                                CpuCacheOperation::invalidate, out_error);
}

bool SyncBuffer(const KmtApi& api, const Device& device, const Buffer& buffer,
                uint64_t offset, uint64_t length, Error* out_error) {
  // XRT uses cache-line synchronization for ranges smaller than the LLC and
  // falls back to allocation-level KMT synchronization for larger ranges.
  // This avoids a fixed KMT transition for command metadata while bounding
  // the linear cost of flushing large buffers such as model weights.
  const uint64_t last_level_cache_size = LastLevelCacheSize();
  if (buffer.cpu_ptr && last_level_cache_size &&
      length < last_level_cache_size) {
    return SyncSmallBufferCpuCache(buffer, offset, length, out_error);
  }
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
  constexpr size_t kBoTableWordOffset =
      kPathBBoTableOffset / sizeof(uint32_t);
  constexpr size_t kBoTableWords = kPathBBoTableSize / sizeof(uint32_t);
  constexpr size_t kBoTableEndBytes =
      kPathBBoTableOffset + kPathBBoTableSize;
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
                            Context& context, const Buffer& buffer,
                            const char* label, Error* out_error) {
  if (buffer.paging_fence_value == 0) return true;
  if (device.mcdm_abi == McdmAbi::compact) {
    // Compact XRT tracks all Map/MakeResident fences in one device-wide
    // watermark and drains it in the HW-queue submit wrapper. Adding per-BO
    // GPU waits changes the first-submit lifecycle and is not part of that ABI.
    return true;
  }
  if (buffer.paging_fence_value <= context.ordered_paging_fence_value) {
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
  if (!CheckStatus(call_name, status, out_error)) return false;
  context.ordered_paging_fence_value = buffer.paging_fence_value;
  return true;
}

void DestroyBuffer(const KmtApi& api, const Device& device, Buffer* buffer) {
  if (!buffer || !buffer->allocation) return;
  if (buffer->paging_fence_value &&
      !WaitForPagingFenceCpu(api, device, buffer->paging_fence_value)) {
    // The paging queue may still own this allocation. Retain the KMT handle;
    // process teardown will reclaim it after the device is closed. Leaking on
    // this exceptional path is safer than freeing storage the KMD may use.
    return;
  }
  const BufferKindInfo kind_info = GetBufferKindInfo(buffer->kind);
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
  if (!CheckStatus("D3DKMTCreateAllocation2(command aperture)", status,
                   out_error)) {
    return false;
  }

  CommandAperture aperture = {};
  aperture.allocation_size = kCommandApertureAllocationSize;
  aperture.gpu_va_size = kCommandApertureGpuVaSize;
  aperture.protocol_gpu_va = kCommandApertureGpuVaBase;
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

  // The compact ABI exposes the status object to firmware by both allocation
  // handle and GPU VA. The legacy ABI uses only the allocation handle.
  if (abi.status_has_gpu_va) {
    D3DDDI_MAPGPUVIRTUALADDRESS status_map = {};
    status_map.hPagingQueue = device.paging_queue;
    status_map.hAllocation = aperture.allocation;
    status_map.SizeInPages = kCommandApertureAllocationSize / kPageSize;
    status_map.Protection.Write = 1;
    status = api.map_gpu_virtual_address(&status_map);
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
  RecordPendingPagingFence(device, map.PagingFenceValue);

  aperture.gpu_va = map.VirtualAddress;

  if (!abi.command_aperture_residency_after_bootstrap) {
    D3DKMT_HANDLE resident_allocs[1] = {aperture.gpu_allocation};
    D3DDDI_MAKERESIDENT resident = {};
    resident.hPagingQueue = device.paging_queue;
    resident.NumAllocations = 1;
    resident.AllocationList = resident_allocs;
    ConfigureMakeResidentFlags(&resident);
    status = api.make_resident(&resident);
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
  aperture.code_gpu_va = aperture.protocol_gpu_va + aperture.code_offset;
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
  if (aperture->protocol_gpu_va >
      std::numeric_limits<uint64_t>::max() - code_offset) {
    SetError(out_error, "command-aperture code GPU address overflows");
    return false;
  }

  aperture->code_allocation = aperture->gpu_allocation;
  aperture->code_offset = code_offset;
  aperture->code_gpu_va = aperture->protocol_gpu_va + code_offset;
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
    aperture->code_allocation = aperture->gpu_allocation;
    aperture->code_gpu_va =
        aperture->protocol_gpu_va + aperture->code_offset;
    aperture->code_size = aperture->gpu_va_size - aperture->code_offset;
    return true;
  }

  D3DDDI_MAPGPUVIRTUALADDRESS map = {};
  map.hPagingQueue = device.paging_queue;
  map.hAllocation = aperture->gpu_allocation;
  map.SizeInPages = kCommandApertureGpuVaSize / kPageSize;
  map.Protection.Write = 1;
  NTSTATUS status = api.map_gpu_virtual_address(&map);
  if (status != 0 && status != kStatusPending) {
    SetErrorFormat(out_error,
                   "D3DKMTMapGpuVirtualAddress(command aperture remap) failed with 0x%08x%s allocation=0x%08x pages=0x%llx",
                   static_cast<uint32_t>(status), NtStatusSuffix(status),
                   static_cast<unsigned>(aperture->gpu_allocation),
                   static_cast<unsigned long long>(map.SizeInPages));
    return false;
  }
  RecordPendingPagingFence(device, map.PagingFenceValue);
  aperture->gpu_va = map.VirtualAddress;
  aperture->code_allocation = aperture->gpu_allocation;
  aperture->code_gpu_va =
      aperture->protocol_gpu_va + aperture->code_offset;
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
  WriteU64(submit_private.data(), 0x10, aperture->protocol_gpu_va);

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
  // Match XRT's qhdl wait path. User-visible deadlines are enforced by the
  // completion-batch notification above this DDI; this worker wait owns the
  // native submission until KMT reports that hardware is done with its memory.
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

  const McdmAbiInfo abi = GetMcdmAbiInfo(device.mcdm_abi);
  std::array<uint8_t, kMaxSubmitPrivatePrefixSize> bootstrap_private = {};
  WriteU32(bootstrap_private.data(), 0x00, 2);
  WriteU64(bootstrap_private.data(), 0x08, aperture->gpu_allocation);
  WriteU64(bootstrap_private.data(), 0x10, aperture->protocol_gpu_va);

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
  const CpuWriteRange range = {offset, length};
  return CommitPathBCodeWrites(api, device, aperture, &range, 1, out_error);
}

bool CommitPathBCodeWrites(const KmtApi& api, const Device& device,
                           const CommandAperture& aperture,
                           const CpuWriteRange* ranges, size_t range_count,
                           Error* out_error) {
  const McdmAbiInfo abi = GetMcdmAbiInfo(device.mcdm_abi);
  switch (abi.command_aperture_write_publish_mode) {
    case CommandApertureWritePublishMode::cpu_cache_flush:
      return PublishCpuWriteRanges(
          aperture.gpu_cpu_ptr, aperture.gpu_va_size, ranges, range_count,
          abi.command_aperture_code_publish_granularity, out_error);
    case CommandApertureWritePublishMode::kmt_invalidate:
      for (size_t i = 0; i < range_count; ++i) {
        if (!SyncCommandApertureCode(api, device, aperture, ranges[i].offset,
                                    ranges[i].length, out_error)) {
          return false;
        }
      }
      return true;
  }
  SetError(out_error, "unknown command-aperture write publication mode");
  return false;
}

bool CopyAndCommitPathBCodeWrites(const CommandAperture& aperture,
                                  const CpuCopyRange* ranges,
                                  size_t range_count, Error* out_error) {
  if (range_count == 0) return true;
  if (!aperture.gpu_cpu_ptr || !ranges) {
    SetError(out_error, "invalid command-aperture copy ranges");
    return false;
  }

  constexpr uintptr_t kCacheLineSize = 64;
  const uintptr_t mapping =
      reinterpret_cast<uintptr_t>(aperture.gpu_cpu_ptr);
  bool used_streaming_stores = false;
  bool used_cached_tail = false;
  static const bool has_clflushopt = []() {
    int registers[4] = {};
    __cpuid(registers, 0);
    if (registers[0] < 7) return false;
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 23)) != 0;
  }();

  for (size_t i = 0; i < range_count; ++i) {
    const CpuCopyRange& range = ranges[i];
    if (range.length == 0) continue;
    if (!range.source || range.offset > aperture.gpu_va_size ||
        range.length > aperture.gpu_va_size - range.offset ||
        range.length > std::numeric_limits<size_t>::max() ||
        range.offset >
            std::numeric_limits<uintptr_t>::max() - mapping) {
      SetError(out_error, "command-aperture copy range is out of bounds");
      return false;
    }
    auto* dst = reinterpret_cast<uint8_t*>(
        mapping + static_cast<uintptr_t>(range.offset));
    const auto* src = static_cast<const uint8_t*>(range.source);
    if ((reinterpret_cast<uintptr_t>(dst) & (kCacheLineSize - 1)) != 0) {
      std::memcpy(dst, src, static_cast<size_t>(range.length));
      const CpuWriteRange write_range = {range.offset, range.length};
      if (!PublishCpuWriteRanges(aperture.gpu_cpu_ptr, aperture.gpu_va_size,
                                 &write_range, 1, 1, out_error)) {
        return false;
      }
      continue;
    }

    const size_t length = static_cast<size_t>(range.length);
    const size_t streamed_length = length & ~(kCacheLineSize - 1);
    for (size_t offset = 0; offset < streamed_length;
         offset += kCacheLineSize) {
      _mm_stream_si128(reinterpret_cast<__m128i*>(dst + offset),
                       _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                           src + offset)));
      _mm_stream_si128(reinterpret_cast<__m128i*>(dst + offset + 16),
                       _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                           src + offset + 16)));
      _mm_stream_si128(reinterpret_cast<__m128i*>(dst + offset + 32),
                       _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                           src + offset + 32)));
      _mm_stream_si128(reinterpret_cast<__m128i*>(dst + offset + 48),
                       _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                           src + offset + 48)));
    }
    used_streaming_stores |= streamed_length != 0;
    if (streamed_length != length) {
      std::memcpy(dst + streamed_length, src + streamed_length,
                  length - streamed_length);
      if (has_clflushopt) {
        FlushCpuCacheLineOptimized(dst + streamed_length);
      } else {
        _mm_clflush(dst + streamed_length);
      }
      used_cached_tail = true;
    }
  }
  if (used_streaming_stores || used_cached_tail) {
    if (has_clflushopt) {
      _mm_sfence();
    } else {
      _mm_mfence();
    }
  }
  return true;
}

bool RefreshPathBCodeMappingAfterWrite(
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
  // invalidation. The marker is the queue-order publication of those slots; a
  // later state-3/chain command may consume them only after this submit.
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

  if (abi.status_has_gpu_va) {
    D3DDDI_MAPGPUVIRTUALADDRESS map = {};
    map.hPagingQueue = device.paging_queue;
    map.hAllocation = ring.allocation;
    map.SizeInPages = kRingSize / 4096;
    map.Protection.Write = 1;
    status = api.map_gpu_virtual_address(&map);
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
  return true;
}


bool SubmitPathBImplNoWait(const KmtApi& api, const Device& device,
                           Context* context, const Buffer& exec_buffer,
                           const void* ert_packet, uint32_t ert_bytes,
                           uint32_t command_state,
                           const PathBChainSubmitInfo* chain_info,
                           uint32_t completion_slot_offset,
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

  if (!IsValidPathBCompletionSlot(ring.size, completion_slot_offset)) {
    SetErrorFormat(out_error,
                   "SubmitPathB invalid completion slot offset=0x%x "
                   "ring_size=0x%llx",
                   completion_slot_offset,
                   static_cast<unsigned long long>(ring.size));
    return false;
  }
  const uint32_t slot_offset = completion_slot_offset;
  uint8_t* slot_cpu = static_cast<uint8_t*>(ring.cpu_ptr) + slot_offset;

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
  out_pending->command_state = chain_info ? 6u : command_state;
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

size_t PathBCompletionCapacity(const Context& context) {
  const size_t slot_count =
      static_cast<size_t>(context.completion_ring.size) /
      kQhdlCompletionSlotSize;
  return slot_count > 1 ? slot_count - 1 : 0;
}

bool IsValidPathBCompletionSlot(uint64_t ring_size, uint32_t slot_offset) {
  return slot_offset >= kQhdlCompletionSlotSize &&
         slot_offset % kQhdlCompletionSlotSize == 0 &&
         slot_offset <= ring_size &&
         kQhdlCompletionSlotSize <= ring_size - slot_offset;
}

bool InitializePathBCompletionSlots(Context* context,
                                    const uint32_t* completion_slot_offsets,
                                    size_t completion_slot_count,
                                    Error* out_error) {
  if (!context || !context->completion_ring_ready ||
      !context->completion_ring.cpu_ptr) {
    SetError(out_error,
             "InitializePathBCompletionSlots called without a completion ring");
    return false;
  }
  if (!completion_slot_offsets || completion_slot_count == 0) {
    SetError(out_error,
             "InitializePathBCompletionSlots called without completion slots");
    return false;
  }

  Buffer& ring = context->completion_ring;
  for (size_t i = 0; i < completion_slot_count; ++i) {
    const uint32_t offset = completion_slot_offsets[i];
    if (!IsValidPathBCompletionSlot(ring.size, offset)) {
      SetErrorFormat(out_error,
                     "InitializePathBCompletionSlots invalid slot offset=0x%x "
                     "ring_size=0x%llx",
                     offset, static_cast<unsigned long long>(ring.size));
      return false;
    }
    for (size_t j = 0; j < i; ++j) {
      if (completion_slot_offsets[j] == offset) {
        SetErrorFormat(
            out_error,
            "InitializePathBCompletionSlots duplicate slot offset=0x%x",
            offset);
        return false;
      }
    }
  }

  // The status allocation is the miniport's coherent qhdl record mapping.
  // Clear every slot before issuing the first command so no later CPU store can
  // share a cache line with a completion that the device is concurrently
  // writing. Do not apply command/data BO cache policy to this driver-owned
  // mapping.
  uint8_t* ring_cpu = static_cast<uint8_t*>(ring.cpu_ptr);
  for (size_t i = 0; i < completion_slot_count; ++i) {
    InitializeCompletionSlot(ring_cpu + completion_slot_offsets[i]);
  }
  std::atomic_thread_fence(std::memory_order_release);
  return true;
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
  if (!IsPathBSubmitComplete(*context, last)) {
    if (!WaitForHwQueueFenceCpu(
            api, device, *context, last.fence_id,
            "D3DKMTWaitForSynchronizationObjectFromCpu(pathb batch)",
            out_error)) {
      return false;
    }
  }
  // Retire each parent through the queue fence interface before consuming its
  // completion slot. The final fence establishes in-order device completion;
  // the per-parent waits mirror the command-specific retirement performed by
  // XRT's runlist wait path and keep miniport completion ownership explicit.
  // These waits observe already-reached fence values and do not serialize
  // parent submission.
  for (size_t i = 0; i < pending_count; ++i) {
    if (!WaitForHwQueueFenceCpu(
            api, device, *context, pending[i].fence_id,
            "D3DKMTWaitForSynchronizationObjectFromCpu(pathb parent retire)",
            out_error)) {
      return false;
    }
  }
  // Every pending parent in a context reports through the same completion
  // ring. The final HWQ fence makes all preceding slots complete; invalidate
  // that ring once, then publish each authoritative slot state into its
  // caller-visible ERT packet. Invalidating every parent exec BO and the same
  // ring once per parent duplicates KMT cache operations without adding an
  // ordering guarantee.
  Buffer ring = pending[0].ring;
  for (size_t i = 1; i < pending_count; ++i) {
    if (pending[i].ring.allocation != ring.allocation) {
      SetError(out_error,
               "pathb batch commands do not share a completion ring");
      return false;
    }
  }
  Error ring_sync_err;
  if (!SyncBuffer(api, device, ring, 0, ring.size, &ring_sync_err)) {
    SetErrorFormat(out_error,
                   "pathb batch completion ring acquire failed: %s",
                   ErrorMessage(&ring_sync_err));
    return false;
  }
  std::atomic_thread_fence(std::memory_order_seq_cst);

  for (size_t i = 0; i < pending_count; ++i) {
    PathBPendingSubmit& p = pending[i];
    volatile uint32_t* const packet_header = p.packet_header;
    uint32_t slot_state = 0;
    std::memcpy(&slot_state, p.slot_cpu, sizeof(slot_state));

    if (packet_header) {
      uint32_t delta = (*packet_header ^ slot_state) & 0xFu;
      *packet_header ^= delta;
    }
    const uint32_t final_state = packet_header ? *packet_header : slot_state;
    if ((final_state & 0xFu) < 4) {
      SetErrorFormat(out_error,
                     "pathb batch command %zu did not complete after final "
                     "fence wait: slot_state=0x%08x slot_offset=0x%x",
                     i, slot_state, p.slot_offset);
      return false;
    }
  }
  return true;
}



bool SubmitPathBChain(const KmtApi& api, const Device& device, Context* context,
                      const Buffer& exec_buffer, const void* ert_packet,
                      uint32_t ert_bytes,
                      const PathBChainSubmitInfo& chain_info,
                      uint32_t completion_slot_offset, uint32_t* packet_header,
                      PathBPendingSubmit* out_pending, Error* out_error) {
  return SubmitPathBImplNoWait(
      api, device, context, exec_buffer, ert_packet, ert_bytes, 6, &chain_info,
      completion_slot_offset, packet_header, out_pending, out_error);
}

bool SubmitPathB(const KmtApi& api, const Device& device, Context* context,
                  const Buffer& exec_buffer, const void* ert_packet,
                  uint32_t ert_bytes, uint32_t command_state,
                  uint32_t completion_slot_offset, uint32_t* packet_header,
                  PathBPendingSubmit* out_pending, Error* out_error) {
  return SubmitPathBImplNoWait(api, device, context, exec_buffer, ert_packet,
                               ert_bytes, command_state, /*chain_info=*/nullptr,
                               completion_slot_offset, packet_header,
                               out_pending, out_error);
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
