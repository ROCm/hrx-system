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

// D3DKMT adapter enumeration is bounded so discovery uses fixed storage and
// retains pre-selection handles without allocation during device bring-up.
constexpr size_t kMaxComputeAdapterHandles = 256;
constexpr size_t kMaxMcdmPrivateDataSize = 0x280;
constexpr size_t kMaxPathBBoTableEntries = 6;
// PARTIAL_ELF command BOs carry this out-of-packet table. The miniport reads
// it independently of the ERT packet, so every submit that consumes the BO
// must SyncBuffer this exact range (or the whole child exec BO). That is a
// command-BO publication domain, not opcode-9.
constexpr size_t kPathBBoTableOffset = 11 * sizeof(uint32_t);
constexpr size_t kPathBBoTableSize =
    2 * kMaxPathBBoTableEntries * sizeof(uint32_t);
constexpr size_t kCompactPathBChainHandleSize = 0x120;

// Miniport-facing runlist child record for the compact MCDM contract. The
// semantics of the three observed state fields are not published, so retain
// neutral names while making the captured byte layout explicit and checked.
struct alignas(uint64_t) CompactPathBChainHandleV1 {
  uint8_t reserved_00[0x18] = {};
  uint64_t requested_size = 0;
  uint64_t xcl_flags = 0;
  uint8_t reserved_28[0x10] = {};
  uint64_t allocation = 0;
  uint8_t reserved_40[0x18] = {};
  uint64_t page_count = 0;
  uint64_t gpu_va = 0;
  uint8_t reserved_68[0x08] = {};
  uint64_t cpu_ptr = 0;
  uint8_t reserved_78[0x08] = {};
  uint64_t observed_state_80 = 0;
  uint8_t reserved_88[0x40] = {};
  uint64_t observed_state_c8 = 0;
  uint64_t observed_state_d0 = 0;
  uint8_t reserved_d8[0x48] = {};
};

static_assert(offsetof(CompactPathBChainHandleV1, requested_size) == 0x18);
static_assert(offsetof(CompactPathBChainHandleV1, xcl_flags) == 0x20);
static_assert(offsetof(CompactPathBChainHandleV1, allocation) == 0x38);
static_assert(offsetof(CompactPathBChainHandleV1, page_count) == 0x58);
static_assert(offsetof(CompactPathBChainHandleV1, gpu_va) == 0x60);
static_assert(offsetof(CompactPathBChainHandleV1, cpu_ptr) == 0x70);
static_assert(offsetof(CompactPathBChainHandleV1, observed_state_80) == 0x80);
static_assert(offsetof(CompactPathBChainHandleV1, observed_state_c8) == 0xc8);
static_assert(offsetof(CompactPathBChainHandleV1, observed_state_d0) == 0xd0);
static_assert(sizeof(CompactPathBChainHandleV1) ==
              kCompactPathBChainHandleSize);

struct Error {
  char message[512] = {};
};

const char* ErrorMessage(const Error* error);

enum class BufferKind {
  host_only,
  cacheable,
  execbuf,
  context_private,
};

// Packet/context layout contract. Distinct from HardwareType: STXH and KRK1
// share `legacy`, and STX2 needs the INF revision to choose between
// `legacy_v2` and `legacy` because one hardware type shipped two layouts.
// Driver package generations (two-DWORD vs compact, 329 vs 3760 vs 3930) are
// also distinct from HardwareType: the same SKU keeps the same hw_type across
// those packages. Strix Halo is always stxh (PCI REV_11), whether the miniport
// answers with two DWORDs {0, 3} or compact {0, 3, 0}.
enum class McdmAbi {
  legacy_v0,
  legacy_v2,
  legacy,
  compact,
};

// PCI SKU from XRT_UMD_ADAPTER_INFO.hw_type
// (D3DKMTQueryAdapterInfo(KMTQAITYPE_UMDRIVERPRIVATE)). Not a driver-package
// version: 329/3760/3930 on Halo all report stxh. Word 0 is kmd_version
// (still 0 in current miniports). Compact packages reject the historical
// two-DWORD prefix with STATUS_BUFFER_TOO_SMALL and accept three DWORDs
// {kmd_version, hw_type, 0}.
enum class HardwareType : uint32_t {
  phx = 0,
  stx = 1,
  stx2 = 2,  // Strix B0 / PCI REV_10; Gorgon Point reuses this PCI identity.
  stxh = 3,  // Strix Halo / PCI REV_11.
  krk1 = 4,  // Krackan / PCI REV_20.
};

const char* HardwareTypeName(uint32_t hw_type);

struct McdmSubmissionPolicy {
  bool supports_command_chaining = false;
  bool uses_shared_command_code_view = false;
  bool submit_completion_is_deferred = false;
};

McdmSubmissionPolicy GetMcdmSubmissionPolicy(McdmAbi abi);
bool SupportsHostBufferReuse(McdmAbi abi);

enum class McdmAbiSource {
  unknown,
  // D3DKMTQueryAdapterInfo(KMTQAITYPE_UMDRIVERPRIVATE) adapter-info prefix.
  identity_query,
};

struct McdmAbiDiagnostics {
  McdmAbi selected_abi = McdmAbi::legacy;
  McdmAbi probed_abi = McdmAbi::legacy;
  McdmAbiSource source = McdmAbiSource::unknown;
  uint32_t kmd_version = 0;
  uint32_t hw_type = 0;
  bool compact_adapter_info = false;
  // Raw UMDRIVERPRIVATE prefix. Word 0 is kmd_version, word 1 is hw_type.
  uint32_t identity_words[3] = {};
  uint32_t identity_word_count = 0;
  uint32_t accepted_identity_count = 0;
  bool identity_accepted = false;
  bool driver_version_disambiguation_required = false;
  bool driver_version_disambiguation_available = false;
  bool compact_query_attempted = false;
  bool legacy_query_attempted = false;
  NTSTATUS compact_query_status = 0;
  NTSTATUS legacy_query_status = 0;
};

struct DriverVersion {
  uint32_t major = 0;
  uint32_t minor = 0;
  uint32_t build = 0;
  uint32_t revision = 0;
};

enum class CommandApertureWritePublishMode {
  cpu_cache_flush,
  kmt_invalidate,
};

struct McdmAbiInfo {
  uint32_t status_private_type;
  uint32_t status_policy;
  uint32_t status_xcl_flags;
  uint32_t submit_private_prefix_size;
  uint32_t setup_private_size;
  uint32_t pathb_private_size;
  uint32_t pathb_packet_offset;
  uint32_t chain_metadata_offset;
  uint32_t pathb_bo_table_entry_count;
  bool status_has_gpu_va;
  bool sync_has_allocation_handle;
  uint64_t command_aperture_code_slot_size;
  CommandApertureWritePublishMode command_aperture_write_publish_mode;
  uint64_t command_aperture_code_publish_granularity;
  bool command_aperture_residency_after_bootstrap;
  bool command_aperture_remap_after_write;
  // Exact D3DDDICB_DESTROYALLOCATION2FLAGS::Value required when destroying a
  // shared resource. Zero retains the legacy per-object teardown behavior.
  uint32_t shared_resource_destroy_flags;
  // Whether teardown explicitly releases mapped GPU VAs before destroying
  // their allocations. WDDM allocation destruction also owns its mapped VA.
  bool explicit_gpu_va_free_on_destroy;
};

McdmAbiInfo GetMcdmAbiInfo(McdmAbi abi);

bool SelectMcdmAbiForDriverVersion(McdmAbi probed_abi,
                                   bool has_driver_version,
                                   const DriverVersion& driver_version,
                                   McdmAbi* out_abi, Error* out_error);

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
  PFND3DKMT_QUERYVIDEOMEMORYINFO query_video_memory_info = nullptr;
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

bool QueryProbedMcdmAbi(const KmtApi& api, D3DKMT_HANDLE adapter,
                        McdmAbi* out_abi, Error* out_error);
bool QueryMcdmAbiDiagnostics(const KmtApi& api, D3DKMT_HANDLE adapter,
                             McdmAbiDiagnostics* out_diagnostics,
                             Error* out_error);

struct Adapter {
  D3DKMT_HANDLE handle = 0;
  LUID luid = {};
  D3DKMT_HANDLE retained_handles[kMaxComputeAdapterHandles] = {};
  size_t retained_handle_count = 0;
};

struct Device {
  D3DKMT_HANDLE adapter = 0;
  D3DKMT_HANDLE retained_adapter_handles[kMaxComputeAdapterHandles] = {};
  size_t retained_adapter_handle_count = 0;
  D3DKMT_HANDLE device = 0;
  D3DKMT_HANDLE paging_queue = 0;
  D3DKMT_HANDLE paging_sync_object = 0;
  void* paging_fence_cpu = nullptr;
  // Highest paging-queue fence returned by Map/MakeResident. XRT drains this
  // device-wide watermark before every HW-queue submit.
  mutable volatile LONG64 pending_paging_fence_value = 0;
  McdmAbi mcdm_abi = McdmAbi::legacy;
  McdmAbiDiagnostics mcdm_abi_diagnostics = {};
  DriverVersion driver_version = {};
  bool has_driver_version = false;
};

struct Buffer {
  BufferKind kind = BufferKind::host_only;
  // Logical BO size exposed to the HAL/runtime.
  uint64_t size = 0;
  // Size requested from the miniport before page rounding. Compact exec BOs
  // include the private-submit prefix in addition to the logical command page.
  uint64_t requested_size = 0;
  // Actual GPU VA reservation size. XRT often requests a non-page-sized BO
  // and maps the page-rounded allocation returned by the driver.
  uint64_t mapped_size = 0;
  D3DKMT_HANDLE allocation = 0;
  D3DKMT_HANDLE resource = 0;
  D3DGPU_VIRTUAL_ADDRESS gpu_va = 0;
  void* cpu_ptr = nullptr;
  UINT64 paging_fence_value = 0;
  // The compact runlist ABI uses a miniport-facing BO record in each chain
  // entry. Keep its representation owned by this DDI layer.
  CompactPathBChainHandleV1 compact_chain_handle;
};

struct Context {
  D3DKMT_HANDLE context = 0;
  D3DKMT_HANDLE hw_queue = 0;
  D3DKMT_HANDLE progress_fence = 0;
  void* progress_fence_cpu = nullptr;
  D3DGPU_VIRTUAL_ADDRESS progress_fence_gpu = 0;
  uint64_t next_fence_id = 1;
  // Highest paging-queue fence ordered onto this hardware queue. Queue order
  // preserves that dependency for later submits, so only newly encountered
  // paging work needs another GPU wait.
  uint64_t ordered_paging_fence_value = 0;
  // Driver writeback in the context-private packet. XRT folds this cookie into
  // the 64 MiB command-aperture allocation private flags.
  uint32_t command_aperture_cookie = 0;
  // Compact-context ABI storage referenced by the 160-byte private packet.
  // It must remain alive until after the hardware queue is destroyed.
  Buffer context_private_buffer;
  // Path B (hwqueue_aie4-style per-dispatch submit) completion state. The
  // negotiated MCDM ABI selects the status object type and submit layout.
  Buffer completion_ring;
  D3DKMT_HANDLE completion_ring_resource = 0;
  bool completion_ring_ready = false;
  bool completion_ring_owned = false;
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
  D3DGPU_VIRTUAL_ADDRESS status_gpu_va = 0;
  // Allocator-selected process VA used by KMT for mapping and bootstrap.
  D3DGPU_VIRTUAL_ADDRESS gpu_va = 0;
  // Context-local address advertised to the MCDM protocol. Each context sees
  // its command aperture at this address even when KMT assigns distinct VAs.
  D3DGPU_VIRTUAL_ADDRESS protocol_gpu_va = 0;
  void* cpu_ptr = nullptr;
  void* gpu_cpu_ptr = nullptr;
  uint64_t cpu_ptr_size = 0;
  // Control-code view inside the 64 MiB aperture BO. The negotiated MCDM ABI
  // determines its offset; callers consume this view without knowing the
  // miniport-specific layout.
  D3DKMT_HANDLE code_allocation = 0;
  D3DKMT_HANDLE code_resource = 0;
  uint64_t code_offset = 0;
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

struct McdmPrivateData {
  uint8_t data[kMaxMcdmPrivateDataSize] = {};
  uint32_t size = 0;
};

McdmPrivateData BuildPathBSetupPrivateData(
    McdmAbi abi, const CommandAperture& aperture);

McdmPrivateData BuildPathBSyncPrivateData(McdmAbi abi,
                                          const CommandAperture& aperture,
                                          uint64_t offset);

McdmPrivateData BuildPathBSubmitPrivateData(
    McdmAbi abi, const Buffer& exec_buffer, const Buffer& completion_ring,
    uint32_t completion_slot_offset, const void* completion_slot_cpu,
    const void* ert_packet, uint32_t ert_bytes, uint32_t command_state,
    const PathBChainSubmitInfo* chain_info);

struct PathBPendingSubmit {
  uint64_t fence_id = 0;
  uint32_t command_state = 0;
  uint8_t* slot_cpu = nullptr;
  uint32_t slot_offset = 0;
  volatile uint32_t* packet_header = nullptr;
  Buffer exec_buffer;
  Buffer ring;
};

// Returns the number of simultaneously pending Path-B submissions supported
// by the context's completion ring after reserving the protocol-owned slot.
size_t PathBCompletionCapacity(const Context& context);

bool IsValidPathBCompletionSlot(uint64_t ring_size, uint32_t slot_offset);

// Clears all completion records reserved for one native submission before any
// command is issued. The completion ring is a driver-owned coherent mapping;
// callers must not apply ordinary BO cache maintenance to it.
bool InitializePathBCompletionSlots(Context* context,
                                    const uint32_t* completion_slot_offsets,
                                    size_t completion_slot_count,
                                    Error* out_error);

struct CpuWriteRange {
  uint64_t offset = 0;
  uint64_t length = 0;
};

struct CpuCopyRange {
  uint64_t offset = 0;
  const void* source = nullptr;
  uint64_t length = 0;
};

bool FindNpuAdapter(const KmtApi& api, Adapter* out_adapter, Error* out_error);

bool CreateDevice(const KmtApi& api, const Adapter& adapter, Device* out_device,
                  Error* out_error);

void DestroyDevice(const KmtApi& api, Device* device);

bool CreateBuffer(const KmtApi& api, const Device& device, BufferKind kind,
                  uint64_t size, Buffer* out_buffer, Error* out_error);

// Makes CPU writes to a Lock2-mapped buffer visible before device execution.
// Buffer ownership must synchronize all writers into the calling thread.
bool PublishBufferCpuWrites(const Buffer& buffer, uint64_t offset,
                            uint64_t length, Error* out_error);

// Invalidates CPU cache lines for a Lock2-mapped buffer after device execution.
// The caller must first wait for the device write to complete.
bool InvalidateBufferCpuReads(const Buffer& buffer, uint64_t offset,
                              uint64_t length, Error* out_error);

// Returns the miniport-facing child handle stored in an ERT_CMD_CHAIN entry.
// The negotiated device contract selects the record shape inside this DDI.
uint64_t GetPathBChainChildHandle(const Device& device, Buffer* buffer);

bool SyncBuffer(const KmtApi& api, const Device& device, const Buffer& buffer,
                uint64_t offset, uint64_t length, Error* out_error);

bool SyncCommandApertureCode(const KmtApi& api, const Device& device,
                             const CommandAperture& aperture, uint64_t offset,
                             uint64_t length, Error* out_error);

bool PopulatePathBBoTable(
    const Device& device, void* command_bo, size_t command_bo_size,
    const D3DGPU_VIRTUAL_ADDRESS* real_bo_gpu_vas,
    size_t real_bo_entry_count, Error* out_error);

bool RefreshCommandApertureGpuMapping(const KmtApi& api, const Device& device,
                                      CommandAperture* aperture,
                                      Error* out_error);

bool RefreshBufferCpuMapping(const KmtApi& api, const Device& device,
                             Buffer* buffer, Error* out_error);

bool WaitForBufferResidency(const KmtApi& api, const Device& device,
                            Context& context, const Buffer& buffer,
                            const char* label, Error* out_error);

void DestroyBuffer(const KmtApi& api, const Device& device, Buffer* buffer);

bool CreateContext(const KmtApi& api, const Device& device,
                   const uint8_t* private_data, size_t private_data_size,
                   Context* out_context, Error* out_error);

void DestroyContext(const KmtApi& api, const Device& device, Context* context);

// Tears down a context and its command aperture in the ABI-defined ownership
// order. Compact MCDM has separate instruction and queue-status allocations;
// the instruction allocation is released before the HW queue and the status
// allocation after it. Legacy MCDM retains its established one-phase aperture
// teardown before context destruction.
void DestroyContextWithCommandAperture(const KmtApi& api,
                                       const Device& device,
                                       Context* context,
                                       CommandAperture* aperture);

bool CreateCommandAperture(const KmtApi& api, const Device& device,
                           const Context& context,
                           CommandAperture* out_aperture, Error* out_error);

// Configures the transaction-code view after reserving the setup payload in
// the command aperture. Compact MCDM uses a context-local 0x8000-byte slot
// allocator, while legacy MCDM retains its fixed code offset.
bool ConfigurePathBCodeRangeForSetupPayload(
    McdmAbi abi, size_t aperture_payload_size, CommandAperture* aperture,
    Error* out_error);

bool EnsureCommandApertureGpuMapping(const KmtApi& api, const Device& device,
                                     CommandAperture* aperture,
                                     Error* out_error);

bool ReleaseCommandApertureGpuMapping(const KmtApi& api, const Device& device,
                                      CommandAperture* aperture,
                                      Error* out_error);

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

// Path-B aperture publication is two independent happens-before edges:
//
//   1. CommitPathBCodeWrite / CopyAndCommitPathBCodeWrites makes CPU stores
//      device-visible (compact: cache flush of each 0x8000 slot; legacy:
//      D3DKMTInvalidateCache). Skip this only when the CPU did not write
//      AND the previous device image is known still resident. Compact Path-B
//      firmware does not retain aperture GPU contents across consumes, so
//      native submit restages from the CPU source of truth and
//      KMT-invalidates the allocation before every consume. Compact Commit
//      is CPU clflush only and does not snoop the NPU cache.
//   2. PublishPathBCodeWrite / opcode-9 end markers publish those slots in
//      HW-queue order so a later state-3/chain command may consume them.
//      Opcode-9 is not a copy and not a sticky mapping. Firmware that
//      retains published slots can appear to tolerate a skip-Commit; that
//      is not a portable MCDM contract. Every consumer must restage then
//      publish the exact range it will execute.
//
// Acquisition validates the mapped range. After execution, start-boundary
// markers release the slots and the final release retires before teardown.
// Keeping those details here lets the native layer use one
// acquire/write/publish/release sequence for both ABIs.
bool AcquirePathBCodeRange(const KmtApi& api, const Device& device,
                           Context* context,
                           const CommandAperture& aperture, uint64_t offset,
                           uint64_t length, Error* out_error);

bool CommitPathBCodeWrite(const KmtApi& api, const Device& device,
                           const CommandAperture& aperture, uint64_t offset,
                           uint64_t length, Error* out_error);

bool CommitPathBCodeWrites(const KmtApi& api, const Device& device,
                           const CommandAperture& aperture,
                           const CpuWriteRange* ranges, size_t range_count,
                           Error* out_error);

// Copies CPU source ranges into the command aperture with non-temporal stores
// and publishes them before a later opcode-9 marker submission.
bool CopyAndCommitPathBCodeWrites(const CommandAperture& aperture,
                                  const CpuCopyRange* ranges,
                                  size_t range_count, Error* out_error);

bool RefreshPathBCodeMappingAfterWrite(
    const KmtApi& api, const Device& device, CommandAperture* aperture,
    Error* out_error);

// Queue-ordered opcode-9 publication of `offset`/`length`. Does not copy
// bytes. Native submit restages (copy+Commit) first, then calls this before
// every state-3/chain consumer of that range.
bool PublishPathBCodeWrite(const KmtApi& api, const Device& device,
                           Context* context,
                           const CommandAperture& aperture, uint64_t offset,
                           uint64_t length, Error* out_error);

bool ReleasePathBCodeRange(const KmtApi& api, const Device& device,
                           Context* context,
                           const CommandAperture& aperture, uint64_t offset,
                           uint64_t length, Error* out_error);

// Path B: per-dispatch hwqueue_aie4-style submit. Reserve an 8-byte completion
// slot, build the driver-negotiated private packet with the ERT packet inline,
// and submit the exec BO via SubmitCommandToHwQueue. `ert_packet`/`ert_bytes`
// are the command BO's ERT packet; `packet_header` is updated with the firmware
// completion state read back from the ring slot.
bool SubmitPathBChain(const KmtApi& api, const Device& device, Context* context,
                      const Buffer& exec_buffer, const void* ert_packet,
                      uint32_t ert_bytes,
                      const PathBChainSubmitInfo& chain_info,
                      uint32_t completion_slot_offset, uint32_t* packet_header,
                      PathBPendingSubmit* out_pending, Error* out_error);

// Single-dispatch path-B issue. Returns the in-flight fence token in
// `out_pending`; wait for it with WaitForPathBSubmits.
bool SubmitPathB(const KmtApi& api, const Device& device, Context* context,
                  const Buffer& exec_buffer, const void* ert_packet,
                  uint32_t ert_bytes, uint32_t command_state,
                  uint32_t completion_slot_offset, uint32_t* packet_header,
                  PathBPendingSubmit* out_pending, Error* out_error);

bool WaitForPathBSubmits(const KmtApi& api, const Device& device,
                         Context* context, PathBPendingSubmit* pending,
                         size_t pending_count, Error* out_error);

// Non-blocking completion poll for an issued (but not yet waited) path-B
// submit: true once the HW progress fence has reached pending.fence_id.
bool IsPathBSubmitComplete(const Context& context,
                           const PathBPendingSubmit& pending);

// Releases command-aperture allocations. Context owners should prefer
// DestroyContextWithCommandAperture so ABI-specific ownership order is kept.
void DestroyCommandAperture(const KmtApi& api, const Device& device,
                            CommandAperture* aperture);

}  // namespace iree::hal::amdxdna::mcdm

#endif  // IREE_AMD_AIE_DRIVER_AMDXDNA_SHIM_WINDOWS_MCDM_KMT_API_H_
