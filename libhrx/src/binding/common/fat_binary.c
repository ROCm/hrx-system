// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "common/fat_binary.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"

#if defined(HRX_ENABLE_ZSTD)
#include <zstd.h>
#endif

#define HRX_FAT_EXECUTABLE_FORMAT_CAPACITY \
  IREE_HAL_STREAMING_FAT_BINARY_FORMAT_CAPACITY

//===----------------------------------------------------------------------===//
// Format magic & on-disk structures
//===----------------------------------------------------------------------===//

// HIP fat-binary wrapper magic (__hipFatBinaryWrapper.magic). HIP uses two:
// the Clang-emitted "HIPF" (0x48495046), and the runtime-emitted sentinel
// 0xBA55FACE that ships via `__hipRegisterFatBinary` in more recent ROCm.
#define HRX_HIP_FAT_MAGIC_HIPF 0x48495046u
#define HRX_HIP_FAT_MAGIC_BA55FACE 0xBA55FACEu
#define HRX_HIP_FAT_VERSION 1

// __CLANG_OFFLOAD_BUNDLE__ (uncompressed bundle).
#define HRX_OFFLOAD_BUNDLE_MAGIC "__CLANG_OFFLOAD_BUNDLE__"
#define HRX_OFFLOAD_BUNDLE_MAGIC_SIZE 24

// CCOB (Compressed Clang Offload Bundle).
#define HRX_CCOB_MAGIC "CCOB"
#define HRX_CCOB_MAGIC_SIZE 4
#define HRX_CCOB_MAGIC_INT 0x424f4343u  // 'C','C','O','B' little-endian

// ELF64 little-endian AMDGPU.
#define HRX_ELF_MAGIC_INT 0x464c457fu  // 0x7f 'E' 'L' 'F'
#define HRX_ELFCLASS64 2
#define HRX_ELFDATA2LSB 1
#define HRX_ELFOSABI_HSA 64
#define HRX_EM_AMDGPU 224
#define HRX_EV_CURRENT 1

#define HRX_ELF_HSA_ABI_VERSION_V3 1
#define HRX_ELF_HSA_ABI_VERSION_V4 2
#define HRX_ELF_HSA_ABI_VERSION_V5 3
#define HRX_ELF_HSA_ABI_VERSION_V6 4

#define HRX_EF_AMDGPU_MACH 0x0FFu
#define HRX_EF_AMDGPU_FEATURE_XNACK_V3 0x100u
#define HRX_EF_AMDGPU_FEATURE_SRAMECC_V3 0x200u
#define HRX_EF_AMDGPU_FEATURE_XNACK_V4 0x300u
#define HRX_EF_AMDGPU_FEATURE_XNACK_UNSUPPORTED_V4 0x000u
#define HRX_EF_AMDGPU_FEATURE_XNACK_ANY_V4 0x100u
#define HRX_EF_AMDGPU_FEATURE_XNACK_OFF_V4 0x200u
#define HRX_EF_AMDGPU_FEATURE_XNACK_ON_V4 0x300u
#define HRX_EF_AMDGPU_FEATURE_SRAMECC_V4 0xC00u
#define HRX_EF_AMDGPU_FEATURE_SRAMECC_UNSUPPORTED_V4 0x000u
#define HRX_EF_AMDGPU_FEATURE_SRAMECC_ANY_V4 0x400u
#define HRX_EF_AMDGPU_FEATURE_SRAMECC_OFF_V4 0x800u
#define HRX_EF_AMDGPU_FEATURE_SRAMECC_ON_V4 0xC00u
#define HRX_EF_AMDGPU_GENERIC_VERSION 0xFF000000u
#define HRX_EF_AMDGPU_GENERIC_VERSION_OFFSET 24

#define HRX_CCOB_METHOD_ZLIB 0
#define HRX_CCOB_METHOD_ZSTD 1

typedef struct hrx_hip_fat_binary_header_t {
  uint32_t magic;
  uint32_t version;
  void* binary;
  void* reserved;
} hrx_hip_fat_binary_header_t;
static_assert(sizeof(hrx_hip_fat_binary_header_t) == 24,
              "HIP fat-binary wrapper must be 24 bytes");

typedef struct hrx_bundle_entry_t {
  uint64_t offset;
  uint64_t size;
  uint64_t triple_size;
} hrx_bundle_entry_t;

typedef struct hrx_ccob_header_v1_t {
  uint8_t magic[4];
  uint16_t version;
  uint16_t method;
  uint32_t uncompressed_size;
  uint64_t hash;
} hrx_ccob_header_v1_t;

typedef struct hrx_ccob_header_v2_t {
  uint8_t magic[4];
  uint16_t version;
  uint16_t method;
  uint32_t file_size;
  uint32_t uncompressed_size;
  uint64_t hash;
} hrx_ccob_header_v2_t;

typedef struct hrx_ccob_header_v3_t {
  uint8_t magic[4];
  uint16_t version;
  uint16_t method;
  uint64_t file_size;
  uint64_t uncompressed_size;
  uint64_t hash;
} hrx_ccob_header_v3_t;

typedef struct hrx_elf64_header_t {
  uint8_t magic[4];
  uint8_t elf_class;
  uint8_t elf_data;
  uint8_t elf_version;
  uint8_t osabi;
  uint8_t abiversion;
  uint8_t padding[7];
  uint16_t type;
  uint16_t machine;
  uint32_t version;
  uint64_t entry;
  uint64_t phoff;
  uint64_t shoff;
  uint32_t flags;
  uint16_t ehsize;
  uint16_t phentsize;
  uint16_t phnum;
  uint16_t shentsize;
  uint16_t shnum;
  uint16_t shstrndx;
} hrx_elf64_header_t;
static_assert(sizeof(hrx_elf64_header_t) == 64,
              "ELF64 header must be 64 bytes");

typedef enum hrx_amdgpu_feature_state_e {
  HRX_AMDGPU_FEATURE_STATE_ANY = 0,
  HRX_AMDGPU_FEATURE_STATE_UNSUPPORTED,
  HRX_AMDGPU_FEATURE_STATE_OFF,
  HRX_AMDGPU_FEATURE_STATE_ON,
} hrx_amdgpu_feature_state_t;

typedef struct hrx_amdgpu_elf_machine_target_t {
  // AMDGPU EF_AMDGPU_MACH_* value.
  uint32_t machine;
  // Processor string represented by |machine|.
  iree_string_view_t processor;
  // True if old V3 e_flags can explicitly encode SRAM ECC off for this target.
  bool sramecc_supported;
  // True if old V3 e_flags can explicitly encode XNACK off for this target.
  bool xnack_supported;
} hrx_amdgpu_elf_machine_target_t;

static const hrx_amdgpu_elf_machine_target_t hrx_amdgpu_elf_machine_targets[] =
    {
#define IREE_AMDGPU_ELF_MACHINE_TARGET(machine, processor, sramecc, xnack) \
  {machine, IREE_SVL(processor), sramecc, xnack},
#include "build_tools/amdgpu/elf_machine_map.inl"
#undef IREE_AMDGPU_ELF_MACHINE_TARGET
};

//===----------------------------------------------------------------------===//
// Format sniffers
//===----------------------------------------------------------------------===//

// A data_length of 0 is treated as "size unknown, sniff from magic only".
// Callers like __hipRegisterFatBinary hand us a raw pointer without a size
// because the HIP ABI doesn't carry one, and we only need the first few bytes
// to detect the container format.
static bool hrx_fat_length_at_least(iree_const_byte_span_t data,
                                    iree_host_size_t min_bytes) {
  return data.data_length == 0 || data.data_length >= min_bytes;
}

static bool hrx_fat_is_elf(iree_const_byte_span_t data) {
  if (!data.data || !hrx_fat_length_at_least(data, 4)) return false;
  uint32_t magic;
  memcpy(&magic, data.data, sizeof(magic));
  return magic == HRX_ELF_MAGIC_INT;
}

static bool hrx_fat_is_uncompressed_bundle(iree_const_byte_span_t data) {
  if (!data.data ||
      !hrx_fat_length_at_least(data, HRX_OFFLOAD_BUNDLE_MAGIC_SIZE)) {
    return false;
  }
  return memcmp(data.data, HRX_OFFLOAD_BUNDLE_MAGIC,
                HRX_OFFLOAD_BUNDLE_MAGIC_SIZE) == 0;
}

static bool hrx_fat_is_ccob(iree_const_byte_span_t data) {
  if (!data.data || !hrx_fat_length_at_least(data, HRX_CCOB_MAGIC_SIZE)) {
    return false;
  }
  return memcmp(data.data, HRX_CCOB_MAGIC, HRX_CCOB_MAGIC_SIZE) == 0;
}

static bool hrx_fat_is_wrapper(iree_const_byte_span_t data) {
  if (!data.data ||
      !hrx_fat_length_at_least(data, sizeof(hrx_hip_fat_binary_header_t))) {
    return false;
  }
  uint32_t magic;
  memcpy(&magic, data.data, sizeof(magic));
  return magic == HRX_HIP_FAT_MAGIC_HIPF || magic == HRX_HIP_FAT_MAGIC_BA55FACE;
}

bool iree_hal_streaming_fat_binary_is_supported(iree_const_byte_span_t data) {
  return hrx_fat_is_elf(data) || hrx_fat_is_uncompressed_bundle(data) ||
         hrx_fat_is_ccob(data) || hrx_fat_is_wrapper(data);
}

//===----------------------------------------------------------------------===//
// ELF / CCOB header validation
//===----------------------------------------------------------------------===//

static const hrx_amdgpu_elf_machine_target_t*
hrx_amdgpu_lookup_elf_machine_target(uint32_t machine) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(hrx_amdgpu_elf_machine_targets); ++i) {
    if (hrx_amdgpu_elf_machine_targets[i].machine == machine) {
      return &hrx_amdgpu_elf_machine_targets[i];
    }
  }
  return NULL;
}

static hrx_amdgpu_feature_state_t hrx_amdgpu_decode_v4_feature(
    uint32_t flags, uint32_t mask, uint32_t any_value, uint32_t off_value,
    uint32_t on_value) {
  const uint32_t value = flags & mask;
  if (value == any_value) {
    return HRX_AMDGPU_FEATURE_STATE_ANY;
  } else if (value == off_value) {
    return HRX_AMDGPU_FEATURE_STATE_OFF;
  } else if (value == on_value) {
    return HRX_AMDGPU_FEATURE_STATE_ON;
  }
  return HRX_AMDGPU_FEATURE_STATE_UNSUPPORTED;
}

static bool hrx_amdgpu_processor_is_generic(iree_string_view_t processor) {
  return iree_string_view_ends_with(processor, IREE_SV("-generic"));
}

static iree_status_t hrx_fat_append_format_string(
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* length, iree_string_view_t value) {
  if (*length + value.size >= executable_format_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU executable format buffer too small");
  }
  memcpy(executable_format + *length, value.data, value.size);
  *length += value.size;
  executable_format[*length] = '\0';
  return iree_ok_status();
}

static iree_status_t hrx_fat_append_format_feature(
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* length, iree_string_view_t name,
    hrx_amdgpu_feature_state_t state) {
  if (state != HRX_AMDGPU_FEATURE_STATE_OFF &&
      state != HRX_AMDGPU_FEATURE_STATE_ON) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(hrx_fat_append_format_string(
      executable_format_capacity, executable_format, length, IREE_SV(":")));
  IREE_RETURN_IF_ERROR(hrx_fat_append_format_string(
      executable_format_capacity, executable_format, length, name));
  return hrx_fat_append_format_string(
      executable_format_capacity, executable_format, length,
      state == HRX_AMDGPU_FEATURE_STATE_ON ? IREE_SV("+") : IREE_SV("-"));
}

static iree_status_t hrx_fat_format_amdgpu_executable_format(
    const hrx_amdgpu_elf_machine_target_t* machine_target,
    hrx_amdgpu_feature_state_t sramecc, hrx_amdgpu_feature_state_t xnack,
    iree_host_size_t executable_format_capacity, char* executable_format) {
  if (executable_format_capacity == 0 || executable_format == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable format output buffer is empty");
  }
  executable_format[0] = '\0';
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(hrx_fat_append_format_string(executable_format_capacity,
                                                    executable_format, &length,
                                                    machine_target->processor));
  IREE_RETURN_IF_ERROR(hrx_fat_append_format_feature(
      executable_format_capacity, executable_format, &length,
      IREE_SV("sramecc"), sramecc));
  return hrx_fat_append_format_feature(executable_format_capacity,
                                       executable_format, &length,
                                       IREE_SV("xnack"), xnack);
}

// Validates an AMDGPU ELF header, computes the total on-disk ELF size from the
// section-header table, and derives the AMDGPU HAL executable format string.
iree_status_t iree_hal_streaming_fat_binary_describe_amdgpu_elf(
    iree_const_byte_span_t elf, iree_host_size_t executable_format_capacity,
    char* executable_format, iree_host_size_t* out_size) {
  if (!hrx_fat_length_at_least(elf, sizeof(hrx_elf64_header_t))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF data too small (got %" PRIhsz ")",
                            elf.data_length);
  }
  hrx_elf64_header_t h;
  memcpy(&h, elf.data, sizeof(h));
  if (h.elf_class != HRX_ELFCLASS64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF class must be 64-bit, got %u", h.elf_class);
  }
  if (h.elf_data != HRX_ELFDATA2LSB) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF must be little-endian, got %u", h.elf_data);
  }
  if (h.elf_version != HRX_EV_CURRENT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported ELF ident version %u", h.elf_version);
  }
  if (h.osabi != HRX_ELFOSABI_HSA) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF OSABI must be HSA (%u), got %u",
                            HRX_ELFOSABI_HSA, h.osabi);
  }
  if (h.machine != HRX_EM_AMDGPU) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF machine must be AMDGPU (%u), got %u",
                            HRX_EM_AMDGPU, h.machine);
  }
  if (h.version != HRX_EV_CURRENT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported ELF version %u", h.version);
  }

  const uint32_t machine = h.flags & HRX_EF_AMDGPU_MACH;
  const hrx_amdgpu_elf_machine_target_t* machine_target =
      hrx_amdgpu_lookup_elf_machine_target(machine);
  if (machine_target == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported AMDGPU ELF machine value 0x%x",
                            machine);
  }

  hrx_amdgpu_feature_state_t sramecc = HRX_AMDGPU_FEATURE_STATE_UNSUPPORTED;
  hrx_amdgpu_feature_state_t xnack = HRX_AMDGPU_FEATURE_STATE_UNSUPPORTED;
  if (h.abiversion == HRX_ELF_HSA_ABI_VERSION_V3) {
    sramecc = iree_all_bits_set(h.flags, HRX_EF_AMDGPU_FEATURE_SRAMECC_V3)
                  ? HRX_AMDGPU_FEATURE_STATE_ON
              : machine_target->sramecc_supported
                  ? HRX_AMDGPU_FEATURE_STATE_OFF
                  : HRX_AMDGPU_FEATURE_STATE_UNSUPPORTED;
    xnack = iree_all_bits_set(h.flags, HRX_EF_AMDGPU_FEATURE_XNACK_V3)
                ? HRX_AMDGPU_FEATURE_STATE_ON
            : machine_target->xnack_supported
                ? HRX_AMDGPU_FEATURE_STATE_OFF
                : HRX_AMDGPU_FEATURE_STATE_UNSUPPORTED;
  } else if (h.abiversion == HRX_ELF_HSA_ABI_VERSION_V4 ||
             h.abiversion == HRX_ELF_HSA_ABI_VERSION_V5 ||
             h.abiversion == HRX_ELF_HSA_ABI_VERSION_V6) {
    sramecc =
        hrx_amdgpu_decode_v4_feature(h.flags, HRX_EF_AMDGPU_FEATURE_SRAMECC_V4,
                                     HRX_EF_AMDGPU_FEATURE_SRAMECC_ANY_V4,
                                     HRX_EF_AMDGPU_FEATURE_SRAMECC_OFF_V4,
                                     HRX_EF_AMDGPU_FEATURE_SRAMECC_ON_V4);
    xnack = hrx_amdgpu_decode_v4_feature(
        h.flags, HRX_EF_AMDGPU_FEATURE_XNACK_V4,
        HRX_EF_AMDGPU_FEATURE_XNACK_ANY_V4, HRX_EF_AMDGPU_FEATURE_XNACK_OFF_V4,
        HRX_EF_AMDGPU_FEATURE_XNACK_ON_V4);
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported AMDGPU HSA code object ABI version %u",
                            h.abiversion);
  }

  const bool is_generic =
      hrx_amdgpu_processor_is_generic(machine_target->processor);
  const uint32_t generic_version = (h.flags & HRX_EF_AMDGPU_GENERIC_VERSION) >>
                                   HRX_EF_AMDGPU_GENERIC_VERSION_OFFSET;
  if (is_generic && h.abiversion != HRX_ELF_HSA_ABI_VERSION_V6) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "generic AMDGPU code object target requires HSA ABI v6");
  }
  if (is_generic && generic_version == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "generic AMDGPU code object target has no generic version");
  }
  if (!is_generic && generic_version != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "non-generic AMDGPU code object target has generic version %u",
        generic_version);
  }

  IREE_RETURN_IF_ERROR(hrx_fat_format_amdgpu_executable_format(
      machine_target, sramecc, xnack, executable_format_capacity,
      executable_format));

  if (h.shoff > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF section table offset is out of range");
  }
  iree_host_size_t section_table_size = 0;
  iree_host_size_t size = 0;
  if (!iree_host_size_checked_mul((iree_host_size_t)h.shentsize,
                                  (iree_host_size_t)h.shnum,
                                  &section_table_size) ||
      !iree_host_size_checked_add((iree_host_size_t)h.shoff, section_table_size,
                                  &size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "ELF section table size overflow");
  }
  if (elf.data_length != 0 && size > elf.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF claims size %" PRIhsz " but only %" PRIhsz
                            " bytes available",
                            size, elf.data_length);
  }
  if (out_size) *out_size = size;
  return iree_ok_status();
}

// Parses a CCOB header and returns details needed to decompress. Handles
// the three on-disk header flavours (v1: size inferred from buffer, v2:
// 32-bit sizes, v3+: 64-bit sizes). When |data.data_length| is 0 we skip
// length checks (see hrx_fat_length_at_least) and report the total on-disk
// file size via |out_file_size| when the header provides one (v2 and v3 do,
// v1 does not and returns 0).
static iree_status_t hrx_fat_parse_ccob(iree_const_byte_span_t data,
                                        uint16_t* out_version,
                                        uint16_t* out_method,
                                        uint64_t* out_uncompressed_size,
                                        uint64_t* out_file_size,
                                        iree_host_size_t* out_payload_offset) {
  if (out_file_size) *out_file_size = 0;
  if (!hrx_fat_length_at_least(data, sizeof(hrx_ccob_header_v1_t))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CCOB header truncated");
  }
  uint16_t version;
  memcpy(&version, data.data + 4, sizeof(version));
  *out_version = version;
  if (version == 1) {
    hrx_ccob_header_v1_t h;
    memcpy(&h, data.data, sizeof(h));
    *out_method = h.method;
    *out_uncompressed_size = h.uncompressed_size;
    *out_payload_offset = sizeof(hrx_ccob_header_v1_t);
  } else if (version == 2) {
    if (!hrx_fat_length_at_least(data, sizeof(hrx_ccob_header_v2_t))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "CCOB v2 header truncated");
    }
    hrx_ccob_header_v2_t h;
    memcpy(&h, data.data, sizeof(h));
    *out_method = h.method;
    *out_uncompressed_size = h.uncompressed_size;
    *out_payload_offset = sizeof(hrx_ccob_header_v2_t);
    if (out_file_size) *out_file_size = h.file_size;
  } else if (version >= 3) {
    if (!hrx_fat_length_at_least(data, sizeof(hrx_ccob_header_v3_t))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "CCOB v3+ header truncated");
    }
    hrx_ccob_header_v3_t h;
    memcpy(&h, data.data, sizeof(h));
    *out_method = h.method;
    *out_uncompressed_size = h.uncompressed_size;
    *out_payload_offset = sizeof(hrx_ccob_header_v3_t);
    if (out_file_size) *out_file_size = h.file_size;
  } else {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "unsupported CCOB version %u", version);
  }
  if (*out_method != HRX_CCOB_METHOD_ZLIB &&
      *out_method != HRX_CCOB_METHOD_ZSTD) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown CCOB compression method %u", *out_method);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Triple matching
//===----------------------------------------------------------------------===//

// A bundle triple is produced by clang/HIP and looks like:
//   host-x86_64-unknown-linux-gnu
//   hipv4-amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-
//   openmp-amdgcn-amd-amdhsa--gfx1100
//   gfx11-generic
// We take whatever trails the last "--" (or lacking that, the trailing
// bare gfx target or final "-" token) as the target arch component, strip
// any feature-suffix (":sramecc+", ":xnack-", ...) and compare
// case-sensitive against the (similarly stripped) candidate target.

static iree_string_view_t hrx_fat_strip_feature_suffix(iree_string_view_t sv) {
  // Features always start at the first ':'.
  for (iree_host_size_t i = 0; i < sv.size; ++i) {
    if (sv.data[i] == ':') return iree_make_string_view(sv.data, i);
  }
  return sv;
}

static iree_string_view_t hrx_fat_triple_target(iree_string_view_t triple) {
  // Prefer the tail after the final "--" if one exists (canonical
  // clang-offload form for GPU triples).
  if (triple.size >= 2) {
    for (iree_host_size_t i = triple.size - 1; i > 0; --i) {
      if (triple.data[i] == '-' && triple.data[i - 1] == '-') {
        return iree_make_string_view(triple.data + i + 1,
                                     triple.size - (i + 1));
      }
    }
  }
  if (triple.size >= 3 && memcmp(triple.data, "gfx", 3) == 0) {
    return triple;
  }
  // Fall back to the last '-'-delimited token.
  for (iree_host_size_t i = triple.size; i > 0; --i) {
    if (triple.data[i - 1] == '-') {
      return iree_make_string_view(triple.data + i, triple.size - i);
    }
  }
  return triple;
}

// Returns true if |triple| targets |target_value| (base gfx name, without
// feature suffixes). Host entries and non-gfx triples return false.
static bool hrx_fat_triple_matches(iree_string_view_t triple,
                                   iree_string_view_t target_value) {
  // Drop host entries entirely.
  static const char kHostPrefix[] = "host-";
  if (triple.size >= sizeof(kHostPrefix) - 1 &&
      memcmp(triple.data, kHostPrefix, sizeof(kHostPrefix) - 1) == 0) {
    return false;
  }
  iree_string_view_t target = hrx_fat_triple_target(triple);
  target = hrx_fat_strip_feature_suffix(target);
  // Only consider gfx-flavoured targets — the streaming layer exclusively
  // feeds AMDGPU today.
  if (target.size < 3 || memcmp(target.data, "gfx", 3) != 0) return false;
  return iree_string_view_equal(target,
                                hrx_fat_strip_feature_suffix(target_value));
}

static bool hrx_fat_triple_matches_any_target(
    iree_string_view_t triple, iree_host_size_t target_count,
    const iree_hal_streaming_fat_binary_target_t* targets,
    iree_host_size_t* out_target_index) {
  for (iree_host_size_t i = 0; i < target_count; ++i) {
    if (iree_string_view_is_empty(targets[i].value)) continue;
    if (hrx_fat_triple_matches(triple, targets[i].value)) {
      *out_target_index = i;
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Match collection
//===----------------------------------------------------------------------===//

static void hrx_fat_extract_clear_matches(
    iree_hal_streaming_fat_binary_extract_t* extract) {
  if (extract->matches) {
    iree_allocator_free(extract->host_allocator, extract->matches);
  }
  extract->matches = NULL;
  extract->match_count = 0;
  extract->match_capacity = 0;
}

static iree_status_t hrx_fat_extract_push(
    iree_hal_streaming_fat_binary_extract_t* extract,
    iree_const_byte_span_t elf_data, iree_string_view_t triple,
    const char* executable_format) {
  if (extract->match_count == extract->match_capacity) {
    iree_host_size_t new_capacity =
        extract->match_capacity ? extract->match_capacity * 2 : 4;
    void* new_matches = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        extract->host_allocator, new_capacity * sizeof(*extract->matches),
        &new_matches));
    if (extract->matches) {
      memcpy(new_matches, extract->matches,
             extract->match_count * sizeof(*extract->matches));
      iree_allocator_free(extract->host_allocator, extract->matches);
    }
    extract->matches = (iree_hal_streaming_fat_binary_elf_t*)new_matches;
    extract->match_capacity = new_capacity;
  }
  extract->matches[extract->match_count].data = elf_data;
  extract->matches[extract->match_count].triple = triple;
  memcpy(extract->matches[extract->match_count].executable_format,
         executable_format, HRX_FAT_EXECUTABLE_FORMAT_CAPACITY);
  extract->match_count++;
  return iree_ok_status();
}

// Walks an uncompressed __CLANG_OFFLOAD_BUNDLE__ and appends every entry
// whose triple matches the best-ranked target into |extract|. Entry payloads
// must be valid AMDGPU ELFs; other entries (host, cpu, ...) are simply skipped.
static iree_status_t hrx_fat_extract_from_bundle(
    iree_const_byte_span_t bundle, iree_host_size_t target_count,
    const iree_hal_streaming_fat_binary_target_t* targets,
    iree_hal_streaming_fat_binary_extract_t* extract) {
  const uint8_t* p = bundle.data + HRX_OFFLOAD_BUNDLE_MAGIC_SIZE;
  const bool is_unbounded = bundle.data_length == 0;
  iree_host_size_t remaining =
      is_unbounded ? IREE_HOST_SIZE_MAX
                   : bundle.data_length - HRX_OFFLOAD_BUNDLE_MAGIC_SIZE;
  if (remaining < sizeof(uint64_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "offload bundle truncated before entry count");
  }
  uint64_t num_entries;
  memcpy(&num_entries, p, sizeof(num_entries));
  p += sizeof(num_entries);
  remaining -= sizeof(num_entries);

  iree_host_size_t selected_target_index = IREE_HOST_SIZE_MAX;
  for (uint64_t i = 0; i < num_entries; ++i) {
    if (remaining < sizeof(hrx_bundle_entry_t)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "offload bundle entry[%" PRIu64 "] truncated", i);
    }
    hrx_bundle_entry_t entry;
    memcpy(&entry, p, sizeof(entry));
    p += sizeof(entry);
    remaining -= sizeof(entry);

    if (remaining < entry.triple_size) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "offload bundle entry[%" PRIu64 "] triple truncated", i);
    }
    iree_string_view_t triple =
        iree_make_string_view((const char*)p, entry.triple_size);
    p += entry.triple_size;
    remaining -= entry.triple_size;

    if (!is_unbounded && (entry.offset > bundle.data_length ||
                          entry.size > bundle.data_length - entry.offset)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "offload bundle entry[%" PRIu64 "] out of range",
                              i);
    }
    iree_host_size_t target_index = IREE_HOST_SIZE_MAX;
    if (!hrx_fat_triple_matches_any_target(triple, target_count, targets,
                                           &target_index)) {
      continue;
    }
    if (target_index > selected_target_index) continue;
    if (target_index < selected_target_index) {
      hrx_fat_extract_clear_matches(extract);
      selected_target_index = target_index;
    }

    iree_const_byte_span_t entry_bytes =
        iree_make_const_byte_span(bundle.data + (iree_host_size_t)entry.offset,
                                  (iree_host_size_t)entry.size);
    if (!hrx_fat_is_elf(entry_bytes)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "offload bundle entry[%" PRIu64
                              "] (triple '%.*s') is not an AMDGPU ELF",
                              i, (int)triple.size, triple.data);
    }
    char executable_format[HRX_FAT_EXECUTABLE_FORMAT_CAPACITY] = {0};
    iree_host_size_t elf_size = 0;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_fat_binary_describe_amdgpu_elf(
        entry_bytes, sizeof(executable_format), executable_format, &elf_size));
    iree_const_byte_span_t tight_elf =
        iree_make_const_byte_span(entry_bytes.data, elf_size);
    IREE_RETURN_IF_ERROR(
        hrx_fat_extract_push(extract, tight_elf, triple, executable_format));
  }
  return iree_ok_status();
}

// Some ROCm libraries (notably Tensile payloads loaded by hipBLASLt) store
// multiple HSACO ELFs back-to-back in a decompressed CCOB instead of wrapping
// them in a Clang offload bundle. Preserve every ELF so hipModuleGetFunction
// can find kernels that live after the first image.
static iree_status_t hrx_fat_extract_concatenated_elves(
    iree_const_byte_span_t data,
    iree_hal_streaming_fat_binary_extract_t* extract) {
  iree_host_size_t offset = 0;
  const iree_host_size_t initial_match_count = extract->match_count;
  while (offset + 4 <= data.data_length) {
    iree_const_byte_span_t remaining = iree_make_const_byte_span(
        data.data + offset, data.data_length - offset);
    if (!hrx_fat_is_elf(remaining)) {
      // Allow zero padding between images.
      if (data.data[offset] == 0) {
        ++offset;
        continue;
      }
      break;
    }
    char executable_format[HRX_FAT_EXECUTABLE_FORMAT_CAPACITY] = {0};
    iree_host_size_t elf_size = 0;
    IREE_RETURN_IF_ERROR(iree_hal_streaming_fat_binary_describe_amdgpu_elf(
        remaining, sizeof(executable_format), executable_format, &elf_size));
    if (elf_size == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "ELF at offset %" PRIhsz " has zero size",
                              offset);
    }
    IREE_RETURN_IF_ERROR(hrx_fat_extract_push(
        extract, iree_make_const_byte_span(remaining.data, elf_size),
        iree_string_view_empty(), executable_format));
    offset += elf_size;
  }
  return extract->match_count > initial_match_count
             ? iree_ok_status()
             : iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "no ELF images found in concatenated payload");
}

// Decompresses a CCOB payload into extract->owned_buffer using zstd, then
// parses the result as either a raw ELF or an offload bundle and collects
// matching ELFs.
static iree_status_t hrx_fat_extract_from_ccob(
    iree_const_byte_span_t ccob, iree_host_size_t target_count,
    const iree_hal_streaming_fat_binary_target_t* targets,
    iree_hal_streaming_fat_binary_extract_t* extract) {
  uint16_t version = 0, method = 0;
  uint64_t uncompressed_size = 0;
  uint64_t file_size = 0;
  iree_host_size_t payload_offset = 0;
  IREE_RETURN_IF_ERROR(hrx_fat_parse_ccob(ccob, &version, &method,
                                          &uncompressed_size, &file_size,
                                          &payload_offset));
  // If the caller passed an unbounded span (e.g. __hipRegisterFatBinary
  // hands us a raw pointer with no ABI length), use the file_size reported
  // by the CCOB header so we can safely bound the compressed payload.
  if (ccob.data_length == 0 && file_size > 0) {
    ccob = iree_make_const_byte_span(ccob.data, (iree_host_size_t)file_size);
  }
  if (method == HRX_CCOB_METHOD_ZLIB) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "CCOB v%u zlib compression is not supported "
                            "(%" PRIu64
                            " bytes uncompressed); recompile with "
                            "zstd or provide an uncompressed bundle",
                            version, uncompressed_size);
  }

#if !defined(HRX_ENABLE_ZSTD)
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "CCOB v%u zstd compression requires building with "
                          "HRX_ENABLE_ZSTD (uncompressed size %" PRIu64 ")",
                          version, uncompressed_size);
#else
  if (payload_offset > ccob.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CCOB payload offset beyond data end");
  }
  iree_const_byte_span_t payload = iree_make_const_byte_span(
      ccob.data + payload_offset, ccob.data_length - payload_offset);

  // Allocate the decompression target upfront; the header tells us the
  // exact size so a single shot is enough.
  void* out_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(extract->host_allocator,
                            (iree_host_size_t)uncompressed_size, &out_buffer));
  extract->owned_buffer = out_buffer;
  extract->owned_buffer_size = (iree_host_size_t)uncompressed_size;

  size_t actual = ZSTD_decompress(out_buffer, (size_t)uncompressed_size,
                                  payload.data, payload.data_length);
  if (ZSTD_isError(actual)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "zstd decompression failed: %s",
                            ZSTD_getErrorName(actual));
  }
  if (actual != (size_t)uncompressed_size) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "zstd size mismatch: expected %" PRIu64 " got %zu",
                            uncompressed_size, actual);
  }

  iree_const_byte_span_t decompressed =
      iree_make_const_byte_span(out_buffer, (iree_host_size_t)actual);
  if (hrx_fat_is_elf(decompressed)) {
    return hrx_fat_extract_concatenated_elves(decompressed, extract);
  }
  if (hrx_fat_is_uncompressed_bundle(decompressed)) {
    return hrx_fat_extract_from_bundle(decompressed, target_count, targets,
                                       extract);
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "CCOB decompressed payload is not an ELF or Clang offload bundle");
#endif  // HRX_ENABLE_ZSTD
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

void iree_hal_streaming_fat_binary_extract_reset(
    iree_hal_streaming_fat_binary_extract_t* extract) {
  if (!extract) return;
  if (extract->owned_buffer) {
    iree_allocator_free(extract->host_allocator, extract->owned_buffer);
  }
  if (extract->matches) {
    iree_allocator_free(extract->host_allocator, extract->matches);
  }
  memset(extract, 0, sizeof(*extract));
}

iree_status_t iree_hal_streaming_fat_binary_extract_for_targets(
    iree_const_byte_span_t data, iree_host_size_t target_count,
    const iree_hal_streaming_fat_binary_target_t* targets,
    iree_allocator_t host_allocator,
    iree_hal_streaming_fat_binary_extract_t* out_extract) {
  IREE_ASSERT_ARGUMENT(out_extract);
  memset(out_extract, 0, sizeof(*out_extract));
  out_extract->host_allocator = host_allocator;

  if (!target_count || !targets) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at least one fat-binary target is required");
  }

  // Peel the HIP fat-binary wrapper if present.
  iree_const_byte_span_t inner = data;
  if (hrx_fat_is_wrapper(data)) {
    hrx_hip_fat_binary_header_t header;
    memcpy(&header, data.data, sizeof(header));
    if (header.version != HRX_HIP_FAT_VERSION) {
      return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                              "HIP fat-binary version %u not supported",
                              header.version);
    }
    if (!header.binary) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HIP fat-binary wrapper has NULL binary pointer");
    }
    // The wrapper points at an absolute address — it does NOT have to lie
    // inside |data|. HIP embeds a pointer to the bundle in a separate
    // section of the host object at link-time.
    inner = iree_make_const_byte_span(header.binary, 0);
  }

  iree_status_t status = iree_ok_status();
  if (hrx_fat_is_ccob(inner)) {
    status =
        hrx_fat_extract_from_ccob(inner, target_count, targets, out_extract);
  } else if (hrx_fat_is_uncompressed_bundle(inner)) {
    status =
        hrx_fat_extract_from_bundle(inner, target_count, targets, out_extract);
  } else if (hrx_fat_is_elf(inner)) {
    char executable_format[HRX_FAT_EXECUTABLE_FORMAT_CAPACITY] = {0};
    iree_host_size_t elf_size = 0;
    status = iree_hal_streaming_fat_binary_describe_amdgpu_elf(
        inner, sizeof(executable_format), executable_format, &elf_size);
    if (iree_status_is_ok(status)) {
      iree_const_byte_span_t tight =
          iree_make_const_byte_span(inner.data, elf_size);
      status = hrx_fat_extract_push(
          out_extract, tight, iree_string_view_empty(), executable_format);
    }
  } else {
    const uint8_t* head =
        inner.data_length >= 4 ? (const uint8_t*)inner.data : NULL;
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unrecognized module binary (expected AMDGPU ELF, Clang offload "
        "bundle, or HIP fat-binary; first 4 bytes: "
        "0x%02x 0x%02x 0x%02x 0x%02x)",
        head ? head[0] : 0, head ? head[1] : 0, head ? head[2] : 0,
        head ? head[3] : 0);
  }

  if (iree_status_is_ok(status) && out_extract->match_count == 0) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "no ELF in fat binary matches any of %" PRIhsz
                              " target candidates (first '%.*s')",
                              target_count, (int)targets[0].value.size,
                              targets[0].value.data);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_fat_binary_extract_reset(out_extract);
  }
  return status;
}
