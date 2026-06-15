// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "common/fat_binary.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"

#if defined(HRX_ENABLE_ZSTD)
#include <zstd.h>
#endif

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
#define HRX_EM_AMDGPU 224

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

// Validates an AMDGPU ELF header and computes the total on-disk ELF size
// from the section-header table. The streaming layer only cares that the
// image is the expected flavour; any semantic validation beyond this is
// the HAL's job.
static iree_status_t hrx_fat_validate_elf(iree_const_byte_span_t elf,
                                          iree_host_size_t* out_size) {
  if (!hrx_fat_length_at_least(elf, sizeof(hrx_elf64_header_t))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF data too small (got %" PRIhsz ")",
                            elf.data_length);
  }
  const hrx_elf64_header_t* h = (const hrx_elf64_header_t*)elf.data;
  if (h->elf_class != HRX_ELFCLASS64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF class must be 64-bit, got %u", h->elf_class);
  }
  if (h->elf_data != HRX_ELFDATA2LSB) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF must be little-endian, got %u", h->elf_data);
  }
  if (h->machine != HRX_EM_AMDGPU) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ELF machine must be AMDGPU (%u), got %u",
                            HRX_EM_AMDGPU, h->machine);
  }
  iree_host_size_t size =
      (iree_host_size_t)(h->shoff + h->shentsize * h->shnum);
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
// We take whatever trails the last "--" (or lacking that, the trailing
// token of the triple) as the target arch component, strip any
// feature-suffix (":sramecc+", ":xnack-", ...) and compare case-sensitive
// against the (similarly stripped) |target_arch| passed in.

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
  // Fall back to the last '-'-delimited token.
  for (iree_host_size_t i = triple.size; i > 0; --i) {
    if (triple.data[i - 1] == '-') {
      return iree_make_string_view(triple.data + i, triple.size - i);
    }
  }
  return triple;
}

// Returns true if |triple| targets |target_arch| (base gfx name, without
// feature suffixes). Host entries and non-gfx triples return false.
static bool hrx_fat_triple_matches(iree_string_view_t triple,
                                   iree_string_view_t target_arch) {
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
  return iree_string_view_equal(target, target_arch);
}

//===----------------------------------------------------------------------===//
// Match collection
//===----------------------------------------------------------------------===//

static iree_status_t hrx_fat_extract_push(
    iree_hal_streaming_fat_binary_extract_t* extract,
    iree_const_byte_span_t elf_data, iree_string_view_t triple) {
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
  extract->match_count++;
  return iree_ok_status();
}

// Walks an uncompressed __CLANG_OFFLOAD_BUNDLE__ and appends every entry
// whose triple matches |target_arch| into |extract|. Entry payloads must
// be valid AMDGPU ELFs; other entries (host, cpu, ...) are simply skipped.
static iree_status_t hrx_fat_extract_from_bundle(
    iree_const_byte_span_t bundle, iree_string_view_t target_arch,
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
    if (!hrx_fat_triple_matches(triple, target_arch)) continue;

    iree_const_byte_span_t entry_bytes =
        iree_make_const_byte_span(bundle.data + (iree_host_size_t)entry.offset,
                                  (iree_host_size_t)entry.size);
    if (!hrx_fat_is_elf(entry_bytes)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "offload bundle entry[%" PRIu64
                              "] (triple '%.*s') is not an AMDGPU ELF",
                              i, (int)triple.size, triple.data);
    }
    iree_host_size_t elf_size = 0;
    IREE_RETURN_IF_ERROR(hrx_fat_validate_elf(entry_bytes, &elf_size));
    iree_const_byte_span_t tight_elf =
        iree_make_const_byte_span(entry_bytes.data, elf_size);
    IREE_RETURN_IF_ERROR(hrx_fat_extract_push(extract, tight_elf, triple));
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
    iree_host_size_t elf_size = 0;
    IREE_RETURN_IF_ERROR(hrx_fat_validate_elf(remaining, &elf_size));
    if (elf_size == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "ELF at offset %" PRIhsz " has zero size",
                              offset);
    }
    IREE_RETURN_IF_ERROR(hrx_fat_extract_push(
        extract, iree_make_const_byte_span(remaining.data, elf_size),
        iree_string_view_empty()));
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
    iree_const_byte_span_t ccob, iree_string_view_t target_arch,
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
    return hrx_fat_extract_from_bundle(decompressed, target_arch, extract);
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

iree_status_t iree_hal_streaming_fat_binary_extract_for_target(
    iree_const_byte_span_t data, iree_string_view_t target_arch,
    iree_allocator_t host_allocator,
    iree_hal_streaming_fat_binary_extract_t* out_extract) {
  IREE_ASSERT_ARGUMENT(out_extract);
  memset(out_extract, 0, sizeof(*out_extract));
  out_extract->host_allocator = host_allocator;

  // Normalize the device-supplied arch (e.g. "gfx942:sramecc+:xnack-")
  // to a bare "gfxNNN" for apples-to-apples comparison with bundle triples.
  iree_string_view_t normalized_target =
      hrx_fat_strip_feature_suffix(target_arch);

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
    status = hrx_fat_extract_from_ccob(inner, normalized_target, out_extract);
  } else if (hrx_fat_is_uncompressed_bundle(inner)) {
    status = hrx_fat_extract_from_bundle(inner, normalized_target, out_extract);
  } else if (hrx_fat_is_elf(inner)) {
    iree_host_size_t elf_size = 0;
    status = hrx_fat_validate_elf(inner, &elf_size);
    if (iree_status_is_ok(status)) {
      iree_const_byte_span_t tight =
          iree_make_const_byte_span(inner.data, elf_size);
      status =
          hrx_fat_extract_push(out_extract, tight, iree_string_view_empty());
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
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "no ELF in fat binary matches target '%.*s' (normalized '%.*s')",
        (int)target_arch.size, target_arch.data, (int)normalized_target.size,
        normalized_target.data);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_fat_binary_extract_reset(out_extract);
  }
  return status;
}
