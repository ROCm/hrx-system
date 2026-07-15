// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "common/kpack.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/io/file_contents.h"

#if defined(HRX_ENABLE_ZSTD)
#include <zstd.h>
#endif

// Fixed bounds. These are generous relative to real kpack metadata (a kernel
// name plus a handful of search paths) and real archive TOCs; exceeding them is
// treated as malformed input rather than silently truncated.
enum {
  KPACK_MP_MAX_DEPTH = 32,      // msgpack nesting guard
  KPACK_MAX_FEATURES = 6,       // ISA feature flags considered for subsetting
  KPACK_PATH_MAX = 4096,        // filesystem path buffer
  KPACK_MAX_LOOKUP_KEY = 1024,  // "<kernel_name>#<co_index>"
  KPACK_MAX_ARCH_CANDIDATES = 64,        // distinct compatible arch strings
  KPACK_MAX_OPEN_ARCHIVES = 64,          // archives opened per resolve
  KPACK_ZSTD_MAX_KERNELS = 1024 * 1024,  // frame-count cap for a zstd blob
};

// Upper bound on an archive's size and on a decompressed code object.
#define KPACK_MAX_FILE_SIZE (2ULL * 1024 * 1024 * 1024)

//===----------------------------------------------------------------------===//
// Debug logging
//===----------------------------------------------------------------------===//

static bool kpack_debug_enabled(void) {
  const char* e = getenv("ROCM_KPACK_DEBUG");
  return e != NULL && e[0] != '\0' && e[0] != '0';
}

#define KPACK_DBG(...)                             \
  do {                                             \
    if (kpack_debug_enabled()) {                   \
      fprintf(stderr, "[HRX kpack] " __VA_ARGS__); \
      fprintf(stderr, "\n");                       \
      fflush(stderr);                              \
    }                                              \
  } while (0)

// True if an environment variable is set to a non-empty, non-"0" value.
static bool kpack_env_flag(const char* name) {
  const char* e = getenv(name);
  return e != NULL && e[0] != '\0' && e[0] != '0';
}

//===----------------------------------------------------------------------===//
// Minimal MessagePack reader
//===----------------------------------------------------------------------===//
// Just enough of https://github.com/msgpack/msgpack/blob/master/spec.md to walk
// the maps/arrays/strings/ints in HIPK metadata and kpack TOCs. MessagePack
// integers are big-endian (unlike the little-endian fixed binary fields in the
// kpack header and zstd blob). No allocation: string payloads borrow from the
// input buffer.

typedef enum {
  KPACK_MP_INVALID = 0,
  KPACK_MP_NIL,
  KPACK_MP_BOOL,
  KPACK_MP_INT,    // signed or unsigned integer
  KPACK_MP_FLOAT,  // f32/f64 (value ignored)
  KPACK_MP_STR,
  KPACK_MP_BIN,
  KPACK_MP_ARRAY,  // |u| = element count; elements follow in the stream
  KPACK_MP_MAP,    // |u| = pair count; key/value pairs follow in the stream
  KPACK_MP_EXT,
} kpack_mp_type_t;

typedef struct {
  kpack_mp_type_t type;
  uint64_t u;           // INT value, or ARRAY/MAP count, or STR/BIN byte length
  const uint8_t* data;  // STR/BIN payload (length |u|)
} kpack_mp_value_t;

typedef struct {
  const uint8_t* p;
  const uint8_t* end;
} kpack_mp_reader_t;

static kpack_mp_reader_t kpack_mp_reader(iree_const_byte_span_t span) {
  kpack_mp_reader_t r = {span.data, span.data + span.data_length};
  return r;
}

// Reads an |n|-byte big-endian unsigned integer, advancing the cursor.
static bool kpack_mp_read_be(kpack_mp_reader_t* r, int n, uint64_t* out) {
  if ((uint64_t)(r->end - r->p) < (uint64_t)n) return false;
  uint64_t v = 0;
  for (int i = 0; i < n; ++i) v = (v << 8) | (uint64_t)r->p[i];
  r->p += n;
  *out = v;
  return true;
}

// Reads one MessagePack value header. STR/BIN/INT/FLOAT/NIL/BOOL/EXT are fully
// consumed; ARRAY/MAP consume only the header (count in |out->u|), leaving the
// elements in the stream for the caller to walk.
static bool kpack_mp_read_value(kpack_mp_reader_t* r, kpack_mp_value_t* out) {
  memset(out, 0, sizeof(*out));
  if (r->p >= r->end) return false;
  uint8_t b = *r->p++;
  uint64_t len = 0;
  if (b <= 0x7f) {  // positive fixint
    out->type = KPACK_MP_INT;
    out->u = b;
    return true;
  }
  if (b >= 0xe0) {  // negative fixint
    out->type = KPACK_MP_INT;
    out->u = (uint64_t)(int64_t)(int8_t)b;
    return true;
  }
  if (b >= 0x80 && b <= 0x8f) {  // fixmap
    out->type = KPACK_MP_MAP;
    out->u = b & 0x0f;
    return true;
  }
  if (b >= 0x90 && b <= 0x9f) {  // fixarray
    out->type = KPACK_MP_ARRAY;
    out->u = b & 0x0f;
    return true;
  }
  if (b >= 0xa0 && b <= 0xbf) {  // fixstr
    len = b & 0x1f;
    if (len > (uint64_t)(r->end - r->p)) return false;
    out->type = KPACK_MP_STR;
    out->u = len;
    out->data = r->p;
    r->p += len;
    return true;
  }
  switch (b) {
    case 0xc0:  // nil
      out->type = KPACK_MP_NIL;
      return true;
    case 0xc2:  // false
    case 0xc3:  // true
      out->type = KPACK_MP_BOOL;
      out->u = (b == 0xc3);
      return true;
    case 0xc4:  // bin8
    case 0xc5:  // bin16
    case 0xc6:  // bin32
      if (!kpack_mp_read_be(r, 1 << (b - 0xc4), &len)) return false;
      if (len > (uint64_t)(r->end - r->p)) return false;
      out->type = KPACK_MP_BIN;
      out->u = len;
      out->data = r->p;
      r->p += len;
      return true;
    case 0xc7:  // ext8
    case 0xc8:  // ext16
    case 0xc9:  // ext32
      if (!kpack_mp_read_be(r, 1 << (b - 0xc7), &len)) return false;
      if (len + 1 > (uint64_t)(r->end - r->p)) return false;  // +1 type byte
      out->type = KPACK_MP_EXT;
      out->u = len;
      r->p += len + 1;
      return true;
    case 0xca:  // float32
      if ((uint64_t)(r->end - r->p) < 4) return false;
      out->type = KPACK_MP_FLOAT;
      r->p += 4;
      return true;
    case 0xcb:  // float64
      if ((uint64_t)(r->end - r->p) < 8) return false;
      out->type = KPACK_MP_FLOAT;
      r->p += 8;
      return true;
    case 0xcc:  // uint8
    case 0xcd:  // uint16
    case 0xce:  // uint32
    case 0xcf:  // uint64
      if (!kpack_mp_read_be(r, 1 << (b - 0xcc), &out->u)) return false;
      out->type = KPACK_MP_INT;
      return true;
    case 0xd0:    // int8
    case 0xd1:    // int16
    case 0xd2:    // int32
    case 0xd3: {  // int64
      int n = 1 << (b - 0xd0);
      uint64_t raw = 0;
      if (!kpack_mp_read_be(r, n, &raw)) return false;
      // Sign-extend from n bytes.
      uint64_t sign_bit = 1ull << (n * 8 - 1);
      if (raw & sign_bit) raw |= ~((sign_bit << 1) - 1);
      out->type = KPACK_MP_INT;
      out->u = raw;
      return true;
    }
    case 0xd4:  // fixext1
    case 0xd5:  // fixext2
    case 0xd6:  // fixext4
    case 0xd7:  // fixext8
    case 0xd8:  // fixext16
      len = (uint64_t)1 << (b - 0xd4);
      if (len + 1 > (uint64_t)(r->end - r->p)) return false;  // +1 type byte
      out->type = KPACK_MP_EXT;
      out->u = len;
      r->p += len + 1;
      return true;
    case 0xd9:  // str8
    case 0xda:  // str16
    case 0xdb:  // str32
      if (!kpack_mp_read_be(r, 1 << (b - 0xd9), &len)) return false;
      if (len > (uint64_t)(r->end - r->p)) return false;
      out->type = KPACK_MP_STR;
      out->u = len;
      out->data = r->p;
      r->p += len;
      return true;
    case 0xdc:  // array16
    case 0xdd:  // array32
      if (!kpack_mp_read_be(r, 2 << (b - 0xdc), &out->u)) return false;
      out->type = KPACK_MP_ARRAY;
      return true;
    case 0xde:  // map16
    case 0xdf:  // map32
      if (!kpack_mp_read_be(r, 2 << (b - 0xde), &out->u)) return false;
      out->type = KPACK_MP_MAP;
      return true;
    default:  // 0xc1 (never used)
      out->type = KPACK_MP_INVALID;
      return false;
  }
}

// Advances the cursor past one complete value, recursing into containers.
static bool kpack_mp_skip_value(kpack_mp_reader_t* r, int depth) {
  if (depth <= 0) return false;
  kpack_mp_value_t v;
  if (!kpack_mp_read_value(r, &v)) return false;
  if (v.type == KPACK_MP_ARRAY) {
    for (uint64_t i = 0; i < v.u; ++i) {
      if (!kpack_mp_skip_value(r, depth - 1)) return false;
    }
  } else if (v.type == KPACK_MP_MAP) {
    for (uint64_t i = 0; i < v.u; ++i) {
      if (!kpack_mp_skip_value(r, depth - 1)) return false;  // key
      if (!kpack_mp_skip_value(r, depth - 1)) return false;  // value
    }
  } else if (v.type == KPACK_MP_INVALID) {
    return false;
  }
  return true;
}

// Captures the byte span of the next complete value and advances past it.
static bool kpack_mp_capture(kpack_mp_reader_t* r,
                             iree_const_byte_span_t* out) {
  const uint8_t* start = r->p;
  if (!kpack_mp_skip_value(r, KPACK_MP_MAX_DEPTH)) return false;
  *out = iree_make_const_byte_span(start, (iree_host_size_t)(r->p - start));
  return true;
}

// Finds the value for string |key| in the MessagePack map at the head of
// |map_bytes|, returning its byte span. Returns false if |map_bytes| is not a
// map or the key is absent.
static bool kpack_mp_map_find(iree_const_byte_span_t map_bytes,
                              iree_string_view_t key,
                              iree_const_byte_span_t* out_value) {
  kpack_mp_reader_t r = kpack_mp_reader(map_bytes);
  kpack_mp_value_t m;
  if (!kpack_mp_read_value(&r, &m) || m.type != KPACK_MP_MAP) return false;
  for (uint64_t i = 0; i < m.u; ++i) {
    // Capture the key as a complete value so a container-typed key (which
    // read_value only consumes the header of) cannot desync the key/value
    // pairing for the rest of the map, then capture the value.
    iree_const_byte_span_t key_span;
    if (!kpack_mp_capture(&r, &key_span)) return false;
    iree_const_byte_span_t value;
    if (!kpack_mp_capture(&r, &value)) return false;
    // Valid kpack/HIPK maps always use string keys; ignore anything else.
    kpack_mp_reader_t kr = kpack_mp_reader(key_span);
    kpack_mp_value_t k;
    if (kpack_mp_read_value(&kr, &k) && k.type == KPACK_MP_STR &&
        iree_string_view_equal(
            iree_make_string_view((const char*)k.data, (iree_host_size_t)k.u),
            key)) {
      *out_value = value;
      return true;
    }
  }
  return false;
}

// Returns the byte span of array element |index| in the MessagePack array at
// the head of |array_bytes|.
static bool kpack_mp_array_at(iree_const_byte_span_t array_bytes,
                              uint64_t index,
                              iree_const_byte_span_t* out_value) {
  kpack_mp_reader_t r = kpack_mp_reader(array_bytes);
  kpack_mp_value_t a;
  if (!kpack_mp_read_value(&r, &a) || a.type != KPACK_MP_ARRAY) return false;
  if (index >= a.u) return false;
  for (uint64_t i = 0; i < index; ++i) {
    if (!kpack_mp_skip_value(&r, KPACK_MP_MAX_DEPTH)) return false;
  }
  return kpack_mp_capture(&r, out_value);
}

static bool kpack_mp_as_str(iree_const_byte_span_t span,
                            iree_string_view_t* out) {
  kpack_mp_reader_t r = kpack_mp_reader(span);
  kpack_mp_value_t v;
  if (!kpack_mp_read_value(&r, &v) || v.type != KPACK_MP_STR) return false;
  *out = iree_make_string_view((const char*)v.data, (iree_host_size_t)v.u);
  return true;
}

static bool kpack_mp_as_u64(iree_const_byte_span_t span, uint64_t* out) {
  kpack_mp_reader_t r = kpack_mp_reader(span);
  kpack_mp_value_t v;
  if (!kpack_mp_read_value(&r, &v) || v.type != KPACK_MP_INT) return false;
  *out = v.u;
  return true;
}

//===----------------------------------------------------------------------===//
// ISA target matching
//===----------------------------------------------------------------------===//

iree_string_view_t iree_hal_streaming_kpack_strip_target_prefix(
    iree_string_view_t isa) {
  static const char kPrefix[] = "amdgcn-amd-amdhsa--";
  const iree_host_size_t plen = sizeof(kPrefix) - 1;
  if (isa.size > plen && memcmp(isa.data, kPrefix, plen) == 0) {
    return iree_make_string_view(isa.data + plen, isa.size - plen);
  }
  return isa;
}

bool iree_hal_streaming_kpack_for_each_compatible_target(
    iree_string_view_t agent_isa,
    iree_hal_streaming_kpack_target_callback_t callback, void* user_data) {
  iree_string_view_t isa =
      iree_hal_streaming_kpack_strip_target_prefix(agent_isa);
  if (isa.size == 0) return false;

  // Split "<processor>[:<feature>]*" on ':'.
  iree_string_view_t processor = isa;
  iree_string_view_t features[KPACK_MAX_FEATURES];
  iree_host_size_t feature_count = 0;
  for (iree_host_size_t i = 0; i < isa.size; ++i) {
    if (isa.data[i] == ':') {
      processor = iree_make_string_view(isa.data, i);
      // Remaining tokens are features.
      iree_host_size_t start = i + 1;
      for (iree_host_size_t j = start; j <= isa.size; ++j) {
        if (j == isa.size || isa.data[j] == ':') {
          if (j > start && feature_count < KPACK_MAX_FEATURES) {
            features[feature_count++] =
                iree_make_string_view(isa.data + start, j - start);
          }
          start = j + 1;
        }
      }
      break;
    }
  }
  if (processor.size == 0) return false;

  const iree_host_size_t n = feature_count;
  if (n == 0) {
    char buf[IREE_HAL_STREAMING_KPACK_TARGET_CAPACITY];
    if (processor.size >= sizeof(buf)) return false;
    memcpy(buf, processor.data, processor.size);
    return callback(iree_make_string_view(buf, processor.size), user_data);
  }

  // Power set of features, descending mask (most specific first). Bit
  // (n-1-i) selects features[i], so dropping from the high bits preserves the
  // original feature order in each candidate.
  const uint32_t full_mask = (1u << n) - 1;
  for (uint32_t mask = full_mask;; --mask) {
    char buf[IREE_HAL_STREAMING_KPACK_TARGET_CAPACITY];
    iree_host_size_t len = 0;
    bool overflow = processor.size >= sizeof(buf);
    if (!overflow) {
      memcpy(buf, processor.data, processor.size);
      len = processor.size;
      for (iree_host_size_t i = 0; i < n; ++i) {
        if (mask & (1u << (n - 1 - i))) {
          if (len + 1 + features[i].size >= sizeof(buf)) {
            overflow = true;
            break;
          }
          buf[len++] = ':';
          memcpy(buf + len, features[i].data, features[i].size);
          len += features[i].size;
        }
      }
    }
    if (!overflow) {
      if (callback(iree_make_string_view(buf, len), user_data)) return true;
    }
    if (mask == 0) break;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// HIPK metadata
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_kpack_parse_metadata(
    iree_const_byte_span_t data,
    iree_hal_streaming_kpack_metadata_t* out_metadata) {
  IREE_ASSERT_ARGUMENT(out_metadata);
  memset(out_metadata, 0, sizeof(*out_metadata));
  if (!data.data || data.data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty HIPK metadata");
  }

  iree_const_byte_span_t value;
  if (!kpack_mp_map_find(data, IREE_SV("kernel_name"), &value) ||
      !kpack_mp_as_str(value, &out_metadata->kernel_name)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HIPK metadata missing string 'kernel_name' (not valid msgpack or "
        "wrong shape)");
  }

  if (!kpack_mp_map_find(data, IREE_SV("kpack_search_paths"), &value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIPK metadata missing 'kpack_search_paths'");
  }
  kpack_mp_reader_t r = kpack_mp_reader(value);
  kpack_mp_value_t arr;
  if (!kpack_mp_read_value(&r, &arr) || arr.type != KPACK_MP_ARRAY) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HIPK metadata 'kpack_search_paths' is not an array");
  }
  for (uint64_t i = 0; i < arr.u; ++i) {
    iree_const_byte_span_t elem;
    if (!kpack_mp_capture(&r, &elem)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HIPK metadata search path array is truncated");
    }
    iree_string_view_t path;
    if (!kpack_mp_as_str(elem, &path)) continue;  // skip non-string entries
    if (out_metadata->search_path_count >=
        IREE_HAL_STREAMING_KPACK_MAX_SEARCH_PATHS) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HIPK metadata has more than %d search paths",
                              IREE_HAL_STREAMING_KPACK_MAX_SEARCH_PATHS);
    }
    out_metadata->search_paths[out_metadata->search_path_count++] = path;
  }

  if (out_metadata->search_path_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIPK metadata has no kpack search paths");
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Archive parsing
//===----------------------------------------------------------------------===//

#define KPACK_ARCHIVE_MAGIC "KPAK"
#define KPACK_ARCHIVE_MAGIC_SIZE 4
#define KPACK_ARCHIVE_HEADER_SIZE 16
#define KPACK_ARCHIVE_VERSION 1

iree_status_t iree_hal_streaming_kpack_archive_open(
    iree_const_byte_span_t archive_bytes,
    iree_hal_streaming_kpack_archive_t* out_archive) {
  IREE_ASSERT_ARGUMENT(out_archive);
  memset(out_archive, 0, sizeof(*out_archive));
  out_archive->archive = archive_bytes;

  if (archive_bytes.data_length < KPACK_ARCHIVE_HEADER_SIZE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kpack archive truncated (%" PRIhsz " bytes, need at least %d)",
        archive_bytes.data_length, KPACK_ARCHIVE_HEADER_SIZE);
  }
  const uint8_t* h = archive_bytes.data;
  if (memcmp(h, KPACK_ARCHIVE_MAGIC, KPACK_ARCHIVE_MAGIC_SIZE) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "not a kpack archive (bad magic 0x%02x 0x%02x 0x%02x 0x%02x)", h[0],
        h[1], h[2], h[3]);
  }
  uint32_t version;
  memcpy(&version, h + 4, sizeof(version));  // little-endian
  if (version != KPACK_ARCHIVE_VERSION) {
    return iree_make_status(IREE_STATUS_INCOMPATIBLE,
                            "unsupported kpack archive version %u", version);
  }
  uint64_t toc_offset;
  memcpy(&toc_offset, h + 8, sizeof(toc_offset));  // little-endian
  if (toc_offset < KPACK_ARCHIVE_HEADER_SIZE ||
      toc_offset >= archive_bytes.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kpack TOC offset %" PRIu64
                            " out of range (file %" PRIhsz " bytes)",
                            toc_offset, archive_bytes.data_length);
  }
  out_archive->version = version;
  out_archive->toc_map = iree_make_const_byte_span(
      h + toc_offset, archive_bytes.data_length - (iree_host_size_t)toc_offset);

  // The TOC is a top-level msgpack map.
  kpack_mp_reader_t r = kpack_mp_reader(out_archive->toc_map);
  kpack_mp_value_t toc;
  if (!kpack_mp_read_value(&r, &toc) || toc.type != KPACK_MP_MAP) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kpack TOC is not a msgpack map");
  }

  // Compression scheme (default "none").
  out_archive->compression = IREE_HAL_STREAMING_KPACK_COMPRESSION_NONE;
  iree_const_byte_span_t value;
  if (kpack_mp_map_find(out_archive->toc_map, IREE_SV("compression_scheme"),
                        &value)) {
    iree_string_view_t scheme;
    if (kpack_mp_as_str(value, &scheme)) {
      if (iree_string_view_equal(scheme, IREE_SV("zstd-per-kernel"))) {
        out_archive->compression =
            IREE_HAL_STREAMING_KPACK_COMPRESSION_ZSTD_PER_KERNEL;
      } else if (!iree_string_view_equal(scheme, IREE_SV("none"))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unsupported kpack compression scheme '%.*s'",
                                (int)scheme.size, scheme.data);
      }
    }
  }

  if (out_archive->compression == IREE_HAL_STREAMING_KPACK_COMPRESSION_NONE) {
    // "blobs" array is consulted per-ordinal at get_kernel time.
    if (kpack_mp_map_find(out_archive->toc_map, IREE_SV("blobs"), &value)) {
      out_archive->blobs_array = value;
    }
  } else {
    uint64_t zstd_offset = 0;
    uint64_t zstd_size = 0;
    if (kpack_mp_map_find(out_archive->toc_map, IREE_SV("zstd_offset"),
                          &value)) {
      kpack_mp_as_u64(value, &zstd_offset);
    }
    if (kpack_mp_map_find(out_archive->toc_map, IREE_SV("zstd_size"), &value)) {
      kpack_mp_as_u64(value, &zstd_size);
    }
    if (zstd_offset < KPACK_ARCHIVE_HEADER_SIZE ||
        zstd_offset > archive_bytes.data_length ||
        zstd_size > archive_bytes.data_length - zstd_offset) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "kpack zstd blob [%" PRIu64 ", %" PRIu64
                              ") out of range (file %" PRIhsz " bytes)",
                              zstd_offset, zstd_offset + zstd_size,
                              archive_bytes.data_length);
    }
    out_archive->zstd_blob =
        iree_make_const_byte_span(h + zstd_offset, (iree_host_size_t)zstd_size);
  }
  return iree_ok_status();
}

static iree_status_t kpack_decompress_none(
    const iree_hal_streaming_kpack_archive_t* archive, uint64_t ordinal,
    iree_allocator_t host_allocator, void** out_kernel,
    iree_host_size_t* out_kernel_size) {
  iree_const_byte_span_t blob;
  if (archive->blobs_array.data_length == 0 ||
      !kpack_mp_array_at(archive->blobs_array, ordinal, &blob)) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "kpack blob ordinal %" PRIu64 " out of range",
                            ordinal);
  }
  iree_const_byte_span_t value;
  uint64_t offset = 0;
  uint64_t size = 0;
  if (!kpack_mp_map_find(blob, IREE_SV("offset"), &value) ||
      !kpack_mp_as_u64(value, &offset) ||
      !kpack_mp_map_find(blob, IREE_SV("size"), &value) ||
      !kpack_mp_as_u64(value, &size)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kpack blob entry missing offset/size");
  }
  if (size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kpack blob entry has zero size");
  }
  if (offset < KPACK_ARCHIVE_HEADER_SIZE ||
      offset > archive->archive.data_length ||
      size > archive->archive.data_length - offset) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kpack blob [%" PRIu64 ", %" PRIu64
                            ") out of range (file %" PRIhsz " bytes)",
                            offset, offset + size,
                            archive->archive.data_length);
  }
  void* buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, (iree_host_size_t)size, &buffer));
  memcpy(buffer, archive->archive.data + offset, (iree_host_size_t)size);
  *out_kernel = buffer;
  *out_kernel_size = (iree_host_size_t)size;
  return iree_ok_status();
}

static iree_status_t kpack_decompress_zstd(
    const iree_hal_streaming_kpack_archive_t* archive, uint64_t ordinal,
    uint64_t original_size, iree_allocator_t host_allocator, void** out_kernel,
    iree_host_size_t* out_kernel_size) {
#if !defined(HRX_ENABLE_ZSTD)
  (void)archive;
  (void)ordinal;
  (void)original_size;
  (void)host_allocator;
  (void)out_kernel;
  (void)out_kernel_size;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "kpack zstd-per-kernel archive requires building HRX with "
      "HRX_ENABLE_ZSTD (install libzstd and rebuild)");
#else
  // Bound the trusted sizes before allocating or walking frames: original_size
  // drives the output allocation and num_kernels drives the frame walk, so an
  // absurd value would be an allocation-amplification / graceful-fail DoS
  // shape. The num_kernels cap matches the reference unpacker's MAX_KERNELS;
  // the original_size cap reuses HRX's whole-file KPACK_MAX_FILE_SIZE limit and
  // has no counterpart in the reference, which bounds the compressed blob but
  // not the decompressed size.
  if (original_size > KPACK_MAX_FILE_SIZE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kpack zstd decompressed size %" PRIu64
                            " exceeds limit %" PRIu64,
                            original_size, (uint64_t)KPACK_MAX_FILE_SIZE);
  }

  // Blob layout: [num_kernels: u32][ (frame_size: u32)(zstd_frame) ]*, all
  // little-endian (these are raw binary fields, not msgpack).
  const uint8_t* p = archive->zstd_blob.data;
  const uint8_t* end = p + archive->zstd_blob.data_length;
  if ((uint64_t)(end - p) < sizeof(uint32_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kpack zstd blob truncated before kernel count");
  }
  uint32_t num_kernels;
  memcpy(&num_kernels, p, sizeof(num_kernels));
  p += sizeof(uint32_t);
  if (num_kernels > KPACK_ZSTD_MAX_KERNELS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kpack zstd kernel count %u exceeds limit %d",
                            num_kernels, KPACK_ZSTD_MAX_KERNELS);
  }
  if (ordinal >= num_kernels) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "kpack zstd ordinal %" PRIu64
                            " out of range (%u frames)",
                            ordinal, num_kernels);
  }
  for (uint32_t i = 0; i <= ordinal; ++i) {
    if ((uint64_t)(end - p) < sizeof(uint32_t)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "kpack zstd frame %u header truncated", i);
    }
    uint32_t frame_size;
    memcpy(&frame_size, p, sizeof(frame_size));
    p += sizeof(uint32_t);
    if (frame_size > (uint64_t)(end - p)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "kpack zstd frame %u data truncated", i);
    }
    if (i == ordinal) {
      void* buffer = NULL;
      IREE_RETURN_IF_ERROR(iree_allocator_malloc(
          host_allocator, (iree_host_size_t)original_size, &buffer));
      size_t produced =
          ZSTD_decompress(buffer, (size_t)original_size, p, frame_size);
      if (ZSTD_isError(produced)) {
        iree_allocator_free(host_allocator, buffer);
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "kpack zstd decompression failed: %s",
                                ZSTD_getErrorName(produced));
      }
      if (produced != (size_t)original_size) {
        iree_allocator_free(host_allocator, buffer);
        return iree_make_status(IREE_STATUS_DATA_LOSS,
                                "kpack zstd size mismatch: expected %" PRIu64
                                " got %zu",
                                original_size, produced);
      }
      *out_kernel = buffer;
      *out_kernel_size = (iree_host_size_t)original_size;
      return iree_ok_status();
    }
    p += frame_size;
  }
  // Unreachable: the loop returns at i == ordinal and ordinal < num_kernels is
  // enforced above; kept as a defensive backstop.
  return iree_make_status(IREE_STATUS_INTERNAL, "kpack zstd frame walk failed");
#endif  // HRX_ENABLE_ZSTD
}

iree_status_t iree_hal_streaming_kpack_archive_get_kernel(
    const iree_hal_streaming_kpack_archive_t* archive,
    iree_string_view_t binary_key, iree_string_view_t arch,
    iree_allocator_t host_allocator, void** out_kernel,
    iree_host_size_t* out_kernel_size) {
  IREE_ASSERT_ARGUMENT(archive);
  IREE_ASSERT_ARGUMENT(out_kernel);
  IREE_ASSERT_ARGUMENT(out_kernel_size);
  *out_kernel = NULL;
  *out_kernel_size = 0;

  iree_const_byte_span_t toc;
  if (!kpack_mp_map_find(archive->toc_map, IREE_SV("toc"), &toc)) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "kpack archive has no 'toc'");
  }
  iree_const_byte_span_t binary_node;
  if (!kpack_mp_map_find(toc, binary_key, &binary_node)) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "kpack archive has no entry for '%.*s'",
                            (int)binary_key.size, binary_key.data);
  }
  iree_const_byte_span_t arch_node;
  if (!kpack_mp_map_find(binary_node, arch, &arch_node)) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND, "kpack entry '%.*s' has no architecture '%.*s'",
        (int)binary_key.size, binary_key.data, (int)arch.size, arch.data);
  }

  iree_const_byte_span_t value;
  uint64_t ordinal = 0;
  uint64_t original_size = 0;
  if (!kpack_mp_map_find(arch_node, IREE_SV("ordinal"), &value) ||
      !kpack_mp_as_u64(value, &ordinal)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kpack entry '%.*s'/'%.*s' missing ordinal",
                            (int)binary_key.size, binary_key.data,
                            (int)arch.size, arch.data);
  }
  // original_size is required for zstd; for "none" the blob size is
  // authoritative.
  if (kpack_mp_map_find(arch_node, IREE_SV("original_size"), &value)) {
    kpack_mp_as_u64(value, &original_size);
  }

  if (archive->compression == IREE_HAL_STREAMING_KPACK_COMPRESSION_NONE) {
    return kpack_decompress_none(archive, ordinal, host_allocator, out_kernel,
                                 out_kernel_size);
  }
  if (archive->compression !=
      IREE_HAL_STREAMING_KPACK_COMPRESSION_ZSTD_PER_KERNEL) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "kpack unsupported compression scheme");
  }
  if (original_size == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kpack zstd entry '%.*s'/'%.*s' missing original_size",
        (int)binary_key.size, binary_key.data, (int)arch.size, arch.data);
  }
  return kpack_decompress_zstd(archive, ordinal, original_size, host_allocator,
                               out_kernel, out_kernel_size);
}

//===----------------------------------------------------------------------===//
// Path resolution and discovery
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_kpack_expand_gfxarch(
    iree_string_view_t pattern, iree_string_view_t arch, char* out,
    iree_host_size_t out_capacity, bool* out_had_placeholder) {
  static const char kPlaceholder[] = "@GFXARCH@";
  const iree_host_size_t plen = sizeof(kPlaceholder) - 1;
  if (out_had_placeholder) *out_had_placeholder = false;
  IREE_ASSERT_ARGUMENT(out);
  if (out_capacity == 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "empty output buffer");
  }

  iree_host_size_t pos = IREE_HOST_SIZE_MAX;
  if (pattern.size >= plen) {
    for (iree_host_size_t i = 0; i + plen <= pattern.size; ++i) {
      if (memcmp(pattern.data + i, kPlaceholder, plen) == 0) {
        pos = i;
        break;
      }
    }
  }

  iree_host_size_t total = pos == IREE_HOST_SIZE_MAX
                               ? pattern.size
                               : pattern.size - plen + arch.size;
  if (total + 1 > out_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "expanded kpack path too long (%" PRIhsz
                            " bytes, capacity %" PRIhsz ")",
                            total, out_capacity);
  }
  if (pos == IREE_HOST_SIZE_MAX) {
    memcpy(out, pattern.data, pattern.size);
    out[pattern.size] = '\0';
    return iree_ok_status();
  }
  if (out_had_placeholder) *out_had_placeholder = true;
  iree_host_size_t len = 0;
  memcpy(out + len, pattern.data, pos);
  len += pos;
  memcpy(out + len, arch.data, arch.size);
  len += arch.size;
  memcpy(out + len, pattern.data + pos + plen, pattern.size - pos - plen);
  len += pattern.size - pos - plen;
  out[len] = '\0';
  return iree_ok_status();
}

// Lexically normalizes "." / ".." components of |path| into |out|. Preserves a
// leading '/'. No filesystem access.
static iree_status_t kpack_normalize_path(iree_string_view_t path, char* out,
                                          iree_host_size_t out_capacity) {
  const bool is_absolute = path.size > 0 && path.data[0] == '/';
  iree_host_size_t out_len = 0;
  // Offsets into |out| where each retained, poppable component begins.
  iree_host_size_t comp_start[256];
  iree_host_size_t comp_count = 0;

#define KPACK_PUT(ch)                                            \
  do {                                                           \
    if (out_len + 1 >= out_capacity)                             \
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,          \
                              "normalized kpack path too long"); \
    out[out_len++] = (ch);                                       \
  } while (0)

  if (is_absolute) KPACK_PUT('/');

  iree_host_size_t i = 0;
  while (i < path.size) {
    // Skip consecutive separators.
    if (path.data[i] == '/') {
      ++i;
      continue;
    }
    iree_host_size_t start = i;
    while (i < path.size && path.data[i] != '/') ++i;
    iree_string_view_t comp =
        iree_make_string_view(path.data + start, i - start);
    if (iree_string_view_equal(comp, IREE_SV("."))) {
      continue;
    }
    if (iree_string_view_equal(comp, IREE_SV(".."))) {
      if (comp_count > 0) {
        out_len = comp_start[--comp_count];  // drop last component + its '/'
      } else if (!is_absolute) {
        // Cannot ascend above a relative root; keep the "..".
        KPACK_PUT('.');
        KPACK_PUT('.');
        KPACK_PUT('/');
      }
      // Absolute root: ".." stays at root (ignored).
      continue;
    }
    if (comp_count >= IREE_ARRAYSIZE(comp_start)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "kpack path has too many components");
    }
    comp_start[comp_count++] = out_len;
    for (iree_host_size_t k = 0; k < comp.size; ++k) KPACK_PUT(comp.data[k]);
    KPACK_PUT('/');
  }

  // Strip a single trailing '/' (but keep a lone root "/").
  if (out_len > 1 && out[out_len - 1] == '/') --out_len;
  if (out_len == 0) KPACK_PUT('.');  // fully-cancelled relative path
  out[out_len] = '\0';
#undef KPACK_PUT
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_kpack_resolve_relative_path(
    iree_string_view_t base_path, iree_string_view_t relative, char* out,
    iree_host_size_t out_capacity) {
  IREE_ASSERT_ARGUMENT(out);
  if (relative.size > 0 && relative.data[0] == '/') {
    // Absolute: normalize directly.
    return kpack_normalize_path(relative, out, out_capacity);
  }

  // Directory of base_path (everything up to and excluding the last '/').
  iree_host_size_t last_slash = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = base_path.size; i > 0; --i) {
    if (base_path.data[i - 1] == '/') {
      last_slash = i - 1;
      break;
    }
  }
  iree_string_view_t dir =
      last_slash == IREE_HOST_SIZE_MAX
          ? IREE_SV(".")
          : iree_make_string_view(base_path.data, last_slash);

  // Join dir + '/' + relative into a temporary, then normalize.
  char joined[KPACK_PATH_MAX];
  iree_host_size_t need = dir.size + 1 + relative.size;
  if (need + 1 > sizeof(joined)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "kpack base+relative path too long");
  }
  iree_host_size_t len = 0;
  memcpy(joined + len, dir.data, dir.size);
  len += dir.size;
  joined[len++] = '/';
  memcpy(joined + len, relative.data, relative.size);
  len += relative.size;
  return kpack_normalize_path(iree_make_string_view(joined, len), out,
                              out_capacity);
}

iree_status_t iree_hal_streaming_kpack_discover_binary_path(
    const void* address_in_binary, char* out, iree_host_size_t out_capacity,
    iree_host_size_t* out_offset) {
  if (!address_in_binary || !out || out_capacity == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kpack_discover_binary_path: null argument");
  }
#if defined(__linux__)
  // Resolve the file backing the mapping that contains |address_in_binary| by
  // scanning /proc/self/maps. dladdr() cannot reliably resolve data segments,
  // and the HIPK wrapper points into a data section.
  FILE* maps = fopen("/proc/self/maps", "r");
  if (!maps) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "kpack_discover_binary_path: cannot open "
                            "/proc/self/maps");
  }
  uintptr_t target = (uintptr_t)address_in_binary;
  char line[KPACK_PATH_MAX + 256];
  // Defer building the not-found status until the scan finds no containing
  // mapping. It carries a formatted (heap-allocated) message and the branches
  // below overwrite |status|, so allocating it eagerly here would leak it
  // whenever a mapping is found. Track whether any file mapping matched
  // instead.
  bool matched = false;
  iree_status_t status = iree_ok_status();
  while (fgets(line, sizeof(line), maps)) {
    // Format: "<low>-<high> <perms> <offset> <dev> <inode> <path>".
    char* dash = strchr(line, '-');
    if (!dash) continue;
    char* endp = NULL;
    uintptr_t low = (uintptr_t)strtoull(line, &endp, 16);
    if (endp != dash) continue;
    uintptr_t high = (uintptr_t)strtoull(dash + 1, &endp, 16);
    if (target < low || target >= high) continue;
    matched = true;

    // endp points just past the high address, at the space before perms.
    char* cursor = endp;
    while (*cursor == ' ') ++cursor;
    while (*cursor && *cursor != ' ') ++cursor;  // skip perms
    while (*cursor == ' ') ++cursor;
    uintptr_t file_offset = (uintptr_t)strtoull(cursor, &endp, 16);
    cursor = endp;
    while (*cursor == ' ') ++cursor;
    while (*cursor && *cursor != ' ') ++cursor;  // skip dev
    while (*cursor == ' ') ++cursor;
    while (*cursor && *cursor != ' ') ++cursor;  // skip inode
    while (*cursor == ' ') ++cursor;             // cursor -> path (or EOL)

    // Trim trailing newline/whitespace.
    char* nl = cursor;
    while (*nl && *nl != '\n') ++nl;
    *nl = '\0';
    if (cursor[0] == '\0' || cursor[0] == '[') {
      // Anonymous mapping or special region ([heap], [stack], ...).
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "kpack_discover_binary_path: address %p is in an anonymous mapping",
          address_in_binary);
      break;
    }
    iree_host_size_t path_len = strlen(cursor);
    if (path_len + 1 > out_capacity) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "kpack_discover_binary_path: path too long for "
                                "buffer (%" PRIhsz " bytes)",
                                path_len);
      break;
    }
    memcpy(out, cursor, path_len + 1);
    if (out_offset)
      *out_offset = (iree_host_size_t)(file_offset + (target - low));
    status = iree_ok_status();
    break;
  }
  fclose(maps);
  if (!matched) {
    status = iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "kpack_discover_binary_path: address %p not in any file mapping",
        address_in_binary);
  }
  return status;
#else
  (void)out_offset;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "kpack_discover_binary_path is not implemented on this platform");
#endif  // __linux__
}

//===----------------------------------------------------------------------===//
// Top-level resolution
//===----------------------------------------------------------------------===//

// Memory-maps the archive at |path|. The mapping owns the bytes and the spans
// parsed from it stay valid until it is freed; the file must not change size
// while mapped. Fails if |path| cannot be mapped (caller skips the path) or is
// larger than KPACK_MAX_FILE_SIZE.
static iree_status_t kpack_map_file(const char* path,
                                    iree_allocator_t host_allocator,
                                    iree_io_file_contents_t** out_mapping) {
  *out_mapping = NULL;
  iree_io_file_contents_t* mapping = NULL;
  IREE_RETURN_IF_ERROR(iree_io_file_contents_map(iree_make_cstring_view(path),
                                                 IREE_IO_FILE_ACCESS_READ,
                                                 host_allocator, &mapping));
  iree_host_size_t length = mapping->const_buffer.data_length;
  if (length > KPACK_MAX_FILE_SIZE) {
    iree_io_file_contents_free(mapping);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "kpack archive '%s' too large (%" PRIhsz " bytes)",
                            path, length);
  }
  *out_mapping = mapping;
  return iree_ok_status();
}

// An opened archive plus its canonical path (for dedup). |archive|'s spans
// point into |mapping| and are valid until it is freed.
typedef struct {
  char canonical[KPACK_PATH_MAX];
  iree_io_file_contents_t* mapping;
  iree_hal_streaming_kpack_archive_t archive;
} kpack_open_archive_t;

// Mutable state threaded through resolution.
typedef struct {
  iree_allocator_t host_allocator;
  // Ranked, de-duplicated compatible architecture candidates.
  char arch[KPACK_MAX_ARCH_CANDIDATES]
           [IREE_HAL_STREAMING_KPACK_TARGET_CAPACITY];
  iree_host_size_t arch_count;
  // Opened archives.
  kpack_open_archive_t* open;
  iree_host_size_t open_count;
} kpack_resolve_state_t;

// Appends |target| to the ranked arch candidate list if not already present.
static bool kpack_arch_collect_cb(iree_string_view_t target, void* user_data) {
  kpack_resolve_state_t* state = (kpack_resolve_state_t*)user_data;
  if (target.size == 0 ||
      target.size >= IREE_HAL_STREAMING_KPACK_TARGET_CAPACITY) {
    return false;
  }
  for (iree_host_size_t i = 0; i < state->arch_count; ++i) {
    if (iree_string_view_equal(
            iree_make_string_view(state->arch[i], strlen(state->arch[i])),
            target)) {
      return false;  // already present
    }
  }
  if (state->arch_count >= KPACK_MAX_ARCH_CANDIDATES) return false;
  memcpy(state->arch[state->arch_count], target.data, target.size);
  state->arch[state->arch_count][target.size] = '\0';
  ++state->arch_count;
  return false;  // continue collecting
}

// Maps and parses the archive at |path| into the resolve's open list
// (deduplicated by canonical path). Missing or unparseable archives are skipped
// (logged), not fatal.
static void kpack_try_open_archive(kpack_resolve_state_t* state,
                                   const char* path) {
  // realpath() requires a buffer of at least PATH_MAX bytes; KPACK_PATH_MAX is
  // sized to match (== PATH_MAX on Linux, the only platform that reaches here).
  // realpath() also doubles as the existence check: it returns NULL if |path|
  // cannot be resolved, and yields the canonical path used to dedup archives.
  char canonical[KPACK_PATH_MAX];
  if (!realpath(path, canonical)) {
    KPACK_DBG("archive not found: %s", path);
    return;
  }
  for (iree_host_size_t i = 0; i < state->open_count; ++i) {
    if (strcmp(state->open[i].canonical, canonical) == 0) return;  // dedup
  }
  if (state->open_count >= KPACK_MAX_OPEN_ARCHIVES) {
    KPACK_DBG("too many archives; ignoring %s", canonical);
    return;
  }

  iree_io_file_contents_t* mapping = NULL;
  iree_status_t status =
      kpack_map_file(canonical, state->host_allocator, &mapping);
  if (!iree_status_is_ok(status)) {
    KPACK_DBG("failed to map %s", canonical);
    iree_status_ignore(status);
    return;
  }
  kpack_open_archive_t* slot = &state->open[state->open_count];
  status = iree_hal_streaming_kpack_archive_open(mapping->const_buffer,
                                                 &slot->archive);
  if (!iree_status_is_ok(status)) {
    KPACK_DBG("failed to parse %s", canonical);
    iree_status_ignore(status);
    iree_io_file_contents_free(mapping);
    return;
  }
  slot->mapping = mapping;
  iree_host_size_t clen = strlen(canonical);
  memcpy(slot->canonical, canonical, clen + 1);
  ++state->open_count;
  KPACK_DBG("opened archive %s", canonical);
}

// Opens every archive in a ':'-separated path list (ROCM_KPACK_PATH / _PREFIX).
static void kpack_open_path_list(kpack_resolve_state_t* state,
                                 const char* list_str) {
  iree_string_view_t list = iree_make_cstring_view(list_str);
  while (list.size > 0) {
    iree_string_view_t entry;
    iree_string_view_t rest;
    iree_string_view_split(list, ':', &entry, &rest);
    list = rest;  // empty when no separator remains, terminating the loop
    if (entry.size == 0) continue;
    char path[KPACK_PATH_MAX];
    if (entry.size + 1 > sizeof(path)) continue;
    memcpy(path, entry.data, entry.size);
    path[entry.size] = '\0';
    kpack_try_open_archive(state, path);
  }
}

iree_status_t iree_hal_streaming_kpack_resolve_code_object(
    const void* hipk_metadata, uint32_t co_index,
    iree_host_size_t target_arch_count, const iree_string_view_t* target_archs,
    iree_allocator_t host_allocator, void** out_code_object,
    iree_host_size_t* out_code_object_size) {
  if (!hipk_metadata || target_arch_count == 0 || !target_archs ||
      !out_code_object || !out_code_object_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kpack resolve: invalid argument");
  }
  *out_code_object = NULL;
  *out_code_object_size = 0;

  if (kpack_env_flag("ROCM_KPACK_DISABLE")) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "kpack disabled via ROCM_KPACK_DISABLE");
  }

  // 1. Parse the HIPK metadata blob (self-bounding within the cap).
  iree_hal_streaming_kpack_metadata_t metadata;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_kpack_parse_metadata(
      iree_make_const_byte_span(hipk_metadata,
                                IREE_HAL_STREAMING_KPACK_MAX_METADATA_SIZE),
      &metadata));

  // 2. Lookup key is "<kernel_name>#<co_index>".
  char lookup_key[KPACK_MAX_LOOKUP_KEY];
  int klen = snprintf(lookup_key, sizeof(lookup_key), "%.*s#%u",
                      (int)metadata.kernel_name.size, metadata.kernel_name.data,
                      co_index);
  if (klen < 0 || (size_t)klen >= sizeof(lookup_key)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "kpack lookup key too long");
  }
  iree_string_view_t lookup_key_view = iree_make_string_view(lookup_key, klen);
  KPACK_DBG("resolving lookup_key='%s', %d search paths, %" PRIhsz " targets",
            lookup_key, (int)metadata.search_path_count, target_arch_count);

  // 3. Discover the owning binary so relative search paths resolve. Best
  // effort: absolute paths and ROCM_KPACK_PATH overrides work without it.
  char binary_path[KPACK_PATH_MAX];
  bool have_binary_path = false;
  iree_status_t disc = iree_hal_streaming_kpack_discover_binary_path(
      hipk_metadata, binary_path, sizeof(binary_path), NULL);
  if (iree_status_is_ok(disc)) {
    have_binary_path = true;
    KPACK_DBG("owning binary: %s", binary_path);
  } else {
    iree_status_ignore(disc);
  }
  iree_string_view_t binary_path_view =
      have_binary_path ? iree_make_string_view(binary_path, strlen(binary_path))
                       : iree_string_view_empty();

  // 4. Build the ranked, de-duplicated architecture candidate list across all
  // requested targets (priority order, then feature-subset specificity).
  kpack_resolve_state_t state;
  memset(&state, 0, sizeof(state));
  state.host_allocator = host_allocator;
  for (iree_host_size_t i = 0; i < target_arch_count; ++i) {
    iree_hal_streaming_kpack_for_each_compatible_target(
        target_archs[i], kpack_arch_collect_cb, &state);
  }
  if (state.arch_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kpack resolve: no usable target architectures");
  }

  state.open = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, KPACK_MAX_OPEN_ARCHIVES * sizeof(kpack_open_archive_t),
      (void**)&state.open);
  if (!iree_status_is_ok(status)) return status;

  // 5. Open candidate archives. ROCM_KPACK_PATH overrides the embedded search
  // paths entirely; otherwise ROCM_KPACK_PATH_PREFIX entries are prepended and
  // embedded search paths are resolved relative to the owning binary, expanding
  // "@GFXARCH@" for each architecture candidate.
  const char* env_override = getenv("ROCM_KPACK_PATH");
  const char* env_prefix = getenv("ROCM_KPACK_PATH_PREFIX");

  if (env_override && env_override[0]) {
    kpack_open_path_list(&state, env_override);
  } else {
    if (env_prefix && env_prefix[0]) {
      kpack_open_path_list(&state, env_prefix);
    }
    for (iree_host_size_t i = 0; i < metadata.search_path_count; ++i) {
      iree_string_view_t rel = metadata.search_paths[i];
      bool is_absolute = rel.size > 0 && rel.data[0] == '/';
      if (!is_absolute && !have_binary_path) {
        KPACK_DBG("skipping relative search path with unknown binary: %.*s",
                  (int)rel.size, rel.data);
        continue;
      }
      char resolved[KPACK_PATH_MAX];
      iree_status_t rs = iree_hal_streaming_kpack_resolve_relative_path(
          binary_path_view, rel, resolved, sizeof(resolved));
      if (!iree_status_is_ok(rs)) {
        iree_status_ignore(rs);
        continue;
      }
      iree_string_view_t resolved_view =
          iree_make_string_view(resolved, strlen(resolved));
      bool has_placeholder =
          iree_string_view_find_char(resolved_view, '@', 0) !=
          IREE_STRING_VIEW_NPOS;
      if (!has_placeholder) {
        kpack_try_open_archive(&state, resolved);
        continue;
      }
      // Expand "@GFXARCH@" for each candidate architecture.
      for (iree_host_size_t a = 0; a < state.arch_count; ++a) {
        char expanded[KPACK_PATH_MAX];
        bool had = false;
        iree_status_t es = iree_hal_streaming_kpack_expand_gfxarch(
            resolved_view, iree_make_cstring_view(state.arch[a]), expanded,
            sizeof(expanded), &had);
        if (!iree_status_is_ok(es)) {
          iree_status_ignore(es);
          continue;
        }
        kpack_try_open_archive(&state, expanded);
      }
    }
  }

  // 6. Arch-first search: for each architecture candidate in priority order,
  // probe every opened archive; the first hit wins.
  iree_status_t last_error = iree_ok_status();
  bool found = false;
  for (iree_host_size_t a = 0; a < state.arch_count && !found; ++a) {
    iree_string_view_t arch =
        iree_make_string_view(state.arch[a], strlen(state.arch[a]));
    for (iree_host_size_t o = 0; o < state.open_count; ++o) {
      void* kernel = NULL;
      iree_host_size_t kernel_size = 0;
      iree_status_t ks = iree_hal_streaming_kpack_archive_get_kernel(
          &state.open[o].archive, lookup_key_view, arch, host_allocator,
          &kernel, &kernel_size);
      if (iree_status_is_ok(ks)) {
        *out_code_object = kernel;
        *out_code_object_size = kernel_size;
        found = true;
        KPACK_DBG("matched arch '%s' in %s (%" PRIhsz " bytes)", state.arch[a],
                  state.open[o].canonical, kernel_size);
        break;
      }
      if (iree_status_code(ks) == IREE_STATUS_NOT_FOUND) {
        iree_status_ignore(ks);  // not in this archive — keep looking
      } else {
        // Corrupt/unsupported archive: remember but keep searching others.
        iree_status_ignore(last_error);
        last_error = ks;
      }
    }
  }

  if (!found) {
    if (iree_status_is_ok(last_error)) {
      if (state.open_count == 0) {
        last_error = iree_make_status(
            IREE_STATUS_NOT_FOUND,
            "no kpack archive found for '%s' (%" PRIhsz
            " search paths; set ROCM_KPACK_DEBUG=1 for details)",
            lookup_key, metadata.search_path_count);
      } else {
        last_error = iree_make_status(
            IREE_STATUS_NOT_FOUND,
            "no kpack code object for '%s' matches target '%.*s' in %" PRIhsz
            " archive(s)",
            lookup_key, (int)target_archs[0].size, target_archs[0].data,
            state.open_count);
      }
    }
    status = last_error;
  } else {
    iree_status_ignore(last_error);
  }

  // Unmap the archives; the returned code object is an independent copy.
  for (iree_host_size_t o = 0; o < state.open_count; ++o) {
    iree_io_file_contents_free(state.open[o].mapping);
  }
  iree_allocator_free(host_allocator, state.open);
  return status;
}
