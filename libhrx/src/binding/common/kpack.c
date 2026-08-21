// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "common/kpack.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/io/file_handle.h"

#if defined(IREE_HAVE_ZSTD)
#include <zstd.h>
#endif

// Fixed bounds. These are generous relative to real kpack metadata (a kernel
// name plus a handful of search paths) and real archive TOCs. Exceeding the
// depth, path, lookup-key, feature, or processor-length bound is rejected
// rather than silently truncated. Exceeding the arch-candidate bound keeps the
// highest-ranked candidates that fit and records why the rest were dropped, so
// a resulting miss carries the reason instead of an unexplained not-found.
enum {
  // msgpack nesting guard. The reference kpack runtime parses through a msgpack
  // library that bounds nesting itself; HRX's hand-rolled reader recurses per
  // container level, so it caps depth to fail closed on a crafted archive
  // rather than overflow the stack, set well above the format's handful of
  // levels.
  KPACK_MP_MAX_DEPTH = 32,
  // Subsettable ISA feature flags expanded from one target ID. The AMDGPU
  // target ID subsets on two features (sramecc, xnack), which this keeps
  // headroom above. A target carrying more is rejected: expanding a truncated
  // feature set would silently generate the wrong candidates and miss the
  // archive keyed on the dropped feature.
  KPACK_MAX_FEATURES = 4,
  // Filesystem path buffer. This is the OS PATH_MAX, so a path that does not
  // fit could not be opened regardless of the buffer.
  KPACK_PATH_MAX = 4096,
  // Archive lookup key "<kernel_name>#<co_index>". kernel_name is the owning
  // binary's install-relative path, whose length the format does not bound;
  // real keys run to tens of bytes, so this buffers well beyond them and
  // rejects a key that would not fit.
  KPACK_MAX_LOOKUP_KEY = 1024,
  // Distinct compatible arch strings ranked in one resolve. Every requested
  // target expands into this one list, each contributing up to
  // 2^KPACK_MAX_FEATURES candidates, so the ceiling scales with how many
  // targets the caller asks for rather than with any single one. The list is
  // ranked most-specific-first and searched in that order, so candidates past
  // the cap are the lowest-ranked fallbacks: dropping them can only miss, never
  // select a lower-ranked target over a higher-ranked one. The drop is recorded
  // so a resulting miss carries the reason. The binding asks for a device's
  // exact target plus a generic fallback, leaving two expansions to share this
  // list.
  KPACK_MAX_ARCH_CANDIDATES = 64,
};

// Upper bound on a single resolved (decompressed) code object. The format does
// not bound a decompressed code object; the reference kpack runtime caps only
// the compressed zstd blob (at 4 GiB). Real AMDGPU code objects are far
// smaller, so this rejects an implausible (corrupt) size before allocating it.
#define KPACK_MAX_CODE_OBJECT_BYTES (256ULL * 1024 * 1024)

//===----------------------------------------------------------------------===//
// Debug logging
//===----------------------------------------------------------------------===//

// True if an environment variable is set to a non-empty, non-"0" value.
static bool kpack_env_flag(const char* name) {
  const char* e = getenv(name);
  return e != NULL && e[0] != '\0' && e[0] != '0';
}

static bool kpack_debug_enabled(void) {
  return kpack_env_flag("ROCM_KPACK_DEBUG");
}

#define KPACK_DBG(...)                             \
  do {                                             \
    if (kpack_debug_enabled()) {                   \
      fprintf(stderr, "[HRX kpack] " __VA_ARGS__); \
      fprintf(stderr, "\n");                       \
      fflush(stderr);                              \
    }                                              \
  } while (0)

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

iree_status_t iree_hal_streaming_kpack_for_each_compatible_target(
    iree_string_view_t agent_isa,
    iree_hal_streaming_kpack_target_callback_t callback, void* user_data) {
  iree_string_view_t isa =
      iree_hal_streaming_kpack_strip_target_prefix(agent_isa);
  if (isa.size == 0) return iree_ok_status();  // no candidates

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
          if (j > start) {
            if (feature_count >= KPACK_MAX_FEATURES) {
              return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                      "kpack ISA target '%.*s' carries more "
                                      "than %d subsettable feature flags",
                                      (int)agent_isa.size, agent_isa.data,
                                      KPACK_MAX_FEATURES);
            }
            features[feature_count++] =
                iree_make_string_view(isa.data + start, j - start);
          }
          start = j + 1;
        }
      }
      break;
    }
  }
  if (processor.size == 0) return iree_ok_status();  // no candidates

  // The processor prefixes every candidate, so a processor that does not fit
  // the target buffer leaves no candidate representable at all.
  if (processor.size >= IREE_HAL_STREAMING_KPACK_TARGET_CAPACITY) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "kpack ISA target processor '%.*s' does not fit the "
        "%d-byte target candidate buffer",
        (int)processor.size, processor.data,
        IREE_HAL_STREAMING_KPACK_TARGET_CAPACITY);
  }

  const iree_host_size_t n = feature_count;
  if (n == 0) {
    char buf[IREE_HAL_STREAMING_KPACK_TARGET_CAPACITY];
    memcpy(buf, processor.data, processor.size);
    // One candidate, so there is no later iteration for an early stop to skip.
    (void)callback(iree_make_string_view(buf, processor.size), user_data);
    return iree_ok_status();
  }

  // Power set of features in descending specificity: candidates carrying more
  // features rank first, and within one cardinality the numerically larger mask
  // ranks first. Bit (n-1-i) selects features[i], so the larger mask keeps
  // earlier-listed features, dropping from the right first and preserving the
  // original feature order in each candidate.
  const uint32_t full_mask = (1u << n) - 1;
  for (int cardinality = (int)n; cardinality >= 0; --cardinality) {
    for (uint32_t mask = full_mask;; --mask) {
      if ((int)iree_math_count_ones_u32(mask) == cardinality) {
        char buf[IREE_HAL_STREAMING_KPACK_TARGET_CAPACITY];
        memcpy(buf, processor.data, processor.size);
        iree_host_size_t len = processor.size;
        bool overflow = false;
        for (iree_host_size_t i = 0; i < n; ++i) {
          if (mask & (1u << (n - 1 - i))) {
            if (len + 1 + features[i].size >= sizeof(buf)) {
              // A single over-long candidate is skipped rather than failed: a
              // shorter subset of the same target still resolves.
              overflow = true;
              break;
            }
            buf[len++] = ':';
            memcpy(buf + len, features[i].data, features[i].size);
            len += features[i].size;
          }
        }
        if (!overflow) {
          if (callback(iree_make_string_view(buf, len), user_data)) {
            return iree_ok_status();
          }
        }
      }
      if (mask == 0) break;
    }
  }
  return iree_ok_status();
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

  // Compression scheme. An absent field means "none"; a present one decides, so
  // a value that is not a scheme name is malformed structure rather than a
  // reason to apply the default and decode the archive under a scheme it never
  // declared.
  out_archive->compression = IREE_HAL_STREAMING_KPACK_COMPRESSION_NONE;
  iree_const_byte_span_t value;
  if (kpack_mp_map_find(out_archive->toc_map, IREE_SV("compression_scheme"),
                        &value)) {
    iree_string_view_t scheme;
    if (!kpack_mp_as_str(value, &scheme)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "kpack TOC 'compression_scheme' is not a string");
    }
    if (iree_string_view_equal(scheme, IREE_SV("zstd-per-kernel"))) {
      out_archive->compression =
          IREE_HAL_STREAMING_KPACK_COMPRESSION_ZSTD_PER_KERNEL;
    } else if (!iree_string_view_equal(scheme, IREE_SV("none"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported kpack compression scheme '%.*s'",
                              (int)scheme.size, scheme.data);
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
  if (size > KPACK_MAX_CODE_OBJECT_BYTES) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kpack code object %" PRIu64
                            " bytes exceeds limit %" PRIu64,
                            size, (uint64_t)KPACK_MAX_CODE_OBJECT_BYTES);
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
#if !defined(IREE_HAVE_ZSTD)
  (void)archive;
  (void)ordinal;
  (void)original_size;
  (void)host_allocator;
  (void)out_kernel;
  (void)out_kernel_size;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "kpack zstd-per-kernel archive requires the HRX HIP binding");
#else
  // Cap the decompressed size before allocating the output buffer for it.
  if (original_size > KPACK_MAX_CODE_OBJECT_BYTES) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kpack zstd decompressed size %" PRIu64 " exceeds limit %" PRIu64,
        original_size, (uint64_t)KPACK_MAX_CODE_OBJECT_BYTES);
  }

  // Blob layout: [num_kernels: u32][ (frame_size: u32)(zstd_frame) ]*, all
  // little-endian (these are raw binary fields, not msgpack). The frame walk
  // below is self-terminating on the blob length, so num_kernels only needs to
  // bound the requested ordinal.
  const uint8_t* p = archive->zstd_blob.data;
  const uint8_t* end = p + archive->zstd_blob.data_length;
  if ((uint64_t)(end - p) < sizeof(uint32_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kpack zstd blob truncated before kernel count");
  }
  uint32_t num_kernels;
  memcpy(&num_kernels, p, sizeof(num_kernels));
  p += sizeof(uint32_t);
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
  // Unreachable: the loop returns at i == ordinal, and ordinal < num_kernels.
  return iree_make_status(IREE_STATUS_INTERNAL, "kpack zstd frame walk failed");
#endif  // IREE_HAVE_ZSTD
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
// Path resolution and mapping queries
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

// Simplifies "." components, duplicate '/' separators, and a trailing '/' of
// |path| into |out|, preserving a leading '/'. These are identity-preserving
// under POSIX pathname resolution regardless of any preceding symlink, so
// collapsing them names the same file. ".." is copied literally, not collapsed:
// its meaning depends on what the preceding component resolves to, so folding
// "dir/.." away here would name a different file than the kernel resolves at
// open time whenever "dir" is a symlink. It is left for the kernel to resolve.
// No filesystem access.
static iree_status_t kpack_normalize_path(iree_string_view_t path, char* out,
                                          iree_host_size_t out_capacity) {
  const bool is_absolute = path.size > 0 && path.data[0] == '/';
  iree_host_size_t out_len = 0;

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
    for (iree_host_size_t k = 0; k < comp.size; ++k) KPACK_PUT(comp.data[k]);
    KPACK_PUT('/');
  }

  // Strip a single trailing '/' (but keep a lone root "/").
  if (out_len > 1 && out[out_len - 1] == '/') --out_len;
  if (out_len == 0) KPACK_PUT('.');  // path named only "." components
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

#if defined(__linux__)

// Fills |out_mapping| from the tail of a /proc/self/maps line describing the
// mapping that contains |address| and ends at |high|. |fields| points just past
// the address range, at the whitespace before the permission bits, and is
// modified in place while trimming the trailing newline.
//
// |out_mapping| and |path_buffer| are initialized by the caller and written
// only once the entry has been parsed in full, so a failure leaves them as the
// caller initialized them rather than half-filled.
static iree_status_t kpack_parse_maps_entry(
    char* fields, const void* address, uintptr_t high, char* path_buffer,
    iree_host_size_t path_capacity,
    iree_hal_streaming_kpack_mapping_t* out_mapping) {
  // Remaining fields: "<perms> <offset> <dev> <inode> <path>".
  char* cursor = fields;
  while (*cursor == ' ') ++cursor;
  const char* permissions = cursor;
  while (*cursor && *cursor != ' ') ++cursor;
  const iree_host_size_t permissions_length =
      (iree_host_size_t)(cursor - permissions);
  // A mapping carries the read bit only when its pages can actually be
  // dereferenced. The ELF loader reserves a library's whole span and protects
  // the segments individually, leaving PROT_NONE alignment gaps that are
  // file-backed and named exactly like the segments around them, so the path is
  // no evidence that the bytes are readable.
  if (permissions_length == 0 || permissions[0] != 'r') {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "kpack_query_mapping: address %p is in a mapping that cannot be read "
        "(permissions '%.*s')",
        address, (int)permissions_length, permissions);
  }

  while (*cursor == ' ') ++cursor;
  while (*cursor && *cursor != ' ') ++cursor;  // skip offset
  while (*cursor == ' ') ++cursor;
  while (*cursor && *cursor != ' ') ++cursor;  // skip dev
  while (*cursor == ' ') ++cursor;
  while (*cursor && *cursor != ' ') ++cursor;  // skip inode
  while (*cursor == ' ') ++cursor;             // cursor -> path (or EOL)

  // Terminate the path at the line's newline.
  char* newline = cursor;
  while (*newline && *newline != '\n') ++newline;
  *newline = '\0';

  const iree_host_size_t readable_bytes =
      (iree_host_size_t)(high - (uintptr_t)address);

  // Three kinds of mapping have a valid extent but no file to resolve paths
  // against: an anonymous one, a special region ([heap], [stack], ...), and one
  // whose backing file has been unlinked. The kernel renders the last as
  // "<path> (deleted)", which names no file that can be opened; reporting it as
  // a path would silently resolve relative search paths against a directory
  // that no longer holds what the path claims.
  const iree_string_view_t path = iree_make_cstring_view(cursor);
  if (path.size == 0 || cursor[0] == '[' ||
      iree_string_view_ends_with(path, IREE_SV(" (deleted)"))) {
    out_mapping->readable_bytes = readable_bytes;
    return iree_ok_status();
  }
  if (path.size + 1 > path_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "kpack_query_mapping: mapping path too long for "
                            "buffer (%" PRIhsz " bytes)",
                            path.size);
  }
  memcpy(path_buffer, path.data, path.size + 1);
  out_mapping->readable_bytes = readable_bytes;
  out_mapping->path = iree_make_string_view(path_buffer, path.size);
  return iree_ok_status();
}

#endif  // __linux__

iree_status_t iree_hal_streaming_kpack_query_mapping(
    const void* address, char* path_buffer, iree_host_size_t path_capacity,
    iree_hal_streaming_kpack_mapping_t* out_mapping) {
  if (!address || !path_buffer || path_capacity == 0 || !out_mapping) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kpack_query_mapping: null argument");
  }
  // Sole owner of the outputs' initial state: the scan below may find no entry
  // to describe, and kpack_parse_maps_entry writes only what it fully parses.
  memset(out_mapping, 0, sizeof(*out_mapping));
  path_buffer[0] = '\0';
#if defined(__linux__)
  FILE* maps = fopen("/proc/self/maps", "r");
  if (!maps) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "kpack_query_mapping: cannot open /proc/self/maps");
  }
  const uintptr_t target = (uintptr_t)address;
  char line[KPACK_PATH_MAX + 256];
  // The scan stops at the one mapping containing |target|, whether or not that
  // mapping can be described. Build the not-found status after the scan: it
  // carries a formatted (heap-allocated) message that would leak on every
  // successful query if it were built eagerly.
  bool found = false;
  iree_status_t status = iree_ok_status();
  while (fgets(line, sizeof(line), maps)) {
    // Format: "<low>-<high> <perms> <offset> <dev> <inode> <path>".
    char* dash = strchr(line, '-');
    if (!dash) continue;
    char* end_pointer = NULL;
    const uintptr_t low = (uintptr_t)strtoull(line, &end_pointer, 16);
    if (end_pointer != dash) continue;
    const uintptr_t high = (uintptr_t)strtoull(dash + 1, &end_pointer, 16);
    if (target < low || target >= high) continue;
    found = true;
    // end_pointer points just past the high address, at the space before perms.
    status = kpack_parse_maps_entry(end_pointer, address, high, path_buffer,
                                    path_capacity, out_mapping);
    break;
  }
  fclose(maps);
  if (iree_status_is_ok(status) && !found) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "kpack_query_mapping: address %p is not in any "
                              "mapping",
                              address);
  }
  return status;
#else
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "kpack_query_mapping is not implemented on this platform");
#endif  // __linux__
}

//===----------------------------------------------------------------------===//
// Top-level resolution
//===----------------------------------------------------------------------===//

// Opens the file at |path| for reading, reporting whether anything is there.
// The open is kept separate from the mapping below because the two answer
// different questions: whether |path| names a member of the search space at
// all, and whether what it names is usable as an archive. A combined
// open-and-map cannot answer the first, since mapping a directory reports
// ENODEV, which maps to IREE_STATUS_NOT_FOUND exactly like the ENOENT of a path
// naming nothing.
static iree_status_t kpack_open_file(const char* path,
                                     iree_allocator_t host_allocator,
                                     iree_io_file_handle_t** out_handle) {
  *out_handle = NULL;
  return iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_SHARE_READ,
      iree_make_cstring_view(path), host_allocator, out_handle);
}

// Memory-maps the whole of an opened file; the mapping owns the bytes and the
// spans parsed from it stay valid until it is released. The mapping retains
// |handle|, so the caller may release its own reference once this returns. The
// file must not change size while mapped.
static iree_status_t kpack_map_file(iree_io_file_handle_t* handle,
                                    iree_allocator_t host_allocator,
                                    iree_io_file_mapping_t** out_mapping) {
  *out_mapping = NULL;
  return iree_io_file_map_view(handle, IREE_IO_FILE_ACCESS_READ, /*offset=*/0,
                               IREE_HOST_SIZE_MAX,
                               IREE_IO_FILE_MAPPING_FLAG_PRIVATE |
                                   IREE_IO_FILE_MAPPING_FLAG_EXCLUDE_FROM_DUMPS,
                               host_allocator, out_mapping);
}

// An opened archive plus the search path spelling it was reached by.
typedef struct {
  // Search path the archive was opened from, exactly as spelled by whoever
  // named it; the dedup key.
  char path[KPACK_PATH_MAX];
  // Mapping owning the archive bytes; |archive|'s spans point into it and are
  // valid until it is released.
  iree_io_file_mapping_t* mapping;
  // Archive parsed over |mapping|'s bytes.
  iree_hal_streaming_kpack_archive_t archive;
} kpack_open_archive_t;

// Mutable state threaded through resolution.
typedef struct {
  // Allocator for archive mappings and the resolved code object.
  iree_allocator_t host_allocator;
  // Ranked, de-duplicated compatible architecture candidates.
  char arch[KPACK_MAX_ARCH_CANDIDATES]
           [IREE_HAL_STREAMING_KPACK_TARGET_CAPACITY];
  // Number of populated entries in |arch|.
  iree_host_size_t arch_count;
  // Archives opened so far, owned; capacity is MAX_OPEN_ARCHIVES entries.
  kpack_open_archive_t* open;
  // Number of populated entries in |open|.
  iree_host_size_t open_count;
  // First problem that kept a candidate out of the search: an archive that is
  // present but unusable, or a search path that cannot be formed. Owned, and
  // reported only if the search matches nothing; a search that finds a code
  // object discards it.
  iree_status_t deferred_error;
} kpack_resolve_state_t;

// Records a problem that kept a candidate archive out of the search. The first
// one recorded wins: it is the most likely root cause, and it is stable with
// respect to whichever unrelated files happen to sit later on the search path.
// Every one is logged, since only one can be returned and a user who fixes that
// one and re-runs would otherwise meet the next only after the round trip.
static void kpack_note_search_error(kpack_resolve_state_t* state,
                                    iree_status_t status) {
  if (kpack_debug_enabled()) {
    // Rendering allocates, so it is gated rather than left to KPACK_DBG.
    char* message = NULL;
    iree_host_size_t message_length = 0;
    if (iree_status_to_string(status, &state->host_allocator, &message,
                              &message_length)) {
      KPACK_DBG("candidate rejected: %.*s", (int)message_length, message);
      iree_allocator_free(state->host_allocator, message);
    } else {
      KPACK_DBG("candidate rejected: %s",
                iree_status_code_string(iree_status_code(status)));
    }
  }
  if (iree_status_is_ok(state->deferred_error)) {
    state->deferred_error = status;
  } else {
    iree_status_ignore(status);
  }
}

// Appends |target| to the ranked arch candidate list if not already present.
// |target| comes from for_each_compatible_target, which yields a nonempty
// string shorter than IREE_HAL_STREAMING_KPACK_TARGET_CAPACITY, so it always
// fits an arch[] slot.
static bool kpack_arch_collect_cb(iree_string_view_t target, void* user_data) {
  kpack_resolve_state_t* state = (kpack_resolve_state_t*)user_data;
  for (iree_host_size_t i = 0; i < state->arch_count; ++i) {
    if (iree_string_view_equal(
            iree_make_string_view(state->arch[i], strlen(state->arch[i])),
            target)) {
      return false;  // already present
    }
  }
  if (state->arch_count >= KPACK_MAX_ARCH_CANDIDATES) {
    // The list holds the highest-ranked candidates; this one and any after it
    // are lower-ranked fallbacks. The search runs highest-first over what is
    // kept, so dropping these can only miss, never select a lower-ranked target
    // over a higher-ranked one. Record why rather than fail: a match on a kept
    // candidate still succeeds, and a miss then carries the reason instead of
    // surfacing as an unexplained not-found.
    kpack_note_search_error(
        state, iree_make_status(
                   IREE_STATUS_RESOURCE_EXHAUSTED,
                   "kpack ranked more than %d compatible target candidates; "
                   "the lowest-ranked fallbacks past the cap were not searched",
                   KPACK_MAX_ARCH_CANDIDATES));
    return false;
  }
  memcpy(state->arch[state->arch_count], target.data, target.size);
  state->arch[state->arch_count][target.size] = '\0';
  ++state->arch_count;
  return false;  // continue collecting
}

// Maps and parses the archive at |path| into the resolve's open list,
// deduplicated by path spelling. |path| is NUL-terminated and shorter than
// KPACK_PATH_MAX; every caller forms it in a buffer of that size.
//
// A path the open reports as naming nothing is not a member of the search
// space: the embedded search paths are speculative, so a miss is the normal
// case and skipping it still evaluates the space completely. Once the open
// succeeds the path names something, and every way that something can fail to
// be a usable archive — it cannot be mapped, it is not an archive, it is a
// directory — is a fact about the user's system that they cannot otherwise
// discover, so it is skipped only after being recorded on |state|.
//
// Returns non-OK only when the search itself cannot continue, which is the one
// case a deferred diagnostic cannot express: more archives than
// MAX_OPEN_ARCHIVES means the ranked search space cannot be evaluated, so "no
// code object found" would not be an honest answer.
static iree_status_t kpack_try_open_archive(kpack_resolve_state_t* state,
                                            const char* path) {
  // The key is the spelling as written, compared verbatim, because no lexical
  // rewrite is an identity function for paths: collapsing ".." does not commute
  // with symlink resolution in either direction, so spellings that collapse
  // alike can name different files and one file can be named by spellings that
  // do not. Keying on a rewrite would therefore drop a distinct archive from
  // the search. Dedup exists only to avoid mapping one archive twice, so it
  // claims no more than it can prove: identical spellings name one file and are
  // opened once, while an alias it cannot see (a "." segment, a symlink, a
  // hardlink, a bind mount) costs one redundant mapping rather than a lost
  // archive.
  for (iree_host_size_t i = 0; i < state->open_count; ++i) {
    if (strcmp(state->open[i].path, path) == 0) {
      return iree_ok_status();  // dedup
    }
  }

  iree_io_file_handle_t* handle = NULL;
  iree_status_t status = kpack_open_file(path, state->host_allocator, &handle);
  if (!iree_status_is_ok(status)) {
    if (iree_status_is_not_found(status)) {
      KPACK_DBG("no archive at %s", path);
      iree_status_ignore(status);
    } else {
      kpack_note_search_error(state, status);
    }
    return iree_ok_status();
  }

  // Past the open, |path| names something. It is no longer a candidate that
  // happens to be absent, so nothing below is skipped silently.
  iree_io_file_mapping_t* mapping = NULL;
  status = kpack_map_file(handle, state->host_allocator, &mapping);
  iree_io_file_handle_release(handle);  // retained by |mapping| on success
  if (!iree_status_is_ok(status)) {
    kpack_note_search_error(
        state, iree_status_annotate_f(
                   status,
                   "kpack search path '%s' names something that cannot be "
                   "mapped as an archive file",
                   path));
    return iree_ok_status();
  }
  iree_hal_streaming_kpack_archive_t archive;
  status = iree_hal_streaming_kpack_archive_open(
      iree_io_file_mapping_contents_ro(mapping), &archive);
  if (!iree_status_is_ok(status)) {
    // The parse sees only bytes, so the path it rejected is named here or
    // nowhere: without it a caller learns some archive is malformed but not
    // which, which is the fact they cannot otherwise discover.
    kpack_note_search_error(
        state,
        iree_status_annotate_f(
            status, "kpack search path '%s' is not a usable archive", path));
    iree_io_file_mapping_release(mapping);
    return iree_ok_status();
  }

  // The cap is tested here, against a real archive: a path that holds nothing
  // or holds an unparseable file never consumed a slot, so it must not be the
  // candidate that reports the set as overfull.
  if (state->open_count >= IREE_HAL_STREAMING_KPACK_MAX_OPEN_ARCHIVES) {
    iree_io_file_mapping_release(mapping);
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "kpack search matched more than %d archives, so the ranked search "
        "cannot be completed and a truncated one would silently select a "
        "lower-ranked target; set ROCM_KPACK_PATH to the specific archive(s) "
        "needed (%s)",
        IREE_HAL_STREAMING_KPACK_MAX_OPEN_ARCHIVES, path);
  }
  kpack_open_archive_t* slot = &state->open[state->open_count];
  slot->archive = archive;
  slot->mapping = mapping;
  memcpy(slot->path, path, strlen(path) + 1);
  ++state->open_count;
  KPACK_DBG("opened archive %s", path);
  return iree_ok_status();
}

// Opens every archive in a ':'-separated path list (ROCM_KPACK_PATH / _PREFIX).
// Returns non-OK only when the search cannot continue (see
// kpack_try_open_archive).
static iree_status_t kpack_open_path_list(kpack_resolve_state_t* state,
                                          const char* list_string) {
  iree_string_view_t list = iree_make_cstring_view(list_string);
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && list.size > 0) {
    iree_string_view_t entry;
    iree_string_view_t rest;
    iree_string_view_split(list, ':', &entry, &rest);
    list = rest;  // empty when no separator remains, terminating the loop
    if (entry.size == 0) continue;
    char path[KPACK_PATH_MAX];
    if (entry.size + 1 > sizeof(path)) {
      kpack_note_search_error(
          state, iree_make_status(
                     IREE_STATUS_OUT_OF_RANGE,
                     "kpack search path entry is too long (%" PRIhsz " bytes)",
                     entry.size));
      continue;
    }
    memcpy(path, entry.data, entry.size);
    path[entry.size] = '\0';
    status = kpack_try_open_archive(state, path);
  }
  return status;
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

  // 1. Query the mapping holding the metadata. This bounds the parse below and
  // names the owning binary in one scan, so the extent and the path are
  // guaranteed to describe the same mapping.
  char binary_path[KPACK_PATH_MAX];
  iree_hal_streaming_kpack_mapping_t mapping;
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_kpack_query_mapping(hipk_metadata, binary_path,
                                             sizeof(binary_path), &mapping),
      "bounding HIPK metadata at %p", hipk_metadata);
  if (mapping.path.size > 0) {
    KPACK_DBG("owning binary: %.*s", (int)mapping.path.size, mapping.path.data);
  }

  // 2. Parse the HIPK metadata blob. The blob carries no length, so the reader
  // is bounded by what is readable at the pointer; the cap then rejects a blob
  // no real metadata approaches.
  iree_hal_streaming_kpack_metadata_t metadata;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_kpack_parse_metadata(
      iree_make_const_byte_span(
          hipk_metadata,
          iree_min(
              mapping.readable_bytes,
              (iree_host_size_t)IREE_HAL_STREAMING_KPACK_MAX_METADATA_SIZE)),
      &metadata));

  // 3. Lookup key is "<kernel_name>#<co_index>".
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

  // 4. Build the ranked, de-duplicated architecture candidate list across all
  // requested targets (priority order, then feature-subset specificity). A
  // target the expansion cannot represent (too many subsettable features, or an
  // over-long processor) fails the resolve rather than contributing a silently
  // wrong candidate set.
  kpack_resolve_state_t state;
  memset(&state, 0, sizeof(state));
  state.host_allocator = host_allocator;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < target_arch_count; ++i) {
    status = iree_hal_streaming_kpack_for_each_compatible_target(
        target_archs[i], kpack_arch_collect_cb, &state);
  }
  if (iree_status_is_ok(status) && state.arch_count == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "kpack resolve: no usable target architectures");
  }
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(state.deferred_error);
    return status;
  }

  state.open = NULL;
  status = iree_allocator_malloc(
      host_allocator,
      IREE_HAL_STREAMING_KPACK_MAX_OPEN_ARCHIVES * sizeof(kpack_open_archive_t),
      (void**)&state.open);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(state.deferred_error);
    return status;
  }

  // 5. Open candidate archives. ROCM_KPACK_PATH overrides the embedded search
  // paths entirely; otherwise ROCM_KPACK_PATH_PREFIX entries are prepended and
  // embedded search paths are resolved relative to the owning binary, expanding
  // "@GFXARCH@" for each architecture candidate.
  const char* env_override = getenv("ROCM_KPACK_PATH");
  const char* env_prefix = getenv("ROCM_KPACK_PATH_PREFIX");

  if (env_override && env_override[0]) {
    status = kpack_open_path_list(&state, env_override);
  } else {
    if (env_prefix && env_prefix[0]) {
      status = kpack_open_path_list(&state, env_prefix);
    }
    for (iree_host_size_t i = 0;
         iree_status_is_ok(status) && i < metadata.search_path_count; ++i) {
      iree_string_view_t rel = metadata.search_paths[i];
      bool is_absolute = rel.size > 0 && rel.data[0] == '/';
      if (!is_absolute && mapping.path.size == 0) {
        // The search path cannot be formed at all: it is relative to the binary
        // owning the metadata, and that mapping has no file to resolve against.
        kpack_note_search_error(
            &state,
            iree_make_status(
                IREE_STATUS_FAILED_PRECONDITION,
                "kpack search path '%.*s' is relative to the binary owning the "
                "metadata, which is a mapping with no backing file to resolve "
                "it against",
                (int)rel.size, rel.data));
        continue;
      }
      char resolved[KPACK_PATH_MAX];
      iree_status_t rs = iree_hal_streaming_kpack_resolve_relative_path(
          mapping.path, rel, resolved, sizeof(resolved));
      if (!iree_status_is_ok(rs)) {
        kpack_note_search_error(&state, rs);
        continue;
      }
      iree_string_view_t resolved_view =
          iree_make_string_view(resolved, strlen(resolved));
      bool has_placeholder =
          iree_string_view_find_char(resolved_view, '@', 0) !=
          IREE_STRING_VIEW_NPOS;
      if (!has_placeholder) {
        status = kpack_try_open_archive(&state, resolved);
        continue;
      }
      // Expand "@GFXARCH@" for each candidate architecture.
      for (iree_host_size_t a = 0;
           iree_status_is_ok(status) && a < state.arch_count; ++a) {
        char expanded[KPACK_PATH_MAX];
        iree_status_t es = iree_hal_streaming_kpack_expand_gfxarch(
            resolved_view, iree_make_cstring_view(state.arch[a]), expanded,
            sizeof(expanded), /*out_had_placeholder=*/NULL);
        if (!iree_status_is_ok(es)) {
          kpack_note_search_error(&state, es);
          continue;
        }
        status = kpack_try_open_archive(&state, expanded);
      }
    }
  }

  // 6. Arch-first search: for each architecture candidate in priority order,
  // probe every opened archive; the first hit wins.
  bool found = false;
  for (iree_host_size_t a = 0;
       iree_status_is_ok(status) && a < state.arch_count && !found; ++a) {
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
                  state.open[o].path, kernel_size);
        break;
      }
      if (iree_status_is_not_found(ks)) {
        iree_status_ignore(ks);  // not in this archive — keep looking
      } else {
        // Corrupt/unsupported archive: remember but keep searching others.
        kpack_note_search_error(&state, ks);
      }
    }
  }

  if (iree_status_is_ok(status) && !found) {
    if (state.open_count == 0) {
      // No archive was opened, so the search space was empty. A candidate that
      // was rejected is then the only explanation there is, and it answers as
      // itself: its code carries the diagnosis (a permissions problem, a
      // malformed file) that a generic "not found" would erase.
      if (!iree_status_is_ok(state.deferred_error)) {
        status = state.deferred_error;
        state.deferred_error = iree_ok_status();
      } else {
        status = iree_make_status(
            IREE_STATUS_NOT_FOUND,
            "no kpack archive found for '%s' (%" PRIhsz
            " search paths; set ROCM_KPACK_DEBUG=1 for details)",
            lookup_key, metadata.search_path_count);
      }
    } else {
      // Archives were opened and searched exhaustively, so the miss is a target
      // mismatch and that is the explanation the caller needs. A rejected
      // candidate is attached to it rather than replacing it: it may still be
      // the root cause, but it cannot outrank a search that actually ran.
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "no kpack code object for '%s' matches target '%.*s' in %" PRIhsz
          " archive(s)",
          lookup_key, (int)target_archs[0].size, target_archs[0].data,
          state.open_count);
      status = iree_status_join(status, state.deferred_error);
      state.deferred_error = iree_ok_status();  // consumed by the join
    }
  }
  // Discarded unless it was transferred to |status| above: the search either
  // succeeded, making the diagnostic moot, or failed terminally, which outranks
  // it.
  iree_status_ignore(state.deferred_error);

  // Unmap the archives; the returned code object is an independent copy.
  for (iree_host_size_t o = 0; o < state.open_count; ++o) {
    iree_io_file_mapping_release(state.open[o].mapping);
  }
  iree_allocator_free(host_allocator, state.open);
  return status;
}
