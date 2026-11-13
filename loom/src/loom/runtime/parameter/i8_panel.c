// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/parameter/i8_panel.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "iree/io/file_handle.h"
#include "loom/util/digest.h"

static uint32_t panel_block_size(uint32_t type) { return type == 8 ? 32 : 256; }
static uint32_t panel_block_bytes(uint32_t type) {
  switch (type) {
    case 8:
      return 34;
    case 12:
      return 144;
    case 13:
      return 176;
    case 14:
      return 210;
    default:
      return 0;
  }
}
static float panel_half(const uint8_t* p) {
  return iree_math_f16_to_f32((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}
static float panel_round_half(double value) {
  return iree_math_f16_to_f32(iree_math_f64_to_f16(value));
}

iree_status_t loom_runtime_i8_panel_decode(uint32_t type,
                                           iree_const_byte_span_t blocks,
                                           iree_host_size_t count,
                                           float* values) {
  uint32_t stride = panel_block_bytes(type), width = panel_block_size(type);
  if (!stride || count > blocks.data_length / stride) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "i8-panel requires complete Q4_K/Q5_K/Q6_K/Q8_0 blocks (type %u)",
        type);
  }
  for (iree_host_size_t b = 0; b < count; ++b) {
    const uint8_t* p = blocks.data + b * stride;
    float* out = values + b * width;
    if (type == 8) {
      float d = panel_half(p);
      for (uint32_t i = 0; i < 32; ++i) {
        out[i] = panel_round_half((double)d * (int8_t)p[2 + i]);
      }
    } else if (type == 12 || type == 13) {
      double d = panel_half(p), dm = panel_half(p + 2);
      for (uint32_t g = 0; g < 8; ++g) {
        unsigned s =
            g < 4 ? p[4 + g] & 63 : (p[8 + g] & 15) | ((p[g] >> 2) & 48);
        unsigned m =
            g < 4 ? p[8 + g] & 63 : (p[8 + g] >> 4) | ((p[4 + g] >> 2) & 48);
        double scale = panel_round_half(d * s),
               zero = panel_round_half(-dm * m);
        for (uint32_t i = 0; i < 32; ++i) {
          unsigned q =
              (p[(type == 12 ? 16 : 48) + (g / 2) * 32 + i] >> (4 * (g % 2))) &
              15;
          if (type == 13) {
            q |= ((p[16 + i] >> g) & 1) << 4;
          }
          out[g * 32 + i] = panel_round_half(scale * q + zero);
        }
      }
    } else {
      double d = panel_half(p + 208);
      for (uint32_t i = 0; i < 256; ++i) {
        uint32_t h = i / 128, g = (i % 128) / 32, lane = i % 32;
        unsigned q = ((p[h * 64 + (g % 2) * 32 + lane] >> (4 * (g / 2))) & 15) |
                     (((p[128 + h * 32 + lane] >> (2 * g)) & 3) << 4);
        double ds = d * (int8_t)p[192 + i / 16];
        double scale = panel_round_half(ds), zero = panel_round_half(-32 * ds);
        out[i] = panel_round_half(scale * q + zero);
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_runtime_i8_panel_pack(const float* values, uint32_t n,
                                         uint32_t k, int8_t* panels,
                                         float* scales) {
  if (!n || !k || n % 128 || k % 128) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "i8-panel v2 requires nonzero N and K divisible by 128");
  }
  for (uint32_t row = 0; row < n; ++row) {
    const float* w = values + (uint64_t)row * k;
    float maximum = 0;
    for (uint32_t c = 0; c < k; ++c) {
      if (!isfinite(w[c])) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "i8-panel encountered nonfinite dequantized weight");
      }
      maximum = fmaxf(maximum, fabsf(w[c]));
    }
    float scale = iree_math_round_to_nearest_bf16(maximum / 127.0f);
    if (scale == 0) {
      scale = 1;
    }
    if (!isfinite(scale)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "i8-panel column scale is not finite");
    }
    scales[row] = scale;
    for (uint32_t c = 0; c < k; ++c) {
      float x = fminf(127.0f, fmaxf(-127.0f, w[c] / scale));
      float lo = floorf(x), fraction = x - lo;
      int q = (int)lo;
      if (fraction > 0.5f || (fraction == 0.5f && (q & 1))) {
        ++q;
      }
      q = iree_min(127, iree_max(-127, q));
      uint64_t at = (uint64_t)(row / 128) * 128 * k +
                    (uint64_t)(c / 128) * 16384 + ((row % 128) / 8) * 1024 +
                    ((c % 128) / 8) * 64 + (c % 8) * 8 + row % 8;
      panels[at] = (int8_t)q;
    }
  }
  return iree_ok_status();
}

#if !defined(_WIN32)
#include <fcntl.h>
#include <stdlib.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define PANEL_LAYERS 64
#define PANEL_N 10496
#define PANEL_K 5120
#define PANEL_BYTES (UINT64_C(64) * 2 * PANEL_N * PANEL_K)
#define SCALE_BYTES (UINT64_C(64) * 2 * PANEL_N * 4)

struct loom_runtime_i8_panel_cache_t {
  iree_allocator_t allocator;
  iree_string_view_t source;
  iree_string_view_t cache;
  char* directory;
  iree_io_file_handle_t* handles[2];
};
typedef struct panel_tensor_t {
  uint64_t offset;
  uint32_t type;
  bool found;
} panel_tensor_t;
typedef struct panel_cursor_t {
  const uint8_t* data;
  uint64_t size, at;
} panel_cursor_t;

static iree_status_t panel_io(const char* operation, const char* path) {
  return iree_make_status(iree_status_code_from_errno(errno),
                          "i8-panel %s '%s': %s", operation, path,
                          strerror(errno));
}
static iree_status_t panel_take(panel_cursor_t* c, uint64_t n,
                                const uint8_t** out) {
  if (n > c->size - c->at) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "truncated i8-panel GGUF metadata");
  }
  *out = c->data + c->at;
  c->at += n;
  return iree_ok_status();
}
static iree_status_t panel_uint(panel_cursor_t* c, unsigned n, uint64_t* out) {
  const uint8_t* p;
  IREE_RETURN_IF_ERROR(panel_take(c, n, &p));
  *out = 0;
  for (unsigned i = 0; i < n; ++i) {
    *out |= (uint64_t)p[i] << (8 * i);
  }
  return iree_ok_status();
}
static iree_status_t panel_string(panel_cursor_t* c, iree_string_view_t* out) {
  uint64_t n;
  const uint8_t* p;
  IREE_RETURN_IF_ERROR(panel_uint(c, 8, &n));
  IREE_RETURN_IF_ERROR(panel_take(c, n, &p));
  *out = iree_make_string_view((const char*)p, n);
  return iree_ok_status();
}
static iree_status_t panel_skip(panel_cursor_t* c, uint32_t type) {
  static const uint8_t sizes[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
  if (type == 8) {
    iree_string_view_t s;
    return panel_string(c, &s);
  }
  if (type == 9) {
    uint64_t element, count;
    IREE_RETURN_IF_ERROR(panel_uint(c, 4, &element));
    IREE_RETURN_IF_ERROR(panel_uint(c, 8, &count));
    if (element >= 13 || element == 9 || count > c->size - c->at) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid i8-panel GGUF array");
    }
    for (uint64_t i = 0; i < count; ++i) {
      IREE_RETURN_IF_ERROR(panel_skip(c, (uint32_t)element));
    }
    return iree_ok_status();
  }
  if (type >= 13) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid i8-panel GGUF metadata type");
  }
  const uint8_t* p;
  return panel_take(c, sizes[type], &p);
}
// Only metadata pages are accessed here; tensor offsets are bounds-checked
// without dereferencing tensor data. A full source hash is a separate step.
static iree_status_t panel_inspect(const uint8_t* data, uint64_t size,
                                   panel_tensor_t tensors[128]) {
  panel_cursor_t c = {data, size, 0};
  uint64_t magic, version, count, kv;
  IREE_RETURN_IF_ERROR(panel_uint(&c, 4, &magic));
  IREE_RETURN_IF_ERROR(panel_uint(&c, 4, &version));
  if (magic != 0x46554747 || (version != 2 && version != 3)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "i8-panel requires GGUF v2/v3");
  }
  IREE_RETURN_IF_ERROR(panel_uint(&c, 8, &count));
  IREE_RETURN_IF_ERROR(panel_uint(&c, 8, &kv));
  if (count > size / 24 || kv > size / 12) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid i8-panel GGUF counts");
  }
  uint64_t alignment = 32;
  for (uint64_t i = 0; i < kv; ++i) {
    iree_string_view_t key;
    uint64_t type;
    IREE_RETURN_IF_ERROR(panel_string(&c, &key));
    IREE_RETURN_IF_ERROR(panel_uint(&c, 4, &type));
    if (iree_string_view_equal(key, IREE_SV("general.alignment"))) {
      if (type != 4) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "GGUF alignment must be uint32");
      }
      IREE_RETURN_IF_ERROR(panel_uint(&c, 4, &alignment));
    } else {
      IREE_RETURN_IF_ERROR(panel_skip(&c, (uint32_t)type));
    }
  }
  for (uint64_t i = 0; i < count; ++i) {
    iree_string_view_t key;
    uint64_t rank, type, offset, dims[4] = {0};
    IREE_RETURN_IF_ERROR(panel_string(&c, &key));
    IREE_RETURN_IF_ERROR(panel_uint(&c, 4, &rank));
    if (!rank || rank > 4) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid i8-panel GGUF tensor rank");
    }
    for (uint64_t j = 0; j < rank; ++j) {
      IREE_RETURN_IF_ERROR(panel_uint(&c, 8, &dims[j]));
    }
    IREE_RETURN_IF_ERROR(panel_uint(&c, 4, &type));
    IREE_RETURN_IF_ERROR(panel_uint(&c, 8, &offset));
    for (unsigned t = 0; t < 128; ++t) {
      char name[64];
      snprintf(name, sizeof(name), "blk.%u.ffn_%s.weight", t / 2,
               t % 2 ? "up" : "gate");
      if (!iree_string_view_equal(key, iree_make_cstring_view(name))) {
        continue;
      }
      if (tensors[t].found || rank != 2 || dims[0] != PANEL_K ||
          dims[1] != 17408 || !panel_block_bytes((uint32_t)type)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "unsupported i8-panel geometry/encoding for %s: rank %" PRIu64
            " K %" PRIu64 " N %" PRIu64 " type %" PRIu64
            " (requires K5120 N17408 Q4_K/Q5_K/Q6_K/Q8_0)",
            name, rank, dims[0], dims[1], type);
      }
      tensors[t] = (panel_tensor_t){offset, (uint32_t)type, true};
    }
  }
  if (!alignment || alignment > UINT32_MAX || (alignment & (alignment - 1)) ||
      c.at > UINT64_MAX - (alignment - 1)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid GGUF alignment");
  }
  uint64_t start = (c.at + alignment - 1) & ~(alignment - 1);
  for (unsigned t = 0; t < 128; ++t) {
    panel_tensor_t* p = &tensors[t];
    uint64_t length = UINT64_C(17408) * PANEL_K / panel_block_size(p->type) *
                      panel_block_bytes(p->type);
    if (!p->found || start > size || p->offset > size - start ||
        length > size - start - p->offset) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "i8-panel requires 64 complete gate/up tensor pairs");
    }
    p->offset += start;
  }
  return iree_ok_status();
}
static void panel_hex(const loom_sha256_t* hash, char out[65]) {
  static const char digits[] = "0123456789abcdef";
  for (unsigned i = 0; i < 32; ++i) {
    out[2 * i] = digits[hash->bytes[i] >> 4];
    out[2 * i + 1] = digits[hash->bytes[i] & 15];
  }
  out[64] = 0;
}
static void panel_qmap(const panel_tensor_t tensors[128], char out[18000]) {
  int at = snprintf(out, 18000, "[");
  for (unsigned t = 0; t < 128; ++t) {
    at += snprintf(out + at, 18000 - at,
                   "%s{\"name\":\"blk.%u.ffn_%s.weight\",\"shape\":[5120,17408]"
                   ",\"type\":%u}",
                   t ? "," : "", t / 2, t % 2 ? "up" : "gate", tensors[t].type);
  }
  snprintf(out + at, 18000 - at, "]");
}
static void panel_digest(const void* data, size_t size, char hex[65]) {
  loom_sha256_t hash;
  loom_sha256_compute(iree_make_const_byte_span(data, size), &hash);
  panel_hex(&hash, hex);
}
static void panel_key(const char* source, const char* qmap, char key[65]) {
  char material[18000];
  int at =
      snprintf(material, sizeof(material),
               "{\"columns\":10496,\"gguf_sha256\":\"%s\",\"k\":5120,\"kernel_"
               "version\":\"resident-full-k-i8-i32-bf16-column-scale-v1\","
               "\"layers\":64,\"layout_version\":\"full-k-colscale-n128-k128-"
               "nfrag-kfrag-k8-n8-v2\",\"quant_type_map\":[",
               source);
  at += snprintf(material + at, sizeof(material) - at, "%s", qmap + 1);
  // qmap already closes the ordered array used by the original content key.
  at += snprintf(
      material + at, sizeof(material) - at,
      ",\"recipe_version\":\"loom-f16-dequant-bf16-absmax-rne-i8-v1\"}");
  panel_digest(material, at, key);
}
static iree_status_t panel_mkdirs(char* path) {
  for (char* p = path + 1;; ++p) {
    if (*p && *p != '/') {
      continue;
    }
    char saved = *p;
    *p = 0;
    int rc = mkdir(path, 0775);
    *p = saved;
    if (rc && errno != EEXIST) {
      return panel_io("mkdir", path);
    }
    if (!saved) {
      break;
    }
  }
  return iree_ok_status();
}
static iree_status_t panel_write(FILE* f, const void* data, size_t size,
                                 const char* path) {
  if (fwrite(data, 1, size, f) != size) {
    return panel_io("write", path);
  }
  return iree_ok_status();
}
static iree_status_t panel_sync(FILE* f, const char* path) {
  if (fflush(f) || fsync(fileno(f))) {
    return panel_io("sync", path);
  }
  return iree_ok_status();
}
static iree_status_t panel_sync_dir(const char* path) {
  int fd = open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0) {
    return panel_io("open", path);
  }
  int rc = fsync(fd);
  int error = errno;
  close(fd);
  errno = error;
  return rc ? panel_io("sync directory", path) : iree_ok_status();
}
// Stat trust is for a local, trusted cache: dev+ino detect replacement;
// size+nanosecond mtime+ctime detect in-place writes, even with restored mtime.
// Userspace cannot set ctime. Privileged tampering, unreliable filesystem
// times, inode reuse with identical timestamps, and concurrent hostile writers
// remain outside this trust model. LOOM_DERIVE_VERIFY=1 forces every content
// hash.
typedef struct panel_identity_t {
  uint64_t dev, ino, size;
  int64_t mtime_ns, ctime_ns;
  char sha256[65];
} panel_identity_t;
#define PANEL_ID_FORMAT                                        \
  "{\"dev\":%" SCNu64 ",\"ino\":%" SCNu64 ",\"size\":%" SCNu64 \
  ",\"mtime_ns\":%" SCNd64 ",\"ctime_ns\":%" SCNd64            \
  ",\"sha256\":\"%64[0-9a-f]\"}"
#define PANEL_ID_PRINT                                         \
  "{\"dev\":%" PRIu64 ",\"ino\":%" PRIu64 ",\"size\":%" PRIu64 \
  ",\"mtime_ns\":%" PRId64 ",\"ctime_ns\":%" PRId64 ",\"sha256\":\"%s\"}"
#define PANEL_ID_ARGS(p) \
  (p)->dev, (p)->ino, (p)->size, (p)->mtime_ns, (p)->ctime_ns, (p)->sha256
#define PANEL_ID_SCAN(p) \
  &(p)->dev, &(p)->ino, &(p)->size, &(p)->mtime_ns, &(p)->ctime_ns, (p)->sha256

static panel_identity_t panel_identity(const struct stat* st) {
  return (panel_identity_t){
      .dev = st->st_dev,
      .ino = st->st_ino,
      .size = st->st_size,
      .mtime_ns = (int64_t)st->st_mtim.tv_sec * INT64_C(1000000000) +
                  st->st_mtim.tv_nsec,
      .ctime_ns = (int64_t)st->st_ctim.tv_sec * INT64_C(1000000000) +
                  st->st_ctim.tv_nsec};
}
static bool panel_identity_equal(const panel_identity_t* a,
                                 const panel_identity_t* b) {
  return a->dev == b->dev && a->ino == b->ino && a->size == b->size &&
         a->mtime_ns == b->mtime_ns && a->ctime_ns == b->ctime_ns;
}
static bool panel_same_stat(const struct stat* a, const struct stat* b) {
  panel_identity_t x = panel_identity(a), y = panel_identity(b);
  return panel_identity_equal(&x, &y);
}
static bool panel_read_record(const char* path, char out[2048]) {
  FILE* f = fopen(path, "r");
  if (!f) {
    return false;
  }
  size_t n = fread(out, 1, 2047, f);
  bool valid = !ferror(f) && feof(f);
  fclose(f);
  out[n] = 0;
  return valid;
}
static iree_status_t panel_atomic_record(const char* path, const char* text,
                                         const char* parent) {
  char stage[4096];
  if (snprintf(stage, sizeof(stage), "%s.partial-XXXXXX", path) >=
      (int)sizeof(stage)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "i8-panel record path too long");
  }
  int fd = mkstemp(stage);
  if (fd < 0) {
    return panel_io("create record", stage);
  }
  FILE* f = fdopen(fd, "w");
  iree_status_t status = iree_ok_status();
  if (!f) {
    status = panel_io("open record stream", stage);
    close(fd);
  } else {
    status = panel_write(f, text, strlen(text), stage);
    if (iree_status_is_ok(status)) {
      status = panel_sync(f, stage);
    }
    if (fclose(f) && iree_status_is_ok(status)) {
      status = panel_io("close", stage);
    }
  }
  if (iree_status_is_ok(status) && rename(stage, path)) {
    status = panel_io("publish record", path);
  }
  if (iree_status_is_ok(status)) {
    status = panel_sync_dir(parent);
  }
  unlink(stage);
  return status;
}
static void panel_source_record(const panel_identity_t* id, const char* qmap,
                                char out[2048]) {
  snprintf(out, 2048,
           "{\"version\":1,\"source\":" PANEL_ID_PRINT
           ",\"qmap_digest\":\"%s\"}\n",
           PANEL_ID_ARGS(id), qmap);
}
static iree_status_t panel_source_hash(const char* source, const char* root,
                                       const void* data, int fd,
                                       const struct stat* before,
                                       const char* qmap_digest, bool verify,
                                       char hash[65], unsigned* hashes) {
  char canonical[4096], name[65], path[4096], lock_path[4096];
  if (!realpath(source, canonical)) {
    return panel_io("canonicalize source", source);
  }
  // One record per canonical path avoids lost updates between source files.
  panel_digest(canonical, strlen(canonical), name);
  if (snprintf(path, sizeof(path), "%s/identity-%s.json", root, name) >=
          (int)sizeof(path) ||
      snprintf(lock_path, sizeof(lock_path), "%s/identity-%s.lock", root,
               name) >= (int)sizeof(lock_path)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "i8-panel identity path too long");
  }
  int lock = open(lock_path, O_CREAT | O_RDWR | O_CLOEXEC, 0664);
  if (lock < 0) {
    return panel_io("open source lock", lock_path);
  }
  iree_status_t status = iree_ok_status();
  if (flock(lock, LOCK_EX)) {
    status = panel_io("lock source", lock_path);
  }
  panel_identity_t current = panel_identity(before), saved = {0};
  char actual[2048], expected[2048], qmap[65] = {0};
  bool hit = false;
  if (iree_status_is_ok(status) && !verify && panel_read_record(path, actual) &&
      sscanf(actual,
             "{\"version\":1,\"source\":" PANEL_ID_FORMAT
             ",\"qmap_digest\":\"%64[0-9a-f]\"}\n",
             PANEL_ID_SCAN(&saved), qmap) == 7 &&
      strlen(saved.sha256) == 64 && strlen(qmap) == 64) {
    panel_source_record(&saved, qmap, expected);
    hit = !strcmp(actual, expected) && !strcmp(qmap, qmap_digest) &&
          panel_identity_equal(&current, &saved);
  }
  if (iree_status_is_ok(status)) {
    if (hit) {
      memcpy(hash, saved.sha256, 65);
    } else {
      ++*hashes;
      panel_digest(data, before->st_size, current.sha256);
      struct stat after, named;
      if (fstat(fd, &after) || stat(source, &named) ||
          !panel_same_stat(before, &after) ||
          !panel_same_stat(before, &named)) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "i8-panel GGUF changed during hashing");
      } else {
        panel_source_record(&current, qmap_digest, expected);
        status = panel_atomic_record(path, expected, root);
        memcpy(hash, current.sha256, 65);
      }
    }
  }
  flock(lock, LOCK_UN);
  close(lock);
  return status;
}
static iree_status_t panel_payload(const char* path, uint64_t length,
                                   const panel_identity_t* saved, bool verify,
                                   panel_identity_t* current,
                                   unsigned* hashes) {
  int fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return panel_io("open", path);
  }
  struct stat before, after;
  iree_status_t status = iree_ok_status();
  if (fstat(fd, &before)) {
    status = panel_io("stat payload", path);
  } else if (!S_ISREG(before.st_mode) || (uint64_t)before.st_size != length) {
    status =
        iree_make_status(IREE_STATUS_DATA_LOSS,
                         "i8-panel cache payload has wrong size: %s", path);
  } else {
    *current = panel_identity(&before);
    if (!verify && saved && panel_identity_equal(saved, current)) {
      memcpy(current->sha256, saved->sha256, 65);
    } else {
      ++*hashes;
      void* data = mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0);
      if (data == MAP_FAILED) {
        status = panel_io("map", path);
      } else {
        panel_digest(data, length, current->sha256);
        munmap(data, length);
        if (fstat(fd, &after) || !panel_same_stat(&before, &after)) {
          status = iree_make_status(
              IREE_STATUS_DATA_LOSS,
              "i8-panel payload changed during hashing: %s", path);
        }
      }
    }
  }
  close(fd);
  return status;
}
static const char* const panel_names[] = {"panels_gu.i8", "colscale_gu.f32"};
static void panel_receipt(const char* key, const panel_identity_t ids[2],
                          char out[2048]) {
  snprintf(
      out, 2048,
      "{\"version\":2,\"cache_key\":\"%s\",\"panels_gu.i8\":" PANEL_ID_PRINT
      ",\"colscale_gu.f32\":" PANEL_ID_PRINT ",\"complete\":true}\n",
      key, PANEL_ID_ARGS(&ids[0]), PANEL_ID_ARGS(&ids[1]));
}
static iree_status_t panel_derive(const uint8_t* data,
                                  const panel_tensor_t tensors[128],
                                  const char* directory, const char* key,
                                  iree_allocator_t allocator) {
  FILE* files[2] = {NULL, NULL};
  char path[4096];
  panel_identity_t ids[2] = {0};
  float* values = NULL;
  int8_t* packed = NULL;
  iree_status_t status = iree_allocator_malloc(
      allocator, 128 * PANEL_K * sizeof(float), (void**)&values);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(allocator, 128 * PANEL_K, (void**)&packed);
  }
  for (unsigned i = 0; i < 2 && iree_status_is_ok(status); ++i) {
    snprintf(path, sizeof(path), "%s/%s", directory, panel_names[i]);
    files[i] = fopen(path, "wbx");
    if (!files[i]) {
      status = panel_io("create", path);
    }
  }
  loom_sha256_state_t states[2];
  for (unsigned i = 0; i < 2; ++i) {
    loom_sha256_initialize(&states[i]);
  }
  for (unsigned t = 0; t < 128 && iree_status_is_ok(status); ++t) {
    uint32_t type = tensors[t].type, stride = panel_block_bytes(type),
             block = panel_block_size(type);
    for (unsigned n = 0; n < PANEL_N && iree_status_is_ok(status); n += 128) {
      uint64_t length = UINT64_C(128) * PANEL_K / block * stride;
      const uint8_t* raw =
          data + tensors[t].offset + (uint64_t)n * PANEL_K / block * stride;
      status = loom_runtime_i8_panel_decode(
          type, iree_make_const_byte_span(raw, length), 128 * PANEL_K / block,
          values);
      float scales[128];
      if (iree_status_is_ok(status)) {
        status =
            loom_runtime_i8_panel_pack(values, 128, PANEL_K, packed, scales);
      }
      if (iree_status_is_ok(status)) {
        status = panel_write(files[0], packed, 128 * PANEL_K, directory);
      }
      if (iree_status_is_ok(status)) {
        status = panel_write(files[1], scales, sizeof(scales), directory);
      }
      if (iree_status_is_ok(status)) {
        loom_sha256_update(&states[0],
                           iree_make_const_byte_span(packed, 128 * PANEL_K));
        loom_sha256_update(&states[1],
                           iree_make_const_byte_span(scales, sizeof(scales)));
      }
    }
  }
  for (unsigned i = 0; i < 2; ++i) {
    loom_sha256_t hash;
    loom_sha256_finalize(&states[i], &hash);
    panel_hex(&hash, ids[i].sha256);
    if (files[i]) {
      if (iree_status_is_ok(status)) {
        status = panel_sync(files[i], directory);
      }
      if (fclose(files[i]) && iree_status_is_ok(status)) {
        status = panel_io("close", directory);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    for (unsigned i = 0; i < 2 && iree_status_is_ok(status); ++i) {
      snprintf(path, sizeof(path), "%s/%s", directory, panel_names[i]);
      struct stat st;
      if (stat(path, &st)) {
        status = panel_io("stat derived payload", path);
      } else {
        panel_identity_t id = panel_identity(&st);
        memcpy(id.sha256, ids[i].sha256, 65);
        ids[i] = id;
      }
    }
  }
  if (iree_status_is_ok(status)) {
    char receipt[2048];
    panel_receipt(key, ids, receipt);
    snprintf(path, sizeof(path), "%s/cache.json", directory);
    FILE* f = fopen(path, "wx");
    if (!f) {
      status = panel_io("create", path);
    } else {
      status = panel_write(f, receipt, strlen(receipt), path);
      if (iree_status_is_ok(status)) {
        status = panel_sync(f, path);
      }
      fclose(f);
    }
  }
  if (iree_status_is_ok(status)) {
    status = panel_sync_dir(directory);
  }
  iree_allocator_free(allocator, packed);
  iree_allocator_free(allocator, values);
  return status;
}
static iree_status_t panel_validate(const char* directory, const char* key,
                                    bool verify, unsigned* hashes) {
  char path[4096], actual[2048], expected[2048], stored_key[65] = {0};
  panel_identity_t saved[2] = {0}, current[2] = {0};
  static const char kReceipt[] = "/cache.json";
  size_t directory_length = strlen(directory);
  if (directory_length + sizeof(kReceipt) > sizeof(path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "i8-panel cache directory path is too long");
  }
  memcpy(path, directory, directory_length);
  memcpy(path + directory_length, kReceipt, sizeof(kReceipt));
  if (!panel_read_record(path, actual)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "i8-panel unreadable cache receipt: %s", path);
  }
  bool legacy = false;
  if (sscanf(actual,
             "{\"version\":2,\"cache_key\":\"%64[0-9a-f]\",\"panels_gu."
             "i8\":" PANEL_ID_FORMAT ",\"colscale_gu.f32\":" PANEL_ID_FORMAT
             ",\"complete\":true}\n",
             stored_key, PANEL_ID_SCAN(&saved[0]),
             PANEL_ID_SCAN(&saved[1])) == 13) {
    panel_receipt(stored_key, saved, expected);
  } else {
    legacy = true;
    if (sscanf(
            actual,
            "{\"cache_key\":\"%64[0-9a-f]\",\"panels_gu.i8\":\"%64[0-9a-f]\","
            "\"colscale_gu.f32\":\"%64[0-9a-f]\",\"complete\":true}\n",
            stored_key, saved[0].sha256, saved[1].sha256) != 3) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "i8-panel invalid cache receipt: %s", path);
    }
    snprintf(expected, sizeof(expected),
             "{\"cache_key\":\"%s\",\"panels_gu.i8\":\"%s\",\"colscale_gu."
             "f32\":\"%s\",\"complete\":true}\n",
             stored_key, saved[0].sha256, saved[1].sha256);
  }
  if (strcmp(actual, expected) || strcmp(key, stored_key) ||
      strlen(saved[0].sha256) != 64 || strlen(saved[1].sha256) != 64) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "i8-panel cache receipt/key mismatch: %s", path);
  }
  bool changed = legacy;
  for (unsigned i = 0; i < 2; ++i) {
    snprintf(path, sizeof(path), "%s/%s", directory, panel_names[i]);
    IREE_RETURN_IF_ERROR(panel_payload(path, i ? SCALE_BYTES : PANEL_BYTES,
                                       legacy ? NULL : &saved[i], verify,
                                       &current[i], hashes));
    if (strcmp(saved[i].sha256, current[i].sha256)) {
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "i8-panel cache payload hash mismatch: %s", path);
    }
    changed |= !panel_identity_equal(&saved[i], &current[i]);
  }
  if (changed) {
    panel_receipt(key, current, expected);
    snprintf(path, sizeof(path), "%s/cache.json", directory);
    IREE_RETURN_IF_ERROR(panel_atomic_record(path, expected, directory));
  }
  return iree_ok_status();
}
static iree_status_t panel_open(const loom_runtime_sources_t* sources,
                                const loom_model_file_t* file,
                                iree_allocator_t allocator,
                                loom_runtime_i8_panel_cache_t** out) {
  struct timespec started, finished;
  clock_gettime(CLOCK_MONOTONIC, &started);
  unsigned source_hashes = 0, payload_hashes = 0;
  const char* verification = getenv("LOOM_DERIVE_VERIFY");
  bool verify = verification && !strcmp(verification, "1");
  char *source = NULL, *root = NULL;
  iree_status_t status = loom_runtime_path_resolve(
      sources->base_directory, file->transform.source, allocator, &source);
  if (iree_status_is_ok(status)) {
    status = loom_runtime_path_resolve(sources->base_directory,
                                       file->transform.cache, allocator, &root);
  }
  int fd = -1, lock = -1;
  void* data = MAP_FAILED;
  struct stat before = {0}, after;
  char key[65], directory[4096], path[4096], stage[4096] = {0};
  panel_tensor_t tensors[128] = {0};
  if (iree_status_is_ok(status)) {
    fd = open(source, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
      status = panel_io("open source", source);
    }
  }
  if (iree_status_is_ok(status) &&
      (fstat(fd, &before) || before.st_size < 24)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "i8-panel requires a complete GGUF source");
  }
  if (iree_status_is_ok(status)) {
    data = mmap(NULL, before.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
      status = panel_io("map source", source);
    }
  }
  if (iree_status_is_ok(status)) {
    status = panel_inspect(data, before.st_size, tensors);
  }
  if (iree_status_is_ok(status)) {
    status = panel_mkdirs(root);
  }
  if (iree_status_is_ok(status)) {
    char qmap[18000], qmap_digest[65], source_hash[65];
    panel_qmap(tensors, qmap);
    panel_digest(qmap, strlen(qmap), qmap_digest);
    status = panel_source_hash(source, root, data, fd, &before, qmap_digest,
                               verify, source_hash, &source_hashes);
    if (iree_status_is_ok(status)) {
      panel_key(source_hash, qmap, key);
    }
  }
  if (iree_status_is_ok(status)) {
    unsigned output =
        file->transform.output == LOOM_MODEL_TRANSFORM_I8_COLSCALE ? 1 : 0;
    char* declared = NULL;
    status = loom_runtime_path_resolve(sources->base_directory, file->path,
                                       allocator, &declared);
    if (iree_status_is_ok(status)) {
      int n = snprintf(path, sizeof(path), "%s/%s/%s", root, key,
                       panel_names[output]);
      if (n >= (int)sizeof(path) || strcmp(path, declared)) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "i8-panel path must match content key: %s", path);
      }
    }
    iree_allocator_free(allocator, declared);
  }
  if (iree_status_is_ok(status)) {
    status = panel_mkdirs(root);
  }
  if (iree_status_is_ok(status)) {
    if (snprintf(directory, sizeof(directory), "%s/%s", root, key) >=
            (int)sizeof(directory) ||
        snprintf(path, sizeof(path), "%s/%s.lock", root, key) >=
            (int)sizeof(path)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "i8-panel cache path too long");
    }
  }
  if (iree_status_is_ok(status)) {
    lock = open(path, O_CREAT | O_RDWR | O_CLOEXEC, 0664);
    if (lock < 0 || flock(lock, LOCK_EX)) {
      status = panel_io("lock", path);
    }
  }
  if (iree_status_is_ok(status)) {
    struct stat st;
    if (!stat(directory, &st)) {
      status = panel_validate(directory, key, verify, &payload_hashes);
      if (iree_status_is_ok(status)) {
        fprintf(stderr, "i8-panel reused %s\n", directory);
      }
    } else if (errno != ENOENT) {
      status = panel_io("stat", directory);
    } else {
      snprintf(stage, sizeof(stage), "%s/.%s.partial-XXXXXX", root, key);
      if (!mkdtemp(stage)) {
        status = panel_io("create stage", stage);
        stage[0] = 0;
      }
      if (iree_status_is_ok(status)) {
        fprintf(stderr, "i8-panel deriving %s\n", directory);
        status = panel_derive(data, tensors, stage, key, allocator);
      }
      if (iree_status_is_ok(status) &&
          (fstat(fd, &after) || !panel_same_stat(&before, &after))) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "i8-panel GGUF changed during derivation");
      }
      if (iree_status_is_ok(status)) {
        if (rename(stage, directory)) {
          status = panel_io("publish", directory);
        } else {
          stage[0] = 0;
          status = panel_sync_dir(root);
        }
      }
      if (iree_status_is_ok(status)) {
        fprintf(stderr, "i8-panel derived %s\n", directory);
      }
    }
  }
  struct stat named;
  if (iree_status_is_ok(status) && (fstat(fd, &after) || stat(source, &named) ||
                                    !panel_same_stat(&before, &after) ||
                                    !panel_same_stat(&before, &named))) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "i8-panel GGUF changed during cache validation");
  }
  clock_gettime(CLOCK_MONOTONIC, &finished);
  fprintf(stderr,
          "i8-panel validation_seconds=%.6f source_hashes=%u payload_hashes=%u "
          "verify=%u\n",
          (double)(finished.tv_sec - started.tv_sec) +
              (double)(finished.tv_nsec - started.tv_nsec) / 1e9,
          source_hashes, payload_hashes, verify);
  loom_runtime_i8_panel_cache_t* cache = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(allocator, sizeof(*cache), (void**)&cache);
  }
  if (cache) {
    cache->allocator = allocator;
    cache->source = file->transform.source;
    cache->cache = file->transform.cache;
    if (iree_status_is_ok(status)) {
      size_t n = strlen(directory) + 1;
      status = iree_allocator_malloc(allocator, n, (void**)&cache->directory);
      if (iree_status_is_ok(status)) {
        memcpy(cache->directory, directory, n);
      }
    }
    for (unsigned i = 0; i < 2 && iree_status_is_ok(status); ++i) {
      snprintf(path, sizeof(path), "%s/%s", directory, panel_names[i]);
      status = iree_io_file_handle_open(
          IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_SHARE_READ |
              IREE_IO_FILE_MODE_ASYNC,
          iree_make_cstring_view(path), allocator, &cache->handles[i]);
    }
  }
  if (stage[0]) {
    for (unsigned i = 0; i < 3; ++i) {
      snprintf(path, sizeof(path), "%s/%s", stage,
               i < 2 ? panel_names[i] : "cache.json");
      unlink(path);
    }
    rmdir(stage);
  }
  if (lock >= 0) {
    flock(lock, LOCK_UN);
    close(lock);
  }
  if (data != MAP_FAILED) {
    munmap(data, before.st_size);
  }
  if (fd >= 0) {
    close(fd);
  }
  iree_allocator_free(allocator, source);
  iree_allocator_free(allocator, root);
  if (iree_status_is_ok(status)) {
    *out = cache;
  } else {
    loom_runtime_i8_panel_cache_free(cache);
  }
  return status;
}

void loom_runtime_i8_panel_cache_free(loom_runtime_i8_panel_cache_t* cache) {
  if (!cache) {
    return;
  }
  for (unsigned i = 0; i < 2; ++i) {
    iree_io_file_handle_release(cache->handles[i]);
  }
  iree_allocator_free(cache->allocator, cache->directory);
  iree_allocator_free(cache->allocator, cache);
}

iree_status_t loom_runtime_i8_panel_append(
    const loom_runtime_sources_t* sources, const loom_model_file_t* file,
    iree_allocator_t allocator, loom_runtime_i8_panel_cache_t** cache,
    iree_io_parameter_index_t* index) {
  if (*cache &&
      (!iree_string_view_equal((*cache)->source, file->transform.source) ||
       !iree_string_view_equal((*cache)->cache, file->transform.cache))) {
    loom_runtime_i8_panel_cache_free(*cache);
    *cache = NULL;
  }
  if (!*cache) {
    IREE_RETURN_IF_ERROR(panel_open(sources, file, allocator, cache));
  }
  unsigned output =
      file->transform.output == LOOM_MODEL_TRANSFORM_I8_COLSCALE ? 1 : 0;
  char expected[4096];
  snprintf(expected, sizeof(expected), "%s/%s", (*cache)->directory,
           panel_names[output]);
  char* declared = NULL;
  IREE_RETURN_IF_ERROR(loom_runtime_path_resolve(
      sources->base_directory, file->path, allocator, &declared));
  bool matches = !strcmp(expected, declared);
  iree_allocator_free(allocator, declared);
  if (!matches) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "i8-panel path must match content key: %s",
                            expected);
  }
  unsigned count = output ? 128 : 2;
  uint64_t length = output ? PANEL_N * 4 : PANEL_BYTES / 2;
  for (unsigned t = 0; t < count; ++t) {
    char name[64];
    if (output) {
      snprintf(name, sizeof(name), "blk.%u.ffn_%s.npu_colscale", t / 2,
               t % 2 ? "up" : "gate");
    } else {
      snprintf(name, sizeof(name), "resident_half%u", t);
    }
    iree_io_parameter_index_entry_t entry = {
        .key = iree_make_cstring_view(name),
        .length = length,
        .type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE,
        .storage.file = {.handle = (*cache)->handles[output],
                         .offset = t * length}};
    IREE_RETURN_IF_ERROR(iree_io_parameter_index_add(index, &entry));
  }
  return iree_ok_status();
}
#else
void loom_runtime_i8_panel_cache_free(loom_runtime_i8_panel_cache_t* cache) {}
iree_status_t loom_runtime_i8_panel_append(
    const loom_runtime_sources_t* sources, const loom_model_file_t* file,
    iree_allocator_t allocator, loom_runtime_i8_panel_cache_t** cache,
    iree_io_parameter_index_t* index) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "i8-panel cache requires POSIX file locking");
}
#endif
