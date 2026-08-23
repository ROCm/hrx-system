// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IR_STRUCTURAL_HASH_H_
#define LOOM_IR_STRUCTURAL_HASH_H_

#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Incremental non-stable hash for compiler-owned structural keys.
//
// Values are mixed as semantic words instead of serial bytes. Byte spans are
// reserved for actual arrays and opaque payloads. Hashes are process-local
// table accelerators: callers must still compare keys for equality and must
// never serialize or otherwise rely on the numeric result remaining stable.

// Returns the initial state for a structural hash.
static inline uint32_t loom_structural_hash_initialize(void) {
  return 374761393u;
}

// Rotates one hash word left by a nonzero fixed amount.
static inline uint32_t loom_structural_hash_rotate_left(uint32_t value,
                                                        uint32_t amount) {
  return (value << amount) | (value >> (32u - amount));
}

// Mixes one 32-bit semantic word into |hash|.
static inline uint32_t loom_structural_hash_mix_u32(uint32_t hash,
                                                    uint32_t value) {
  hash += value * 3266489917u;
  hash = loom_structural_hash_rotate_left(hash, 17u);
  return hash * 668265263u;
}

// Mixes one 16-bit semantic word into |hash|.
static inline uint32_t loom_structural_hash_mix_u16(uint32_t hash,
                                                    uint16_t value) {
  return loom_structural_hash_mix_u32(hash, value);
}

// Mixes one 8-bit semantic word into |hash|.
static inline uint32_t loom_structural_hash_mix_u8(uint32_t hash,
                                                   uint8_t value) {
  return loom_structural_hash_mix_u32(hash, value);
}

// Mixes one 64-bit semantic word into |hash|.
static inline uint32_t loom_structural_hash_mix_u64(uint32_t hash,
                                                    uint64_t value) {
  hash = loom_structural_hash_mix_u32(hash, (uint32_t)value);
  return loom_structural_hash_mix_u32(hash, (uint32_t)(value >> 32));
}

// Mixes one byte span and its length into |hash| four bytes at a time.
static inline uint32_t loom_structural_hash_mix_bytes(uint32_t hash,
                                                      const void* data,
                                                      iree_host_size_t length) {
  hash = loom_structural_hash_mix_u64(hash, (uint64_t)length);
  const uint8_t* bytes = (const uint8_t*)data;
  while (length >= sizeof(uint32_t)) {
    uint32_t word = 0;
    memcpy(&word, bytes, sizeof(word));
    hash = loom_structural_hash_mix_u32(hash, word);
    bytes += sizeof(word);
    length -= sizeof(word);
  }
  if (length != 0) {
    uint32_t tail = 0;
    memcpy(&tail, bytes, length);
    hash = loom_structural_hash_mix_u32(hash, tail);
  }
  return hash;
}

// Finalizes a structural hash after all fields have been mixed.
static inline uint32_t loom_structural_hash_finalize(uint32_t hash) {
  hash ^= hash >> 15;
  hash *= 2246822519u;
  hash ^= hash >> 13;
  hash *= 3266489917u;
  hash ^= hash >> 16;
  return hash;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_IR_STRUCTURAL_HASH_H_
