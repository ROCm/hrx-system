// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/encoding.h"

//===----------------------------------------------------------------------===//
// Encoding equality and hashing
//===----------------------------------------------------------------------===//

static uint32_t loom_encoding_hash_bytes(const void* data,
                                         iree_host_size_t length,
                                         uint32_t hash) {
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < length; ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
}

bool loom_encoding_equal(const loom_encoding_t* a, const loom_encoding_t* b) {
  if (a->name_id != b->name_id) {
    return false;
  }
  if (a->attribute_count != b->attribute_count) {
    return false;
  }
  for (uint8_t i = 0; i < a->attribute_count; ++i) {
    if (a->attributes[i].name_id != b->attributes[i].name_id) {
      return false;
    }
    if (!loom_attribute_equal(&a->attributes[i].value,
                              &b->attributes[i].value)) {
      return false;
    }
  }
  return true;
}

uint32_t loom_encoding_hash(const loom_encoding_t* encoding) {
  uint32_t hash = 2166136261u;
  hash = loom_encoding_hash_bytes(&encoding->name_id, sizeof(encoding->name_id),
                                  hash);
  hash = loom_encoding_hash_bytes(&encoding->attribute_count,
                                  sizeof(encoding->attribute_count), hash);
  for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
    hash =
        loom_encoding_hash_bytes(&encoding->attributes[i].name_id,
                                 sizeof(encoding->attributes[i].name_id), hash);
    uint32_t value_hash = loom_attribute_hash(&encoding->attributes[i].value);
    hash = loom_encoding_hash_bytes(&value_hash, sizeof(value_hash), hash);
  }
  return hash;
}
