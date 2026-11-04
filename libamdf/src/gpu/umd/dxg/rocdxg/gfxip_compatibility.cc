// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shared/include/utils.h"

namespace wsl::thunk {
namespace {

constexpr GfxipTable kGfxipTable[] = {
    {0x7448, 11, 0, 0}, {0x744C, 11, 0, 0}, {0x745E, 11, 0, 0},
    {0x7449, 11, 0, 0}, {0x744A, 11, 0, 0}, {0x744B, 11, 0, 0},
    {0x7470, 11, 0, 1}, {0x747E, 11, 0, 1}, {0x7590, 12, 0, 0},
    {0x7550, 12, 0, 1}, {0x7551, 12, 0, 1}, {0x150E, 11, 5, 0},
    {0x1586, 11, 5, 1}, {0x1114, 11, 5, 2}, {0x1900, 11, 0, 3},
    {0x1902, 11, 5, 3},
};

const GfxipTable* FindGfxipEntry(uint16_t device_id) {
  for (const GfxipTable& entry : kGfxipTable) {
    if (entry.device_id == device_id) {
      return &entry;
    }
  }
  return nullptr;
}

}  // namespace

bool QueryAdapterSupported(unsigned int device_id) {
  return FindGfxipEntry(static_cast<uint16_t>(device_id)) != nullptr;
}

bool LookupGfxipEntry(uint16_t device_id, GfxipTable* out) {
  if (out == nullptr) {
    return false;
  }
  const GfxipTable* entry = FindGfxipEntry(device_id);
  if (entry == nullptr) {
    return false;
  }
  *out = *entry;
  return true;
}

}  // namespace wsl::thunk
