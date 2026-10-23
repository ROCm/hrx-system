// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "native_abi.h"

extern "C" Pair loom_pair(Pair value) {
  return {value.second + 7u, value.first * 3u};
}

extern "C" Mixed loom_mixed(Mixed value) {
  return {value.value * 2.0f + 0.25f, value.tag + 9u};
}

extern "C" Large loom_large(Large value) {
  return {value.third + 11ull, value.first + 13ull, value.second + 17ull};
}

extern "C" Words4 loom_words(Words4 value) {
  return value + Words4{1u, 2u, 3u, 4u};
}

extern "C" const unsigned* loom_advance(const unsigned* base) {
  return base + 3;
}

extern "C" Pair loom_call_native_pair(Pair value) { return native_pair(value); }

extern "C" Mixed loom_call_native_mixed(Mixed value) {
  return native_mixed(value);
}

extern "C" Large loom_call_native_large(Large value) {
  return native_large(value);
}

extern "C" Words4 loom_call_native_words(Words4 value) {
  return native_words(value + Words4{10u, 20u, 30u, 40u});
}

extern "C" const unsigned* loom_call_native_advance(const unsigned* base) {
  return native_advance(base + 1);
}
