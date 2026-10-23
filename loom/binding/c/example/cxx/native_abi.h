// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_BINDING_C_EXAMPLE_CXX_NATIVE_ABI_H_
#define LOOM_BINDING_C_EXAMPLE_CXX_NATIVE_ABI_H_

struct Pair {
  unsigned first;
  unsigned second;
};

struct Mixed {
  float value;
  unsigned tag;
};

struct Large {
  unsigned long long first;
  unsigned long long second;
  unsigned long long third;
};

using Words4 = unsigned __attribute__((vector_size(16)));

// Definitions compiled by Loom and called from the native translation unit.
extern "C" Pair loom_pair(Pair value);
extern "C" Mixed loom_mixed(Mixed value);
extern "C" Large loom_large(Large value);
extern "C" Words4 loom_words(Words4 value);
extern "C" const unsigned* loom_advance(const unsigned* base);

// Definitions compiled by the native toolchain and called from Loom code.
extern "C" Pair native_pair(Pair value);
extern "C" Mixed native_mixed(Mixed value);
extern "C" Large native_large(Large value);
extern "C" Words4 native_words(Words4 value);
extern "C" const unsigned* native_advance(const unsigned* base);

// Loom definitions that exercise calls into the native translation unit.
extern "C" Pair loom_call_native_pair(Pair value);
extern "C" Mixed loom_call_native_mixed(Mixed value);
extern "C" Large loom_call_native_large(Large value);
extern "C" Words4 loom_call_native_words(Words4 value);
extern "C" const unsigned* loom_call_native_advance(const unsigned* base);

#endif  // LOOM_BINDING_C_EXAMPLE_CXX_NATIVE_ABI_H_
