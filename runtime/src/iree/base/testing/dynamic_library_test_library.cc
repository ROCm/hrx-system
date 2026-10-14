// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#if defined(_WIN32)
#define IREE_SYM_EXPORT __declspec(dllexport)
#define IREE_SYM_NOINLINE __declspec(noinline)
#else
#define IREE_SYM_EXPORT __attribute__((visibility("default")))
#define IREE_SYM_NOINLINE __attribute__((noinline))
#endif  // _WIN32

// Private code supplies a symbol and source line absent from the DLL exports.
static IREE_SYM_NOINLINE int private_times_two(int value) { return value * 2; }

int IREE_SYM_EXPORT times_two(int value) { return private_times_two(value); }

typedef int (*private_function_t)(int);
private_function_t IREE_SYM_EXPORT get_private_function() {
  return private_times_two;
}

int IREE_SYM_EXPORT next_value() {
  static int value = 0;
  return ++value;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
