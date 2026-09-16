// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_HIP_HIP_DSO_TEST_UTIL_H_
#define LIBHRX_SRC_BINDING_HIP_HIP_DSO_TEST_UTIL_H_

#include <string>

namespace hrx::hip::testing {

// Owns an RTLD_LOCAL handle to the exact HIP DSO produced by the current build.
class HipDso {
 public:
  HipDso() = default;
  ~HipDso();

  HipDso(const HipDso&) = delete;
  HipDso& operator=(const HipDso&) = delete;

  // Opens and verifies the configured build artifact.
  bool Open();

  // Closes the DSO. Returns false when the loader rejects the close request.
  bool Close();

  // Resolves exported |name| and verifies that the symbol belongs to this DSO.
  template <typename T>
  T Resolve(const char* name) {
    return reinterpret_cast<T>(ResolveRaw(name));
  }

  // Resolves an unexported test-control symbol from this exact unstripped DSO.
  // This does not widen the DSO symbol surface under test.
  template <typename T>
  T ResolveLocalForTest(const char* name) {
    return reinterpret_cast<T>(ResolveLocalForTestRaw(name));
  }

  // True while this object owns a loader reference.
  bool is_open() const { return handle_ != nullptr; }

  // Diagnostic for the most recent failed operation.
  const std::string& error() const { return error_; }

 private:
  void* ResolveRaw(const char* name);
  void* ResolveLocalForTestRaw(const char* name);

  // Handle returned by dlopen for the configured artifact.
  void* handle_ = nullptr;
  // Canonical path used to verify both the handle and resolved symbols.
  std::string canonical_path_;
  // Diagnostic retained when an operation fails.
  std::string error_;
};

}  // namespace hrx::hip::testing

#endif  // LIBHRX_SRC_BINDING_HIP_HIP_DSO_TEST_UTIL_H_
