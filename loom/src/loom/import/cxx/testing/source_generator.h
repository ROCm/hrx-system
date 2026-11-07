// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_TESTING_SOURCE_GENERATOR_H_
#define LOOM_IMPORT_CXX_TESTING_SOURCE_GENERATOR_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace loom::cxx_import::testing {

// Maximum byte count consumed from one fuzzer mutation.
inline constexpr size_t kMaximumMutationByteCount = 64;

// Structural ceiling for generated translation units.
inline constexpr size_t kMaximumGeneratedSourceByteCount = 8192;

// Generates one bounded, semantically valid C++ translation unit.
std::string GenerateSource(std::span<const uint8_t> bytes);

}  // namespace loom::cxx_import::testing

#endif  // LOOM_IMPORT_CXX_TESTING_SOURCE_GENERATOR_H_
