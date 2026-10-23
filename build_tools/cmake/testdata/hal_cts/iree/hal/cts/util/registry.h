// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The build fixture supplies only the registry dependency. Embedding, source
// generation, compilation, and transitive registration retention use the real
// build rules. The runtime testdata_test exercises the actual CTS registry.
#pragma once

#include <cstdint>
#include <string_view>

using iree_string_view_t = std::string_view;
struct iree_const_byte_span_t {
  // Borrowed embedded bytes.
  const uint8_t* data;
  // Number of embedded bytes.
  size_t data_length;
};
inline bool iree_string_view_equal(iree_string_view_t lhs,
                                   iree_string_view_t rhs) {
  return lhs == rhs;
}
inline iree_string_view_t iree_make_cstring_view(const char* value) {
  return value;
}
inline iree_const_byte_span_t iree_make_const_byte_span(const uint8_t* data,
                                                        size_t size) {
  return {data, size};
}
inline iree_const_byte_span_t iree_const_byte_span_empty() { return {}; }

namespace iree::hal::cts {
struct ExecutableTarget {
  // GTest-compatible name.
  const char* name;
  // Executable family.
  const char* family;
  // Family-owned artifact identity.
  const char* target_key;
  // Looks up bytes by logical name.
  iree_const_byte_span_t (*data_fn)(iree_string_view_t);
};
struct CtsRegistry {
  static void RegisterExecutableTarget(const char* backend,
                                       ExecutableTarget target);
};
}  // namespace iree::hal::cts
