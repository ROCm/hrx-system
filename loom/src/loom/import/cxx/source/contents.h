// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SOURCE_CONTENTS_H_
#define LOOM_IMPORT_CXX_SOURCE_CONTENTS_H_

#include <cxx/source_location.h>

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "loom/import/cxx/source/options.h"

namespace cxx {
class Preprocessor;
}  // namespace cxx

namespace loom::cxx_import {

inline iree_string_view_t view(std::string_view value) {
  return iree_make_string_view(value.data(), value.size());
}

inline std::string string(iree_string_view_t value) {
  return value.size ? std::string(value.data, value.size) : std::string();
}

// Exact bytes and byte offsets for one physical C/C++ source range.
struct PhysicalSourceRange {
  // Complete retained source containing the range.
  std::string_view source;
  // Byte offset of the first source position.
  size_t start = 0;
  // Byte offset one past the final source position.
  size_t end = 0;
};

// One immutable include lookup captured while constructing a preamble.
struct SourceContentSnapshot {
  // Exact candidate path queried by the preprocessor.
  std::string filename;
  // Owned source bytes, or nullopt for a retained missing candidate.
  std::optional<std::string> source;
  // Byte offsets of physical line starts, populated for found sources.
  std::vector<size_t> line_offsets;
};

// Owns every include result resolved for one frontend invocation and provides
// exact physical source bytes independently of frontend token ownership.
// Missing include results remain cached so repeated search and __has_include
// queries observe one provider snapshot.
class SourceContents {
 public:
  explicit SourceContents(const loom_cxx_import_options_t& options);

  SourceContents(const SourceContents&) = delete;
  SourceContents& operator=(const SourceContents&) = delete;

  // Binds the frontend source store and physical primary filename. Both outlive
  // this object and become the source of primary-file bytes.
  void bind(cxx::Preprocessor* preprocessor, std::string primary_filename);

  // Retains original primary bytes when the frontend consumes a transformed
  // copy, such as a source with its restored preamble masked.
  void retain_primary(std::string source);

  // Borrows one immutable prefix snapshot for this frontend lifetime. Entries
  // retain both successful and missing lookups from the producing invocation.
  void adopt(std::span<const SourceContentSnapshot> snapshot);

  // Moves all invocation-local include results into a sorted immutable
  // snapshot. The producing frontend no longer resolves source bytes afterward.
  std::vector<SourceContentSnapshot> take_snapshot();

  // Returns the retained result for an include candidate, resolving it once.
  const std::optional<std::string>& resolve(const std::string& path);

  // Returns exact retained bytes for a physical filename. Every filename
  // projected from a parsed source location must have been retained here.
  std::string_view source(std::string_view filename) const;

  // Maps physical one-based UTF-8 line/column positions back to byte offsets.
  PhysicalSourceRange range(const cxx::SourcePosition& first,
                            const cxx::SourcePosition& last) const;

  // Virtual system include root, empty when embedded includes are disabled.
  std::string_view builtin_root() const { return builtin_root_; }
  // Physical filename supplied for the primary source.
  std::string_view primary_filename() const { return primary_filename_; }

 private:
  struct TransparentStringHash {
    using is_transparent = void;
    size_t operator()(std::string_view value) const {
      return std::hash<std::string_view>{}(value);
    }
  };

  struct Entry {
    // Owned provider or filesystem result, empty for a retained miss.
    std::optional<std::string> source;
    // Lazily computed byte offsets of physical line starts.
    mutable std::vector<size_t> line_offsets;
  };

  const SourceContentSnapshot* snapshot_entry(std::string_view filename) const;
  static void build_line_offsets(std::string_view source,
                                 std::vector<size_t>& line_offsets);
  static size_t offset(std::string_view source,
                       const std::vector<size_t>& line_offsets,
                       const cxx::SourcePosition& position);

  // Borrowed provider state for this invocation.
  loom_cxx_source_provider_t provider_;
  // Enabled virtual root for this invocation, empty for external-only lookup.
  std::string_view builtin_root_;
  // Frontend-owned source storage used for the primary file.
  cxx::Preprocessor* preprocessor_ = nullptr;
  // Physical primary filename corresponding to the frontend's main source.
  std::string primary_filename_;
  // Original primary bytes when the frontend source was transformed.
  std::optional<std::string> primary_source_;
  // Lazily computed byte offsets of primary-source line starts.
  mutable std::vector<size_t> primary_line_offsets_;
  // Candidate path to immutable source bytes, or a retained miss.
  std::unordered_map<std::string, Entry, TransparentStringHash, std::equal_to<>>
      entries_;
  // Borrowed sorted prefix lookups owned by an immutable preamble handle.
  std::span<const SourceContentSnapshot> snapshot_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_SOURCE_CONTENTS_H_
