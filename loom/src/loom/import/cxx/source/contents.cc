// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/contents.h"

#include <cxx/preprocessor.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "loom/import/cxx/source/catalog.h"
#include "loom/import/cxx/source/error.h"

namespace loom::cxx_import {
namespace {

// Advances by the same UTF-8 code-point columns used by the CXX source map.
size_t advance_columns(std::string_view source, size_t offset, size_t count) {
  while (count-- && offset < source.size()) {
    const auto lead = static_cast<uint8_t>(source[offset]);
    size_t length = 1;
    if ((lead >> 5) == 0x06) {
      length = 2;
    } else if ((lead >> 4) == 0x0e) {
      length = 3;
    } else if ((lead >> 3) == 0x1e) {
      length = 4;
    }
    size_t candidate = offset + 1;
    bool complete = length == 1;
    for (size_t index = 1; index < length && candidate < source.size();
         ++index) {
      if ((static_cast<uint8_t>(source[candidate]) & 0xc0) != 0x80) {
        break;
      }
      ++candidate;
      complete = index + 1 == length;
    }
    offset = complete ? candidate : offset + 1;
  }
  return offset;
}

}  // namespace

SourceContents::SourceContents(const loom_cxx_import_options_t& options)
    : provider_(options.source_provider),
      builtin_root_(iree_any_bit_set(options.flags,
                                     LOOM_CXX_IMPORT_FLAG_NO_BUILTIN_INCLUDES)
                        ? std::string_view()
                        : builtin_include_root()) {}

void SourceContents::bind(cxx::Preprocessor* preprocessor,
                          std::string primary_filename) {
  preprocessor_ = preprocessor;
  primary_filename_ = std::move(primary_filename);
}

void SourceContents::retain_primary(std::string source) {
  primary_source_ = std::move(source);
}

void SourceContents::adopt(std::span<const SourceContentSnapshot> snapshot) {
  snapshot_ = snapshot;
}

std::vector<SourceContentSnapshot> SourceContents::take_snapshot() {
  std::vector<SourceContentSnapshot> result;
  result.reserve(entries_.size());
  for (auto& [filename, entry] : entries_) {
    SourceContentSnapshot snapshot = {
        .filename = filename,
        .source = std::move(entry.source),
    };
    if (snapshot.source) {
      build_line_offsets(*snapshot.source, snapshot.line_offsets);
    }
    result.push_back(std::move(snapshot));
  }
  entries_.clear();
  std::ranges::sort(result, {}, &SourceContentSnapshot::filename);
  return result;
}

const std::optional<std::string>& SourceContents::resolve(
    const std::string& path) {
  if (auto found = entries_.find(path); found != entries_.end()) {
    return found->second.source;
  }
  if (const auto* snapshot = snapshot_entry(path)) {
    return snapshot->source;
  }
  auto entry = entries_.try_emplace(path).first;
  if (!builtin_root_.empty() && path.starts_with(builtin_root_) &&
      path.size() > builtin_root_.size() && path[builtin_root_.size()] == '/') {
    auto contents = builtin_include(
        std::string_view(path).substr(builtin_root_.size() + 1));
    if (contents) {
      entry->second.source = *contents;
    }
    return entry->second.source;
  }
  if (provider_.fn) {
    bool found = false;
    iree_string_view_t contents = iree_string_view_empty();
    check(provider_.fn(provider_.user_data, view(path), &found, &contents));
    if (found) {
      entry->second.source = string(contents);
    }
    return entry->second.source;
  }
  std::error_code error;
  auto status = std::filesystem::status(path, error);
  if (error == std::errc::no_such_file_or_directory ||
      error == std::errc::not_a_directory) {
    return entry->second.source;
  }
  if (error) {
    throw StatusError(iree_make_status(IREE_STATUS_UNAVAILABLE,
                                       "cannot inspect include '%s': %s",
                                       path.c_str(), error.message().c_str()));
  }
  if (!std::filesystem::is_regular_file(status)) {
    return entry->second.source;
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw StatusError(iree_make_status(
        IREE_STATUS_UNAVAILABLE, "cannot open include '%s'", path.c_str()));
  }
  entry->second.source.emplace(std::istreambuf_iterator<char>(input),
                               std::istreambuf_iterator<char>());
  if (input.bad()) {
    throw StatusError(iree_make_status(
        IREE_STATUS_DATA_LOSS, "cannot read include '%s'", path.c_str()));
  }
  return entry->second.source;
}

std::string_view SourceContents::source(std::string_view filename) const {
  if (filename == primary_filename_) {
    if (primary_source_) {
      return *primary_source_;
    }
    if (!preprocessor_) {
      throw std::logic_error("C/C++ primary source is not bound");
    }
    return preprocessor_->source(preprocessor_->mainSourceFileId());
  }
  auto found = entries_.find(filename);
  if (found != entries_.end() && found->second.source) {
    return *found->second.source;
  }
  if (const auto* snapshot = snapshot_entry(filename);
      snapshot && snapshot->source) {
    return *snapshot->source;
  }
  throw std::logic_error("C/C++ physical source was not retained: " +
                         std::string(filename));
}

const SourceContentSnapshot* SourceContents::snapshot_entry(
    std::string_view filename) const {
  auto found = std::ranges::lower_bound(snapshot_, filename, {},
                                        &SourceContentSnapshot::filename);
  if (found == snapshot_.end() || found->filename != filename) {
    return nullptr;
  }
  return &*found;
}

void SourceContents::build_line_offsets(std::string_view source,
                                        std::vector<size_t>& line_offsets) {
  line_offsets.push_back(0);
  for (size_t index = 0; index < source.size(); ++index) {
    if (source[index] == '\n') {
      line_offsets.push_back(index + 1);
    }
  }
}

size_t SourceContents::offset(std::string_view source,
                              const std::vector<size_t>& line_offsets,
                              const cxx::SourcePosition& position) {
  if (!position.line || !position.column ||
      position.line > line_offsets.size()) {
    throw std::logic_error("C/C++ physical source position is invalid");
  }
  const size_t line_offset = line_offsets[position.line - 1];
  const size_t result =
      advance_columns(source, line_offset, position.column - 1);
  const size_t line_end = position.line < line_offsets.size()
                              ? line_offsets[position.line]
                              : source.size();
  if (result > line_end) {
    throw std::logic_error("C/C++ physical source column is invalid");
  }
  return result;
}

PhysicalSourceRange SourceContents::range(
    const cxx::SourcePosition& first, const cxx::SourcePosition& last) const {
  if (first.fileName != last.fileName) {
    throw std::logic_error("C/C++ physical source range crosses files");
  }
  if (first.fileName == primary_filename_) {
    auto contents = source(first.fileName);
    if (primary_line_offsets_.empty()) {
      build_line_offsets(contents, primary_line_offsets_);
    }
    return {
        .source = contents,
        .start = offset(contents, primary_line_offsets_, first),
        .end = offset(contents, primary_line_offsets_, last),
    };
  }
  if (auto found = entries_.find(first.fileName);
      found != entries_.end() && found->second.source) {
    const auto& retained = found->second;
    auto contents = std::string_view(*retained.source);
    if (retained.line_offsets.empty()) {
      build_line_offsets(contents, retained.line_offsets);
    }
    return {
        .source = contents,
        .start = offset(contents, retained.line_offsets, first),
        .end = offset(contents, retained.line_offsets, last),
    };
  }
  const auto* snapshot = snapshot_entry(first.fileName);
  if (!snapshot || !snapshot->source) {
    throw std::logic_error("C/C++ physical source was not retained: " +
                           std::string(first.fileName));
  }
  auto contents = std::string_view(*snapshot->source);
  return {
      .source = contents,
      .start = offset(contents, snapshot->line_offsets, first),
      .end = offset(contents, snapshot->line_offsets, last),
  };
}

}  // namespace loom::cxx_import
