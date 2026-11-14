// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/preamble.h"

#include <cxx/lexer.h>
#include <cxx/pch.h>
#include <cxx/preprocessor.h>

#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "iree/base/internal/atomics.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/frontend.h"

namespace {

using loom::cxx_import::SourceContentSnapshot;

std::string copy_string(iree_string_view_t value) {
  return value.size ? std::string(value.data, value.size) : std::string();
}

std::string_view string_view(iree_string_view_t value) {
  return std::string_view(value.data ? value.data : "", value.size);
}

void append_part(std::string& destination, std::string_view value) {
  destination.append(std::to_string(value.size()));
  destination.push_back(':');
  destination.append(value);
}

class FrontendIdentity {
 public:
  explicit FrontendIdentity(const loom_cxx_import_options_t& options)
      : standard_(copy_string(loom_cxx_import_options_standard(&options))),
        triple_(copy_string(options.triple)),
        data_model_(options.data_model),
        hosted_(iree_all_bits_set(options.flags, LOOM_CXX_IMPORT_FLAG_HOSTED)),
        builtin_includes_(!iree_any_bit_set(
            options.flags, LOOM_CXX_IMPORT_FLAG_NO_BUILTIN_INCLUDES)) {
    include_paths_.reserve(options.include_path_count);
    for (iree_host_size_t i = 0; i < options.include_path_count; ++i) {
      include_paths_.push_back(copy_string(options.include_paths[i]));
    }
    system_include_paths_.reserve(options.system_include_path_count);
    for (iree_host_size_t i = 0; i < options.system_include_path_count; ++i) {
      system_include_paths_.push_back(
          copy_string(options.system_include_paths[i]));
    }
    defines_.reserve(options.define_count);
    for (iree_host_size_t i = 0; i < options.define_count; ++i) {
      defines_.emplace_back(copy_string(options.defines[i].name),
                            copy_string(options.defines[i].value));
    }
  }

  bool matches(const loom_cxx_import_options_t& options) const {
    const auto standard =
        string_view(loom_cxx_import_options_standard(&options));
    const bool hosted =
        iree_all_bits_set(options.flags, LOOM_CXX_IMPORT_FLAG_HOSTED);
    const bool builtin_includes = !iree_any_bit_set(
        options.flags, LOOM_CXX_IMPORT_FLAG_NO_BUILTIN_INCLUDES);
    if (standard != standard_ || string_view(options.triple) != triple_ ||
        options.data_model != data_model_ || hosted != hosted_ ||
        builtin_includes != builtin_includes_ ||
        options.include_path_count != include_paths_.size() ||
        options.system_include_path_count != system_include_paths_.size() ||
        options.define_count != defines_.size()) {
      return false;
    }
    for (iree_host_size_t i = 0; i < options.include_path_count; ++i) {
      if (string_view(options.include_paths[i]) != include_paths_[i]) {
        return false;
      }
    }
    for (iree_host_size_t i = 0; i < options.system_include_path_count; ++i) {
      if (string_view(options.system_include_paths[i]) !=
          system_include_paths_[i]) {
        return false;
      }
    }
    for (iree_host_size_t i = 0; i < options.define_count; ++i) {
      if (string_view(options.defines[i].name) != defines_[i].first ||
          string_view(options.defines[i].value) != defines_[i].second) {
        return false;
      }
    }
    return true;
  }

  cxx::PrecompiledHeaderKeys keys() const {
    std::string target;
    append_part(target, triple_);
    target.append(std::to_string(data_model_));
    target.push_back(hosted_ ? 'h' : 'f');
    std::string options;
    options.push_back(builtin_includes_ ? '1' : '0');
    for (const auto& path : include_paths_) {
      append_part(options, path);
    }
    options.push_back('|');
    for (const auto& path : system_include_paths_) {
      append_part(options, path);
    }
    options.push_back('|');
    for (const auto& [name, value] : defines_) {
      append_part(options, name);
      append_part(options, value);
    }
    return {
        .serializationAbi = cxx::precompiledHeaderSerializationAbi(),
        .targetKey = std::move(target),
        .languageKey = standard_,
        .optionDigest = std::move(options),
    };
  }

 private:
  // Effective frontend language-standard spelling.
  std::string standard_;
  // Source ABI triple controlling predefined target facts.
  std::string triple_;
  // Source integer and pointer data model.
  loom_cxx_data_model_t data_model_;
  // Whether the frontend is configured as a hosted Linux compile.
  bool hosted_;
  // Whether the embedded include catalog participates in lookup.
  bool builtin_includes_;
  // Ordered user include search roots.
  std::vector<std::string> include_paths_;
  // Ordered system include search roots.
  std::vector<std::string> system_include_paths_;
  // Ordered preprocessor definitions and replacement text.
  std::vector<std::pair<std::string, std::string>> defines_;
};

bool is_spliced_newline(std::string_view source, size_t newline) {
  auto index = newline;
  if (index > 0 && source[index - 1] == '\r') {
    --index;
  }
  return index > 0 && source[index - 1] == '\\';
}

size_t end_of_directive_line(std::string_view source, size_t offset) {
  if (offset == 0 || offset >= source.size()) {
    return offset;
  }
  if (source[offset - 1] == '\n') {
    return offset;
  }
  while (offset < source.size()) {
    const auto newline = source.find('\n', offset);
    if (newline == std::string_view::npos) {
      break;
    }
    if (!is_spliced_newline(source, newline)) {
      return newline + 1;
    }
    offset = newline + 1;
  }
  return source.size();
}

void mask_preamble(std::string& source, size_t size) {
  const auto end = end_of_directive_line(source, size);
  for (size_t i = 0; i < end; ++i) {
    if (source[i] != '\n' && source[i] != '\r') {
      source[i] = ' ';
    }
  }
}

std::string writer_error(const cxx::PrecompiledHeaderWriter& writer) {
  std::string message = "C/C++ source prefix cannot be serialized";
  for (const auto& error : writer.errors()) {
    message.append(": ");
    message.append(error);
  }
  return message;
}

}  // namespace

struct loom_cxx_preamble_t {
  loom_cxx_preamble_t(
      iree_allocator_t allocator, FrontendIdentity frontend_identity,
      std::string source_prefix, std::vector<uint8_t> archive,
      std::vector<SourceContentSnapshot> source_snapshot) noexcept
      : host_allocator(allocator),
        identity(std::move(frontend_identity)),
        prefix(std::move(source_prefix)),
        bytes(std::move(archive)),
        sources(std::move(source_snapshot)) {
    iree_atomic_ref_count_init(&ref_count);
  }

  // Shared by caller catalogs and concurrent compilation tasks.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning this handle allocation.
  iree_allocator_t host_allocator;
  // Exact frontend options defining the serialized semantic prefix.
  FrontendIdentity identity;
  // Exact leading primary-source bytes required for restoration.
  std::string prefix;
  // Immutable CXX precompiled-header archive.
  std::vector<uint8_t> bytes;
  // Sorted immutable prefix include hits and misses.
  std::vector<SourceContentSnapshot> sources;
};

iree_status_t loom_cxx_preamble_create(iree_string_view_t source,
                                       iree_string_view_t filename,
                                       const loom_cxx_import_options_t* options,
                                       iree_allocator_t host_allocator,
                                       loom_cxx_preamble_t** out_preamble) {
  IREE_ASSERT_ARGUMENT(out_preamble);
  *out_preamble = nullptr;
  loom_cxx_import_options_t defaults;
  if (!options) {
    loom_cxx_import_options_initialize(&defaults);
    options = &defaults;
  }
  IREE_RETURN_IF_ERROR(loom_cxx_import_options_validate(options));
  using namespace loom::cxx_import;
  try {
    auto producer_options = *options;
    producer_options.preamble = nullptr;
    Frontend producer(filename, producer_options);
    producer.preprocess(source, /*preamble_only=*/true);
    const auto preamble_size = producer.unit().preprocessor()->preambleSize();
    if (!preamble_size || !*preamble_size) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "C/C++ source has no reusable leading include preamble");
    }
    producer.parse();
    FrontendIdentity identity(producer_options);
    cxx::PrecompiledHeaderWriter writer(&producer.unit(), identity.keys());
    writer.setPreprocessorState(
        producer.unit().preprocessor()->preambleState());
    auto bytes = writer();
    if (bytes.empty()) {
      auto message = writer_error(writer);
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "%s",
                              message.c_str());
    }
    std::string prefix(source.data, *preamble_size);
    auto sources = producer.contents().take_snapshot();
    void* storage = nullptr;
    check(iree_allocator_malloc(host_allocator, sizeof(loom_cxx_preamble_t),
                                &storage));
    *out_preamble = new (storage) loom_cxx_preamble_t(
        host_allocator, std::move(identity), std::move(prefix),
        std::move(bytes), std::move(sources));
    return iree_ok_status();
  } catch (StatusError& error) {
    return error.release();
  } catch (const SourceRejected&) {
    return iree_ok_status();
  } catch (const std::bad_alloc&) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "C/C++ preamble allocation failed");
  } catch (const std::exception& error) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "C/C++ preamble creation failed: %s", error.what());
  }
}

void loom_cxx_preamble_retain(loom_cxx_preamble_t* preamble) {
  if (preamble) {
    iree_atomic_ref_count_inc(&preamble->ref_count);
  }
}

void loom_cxx_preamble_release(loom_cxx_preamble_t* preamble) {
  if (preamble && iree_atomic_ref_count_dec(&preamble->ref_count) == 1) {
    const iree_allocator_t host_allocator = preamble->host_allocator;
    preamble->~loom_cxx_preamble_t();
    iree_allocator_free(host_allocator, preamble);
  }
}

namespace loom::cxx_import {

bool restore_preamble(const loom_cxx_preamble_t* preamble,
                      iree_string_view_t source,
                      const loom_cxx_import_options_t& options,
                      Frontend& frontend) {
  if (!preamble || !preamble->identity.matches(options)) {
    return false;
  }
  const auto source_view = string_view(source);
  if (!source_view.starts_with(preamble->prefix)) {
    return false;
  }
  const auto suffix = source_view.substr(preamble->prefix.size());
  if (!suffix.empty()) {
    cxx::Lexer lexer(suffix);
    lexer.setPreprocessing(true);
    if (lexer.next() == cxx::TokenKind::T_HASH) {
      return false;
    }
  }
  cxx::PrecompiledHeaderReader reader(&frontend.unit(),
                                      preamble->identity.keys());
  if (!reader(preamble->bytes)) {
    throw std::logic_error("immutable C/C++ preamble failed to restore: " +
                           reader.error());
  }
  frontend.contents().adopt(preamble->sources);
  frontend.contents().retain_primary(string(source));
  std::string masked_source(source_view);
  mask_preamble(masked_source, preamble->prefix.size());
  frontend.preprocess(view(masked_source));
  frontend.resume();
  return true;
}

}  // namespace loom::cxx_import
