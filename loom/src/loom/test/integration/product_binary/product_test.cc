// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include "iree/base/api.h"
#include "iree/base/internal/json.h"

namespace {

struct Options {
  std::filesystem::path primary_file;
  std::string primary_magic;
  std::filesystem::path manifest_file;
  std::filesystem::path artifact_directory;
  std::filesystem::path kernel_file;
  std::string kernel_magic;
  std::filesystem::path dependency_report_file;
  std::string dependency_component;
};

std::optional<std::string_view> ParseValue(std::string_view argument,
                                           std::string_view name) {
  const std::string prefix = "--" + std::string(name) + "=";
  if (argument.size() < prefix.size() ||
      argument.substr(0, prefix.size()) != prefix) {
    return std::nullopt;
  }
  return argument.substr(prefix.size());
}

bool ParseOptions(int argc, char** argv, Options& options) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument = argv[i];
    if (auto value = ParseValue(argument, "primary")) {
      options.primary_file = *value;
    } else if (auto value = ParseValue(argument, "primary_magic")) {
      options.primary_magic = *value;
    } else if (auto value = ParseValue(argument, "manifest")) {
      options.manifest_file = *value;
    } else if (auto value = ParseValue(argument, "artifacts")) {
      options.artifact_directory = *value;
    } else if (auto value = ParseValue(argument, "kernel")) {
      options.kernel_file = *value;
    } else if (auto value = ParseValue(argument, "kernel_magic")) {
      options.kernel_magic = *value;
    } else if (auto value = ParseValue(argument, "dependency_report")) {
      options.dependency_report_file = *value;
    } else if (auto value = ParseValue(argument, "dependency_component")) {
      options.dependency_component = *value;
    } else {
      std::cerr << "Unknown product test argument: " << argument << "\n";
      return false;
    }
  }
  if (options.primary_file.empty()) {
    std::cerr << "Product test requires --primary.\n";
    return false;
  }
  return true;
}

bool RequireNonemptyFile(const std::filesystem::path& path,
                         std::string_view description) {
  std::error_code error;
  const auto size = std::filesystem::file_size(path, error);
  if (error || size == 0) {
    std::cerr << "Missing or empty " << description << ": " << path << " ("
              << error.message() << ")\n";
    return false;
  }
  return true;
}

bool RequireMagic(const std::filesystem::path& path, std::string_view expected,
                  std::string_view description) {
  if (!RequireNonemptyFile(path, description)) return false;
  std::ifstream stream(path, std::ios::binary);
  std::array<uint8_t, 4> bytes = {};
  stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (stream.gcount() != static_cast<std::streamsize>(bytes.size())) {
    std::cerr << description << " is shorter than four bytes: " << path << "\n";
    return false;
  }
  constexpr char kHexDigits[] = "0123456789abcdef";
  std::string actual;
  actual.reserve(bytes.size() * 2);
  for (uint8_t byte : bytes) {
    actual.push_back(kHexDigits[byte >> 4]);
    actual.push_back(kHexDigits[byte & 0x0F]);
  }
  if (actual != expected) {
    std::cerr << description << " has magic " << actual << "; expected "
              << expected << ": " << path << "\n";
    return false;
  }
  return true;
}

std::optional<std::string> ReadFile(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) return std::nullopt;
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

bool RequireOk(iree_status_t status, std::string_view description) {
  if (iree_status_is_ok(status)) return true;
  std::cerr << description << ":\n";
  iree_status_fprint(stderr, status);
  iree_status_free(status);
  return false;
}

bool LookupString(iree_string_view_t object, iree_string_view_t key,
                  std::string& out_value) {
  std::array<char, 256> storage = {};
  iree_host_size_t length = 0;
  if (!RequireOk(iree_json_lookup_string(object, key,
                                         iree_make_mutable_string_view(
                                             storage.data(), storage.size()),
                                         &length),
                 "Invalid JSON string field")) {
    return false;
  }
  out_value.assign(storage.data(), length);
  return true;
}

bool LookupUint64(iree_string_view_t object, iree_string_view_t key,
                  uint64_t& out_value) {
  iree_string_view_t value = iree_string_view_empty();
  return RequireOk(iree_json_lookup_object_value(object, key, &value),
                   "Missing JSON integer field") &&
         RequireOk(iree_json_parse_uint64(value, &out_value),
                   "Invalid JSON integer field");
}

bool LookupArray(iree_string_view_t object, iree_string_view_t key,
                 iree_host_size_t expected_length,
                 iree_string_view_t& out_array) {
  if (!RequireOk(iree_json_lookup_object_value(object, key, &out_array),
                 "Missing JSON array")) {
    return false;
  }
  iree_host_size_t length = 0;
  if (!RequireOk(iree_json_array_length(out_array, &length),
                 "Invalid JSON array")) {
    return false;
  }
  if (length != expected_length) {
    std::cerr << "Command manifest array has " << length
              << " elements; expected " << expected_length << ".\n";
    return false;
  }
  return true;
}

bool RequireDependencyReport(const Options& options) {
  if (!RequireNonemptyFile(options.dependency_report_file,
                           "dependency report")) {
    return false;
  }
  const auto report = ReadFile(options.dependency_report_file);
  if (!report) {
    std::cerr << "Cannot read dependency report: "
              << options.dependency_report_file << "\n";
    return false;
  }

  iree_string_view_t remaining =
      iree_make_string_view(report->data(), report->size());
  iree_string_view_t root = iree_string_view_empty();
  if (!RequireOk(iree_json_consume_object(&remaining, &root),
                 "Invalid dependency report") ||
      !RequireOk(iree_json_consume_insignificant(&remaining),
                 "Invalid dependency report trailing content") ||
      !iree_string_view_is_empty(remaining)) {
    std::cerr << "Dependency report has trailing content: "
              << options.dependency_report_file << "\n";
    return false;
  }

  uint64_t schema_version = 0;
  std::string component;
  bool succeeded = false;
  if (!LookupUint64(root, IREE_SV("schema_version"), schema_version) ||
      schema_version != 1 ||
      !LookupString(root, IREE_SV("component"), component) ||
      !RequireOk(iree_json_lookup_bool(root, IREE_SV("succeeded"), &succeeded),
                 "Invalid dependency report succeeded field") ||
      !succeeded) {
    std::cerr << "Dependency report has an unexpected schema: "
              << options.dependency_report_file << "\n";
    return false;
  }
  if (component != options.dependency_component) {
    std::cerr << "Dependency report component is " << component << "; expected "
              << options.dependency_component << ".\n";
    return false;
  }
  return true;
}

bool RequireManifest(const Options& options) {
  if (!RequireNonemptyFile(options.manifest_file, "command manifest")) {
    return false;
  }
  const auto manifest = ReadFile(options.manifest_file);
  if (!manifest) {
    std::cerr << "Cannot read command manifest: " << options.manifest_file
              << "\n";
    return false;
  }

  iree_string_view_t remaining =
      iree_make_string_view(manifest->data(), manifest->size());
  iree_string_view_t root = iree_string_view_empty();
  if (!RequireOk(iree_json_consume_object(&remaining, &root),
                 "Invalid command manifest") ||
      !RequireOk(iree_json_consume_insignificant(&remaining),
                 "Invalid command manifest trailing content") ||
      !iree_string_view_is_empty(remaining)) {
    std::cerr << "Command manifest has trailing content: "
              << options.manifest_file << "\n";
    return false;
  }

  uint64_t schema_version = 0;
  std::string format;
  if (!LookupUint64(root, IREE_SV("schema_version"), schema_version) ||
      schema_version != 2 || !LookupString(root, IREE_SV("format"), format) ||
      format != "loom-command-set") {
    std::cerr << "Command manifest has an unexpected schema: "
              << options.manifest_file << "\n";
    return false;
  }

  iree_string_view_t programs = iree_string_view_empty();
  iree_string_view_t program = iree_string_view_empty();
  std::string program_symbol;
  std::string program_artifact;
  uint64_t program_byte_length = 0;
  if (!LookupArray(root, IREE_SV("programs"), 1, programs) ||
      !RequireOk(iree_json_array_get(programs, 0, &program),
                 "Invalid command manifest program") ||
      !LookupString(program, IREE_SV("symbol"), program_symbol) ||
      program_symbol != "product_command" ||
      !LookupString(program, IREE_SV("artifact"), program_artifact) ||
      program_artifact != "program-0.loomcmd" ||
      !LookupUint64(program, IREE_SV("byte_length"), program_byte_length)) {
    std::cerr << "Command manifest does not describe product_command: "
              << options.manifest_file << "\n";
    return false;
  }

  iree_string_view_t requirements = iree_string_view_empty();
  iree_string_view_t requirement = iree_string_view_empty();
  uint64_t requirement_ordinal = 0;
  if (!LookupArray(program, IREE_SV("entry_requirements"), 1, requirements) ||
      !RequireOk(iree_json_array_get(requirements, 0, &requirement),
                 "Invalid command entry requirement") ||
      !RequireOk(iree_json_parse_uint64(requirement, &requirement_ordinal),
                 "Invalid command entry requirement") ||
      requirement_ordinal != 0) {
    std::cerr << "Command program has an unexpected entry requirement.\n";
    return false;
  }

  iree_string_view_t entries = iree_string_view_empty();
  iree_string_view_t entry = iree_string_view_empty();
  std::string entry_symbol;
  if (!LookupArray(root, IREE_SV("entries"), 1, entries) ||
      !RequireOk(iree_json_array_get(entries, 0, &entry),
                 "Invalid command manifest entry") ||
      !LookupString(entry, IREE_SV("symbol"), entry_symbol) ||
      entry_symbol != "product_kernel") {
    std::cerr << "Command manifest does not describe product_kernel.\n";
    return false;
  }

  const auto program_path = options.artifact_directory / program_artifact;
  if (!RequireNonemptyFile(program_path, "command program artifact")) {
    return false;
  }
  std::error_code error;
  const auto actual_byte_length =
      std::filesystem::file_size(program_path, error);
  if (error || actual_byte_length != program_byte_length) {
    std::cerr << "Command program byte length does not match its manifest: "
              << program_path << "\n";
    return false;
  }
  return RequireMagic(options.kernel_file, options.kernel_magic,
                      "command kernel product");
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, options)) return 1;
  if (!options.primary_magic.empty() &&
      !RequireMagic(options.primary_file, options.primary_magic,
                    "primary product")) {
    return 1;
  }
  if (options.primary_magic.empty() &&
      !RequireNonemptyFile(options.primary_file, "primary product")) {
    return 1;
  }

  const bool has_dependency_report = !options.dependency_report_file.empty();
  if (has_dependency_report != !options.dependency_component.empty()) {
    std::cerr << "Dependency report arguments must be supplied together.\n";
    return 1;
  }
  if (has_dependency_report && !RequireDependencyReport(options)) return 1;

  const bool has_command_product = !options.manifest_file.empty();
  if (has_command_product != !options.artifact_directory.empty() ||
      has_command_product != !options.kernel_file.empty() ||
      has_command_product != !options.kernel_magic.empty()) {
    std::cerr << "Command product arguments must be supplied together.\n";
    return 1;
  }
  return !has_command_product || RequireManifest(options) ? 0 : 1;
}
