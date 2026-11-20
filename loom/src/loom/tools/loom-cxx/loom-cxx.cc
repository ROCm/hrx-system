// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-cxx: single-source C++ offload compiler driver.
//
// Compiles one hosted C++ translation unit to one host object that also
// carries the unit's device code. The source is preprocessed and type-checked
// once, by cxx with the hosted profile for the host triple. Both back ends
// consume that one translation unit:
//
//   1. Host: cxx's MLIR/LLVM pipeline lowers the unit to optimized LLVM IR.
//      Host code runs first: the kernel_ref descriptors it references are the
//      device pass's roots, and a unit that references none needs no device
//      pass. The importer only reads the unit's AST (it keeps its range-for
//      expansions outside it), so the host code is unaffected by the import.
//   2. Device: if host code references loom::kernel::kernel_ref descriptors,
//      Loom's importer translates the kernels its host code references, and
//      the Loom compiler compiles them for --loom-target. A kernel the
//      importer or the device compiler rejects fails the compile.
//   3. Output: the device code object is embedded in the host module with an
//      embedded_code_object record that tells a runtime how to load it. No
//      static constructor registers or loads it: a runtime loads it when it
//      first launches one of its kernels. Every kernel_ref descriptor the
//      host references is defined in the host module with internal linkage,
//      so the object exports no kernel_ref symbol and each object's launches
//      resolve to its own descriptors and its own code object
//      (loomcxx/kernel_ref.h). A descriptor names the kernel by the ordinal
//      of its export in the code object, and points to a host function, local
//      to the object, that packs the kernel arguments: it combines the
//      importer's host leaves (source offsets and sizes in the host objects)
//      with the kernel argument layout that the device compiler reports for
//      the compiled kernel in its artifact manifest. The module is emitted as
//      one object file.

#include <cxx/memory_layout.h>
#include <cxx/mlir/llvm_codegen.h>
#include <cxx/preprocessor.h>
#include <cxx/toolchain.h>
#include <cxx/translation_unit.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <stdio.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "iree/base/api.h"
#include "loom/error/diagnostic.h"
#include "loom/import/cxx/unit.h"
#include "loom/tooling/cli/loomc_result.h"
#include "loomc/artifact.h"
#include "loomc/artifact_manifest.h"
#include "loomc/compile.h"
#include "loomc/config.h"
#include "loomc/context.h"
#include "loomc/emit.h"
#include "loomc/import/cxx.h"
#include "loomc/iree.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/target.h"
#include "loomc/target/configured.h"
#include "loomc/target/configured_ir.h"
#include "loomc/workspace.h"

namespace {

constexpr std::string_view kTool = "loom-cxx";
// Itanium mangling prefix of loom::kernel::kernel_ref<...> variables.
constexpr std::string_view kKernelRefPrefix = "_ZN4loom6kernel10kernel_ref";
// loom::kernel::descriptor_version in loomcxx/kernel_ref.h.
constexpr uint32_t kDescriptorVersion = 3;
// KernelLaunchLeaf::kind of a device pointer passed as a buffer binding.
constexpr uint32_t kLeafBuffer = 1;

struct Options {
  std::string source;
  std::string output;
  bool compile = false;
  std::string triple = "x86_64-linux-gnu";
  std::string standard;
  int optimization_level = 0;
  std::vector<std::string> include_paths;
  std::vector<std::string> system_include_paths;
  // Directory holding Loom's loomcxx/ headers, searched after every -isystem
  // directory, or empty when none was found (see find_loomcxx_include_dir).
  std::string loomcxx_include_path;
  std::vector<std::string> defines;
  // Macros removed after every -D definition, as GCC orders them.
  std::vector<std::string> undefines;
  // Dependency file options (-MD/-MMD/-MP/-MF/-MT/-MQ).
  bool dependency_file = false;
  bool dependency_skip_system = false;
  bool dependency_phony = false;
  std::string dependency_path;
  std::string dependency_target;
  // Device options.
  std::string loom_target;
  std::string loom_format;
  bool stats = false;
  bool verbose = false;
  std::string keep_directory;
  std::vector<std::string> ignored;
};

[[noreturn]] void usage_error(const std::string& message) {
  std::fprintf(stderr, "%s: error: %s\n", kTool.data(), message.c_str());
  std::fprintf(stderr, "run '%s --help' for usage\n", kTool.data());
  std::exit(1);
}

void print_help() {
  std::printf(
      "loom-cxx: single-source C++ offload compiler driver.\n"
      "\n"
      "Usage: loom-cxx [options] -c source.cpp -o object.o\n"
      "\n"
      "Parses the hosted C++ source once, generates host code from it with\n"
      "cxx's MLIR/LLVM pipeline, imports the kernels it references with Loom's\n"
      "C++ importer, compiles them for the device target and emits one host\n"
      "object that embeds the device code and defines the kernel descriptors.\n"
      "\n"
      "Compiler options:\n"
      "  -c                    compile to an object (required)\n"
      "  -o FILE               output object\n"
      "  -I DIR, -isystem DIR  user and system include directories; the only\n"
      "                        implicit one holds Loom's loomcxx/ headers\n"
      "  -D NAME[=VALUE]       macro definition\n"
      "  -U NAME               macro removal, after every -D\n"
      "  -std=STD              c++17, c++20 (default), ...; gnu++NN reads as\n"
      "                        c++NN\n"
      "  -O0 -O1 -O2 -O3 -Os   host optimization level\n"
      "  -MD -MMD -MP          dependency file, with -MF FILE, -MT/-MQ TARGET\n"
      "  --target=TRIPLE       host triple (default x86_64-linux-gnu)\n"
      "  -W* -f* -g* -m* -w -v -pthread -pipe -Xclang ARG are ignored.\n"
      "\n"
      "Loom options:\n"
      "  --loom-target=FAMILY:SELECTOR\n"
      "      device target, such as amdgpu:gfx90a\n"
      "  --loom-format=FORMAT\n"
      "      device artifact format (default amdgpu-hsaco for amdgpu)\n"
      "  --loom-stats\n"
      "      report parse counts and phase times\n"
      "  --loom-keep=DIR\n"
      "      keep the device intermediates in DIR\n"
      "  --loom-verbose\n"
      "      print every device-pass diagnostic\n");
}

bool starts_with(std::string_view value, std::string_view prefix) {
  return value.substr(0, prefix.size()) == prefix;
}

int parse_optimization_level(std::string_view flag) {
  auto level = flag.substr(2);
  if (level.empty()) {
    return 1;
  }
  if (level == "s" || level == "z" || level == "g") {
    return 2;
  }
  if (level == "fast") {
    return 3;
  }
  if (level.size() == 1 && level[0] >= '0' && level[0] <= '9') {
    return std::min(level[0] - '0', 3);
  }
  usage_error(std::format("unknown optimization level '{}'", flag));
}

// The directory holding Loom's loomcxx/ headers, which every compile searches
// after the -isystem directories, as clang searches its resource directory:
// include/ next to the binary in the build tree (CMake links it to the source
// headers), or lib/loom-cxx/include beside the binary's bin/ directory in an
// installation. Empty when neither holds loomcxx/kernel_ref.h.
std::string find_loomcxx_include_dir(const char* argv0) {
  static int anchor;
  const std::filesystem::path executable =
      llvm::sys::fs::getMainExecutable(argv0, &anchor);
  if (executable.empty()) {
    return {};
  }
  const std::filesystem::path directory = executable.parent_path();
  for (const auto& candidate :
       {directory / "include",
        directory.parent_path() / "lib" / "loom-cxx" / "include"}) {
    std::error_code error;
    if (std::filesystem::exists(candidate / "loomcxx" / "kernel_ref.h",
                                error)) {
      return candidate.lexically_normal().string();
    }
  }
  return {};
}

Options parse_options(int argc, char** argv) {
  Options options;
  std::vector<std::string_view> args(argv + 1, argv + argc);
  auto value = [&](size_t& i, std::string_view flag) -> std::string {
    if (i + 1 >= args.size()) {
      usage_error(std::format("{} needs a value", flag));
    }
    return std::string(args[++i]);
  };
  // Joined (-Ifoo) or separate (-I foo) argument of a short option.
  auto joined = [&](size_t& i, std::string_view flag) -> std::string {
    if (args[i].size() > flag.size()) {
      return std::string(args[i].substr(flag.size()));
    }
    return value(i, flag);
  };
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (arg == "--help" || arg == "-h") {
      print_help();
      std::exit(0);
    } else if (arg == "-c") {
      options.compile = true;
    } else if (starts_with(arg, "-o")) {
      options.output = joined(i, "-o");
    } else if (starts_with(arg, "-isystem")) {
      options.system_include_paths.push_back(joined(i, "-isystem"));
    } else if (starts_with(arg, "-I")) {
      options.include_paths.push_back(joined(i, "-I"));
    } else if (starts_with(arg, "-D")) {
      options.defines.push_back(joined(i, "-D"));
    } else if (starts_with(arg, "-U")) {
      options.undefines.push_back(joined(i, "-U"));
    } else if (starts_with(arg, "-std=")) {
      auto standard = std::string(arg.substr(5));
      if (starts_with(standard, "gnu++")) {
        standard = "c++" + standard.substr(5);
      }
      options.standard = standard;
    } else if (starts_with(arg, "-O")) {
      options.optimization_level = parse_optimization_level(arg);
    } else if (arg == "-MD" || arg == "-MMD") {
      options.dependency_file = true;
      options.dependency_skip_system = arg == "-MMD";
    } else if (arg == "-MP") {
      options.dependency_phony = true;
    } else if (arg == "-MF") {
      options.dependency_path = value(i, arg);
    } else if (arg == "-MT" || arg == "-MQ") {
      options.dependency_target = value(i, arg);
    } else if (starts_with(arg, "--target=")) {
      options.triple = std::string(arg.substr(9));
    } else if (arg == "-target") {
      options.triple = value(i, arg);
    } else if (arg == "-x") {
      if (value(i, arg) != "c++") {
        usage_error("only C++ sources are supported");
      }
    } else if (starts_with(arg, "--loom-target=")) {
      options.loom_target = std::string(arg.substr(14));
    } else if (starts_with(arg, "--loom-format=")) {
      options.loom_format = std::string(arg.substr(14));
    } else if (arg == "--loom-stats") {
      options.stats = true;
    } else if (arg == "--loom-verbose") {
      options.verbose = true;
    } else if (starts_with(arg, "--loom-keep=")) {
      options.keep_directory = std::string(arg.substr(12));
    } else if (arg == "-Xclang") {
      options.ignored.push_back(std::string(arg) + " " + value(i, arg));
    } else if (starts_with(arg, "-W") || starts_with(arg, "-f") ||
               starts_with(arg, "-g") || starts_with(arg, "-m") ||
               arg == "-w" || arg == "-v" || arg == "-pthread" ||
               arg == "-pipe" || arg == "-pedantic") {
      // Diagnostics, code-model and debug flags of the surrounding build.
      // The hosted profile fixes exceptions off and non-PIC host code. Build
      // systems pass -v while identifying the compiler (CMake's ABI check).
      options.ignored.push_back(std::string(arg));
    } else if (starts_with(arg, "-")) {
      usage_error(std::format("unsupported option '{}'", arg));
    } else {
      if (!options.source.empty()) {
        usage_error("expected exactly one source file");
      }
      options.source = std::string(arg);
    }
  }
  if (!options.compile) {
    usage_error("only compilation (-c) is supported");
  }
  if (options.source.empty()) {
    usage_error("no source file");
  }
  if (options.output.empty()) {
    options.output = std::filesystem::path(options.source)
                         .filename()
                         .replace_extension(".o")
                         .string();
  }
  options.loomcxx_include_path = find_loomcxx_include_dir(argv[0]);
  if (!options.loom_target.empty() && options.loom_format.empty()) {
    if (starts_with(options.loom_target, "amdgpu:")) {
      options.loom_format = "amdgpu-hsaco";
    } else {
      usage_error("--loom-format is required for this --loom-target");
    }
  }
  return options;
}

// ---------------------------------------------------------------------------
// Small RAII and status helpers
// ---------------------------------------------------------------------------

template <typename T, void (*Release)(T*)>
struct Releaser {
  void operator()(T* value) const { Release(value); }
};
template <typename T, void (*Release)(T*)>
using Handle = std::unique_ptr<T, Releaser<T, Release>>;

using TargetEnvironment =
    Handle<loomc_target_environment_t, loomc_target_environment_release>;
using Context = Handle<loomc_context_t, loomc_context_release>;
using Workspace = Handle<loomc_workspace_t, loomc_workspace_release>;
using Source = Handle<loomc_source_t, loomc_source_release>;
using Module = Handle<loomc_module_t, loomc_module_release>;
using Result = Handle<loomc_result_t, loomc_result_release>;
using Unit = Handle<loomc_cxx_unit_t, loomc_cxx_unit_release>;
using Compiler = Handle<loomc_compiler_t, loomc_compiler_release>;
using PassProgram = Handle<loomc_pass_program_t, loomc_pass_program_release>;
using TargetProfile =
    Handle<loomc_target_profile_t, loomc_target_profile_release>;

// Thrown for infrastructure failures after the message has been printed.
struct Failure {};

void check(loomc_status_t status, std::string_view what) {
  if (loomc_status_is_ok(status)) {
    return;
  }
  iree_status_t iree_status = iree_status_from_loomc(status);
  std::fprintf(stderr, "%s: error: %.*s: ", kTool.data(), (int)what.size(),
               what.data());
  iree_status_fprint(stderr, iree_status);
  iree_status_free(iree_status);
  throw Failure{};
}

void check_iree(iree_status_t status, std::string_view what) {
  if (iree_status_is_ok(status)) {
    return;
  }
  std::fprintf(stderr, "%s: error: %.*s: ", kTool.data(), (int)what.size(),
               what.data());
  iree_status_fprint(stderr, status);
  iree_status_free(status);
  throw Failure{};
}

// Prints a result's diagnostics to a string; returns whether it succeeded.
bool print_result(const loomc_result_t* result, std::string& text) {
  char* buffer = nullptr;
  size_t size = 0;
  FILE* stream = open_memstream(&buffer, &size);
  if (!stream) {
    std::fprintf(stderr, "%s: error: cannot capture diagnostics\n",
                 kTool.data());
    throw Failure{};
  }
  bool succeeded = false;
  iree_status_t status =
      loom_tooling_cli_print_loomc_result(stream, result, &succeeded);
  std::fclose(stream);
  text.append(buffer, size);
  std::free(buffer);
  check_iree(status, "printing diagnostics");
  return succeeded;
}

std::vector<std::string_view> lines(std::string_view text) {
  std::vector<std::string_view> result;
  while (!text.empty()) {
    auto end = text.find('\n');
    result.push_back(text.substr(0, end));
    if (end == std::string_view::npos) {
      break;
    }
    text.remove_prefix(end + 1);
  }
  return result;
}

std::optional<std::string> read_file(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return std::nullopt;
  }
  std::stringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

// ---------------------------------------------------------------------------
// AMDGPU code object exports
// ---------------------------------------------------------------------------

// Reads the MessagePack values of an AMDGPU code object metadata note.
class MsgPackReader {
 public:
  explicit MsgPackReader(std::string_view data) : data_(data) {}

  // Reads a string; false when the next value is not one.
  bool read_string(std::string_view& value) {
    const auto tag = byte();
    if (!tag) {
      return false;
    }
    uint64_t length = 0;
    if (*tag >= 0xa0 && *tag <= 0xbf) {
      length = *tag & 0x1f;
    } else if (*tag == 0xd9 || *tag == 0xda || *tag == 0xdb) {
      const auto read = unsigned_value(1u << (*tag - 0xd9));
      if (!read) {
        return false;
      }
      length = *read;
    } else {
      return false;
    }
    return bytes(length, value);
  }

  // Reads an array (or map) header and its element (or pair) count; false
  // when the next value is not one.
  bool read_array(uint64_t& count) { return read_container(false, count); }
  bool read_map(uint64_t& count) { return read_container(true, count); }

  // Skips the next value, with everything it contains.
  bool skip() {
    const auto tag = byte();
    if (!tag) {
      return false;
    }
    std::string_view ignored;
    uint64_t count = 0;
    const uint8_t t = *tag;
    if (t <= 0x7f || t >= 0xe0 || t == 0xc0 || t == 0xc2 || t == 0xc3) {
      return true;
    }
    if (t >= 0x80 && t <= 0x9f) {
      count = t & 0x0f;
      return skip_values(t <= 0x8f ? 2 * count : count);
    }
    if (t >= 0xa0 && t <= 0xbf) {
      return bytes(t & 0x1f, ignored);
    }
    if (t >= 0xc4 && t <= 0xc6) {  // bin 8/16/32
      const auto length = unsigned_value(1u << (t - 0xc4));
      return length && bytes(*length, ignored);
    }
    if (t >= 0xc7 && t <= 0xc9) {  // ext 8/16/32: length, type, data
      const auto length = unsigned_value(1u << (t - 0xc7));
      return length && bytes(*length + 1, ignored);
    }
    if (t == 0xca || t == 0xcb) {  // float 32/64
      return bytes(t == 0xca ? 4 : 8, ignored);
    }
    if (t >= 0xcc && t <= 0xd3) {  // uint and int 8/16/32/64
      return bytes(1u << ((t - 0xcc) & 3), ignored);
    }
    if (t >= 0xd4 && t <= 0xd8) {  // fixext 1/2/4/8/16: type, data
      return bytes((1u << (t - 0xd4)) + 1, ignored);
    }
    if (t >= 0xd9 && t <= 0xdb) {  // str 8/16/32
      const auto length = unsigned_value(1u << (t - 0xd9));
      return length && bytes(*length, ignored);
    }
    if (t >= 0xdc && t <= 0xdf) {  // array 16/32, map 16/32
      const auto read = unsigned_value(t == 0xdc || t == 0xde ? 2 : 4);
      return read && skip_values(t >= 0xde ? 2 * *read : *read);
    }
    return false;
  }

 private:
  std::optional<uint8_t> byte() {
    if (data_.empty()) {
      return std::nullopt;
    }
    const auto value = static_cast<uint8_t>(data_.front());
    data_.remove_prefix(1);
    return value;
  }
  // Big-endian unsigned integer of |size| bytes.
  std::optional<uint64_t> unsigned_value(unsigned size) {
    std::string_view value;
    if (!bytes(size, value)) {
      return std::nullopt;
    }
    uint64_t result = 0;
    for (char c : value) {
      result = (result << 8) | static_cast<uint8_t>(c);
    }
    return result;
  }
  bool bytes(uint64_t size, std::string_view& value) {
    if (size > data_.size()) {
      return false;
    }
    value = data_.substr(0, size);
    data_.remove_prefix(size);
    return true;
  }
  bool skip_values(uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
      if (!skip()) {
        return false;
      }
    }
    return true;
  }
  bool read_container(bool map, uint64_t& count) {
    const auto tag = byte();
    if (!tag) {
      return false;
    }
    const uint8_t fixed = map ? 0x80 : 0x90;
    if (*tag >= fixed && *tag <= fixed + 0x0f) {
      count = *tag & 0x0f;
      return true;
    }
    const uint8_t sized = map ? 0xde : 0xdc;
    if (*tag != sized && *tag != sized + 1) {
      return false;
    }
    const auto read = unsigned_value(*tag == sized ? 2 : 4);
    if (!read) {
      return false;
    }
    count = *read;
    return true;
  }

  std::string_view data_;
};

// Export names of the AMDGPU code object |image| in export ordinal order:
// the order of its amdhsa.kernels metadata, in which HRX's loader numbers the
// exports (hrx_executable_export_info). An export is named by its .name, or
// by its .symbol without the ".kd" suffix, as the loader names it. Returns
// false with |error| set when |image| is not such a code object.
bool amdgpu_export_names(std::string_view image,
                         std::vector<std::string>& names, std::string& error) {
  names.clear();
  auto read = [&](uint64_t offset, unsigned size) -> std::optional<uint64_t> {
    if (offset > image.size() || size > image.size() - offset) {
      return std::nullopt;
    }
    uint64_t value = 0;
    for (unsigned i = 0; i < size; ++i) {  // little-endian
      value |= uint64_t(static_cast<uint8_t>(image[offset + i])) << (8 * i);
    }
    return value;
  };
  if (image.size() < 64 || image.substr(0, 4) != "\x7f" "ELF" ||
      image[4] != 2 || image[5] != 1) {
    error = "not a little-endian ELF64 code object";
    return false;
  }
  const uint64_t section_offset = *read(0x28, 8);
  const uint64_t section_size = *read(0x3a, 2);
  const uint64_t section_count = *read(0x3c, 2);
  constexpr uint32_t kSectionNote = 7;
  constexpr uint32_t kNoteAmdgpuMetadata = 32;
  bool found = false;
  for (uint64_t i = 0; i < section_count && !found; ++i) {
    const uint64_t header = section_offset + i * section_size;
    const auto type = read(header + 4, 4);
    const auto offset = read(header + 0x18, 8);
    const auto size = read(header + 0x20, 8);
    if (!type || !offset || !size || *offset > image.size() ||
        *size > image.size() - *offset) {
      error = "malformed section header";
      return false;
    }
    if (*type != kSectionNote) {
      continue;
    }
    // Notes: name size, descriptor size and type, then the name and the
    // descriptor, each padded to 4 bytes.
    for (uint64_t note = *offset; note + 12 <= *offset + *size;) {
      const uint64_t name_size = *read(note, 4);
      const uint64_t desc_size = *read(note + 4, 4);
      const uint64_t note_type = *read(note + 8, 4);
      const uint64_t name = note + 12;
      const uint64_t desc = name + ((name_size + 3) & ~uint64_t(3));
      const uint64_t next = desc + ((desc_size + 3) & ~uint64_t(3));
      if (next > *offset + *size) {
        error = "malformed note";
        return false;
      }
      note = next;
      if (note_type != kNoteAmdgpuMetadata ||
          image.substr(name, name_size).substr(0, 6) != "AMDGPU") {
        continue;
      }
      found = true;
      MsgPackReader reader(image.substr(desc, desc_size));
      uint64_t entries = 0;
      if (!reader.read_map(entries)) {
        error = "malformed AMDGPU metadata";
        return false;
      }
      for (uint64_t entry = 0; entry < entries; ++entry) {
        std::string_view key;
        if (!reader.read_string(key)) {
          error = "malformed AMDGPU metadata";
          return false;
        }
        if (key != "amdhsa.kernels") {
          if (!reader.skip()) {
            error = "malformed AMDGPU metadata";
            return false;
          }
          continue;
        }
        uint64_t kernels = 0;
        if (!reader.read_array(kernels)) {
          error = "malformed amdhsa.kernels";
          return false;
        }
        for (uint64_t k = 0; k < kernels; ++k) {
          uint64_t fields = 0;
          if (!reader.read_map(fields)) {
            error = "malformed amdhsa.kernels";
            return false;
          }
          std::string_view kernel_name, symbol;
          for (uint64_t f = 0; f < fields; ++f) {
            std::string_view field;
            if (!reader.read_string(field)) {
              error = "malformed amdhsa.kernels";
              return false;
            }
            std::string_view* value = field == ".name"     ? &kernel_name
                                      : field == ".symbol" ? &symbol
                                                           : nullptr;
            if (value ? !reader.read_string(*value) : !reader.skip()) {
              error = "malformed amdhsa.kernels";
              return false;
            }
          }
          if (kernel_name.empty() && symbol.ends_with(".kd")) {
            kernel_name = symbol.substr(0, symbol.size() - 3);
          }
          names.emplace_back(kernel_name);
        }
      }
      break;
    }
  }
  if (!found) {
    error = "no AMDGPU metadata note";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

class Driver {
 public:
  explicit Driver(Options options) : options_(std::move(options)) {}

  int run();

 private:
  void make_context(bool ir_only, TargetEnvironment& environment,
                    Context& context, Workspace& workspace);
  loomc_cxx_import_options_t import_options() const;
  bool parse();
  void write_dependency_file();
  bool generate_host_code();
  std::vector<std::string> referenced_kernel_refs() const;
  // Runs the device pass in a work directory. Returns false when the pass
  // failed and the failure policy fails the compile.
  bool run_device_pass();
  // Imports and compiles the device code. Returns false when the device pass
  // failed.
  bool device_pass();
  // Imports the kernels of kernel_refs_.
  bool import_device_module(std::string& bytecode_out, bool& has_kernels);
  bool compile_device_module(const std::string& bytecode);
  void read_argument_layouts(const loomc_result_t* result);
  void read_export_ordinals();
  [[noreturn]] void device_failure(std::string_view stage,
                                   std::string_view log);
  void embed_device_code();
  void define_kernel_descriptors(llvm::GlobalVariable* code_object);
  // One kernel argument a packing function copies: |length| bytes from
  // |source_offset| in host parameter object |parameter| to |kernarg_offset|.
  struct ArgumentCopy {
    uint32_t parameter;
    uint64_t source_offset;
    uint64_t kernarg_offset;
    uint64_t length;
  };
  llvm::Function* define_packing_function(
      const std::vector<ArgumentCopy>& copies, uint64_t kernarg_size,
      uint32_t parameter_count, size_t index);
  void localize_descriptor_users(
      const std::vector<llvm::GlobalVariable*>& descriptors);
  void emit_object();
  std::string temp_path(std::string_view name) const;

  Options options_;
  std::string source_text_;
  // Import side: IR-only target environment, as loom-import-cxx uses.
  TargetEnvironment import_environment_;
  Context import_context_;
  Workspace import_workspace_;
  Unit unit_;
  // Host side.
  std::unique_ptr<llvm::TargetMachine> target_machine_;
  std::unique_ptr<llvm::LLVMContext> llvm_context_;
  std::unique_ptr<llvm::Module> host_module_;
  std::vector<std::string> kernel_refs_;
  // Device side.
  std::string work_directory_;
  bool keep_work_directory_ = false;
  // Host launch facts of the kernel_ref specializations, from the importer.
  std::vector<loom::cxx_import::KernelLaunch> launches_;
  // Kernel argument layout of one compiled kernel export, from the device
  // compiler's artifact manifest.
  struct ArgumentLayout {
    // Byte offset and payload length of each argument, in argument order.
    struct Argument {
      bool buffer = false;
      uint64_t offset = 0;
      uint64_t length = 0;
      bool present = false;
    };
    uint64_t segment_size = 0;
    std::vector<Argument> arguments;
  };
  std::unordered_map<std::string, ArgumentLayout> layouts_;
  // Export ordinal of each compiled kernel in the code object, by export name.
  std::unordered_map<std::string, uint32_t> export_ordinals_;
  std::string code_object_;
  int import_count_ = 0;
  int device_compile_count_ = 0;
  // Phase times in seconds, for --loom-stats.
  double parse_seconds_ = 0, host_seconds_ = 0, import_seconds_ = 0,
         device_compile_seconds_ = 0, emit_seconds_ = 0;
};

std::string Driver::temp_path(std::string_view name) const {
  return (std::filesystem::path(work_directory_) / name).string();
}

void Driver::make_context(bool ir_only, TargetEnvironment& environment,
                          Context& context, Workspace& workspace) {
  const loomc_allocator_t allocator = loomc_allocator_system();
  loomc_target_environment_t* raw_environment = nullptr;
  check(ir_only ? loomc_target_environment_create_configured_ir(
                      allocator, &raw_environment)
                : loomc_target_environment_create_configured(allocator,
                                                             &raw_environment),
        "creating the target environment");
  environment.reset(raw_environment);
  loomc_context_target_options_t target_options = {};
  target_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
  target_options.structure_size = sizeof(target_options);
  target_options.target_environment = environment.get();
  loomc_context_options_t context_options = {};
  context_options.type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS;
  context_options.structure_size = sizeof(context_options);
  context_options.next = &target_options;
  loomc_context_t* raw_context = nullptr;
  check(loomc_context_create(&context_options, allocator, &raw_context),
        "creating the compiler context");
  context.reset(raw_context);
  loomc_workspace_options_t workspace_options = {};
  workspace_options.type = LOOMC_STRUCTURE_TYPE_WORKSPACE_OPTIONS;
  workspace_options.structure_size = sizeof(workspace_options);
  // As loom-import-cxx: 64 KiB blocks avoid per-block mappings on allocators
  // with a 128 KiB mapping threshold. The compiler uses default blocks.
  workspace_options.block_size = ir_only ? 64 * 1024 : 0;
  loomc_workspace_t* raw_workspace = nullptr;
  check(loomc_workspace_create(ir_only ? &workspace_options : nullptr,
                               allocator, &raw_workspace),
        "creating the compiler workspace");
  workspace.reset(raw_workspace);
}

// Source configuration of the one parse, also passed to every import.
loomc_cxx_import_options_t Driver::import_options() const {
  loomc_cxx_import_options_t options = {};
  options.type = LOOMC_STRUCTURE_TYPE_CXX_IMPORT_OPTIONS;
  options.structure_size = sizeof(options);
  options.standard = loomc_make_cstring_view(options_.standard.c_str());
  options.triple = loomc_make_cstring_view(options_.triple.c_str());
  options.data_model = LOOMC_CXX_DATA_MODEL_LP64;
  options.fp_model = LOOMC_CXX_FP_MODEL_PRECISE;
  // A host compile: the hosted profile for the triple, the include
  // directories on the command line and loom-cxx's loomcxx/ headers. Device
  // import leaves mutable globals on the host.
  options.flags = LOOMC_CXX_IMPORT_FLAG_HOSTED |
                  LOOMC_CXX_IMPORT_FLAG_NO_BUILTIN_INCLUDES |
                  LOOMC_CXX_IMPORT_FLAG_SKIP_MUTABLE_GLOBALS;
  return options;
}

bool Driver::parse() {
  auto start = Clock::now();
  auto text = read_file(options_.source);
  if (!text) {
    std::fprintf(stderr, "%s: error: cannot read '%s'\n", kTool.data(),
                 options_.source.c_str());
    return false;
  }
  source_text_ = std::move(*text);
  make_context(/*ir_only=*/true, import_environment_, import_context_,
               import_workspace_);

  std::vector<loomc_string_view_t> includes, system_includes;
  for (const auto& path : options_.include_paths) {
    includes.push_back(loomc_make_cstring_view(path.c_str()));
  }
  for (const auto& path : options_.system_include_paths) {
    system_includes.push_back(loomc_make_cstring_view(path.c_str()));
  }
  if (!options_.loomcxx_include_path.empty()) {
    system_includes.push_back(
        loomc_make_cstring_view(options_.loomcxx_include_path.c_str()));
  }
  std::vector<std::string> define_names, define_values;
  for (const auto& define : options_.defines) {
    auto equal = define.find('=');
    define_names.push_back(define.substr(0, equal));
    define_values.push_back(equal == std::string::npos
                                ? std::string("1")
                                : define.substr(equal + 1));
  }
  std::vector<loomc_cxx_define_t> defines(options_.defines.size());
  for (size_t i = 0; i < defines.size(); ++i) {
    defines[i].name = loomc_make_cstring_view(define_names[i].c_str());
    defines[i].value = loomc_make_cstring_view(define_values[i].c_str());
  }
  std::vector<loomc_string_view_t> undefines;
  for (const auto& name : options_.undefines) {
    undefines.push_back(loomc_make_cstring_view(name.c_str()));
  }
  loomc_cxx_import_options_t options = import_options();
  options.include_paths = includes.data();
  options.include_path_count = includes.size();
  options.system_include_paths = system_includes.data();
  options.system_include_path_count = system_includes.size();
  options.defines = defines.data();
  options.define_count = defines.size();
  options.undefines = undefines.data();
  options.undefine_count = undefines.size();

  loomc_source_options_t source_options = {};
  source_options.identifier = loomc_make_cstring_view(options_.source.c_str());
  source_options.contents =
      loomc_make_byte_span(source_text_.data(), source_text_.size());
  source_options.storage = LOOMC_SOURCE_STORAGE_BORROWED;
  loomc_source_t* raw_source = nullptr;
  check(loomc_source_create(&source_options, loomc_allocator_system(),
                            &raw_source),
        "creating the source");
  Source source(raw_source);

  loomc_cxx_unit_t* raw_unit = nullptr;
  loomc_result_t* raw_result = nullptr;
  check(loomc_cxx_unit_parse(import_context_.get(), source.get(), &options,
                             loomc_allocator_system(), &raw_unit, &raw_result),
        "parsing");
  unit_.reset(raw_unit);
  Result result(raw_result);
  std::string text_out;
  const bool succeeded = print_result(result.get(), text_out);
  std::fputs(text_out.c_str(), stderr);
  parse_seconds_ = seconds_since(start);
  return succeeded && unit_;
}

void Driver::write_dependency_file() {
  if (!options_.dependency_file) {
    return;
  }
  auto& unit = loom::cxx_import::unit_translation_unit(
      reinterpret_cast<loom_cxx_unit_t*>(unit_.get()));
  std::vector<std::string> dependencies = {options_.source};
  std::unordered_set<std::string> seen(dependencies.begin(),
                                       dependencies.end());
  for (const auto& [file, system] : unit.preprocessor()->includedFiles()) {
    if (options_.dependency_skip_system && system) {
      continue;
    }
    if (seen.insert(file).second) {
      dependencies.push_back(file);
    }
  }
  std::string path = options_.dependency_path;
  if (path.empty()) {
    path =
        std::filesystem::path(options_.output).replace_extension(".d").string();
  }
  std::string target = options_.dependency_target.empty()
                           ? options_.output
                           : options_.dependency_target;
  std::ofstream out(path);
  out << target << ':';
  for (const auto& dependency : dependencies) {
    out << " \\\n  " << dependency;
  }
  out << '\n';
  if (options_.dependency_phony) {
    for (size_t i = 1; i < dependencies.size(); ++i) {
      out << '\n' << dependencies[i] << ":\n";
    }
  }
  if (!out) {
    std::fprintf(stderr, "%s: error: cannot write '%s'\n", kTool.data(),
                 path.c_str());
    throw Failure{};
  }
}

bool Driver::generate_host_code() {
  auto start = Clock::now();
  auto* native_unit = reinterpret_cast<loom_cxx_unit_t*>(unit_.get());
  // Host code generation reports through the unit's frontend diagnostics.
  loom_cxx_unit_set_diagnostic_sink(
      native_unit,
      loom_diagnostic_sink_t{loom_diagnostic_stderr_sink, nullptr});
  auto& unit = loom::cxx_import::unit_translation_unit(native_unit);
  const auto& toolchain = loom::cxx_import::unit_toolchain(native_unit);
  toolchain.applyEntryPointAbi(&unit);

  std::string error;
  target_machine_ = cxx::createTargetMachine(
      toolchain.memoryLayout()->triple(), options_.optimization_level, error);
  if (!target_machine_) {
    std::fprintf(stderr, "%s: error: %s\n", kTool.data(), error.c_str());
    return false;
  }
  llvm_context_ = std::make_unique<llvm::LLVMContext>();
  cxx::LLVMCodegenOptions codegen_options;
  codegen_options.optimizationLevel = options_.optimization_level;
  host_module_ = cxx::generateLLVMModule(
      &unit, *llvm_context_, target_machine_.get(), codegen_options, error);
  loom_cxx_unit_set_diagnostic_sink(native_unit, loom_diagnostic_sink_t{});
  if (!host_module_ || loom::cxx_import::unit_has_error(native_unit)) {
    if (!error.empty()) {
      std::fprintf(stderr, "%s: error: %s\n", kTool.data(), error.c_str());
    }
    return false;
  }
  kernel_refs_ = referenced_kernel_refs();
  host_seconds_ = seconds_since(start);
  return true;
}

// Descriptors the host code references, as an object file's undefined
// kernel_ref symbols would list them.
std::vector<std::string> Driver::referenced_kernel_refs() const {
  std::set<std::string> names;
  for (const auto& global : host_module_->globals()) {
    if (global.isDeclaration() &&
        starts_with(global.getName().str(), kKernelRefPrefix)) {
      names.insert(global.getName().str());
    }
  }
  return {names.begin(), names.end()};
}

void Driver::device_failure(std::string_view stage, std::string_view log) {
  std::fprintf(stderr, "%s: %.*s failed for %s\n", kTool.data(),
               (int)stage.size(), stage.data(), options_.source.c_str());
  int printed = 0;
  auto log_lines = lines(log);
  for (size_t i = 0; i < log_lines.size() && printed < 40; ++i) {
    if (log_lines[i].find("error [") == std::string_view::npos) {
      continue;
    }
    for (size_t j = i; j < std::min(i + 4, log_lines.size()); ++j, ++printed) {
      std::fprintf(stderr, "  %.*s\n", (int)log_lines[j].size(),
                   log_lines[j].data());
    }
  }
  throw Failure{};
}

bool Driver::import_device_module(std::string& bytecode, bool& has_kernels) {
  auto start = Clock::now();
  // The kernels to import are exactly those whose descriptors the host code
  // references.
  std::vector<loomc_string_view_t> kernel_refs;
  for (const auto& symbol : kernel_refs_) {
    kernel_refs.push_back(loomc_make_cstring_view(symbol.c_str()));
  }
  loomc_cxx_import_options_t options = import_options();
  options.kernel_refs = kernel_refs.data();
  options.kernel_ref_count = kernel_refs.size();

  auto* native_unit = reinterpret_cast<loom_cxx_unit_t*>(unit_.get());
  loomc_module_t* raw_module = nullptr;
  loomc_result_t* raw_result = nullptr;
  check(loomc_module_import_cxx_unit(
            import_context_.get(), import_workspace_.get(), unit_.get(),
            &options, loomc_allocator_system(), &raw_module, &raw_result),
        "importing device code");
  ++import_count_;
  Module module(raw_module);
  Result result(raw_result);
  std::string log;
  bool succeeded = print_result(result.get(), log);
  if (options_.verbose) {
    std::fputs(log.c_str(), stderr);
  }

  if (!succeeded || !module) {
    import_seconds_ += seconds_since(start);
    device_failure("device import", log);
    return false;
  }
  launches_ = loom::cxx_import::unit_kernel_launches(native_unit);
  has_kernels = std::any_of(launches_.begin(), launches_.end(),
                            [](const auto& launch) { return launch.available; });
  if (!has_kernels) {
    import_seconds_ += seconds_since(start);
    return true;
  }

  // Cleanup and verification, as loom-import-cxx runs before writing.
  loomc_pass_program_t* raw_program = nullptr;
  raw_result = nullptr;
  check(loomc_pass_program_create_from_pipeline_text(
            import_context_.get(),
            loomc_make_cstring_view("combine,cse,dce,location-dce"), nullptr,
            loomc_allocator_system(), &raw_program, &raw_result),
        "creating the import cleanup pipeline");
  PassProgram program(raw_program);
  result.reset(raw_result);
  if (result && !print_result(result.get(), log)) {
    device_failure("device import cleanup", log);
    return false;
  }
  raw_result = nullptr;
  check(loomc_pass_program_run(program.get(), import_workspace_.get(),
                               module.get(), nullptr, loomc_allocator_system(),
                               &raw_result),
        "running the import cleanup pipeline");
  result.reset(raw_result);
  if (!print_result(result.get(), log)) {
    device_failure("device import cleanup", log);
    return false;
  }
  raw_result = nullptr;
  check(loomc_module_verify(module.get(), nullptr, loomc_allocator_system(),
                            &raw_result),
        "verifying the device module");
  result.reset(raw_result);
  if (!print_result(result.get(), log)) {
    device_failure("device import verification", log);
    return false;
  }
  loomc_module_serialize_options_t serialize_options = {};
  serialize_options.type = LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS;
  serialize_options.structure_size = sizeof(serialize_options);
  serialize_options.format = LOOMC_SOURCE_FORMAT_BYTECODE;
  loomc_source_t* raw_serialized = nullptr;
  check(loomc_module_serialize_to_source(module.get(), &serialize_options,
                                         loomc_allocator_system(),
                                         &raw_serialized),
        "serializing the device module");
  Source serialized(raw_serialized);
  const auto contents = loomc_source_contents(serialized.get());
  bytecode.assign(reinterpret_cast<const char*>(contents.data),
                  contents.data_length);
  if (keep_work_directory_) {
    std::ofstream(temp_path("device.loombc"), std::ios::binary) << bytecode;
  }
  import_seconds_ += seconds_since(start);
  return true;
}

bool Driver::compile_device_module(const std::string& bytecode) {
  auto start = Clock::now();
  // The device compiler gets the serialized module, exactly as a separate
  // loom-compile process would read it.
  TargetEnvironment environment;
  Context context;
  Workspace workspace;
  make_context(/*ir_only=*/false, environment, context, workspace);
  loomc_compiler_t* raw_compiler = nullptr;
  check(loomc_compiler_create(context.get(), nullptr, loomc_allocator_system(),
                              &raw_compiler),
        "creating the device compiler");
  Compiler compiler(raw_compiler);
  const std::string identifier = temp_path("device.loombc");
  loomc_source_options_t source_options = {};
  source_options.identifier = loomc_make_cstring_view(identifier.c_str());
  source_options.contents =
      loomc_make_byte_span(bytecode.data(), bytecode.size());
  source_options.storage = LOOMC_SOURCE_STORAGE_BORROWED;
  source_options.format = LOOMC_SOURCE_FORMAT_BYTECODE;
  loomc_source_t* raw_source = nullptr;
  check(loomc_source_create(&source_options, loomc_allocator_system(),
                            &raw_source),
        "creating the device source");
  Source source(raw_source);
  loomc_module_t* raw_module = nullptr;
  loomc_result_t* raw_result = nullptr;
  check(loomc_module_deserialize_from_source(
            context.get(), workspace.get(), source.get(), nullptr,
            loomc_allocator_system(), &raw_module, &raw_result),
        "reading the device module");
  Module module(raw_module);
  Result result(raw_result);
  std::string log;
  if (!print_result(result.get(), log) || !module) {
    device_failure("device compile", log);
    return false;
  }
  loomc_target_profile_t* raw_profile = nullptr;
  check(loomc_target_profile_select(
            environment.get(),
            loomc_make_cstring_view(options_.loom_target.c_str()),
            loomc_allocator_system(), &raw_profile),
        "selecting the device target");
  TargetProfile profile(raw_profile);

  loomc_config_options_t config = {};
  config.flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
  // The details manifest reports each kernel's argument layout as the device
  // compiler decided it.
  loomc_artifact_manifest_options_t manifest_options = {};
  manifest_options.type = LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS;
  manifest_options.structure_size = sizeof(manifest_options);
  manifest_options.mode = LOOMC_ARTIFACT_MANIFEST_MODE_DETAILS;
  loomc_emit_options_t emit_options = {};
  emit_options.type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS;
  emit_options.structure_size = sizeof(emit_options);
  emit_options.next = &manifest_options;
  emit_options.artifact_format =
      loomc_make_cstring_view(options_.loom_format.c_str());
  emit_options.artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY;
  loomc_compile_artifact_options_t compile_options = {};
  compile_options.type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS;
  compile_options.structure_size = sizeof(compile_options);
  compile_options.target_profile = profile.get();
  compile_options.config = &config;
  compile_options.emit_options = &emit_options;
  raw_result = nullptr;
  check(loomc_compile_artifact(compiler.get(), workspace.get(),
                               /*pass_program=*/nullptr, module.get(),
                               &compile_options, loomc_allocator_system(),
                               &raw_result),
        "compiling device code");
  ++device_compile_count_;
  result.reset(raw_result);
  const bool succeeded = print_result(result.get(), log);
  if (options_.verbose) {
    std::fputs(log.c_str(), stderr);
  }
  device_compile_seconds_ += seconds_since(start);
  if (!succeeded) {
    device_failure("device compile", log);
    return false;
  }
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result.get());
       ++i) {
    const auto* artifact = loomc_result_artifact_at(result.get(), i);
    if (artifact->kind != LOOMC_ARTIFACT_KIND_EXECUTABLE) {
      continue;
    }
    loomc_source_t* raw_artifact = nullptr;
    check(loomc_artifact_create_source(artifact, LOOMC_SOURCE_FORMAT_UNKNOWN,
                                       loomc_allocator_system(), &raw_artifact),
          "reading the device code object");
    Source artifact_source(raw_artifact);
    const auto contents = loomc_source_contents(artifact_source.get());
    code_object_.assign(reinterpret_cast<const char*>(contents.data),
                        contents.data_length);
  }
  if (code_object_.empty()) {
    device_failure("device compile", "error [TOOL]: no code object\n");
    return false;
  }
  if (keep_work_directory_) {
    std::ofstream(temp_path("device.code"), std::ios::binary) << code_object_;
  }
  read_argument_layouts(result.get());
  read_export_ordinals();
  return true;
}

// Numbers the exports of the code object as the runtime's loader does, so a
// descriptor names its kernel by export ordinal.
void Driver::read_export_ordinals() {
  export_ordinals_.clear();
  if (options_.loom_format != "amdgpu-hsaco") {
    std::fprintf(stderr,
                 "%s: error: kernel descriptors need the export ordinals of "
                 "the code object, which loom-cxx reads only from the "
                 "amdgpu-hsaco format, not '%s'\n",
                 kTool.data(), options_.loom_format.c_str());
    throw Failure{};
  }
  std::vector<std::string> names;
  std::string error;
  if (!amdgpu_export_names(code_object_, names, error)) {
    std::fprintf(stderr, "%s: error: device code object: %s\n", kTool.data(),
                 error.c_str());
    throw Failure{};
  }
  for (size_t i = 0; i < names.size(); ++i) {
    if (!export_ordinals_.emplace(names[i], static_cast<uint32_t>(i)).second) {
      std::fprintf(stderr,
                   "%s: error: device code object: export '%s' appears twice\n",
                   kTool.data(), names[i].c_str());
      throw Failure{};
    }
  }
}

// Reads the kernel argument layout of every compiled kernel export from the
// artifact manifest of |result|.
void Driver::read_argument_layouts(const loomc_result_t* result) {
  layouts_.clear();
  std::string manifest;
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const auto* artifact = loomc_result_artifact_at(result, i);
    if (artifact->kind != LOOMC_ARTIFACT_KIND_REPORT ||
        std::string_view(artifact->format.data, artifact->format.size) !=
            LOOMC_ARTIFACT_FORMAT_ARTIFACT_MANIFEST_JSON) {
      continue;
    }
    loomc_source_t* raw_artifact = nullptr;
    check(loomc_artifact_create_source(artifact, LOOMC_SOURCE_FORMAT_UNKNOWN,
                                       loomc_allocator_system(), &raw_artifact),
          "reading the device artifact manifest");
    Source artifact_source(raw_artifact);
    const auto contents = loomc_source_contents(artifact_source.get());
    manifest.assign(reinterpret_cast<const char*>(contents.data),
                    contents.data_length);
  }
  if (keep_work_directory_) {
    std::ofstream(temp_path("device.manifest.json")) << manifest;
  }
  auto fail = [&](const std::string& message) {
    std::fprintf(stderr, "%s: error: device artifact manifest: %s\n",
                 kTool.data(), message.c_str());
    throw Failure{};
  };
  if (manifest.empty()) {
    fail("the device compiler produced none");
  }
  auto parsed = llvm::json::parse(manifest);
  if (!parsed) {
    fail(llvm::toString(parsed.takeError()));
  }
  const auto* functions = parsed->getAsObject()
                              ? parsed->getAsObject()->getArray("functions")
                              : nullptr;
  if (!functions) {
    fail("no functions");
  }
  for (const auto& value : *functions) {
    const auto* function = value.getAsObject();
    const auto name = function ? function->getString("name") : std::nullopt;
    const auto* interface =
        function ? function->getObject("interface") : nullptr;
    if (!name || !interface) {
      fail("malformed function");
    }
    ArgumentLayout layout;
    const auto segment_size = interface->getInteger("argument_byte_length");
    const auto parameter_count = interface->getInteger("parameter_count");
    // A kernel without parameters has no argument segment.
    if (!parameter_count || (!segment_size && *parameter_count != 0)) {
      fail(std::format("no kernel argument layout for '{}'", name->str()));
    }
    layout.segment_size = static_cast<uint64_t>(segment_size.value_or(0));
    layout.arguments.resize(static_cast<size_t>(*parameter_count));
    if (const auto* parameters = interface->getArray("parameters")) {
      for (const auto& parameter_value : *parameters) {
        const auto* parameter = parameter_value.getAsObject();
        const auto index = parameter ? parameter->getInteger("index") : std::nullopt;
        const auto kind = parameter ? parameter->getString("kind") : std::nullopt;
        const auto offset =
            parameter ? parameter->getInteger("byte_offset") : std::nullopt;
        const auto length =
            parameter ? parameter->getInteger("byte_length") : std::nullopt;
        if (!index || !kind || !offset || !length || *index < 0 ||
            static_cast<size_t>(*index) >= layout.arguments.size()) {
          fail(std::format("malformed argument of '{}'", name->str()));
        }
        auto& argument = layout.arguments[static_cast<size_t>(*index)];
        argument.buffer = *kind == "binding";
        argument.offset = static_cast<uint64_t>(*offset);
        argument.length = static_cast<uint64_t>(*length);
        argument.present = true;
      }
    }
    layouts_[name->str()] = std::move(layout);
  }
}

bool Driver::device_pass() {
  if (options_.loom_target.empty()) {
    std::fprintf(stderr,
                 "%s: error: %s launches kernels; --loom-target is required\n",
                 kTool.data(), options_.source.c_str());
    throw Failure{};
  }
  std::string bytecode;
  bool has_kernels = false;
  if (!import_device_module(bytecode, has_kernels)) {
    return false;
  }
  if (!has_kernels) {
    return true;
  }
  return compile_device_module(bytecode);
}

void Driver::embed_device_code() {
  llvm::GlobalVariable* record = nullptr;
  if (!code_object_.empty()) {
    // The code object and its loom::kernel::embedded_code_object record:
    // target family, target key, data and size.
    auto& context = host_module_->getContext();
    auto* data = llvm::ConstantDataArray::get(
        context, llvm::ArrayRef<uint8_t>(
                     reinterpret_cast<const uint8_t*>(code_object_.data()),
                     code_object_.size()));
    auto* blob = new llvm::GlobalVariable(
        *host_module_, data->getType(), /*isConstant=*/true,
        llvm::GlobalValue::PrivateLinkage, data, "__loom_cxx_code_object_data");
    blob->setAlignment(llvm::Align(16));
    auto string = [&](std::string_view value, std::string_view name) {
      auto* text = llvm::ConstantDataArray::getString(context, value);
      auto* global = new llvm::GlobalVariable(
          *host_module_, text->getType(), /*isConstant=*/true,
          llvm::GlobalValue::PrivateLinkage, text, name);
      global->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
      return global;
    };
    const std::string_view target = options_.loom_target;
    const auto colon = target.find(':');
    auto* pointer_type = llvm::PointerType::getUnqual(context);
    auto* size_type = host_module_->getDataLayout().getIntPtrType(context);
    auto* record_type = llvm::StructType::get(
        context, {pointer_type, pointer_type, pointer_type, size_type});
    record = new llvm::GlobalVariable(
        *host_module_, record_type, /*isConstant=*/true,
        llvm::GlobalValue::InternalLinkage,
        llvm::ConstantStruct::get(
            record_type,
            {string(target.substr(0, colon), "__loom_cxx_target_family"),
             string(colon == std::string_view::npos ? std::string_view()
                                                    : target.substr(colon + 1),
                    "__loom_cxx_target_key"),
             blob, llvm::ConstantInt::get(size_type, code_object_.size())}),
        "__loom_cxx_code_object");
    record->setAlignment(llvm::Align(8));
  }
  define_kernel_descriptors(record);
}

// Defines every kernel_ref descriptor the host module references, in the
// module, with internal linkage (layout in loomcxx/kernel_ref.h). A kernel is
// available when the importer imported it and the device compiler compiled
// it; its descriptor names this object's code object record and the kernel's
// export ordinal there, and points to a packing function for the compiled
// kernel's argument layout. Other references get descriptors without the
// available flag.
void Driver::define_kernel_descriptors(llvm::GlobalVariable* code_object) {
  if (kernel_refs_.empty()) {
    return;
  }
  auto& context = host_module_->getContext();
  auto* i32 = llvm::Type::getInt32Ty(context);
  auto* pointer = llvm::PointerType::getUnqual(context);
  auto* descriptor_type = llvm::StructType::get(
      context, {i32, i32, pointer, llvm::ArrayType::get(i32, 3), i32, i32, i32,
                pointer, pointer});
  auto u32 = [&](uint64_t value) -> llvm::Constant* {
    return llvm::ConstantInt::get(i32, value);
  };
  auto fail = [&](const std::string& symbol, const std::string& message) {
    std::fprintf(stderr, "%s: error: kernel descriptor %s: %s\n", kTool.data(),
                 symbol.c_str(), message.c_str());
    throw Failure{};
  };
  std::vector<llvm::GlobalVariable*> descriptors;
  std::unordered_map<std::string, const loom::cxx_import::KernelLaunch*>
      launches;
  for (const auto& launch : launches_) {
    launches.emplace(launch.symbol, &launch);
  }
  for (size_t index = 0; index < kernel_refs_.size(); ++index) {
    const std::string& symbol = kernel_refs_[index];
    auto* declaration = host_module_->getGlobalVariable(symbol);
    if (!declaration || !declaration->isDeclaration()) {
      continue;
    }
    auto found = launches.find(symbol);
    const auto* launch = found == launches.end() ? nullptr : found->second;
    const ArgumentLayout* layout = nullptr;
    uint32_t export_ordinal = 0;
    if (launch && launch->available && code_object) {
      auto layout_found = layouts_.find(launch->export_name);
      auto ordinal_found = export_ordinals_.find(launch->export_name);
      if (layout_found == layouts_.end() ||
          ordinal_found == export_ordinals_.end()) {
        fail(symbol, std::format("the device code object has no export '{}'",
                                 launch->export_name));
      }
      layout = &layout_found->second;
      export_ordinal = ordinal_found->second;
    }
    const std::string name =
        launch && !launch->export_name.empty() ? launch->export_name : symbol;
    auto* name_data = llvm::ConstantDataArray::getString(context, name);
    auto* name_global = new llvm::GlobalVariable(
        *host_module_, name_data->getType(), /*isConstant=*/true,
        llvm::GlobalValue::PrivateLinkage, name_data,
        std::format("__loom_cxx_kernel_name.{}", index));
    name_global->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

    std::array<llvm::Constant*, 3> workgroup_size = {u32(0), u32(0), u32(0)};
    llvm::Constant* pack = llvm::ConstantPointerNull::get(pointer);
    llvm::Constant* code_object_reference =
        llvm::ConstantPointerNull::get(pointer);
    uint64_t segment_size = 0, parameter_count = 0;
    if (layout) {
      // Leaf i is argument i of the compiled kernel.
      if (layout->arguments.size() != launch->leaves.size()) {
        fail(symbol,
             std::format("'{}' has {} kernel arguments but {} host leaves",
                         launch->export_name, layout->arguments.size(),
                         launch->leaves.size()));
      }
      std::vector<ArgumentCopy> copies;
      for (size_t i = 0; i < launch->leaves.size(); ++i) {
        const auto& leaf = launch->leaves[i];
        const auto& argument = layout->arguments[i];
        // A pack function copies one host subobject per argument. One that
        // flattens to several arguments repeats its leaf, and no host bytes
        // hold the separate arguments.
        if (i > 0) {
          const auto& previous = launch->leaves[i - 1];
          if (leaf.kind != kLeafBuffer && previous.kind != kLeafBuffer &&
              leaf.parameter == previous.parameter &&
              leaf.source_offset == previous.source_offset &&
              leaf.size == previous.size) {
            fail(symbol,
                 std::format("parameter {} of '{}' has a host subobject at "
                             "offset {} that flattens to several kernel "
                             "arguments; loom-cxx packs one argument per "
                             "host subobject",
                             leaf.parameter, launch->export_name,
                             leaf.source_offset));
          }
        }
        const bool buffer = leaf.kind == kLeafBuffer;
        // A buffer argument carries the host pointer; a by-value argument
        // carries its payload, the low-address bytes of the host object.
        if (!argument.present || argument.buffer != buffer ||
            argument.length > leaf.size ||
            (buffer && argument.length != leaf.size) ||
            argument.offset + argument.length > layout->segment_size ||
            leaf.parameter >= launch->parameter_count) {
          fail(symbol,
               std::format("argument {} of '{}' (offset {}, {} bytes{}) does "
                           "not hold the host leaf ({} bytes{})",
                           i, launch->export_name, argument.offset,
                           argument.length, argument.buffer ? ", buffer" : "",
                           leaf.size, buffer ? ", pointer" : ""));
        }
        copies.push_back({leaf.parameter, leaf.source_offset, argument.offset,
                          argument.length});
      }
      // Arguments must not overlap: the kernarg bytes no argument covers are
      // zeroed.
      std::vector<ArgumentCopy> sorted = copies;
      std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.kernarg_offset < b.kernarg_offset;
      });
      for (size_t i = 1; i < sorted.size(); ++i) {
        if (sorted[i - 1].kernarg_offset + sorted[i - 1].length >
            sorted[i].kernarg_offset) {
          fail(symbol, std::format("kernel arguments of '{}' overlap at "
                                   "kernarg offset {}",
                                   launch->export_name,
                                   sorted[i].kernarg_offset));
        }
      }
      pack = define_packing_function(copies, layout->segment_size,
                                     launch->parameter_count, index);
      for (size_t i = 0; i < 3; ++i) {
        workgroup_size[i] = u32(launch->workgroup_size[i]);
      }
      segment_size = layout->segment_size;
      parameter_count = launch->parameter_count;
      code_object_reference = code_object;
    }
    auto* initializer = llvm::ConstantStruct::get(
        descriptor_type,
        {u32(kDescriptorVersion), u32(layout ? 1 : 0), name_global,
         llvm::ConstantArray::get(llvm::ArrayType::get(i32, 3), workgroup_size),
         u32(segment_size), u32(parameter_count), u32(export_ordinal),
         code_object_reference, pack});
    // A local definition replaces the external declaration: no kernel_ref
    // symbol leaves the object, and no other object's descriptor can
    // satisfy (or replace) this object's references.
    auto* descriptor = new llvm::GlobalVariable(
        *host_module_, descriptor_type, /*isConstant=*/true,
        llvm::GlobalValue::InternalLinkage, initializer);
    descriptor->setAlignment(llvm::Align(8));
    descriptor->takeName(declaration);
    declaration->replaceAllUsesWith(descriptor);
    declaration->eraseFromParent();
    descriptors.push_back(descriptor);
  }
  localize_descriptor_users(descriptors);
}

// Defines the loom::kernel::pack_function of one compiled kernel, with
// internal linkage: void(void* kernarg, const void* const* parameters). It
// writes all |kernarg_size| bytes of the kernel argument segment at |kernarg|:
// each argument's payload from its host parameter object, in pieces of at
// most 8 bytes with unaligned loads and stores, and zeros in every byte no
// argument covers, including the rest of the slot of a sub-word scalar. It
// reads only its arguments and references nothing else, no device code in
// particular.
llvm::Function* Driver::define_packing_function(
    const std::vector<ArgumentCopy>& copies, uint64_t kernarg_size,
    uint32_t parameter_count, size_t index) {
  auto& context = host_module_->getContext();
  auto* pointer = llvm::PointerType::getUnqual(context);
  auto* function = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(context),
                              {pointer, pointer}, /*isVarArg=*/false),
      llvm::GlobalValue::InternalLinkage,
      std::format("__loom_cxx_pack.{}", index), *host_module_);
  function->addFnAttr(llvm::Attribute::NoUnwind);
  function->addFnAttr(llvm::Attribute::NoRecurse);
  function->addFnAttr(llvm::Attribute::WillReturn);
  llvm::Value* kernarg = function->getArg(0);
  llvm::Value* parameters = function->getArg(1);
  kernarg->setName("kernarg");
  parameters->setName("parameters");
  llvm::IRBuilder<> builder(
      llvm::BasicBlock::Create(context, "entry", function));
  auto* i8 = builder.getInt8Ty();
  auto at = [&](llvm::Value* base, uint64_t offset) -> llvm::Value* {
    return offset ? builder.CreateConstInBoundsGEP1_64(i8, base, offset)
                  : base;
  };
  // Pieces of 8, 4, 2 and 1 bytes covering [offset, offset + length).
  auto pieces = [](uint64_t offset, uint64_t length, auto&& piece) {
    for (uint64_t end = offset + length; offset < end;) {
      unsigned bytes = 8;
      while (bytes > end - offset) {
        bytes /= 2;
      }
      piece(offset, bytes);
      offset += bytes;
    }
  };
  const auto& data_layout = host_module_->getDataLayout();
  const uint64_t pointer_size = data_layout.getPointerSize();
  std::vector<llvm::Value*> sources(parameter_count, nullptr);
  auto source = [&](uint32_t parameter) {
    if (!sources[parameter]) {
      sources[parameter] = builder.CreateAlignedLoad(
          pointer, at(parameters, parameter * pointer_size),
          data_layout.getPointerABIAlignment(0),
          std::format("parameter{}", parameter));
    }
    return sources[parameter];
  };
  std::vector<std::pair<uint64_t, uint64_t>> covered;
  for (const auto& copy : copies) {
    llvm::Value* from = source(copy.parameter);
    pieces(copy.kernarg_offset, copy.length, [&](uint64_t offset,
                                                 unsigned bytes) {
      auto* type = builder.getIntNTy(8 * bytes);
      const uint64_t source_offset =
          copy.source_offset + (offset - copy.kernarg_offset);
      auto* value = builder.CreateAlignedLoad(
          type, at(from, source_offset), llvm::Align(1));
      builder.CreateAlignedStore(value, at(kernarg, offset), llvm::Align(1));
    });
    covered.emplace_back(copy.kernarg_offset, copy.length);
  }
  std::sort(covered.begin(), covered.end());
  uint64_t next = 0;
  auto zero = [&](uint64_t offset, unsigned bytes) {
    builder.CreateAlignedStore(builder.getIntN(8 * bytes, 0),
                               at(kernarg, offset), llvm::Align(1));
  };
  for (const auto& [offset, length] : covered) {
    if (offset > next) {
      pieces(next, offset - next, zero);
    }
    next = std::max(next, offset + length);
  }
  if (kernarg_size > next) {
    pieces(next, kernarg_size - next, zero);
  }
  builder.CreateRetVoid();
  return function;
}

// Gives local linkage to every mergeable (inline, template) definition that
// refers to a descriptor, directly or through other definitions. The linker
// keeps one copy of a mergeable definition, which would make every object's
// launches through it use one object's descriptors (and code object): a
// kernel dropped in that object would be unavailable everywhere. Definitions
// with strong external linkage are unique to the object and stop the walk.
void Driver::localize_descriptor_users(
    const std::vector<llvm::GlobalVariable*>& descriptors) {
  std::vector<llvm::GlobalValue*> worklist(descriptors.begin(),
                                           descriptors.end());
  std::unordered_set<llvm::GlobalValue*> visited(worklist.begin(),
                                                 worklist.end());
  auto reach = [&](llvm::GlobalValue* user) {
    if (!user || user->isDeclaration() || !visited.insert(user).second) {
      return;
    }
    if (!user->hasLocalLinkage()) {
      if (!user->isWeakForLinker()) {
        return;  // strong definition: unique to this object
      }
      user->setLinkage(llvm::GlobalValue::InternalLinkage);
      if (auto* object = llvm::dyn_cast<llvm::GlobalObject>(user)) {
        object->setComdat(nullptr);
      }
    }
    worklist.push_back(user);
  };
  while (!worklist.empty()) {
    llvm::GlobalValue* value = worklist.back();
    worklist.pop_back();
    std::vector<llvm::User*> users(value->user_begin(), value->user_end());
    while (!users.empty()) {
      llvm::User* user = users.back();
      users.pop_back();
      if (auto* instruction = llvm::dyn_cast<llvm::Instruction>(user)) {
        reach(instruction->getFunction());
      } else if (auto* global = llvm::dyn_cast<llvm::GlobalValue>(user)) {
        reach(global);
      } else if (llvm::isa<llvm::Constant>(user)) {
        users.insert(users.end(), user->user_begin(), user->user_end());
      }
    }
  }
}

void Driver::emit_object() {
  auto start = Clock::now();
  // Written next to the output and renamed, so a failed compile leaves no
  // partial object behind.
  const std::string partial =
      std::format("{}.loom-cxx-{}.tmp", options_.output, (long)getpid());
  {
    std::error_code error_code;
    llvm::raw_fd_ostream out(partial, error_code);
    if (error_code) {
      std::fprintf(stderr, "%s: error: cannot open '%s': %s\n", kTool.data(),
                   partial.c_str(), error_code.message().c_str());
      throw Failure{};
    }
    std::string error;
    if (!cxx::emitMachineCode(*host_module_, target_machine_.get(), out,
                              /*assembly=*/false, error)) {
      std::fprintf(stderr, "%s: error: %s\n", kTool.data(), error.c_str());
      std::filesystem::remove(partial);
      throw Failure{};
    }
  }
  std::error_code error_code;
  std::filesystem::rename(partial, options_.output, error_code);
  if (error_code) {
    std::fprintf(stderr, "%s: error: cannot write '%s': %s\n", kTool.data(),
                 options_.output.c_str(), error_code.message().c_str());
    std::filesystem::remove(partial);
    throw Failure{};
  }
  emit_seconds_ = seconds_since(start);
}

bool Driver::run_device_pass() {
  if (!options_.keep_directory.empty()) {
    work_directory_ = options_.keep_directory;
    keep_work_directory_ = true;
    std::filesystem::create_directories(work_directory_);
  } else {
    auto base = std::filesystem::temp_directory_path() /
                std::format("loom-cxx-{}-XXXXXX", (long)getpid());
    std::string pattern = base.string();
    if (!mkdtemp(pattern.data())) {
      std::fprintf(stderr, "%s: error: cannot create a temporary directory\n",
                   kTool.data());
      return false;
    }
    work_directory_ = pattern;
  }
  bool ok = false;
  try {
    ok = device_pass();
  } catch (...) {
    if (!keep_work_directory_) {
      std::filesystem::remove_all(work_directory_);
    }
    throw;
  }
  if (!keep_work_directory_) {
    std::filesystem::remove_all(work_directory_);
  }
  return ok;
}

int Driver::run() {
  const auto before = loom::cxx_import::frontend_statistics();
  if (!parse()) {
    return 1;
  }
  write_dependency_file();
  // The device pass imports the kernels of the descriptors host code
  // references, so host code generation runs first.
  if (!generate_host_code()) {
    return 1;
  }
  if (!kernel_refs_.empty() && !run_device_pass()) {
    return 1;
  }
  embed_device_code();
  emit_object();

  if (options_.stats) {
    const auto after = loom::cxx_import::frontend_statistics();
    std::fprintf(
        stderr,
        "%s: stats: %s: preprocess=%llu parse=%llu translation_unit=%p "
        "host_codegen=1 imports=%d device_compiles=%d kernel_refs=%zu "
        "code_object_bytes=%zu times_s: parse=%.3f host=%.3f import=%.3f "
        "device_compile=%.3f emit=%.3f\n",
        kTool.data(), options_.source.c_str(),
        (unsigned long long)(after.preprocess_count - before.preprocess_count),
        (unsigned long long)(after.parse_count - before.parse_count),
        (void*)&loom::cxx_import::unit_translation_unit(
            reinterpret_cast<loom_cxx_unit_t*>(unit_.get())),
        import_count_, device_compile_count_, kernel_refs_.size(),
        code_object_.size(), parse_seconds_, host_seconds_, import_seconds_,
        device_compile_seconds_, emit_seconds_);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options options = parse_options(argc, argv);
  if (options.stats || getenv("LOOM_CXX_STATS")) {
    options.stats = true;
  }
  try {
    Driver driver(std::move(options));
    return driver.run();
  } catch (const Failure&) {
    return 1;
  }
}
