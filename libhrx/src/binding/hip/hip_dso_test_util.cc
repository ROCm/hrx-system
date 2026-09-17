// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "binding/hip/hip_dso_test_util.h"

#include <dlfcn.h>
#include <elf.h>
#include <limits.h>
#include <link.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

namespace hrx::hip::testing {
namespace {

const char* ConfiguredPath() {
  const char* environment_path = std::getenv("HRX_TEST_LIBAMDHIP64");
  if (environment_path && environment_path[0] != '\0') {
    return environment_path;
  }
#ifdef HRX_TEST_LIBAMDHIP64_PATH
  return HRX_TEST_LIBAMDHIP64_PATH;
#else
  return nullptr;
#endif
}

std::string CanonicalPath(const char* path) {
  char canonical_path[PATH_MAX];
  return path && realpath(path, canonical_path) ? canonical_path : "";
}

}  // namespace

HipDso::~HipDso() {
  if (handle_) {
    dlclose(handle_);
  }
}

bool HipDso::Open() {
  if (handle_) {
    return true;
  }
  error_.clear();

  const char* configured_path = ConfiguredPath();
  if (!configured_path) {
    error_ = "the build did not provide a libamdhip64 artifact";
    return false;
  }
  canonical_path_ = CanonicalPath(configured_path);
  if (canonical_path_.empty()) {
    error_ = std::string("cannot canonicalize ") + configured_path;
    return false;
  }

  dlerror();
  handle_ = dlopen(configured_path, RTLD_NOW | RTLD_LOCAL);
  if (!handle_) {
    const char* loader_error = dlerror();
    error_ = std::string("cannot dlopen ") + configured_path + ": " +
             (loader_error ? loader_error : "unknown loader error");
    return false;
  }

  link_map* loaded_map = nullptr;
  if (dlinfo(handle_, RTLD_DI_LINKMAP, &loaded_map) != 0 || !loaded_map ||
      CanonicalPath(loaded_map->l_name) != canonical_path_) {
    error_ = "the loader opened a different libamdhip64 artifact";
    dlclose(handle_);
    handle_ = nullptr;
    return false;
  }
  return true;
}

bool HipDso::Close() {
  if (!handle_) {
    return true;
  }
  void* handle = handle_;
  handle_ = nullptr;
  canonical_path_.clear();
  if (dlclose(handle) == 0) {
    return true;
  }
  const char* loader_error = dlerror();
  error_ = loader_error ? loader_error : "dlclose failed";
  return false;
}

void* HipDso::ResolveRaw(const char* name) {
  error_.clear();
  if (!handle_ || !name || name[0] == '\0') {
    error_ = "symbol resolution requires an open DSO and a non-empty name";
    return nullptr;
  }

  dlerror();
  void* symbol = dlsym(handle_, name);
  const char* loader_error = dlerror();
  if (loader_error || !symbol) {
    error_ = std::string("cannot resolve ") + name + ": " +
             (loader_error ? loader_error : "symbol not found");
    return nullptr;
  }

  Dl_info symbol_info = {};
  if (dladdr(symbol, &symbol_info) == 0 || !symbol_info.dli_fname ||
      CanonicalPath(symbol_info.dli_fname) != canonical_path_) {
    error_ = std::string(name) + " resolved from a different DSO";
    return nullptr;
  }
  return symbol;
}

void* HipDso::ResolveLocalForTestRaw(const char* name) {
  error_.clear();
  if (!handle_ || !name || name[0] == 0) {
    error_ =
        "local symbol resolution requires an open DSO and a non-empty name";
    return nullptr;
  }

  std::ifstream file(canonical_path_, std::ios::binary | std::ios::ate);
  if (!file) {
    error_ = "cannot read the exact DSO for local symbol resolution";
    return nullptr;
  }
  const std::streamoff file_size = file.tellg();
  if (file_size < static_cast<std::streamoff>(sizeof(Elf64_Ehdr))) {
    error_ = "the exact DSO has an invalid ELF header";
    return nullptr;
  }
  std::vector<char> image(static_cast<size_t>(file_size));
  file.seekg(0, std::ios::beg);
  if (!file.read(image.data(), static_cast<std::streamsize>(image.size()))) {
    error_ = "cannot read the complete exact DSO";
    return nullptr;
  }

  const auto* header = reinterpret_cast<const Elf64_Ehdr*>(image.data());
  if (std::memcmp(header->e_ident, ELFMAG, SELFMAG) != 0 ||
      header->e_ident[EI_CLASS] != ELFCLASS64 ||
      header->e_ident[EI_DATA] != ELFDATA2LSB || header->e_type != ET_DYN ||
      header->e_shentsize != sizeof(Elf64_Shdr) || header->e_shnum == 0 ||
      header->e_shoff > image.size() ||
      header->e_shnum > (image.size() - header->e_shoff) / sizeof(Elf64_Shdr)) {
    error_ = "the exact DSO has an unsupported ELF layout";
    return nullptr;
  }

  const auto* sections = reinterpret_cast<const Elf64_Shdr*>(
      image.data() + static_cast<size_t>(header->e_shoff));
  Elf64_Addr symbol_value = 0;
  bool found = false;
  for (Elf64_Half i = 0; i < header->e_shnum && !found; ++i) {
    const Elf64_Shdr& symbol_section = sections[i];
    if (symbol_section.sh_type != SHT_SYMTAB ||
        symbol_section.sh_entsize != sizeof(Elf64_Sym) ||
        symbol_section.sh_link >= header->e_shnum ||
        symbol_section.sh_offset > image.size() ||
        symbol_section.sh_size > image.size() - symbol_section.sh_offset) {
      continue;
    }
    const Elf64_Shdr& string_section = sections[symbol_section.sh_link];
    if (string_section.sh_type != SHT_STRTAB ||
        string_section.sh_offset > image.size() ||
        string_section.sh_size > image.size() - string_section.sh_offset) {
      continue;
    }
    const auto* symbols = reinterpret_cast<const Elf64_Sym*>(
        image.data() + static_cast<size_t>(symbol_section.sh_offset));
    const char* strings =
        image.data() + static_cast<size_t>(string_section.sh_offset);
    const size_t symbol_count =
        static_cast<size_t>(symbol_section.sh_size / symbol_section.sh_entsize);
    for (size_t j = 0; j < symbol_count; ++j) {
      const Elf64_Sym& symbol = symbols[j];
      if (symbol.st_name >= string_section.sh_size ||
          symbol.st_shndx == SHN_UNDEF ||
          ELF64_ST_TYPE(symbol.st_info) != STT_FUNC) {
        continue;
      }
      const char* symbol_name = strings + symbol.st_name;
      const size_t remaining =
          static_cast<size_t>(string_section.sh_size - symbol.st_name);
      if (!std::memchr(symbol_name, 0, remaining) ||
          std::strcmp(symbol_name, name) != 0) {
        continue;
      }
      symbol_value = symbol.st_value;
      found = true;
      break;
    }
  }
  if (!found) {
    error_ = std::string("cannot resolve private test symbol ") + name +
             " from the unstripped exact DSO";
    return nullptr;
  }

  link_map* loaded_map = nullptr;
  if (dlinfo(handle_, RTLD_DI_LINKMAP, &loaded_map) != 0 || !loaded_map) {
    error_ = "cannot determine the exact DSO load address";
    return nullptr;
  }
  void* symbol = reinterpret_cast<void*>(loaded_map->l_addr + symbol_value);
  Dl_info symbol_info = {};
  if (dladdr(symbol, &symbol_info) == 0 || !symbol_info.dli_fname ||
      CanonicalPath(symbol_info.dli_fname) != canonical_path_) {
    error_ = std::string(name) + " resolved from a different DSO";
    return nullptr;
  }
  return symbol;
}

}  // namespace hrx::hip::testing
