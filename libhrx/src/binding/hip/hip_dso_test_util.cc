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
#include <limits.h>
#include <link.h>

#include <cstdlib>

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
  if (handle_) dlclose(handle_);
}

bool HipDso::Open() {
  if (handle_) return true;
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
  if (!handle_) return true;
  void* handle = handle_;
  handle_ = nullptr;
  canonical_path_.clear();
  if (dlclose(handle) == 0) return true;
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

}  // namespace hrx::hip::testing
