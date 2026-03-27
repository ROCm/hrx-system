// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include "pyre_loader.hpp"

#include <dlfcn.h>
#include <cstdlib>

std::string PyreLoader::library_path_;

void PyreLoader::setLibraryPath(const std::string& path) {
  library_path_ = path;
}

PyreLoader& PyreLoader::instance() {
  static PyreLoader loader;
  return loader;
}

PyreLoader::PyreLoader() {
  std::string path = library_path_;
  if (path.empty()) {
    const char* env = std::getenv("PYRE_LIBRARY");
    if (env) path = env;
  }
  if (path.empty()) {
    path = "libpyre.so";
  }
  load(path);
}

PyreLoader::~PyreLoader() {
  if (handle_) {
    dlclose(handle_);
  }
}

void* PyreLoader::loadSymbol(const char* name) {
  void* sym = dlsym(handle_, name);
  if (!sym) {
    throw PyreLoaderError(std::string("Failed to load symbol: ") + name +
                          " (" + dlerror() + ")");
  }
  return sym;
}

void PyreLoader::load(const std::string& path) {
  handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle_) {
    throw PyreLoaderError(std::string("Failed to load ") + path +
                          ": " + dlerror());
  }

#define LOAD(name) name = (decltype(name))loadSymbol("pyre_" #name)
#define LOAD_FULL(field, sym) field = (decltype(field))loadSymbol(#sym)

  LOAD_FULL(runtime_version, pyre_runtime_version);
  LOAD_FULL(make_status, pyre_make_status);
  LOAD_FULL(status_code, pyre_status_code);
  LOAD_FULL(status_to_string, pyre_status_to_string);
  LOAD_FULL(status_free_message, pyre_status_free_message);
  LOAD_FULL(status_ignore, pyre_status_ignore);

  LOAD(gpu_initialize);
  LOAD(gpu_shutdown);
  LOAD(gpu_device_count);
  LOAD(gpu_device_get);
  LOAD(cpu_initialize);
  LOAD(cpu_shutdown);
  LOAD(cpu_device_count);
  LOAD(cpu_device_get);

  LOAD(device_get_property);
  LOAD(device_synchronize);
  LOAD(device_get_type);
  LOAD(device_retain);
  LOAD(device_release);

  LOAD(semaphore_create);
  LOAD(semaphore_retain);
  LOAD(semaphore_release);
  LOAD(semaphore_query);
  LOAD(semaphore_wait);
  LOAD(semaphore_signal);

  LOAD(stream_create);
  LOAD(stream_retain);
  LOAD(stream_release);
  LOAD(stream_synchronize);
  LOAD(stream_query);
  LOAD(stream_flush);
  LOAD(stream_get_semaphore);
  LOAD(stream_get_timeline_position);
  LOAD(stream_wait_on);

  LOAD(buffer_allocate);
  LOAD(buffer_retain);
  LOAD(buffer_release);
  LOAD(buffer_map);
  LOAD(buffer_unmap);
  LOAD(buffer_get_device_ptr);

  LOAD(stream_fill_buffer);
  LOAD(stream_copy_buffer);
  LOAD(stream_update_buffer);

  LOAD(queue_fill);
  LOAD(queue_copy);
  LOAD(queue_barrier);

#undef LOAD
#undef LOAD_FULL
}
