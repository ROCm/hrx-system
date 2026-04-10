// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_loader.hpp"

#include <cstdlib>
#include <dlfcn.h>

std::string HrxLoader::library_path_;

void HrxLoader::setLibraryPath(const std::string &path) {
  library_path_ = path;
}

HrxLoader &HrxLoader::instance() {
  static HrxLoader loader;
  return loader;
}

HrxLoader::HrxLoader() {
  std::string path = library_path_;
  if (path.empty()) {
    const char *env = std::getenv("HRX_LIBRARY");
    if (env)
      path = env;
  }
  if (path.empty()) {
    path = "libhrx.so";
  }
  load(path);
}

HrxLoader::~HrxLoader() {
  if (handle_) {
    dlclose(handle_);
  }
}

void *HrxLoader::loadSymbol(const char *name) {
  void *sym = dlsym(handle_, name);
  if (!sym) {
    throw HrxLoaderError(std::string("Failed to load symbol: ") + name + " (" +
                         dlerror() + ")");
  }
  return sym;
}

void HrxLoader::load(const std::string &path) {
  handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle_) {
    throw HrxLoaderError(std::string("Failed to load ") + path + ": " +
                         dlerror());
  }

#define LOAD(name) name = (decltype(name))loadSymbol("hrx_" #name)
#define LOAD_FULL(field, sym) field = (decltype(field))loadSymbol(#sym)

  host_allocator_system_ptr =
      (hrx_host_allocator_t *)loadSymbol("hrx_host_allocator_system_value");

  LOAD(host_allocator_malloc);
  LOAD(host_allocator_malloc_uninitialized);
  LOAD(host_allocator_realloc);
  LOAD(host_allocator_clone);
  LOAD(host_allocator_free);
  LOAD(host_allocator_malloc_aligned);
  LOAD(host_allocator_realloc_aligned);
  LOAD(host_allocator_free_aligned);

  LOAD_FULL(runtime_version, hrx_runtime_version);
  LOAD_FULL(make_status, hrx_make_status);
  LOAD_FULL(status_code, hrx_status_code);
  LOAD_FULL(status_to_string, hrx_status_to_string);
  LOAD_FULL(status_free_message, hrx_status_free_message);
  LOAD_FULL(status_ignore, hrx_status_ignore);

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
  LOAD(stream_get_device);
  LOAD(stream_get_timeline_position);
  LOAD(stream_wait_on);
  LOAD(stream_dispatch);
  LOAD(stream_execution_barrier);

  LOAD(device_allocator);
  LOAD(allocator_retain);
  LOAD(allocator_release);
  LOAD(allocator_allocate_buffer);
  LOAD(allocator_import_buffer);

  LOAD(buffer_allocate);
  LOAD(buffer_retain);
  LOAD(buffer_release);
  LOAD(buffer_map);
  LOAD(buffer_unmap);
  LOAD(buffer_get_device_ptr);
  LOAD(buffer_get_size);

  LOAD(synchronous_h2d);
  LOAD(synchronous_d2h);

  LOAD(stream_fill_buffer);
  LOAD(stream_copy_buffer);
  LOAD(stream_update_buffer);

  LOAD(executable_load_data);
  LOAD(executable_load_file);
  LOAD(executable_retain);
  LOAD(executable_release);
  LOAD(executable_export_count);
  LOAD(executable_export_info);
  LOAD(executable_lookup_export_by_name);

  LOAD(queue_fill);
  LOAD(queue_copy);
  LOAD(queue_barrier);
  LOAD(queue_dispatch);
  LOAD(queue_host_call);

  LOAD(module_load_vmfb);
  LOAD(module_retain);
  LOAD(module_release);
  LOAD(module_lookup_function);
  LOAD(function_retain);
  LOAD(function_release);
  LOAD(function_invoke);

  LOAD(value_list_create);
  LOAD(value_list_retain);
  LOAD(value_list_release);
  LOAD(value_list_size);
  LOAD(value_list_push_i64);
  LOAD(value_list_get_i64);
  LOAD(value_list_push_null_ref);
  LOAD(value_list_push_buffer);
  LOAD(value_list_push_buffer_view);
  LOAD(value_list_push_fence);

  LOAD(fence_create);
  LOAD(fence_create_at);
  LOAD(fence_retain);
  LOAD(fence_release);
  LOAD(fence_insert);
  LOAD(fence_extend);
  LOAD(fence_signal);
  LOAD(fence_wait);

  LOAD(buffer_view_create);
  LOAD(buffer_view_retain);
  LOAD(buffer_view_release);
  LOAD(buffer_view_rank);
  LOAD(buffer_view_dim);

  LOAD(allocator_query_virtual_memory);
  LOAD(allocator_virtual_memory_reserve);
  LOAD(allocator_virtual_memory_release);
  LOAD(allocator_physical_memory_allocate);
  LOAD(allocator_physical_memory_free);
  LOAD(allocator_virtual_memory_map);
  LOAD(allocator_virtual_memory_unmap);
  LOAD(allocator_virtual_memory_protect);

#undef LOAD
#undef LOAD_FULL
}
