// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "hrx_loader.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdlib>

std::string HrxLoader::library_path_;

HrxDynamicLibrary::~HrxDynamicLibrary() {
  if (!handle_) {
    return;
  }
#if defined(_WIN32)
  FreeLibrary(reinterpret_cast<HMODULE>(handle_));
#else
  dlclose(handle_);
#endif
}

void HrxDynamicLibrary::load(const std::string& path) {
  if (handle_) {
    throw HrxLoaderError("Dynamic library is already loaded");
  }
#if defined(_WIN32)
  handle_ = reinterpret_cast<void*>(LoadLibraryA(path.c_str()));
  if (!handle_) {
    throw HrxLoaderError("Failed to load " + path);
  }
#else
  handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle_) {
    throw HrxLoaderError(std::string("Failed to load ") + path + ": " +
                         dlerror());
  }
#endif
}

void* HrxDynamicLibrary::loadSymbol(const char* name) {
#if defined(_WIN32)
  void* symbol = reinterpret_cast<void*>(
      GetProcAddress(reinterpret_cast<HMODULE>(handle_), name));
#else
  void* symbol = dlsym(handle_, name);
#endif
  if (!symbol) {
#if defined(_WIN32)
    throw HrxLoaderError(std::string("Failed to load symbol: ") + name);
#else
    throw HrxLoaderError(std::string("Failed to load symbol: ") + name + " (" +
                         dlerror() + ")");
#endif
  }
  return symbol;
}

void HrxLoader::setLibraryPath(const std::string& path) {
  library_path_ = path;
}

HrxLoader& HrxLoader::instance() {
  static HrxLoader loader;
  return loader;
}

HrxLoader::HrxLoader() {
  std::string path = library_path_;
  if (path.empty()) {
    const char* env = std::getenv("HRX_LIBRARY");
    if (env) {
      path = env;
    }
  }
  if (path.empty()) {
    path = "libhrx.so";
  }
  load(path);
}

HrxLoader::~HrxLoader() = default;

void* HrxLoader::loadSymbol(const char* name) {
  return library_.loadSymbol(name);
}

void HrxLoader::load(const std::string& path) {
  library_.load(path);

#define LOAD(name) name = (decltype(name))loadSymbol("hrx_" #name)
#define LOAD_FULL(field, sym) field = (decltype(field))loadSymbol(#sym)

  LOAD_FULL(host_allocator_system_fn, hrx_host_allocator_system);

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

  LOAD(runtime_create);
  LOAD(runtime_retain);
  LOAD(runtime_release);
  LOAD(device_group_create);
  LOAD(device_group_retain);
  LOAD(device_group_release);
  LOAD(device_group_device_count);
  LOAD(device_group_get_device);

  LOAD(gpu_initialize);
  LOAD(gpu_shutdown);
  LOAD(gpu_device_count);
  LOAD(gpu_device_get);
  LOAD(cpu_initialize);
  LOAD(cpu_shutdown);
  LOAD(cpu_device_count);
  LOAD(cpu_device_get);

  LOAD(device_get_property);
  LOAD(device_get_type);
  LOAD(device_retain);
  LOAD(device_release);

  LOAD(device_queue_family_count);
  LOAD(device_query_queue_family);
  LOAD(device_query_queue_execution_resource);
  LOAD(device_query_queue_execution_resource_group);
  LOAD(device_get_queue);
  LOAD(queue_acquire);
  LOAD(queue_retain);
  LOAD(queue_release);
  LOAD(queue_get_device);
  LOAD(queue_get_family_ordinal);
  LOAD(queue_query_params);

  LOAD(semaphore_create);
  LOAD(semaphore_retain);
  LOAD(semaphore_release);
  LOAD(semaphore_query);
  LOAD(semaphore_wait);
  LOAD(semaphore_signal);

  LOAD(stream_create);
  LOAD(stream_retain);
  LOAD(stream_release);
  LOAD(stream_get_queue);
  LOAD(stream_synchronize);
  LOAD(stream_wait);
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
  LOAD(stream_copy_h2d);
  LOAD(stream_copy_d2h);

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

  LOAD(graph_create);
  LOAD(graph_release);
  LOAD(graph_add_kernel_node);
  LOAD(graph_add_copy_buffer_node);
  LOAD(graph_add_fill_buffer_node);
  LOAD(graph_add_host_call_node);
  LOAD(graph_instantiate);
  LOAD(graph_exec_release);
  LOAD(graph_exec_launch);

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
  LOAD(allocator_virtual_memory_protect_scoped);

  LOAD(device_query_host_memory_scope);
  LOAD(device_query_queue_family_memory_scope);
  LOAD(virtual_buffer_query_capabilities);
  LOAD(virtual_buffer_create);
  LOAD(virtual_buffer_query_info);
  LOAD(virtual_buffer_resolve_projection);
  LOAD(virtual_buffer_detach);
  LOAD(allocator_query_mapping_requirements);
  LOAD(queue_map);

#undef LOAD
#undef LOAD_FULL
}
