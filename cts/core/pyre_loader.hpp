// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// Dynamic loader for libpyre.so. Mirrors hip-cts HipLoader pattern.
// Thread-safe singleton, loads via dlopen at runtime.

#ifndef PYRE_CTS_LOADER_HPP
#define PYRE_CTS_LOADER_HPP

#include "pyre_runtime.h"

#include <stdexcept>
#include <string>

class PyreLoaderError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class PyreLoader {
 public:
  static PyreLoader& instance();
  static void setLibraryPath(const std::string& path);

  // Host allocator. system_value is a data symbol loaded via dlsym.
  pyre_host_allocator_t* host_allocator_system_ptr;
  pyre_host_allocator_t host_allocator_system() {
    return *host_allocator_system_ptr;
  }

  decltype(&pyre_host_allocator_malloc) host_allocator_malloc;
  decltype(&pyre_host_allocator_malloc_uninitialized) host_allocator_malloc_uninitialized;
  decltype(&pyre_host_allocator_realloc) host_allocator_realloc;
  decltype(&pyre_host_allocator_clone) host_allocator_clone;
  decltype(&pyre_host_allocator_free) host_allocator_free;
  decltype(&pyre_host_allocator_malloc_aligned) host_allocator_malloc_aligned;
  decltype(&pyre_host_allocator_realloc_aligned) host_allocator_realloc_aligned;
  decltype(&pyre_host_allocator_free_aligned) host_allocator_free_aligned;

  // Runtime version.
  decltype(&pyre_runtime_version) runtime_version;

  // Status API.
  decltype(&pyre_make_status) make_status;
  decltype(&pyre_status_code) status_code;
  decltype(&pyre_status_to_string) status_to_string;
  decltype(&pyre_status_free_message) status_free_message;
  decltype(&pyre_status_ignore) status_ignore;

  // GPU lifecycle.
  decltype(&pyre_gpu_initialize) gpu_initialize;
  decltype(&pyre_gpu_shutdown) gpu_shutdown;
  decltype(&pyre_gpu_device_count) gpu_device_count;
  decltype(&pyre_gpu_device_get) gpu_device_get;

  // CPU lifecycle.
  decltype(&pyre_cpu_initialize) cpu_initialize;
  decltype(&pyre_cpu_shutdown) cpu_shutdown;
  decltype(&pyre_cpu_device_count) cpu_device_count;
  decltype(&pyre_cpu_device_get) cpu_device_get;

  // Device ops.
  decltype(&pyre_device_get_property) device_get_property;
  decltype(&pyre_device_synchronize) device_synchronize;
  decltype(&pyre_device_get_type) device_get_type;
  decltype(&pyre_device_retain) device_retain;
  decltype(&pyre_device_release) device_release;

  // Semaphores.
  decltype(&pyre_semaphore_create) semaphore_create;
  decltype(&pyre_semaphore_retain) semaphore_retain;
  decltype(&pyre_semaphore_release) semaphore_release;
  decltype(&pyre_semaphore_query) semaphore_query;
  decltype(&pyre_semaphore_wait) semaphore_wait;
  decltype(&pyre_semaphore_signal) semaphore_signal;

  // Streams.
  decltype(&pyre_stream_create) stream_create;
  decltype(&pyre_stream_retain) stream_retain;
  decltype(&pyre_stream_release) stream_release;
  decltype(&pyre_stream_synchronize) stream_synchronize;
  decltype(&pyre_stream_query) stream_query;
  decltype(&pyre_stream_flush) stream_flush;
  decltype(&pyre_stream_get_semaphore) stream_get_semaphore;
  decltype(&pyre_stream_get_device) stream_get_device;
  decltype(&pyre_stream_get_timeline_position) stream_get_timeline_position;
  decltype(&pyre_stream_wait_on) stream_wait_on;
  decltype(&pyre_stream_dispatch) stream_dispatch;
  decltype(&pyre_stream_execution_barrier) stream_execution_barrier;

  // Allocator.
  decltype(&pyre_device_allocator) device_allocator;
  decltype(&pyre_allocator_retain) allocator_retain;
  decltype(&pyre_allocator_release) allocator_release;
  decltype(&pyre_allocator_allocate_buffer) allocator_allocate_buffer;
  decltype(&pyre_allocator_import_buffer) allocator_import_buffer;

  // Buffers.
  decltype(&pyre_buffer_allocate) buffer_allocate;
  decltype(&pyre_buffer_retain) buffer_retain;
  decltype(&pyre_buffer_release) buffer_release;
  decltype(&pyre_buffer_map) buffer_map;
  decltype(&pyre_buffer_unmap) buffer_unmap;
  decltype(&pyre_buffer_get_device_ptr) buffer_get_device_ptr;
  decltype(&pyre_buffer_get_size) buffer_get_size;

  // Synchronous transfers.
  decltype(&pyre_synchronous_h2d) synchronous_h2d;
  decltype(&pyre_synchronous_d2h) synchronous_d2h;

  // Stream ops.
  decltype(&pyre_stream_fill_buffer) stream_fill_buffer;
  decltype(&pyre_stream_copy_buffer) stream_copy_buffer;
  decltype(&pyre_stream_update_buffer) stream_update_buffer;

  // Direct executables.
  decltype(&pyre_executable_load_data) executable_load_data;
  decltype(&pyre_executable_load_file) executable_load_file;
  decltype(&pyre_executable_retain) executable_retain;
  decltype(&pyre_executable_release) executable_release;
  decltype(&pyre_executable_export_count) executable_export_count;
  decltype(&pyre_executable_export_info) executable_export_info;
  decltype(&pyre_executable_lookup_export_by_name) executable_lookup_export_by_name;

  // Queue ops.
  decltype(&pyre_queue_fill) queue_fill;
  decltype(&pyre_queue_copy) queue_copy;
  decltype(&pyre_queue_barrier) queue_barrier;
  decltype(&pyre_queue_dispatch) queue_dispatch;
  decltype(&pyre_queue_host_call) queue_host_call;

  // VM modules/functions.
  decltype(&pyre_module_load_vmfb) module_load_vmfb;
  decltype(&pyre_module_retain) module_retain;
  decltype(&pyre_module_release) module_release;
  decltype(&pyre_module_lookup_function) module_lookup_function;
  decltype(&pyre_function_retain) function_retain;
  decltype(&pyre_function_release) function_release;
  decltype(&pyre_function_invoke) function_invoke;

  // VM value lists.
  decltype(&pyre_value_list_create) value_list_create;
  decltype(&pyre_value_list_retain) value_list_retain;
  decltype(&pyre_value_list_release) value_list_release;
  decltype(&pyre_value_list_size) value_list_size;
  decltype(&pyre_value_list_push_i64) value_list_push_i64;
  decltype(&pyre_value_list_get_i64) value_list_get_i64;
  decltype(&pyre_value_list_push_null_ref) value_list_push_null_ref;
  decltype(&pyre_value_list_push_buffer) value_list_push_buffer;
  decltype(&pyre_value_list_push_buffer_view) value_list_push_buffer_view;
  decltype(&pyre_value_list_push_fence) value_list_push_fence;

  // Fences.
  decltype(&pyre_fence_create) fence_create;
  decltype(&pyre_fence_create_at) fence_create_at;
  decltype(&pyre_fence_retain) fence_retain;
  decltype(&pyre_fence_release) fence_release;
  decltype(&pyre_fence_insert) fence_insert;
  decltype(&pyre_fence_extend) fence_extend;
  decltype(&pyre_fence_signal) fence_signal;
  decltype(&pyre_fence_wait) fence_wait;

  // Buffer views.
  decltype(&pyre_buffer_view_create) buffer_view_create;
  decltype(&pyre_buffer_view_retain) buffer_view_retain;
  decltype(&pyre_buffer_view_release) buffer_view_release;
  decltype(&pyre_buffer_view_rank) buffer_view_rank;
  decltype(&pyre_buffer_view_dim) buffer_view_dim;

  // Virtual memory.
  decltype(&pyre_allocator_query_virtual_memory) allocator_query_virtual_memory;
  decltype(&pyre_allocator_virtual_memory_reserve) allocator_virtual_memory_reserve;
  decltype(&pyre_allocator_virtual_memory_release) allocator_virtual_memory_release;
  decltype(&pyre_allocator_physical_memory_allocate) allocator_physical_memory_allocate;
  decltype(&pyre_allocator_physical_memory_free) allocator_physical_memory_free;
  decltype(&pyre_allocator_virtual_memory_map) allocator_virtual_memory_map;
  decltype(&pyre_allocator_virtual_memory_unmap) allocator_virtual_memory_unmap;
  decltype(&pyre_allocator_virtual_memory_protect) allocator_virtual_memory_protect;

 private:
  PyreLoader();
  ~PyreLoader();
  void load(const std::string& path);
  void* loadSymbol(const char* name);

  void* handle_ = nullptr;
  static std::string library_path_;
};

inline PyreLoader& pyre() { return PyreLoader::instance(); }

#endif  // PYRE_CTS_LOADER_HPP
