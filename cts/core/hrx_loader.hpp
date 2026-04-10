// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Dynamic loader for libhrx.so. Mirrors hip-cts HipLoader pattern.
// Thread-safe singleton, loads via dlopen at runtime.

#ifndef HRX_CTS_LOADER_HPP
#define HRX_CTS_LOADER_HPP

#include "hrx_runtime.h"

#include <stdexcept>
#include <string>

class HrxLoaderError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class HrxLoader {
 public:
  static HrxLoader& instance();
  static void setLibraryPath(const std::string& path);

  // Host allocator. system_value is a data symbol loaded via dlsym.
  hrx_host_allocator_t* host_allocator_system_ptr;
  hrx_host_allocator_t host_allocator_system() {
    return *host_allocator_system_ptr;
  }

  decltype(&hrx_host_allocator_malloc) host_allocator_malloc;
  decltype(&hrx_host_allocator_malloc_uninitialized) host_allocator_malloc_uninitialized;
  decltype(&hrx_host_allocator_realloc) host_allocator_realloc;
  decltype(&hrx_host_allocator_clone) host_allocator_clone;
  decltype(&hrx_host_allocator_free) host_allocator_free;
  decltype(&hrx_host_allocator_malloc_aligned) host_allocator_malloc_aligned;
  decltype(&hrx_host_allocator_realloc_aligned) host_allocator_realloc_aligned;
  decltype(&hrx_host_allocator_free_aligned) host_allocator_free_aligned;

  // Runtime version.
  decltype(&hrx_runtime_version) runtime_version;

  // Status API.
  decltype(&hrx_make_status) make_status;
  decltype(&hrx_status_code) status_code;
  decltype(&hrx_status_to_string) status_to_string;
  decltype(&hrx_status_free_message) status_free_message;
  decltype(&hrx_status_ignore) status_ignore;

  // GPU lifecycle.
  decltype(&hrx_gpu_initialize) gpu_initialize;
  decltype(&hrx_gpu_shutdown) gpu_shutdown;
  decltype(&hrx_gpu_device_count) gpu_device_count;
  decltype(&hrx_gpu_device_get) gpu_device_get;

  // CPU lifecycle.
  decltype(&hrx_cpu_initialize) cpu_initialize;
  decltype(&hrx_cpu_shutdown) cpu_shutdown;
  decltype(&hrx_cpu_device_count) cpu_device_count;
  decltype(&hrx_cpu_device_get) cpu_device_get;

  // Device ops.
  decltype(&hrx_device_get_property) device_get_property;
  decltype(&hrx_device_synchronize) device_synchronize;
  decltype(&hrx_device_get_type) device_get_type;
  decltype(&hrx_device_retain) device_retain;
  decltype(&hrx_device_release) device_release;

  // Semaphores.
  decltype(&hrx_semaphore_create) semaphore_create;
  decltype(&hrx_semaphore_retain) semaphore_retain;
  decltype(&hrx_semaphore_release) semaphore_release;
  decltype(&hrx_semaphore_query) semaphore_query;
  decltype(&hrx_semaphore_wait) semaphore_wait;
  decltype(&hrx_semaphore_signal) semaphore_signal;

  // Streams.
  decltype(&hrx_stream_create) stream_create;
  decltype(&hrx_stream_retain) stream_retain;
  decltype(&hrx_stream_release) stream_release;
  decltype(&hrx_stream_synchronize) stream_synchronize;
  decltype(&hrx_stream_query) stream_query;
  decltype(&hrx_stream_flush) stream_flush;
  decltype(&hrx_stream_get_semaphore) stream_get_semaphore;
  decltype(&hrx_stream_get_device) stream_get_device;
  decltype(&hrx_stream_get_timeline_position) stream_get_timeline_position;
  decltype(&hrx_stream_wait_on) stream_wait_on;
  decltype(&hrx_stream_dispatch) stream_dispatch;
  decltype(&hrx_stream_execution_barrier) stream_execution_barrier;

  // Allocator.
  decltype(&hrx_device_allocator) device_allocator;
  decltype(&hrx_allocator_retain) allocator_retain;
  decltype(&hrx_allocator_release) allocator_release;
  decltype(&hrx_allocator_allocate_buffer) allocator_allocate_buffer;
  decltype(&hrx_allocator_import_buffer) allocator_import_buffer;

  // Buffers.
  decltype(&hrx_buffer_allocate) buffer_allocate;
  decltype(&hrx_buffer_retain) buffer_retain;
  decltype(&hrx_buffer_release) buffer_release;
  decltype(&hrx_buffer_map) buffer_map;
  decltype(&hrx_buffer_unmap) buffer_unmap;
  decltype(&hrx_buffer_get_device_ptr) buffer_get_device_ptr;
  decltype(&hrx_buffer_get_size) buffer_get_size;

  // Synchronous transfers.
  decltype(&hrx_synchronous_h2d) synchronous_h2d;
  decltype(&hrx_synchronous_d2h) synchronous_d2h;

  // Stream ops.
  decltype(&hrx_stream_fill_buffer) stream_fill_buffer;
  decltype(&hrx_stream_copy_buffer) stream_copy_buffer;
  decltype(&hrx_stream_update_buffer) stream_update_buffer;

  // Direct executables.
  decltype(&hrx_executable_load_data) executable_load_data;
  decltype(&hrx_executable_load_file) executable_load_file;
  decltype(&hrx_executable_retain) executable_retain;
  decltype(&hrx_executable_release) executable_release;
  decltype(&hrx_executable_export_count) executable_export_count;
  decltype(&hrx_executable_export_info) executable_export_info;
  decltype(&hrx_executable_lookup_export_by_name) executable_lookup_export_by_name;

  // Queue ops.
  decltype(&hrx_queue_fill) queue_fill;
  decltype(&hrx_queue_copy) queue_copy;
  decltype(&hrx_queue_barrier) queue_barrier;
  decltype(&hrx_queue_dispatch) queue_dispatch;
  decltype(&hrx_queue_host_call) queue_host_call;

  // VM modules/functions.
  decltype(&hrx_module_load_vmfb) module_load_vmfb;
  decltype(&hrx_module_retain) module_retain;
  decltype(&hrx_module_release) module_release;
  decltype(&hrx_module_lookup_function) module_lookup_function;
  decltype(&hrx_function_retain) function_retain;
  decltype(&hrx_function_release) function_release;
  decltype(&hrx_function_invoke) function_invoke;

  // VM value lists.
  decltype(&hrx_value_list_create) value_list_create;
  decltype(&hrx_value_list_retain) value_list_retain;
  decltype(&hrx_value_list_release) value_list_release;
  decltype(&hrx_value_list_size) value_list_size;
  decltype(&hrx_value_list_push_i64) value_list_push_i64;
  decltype(&hrx_value_list_get_i64) value_list_get_i64;
  decltype(&hrx_value_list_push_null_ref) value_list_push_null_ref;
  decltype(&hrx_value_list_push_buffer) value_list_push_buffer;
  decltype(&hrx_value_list_push_buffer_view) value_list_push_buffer_view;
  decltype(&hrx_value_list_push_fence) value_list_push_fence;

  // Fences.
  decltype(&hrx_fence_create) fence_create;
  decltype(&hrx_fence_create_at) fence_create_at;
  decltype(&hrx_fence_retain) fence_retain;
  decltype(&hrx_fence_release) fence_release;
  decltype(&hrx_fence_insert) fence_insert;
  decltype(&hrx_fence_extend) fence_extend;
  decltype(&hrx_fence_signal) fence_signal;
  decltype(&hrx_fence_wait) fence_wait;

  // Buffer views.
  decltype(&hrx_buffer_view_create) buffer_view_create;
  decltype(&hrx_buffer_view_retain) buffer_view_retain;
  decltype(&hrx_buffer_view_release) buffer_view_release;
  decltype(&hrx_buffer_view_rank) buffer_view_rank;
  decltype(&hrx_buffer_view_dim) buffer_view_dim;

  // Virtual memory.
  decltype(&hrx_allocator_query_virtual_memory) allocator_query_virtual_memory;
  decltype(&hrx_allocator_virtual_memory_reserve) allocator_virtual_memory_reserve;
  decltype(&hrx_allocator_virtual_memory_release) allocator_virtual_memory_release;
  decltype(&hrx_allocator_physical_memory_allocate) allocator_physical_memory_allocate;
  decltype(&hrx_allocator_physical_memory_free) allocator_physical_memory_free;
  decltype(&hrx_allocator_virtual_memory_map) allocator_virtual_memory_map;
  decltype(&hrx_allocator_virtual_memory_unmap) allocator_virtual_memory_unmap;
  decltype(&hrx_allocator_virtual_memory_protect) allocator_virtual_memory_protect;

 private:
  HrxLoader();
  ~HrxLoader();
  void load(const std::string& path);
  void* loadSymbol(const char* name);

  void* handle_ = nullptr;
  static std::string library_path_;
};

inline HrxLoader& hrx() { return HrxLoader::instance(); }

#endif  // HRX_CTS_LOADER_HPP
