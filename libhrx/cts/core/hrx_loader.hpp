// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Dynamic loader for libhrx.so. Mirrors hip-cts HipLoader pattern.
// Thread-safe singleton, loads via dlopen at runtime.

#ifndef HRX_CTS_LOADER_HPP
#define HRX_CTS_LOADER_HPP

#include <stdexcept>
#include <string>

#include "hrx_runtime.h"

class HrxLoaderError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class HrxDynamicLibrary {
 public:
  HrxDynamicLibrary() = default;
  HrxDynamicLibrary(const HrxDynamicLibrary&) = delete;
  HrxDynamicLibrary& operator=(const HrxDynamicLibrary&) = delete;
  ~HrxDynamicLibrary();

  void load(const std::string& path);
  void* loadSymbol(const char* name);

 private:
  void* handle_ = nullptr;
};

class HrxLoader {
 public:
  static HrxLoader& instance();
  static void setLibraryPath(const std::string& path);

  // Host allocator.
  decltype(&hrx_host_allocator_system) host_allocator_system_fn;
  hrx_host_allocator_t host_allocator_system() {
    return host_allocator_system_fn();
  }

  decltype(&hrx_host_allocator_malloc) host_allocator_malloc;
  decltype(&hrx_host_allocator_malloc_uninitialized)
      host_allocator_malloc_uninitialized;
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

  // Explicit runtime and immutable groups.
  decltype(&hrx_runtime_create) runtime_create;
  decltype(&hrx_runtime_retain) runtime_retain;
  decltype(&hrx_runtime_release) runtime_release;
  decltype(&hrx_runtime_query_available_devices)
      runtime_query_available_devices;
  decltype(&hrx_device_group_create) device_group_create;
  decltype(&hrx_device_group_retain) device_group_retain;
  decltype(&hrx_device_group_release) device_group_release;
  decltype(&hrx_device_group_device_count) device_group_device_count;
  decltype(&hrx_device_group_get_device) device_group_get_device;
  decltype(&hrx_device_get_device_group) device_get_device_group;

  decltype(&hrx_slab_pool_create) slab_pool_create;
  decltype(&hrx_slab_cache_create) slab_cache_create;
  decltype(&hrx_tlsf_pool_query_backing_request)
      tlsf_pool_query_backing_request;
  decltype(&hrx_fixed_block_pool_query_backing_request)
      fixed_block_pool_query_backing_request;
  decltype(&hrx_tlsf_pool_create) tlsf_pool_create;
  decltype(&hrx_fixed_block_pool_create) fixed_block_pool_create;
  decltype(&hrx_tlsf_pool_create_from_buffer) tlsf_pool_create_from_buffer;
  decltype(&hrx_fixed_block_pool_create_from_buffer)
      fixed_block_pool_create_from_buffer;
  decltype(&hrx_mem_pool_retain) mem_pool_retain;
  decltype(&hrx_mem_pool_release) mem_pool_release;
  decltype(&hrx_mem_pool_get_device_group) mem_pool_get_device_group;
  decltype(&hrx_mem_pool_trim) mem_pool_trim;
  decltype(&hrx_mem_pool_query_capabilities) mem_pool_query_capabilities;
  decltype(&hrx_mem_pool_query_host_access) mem_pool_query_host_access;
  decltype(&hrx_mem_pool_query_atomic_operations)
      mem_pool_query_atomic_operations;
  decltype(&hrx_mem_pool_query_zero_compute_atomic_operations)
      mem_pool_query_zero_compute_atomic_operations;
  decltype(&hrx_buffer_query_atomic_operations) buffer_query_atomic_operations;
  decltype(&hrx_buffer_query_zero_compute_atomic_operations)
      buffer_query_zero_compute_atomic_operations;
  decltype(&hrx_mem_pool_query_stats) mem_pool_query_stats;
  decltype(&hrx_mem_pool_transition_table) mem_pool_transition_table;
  decltype(&hrx_buffer_transition_table) buffer_transition_table;
  decltype(&hrx_memory_transition_prepare_pair) memory_transition_prepare_pair;
  decltype(&hrx_memory_transition_recipe) memory_transition_recipe;
  decltype(&hrx_buffer_mapping_memory_barrier) buffer_mapping_memory_barrier;
  decltype(&hrx_device_group_topology_node_count)
      device_group_topology_node_count;
  decltype(&hrx_device_group_query_topology_node)
      device_group_query_topology_node;
  decltype(&hrx_queue_alloca) queue_alloca;
  decltype(&hrx_queue_dealloca) queue_dealloca;

  // Device ops.
  decltype(&hrx_device_get_property) device_get_property;
  decltype(&hrx_device_get_type) device_get_type;
  decltype(&hrx_device_retain) device_retain;
  decltype(&hrx_device_release) device_release;

  // Exact queues.
  decltype(&hrx_device_queue_family_count) device_queue_family_count;
  decltype(&hrx_device_query_queue_family) device_query_queue_family;
  decltype(&hrx_device_query_queue_execution_resource)
      device_query_queue_execution_resource;
  decltype(&hrx_device_query_queue_execution_resource_group)
      device_query_queue_execution_resource_group;
  decltype(&hrx_device_get_queue) device_get_queue;
  decltype(&hrx_queue_acquire) queue_acquire;
  decltype(&hrx_queue_retain) queue_retain;
  decltype(&hrx_queue_release) queue_release;
  decltype(&hrx_queue_get_device) queue_get_device;
  decltype(&hrx_queue_get_family_ordinal) queue_get_family_ordinal;
  decltype(&hrx_queue_query_params) queue_query_params;

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
  decltype(&hrx_stream_get_queue) stream_get_queue;
  decltype(&hrx_stream_synchronize) stream_synchronize;
  decltype(&hrx_stream_wait) stream_wait;
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
  decltype(&hrx_buffer_query_address) buffer_query_address;
  decltype(&hrx_buffer_get_size) buffer_get_size;

  // Host/device transfers.
  decltype(&hrx_synchronous_h2d) synchronous_h2d;
  decltype(&hrx_synchronous_d2h) synchronous_d2h;
  decltype(&hrx_stream_copy_h2d) stream_copy_h2d;
  decltype(&hrx_stream_copy_d2h) stream_copy_d2h;

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
  decltype(&hrx_executable_lookup_export_by_name)
      executable_lookup_export_by_name;

  // Queue ops.
  decltype(&hrx_queue_fill) queue_fill;
  decltype(&hrx_queue_copy) queue_copy;
  decltype(&hrx_queue_barrier) queue_barrier;
  decltype(&hrx_queue_dispatch) queue_dispatch;
  decltype(&hrx_queue_host_call) queue_host_call;

  // Graphs.
  decltype(&hrx_graph_create) graph_create;
  decltype(&hrx_graph_release) graph_release;
  decltype(&hrx_graph_add_kernel_node) graph_add_kernel_node;
  decltype(&hrx_graph_add_copy_buffer_node) graph_add_copy_buffer_node;
  decltype(&hrx_graph_add_fill_buffer_node) graph_add_fill_buffer_node;
  decltype(&hrx_graph_add_host_call_node) graph_add_host_call_node;
  decltype(&hrx_graph_instantiate) graph_instantiate;
  decltype(&hrx_graph_exec_release) graph_exec_release;
  decltype(&hrx_graph_exec_launch) graph_exec_launch;

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

  // Legacy scalar VMM retained for ROCr-backed HIP.
  decltype(&hrx_allocator_query_virtual_memory) allocator_query_virtual_memory;
  decltype(&hrx_allocator_virtual_memory_reserve)
      allocator_virtual_memory_reserve;
  decltype(&hrx_allocator_virtual_memory_release)
      allocator_virtual_memory_release;
  decltype(&hrx_allocator_physical_memory_allocate)
      allocator_physical_memory_allocate;
  decltype(&hrx_allocator_physical_memory_free) allocator_physical_memory_free;
  decltype(&hrx_allocator_virtual_memory_map) allocator_virtual_memory_map;
  decltype(&hrx_allocator_virtual_memory_unmap) allocator_virtual_memory_unmap;
  decltype(&hrx_allocator_virtual_memory_protect)
      allocator_virtual_memory_protect;
  decltype(&hrx_allocator_virtual_memory_protect_scoped)
      allocator_virtual_memory_protect_scoped;

  // Address owners and batched mappings.
  decltype(&hrx_device_query_host_memory_scope) device_query_host_memory_scope;
  decltype(&hrx_device_query_queue_family_memory_scope)
      device_query_queue_family_memory_scope;
  decltype(&hrx_virtual_buffer_query_capabilities)
      virtual_buffer_query_capabilities;
  decltype(&hrx_virtual_buffer_create) virtual_buffer_create;
  decltype(&hrx_virtual_buffer_query_info) virtual_buffer_query_info;
  decltype(&hrx_virtual_buffer_resolve_projection)
      virtual_buffer_resolve_projection;
  decltype(&hrx_virtual_buffer_detach) virtual_buffer_detach;
  decltype(&hrx_mem_pool_query_mapping_requirements)
      mem_pool_query_mapping_requirements;
  decltype(&hrx_queue_map) queue_map;

 private:
  HrxLoader();
  ~HrxLoader();
  void load(const std::string& path);
  void* loadSymbol(const char* name);

  HrxDynamicLibrary library_;
  static std::string library_path_;
};

inline HrxLoader& hrx() { return HrxLoader::instance(); }

#endif  // HRX_CTS_LOADER_HPP
