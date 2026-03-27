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
  decltype(&pyre_stream_get_timeline_position) stream_get_timeline_position;
  decltype(&pyre_stream_wait_on) stream_wait_on;

  // Buffers.
  decltype(&pyre_buffer_allocate) buffer_allocate;
  decltype(&pyre_buffer_retain) buffer_retain;
  decltype(&pyre_buffer_release) buffer_release;
  decltype(&pyre_buffer_map) buffer_map;
  decltype(&pyre_buffer_unmap) buffer_unmap;
  decltype(&pyre_buffer_get_device_ptr) buffer_get_device_ptr;

  // Stream ops.
  decltype(&pyre_stream_fill_buffer) stream_fill_buffer;
  decltype(&pyre_stream_copy_buffer) stream_copy_buffer;
  decltype(&pyre_stream_update_buffer) stream_update_buffer;

  // Queue ops.
  decltype(&pyre_queue_fill) queue_fill;
  decltype(&pyre_queue_copy) queue_copy;
  decltype(&pyre_queue_barrier) queue_barrier;

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
