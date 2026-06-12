// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/hal/drivers/amdgpu/util/global_table.h"
#include "iree/testing/benchmark.h"

#define IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_NAME_CAPACITY 32

typedef enum iree_hal_amdgpu_global_table_benchmark_mode_e {
  IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_LOOKUP_FIRST = 0,
  IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_LOOKUP_LAST,
  IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_LOOKUP_CYCLE,
  IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_BUFFER_CACHED,
} iree_hal_amdgpu_global_table_benchmark_mode_t;

typedef struct iree_hal_amdgpu_global_table_benchmark_config_t {
  // Operation measured by this benchmark row.
  iree_hal_amdgpu_global_table_benchmark_mode_t mode;

  // Number of executable globals prepopulated into the table.
  iree_host_size_t entry_count;
} iree_hal_amdgpu_global_table_benchmark_config_t;

typedef struct iree_hal_amdgpu_global_table_benchmark_name_table_t {
  // Host allocator used for name table storage.
  iree_allocator_t host_allocator;

  // Number of names in |names|.
  iree_host_size_t count;

  // String views into |storage|.
  iree_string_view_t* names;

  // Fixed-width NUL-terminated string storage for all names.
  char* storage;
} iree_hal_amdgpu_global_table_benchmark_name_table_t;

typedef struct iree_hal_amdgpu_global_table_benchmark_resolver_t {
  // Number of names the resolver accepts.
  iree_host_size_t global_count;

  // Buffer returned for every accepted global.
  iree_hal_buffer_t* buffer;
} iree_hal_amdgpu_global_table_benchmark_resolver_t;

typedef struct iree_hal_amdgpu_global_table_benchmark_fixture_t {
  // Heap-backed fake buffer retained by cached table entries.
  iree_hal_buffer_t* fake_buffer;

  // Resolver state passed to the table.
  iree_hal_amdgpu_global_table_benchmark_resolver_t resolver;

  // Persistent benchmark names used by lookup calls.
  iree_hal_amdgpu_global_table_benchmark_name_table_t name_table;

  // Global table under benchmark.
  iree_hal_amdgpu_global_table_t table;
} iree_hal_amdgpu_global_table_benchmark_fixture_t;

static bool iree_hal_amdgpu_global_table_benchmark_try_parse_name(
    iree_string_view_t name, iree_host_size_t global_count,
    iree_host_size_t* out_global_ordinal) {
  *out_global_ordinal = 0;

  static const iree_string_view_t prefix = IREE_SV("global_");
  if (!iree_string_view_starts_with(name, prefix)) {
    return false;
  }
  iree_string_view_t ordinal_string =
      iree_string_view_strip_prefix(name, prefix);
  uint64_t global_ordinal = 0;
  if (!iree_string_view_atoi_uint64_base(ordinal_string, 10, &global_ordinal)) {
    return false;
  }
  if (global_ordinal >= global_count) {
    return false;
  }

  *out_global_ordinal = (iree_host_size_t)global_ordinal;
  return true;
}

static iree_status_t iree_hal_amdgpu_global_table_benchmark_try_verify(
    void* user_data, iree_string_view_t name,
    iree_host_size_t verification_physical_device_ordinal, bool* out_found,
    iree_device_size_t* out_byte_length) {
  (void)verification_physical_device_ordinal;
  *out_found = false;
  *out_byte_length = 0;

  iree_hal_amdgpu_global_table_benchmark_resolver_t* resolver =
      (iree_hal_amdgpu_global_table_benchmark_resolver_t*)user_data;
  iree_host_size_t global_ordinal = 0;
  if (!iree_hal_amdgpu_global_table_benchmark_try_parse_name(
          name, resolver->global_count, &global_ordinal)) {
    return iree_ok_status();
  }

  *out_found = true;
  *out_byte_length = 64;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_global_table_benchmark_create_buffer(
    void* user_data, iree_string_view_t name,
    iree_device_size_t expected_byte_length,
    iree_hal_queue_affinity_t selected_queue_affinity,
    iree_host_size_t physical_device_ordinal, iree_hal_buffer_t** out_buffer) {
  (void)expected_byte_length;
  (void)selected_queue_affinity;
  (void)physical_device_ordinal;
  iree_hal_amdgpu_global_table_benchmark_resolver_t* resolver =
      (iree_hal_amdgpu_global_table_benchmark_resolver_t*)user_data;

  iree_host_size_t global_ordinal = 0;
  if (!iree_hal_amdgpu_global_table_benchmark_try_parse_name(
          name, resolver->global_count, &global_ordinal)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "verified benchmark global disappeared");
  }

  iree_hal_buffer_retain(resolver->buffer);
  *out_buffer = resolver->buffer;
  return iree_ok_status();
}

static iree_hal_amdgpu_queue_affinity_domain_t
iree_hal_amdgpu_global_table_benchmark_domain(void) {
  return (iree_hal_amdgpu_queue_affinity_domain_t){
      /*.supported_affinity=*/0x1ull,
      /*.physical_device_count=*/1,
      /*.queue_count_per_physical_device=*/1,
  };
}

static void iree_hal_amdgpu_global_table_benchmark_buffer_release(
    void* user_data, iree_hal_buffer_t* buffer) {
  (void)buffer;
  iree_allocator_free_aligned(iree_allocator_system(), user_data);
}

static iree_status_t iree_hal_amdgpu_global_table_benchmark_create_heap_buffer(
    iree_hal_buffer_t** out_buffer) {
  *out_buffer = NULL;

  void* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      iree_allocator_system(), 64, IREE_HAL_HEAP_BUFFER_ALIGNMENT,
      /*offset=*/0, &storage));

  iree_hal_buffer_release_callback_t release_callback = {
      /*.fn=*/iree_hal_amdgpu_global_table_benchmark_buffer_release,
      /*.user_data=*/storage,
  };
  iree_status_t status = iree_hal_heap_buffer_wrap(
      iree_hal_buffer_placement_undefined(), IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_USAGE_DEFAULT, 64, iree_make_byte_span(storage, 64),
      release_callback, iree_allocator_system(), out_buffer);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free_aligned(iree_allocator_system(), storage);
  }
  return status;
}

static void iree_hal_amdgpu_global_table_benchmark_name_table_deinitialize(
    iree_hal_amdgpu_global_table_benchmark_name_table_t* name_table) {
  iree_allocator_free(name_table->host_allocator, name_table->storage);
  iree_allocator_free(name_table->host_allocator, name_table->names);
  memset(name_table, 0, sizeof(*name_table));
}

static iree_status_t
iree_hal_amdgpu_global_table_benchmark_name_table_initialize(
    iree_host_size_t count, iree_allocator_t host_allocator,
    iree_hal_amdgpu_global_table_benchmark_name_table_t* out_name_table) {
  IREE_ASSERT_ARGUMENT(out_name_table);
  memset(out_name_table, 0, sizeof(*out_name_table));
  out_name_table->host_allocator = host_allocator;
  out_name_table->count = count;

  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, count, sizeof(out_name_table->names[0]),
      (void**)&out_name_table->names);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, count,
        IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_NAME_CAPACITY,
        (void**)&out_name_table->storage);
  }

  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    char* storage = out_name_table->storage +
                    i * IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_NAME_CAPACITY;
    int name_length =
        snprintf(storage, IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_NAME_CAPACITY,
                 "global_%04" PRIhsz, i);
    if (name_length <= 0 ||
        (iree_host_size_t)name_length >=
            IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_NAME_CAPACITY) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "benchmark global name for ordinal %" PRIhsz
          " exceeds storage capacity %" PRIhsz,
          i,
          (iree_host_size_t)
              IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_NAME_CAPACITY);
    } else {
      out_name_table->names[i] =
          iree_make_string_view(storage, (iree_host_size_t)name_length);
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_global_table_benchmark_name_table_deinitialize(
        out_name_table);
  }
  return status;
}

static void iree_hal_amdgpu_global_table_benchmark_fixture_deinitialize(
    iree_hal_amdgpu_global_table_benchmark_fixture_t* fixture) {
  iree_hal_amdgpu_global_table_deinitialize(&fixture->table);
  iree_hal_amdgpu_global_table_benchmark_name_table_deinitialize(
      &fixture->name_table);
  iree_hal_buffer_release(fixture->fake_buffer);
  memset(fixture, 0, sizeof(*fixture));
}

static iree_status_t iree_hal_amdgpu_global_table_benchmark_fixture_initialize(
    const iree_hal_amdgpu_global_table_benchmark_config_t* config,
    iree_hal_amdgpu_global_table_benchmark_fixture_t* out_fixture) {
  IREE_ASSERT_ARGUMENT(config);
  IREE_ASSERT_ARGUMENT(out_fixture);
  memset(out_fixture, 0, sizeof(*out_fixture));

  iree_status_t status =
      iree_hal_amdgpu_global_table_benchmark_create_heap_buffer(
          &out_fixture->fake_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_global_table_benchmark_name_table_initialize(
        config->entry_count, iree_allocator_system(), &out_fixture->name_table);
  }
  if (iree_status_is_ok(status)) {
    out_fixture->resolver.global_count = config->entry_count;
    out_fixture->resolver.buffer = out_fixture->fake_buffer;
  }

  const iree_hal_amdgpu_global_table_params_t params = {
      /*.host_allocator=*/iree_allocator_system(),
      /*.queue_affinity_domain=*/
      iree_hal_amdgpu_global_table_benchmark_domain(),
      /*.loaded_physical_device_mask=*/0x1ull,
      /*.physical_device_count=*/1,
      /*.resolver=*/
      {
          /*.user_data=*/&out_fixture->resolver,
          /*.try_verify=*/iree_hal_amdgpu_global_table_benchmark_try_verify,
          /*.create_buffer=*/
          iree_hal_amdgpu_global_table_benchmark_create_buffer,
      },
  };
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_amdgpu_global_table_initialize(&params, &out_fixture->table);
  }

  for (iree_host_size_t i = 0;
       i < out_fixture->name_table.count && iree_status_is_ok(status); ++i) {
    iree_hal_executable_global_t global = iree_hal_executable_global_invalid();
    status = iree_hal_amdgpu_global_table_lookup(
        &out_fixture->table, out_fixture->name_table.names[i], &global);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_global_table_benchmark_fixture_deinitialize(out_fixture);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_global_table_benchmark_run_lookup(
    iree_hal_amdgpu_global_table_benchmark_mode_t mode,
    iree_hal_amdgpu_global_table_benchmark_fixture_t* fixture,
    iree_benchmark_state_t* benchmark_state) {
  iree_host_size_t global_ordinal = 0;
  if (mode == IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_LOOKUP_LAST) {
    global_ordinal = fixture->name_table.count - 1;
  }

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_hal_executable_global_t cached_global =
        iree_hal_executable_global_invalid();
    status = iree_hal_amdgpu_global_table_lookup(
        &fixture->table, fixture->name_table.names[global_ordinal],
        &cached_global);
    iree_optimization_barrier(cached_global.value);

    if (mode == IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_LOOKUP_CYCLE) {
      ++global_ordinal;
      if (global_ordinal == fixture->name_table.count) {
        global_ordinal = 0;
      }
    }
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_global_table_benchmark_run_buffer(
    iree_hal_executable_global_t global,
    iree_hal_amdgpu_global_table_benchmark_fixture_t* fixture,
    iree_benchmark_state_t* benchmark_state) {
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_hal_buffer_t* cached_buffer = NULL;
    status = iree_hal_amdgpu_global_table_buffer(
        &fixture->table, global, IREE_HAL_QUEUE_AFFINITY_ANY, &cached_buffer);
    iree_optimization_barrier(cached_buffer);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_global_table_benchmark_run(
    const iree_benchmark_def_t* benchmark_def,
    iree_benchmark_state_t* benchmark_state) {
  const iree_hal_amdgpu_global_table_benchmark_config_t* config =
      (const iree_hal_amdgpu_global_table_benchmark_config_t*)
          benchmark_def->user_data;

  iree_hal_amdgpu_global_table_benchmark_fixture_t fixture;
  iree_hal_executable_global_t buffer_global =
      iree_hal_executable_global_invalid();

  iree_status_t status =
      iree_hal_amdgpu_global_table_benchmark_fixture_initialize(config,
                                                                &fixture);
  if (iree_status_is_ok(status) &&
      config->mode == IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_BUFFER_CACHED) {
    status = iree_hal_amdgpu_global_table_lookup(
        &fixture.table, fixture.name_table.names[fixture.name_table.count - 1],
        &buffer_global);
  }
  if (iree_status_is_ok(status) &&
      config->mode == IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_BUFFER_CACHED) {
    iree_hal_buffer_t* cached_buffer = NULL;
    status = iree_hal_amdgpu_global_table_buffer(&fixture.table, buffer_global,
                                                 IREE_HAL_QUEUE_AFFINITY_ANY,
                                                 &cached_buffer);
    iree_optimization_barrier(cached_buffer);
  }

  if (iree_status_is_ok(status)) {
    switch (config->mode) {
      case IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_LOOKUP_FIRST:
      case IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_LOOKUP_LAST:
      case IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_LOOKUP_CYCLE:
        status = iree_hal_amdgpu_global_table_benchmark_run_lookup(
            config->mode, &fixture, benchmark_state);
        break;
      case IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_BUFFER_CACHED:
        status = iree_hal_amdgpu_global_table_benchmark_run_buffer(
            buffer_global, &fixture, benchmark_state);
        break;
    }
  }

  iree_hal_amdgpu_global_table_benchmark_fixture_deinitialize(&fixture);
  return status;
}

#define IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_REGISTER(                       \
    suffix, name, mode_value, entry_count_value)                               \
  static const iree_hal_amdgpu_global_table_benchmark_config_t                 \
      iree_hal_amdgpu_global_table_benchmark_config_##suffix = {               \
          /*.mode=*/mode_value,                                                \
          /*.entry_count=*/entry_count_value,                                  \
  };                                                                           \
  static const iree_benchmark_def_t                                            \
      iree_hal_amdgpu_global_table_benchmark_def_##suffix = {                  \
          /*.flags=*/{},                                                       \
          /*.time_unit=*/IREE_BENCHMARK_UNIT_NANOSECOND,                       \
          /*.minimum_duration_ns=*/{},                                         \
          /*.iteration_count=*/{},                                             \
          /*.run=*/iree_hal_amdgpu_global_table_benchmark_run, /*.user_data=*/ \
          &iree_hal_amdgpu_global_table_benchmark_config_##suffix,             \
  };                                                                           \
  static const iree_benchmark_def_t*                                           \
      iree_hal_amdgpu_global_table_benchmark_registration_##suffix             \
          IREE_ATTRIBUTE_UNUSED = iree_benchmark_register(                     \
              iree_make_cstring_view(name),                                    \
              &iree_hal_amdgpu_global_table_benchmark_def_##suffix)

#define IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_REGISTER_FOR_COUNT(suffix,      \
                                                                  entry_count) \
  IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_REGISTER(                             \
      lookup_first_##suffix,                                                   \
      "BM_GlobalTableLookupCachedFirst/entries:" #entry_count,                 \
      IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_LOOKUP_FIRST, entry_count);       \
  IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_REGISTER(                             \
      lookup_last_##suffix,                                                    \
      "BM_GlobalTableLookupCachedLast/entries:" #entry_count,                  \
      IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_LOOKUP_LAST, entry_count);        \
  IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_REGISTER(                             \
      lookup_cycle_##suffix,                                                   \
      "BM_GlobalTableLookupCachedCycle/entries:" #entry_count,                 \
      IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_LOOKUP_CYCLE, entry_count);       \
  IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_REGISTER(                             \
      buffer_cached_##suffix,                                                  \
      "BM_GlobalTableBufferCached/entries:" #entry_count,                      \
      IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_BUFFER_CACHED, entry_count)

IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_REGISTER_FOR_COUNT(1, 1);
IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_REGISTER_FOR_COUNT(8, 8);
IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_REGISTER_FOR_COUNT(64, 64);
IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_REGISTER_FOR_COUNT(256, 256);
IREE_HAL_AMDGPU_GLOBAL_TABLE_BENCHMARK_REGISTER_FOR_COUNT(1024, 1024);
