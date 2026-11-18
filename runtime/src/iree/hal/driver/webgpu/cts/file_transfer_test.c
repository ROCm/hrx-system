// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A wasm reactor: each exported step returns to the JS event loop. The host
// resumes only when the real proactor publishes the required semaphore value.
#include <stdio.h>

#include "iree/async/util/proactor_pool.h"
#include "iree/hal/api.h"
#include "iree/hal/driver/webgpu/webgpu_device.h"
#include "iree/hal/driver/webgpu/webgpu_queue.h"
#include "iree/io/file_handle.h"

#define EXPORT(name) __attribute__((export_name(#name)))
#define ROW_COUNT 65
#define RECORD_CAPACITY 68
#define ORIGIN 33
#define FILE_STRIDE 192
#define BUFFER_STRIDE 128
#define FILE_LENGTH (2 * ORIGIN + ROW_COUNT * FILE_STRIDE)
#define BUFFER_LENGTH (2 * ORIGIN + ROW_COUNT * BUFFER_STRIDE)
#define BULK_LENGTH (3 * 1024 * 1024 + 31)
#define MAX_LENGTH (BULK_LENGTH + 2 * ORIGIN)

enum {
  GATE,
  INITIALIZED,
  READY,
  LOADED,
  WRITTEN,
  FINISHED,
  INDEPENDENT_READ,
  INDEPENDENT_FINISHED,
  SEMAPHORE_COUNT,
};

static iree_async_proactor_pool_t* proactor_pool;
static iree_async_proactor_t* proactor;
static iree_hal_device_t* device;
static iree_hal_device_group_t* group;
static iree_hal_pool_t* pool;
static iree_hal_webgpu_queue_t* queue;
static iree_hal_semaphore_t* semaphores[SEMAPHORE_COUNT];
static uint64_t one = 1;
static uint8_t input[MAX_LENGTH];
static uint8_t output[MAX_LENGTH];
static uint8_t expected_file[MAX_LENGTH];
static uint8_t expected_memory[MAX_LENGTH];
static uint8_t initial[MAX_LENGTH];
static uint8_t expected[MAX_LENGTH];
static uint8_t actual[MAX_LENGTH];
static uint64_t metadata_words[8 + RECORD_CAPACITY * 8];
static uint32_t independent_input = 0x35791ACE;
static uint32_t independent_output;
static uint32_t released_files;
static uint32_t imported_files;
static bool invalid;
static size_t current_file_length;
static size_t current_buffer_length;

static int report(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return 0;
  }
  iree_status_fprint(stderr, status);
  return (int)iree_status_consume_code(status);
}

static iree_hal_semaphore_list_t point(int index) {
  return (iree_hal_semaphore_list_t){1, &semaphores[index], &one};
}

static void file_released(void* user_data,
                          iree_io_file_handle_primitive_t primitive) {
  (void)user_data;
  (void)primitive;
  ++released_files;
}

static iree_status_t import_file(void* bytes, size_t length, int fd,
                                 iree_hal_file_t** out_file) {
  iree_io_file_handle_t* handle = NULL;
  const iree_io_file_handle_release_callback_t release = {file_released, NULL};
  iree_status_t status;
  if (fd >= 0) {
    const iree_io_file_handle_primitive_t primitive = {
        .type = IREE_IO_FILE_HANDLE_TYPE_FD, .value = {.fd = fd}};
    status = iree_io_file_handle_wrap(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE, 0, primitive,
        release, iree_allocator_system(), &handle);
  } else {
    status = iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(bytes, length), release, iree_allocator_system(),
        &handle);
  }
  if (iree_status_is_ok(status)) {
    ++imported_files;
    status = iree_hal_file_import(
        device, NULL, 1,
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, handle, 0,
        out_file);
    iree_io_file_handle_release(handle);
  }
  return status;
}

static iree_status_t allocate(size_t length, iree_hal_buffer_t** out_buffer) {
  return iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(length, 1),
      iree_infinite_timeout(), out_buffer);
}

// Each operand is captured before this stack frame and its file reference go
// away. The backing host bytes remain owned by the test until final completion.
static iree_status_t linear(iree_hal_memory_access_t access,
                            iree_hal_semaphore_list_t waits,
                            iree_hal_semaphore_list_t signals, void* bytes,
                            size_t length, iree_hal_buffer_t* buffer) {
  iree_hal_file_t* file = NULL;
  IREE_RETURN_IF_ERROR(import_file(bytes, length, -1, &file));
  iree_status_t status =
      access == IREE_HAL_MEMORY_ACCESS_READ
          ? iree_hal_queue_read_linear((iree_hal_queue_t*)queue, waits, signals,
                                       file, 0, buffer, 0, length, NULL, 0)
          : iree_hal_queue_write_linear((iree_hal_queue_t*)queue, waits,
                                        signals, buffer, 0, file, 0, length,
                                        NULL, 0);
  iree_hal_file_release(file);
  return status;
}

EXPORT(initialize)
int initialize(uint32_t device_handle) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_status_t status = iree_async_proactor_pool_create(
      1, NULL, iree_async_proactor_pool_options_default(), allocator,
      &proactor_pool);
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_get(proactor_pool, 0, &proactor);
  }
  iree_hal_device_create_params_t params =
      iree_hal_device_create_params_default();
  params.proactor_pool = proactor_pool;
  if (iree_status_is_ok(status)) {
    status = iree_hal_webgpu_device_create(iree_make_cstring_view("webgpu"),
                                           device_handle, 0, &params, allocator,
                                           &device);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_group_create_from_device(device, NULL, allocator,
                                                      &group);
  }
  if (iree_status_is_ok(status)) {
    queue = (iree_hal_webgpu_queue_t*)iree_hal_device_queue(device, 0, 0);
    const iree_hal_pool_family_access_t family = {
        .family = iree_hal_device_queue_family(device, 0),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER};
    const iree_hal_pool_scope_t scope = {.family_count = 1,
                                         .families = &family};
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    status =
        iree_hal_slab_pool_create(group, scope, &options, allocator, &pool);
  }
  return report(status);
}

EXPORT(proactor_drain)
int proactor_drain(void) {
  iree_host_size_t completed = 0;
  iree_status_t status =
      iree_async_proactor_poll(proactor, iree_immediate_timeout(), &completed);
  if (iree_status_is_deadline_exceeded(status)) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }
  return report(status);
}

EXPORT(begin)
int begin(uint32_t form, uint32_t count) {
  const bool indirect = (form & 2) != 0;
  const bool strided = (form & 1) != 0;
  const bool bulk = (form & 4) != 0;
  const uint32_t origin = bulk && strided ? 32 : ORIGIN;
  const uint64_t row_length =
      bulk ? (strided ? 1024 * 1024 - 4 : BULK_LENGTH) : 90;
  const uint64_t file_stride = bulk ? 1024 * 1024 : FILE_STRIDE;
  const uint64_t buffer_stride = bulk ? 1024 * 1024 - 1 : BUFFER_STRIDE;
  current_file_length = bulk ? MAX_LENGTH : FILE_LENGTH;
  current_buffer_length = bulk ? MAX_LENGTH : BUFFER_LENGTH;
  invalid = count > ROW_COUNT;
  imported_files = released_files = 0;
  independent_output = 0;
  for (size_t i = 0; i < current_file_length; ++i) {
    input[i] = (uint8_t)(i * 17 + 3);
  }
  memset(output, 0xCD, current_file_length);
  memset(expected_file, 0xCD, current_file_length);
  memset(expected_memory, 0xCD, current_file_length);
  memset(initial, 0xA5, current_buffer_length);
  memcpy(expected, initial, current_buffer_length);
  memset(actual, 0, current_buffer_length);
  memset(metadata_words, 0xFF, sizeof(metadata_words));
  metadata_words[0] = strided ? 1 : count;
  iree_hal_file_transfer_t rows[ROW_COUNT] = {0};
  for (uint32_t i = 0; i < iree_min(count, ROW_COUNT); ++i) {
    const uint32_t file_index = !strided && i >= 32 ? i & 1 : 0;
    rows[i] = (iree_hal_file_transfer_t){
        file_index, 0,  i * file_stride, i * buffer_stride, row_length, 1, 1,
        {0},        {0}};
    metadata_words[8 + i * 8] = i * file_stride | ((uint64_t)file_index << 48);
    metadata_words[9 + i * 8] = i * buffer_stride;
    metadata_words[10 + i * 8] = row_length;
    memcpy(expected + origin + i * buffer_stride,
           input + origin + i * file_stride, row_length);
    memcpy((file_index ? expected_memory : expected_file) + origin +
               i * file_stride,
           input + origin + i * file_stride, row_length);
  }
  if (strided) {
    const uint64_t row_count = count == ROW_COUNT ? 5 : count;
    const uint64_t slice_count = count == ROW_COUNT ? 13 : 1;
    rows[0] =
        (iree_hal_file_transfer_t){0,
                                   0,
                                   0,
                                   0,
                                   row_length,
                                   row_count,
                                   slice_count,
                                   {file_stride, row_count * file_stride},
                                   {buffer_stride, row_count * buffer_stride}};
    metadata_words[8] = metadata_words[9] = 0;
    metadata_words[10] = row_length | (UINT64_C(1) << 48);
    metadata_words[11] = file_stride | (row_count << 48);
    metadata_words[12] = buffer_stride;
    metadata_words[13] = row_count * file_stride | (slice_count << 48);
    metadata_words[14] = row_count * buffer_stride;
  } else if (invalid) {
    metadata_words[8 + ROW_COUNT * 8] = UINT64_C(2) << 48;
    metadata_words[9 + ROW_COUNT * 8] = 0;
    metadata_words[10 + ROW_COUNT * 8] = row_length;
  }
  iree_status_t status = iree_ok_status();
  for (int i = 0; i < SEMAPHORE_COUNT && iree_status_is_ok(status); ++i) {
    status = iree_hal_semaphore_create(device, 1, 0, 0, &semaphores[i]);
  }
  iree_hal_buffer_t* payload = NULL;
  iree_hal_buffer_t* metadata = NULL;
  iree_hal_buffer_t* independent = NULL;
  if (iree_status_is_ok(status)) {
    status = allocate(current_buffer_length, &payload);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_t* allocation = NULL;
    const size_t offset = strided ? 3 : 0;
    status = allocate(sizeof(metadata_words) + offset, &allocation);
    if (iree_status_is_ok(status)) {
      // Logical metadata offsets are aligned, but a HAL view may begin at
      // an unaligned native offset. Acquiring it still preserves exact bytes.
      status =
          iree_hal_buffer_subspan(allocation, offset, sizeof(metadata_words),
                                  iree_allocator_system(), &metadata);
    }
    iree_hal_buffer_release(allocation);
  }
  if (iree_status_is_ok(status)) {
    status = allocate(4, &independent);
  }
  iree_hal_file_t* files[4] = {NULL};
  if (iree_status_is_ok(status)) {
    status = import_file(NULL, current_file_length, 1, &files[0]);
  }
  if (iree_status_is_ok(status)) {
    status = import_file(input, current_file_length, -1, &files[1]);
  }
  if (iree_status_is_ok(status)) {
    status = import_file(NULL, current_file_length, 2, &files[2]);
  }
  if (iree_status_is_ok(status)) {
    status = import_file(output, current_file_length, -1, &files[3]);
  }
  if (iree_status_is_ok(status)) {
    status =
        linear(IREE_HAL_MEMORY_ACCESS_READ, iree_hal_semaphore_list_empty(),
               point(INITIALIZED), initial, current_buffer_length, payload);
  }
  iree_hal_semaphore_t* producers[] = {semaphores[GATE],
                                       semaphores[INITIALIZED]};
  uint64_t values[] = {1, 1};
  if (iree_status_is_ok(status)) {
    status =
        linear(IREE_HAL_MEMORY_ACCESS_READ,
               (iree_hal_semaphore_list_t){2, producers, values}, point(READY),
               metadata_words, sizeof(metadata_words), metadata);
  }
  iree_hal_file_binding_t sources[] = {
      {files[0], origin, current_file_length - 2 * origin},
      {files[1], origin, current_file_length - 2 * origin}};
  iree_hal_file_binding_t targets[] = {
      {files[2], origin, current_file_length - 2 * origin},
      {files[3], origin, current_file_length - 2 * origin}};
  iree_hal_buffer_binding_t buffer = {payload, origin,
                                      current_buffer_length - 2 * origin};
  iree_hal_file_transfer_list_t list = {0};
  if (indirect) {
    list.type = IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT;
    list.indirect.records =
        (iree_hal_buffer_binding_t){metadata, 64, RECORD_CAPACITY * 64};
    list.indirect.count_buffer = metadata;
    list.indirect.count_offset = 0;
  } else {
    list.host.count = strided ? 1 : count;
    list.host.transfers = rows;
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_read((iree_hal_queue_t*)queue, point(READY),
                                 point(LOADED),
                                 (iree_hal_file_binding_table_t){2, sources},
                                 (iree_hal_buffer_binding_table_t){1, &buffer},
                                 &list, NULL, IREE_HAL_READ_FLAG_NONE);
  }
  if (iree_status_is_ok(status) && !invalid) {
    status = iree_hal_queue_write((iree_hal_queue_t*)queue, point(LOADED),
                                  point(WRITTEN),
                                  (iree_hal_buffer_binding_table_t){1, &buffer},
                                  (iree_hal_file_binding_table_t){2, targets},
                                  &list, NULL, IREE_HAL_WRITE_FLAG_NONE);
  }
  if (iree_status_is_ok(status) && !invalid) {
    status = linear(IREE_HAL_MEMORY_ACCESS_WRITE, point(WRITTEN),
                    point(FINISHED), actual, current_buffer_length, payload);
  }
  if (iree_status_is_ok(status)) {
    status =
        linear(IREE_HAL_MEMORY_ACCESS_READ, iree_hal_semaphore_list_empty(),
               point(INDEPENDENT_READ), &independent_input, 4, independent);
  }
  if (iree_status_is_ok(status)) {
    status = linear(IREE_HAL_MEMORY_ACCESS_WRITE, point(INDEPENDENT_READ),
                    point(INDEPENDENT_FINISHED), &independent_output, 4,
                    independent);
  }
  // Captures must be sufficient after all caller-owned references and HOST
  // descriptions disappear, including before the producer gate opens.
  memset(rows, 0xFF, sizeof(rows));
  for (int i = 0; i < 4; ++i) {
    iree_hal_file_release(files[i]);
  }
  iree_hal_buffer_release(independent);
  iree_hal_buffer_release(metadata);
  iree_hal_buffer_release(payload);
  return report(status);
}

EXPORT(query)
int query(uint32_t independent) {
  uint64_t value = 0;
  iree_status_t status =
      iree_hal_semaphore_query(semaphores[independent ? INDEPENDENT_FINISHED
                                          : invalid   ? LOADED
                                                      : FINISHED],
                               &value);
  if (!iree_status_is_ok(status)) {
    return -report(status);
  }
  return value == 1;
}

EXPORT(open_gate)
int open_gate(void) {
  uint64_t value = 0;
  iree_status_t status = iree_hal_semaphore_query(semaphores[READY], &value);
  if (iree_status_is_ok(status) &&
      (value || independent_output != independent_input)) {
    status =
        iree_make_status(IREE_STATUS_DATA_LOSS,
                         "independent progress or producer ordering failed");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_signal(semaphores[GATE], 1, NULL);
  }
  return report(status);
}

EXPORT(finish)
int finish(void) {
  iree_status_t status = iree_ok_status();
  if ((!invalid && (memcmp(actual, expected, current_buffer_length) ||
                    memcmp(output, expected_memory, current_file_length))) ||
      released_files != imported_files) {
    status =
        iree_make_status(IREE_STATUS_DATA_LOSS,
                         "byte preservation or file retirement failed (%u/%u)",
                         released_files, imported_files);
  }
  for (int i = 0; i < SEMAPHORE_COUNT; ++i) {
    iree_hal_semaphore_release(semaphores[i]);
  }
  return report(status);
}

EXPORT(expected_file_pointer)
uint32_t expected_file_pointer(void) {
  return (uint32_t)(uintptr_t)expected_file;
}

EXPORT(file_length)
uint32_t file_length(uint32_t form) {
  return form & 4 ? MAX_LENGTH : FILE_LENGTH;
}

EXPORT(shutdown)
void shutdown(void) {
  iree_hal_pool_release(pool);
  iree_hal_device_group_release(group);
  iree_hal_device_release(device);
  iree_async_proactor_pool_release(proactor_pool);
}
