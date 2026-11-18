// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_FILE_TRANSFER_H_
#define IREE_HAL_UTILS_FILE_TRANSFER_H_

#include "iree/hal/file.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable operation data placed in caller-owned storage. The enclosing
// operation owns all references; this value has no separate resource lifetime.
typedef struct iree_hal_file_transfer_batch_t {
  // Captured, qualified file windows; the enclosing operation retains them.
  iree_hal_file_binding_table_t files;
  // Captured, qualified payload windows with resolved explicit lengths.
  iree_hal_buffer_binding_table_t buffers;
  // Captured host values or indirect metadata, whose bytes remain unread.
  iree_hal_file_transfer_list_t transfers;
} iree_hal_file_transfer_batch_t;

// Validates external list/table shapes and computes their capture storage.
// No resources or indirect bytes are accessed. The enclosing operation places
// this extent in its existing allocation, with system allocator alignment.
iree_status_t iree_hal_file_transfer_batch_storage_size(
    iree_hal_file_binding_table_t files,
    iree_hal_buffer_binding_table_t buffers,
    const iree_hal_file_transfer_list_t* transfers, iree_host_size_t* out_size);

// Captures operands into storage sized by batch_storage_size for these same
// inputs. file_access is READ or WRITE; payload access is the opposite.
// Resource permissions and host descriptions are validated now; indirect bytes
// remain unread until dependency-ready acquisition. The enclosing operation
// retains captured references after success and owns storage even on failure.
// No allocation, signal, I/O, cache action, or resource retain occurs here.
iree_status_t iree_hal_file_transfer_batch_capture(
    iree_hal_queue_t* queue, iree_hal_memory_access_t file_access,
    iree_hal_file_binding_table_t files,
    iree_hal_buffer_binding_table_t buffers,
    const iree_hal_file_transfer_list_t* transfers, void* storage,
    iree_hal_file_transfer_batch_t* out_batch);

// A contiguous extent in the original binding coordinates. A backend can
// admit only a prefix when constrained by staging or native request capacity.
typedef struct iree_hal_file_transfer_range_t {
  // File table index, already validated.
  uint32_t file_binding;
  // Buffer table index, already validated.
  uint32_t buffer_binding;
  // First byte relative to the file binding's origin.
  uint64_t file_offset;
  // First byte relative to the buffer binding's origin.
  iree_device_size_t buffer_offset;
  // Contiguous bytes remaining; zero indicates end of the acquired prefix.
  uint64_t length;
} iree_hal_file_transfer_range_t;

// Single-progress-owner cursor. Space is constant in row count and logical
// length. Adjacent rows/slices collapse when both endpoints are contiguous.
typedef struct iree_hal_file_transfer_iterator_t {
  // Borrowed immutable capture, live through iteration.
  const iree_hal_file_transfer_batch_t* batch;
  // Next acquired indirect record, or NULL for captured host descriptions.
  const iree_hal_file_transfer_record_t* records;
  // Live prefix, sampled once after waits and bounded by declared capacity.
  iree_host_size_t count;
  // Index of the next description not yet decoded.
  iree_host_size_t index;
  // End of the acquired prefix, at most count. A page can end between records.
  iree_host_size_t acquired;
  // Current normalized description; row_length zero means it is exhausted.
  iree_hal_file_transfer_t current;
  // Row within the current slice.
  uint64_t row;
  // Slice within the current description.
  uint64_t slice;
  // Admitted prefix of the current contiguous row.
  uint64_t position;
} iree_hal_file_transfer_iterator_t;

// Begins iteration after producer waits and metadata visibility. For INDIRECT,
// count is the sampled live count, checked against the captured capacity.
// records points to that readable prefix or is NULL until the caller acquires
// it with iterator_acquire before the first peek. HOST uses captured
// values and ignores these two parameters. Metadata stays stable until retired.
iree_status_t iree_hal_file_transfer_iterator_initialize(
    const iree_hal_file_transfer_batch_t* batch, uint64_t count,
    const iree_hal_file_transfer_record_t* records,
    iree_hal_file_transfer_iterator_t* out_iterator);

// Makes the next consecutive page readable. The caller has acquired exactly
// this prefix after producer dependencies, within the validated live count.
// Decoded geometry owns its values, so prior pages can be reused immediately.
// This trusted handoff performs no validation, allocation, or synchronization.
void iree_hal_file_transfer_iterator_acquire(
    iree_hal_file_transfer_iterator_t* iterator,
    const iree_hal_file_transfer_record_t* records, iree_host_size_t count);

// Returns the next contiguous range without advancing it. Indirect descriptions
// are decoded and validated exactly once as encountered. Host descriptions
// consume capture-time validation. A zero length means end of the acquired
// prefix; index < count then requires acquiring more records. Other
// output fields are then unspecified. A malformed indirect record fails before
// returning its first range, without undoing earlier admissions.
iree_status_t iree_hal_file_transfer_iterator_peek(
    iree_hal_file_transfer_iterator_t* iterator,
    iree_hal_file_transfer_range_t* out_range);

// Consumes a nonempty prefix of the last returned range. This is trusted native
// admission information and cannot fail. No allocation or synchronization.
void iree_hal_file_transfer_iterator_advance(
    iree_hal_file_transfer_iterator_t* iterator, uint64_t length);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_UTILS_FILE_TRANSFER_H_
