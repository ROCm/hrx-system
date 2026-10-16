// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_FABRIC_FILE_TRANSFER_H_
#define IREE_HAL_DRIVERS_AMD_FABRIC_FILE_TRANSFER_H_

#include "amdf/amdf.h"
#include "iree/async/operations/file.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Consumes terminal I/O status on the imported file's proactor. Successful
// reads still require the engine's host publication or native copy before
// their bytes become visible to buffer consumers.
typedef struct iree_hal_amd_fabric_file_transfer_callback_t {
  // Called exactly once after an accepted chunk retires, including on failure.
  void (*fn)(void* user_data, iree_status_t status);
  // Borrowed engine operation owning the transfer storage.
  void* user_data;
} iree_hal_amd_fabric_file_transfer_callback_t;

typedef struct iree_hal_amd_fabric_file_host_range_t
    iree_hal_amd_fabric_file_host_range_t;

// Placed state for a file transfer through a host execution binding or an
// engine's native buffer copy. The engine owns dependency resolution, staging,
// cache boundaries, and final signaling. Native completion only advances I/O.
// Initialization and deinitialization bracket accepted I/O and the engine's
// final cache work.
typedef struct iree_hal_amd_fabric_file_transfer_t {
  // Allocator for ordinary host staging, independent of device memory pools.
  iree_allocator_t host_allocator;
  // Retained file keeping its imported handle or host allocation live.
  iree_hal_file_t* file;
  // Retained logical buffer; backing retirement remains caller-ordered.
  iree_hal_buffer_t* buffer;
  // Exactly READ (file to buffer) or WRITE (buffer to file).
  iree_hal_memory_access_t file_access;
  // Byte position within the file.
  uint64_t file_offset;
  // Byte position within the logical buffer.
  iree_device_size_t buffer_offset;
  // Exact requested byte count.
  iree_device_size_t length;
  // Borrowed host file storage when no native I/O handle is present.
  iree_byte_span_t file_contents;
  // Host execution view resolved only after producer dependencies complete.
  iree_byte_span_t buffer_contents;
  // Private views of physical sources underlying an execution-only virtual
  // buffer. These views borrow the application's installed backing lifetimes.
  struct {
    // Borrowed native API, dominated by the engine's retained device.
    const amdf_api_t* api;
    // Borrowed destination for native view cleanup diagnostics.
    iree_hal_device_event_sink_t event_sink;
    // Owned host views in logical file order; no physical backing is retained.
    iree_hal_amd_fabric_file_host_range_t* ranges;
    // Number of initialized ranges, including any partially prepared tail.
    iree_host_size_t count;
    // Current I/O range; advances only after its complete extent retires.
    iree_host_size_t index;
  } host;
  // Platform I/O state, unused for memory files.
  struct {
    // Native handle retained through |file|; its proactor remains borrowed.
    iree_async_file_t* file;
    // Owned ordinary host staging reused by file chunks, or empty.
    iree_byte_span_t staging;
    // Current chunk, borrowed from staging or the buffer's host execution view.
    struct {
      // Transfer-relative position of the chunk's first byte.
      uint64_t offset;
      // Complete chunk extent, including any successfully completed prefix.
      iree_byte_span_t contents;
    } chunk;
    // Byte count completed across all native requests.
    uint64_t bytes_transferred;
    // Terminal callback returning ownership to the engine.
    iree_hal_amd_fabric_file_transfer_callback_t callback;
    union {
      // Placed read request reused after each partial completion.
      iree_async_file_read_operation_t read;
      // Placed write request reused after each partial completion.
      iree_async_file_write_operation_t write;
    };
  } async;
} iree_hal_amd_fabric_file_transfer_t;

// Captures a nonempty transfer already validated by HAL queue_read/write.
// Supports memory files and platform files imported for async I/O. Failure
// leaves an empty transfer that may be deinitialized. Buffer storage need not
// be committed yet.
iree_status_t iree_hal_amd_fabric_file_transfer_initialize(
    iree_hal_file_t* file, iree_hal_memory_access_t file_access,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length,
    iree_allocator_t host_allocator,
    iree_hal_amd_fabric_file_transfer_t* out_transfer);

// Releases captured operands after all native I/O and host work have retired,
// or when an operation is rejected before starting. Does not free placed state.
void iree_hal_amd_fabric_file_transfer_deinitialize(
    iree_hal_amd_fabric_file_transfer_t* transfer);

// Prepares direct buffered I/O through the physical sources of an AMDF virtual
// buffer, after mapping dependencies complete and before executing barriers.
// Queue visibility is combined over the entire operation; CPU actions are
// qualified against each physical source and performed only when required.
// Unsupported backing returns OK with available=false and no acquired views.
// Mapping failures, holes, native errors, and allocation failures propagate.
// Success with available=true retains host views until deinitialization,
// without adding source ownership or public host-mapping permissions to the
// buffer.
iree_status_t iree_hal_amd_fabric_file_transfer_prepare_virtual(
    iree_hal_amd_fabric_file_transfer_t* transfer, const amdf_api_t* api,
    const iree_hal_queue_family_t* family, uint32_t native_family_ordinal,
    iree_hal_device_event_sink_t event_sink,
    iree_hal_memory_effects_t* out_before, iree_hal_memory_effects_t* out_after,
    bool* out_available);

// Publishes the complete direct-host read after every native file request has
// retired. Private physical views use their exact pair recipes; ordinary host
// bindings use the buffer's public host cache contract.
iree_status_t iree_hal_amd_fabric_file_transfer_flush(
    iree_hal_amd_fabric_file_transfer_t* transfer);

// Resolves committed buffer storage and acquires source bytes on the host
// service, after dependencies and the caller's before barrier. Memory files
// complete their copy and publication here. Platform files prepare one bounded
// chunk for submission. After a successful chunk, call finish and then prepare
// again while bytes_transferred is less than length. Staging is allocated once
// on first preparation and reused until the transfer completes.
iree_status_t iree_hal_amd_fabric_file_transfer_prepare(
    iree_hal_amd_fabric_file_transfer_t* transfer);

// Prepares the next platform-file chunk in an engine-owned ordinary host span.
// The file must have an async handle. Geometry can reduce the supplied extent;
// async.chunk records the exact admitted span and transfer-relative offset.
// No buffer mapping, host cache maintenance, allocation, or copy occurs here.
// GPU engines use this for staging, then order the native buffer copy
// themselves. Contents remain borrowed until the accepted I/O and engine copy
// both retire.
iree_status_t iree_hal_amd_fabric_file_transfer_prepare_chunk(
    iree_hal_amd_fabric_file_transfer_t* transfer, iree_byte_span_t contents);

// Submits the prepared chunk, continuing partial requests until the exact
// chunk completes or an error occurs. Failure retains caller ownership and does
// not invoke |callback|. Success transfers ownership through the final callback
// on the file's proactor, which may differ from the engine's progress owner.
iree_status_t iree_hal_amd_fabric_file_transfer_submit(
    iree_hal_amd_fabric_file_transfer_t* transfer,
    iree_hal_amd_fabric_file_transfer_callback_t callback);

// Copies staged read bytes and publishes the completed chunk on the host
// service before reuse. The final chunk releases staging; the engine applies
// its after barrier and signals only after every chunk has completed.
iree_status_t iree_hal_amd_fabric_file_transfer_finish(
    iree_hal_amd_fabric_file_transfer_t* transfer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_FABRIC_FILE_TRANSFER_H_
