// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cross-process file handle transfer for SHM carriers.
//
// SHM ring entries cannot carry platform descriptor rights. Cross-process SHM
// handshakes therefore keep their bootstrap channel open and use this component
// to transfer rights while the normal carrier/control payload carries an opaque
// transfer payload. POSIX uses SCM_RIGHTS over the retained channel. Windows
// duplicates HANDLEs into the peer process at export time and carries the peer
// HANDLE value in the transfer payload.

#ifndef IREE_NET_CARRIER_SHM_FILE_TRANSFER_H_
#define IREE_NET_CARRIER_SHM_FILE_TRANSFER_H_

#include "iree/async/primitive.h"
#include "iree/base/api.h"
#include "iree/io/file_handle.h"
#include "iree/net/carrier.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_shm_file_transfer_t iree_net_shm_file_transfer_t;

// Creates a file transfer sideband that owns |channel|. Platforms that require
// peer identity derive it from the connected channel.
iree_status_t iree_net_shm_file_transfer_create(
    iree_async_primitive_t channel, iree_allocator_t host_allocator,
    iree_net_shm_file_transfer_t** out_transfer);

// Releases the transfer sideband and closes any pending imported descriptors.
void iree_net_shm_file_transfer_release(iree_net_shm_file_transfer_t* transfer);

// Returns the platform transfer mechanism supported by |transfer|.
iree_net_file_handle_transfer_type_t iree_net_shm_file_transfer_type(
    const iree_net_shm_file_transfer_t* transfer);

// Carrier vtable implementation for query_file_handle_transfer.
iree_status_t iree_net_shm_file_transfer_query(
    iree_net_shm_file_transfer_t* transfer, iree_io_file_handle_t* file_handle,
    iree_net_file_handle_transfer_type_t* out_transfer_type,
    iree_host_size_t* out_payload_length);

// Carrier vtable implementation for export_file_handle.
iree_status_t iree_net_shm_file_transfer_export(
    iree_net_shm_file_transfer_t* transfer, iree_io_file_handle_t* file_handle,
    iree_net_file_handle_transfer_type_t transfer_type,
    iree_byte_span_t transfer_payload);

// Carrier vtable implementation for import_file_handle.
iree_status_t iree_net_shm_file_transfer_import(
    iree_net_shm_file_transfer_t* transfer,
    iree_net_file_handle_transfer_type_t transfer_type,
    iree_const_byte_span_t transfer_payload, iree_allocator_t host_allocator,
    iree_io_file_handle_t** out_file_handle);

// Carrier vtable implementation for release_file_handle_transfer.
iree_status_t iree_net_shm_file_transfer_release_export(
    iree_net_shm_file_transfer_t* transfer,
    iree_net_file_handle_transfer_type_t transfer_type,
    iree_const_byte_span_t transfer_payload);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_SHM_FILE_TRANSFER_H_
