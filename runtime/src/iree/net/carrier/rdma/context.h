// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared RDMA device context.
//
// The RDMA context owns the dynamically loaded rdma-core libraries, the opened
// verbs device context, the selected active port/GID, and a protection domain.
// Carriers, transport factories, and GPU-direct setup code share this object so
// that expensive device discovery and PD creation are cold-path operations.

#ifndef IREE_NET_CARRIER_RDMA_CONTEXT_H_
#define IREE_NET_CARRIER_RDMA_CONTEXT_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/net/carrier/rdma/librdmacm.h"
#include "iree/net/carrier/rdma/libverbs.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_rdma_context_t iree_net_rdma_context_t;

// Options for selecting and opening an RDMA device context.
typedef struct iree_net_rdma_context_options_t {
  // Optional library search paths for libibverbs/librdmacm.
  iree_string_view_list_t library_search_paths;

  // Optional verbs device name, such as "mlx5_0" or "rxe0".
  // Empty selects the first device with a usable active port.
  iree_string_view_t device_name;

  // One-based port number. 0 selects the first active port on the device.
  uint8_t port_number;

  // GID table index used for RoCE addressing. 0 selects the default GID.
  uint8_t gid_index;
} iree_net_rdma_context_options_t;

// Returns default RDMA context options.
static inline iree_net_rdma_context_options_t
iree_net_rdma_context_options_default(void) {
  iree_net_rdma_context_options_t options;
  memset(&options, 0, sizeof(options));
  options.library_search_paths = iree_string_view_list_empty();
  return options;
}

// Creates an RDMA context over the selected device and active port.
//
// Returns UNAVAILABLE when rdma-core libraries are present but no usable RDMA
// device/port is available. Returns NOT_FOUND when a requested device name does
// not exist.
IREE_API_EXPORT iree_status_t iree_net_rdma_context_create(
    iree_net_rdma_context_options_t options, iree_allocator_t host_allocator,
    iree_net_rdma_context_t** out_context);

// Retains an RDMA context.
IREE_API_EXPORT void iree_net_rdma_context_retain(
    iree_net_rdma_context_t* context);

// Releases an RDMA context and destroys it when the last reference drops.
IREE_API_EXPORT void iree_net_rdma_context_release(
    iree_net_rdma_context_t* context);

// Returns the dynamically loaded libibverbs symbols owned by the context.
IREE_API_EXPORT const iree_net_libverbs_t* iree_net_rdma_context_libverbs(
    const iree_net_rdma_context_t* context);

// Returns the dynamically loaded librdmacm symbols owned by the context.
IREE_API_EXPORT const iree_net_librdmacm_t* iree_net_rdma_context_librdmacm(
    const iree_net_rdma_context_t* context);

// Returns the opened verbs device context.
IREE_API_EXPORT struct ibv_context* iree_net_rdma_context_device(
    const iree_net_rdma_context_t* context);

// Returns the protection domain used for queue pairs and memory regions.
IREE_API_EXPORT struct ibv_pd* iree_net_rdma_context_protection_domain(
    const iree_net_rdma_context_t* context);

// Returns the selected verbs device name.
IREE_API_EXPORT iree_string_view_t
iree_net_rdma_context_device_name(const iree_net_rdma_context_t* context);

// Returns the selected one-based port number.
IREE_API_EXPORT uint8_t
iree_net_rdma_context_port_number(const iree_net_rdma_context_t* context);

// Returns the selected GID table index.
IREE_API_EXPORT uint8_t
iree_net_rdma_context_gid_index(const iree_net_rdma_context_t* context);

// Returns the selected port attributes.
IREE_API_EXPORT const struct ibv_port_attr*
iree_net_rdma_context_port_attributes(const iree_net_rdma_context_t* context);

// Returns the selected GID value.
IREE_API_EXPORT const union ibv_gid* iree_net_rdma_context_gid(
    const iree_net_rdma_context_t* context);

// Registers ordinary host memory in the context protection domain.
//
// Registration is a cold-path operation. Callers should register slab/provider
// memory once and reuse the returned MR instead of registering per transfer.
IREE_API_EXPORT iree_status_t iree_net_rdma_context_register_host_memory(
    iree_net_rdma_context_t* context, void* base_ptr, iree_host_size_t length,
    int access_flags, struct ibv_mr** out_memory_region);

// Registers a dma-buf file descriptor in the context protection domain.
IREE_API_EXPORT iree_status_t iree_net_rdma_context_register_dmabuf(
    iree_net_rdma_context_t* context, uint64_t offset, iree_host_size_t length,
    uint64_t iova, int file_descriptor, int access_flags,
    struct ibv_mr** out_memory_region);

// Deregisters a memory region created by this context.
IREE_API_EXPORT iree_status_t iree_net_rdma_context_deregister_memory(
    iree_net_rdma_context_t* context, struct ibv_mr* memory_region);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_CONTEXT_H_
