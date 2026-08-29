// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_PRODUCT_STORAGE_H_
#define LOOMC_PRODUCT_STORAGE_H_

#include "loomc/product.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocates an immutable request and transfers |*inout_source| on success.
//
// Roots must be unique and sorted by module then symbol ordinal. Bindings must
// be sorted by requirement ordinal and refer only to supplied roots. These are
// trusted producer invariants rather than public validation obligations.
LOOMC_API_PRIVATE loomc_status_t loomc_request_create_take_source(
    loomc_source_t** inout_source, const loomc_request_root_t* roots,
    loomc_host_size_t root_count, const loomc_request_binding_t* bindings,
    loomc_host_size_t binding_count, loomc_allocator_t allocator,
    loomc_request_t** out_request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_PRODUCT_STORAGE_H_
