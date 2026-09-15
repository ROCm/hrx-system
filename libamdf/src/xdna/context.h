// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_CONTEXT_H_
#define AMDF_SRC_XDNA_CONTEXT_H_

#include "amdf/xdna.h"
#include "libamdf/src/xdna/umd/context.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Admits one program-independent schedulable XDNA context.
amdf_status_t AMDF_CALL amdf_xdna_context_create(
    amdf_device_t* device, const amdf_xdna_context_create_info_t* create_info,
    amdf_xdna_context_t** out_context);

// Copies immutable context identity and logical admission information.
amdf_status_t AMDF_CALL amdf_xdna_context_query_info(
    amdf_xdna_context_t* context, amdf_xdna_context_info_t* out_info);

// Copies complete fixed placement, or returns UNSUPPORTED without output.
amdf_status_t AMDF_CALL amdf_xdna_context_query_placement_info(
    amdf_xdna_context_t* context, amdf_xdna_context_placement_info_t* out_info);

// Returns the ordinary-address-domain device borrowed by one context.
amdf_device_t* amdf_xdna_context_get_device(amdf_xdna_context_t* context);

// Returns the exact borrowed private descriptor of a live context.
amdf_memory_scope_t* amdf_xdna_context_get_memory_scope(
    amdf_xdna_context_t* context);

// Enumerates complete private storage descriptors without native operations.
amdf_status_t AMDF_CALL amdf_xdna_context_enumerate_memory_scopes(
    amdf_xdna_context_t* context, uint32_t capacity,
    amdf_memory_scope_t** scopes, uint32_t* out_count);

// Returns immutable information borrowed from one XDNA context.
const amdf_xdna_context_info_t* amdf_xdna_context_get_info(
    const amdf_xdna_context_t* context);

// Returns the UMD context borrowed from one public XDNA context.
amdf_xdna_umd_context_t* amdf_xdna_context_get_umd(
    amdf_xdna_context_t* context);

// Registers one child borrowing a context.
amdf_status_t amdf_xdna_context_register_child(amdf_xdna_context_t* context);

// Releases one context child borrow.
void amdf_xdna_context_unregister_child(amdf_xdna_context_t* context);

// Destroys one context with no remaining children.
amdf_status_t AMDF_CALL amdf_xdna_context_destroy(amdf_xdna_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_XDNA_CONTEXT_H_
