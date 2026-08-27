// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_CONTEXT_CACHE_H_
#define IREE_HAL_DRIVERS_AMDXDNA_CONTEXT_CACHE_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdxdna/native.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdxdna_device iree_hal_amdxdna_device;
typedef struct iree_hal_amdxdna_device_context_cache_t
    iree_hal_amdxdna_device_context_cache_t;

// Injectable native-context operations used by hermetic cache-policy tests and
// optional device integration hooks. NULL callbacks use the native DDI
// directly. The retain/release callbacks retain/release opaque native context
// refs with the same ownership contract as
// iree_hal_amdxdna_native_context_ref_{retain,release}.
typedef struct iree_hal_amdxdna_context_cache_ops_t {
  iree_status_t (*create_context)(
      void* user_data,
      const iree_hal_amdxdna_native_c_context_image_t* context_image,
      bool* out_context_pool_exhausted,
      iree_hal_amdxdna_native_context_ref_t** out_context_ref);
  iree_hal_amdxdna_native_context_ref_t* (*retain_context)(
      void* user_data, iree_hal_amdxdna_native_context_ref_t* context_ref);
  void (*release_context)(void* user_data,
                          iree_hal_amdxdna_native_context_ref_t* context_ref);
  // Optional hook invoked before the cache releases an entry-owned context ref.
  // Device-level users can use this to invalidate resources keyed by the
  // context/queue lifetime. The callback must not retain |context_ref|.
  void (*before_release_context)(
      void* user_data, iree_hal_amdxdna_native_context_ref_t* context_ref);
} iree_hal_amdxdna_context_cache_ops_t;

// Full identity of a cached hardware context. Two contexts are interchangeable
// only when they use the same native context image inputs. For Linux KMQ PDI
// contexts this is PDI + CU name; for native-xclbin backends this is the xclbin
// image. Dispatch control code is command identity, not context identity: the
// native context constructor does not consume it, and splitting one logical PDI
// context by control-code content creates redundant hwctx objects.
typedef struct iree_hal_amdxdna_context_cache_key_t {
  iree_const_byte_span_t pdi;
  iree_const_byte_span_t xclbin;
  iree_string_view_t kernel_name;
} iree_hal_amdxdna_context_cache_key_t;

// Returns true when a context cached under `lhs` may be safely reused to
// satisfy a request bearing `rhs`, i.e. all native context-image key fields are
// byte-for-byte equal. The cache uses this for lookup; it is also exported so
// the reuse-safety contract can be unit tested without a device.
bool iree_hal_amdxdna_context_cache_key_equal(
    const iree_hal_amdxdna_context_cache_key_t* lhs,
    const iree_hal_amdxdna_context_cache_key_t* rhs);

// Creates a bounded device context cache sized by `hardware_context_budget`
// (from iree_hal_amdxdna_native_c_device_caps_t::max_hardware_contexts). See
// iree_hal_amdxdna_context_cache_resolve_capacity for how the capacity is
// chosen.
iree_hal_amdxdna_device_context_cache_t*
iree_hal_amdxdna_device_context_cache_create(
    iree_allocator_t host_allocator, iree_host_size_t hardware_context_budget);

// Creates a cache using injected context operations. Intended for hermetic
// policy/lifetime tests; production code should use
// iree_hal_amdxdna_device_context_cache_create.
iree_hal_amdxdna_device_context_cache_t*
iree_hal_amdxdna_device_context_cache_create_with_ops(
    iree_allocator_t host_allocator, iree_host_size_t hardware_context_budget,
    const iree_hal_amdxdna_context_cache_ops_t* ops, void* user_data);

// Resolves the cache capacity: IREE_HAL_AMDXDNA_CONTEXT_CACHE_CAPACITY if set
// (0 disables the bound), else the device budget if nonzero, else a built-in
// default. Exported so the policy can be unit tested without a device.
iree_host_size_t iree_hal_amdxdna_context_cache_resolve_capacity(
    iree_host_size_t hardware_context_budget);

void iree_hal_amdxdna_device_context_cache_destroy(
    iree_hal_amdxdna_device_context_cache_t* context_cache);

void iree_hal_amdxdna_device_context_cache_clear(
    iree_hal_amdxdna_device_context_cache_t* context_cache);

// Implements lookup/create independently of the HAL device wrapper. Exported
// for hermetic cache-policy tests; production callers use
// iree_hal_amdxdna_device_get_or_create_context below.
iree_status_t iree_hal_amdxdna_context_cache_get_or_create(
    iree_hal_amdxdna_device_context_cache_t* context_cache,
    iree_hal_amdxdna_native_device_t* native_device,
    uint32_t context_image_models, iree_const_byte_span_t pdi,
    iree_const_byte_span_t xclbin, iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_context_ref_t** out_context_ref);

// Returns a native context for the (non-empty) control-packet bootstrap
// `pdi`/`xclbin` and CU/export name, creating and caching it on first use. A
// cached PDI context is reused when PDI + kernel name match; an xclbin-native
// context is keyed by xclbin content because that backend repatches control
// streams per dispatch.
iree_status_t iree_hal_amdxdna_device_get_or_create_context(
    iree_hal_amdxdna_device* device, iree_const_byte_span_t pdi,
    iree_const_byte_span_t xclbin, iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_context_ref_t** out_context_ref);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_CONTEXT_CACHE_H_
