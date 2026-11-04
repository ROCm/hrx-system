// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_WDDM_PROVIDER_FAKE_API_H_
#define AMDF_SRC_GPU_UMD_WDDM_PROVIDER_FAKE_API_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Restores the fake provider's default state and observations.
void amdf_test_wddm_provider_reset(void);

// Selects the number of rejected adapter-close attempts.
void amdf_test_wddm_provider_set_adapter_close_failures(uint32_t failure_count);

// Returns the number of successfully opened adapters.
uint32_t amdf_test_wddm_provider_query_adapter_open_success_count(void);

// Returns the number of attempted adapter closes.
uint32_t amdf_test_wddm_provider_query_adapter_close_attempt_count(void);

// Returns the number of successfully closed adapters.
uint32_t amdf_test_wddm_provider_query_adapter_close_success_count(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_GPU_UMD_WDDM_PROVIDER_FAKE_API_H_
