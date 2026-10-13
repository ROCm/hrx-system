// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_MODULE_HAL_MODULE_H_
#define IREE_MODULE_HAL_MODULE_H_

#include "iree/vm/environment.h"
#include "iree/vm/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates the optional native "hal" module using the HAL reference family
// already registered in |environment|. The module owns immutable type bindings
// and no per-process state. Calls translate borrowed VM references and verified
// static callsite records into native HAL descriptors. Queue work, pool
// backing, mapping, synchronization, and completion follow the HAL's caller
// contracts; the module neither tracks pending work nor waits during teardown.
//
// The module does not retain |environment|. The registered type provider's code
// and descriptors must outlive the module and all escaped reference owners.
// Success transfers one module owner to |out_module|; failure leaves it null.
IREE_API_EXPORT iree_status_t iree_hal_module_create(
    iree_vm_environment_t* environment, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_MODULE_HAL_MODULE_H_
