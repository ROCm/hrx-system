// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BUFFER_PROVIDER_H_
#define IREE_VM_BUFFER_PROVIDER_H_

#include "iree/vm/ref.h"

// Returns the process-static provider table for the core "vm" family.
const iree_vm_ref_type_table_t* iree_vm_buffer_provider_table(void);

#endif  // IREE_VM_BUFFER_PROVIDER_H_
