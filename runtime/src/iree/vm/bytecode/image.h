// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_IMAGE_H_
#define IREE_VM_BYTECODE_IMAGE_H_

#include "iree/vm/bytecode/module_storage.h"

// Retains one private immutable image owner. A null image is ignored.
void iree_vm_bytecode_image_retain(iree_vm_bytecode_image_t* image);

// Releases one private immutable image owner. Final release deallocates
// transferred image storage and the complete module slab.
void iree_vm_bytecode_image_release(iree_vm_bytecode_image_t* image);

#endif  // IREE_VM_BYTECODE_IMAGE_H_
