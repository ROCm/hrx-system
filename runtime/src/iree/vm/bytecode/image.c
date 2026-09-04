// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/image.h"

void iree_vm_bytecode_image_retain(iree_vm_bytecode_image_t* image) {
  if (IREE_LIKELY(image)) {
    iree_atomic_ref_count_inc(&image->ref_count);
  }
}

void iree_vm_bytecode_image_release(iree_vm_bytecode_image_t* image) {
  if (IREE_LIKELY(image) && iree_atomic_ref_count_dec(&image->ref_count) == 1) {
    const iree_vm_bytecode_module_storage_t storage = image->storage;
    const iree_allocator_t host_allocator = image->host_allocator;
    iree_allocator_free(storage.deallocator, (void*)storage.contents.data);
    iree_allocator_free_aligned(host_allocator, image);
  }
}
