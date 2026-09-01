// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_TOOLING_DUMP_H_
#define IREE_VM_BYTECODE_TOOLING_DUMP_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Streaming sink used by the bytecode module dumper.
//
// |text| is borrowed and valid only for the duration of the call. Returning a
// non-OK status terminates the dump and propagates that status to the caller.
typedef iree_status_t(IREE_API_PTR* iree_vm_bytecode_dump_write_fn_t)(
    void* user_data, iree_string_view_t text);

// Callback invoked for each text fragment emitted by the dumper.
typedef struct iree_vm_bytecode_dump_write_callback_t {
  // Function receiving the next text fragment.
  iree_vm_bytecode_dump_write_fn_t fn;
  // Opaque callback state passed to |fn|.
  void* user_data;
} iree_vm_bytecode_dump_write_callback_t;

// Verifies and dumps one immutable bytecode module image as deterministic text.
//
// The dump includes the container directory, architectural requirements,
// public declaration reflection, authored presentation and metadata when
// present, private function storage, and decoded instruction records. No
// runtime type provider or executable instruction implementation is required.
// |contents| must be nonempty with an eight-byte-aligned base address.
// |contents| and |module_name| need only remain live for this call. Failure may
// leave output already accepted by |write_callback| as a valid dump prefix.
IREE_API_EXPORT iree_status_t iree_vm_bytecode_module_dump(
    iree_string_view_t module_name, iree_const_byte_span_t contents,
    iree_vm_bytecode_dump_write_callback_t write_callback,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_VM_BYTECODE_TOOLING_DUMP_H_
