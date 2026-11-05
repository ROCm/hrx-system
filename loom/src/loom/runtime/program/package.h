// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_PROGRAM_PACKAGE_H_
#define LOOM_RUNTIME_PROGRAM_PACKAGE_H_

#include "iree/base/api.h"
#include "loomc/source.h"

#ifdef __cplusplus
extern "C" {
#endif

// One immutable input member. Names use UTF-8 and '/' separators, with no
// absolute paths, empty components, '.', '..', backslashes or drive prefixes.
typedef struct loom_runtime_package_file_t {
  // Package-relative portable filename, independent of the server filesystem.
  iree_string_view_t path;
  // Exact file bytes, including binary source and tokenizer/configuration data.
  iree_const_byte_span_t contents;
} loom_runtime_package_file_t;

// Immutable source environment. Retained references and opened sources may be
// shared across threads. No filesystem resources or transport claims are held.
typedef struct loom_runtime_package_t loom_runtime_package_t;

// Copies all names and payloads. document names the management JSON inside the
// package namespace and establishes the directory for relative references; it
// need not appear in files. Duplicate names reject. Failure publishes nothing.
iree_status_t loom_runtime_package_create(
    iree_string_view_t document, iree_host_size_t file_count,
    const loom_runtime_package_file_t* files, iree_allocator_t host_allocator,
    loom_runtime_package_t** out_package);

// NULL is accepted. Final release has no I/O or asynchronous cleanup.
void loom_runtime_package_retain(loom_runtime_package_t* package);
void loom_runtime_package_release(loom_runtime_package_t* package);

// Opens a relative locator against the package document directory. '.' and
// '..' are resolved lexically; escaping the package rejects. Missing members
// return NOT_FOUND, never a filesystem fallback. The returned compiler source
// retains its bytes independently of the package and autodetects text/bytecode
// when consumed by the compiler. Non-program assets use source_contents.
iree_status_t loom_runtime_package_open(loom_runtime_package_t* package,
                                        iree_string_view_t locator,
                                        loomc_source_t** out_source);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_PROGRAM_PACKAGE_H_
