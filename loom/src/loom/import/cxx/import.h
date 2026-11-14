// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_IMPORT_H_
#define LOOM_IMPORT_CXX_IMPORT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/import/cxx/source/options.h"
#include "loom/import/cxx/source/preamble.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

// Imports one C/C++ translation unit into an ordinary verified Loom module.
// The finalized context must contain the core dialects. Context and block pool
// outlive the returned module; all source/AST storage can be released on
// return. No cleanup or optimization passes run here. On success the caller
// owns *out_module and frees it with loom_module_free(). Source rejection emits
// diagnostics and returns OK with *out_module=NULL. Infrastructure and option
// failures return a non-OK status with *out_module=NULL.
iree_status_t loom_cxx_import(iree_string_view_t source,
                              iree_string_view_t filename,
                              loom_context_t* context,
                              iree_arena_block_pool_t* block_pool,
                              const loom_cxx_import_options_t* options,
                              iree_allocator_t host_allocator,
                              loom_module_t** out_module);

// One preprocessed and type-checked C/C++ translation unit that several
// consumers share, such as a host code generator and the importer, so the
// source is parsed once. loom/import/cxx/unit.h exposes its frontend state to
// C++ consumers.
typedef struct loom_cxx_unit_t loom_cxx_unit_t;

// Preprocesses and checks one translation unit with the source configuration
// in |options|: standard, triple, data model, include paths, defines, source
// provider, preamble and the NO_BUILTIN_INCLUDES flag. Diagnostics
// go to options->diagnostic_sink. On success the caller owns *out_unit and
// frees it with loom_cxx_unit_free(). Source rejection emits diagnostics and
// returns OK with *out_unit=NULL. Borrowed option storage and the source
// provider may be released on return.
iree_status_t loom_cxx_unit_parse(iree_string_view_t source,
                                  iree_string_view_t filename,
                                  const loom_cxx_import_options_t* options,
                                  iree_allocator_t host_allocator,
                                  loom_cxx_unit_t** out_unit);

// Frees a unit returned by loom_cxx_unit_parse(). NULL is allowed.
void loom_cxx_unit_free(loom_cxx_unit_t* unit);

// Replaces the sink receiving diagnostics raised by later uses of |unit|,
// such as host code generation. A NULL callback silently counts errors.
void loom_cxx_unit_set_diagnostic_sink(loom_cxx_unit_t* unit,
                                       loom_diagnostic_sink_t sink);

// Imports an already parsed translation unit, as loom_cxx_import() does after
// parsing. Translation options apply: roots, floating-point model, Low
// assembly environment and source observer.
// The source configuration was fixed by loom_cxx_unit_parse(); options->
// diagnostic_sink becomes the unit's sink. The unit can be imported again;
// translation may instantiate further templates in it but does not reparse
// it.
iree_status_t loom_cxx_import_unit(loom_cxx_unit_t* unit,
                                   loom_context_t* context,
                                   iree_arena_block_pool_t* block_pool,
                                   const loom_cxx_import_options_t* options,
                                   iree_allocator_t host_allocator,
                                   loom_module_t** out_module);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_IMPORT_CXX_IMPORT_H_
