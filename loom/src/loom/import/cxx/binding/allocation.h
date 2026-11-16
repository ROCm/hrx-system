// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_ALLOCATION_H_
#define LOOM_IMPORT_CXX_BINDING_ALLOCATION_H_

#include <cxx/ast_fwd.h>
#include <cxx/const_value.h>
#include <cxx/cxx_fwd.h>

#include <cstdint>
#include <optional>
#include <string_view>

#include "loom/ir/types.h"

namespace loom::cxx_import {

class Diagnostics;

// Admits a constant, positive power-of-two byte alignment at the source
// boundary. Allocation bindings retain the result for every emitted call.
int64_t read_byte_alignment(cxx::TranslationUnit& unit,
                            Diagnostics& diagnostics,
                            const std::optional<cxx::ConstValue>& value,
                            std::string_view operation, cxx::AST* owner);

// Admits the allocatable memory-space family shared by buffers and views.
loom_value_fact_memory_space_t read_allocation_space(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    const std::optional<cxx::ConstValue>& value, std::string_view operation,
    cxx::AST* owner);

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_ALLOCATION_H_
