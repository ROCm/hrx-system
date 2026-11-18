// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_VALUE_COMPONENT_NAMES_H_
#define LOOM_IMPORT_CXX_VALUE_COMPONENT_NAMES_H_

#include <string>
#include <string_view>
#include <vector>

#include "loom/import/cxx/value/representation.h"

namespace loom::cxx_import {

// Appends source-level names in the exact order of the partition's SSA values.
// Source admission uses these names for record fields and columnar views.
void append_component_names(const Partition& partition, std::string_view prefix,
                            std::vector<std::string>& output);

// A source alias keeps an SSA value's first name. Composite values use the same
// field, extent, encoding, and payload names as their admitted types.
loom_value_id_t name_value(loom_module_t* module, loom_value_id_t value,
                           std::string_view hint);
Value name_value(loom_module_t* module, Value value, std::string_view hint);

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_VALUE_COMPONENT_NAMES_H_
