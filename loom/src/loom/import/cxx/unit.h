// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_UNIT_H_
#define LOOM_IMPORT_CXX_UNIT_H_

#include <cxx/toolchain.h>
#include <cxx/translation_unit.h>

#include <cstdint>
#include <string>
#include <vector>

#include "loom/import/cxx/import.h"
#include "loom/import/cxx/source/frontend.h"
#include "loom/import/cxx/symbol/kernel_launch.h"

// C++ view of a parsed loom_cxx_unit_t for consumers that share the frontend
// translation unit with the importer, such as a host code generator in a
// single-source compiler driver. The unit keeps ownership of everything here.

namespace loom::cxx_import {

// The checked frontend translation unit. A consumer may read it and complete
// template instantiations in it; diagnostics it raises go to the unit's sink.
cxx::TranslationUnit& unit_translation_unit(loom_cxx_unit_t* unit);

// The source profile that configured the unit: triple, memory layout,
// language and exception support.
const cxx::Toolchain& unit_toolchain(const loom_cxx_unit_t* unit);

// Whether an error diagnostic was reported through the unit since it was
// parsed, for example by a host code generator.
bool unit_has_error(const loom_cxx_unit_t* unit);

// Usage counts of one unit.
struct UnitStatistics {
  // Completed loom_cxx_import_unit() calls on the unit.
  uint64_t import_count = 0;
};
UnitStatistics unit_statistics(const loom_cxx_unit_t* unit);

// Host launch facts of every kernel_ref root of the most recent successful
// import, in request order.
// A single-source driver combines these with the native argument layout of
// the compiled kernels to define the host's descriptors.
const std::vector<KernelLaunch>& unit_kernel_launches(
    const loom_cxx_unit_t* unit);

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_UNIT_H_
