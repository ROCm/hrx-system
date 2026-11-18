// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SOURCE_LOCATIONS_H_
#define LOOM_IMPORT_CXX_SOURCE_LOCATIONS_H_

#include <cxx/ast_fwd.h>
#include <cxx/cxx_fwd.h>

#include <unordered_map>

#include "loom/import/cxx/source/options.h"
#include "loom/ir/module.h"

namespace loom::cxx_import {

class Diagnostics;

// Projects each reached source range once into module-owned provenance. The
// source and module outlive this index; output locations retain no AST storage.
// Synthesized AST nodes without source tokens map to LOOM_LOCATION_UNKNOWN.
class Locations {
 public:
  Locations(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
            loom_module_t* module,
            loom_cxx_source_observer_t source_observer = {})
      : unit_(unit),
        diagnostics_(diagnostics),
        module_(module),
        source_observer_(source_observer) {}

  // Returns the retained location, diagnosing unrepresentable source ranges.
  loom_location_id_t get(cxx::AST* ast);

 private:
  // Borrowed tokens and source positions for this translation unit.
  cxx::TranslationUnit& unit_;
  // Source rejection boundary for unrepresentable source coordinates.
  Diagnostics& diagnostics_;
  // Owns copied source identities and resulting location records.
  loom_module_t* module_;
  // Optional consumer of exact sources retained by translated locations.
  loom_cxx_source_observer_t source_observer_;
  // Retained results keyed by source identity, valid until AST destruction.
  std::unordered_map<cxx::AST*, loom_location_id_t> locations_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_SOURCE_LOCATIONS_H_
