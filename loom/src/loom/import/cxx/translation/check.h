// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_TRANSLATION_CHECK_H_
#define LOOM_IMPORT_CXX_TRANSLATION_CHECK_H_

struct loom_builder_t;

namespace cxx {
class TranslationUnit;
}  // namespace cxx

namespace loom::cxx_import {

class Diagnostics;
class Functions;
struct FunctionBody;
class Intrinsics;
class Locations;
class Scalars;
class Types;

// Projects an admitted check case into existing check sources, direct function
// invocations, and terminal expectations. The caller has entered the case body.
// All collaborators and source AST remain live until translation returns.
void translate_check_case_body(cxx::TranslationUnit& unit,
                               Diagnostics& diagnostics, Functions& functions,
                               Intrinsics& intrinsics, Types& types,
                               Scalars& scalars, Locations& locations,
                               loom_builder_t& builder,
                               const FunctionBody& body);

// Projects an admitted scenario into finite trial recipes and their terminal
// target-only or differential actions. The caller has entered the scenario
// body, whose optional configuration arguments are already reserved.
void translate_check_scenario_body(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Functions& functions,
    Intrinsics& intrinsics, Types& types, Scalars& scalars,
    Locations& locations, loom_builder_t& builder, const FunctionBody& body);

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_TRANSLATION_CHECK_H_
