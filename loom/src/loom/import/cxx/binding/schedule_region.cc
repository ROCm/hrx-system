// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/schedule_region.h"

#include <cxx/ast.h>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/frontend.h"

namespace loom::cxx_import {

std::optional<ScheduleRegion> ScheduleRegion::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::StatementAST* statement) {
  std::optional<Kind> kind;
  cxx::AttributeAST* source = nullptr;
  visit_loom_attributes(
      unit, statement_attributes(statement),
      [&](std::string_view name, cxx::AttributeAST* attribute) {
        if (name != "serial" && name != "concurrent") {
          return;
        }
        if (source) {
          diagnostics.reject(
              unit, attribute,
              "one serial or concurrent schedule is allowed per statement");
        }
        if (attribute->attributeArgumentClause) {
          diagnostics.reject(unit, attribute,
                             "schedule region attributes take no arguments");
        }
        source = attribute;
        kind = name == "serial" ? Kind::Serial : Kind::Concurrent;
      });
  if (!kind) {
    return std::nullopt;
  }
  auto* body = cxx::ast_cast<cxx::CompoundStatementAST>(statement);
  if (!body) {
    diagnostics.reject(
        unit, source,
        "serial and concurrent require an attributed compound statement");
  }
  return ScheduleRegion{*kind, body};
}

}  // namespace loom::cxx_import
