// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/objects.h"

#include <cxx/ast.h>
#include <cxx/symbols.h>

namespace loom::cxx_import {

const cxx::Type* source_object_type(cxx::Symbol* symbol) {
  auto* base = cxx::symbol_cast<cxx::BaseClassSymbol>(symbol);
  return base ? base->symbol()->type() : symbol->type();
}

uint64_t source_subobject_offset(cxx::Symbol* symbol) {
  if (auto* field = cxx::symbol_cast<cxx::FieldSymbol>(symbol)) {
    return *field->offsetInClass();
  }
  auto* base = static_cast<cxx::BaseClassSymbol*>(symbol);
  auto* enclosing = static_cast<cxx::ClassSymbol*>(base->parent());
  auto* definition = static_cast<cxx::ClassSymbol*>(base->symbol());
  return enclosing->layout()->getBaseInfo(definition, false)->offset;
}

bool is_class_adjustment(const cxx::ImplicitCastExpressionAST* expression) {
  return expression->castKind ==
             cxx::ImplicitCastKind::kDerivedToBaseConversion ||
         expression->castKind ==
             cxx::ImplicitCastKind::kBaseToDerivedConversion;
}

}  // namespace loom::cxx_import
