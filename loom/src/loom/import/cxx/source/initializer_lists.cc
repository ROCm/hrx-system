// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/initializer_lists.h"

#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/initialization.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/types.h>

#include <array>

#include "loom/import/cxx/source/frontend.h"

namespace loom::cxx_import {

const InitializerList& InitializerLists::retain(
    cxx::BracedInitListAST* source) {
  auto traits = unit_.typeTraits();
  auto* element = traits.initializer_list_element_type(source->type);
  auto* record =
      cxx::unqualified_cast<cxx::ClassType>(source->type)->definition();
  std::array<cxx::FieldSymbol*, 2> fields = {};
  size_t count = 0;
  for (auto* member : record->members()) {
    if (auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member);
        field && !field->isStatic()) {
      if (count < fields.size()) {
        fields[count] = field;
      }
      ++count;
    }
  }
  auto* pointer = traits.add_pointer(traits.add_const(element));
  if (count != 2 || !record->baseClasses().empty() ||
      !traits.is_same(fields[0]->type(), pointer) ||
      (!traits.is_same(fields[1]->type(), pointer) &&
       !traits.is_integral(fields[1]->type()))) {
    diagnostics_.reject(unit_, source,
                        "initializer_list requires a const-element pointer "
                        "followed by an element count or end pointer");
  }
  size_t extent = 0;
  for (auto* clause = source->expressionList; clause; clause = clause->next) {
    ++extent;
  }
  auto* backing = cxx::BracedInitListAST::create(unit_.arena());
  backing->lbraceLoc = source->lbraceLoc;
  backing->rbraceLoc = source->rbraceLoc;
  backing->expressionList = *cxx::initializerListSlot(unit_.arena(), source);
  backing->type =
      unit_.control()->getBoundedArrayType(traits.add_const(element), extent);
  backing->valueCategory = cxx::ValueCategory::kPrValue;
  return lists_.emplace(source, InitializerList{backing, fields[0], fields[1]})
      .first->second;
}

const InitializerList* InitializerLists::get(
    cxx::BracedInitListAST* source) const {
  auto found = lists_.find(source);
  return found == lists_.end() ? nullptr : &found->second;
}

}  // namespace loom::cxx_import
