// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/lifetime/destruction.h"

#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <limits>

#include "iree/base/api.h"

namespace loom::cxx_import {

std::optional<Destruction> destruction_for(cxx::TranslationUnit& unit,
                                           const cxx::Type* type) {
  auto traits = unit.typeTraits();
  if (traits.has_trivial_destructor(type)) {
    return std::nullopt;
  }

  size_t element_count = 1;
  type = traits.remove_cv(type);
  while (auto* array = cxx::type_cast<cxx::BoundedArrayType>(type)) {
    IREE_ASSERT_LE(array->size(),
                   std::numeric_limits<size_t>::max() / element_count);
    element_count *= array->size();
    type = traits.remove_cv(array->elementType());
  }

  auto* class_type = cxx::type_cast<cxx::ClassType>(type);
  IREE_ASSERT(class_type);
  auto* record = class_type->definition();
  traits.require_complete_class(record);
  record = class_type->definition();
  IREE_ASSERT(record && record->isComplete());
  auto* destructor = record->destructor();
  IREE_ASSERT(destructor && !destructor->isDeleted());
  return Destruction{type, destructor, element_count};
}

}  // namespace loom::cxx_import
