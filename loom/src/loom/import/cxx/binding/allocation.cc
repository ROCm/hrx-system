// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/allocation.h"

#include <string>
#include <variant>

#include "loom/import/cxx/source/source.h"

namespace loom::cxx_import {

int64_t read_byte_alignment(cxx::TranslationUnit& unit,
                            Diagnostics& diagnostics,
                            const std::optional<cxx::ConstValue>& value,
                            std::string_view operation, cxx::AST* owner) {
  auto* alignment = value ? std::get_if<cxx::ConstInt>(&*value) : nullptr;
  if (!alignment || alignment->isNegative() ||
      alignment->toUWide() > INT64_MAX || alignment->toUWide() == 0 ||
      (alignment->toUWide() & (alignment->toUWide() - 1)) != 0) {
    diagnostics.reject(
        unit, owner,
        std::string(operation) +
            " alignment must be a positive power-of-two byte count");
  }
  return static_cast<int64_t>(alignment->toUIntMax());
}

loom_value_fact_memory_space_t read_allocation_space(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    const std::optional<cxx::ConstValue>& value, std::string_view operation,
    cxx::AST* owner) {
  auto* space = value ? std::get_if<cxx::ConstInt>(&*value) : nullptr;
  if (!space || space->isNegative() || space->toUWide() > UINT8_MAX) {
    diagnostics.reject(
        unit, owner,
        std::string(operation) + " memory space must be a constant enum");
  }
  auto memory_space =
      static_cast<loom_value_fact_memory_space_t>(space->toUIntMax());
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
      return memory_space;
    default:
      diagnostics.reject(unit, owner,
                         std::string(operation) +
                             " requires an allocatable memory space: global, "
                             "workgroup, private, host, or generic");
  }
}

}  // namespace loom::cxx_import
