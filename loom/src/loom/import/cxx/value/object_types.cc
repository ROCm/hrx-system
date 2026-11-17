// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cxx/symbols.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/value/types.h"
#include "loom/ir/module.h"
#include "loom/ops/type/ops.h"

namespace loom::cxx_import {

void Types::declare_object_import(cxx::ClassSymbol* source,
                                  std::string_view namespace_name,
                                  std::string_view payload_name,
                                  std::string_view declaration_name) {
  loom_string_id_t namespace_id, payload_id, declaration_id;
  check(loom_module_intern_string(module_, cxx_import::view(namespace_name),
                                  &namespace_id));
  check(loom_module_intern_string(module_, cxx_import::view(payload_name),
                                  &payload_id));
  check(loom_module_intern_string(module_, cxx_import::view(declaration_name),
                                  &declaration_id));
  loom_symbol_ref_t symbol = {};
  check(loom_module_add_symbol(module_, declaration_id, &symbol.symbol_id));
  loom_builder_t builder;
  loom_builder_initialize(module_, &module_->arena, loom_module_block(module_),
                          &builder);
  loom_op_t* op;
  check(loom_type_decl_build(&builder, namespace_id, payload_id, symbol,
                             LOOM_LOCATION_UNKNOWN, &op));
  object_imports_.emplace(
      source, loom_type_reference(LOOM_REFERENCE_OWNERSHIP_BORROWED,
                                  namespace_id, payload_id, symbol.symbol_id));
}

loom_type_t Types::borrowed_object_type(cxx::ClassSymbol* payload,
                                        cxx::AST* owner) const {
  auto found = object_imports_.find(payload->canonical());
  if (found == object_imports_.end()) {
    diagnostics_.reject(
        unit_, owner,
        "object payload requires an explicit loom::import identity");
  }
  const auto identity = found->second;
  return loom_type_reference(LOOM_REFERENCE_OWNERSHIP_BORROWED,
                             loom_type_reference_namespace_id(identity),
                             loom_type_reference_name_id(identity),
                             loom_type_reference_declaration_id(identity));
}

const ObjectPartition* Types::object(const cxx::ClassType* input,
                                     loom_reference_ownership_t ownership,
                                     cxx::AST* owner) {
  auto* source = input->definition();
  if (auto found = objects_.find(source); found != objects_.end()) {
    return found->second.get();
  }
  auto traits = unit_.typeTraits();
  if (source) {
    traits.require_complete_class(source);
    source = input->definition();
  }
  if (!source || !source->isComplete() || source->isUnion() ||
      !source->baseClasses().empty() || !traits.is_trivially_copyable(input) ||
      !traits.has_trivial_destructor(input)) {
    diagnostics_.reject(unit_, owner,
                        "object handle primitives require a complete trivial "
                        "class without bases");
  }
  auto arguments = source->templateArguments();
  auto* payload = arguments.size() == 1
                      ? cxx::type_cast<cxx::ClassType>(
                            cxx::template_argument_type(arguments[0]))
                      : nullptr;
  if (!payload) {
    diagnostics_.reject(unit_, owner,
                        "object handles require one imported payload class");
  }
  auto result = std::make_unique<ObjectPartition>();
  result->kind = ValueKind::Object;
  result->component_count = 1;
  result->source = source;
  const auto identity = borrowed_object_type(payload->symbol(), owner);
  result->type =
      loom_type_reference(ownership, loom_type_reference_namespace_id(identity),
                          loom_type_reference_name_id(identity),
                          loom_type_reference_declaration_id(identity));
  auto* admitted = result.get();
  objects_.emplace(source, std::move(result));
  return admitted;
}

}  // namespace loom::cxx_import
