// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/symbol/immutable_objects.h"

#include <cxx/ast.h>
#include <cxx/binder.h>
#include <cxx/control.h>
#include <cxx/external_name_encoder.h>
#include <cxx/memory_layout.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <algorithm>
#include <cctype>
#include <string>

#include "loom/import/cxx/source/constants.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/expressions.h"
#include "loom/ops/global/ops.h"

namespace loom::cxx_import {
namespace {

bool contains_mutable_subobject(cxx::TranslationUnit& unit,
                                const cxx::Type* type) {
  auto traits = unit.typeTraits();
  auto* element = traits.remove_cv(traits.remove_all_extents(type));
  auto* class_type = cxx::type_cast<cxx::ClassType>(element);
  if (!class_type) {
    return false;
  }
  auto* definition = class_type->definition();
  traits.requireCompleteClass(definition);
  if (!definition || !definition->isComplete()) {
    return true;
  }
  for (auto* member : traits.aggregate_elements(definition)) {
    auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member);
    if (field && field->isMutable()) {
      return true;
    }
    if (contains_mutable_subobject(unit,
                                   traits.aggregate_element_type(member))) {
      return true;
    }
  }
  return false;
}

std::string qualified_object_name(cxx::VariableSymbol* variable,
                                  char separator) {
  std::string spelling = cxx::to_string(variable->name());
  for (auto* owner : variable->enclosingSymbols()) {
    auto* space = cxx::symbol_cast<cxx::NamespaceSymbol>(owner);
    if (space && space->name()) {
      spelling = cxx::to_string(space->name()) + separator + spelling;
    }
  }
  return spelling;
}

std::string private_object_name(cxx::VariableSymbol* variable) {
  auto spelling = qualified_object_name(variable, '_');
  for (char& character : spelling) {
    if (!std::isalnum(static_cast<unsigned char>(character)) &&
        character != '_') {
      character = '_';
    }
  }
  return spelling;
}

std::string value_name(std::string spelling) {
  for (char& character : spelling) {
    if (!std::isalnum(static_cast<unsigned char>(character)) &&
        character != '_') {
      character = '_';
    }
  }
  return spelling;
}

std::string external_object_name(cxx::TranslationUnit& unit,
                                 Diagnostics& diagnostics,
                                 cxx::VariableSymbol* variable,
                                 cxx::AST* owner) {
  if (variable->isSpecialization() || variable->isTemplatePattern()) {
    diagnostics.reject(
        unit, owner,
        "linkable immutable object template specializations are unsupported");
  }
  if (!cxx::name_cast<cxx::Identifier>(variable->name())) {
    diagnostics.reject(unit, owner,
                       "linkable immutable object name is unsupported");
  }
  auto spelling = qualified_object_name(variable, '.');
  if (!is_symbol_name(spelling)) {
    diagnostics.reject(unit, owner,
                       "linkable immutable object name is not a Loom symbol");
  }
  return spelling;
}

bool is_thread_local(cxx::VariableSymbol* variable) {
  return variable->isThreadLocal() ||
         std::ranges::any_of(variable->redeclarations(),
                             &cxx::VariableSymbol::isThreadLocal);
}

bool is_inline(cxx::VariableSymbol* variable) {
  return variable->isInline() ||
         std::ranges::any_of(variable->redeclarations(),
                             &cxx::VariableSymbol::isInline);
}

int64_t explicit_alignment(cxx::VariableSymbol* variable) {
  int64_t alignment = variable->explicitAlignment();
  for (auto* declaration : variable->redeclarations()) {
    alignment = std::max<int64_t>(alignment, declaration->explicitAlignment());
  }
  return alignment;
}

}  // namespace

void ImmutableObjects::retain_definition(cxx::VariableSymbol* variable,
                                         cxx::AST* owner) {
  if (unit_.typeTraits().is_reference(variable->type()) ||
      !cxx::Binder::declaresVariableDefinition(variable)) {
    return;
  }
  auto* canonical = variable->canonical();
  if (!cxx::symbol_cast<cxx::NamespaceSymbol>(canonical->parent()) ||
      cxx::has_internal_linkage(canonical) || is_inline(canonical)) {
    return;
  }
  reference(variable, owner);
}

ImmutableObject ImmutableObjects::reference(cxx::VariableSymbol* variable,
                                            cxx::AST* owner) {
  auto* canonical = variable->canonical();
  if (auto found = variables_.find(canonical); found != variables_.end()) {
    return emissions_[found->second].object;
  }
  if (auto found = locals_.find(canonical); found != locals_.end()) {
    return materialize(canonical, std::move(found->second));
  }
  if (!cxx::symbol_cast<cxx::NamespaceSymbol>(canonical->parent())) {
    diagnostics_.reject(unit_, owner,
                        "immutable storage requires static storage duration");
  }
  const bool has_internal_linkage = cxx::has_internal_linkage(canonical);
  auto* definition = canonical->definition();
  if (!definition && cxx::Binder::declaresVariableDefinition(canonical)) {
    definition = canonical;
  }
  const auto kind = !definition            ? EmissionKind::Declaration
                    : has_internal_linkage ? EmissionKind::PrivateDefinition
                    : is_inline(canonical) ? EmissionKind::LinkOnceDefinition
                                           : EmissionKind::PublicDefinition;
  auto spelling =
      has_internal_linkage
          ? private_object_name(canonical)
          : external_object_name(unit_, diagnostics_, canonical, owner);
  return materialize(canonical,
                     declare(canonical, owner, kind, std::move(spelling)));
}

void ImmutableObjects::retain_locals(
    cxx::FunctionSymbol* function,
    std::span<cxx::InitDeclaratorAST* const> declarations) {
  if (declarations.empty()) {
    return;
  }
  bool shared = true;
  for (auto* enclosing = function; enclosing;
       enclosing = enclosing->enclosingFunction()) {
    if (!cxx::is_inline_or_templated(enclosing) ||
        cxx::has_internal_linkage(enclosing) ||
        cxx::is_declared_with_internal_type(enclosing) ||
        cxx::has_internal_template_argument(enclosing)) {
      shared = false;
      break;
    }
  }
  std::string prefix;
  if (shared) {
    prefix = "__loom_cxx_static." +
             cxx::ExternalNameEncoder(&unit_).encode(function) + ".";
    diagnostics_.finish();
  }
  for (size_t ordinal = 0; ordinal < declarations.size(); ++ordinal) {
    auto* declaration = declarations[ordinal];
    auto* variable =
        cxx::symbol_cast<cxx::VariableSymbol>(declaration->symbol)->canonical();
    if (locals_.contains(variable)) {
      continue;
    }
    locals_.emplace(variable, declare(variable, declaration,
                                      shared ? EmissionKind::LinkOnceDefinition
                                             : EmissionKind::PrivateDefinition,
                                      shared ? prefix + std::to_string(ordinal)
                                             : private_object_name(variable)));
  }
}

ImmutableObjects::Declaration ImmutableObjects::declare(
    cxx::VariableSymbol* canonical, cxx::AST* owner, EmissionKind kind,
    std::string spelling) {
  auto* definition = canonical->definition();
  if (!definition && cxx::Binder::declaresVariableDefinition(canonical)) {
    definition = canonical;
  }
  auto traits = unit_.typeTraits();
  auto* source_type = definition ? definition->type() : canonical->type();
  if (is_thread_local(canonical)) {
    diagnostics_.reject(unit_, owner,
                        "immutable storage cannot have thread-local storage");
  }
  if (!definition && cxx::has_internal_linkage(canonical)) {
    diagnostics_.reject(
        unit_, owner,
        "immutable storage with internal linkage requires an object "
        "definition");
  }
  if (!traits.is_const(source_type) || traits.is_volatile(source_type) ||
      contains_mutable_subobject(unit_, source_type)) {
    diagnostics_.reject(
        unit_, owner,
        "immutable storage requires a nonvolatile deeply const object");
  }
  auto natural_alignment =
      unit_.control()->memoryLayout()->alignmentOf(source_type);
  if (!natural_alignment) {
    diagnostics_.reject(unit_, owner, "unknown immutable object alignment");
  }
  if (definition && !definition->constValue()) {
    diagnostics_.reject(unit_, owner,
                        "immutable storage requires constant initialization");
  }
  if (definition && !traits.is_trivially_copyable(source_type)) {
    diagnostics_.reject(
        unit_, owner,
        "immutable storage requires a fully initialized, trivially "
        "copyable, address-free object representation");
  }
  return {
      .definition = definition,
      .type = source_type,
      .alignment =
          std::max<int64_t>(*natural_alignment, explicit_alignment(canonical)),
      .kind = kind,
      .spelling = std::move(spelling),
      .source = owner,
  };
}

ImmutableObject ImmutableObjects::materialize(cxx::VariableSymbol* canonical,
                                              Declaration declaration) {
  bool is_utf8 = false;
  std::vector<uint8_t> contents;
  if (auto* definition = declaration.definition) {
    auto representation = object_representation(
        unit_, declaration.type, *definition->constValue(),
        string_initializer(definition->initializer()));
    if (!representation) {
      diagnostics_.reject(
          unit_, declaration.source,
          "immutable storage requires a fully initialized, trivially "
          "copyable, address-free object representation");
    }
    contents = std::move(representation->bytes);
    is_utf8 = representation->is_utf8;
  }
  auto spelling = std::move(declaration.spelling);
  if (declaration.kind == EmissionKind::PrivateDefinition) {
    spelling = names_.unique(std::move(spelling));
  } else {
    names_.reserve(spelling, declaration.source);
  }
  loom_string_id_t name;
  check(loom_module_intern_string(module_, view(spelling), &name));
  auto value_spelling = value_name(spelling);
  loom_string_id_t value_name_id;
  check(
      loom_module_intern_string(module_, view(value_spelling), &value_name_id));
  loom_symbol_id_t symbol;
  check(loom_module_add_symbol(module_, name, &symbol));
  ImmutableObject object = {
      .symbol = {0, symbol},
      .value_name = loom_string_table_get(&module_->strings, value_name_id),
      .type = declaration.type,
      .alignment = declaration.alignment,
  };
  variables_.emplace(canonical, emissions_.size());
  emissions_.push_back({
      .object = object,
      .kind = declaration.kind,
      .is_utf8 = is_utf8,
      .contents = std::move(contents),
      .source = declaration.source,
  });
  return object;
}

void ImmutableObjects::build(loom_builder_t* builder) {
  for (const auto& emission : emissions_) {
    loom_op_t* op;
    if (emission.kind == EmissionKind::Declaration) {
      check(loom_global_rodata_decl_build(builder, emission.object.symbol,
                                          locations_.get(emission.source),
                                          &op));
      continue;
    }
    loom_global_rodata_def_build_flags_t flags =
        emission.object.alignment > 1
            ? LOOM_GLOBAL_RODATA_DEF_BUILD_FLAG_HAS_ALIGNMENT
            : 0;
    uint8_t visibility = 0;
    uint8_t linkage = 0;
    if (emission.kind == EmissionKind::PublicDefinition ||
        emission.kind == EmissionKind::LinkOnceDefinition) {
      flags |= LOOM_GLOBAL_RODATA_DEF_BUILD_FLAG_HAS_VISIBILITY;
      visibility = LOOM_GLOBAL_RODATA_DEF_VISIBILITY_PUBLIC;
    }
    if (emission.kind == EmissionKind::LinkOnceDefinition) {
      flags |= LOOM_GLOBAL_RODATA_DEF_BUILD_FLAG_HAS_LINKAGE;
      linkage = LOOM_SYMBOL_LINKAGE_LINK_ONCE;
    }
    check(loom_global_rodata_def_build(
        builder, flags, visibility, linkage, emission.object.symbol,
        emission.object.alignment, loom_symbol_ref_array_empty(),
        iree_make_const_byte_span(emission.contents.data(),
                                  emission.contents.size()),
        locations_.get(emission.source), &op));
    if (emission.is_utf8) {
      // Reuse the builder-owned object bytes, including terminator and padding.
      auto contents = loom_global_rodata_def_contents(op);
      check(loom_global_rodata_def_set_contents(
          module_, op,
          loom_attr_utf8(iree_make_string_view(
              reinterpret_cast<const char*>(contents.data),
              contents.data_length))));
    }
  }
}

}  // namespace loom::cxx_import
