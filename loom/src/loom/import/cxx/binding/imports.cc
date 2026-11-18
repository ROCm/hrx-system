// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/imports.h"

#include <cxx/ast.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/symbols.h>
#include <cxx/translation_unit.h>
#include <cxx/type_traits.h>
#include <cxx/types.h>

#include <string_view>
#include <vector>

#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/diagnostics.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/source/expressions.h"

namespace loom::cxx_import {
namespace {

cxx::ClassSymbol* template_payload(cxx::ClassSymbol* source) {
  auto arguments = source ? source->templateArguments()
                          : std::span<const cxx::TemplateArgument>{};
  auto* type = arguments.size() == 1
                   ? cxx::template_argument_type(arguments.front())
                   : nullptr;
  auto* payload = cxx::type_cast<cxx::ClassType>(type);
  return payload ? payload->symbol()->canonical() : nullptr;
}

cxx::ClassSymbol* protocol_payload(cxx::ClassSymbol* source) {
  return template_annotation(source, "protocol") ? template_payload(source)
                                                 : nullptr;
}

cxx::ClassSymbol* protocol_payload(cxx::FunctionSymbol* function) {
  return protocol_payload(cxx::symbol_cast<cxx::ClassSymbol>(
      function ? function->parent() : nullptr));
}

cxx::ClassSymbol* borrowed_handle_payload(cxx::TranslationUnit& unit,
                                          const cxx::Type* input) {
  auto traits = unit.typeTraits();
  auto* type = cxx::type_cast<cxx::ClassType>(
      cxx::unattributed_type(traits.remove_cv(traits.remove_reference(input))));
  auto* source = type ? type->definition() : nullptr;
  const auto* binding = source ? template_annotation(source, "type") : nullptr;
  return binding && binding->arguments.size() == 1 &&
                 binding->arguments.front()->name() == "ptr"
             ? template_payload(source)
             : nullptr;
}

}  // namespace

void Imports::declaration(cxx::Symbol* symbol,
                          cxx::List<cxx::AttributeSpecifierAST*>* attributes,
                          cxx::AST* owner) {
  auto* function = cxx::symbol_cast<cxx::FunctionSymbol>(symbol);
  auto* object = cxx::symbol_cast<cxx::ClassSymbol>(symbol);
  cxx::AttributeAST* source = nullptr;
  visit_loom_attributes(
      unit_, attributes,
      [&](std::string_view name, cxx::AttributeAST* attribute) {
        if (name != "import") {
          return;
        }
        if (source) {
          diagnostics_.reject(unit_, attribute,
                              "one import binding is allowed per declaration");
        }
        source = attribute;
        if (object) {
          if (object->definition() || object->isUnion() ||
              object->isTemplatePattern() ||
              !cxx::symbol_cast<cxx::NamespaceSymbol>(object->parent())) {
            diagnostics_.reject(
                unit_, attribute,
                "object imports require incomplete namespace-scope classes");
          }
        } else if (!function || function->isTemplatePattern()) {
          diagnostics_.reject(
              unit_, attribute,
              "import bindings require concrete namespace-scope functions "
              "or incomplete object classes");
        }
        auto* receiver_payload = protocol_payload(function);
        if (function && !receiver_payload &&
            !cxx::symbol_cast<cxx::NamespaceSymbol>(function->parent())) {
          diagnostics_.reject(
              unit_, attribute,
              "import bindings require concrete namespace-scope functions "
              "or protocol methods");
        }
        if (receiver_payload) {
          auto* protocol_class =
              cxx::symbol_cast<cxx::ClassSymbol>(function->parent());
          if (!function->isImplicitObjectMemberFunction() ||
              function->isConstructor() || function->isDestructor() ||
              function->isVirtual() || !protocol_class ||
              !protocol_class->baseClasses().empty()) {
            diagnostics_.reject(
                unit_, attribute,
                "protocol method imports require ordinary nonvirtual "
                "instance methods on a class without bases");
          }
          for (auto* member : protocol_class->members()) {
            auto* field = cxx::symbol_cast<cxx::FieldSymbol>(member);
            if (field && !field->isStatic()) {
              diagnostics_.reject(unit_, attribute,
                                  "protocols cannot contain instance storage");
            }
          }
          if (!annotation(receiver_payload, "import")) {
            diagnostics_.reject(
                unit_, attribute,
                "protocol methods require imported payload types");
          }
        }
        if (cxx::has_internal_linkage(symbol)) {
          diagnostics_.reject(unit_, attribute,
                              "import bindings require external linkage");
        }
        std::vector<std::string> arguments;
        auto* clause = attribute->attributeArgumentClause;
        for (auto* argument :
             cxx::ListView{clause ? clause->expressionList : nullptr}) {
          auto* literal =
              cxx::ast_cast<cxx::StringLiteralExpressionAST>(argument);
          auto spelling =
              literal ? literal->literal->stringValue() : std::string_view{};
          if (!literal || spelling.empty() ||
              spelling.find('\0') != std::string_view::npos) {
            diagnostics_.reject(
                unit_, argument,
                "import names require nonempty NUL-free string literals");
          }
          arguments.emplace_back(spelling);
        }
        if (arguments.empty() || arguments.size() > 2) {
          diagnostics_.reject(
              unit_, attribute,
              "import requires a module string and optional symbol string");
        }
        const std::string module = arguments[0];
        const std::optional<std::string> external_name =
            arguments.size() == 2 ? std::optional<std::string>{arguments[1]}
                                  : std::nullopt;
        if (receiver_payload && !external_name) {
          diagnostics_.reject(
              unit_, attribute,
              "protocol method imports require an explicit runtime symbol");
        }
        if (object) {
          if (!external_name) {
            diagnostics_.reject(
                unit_, attribute,
                "object imports require a namespace and payload name");
          }
          auto [index, inserted] =
              object_indices_.try_emplace(object->canonical(), objects_.size());
          if (inserted) {
            objects_.push_back(
                {object->canonical(), {module, external_name, attribute}});
          } else {
            const auto& previous = objects_[index->second].binding;
            if (previous.module != module || previous.symbol != external_name) {
              diagnostics_.reject(
                  unit_, owner,
                  "conflicting import bindings across object redeclarations");
            }
          }
          return;
        }
        auto [previous, inserted] = imports_.try_emplace(
            function->canonical(),
            ImportBinding{module, external_name, attribute, receiver_payload});
        if (!inserted &&
            (previous->second.module != module ||
             previous->second.symbol != external_name ||
             previous->second.receiver_payload != receiver_payload)) {
          diagnostics_.reject(
              unit_, owner,
              "conflicting import bindings across function redeclarations");
        }
      });
}

const ImportBinding* Imports::lookup(cxx::FunctionSymbol* function) const {
  auto found = imports_.find(function->canonical());
  return found == imports_.end() ? nullptr : &found->second;
}

const ImportBinding* Imports::object_method(
    cxx::FunctionSymbol* function) const {
  const auto* binding = lookup(function);
  return binding && binding->receiver_payload ? binding : nullptr;
}

bool Imports::is_protocol_operator(cxx::FunctionSymbol* function) const {
  auto* name =
      function ? cxx::name_cast<cxx::OperatorId>(function->name()) : nullptr;
  if (!name || name->op() != cxx::TokenKind::T_MINUS_GREATER ||
      !function->isImplicitObjectMemberFunction() ||
      !function->parameters().empty()) {
    return false;
  }
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  auto traits = unit_.typeTraits();
  auto* return_type = signature && traits.is_pointer(signature->returnType())
                          ? cxx::type_cast<cxx::ClassType>(
                                traits.remove_pointer(signature->returnType()))
                          : nullptr;
  auto* protocol_class = return_type ? return_type->definition() : nullptr;
  auto* protocol_payload_type = protocol_payload(protocol_class);
  auto* handle_payload =
      borrowed_handle_payload(unit_, implicit_object_type(unit_, function));
  return protocol_payload_type && handle_payload &&
         protocol_payload_type == handle_payload;
}

std::optional<ObjectMethodReceiver> Imports::object_method_receiver(
    cxx::FunctionSymbol* function, cxx::CallExpressionAST* call) const {
  const auto* binding = object_method(function);
  if (!binding) {
    return std::nullopt;
  }
  auto* designator = unwrap_expression(call ? call->baseExpression : nullptr);
  auto* member = cxx::ast_cast<cxx::MemberExpressionAST>(designator);
  auto* arrow_call =
      member && member->accessOp == cxx::TokenKind::T_MINUS_GREATER
          ? cxx::ast_cast<cxx::CallExpressionAST>(
                unwrap_expression(member->baseExpression))
          : nullptr;
  auto* arrow = direct_callee(arrow_call);
  auto* arrow_designator =
      unwrap_expression(arrow_call ? arrow_call->baseExpression : nullptr);
  auto* arrow_member =
      cxx::ast_cast<cxx::MemberExpressionAST>(arrow_designator);
  if (!arrow_call || arrow_call->expressionList ||
      !is_protocol_operator(arrow) || !arrow_member ||
      arrow_member->accessOp != cxx::TokenKind::T_DOT) {
    diagnostics_.reject(
        unit_, call, "protocol methods require a matching loom::ptr receiver");
  }
  auto* handle = arrow_member->baseExpression;
  auto* payload = borrowed_handle_payload(unit_, handle->type);
  if (!payload || payload != binding->receiver_payload) {
    diagnostics_.reject(
        unit_, call, "protocol method receiver has a mismatched payload type");
  }
  return ObjectMethodReceiver{handle, payload};
}

}  // namespace loom::cxx_import
