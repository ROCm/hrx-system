// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SOURCE_ATTRIBUTES_H_
#define LOOM_IMPORT_CXX_SOURCE_ATTRIBUTES_H_

#include <cxx/ast_fwd.h>
#include <cxx/cxx_fwd.h>
#include <cxx/names_fwd.h>
#include <cxx/symbols_fwd.h>

#include <string_view>

namespace cxx {
struct Attribute;
}

namespace loom::cxx_import {

class Diagnostics;

// Returns a semantic attribute attached to a resolved source symbol, or null
// when the canonical declaration has no attribute with this spelling.
const cxx::Attribute* semantic_attribute(cxx::Symbol* symbol,
                                         std::string_view attribute_namespace,
                                         std::string_view spelling);

// Returns a semantic Loom annotation attached to a resolved source symbol, or
// null when the canonical declaration has no annotation with this spelling.
inline const cxx::Attribute* annotation(cxx::Symbol* symbol,
                                        std::string_view spelling) {
  return semantic_attribute(symbol, "loom", spelling);
}

// An implicit kernel specialization takes its role from all declarations of
// its primary template. An explicit specialization declares its own role.
bool is_kernel_function(cxx::FunctionSymbol* function);

// Whether the declaration binds exactly this operation and optional selector.
bool has_op_binding(cxx::Symbol* symbol, std::string_view operation,
                    std::string_view selector = {});

// Returns a Loom annotation attached to a class specialization or its primary
// template. Specialization annotations take precedence when both are present.
const cxx::Attribute* template_annotation(cxx::ClassSymbol* source,
                                          std::string_view spelling);

// Source positions with distinct binding owners. Namespace declarations,
// function prototypes, parameters and automatic declarations each admit their
// own leading attributes; all other source positions are local and unowned.
enum class BindingAttributeScope {
  Declaration,
  Function,
  Parameter,
  StagedParameter,
  LocalDeclaration,
  Statement,
  CheckStatement,
  Local,
};

// Returns the raw leading attribute list carried by one statement or automatic
// declaration. C++ represents a declaration statement's leading attributes on
// its nested simple declaration.
cxx::List<cxx::AttributeSpecifierAST*>* statement_attributes(
    cxx::StatementAST* statement);

// Rejects binding annotations outside their owning source positions.
void reject_misplaced_binding_attributes(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::List<cxx::AttributeSpecifierAST*>* attributes,
    BindingAttributeScope scope = BindingAttributeScope::Local);

// Checks declarator/type positions once. Only the selected function prototype
// admits parameter bindings; nested function types have no binding owner.
void reject_misplaced_binding_declarator(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::DeclaratorAST* declarator,
    cxx::FunctionDeclaratorChunkAST* parameter_owner = nullptr,
    BindingAttributeScope parameter_scope = BindingAttributeScope::Parameter);

// Checks one statement's leading attributes during its owner's body walk.
// Scope distinguishes ordinary translated statements from check records,
// whose bodies admit a different binding vocabulary.
void reject_misplaced_binding_statement(cxx::TranslationUnit& unit,
                                        Diagnostics& diagnostics,
                                        cxx::StatementAST* statement,
                                        BindingAttributeScope scope);

// Queries a semantic Loom annotation attached to a resolved source symbol.
inline bool annotated(cxx::Symbol* symbol, std::string_view spelling) {
  return annotation(symbol, spelling) != nullptr;
}

// Walks source attributes in spelling order, including repeated annotations
// and numeric arguments omitted from the frontend's semantic attribute map.
// Each returned name borrows the frontend's interned spelling.
class LoomAttributes {
 public:
  LoomAttributes(cxx::TranslationUnit& unit,
                 cxx::List<cxx::AttributeSpecifierAST*>* specifiers)
      : unit_(unit), specifiers_(specifiers) {}
  cxx::AttributeAST* next(std::string_view& name);

 private:
  // Source token owner for a C++ attribute-using-prefix.
  cxx::TranslationUnit& unit_;
  // Remaining attribute specifiers on the current source construct.
  cxx::List<cxx::AttributeSpecifierAST*>* specifiers_;
  // Remaining comma-separated attributes in the active specifier.
  cxx::List<cxx::AttributeAST*>* attributes_ = nullptr;
  // Shared namespace of that specifier, when written as [[using ns: ...]].
  const cxx::Identifier* prefix_ = nullptr;
};

template <typename Visitor>
void visit_loom_attributes(cxx::TranslationUnit& unit,
                           cxx::List<cxx::AttributeSpecifierAST*>* attributes,
                           Visitor visitor) {
  if (attributes) {
    LoomAttributes cursor(unit, attributes);
    std::string_view name;
    while (auto* attribute = cursor.next(name)) {
      visitor(name, attribute);
    }
  }
}

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_SOURCE_ATTRIBUTES_H_
