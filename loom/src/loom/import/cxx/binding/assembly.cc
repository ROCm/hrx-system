// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/assembly.h"

#include <cxx/ast.h>
#include <cxx/literals.h>
#include <cxx/names.h>
#include <cxx/preprocessor.h>
#include <cxx/symbols.h>
#include <cxx/types.h>

#include <string>
#include <vector>

#include "loom/format/text/parser.h"
#include "loom/import/cxx/source/attributes.h"
#include "loom/import/cxx/source/error.h"
#include "loom/import/cxx/symbol/names.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/verify/verify.h"

namespace loom::cxx_import {

void AssemblyFragments::record(loom_func_like_t function,
                               loom_source_entry_t source) {
  functions_.push_back(function);
  if (source.source_id == LOOM_SOURCE_ID_INVALID) {
    return;
  }
  if (source.source_id >= sources_.size()) {
    sources_.resize(source.source_id + 1,
                    {.source_id = LOOM_SOURCE_ID_INVALID});
  }
  sources_[source.source_id] = source;
}

void AssemblyFragments::verify(loom_module_t* module,
                               loom_diagnostic_sink_t sink) const {
  if (functions_.empty()) {
    return;
  }
  loom_source_table_resolver_t sources = {module, sources_.data(),
                                          sources_.size()};
  const loom_verify_options_t options = {
      .sink = sink,
      .max_errors = 20,
      .source_resolver = {loom_source_table_resolve, &sources},
  };
  loom_verify_result_t result = {};
  check(loom_verify_functions(module, functions_.data(), functions_.size(),
                              &options, &result));
  if (result.error_count) {
    throw SourceRejected();
  }
}

std::optional<AssemblyIntrinsic> AssemblyIntrinsic::resolve(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
    cxx::AST* owner) {
  if (!supports(attribute.arguments[0]->name())) {
    return std::nullopt;
  }
  auto arguments = function->templateArguments();
  if (attribute.arguments.size() != 1 || arguments.size() < 2) {
    diagnostics.reject(
        unit, owner,
        "assembly requires leading contract and result type arguments");
  }
  auto* contract_type =
      cxx::type_cast<cxx::ClassType>(cxx::template_argument_type(arguments[0]));
  auto* contract = contract_type ? contract_type->definition() : nullptr;
  auto* declaration =
      contract ? cxx::ast_cast<cxx::ClassSpecifierAST>(contract->declaration())
               : nullptr;
  std::string_view representation;
  if (declaration) {
    visit_loom_attributes(
        unit, declaration->attributeList,
        [&](std::string_view name, cxx::AttributeAST* binding) {
          if (name != "representation") {
            return;
          }
          auto* clause = binding->attributeArgumentClause;
          auto* arguments = clause ? clause->expressionList : nullptr;
          auto* literal = arguments && !arguments->next
                              ? cxx::ast_cast<cxx::StringLiteralExpressionAST>(
                                    arguments->value)
                              : nullptr;
          if (!representation.empty() || !literal ||
              literal->literal->stringValue().empty()) {
            diagnostics.reject(unit, binding,
                               "assembly contract requires one nonempty string "
                               "representation binding");
          }
          representation = literal->literal->stringValue();
        });
  }
  if (representation.empty()) {
    diagnostics.reject(unit, owner,
                       "assembly contract type requires loom::representation");
  }
  auto* signature = cxx::type_cast<cxx::FunctionType>(function->type());
  const auto& parameters = signature->parameterTypes();
  auto* source = parameters.empty()
                     ? nullptr
                     : cxx::type_cast<cxx::PointerType>(
                           types.unqualified(parameters.front()));
  if (!source ||
      types.unqualified(source->elementType())->kind() !=
          cxx::TypeKind::kChar ||
      !unit.typeTraits().is_const(source->elementType()) ||
      signature->isVariadic()) {
    diagnostics.reject(unit, owner,
                       "assembly first parameter must be const char*");
  }
  for (auto* parameter : std::span(parameters).subspan(1)) {
    if (types.partition(parameter, owner).kind != ValueKind::SSA) {
      diagnostics.reject(unit, owner,
                         "assembly operands require scalar or vector values");
    }
  }
  auto* result = types.unqualified(signature->returnType());
  auto* selected_result = cxx::template_argument_type(arguments[1]);
  if (!selected_result || result != types.unqualified(selected_result)) {
    diagnostics.reject(
        unit, owner,
        "assembly return type must match its result type argument");
  }
  loom_type_t result_type = loom_type_none();
  if (result->kind() != cxx::TypeKind::kVoid) {
    if (types.partition(result, owner).kind != ValueKind::SSA) {
      diagnostics.reject(
          unit, owner,
          "assembly result requires a scalar, vector, or void type");
    }
    result_type = types.get(result, owner);
  }
  return AssemblyIntrinsic(representation, parameters.size() - 1, result_type);
}

loom_symbol_ref_t AssemblyIntrinsic::fragment(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics,
    cxx::ExpressionAST* source, SymbolNames& names,
    AssemblyFragments& fragments, loom_module_t* module,
    const loom_cxx_import_options_t& options) {
  while (auto* cast = cxx::ast_cast<cxx::ImplicitCastExpressionAST>(source)) {
    source = cast->expression;
  }
  auto* literal = cxx::ast_cast<cxx::StringLiteralExpressionAST>(source);
  if (!literal) {
    diagnostics.reject(unit, source, "assembly requires a raw string literal");
  }
  if (auto found = fragments_.find(literal); found != fragments_.end()) {
    return found->second;
  }
  auto token = unit.tokenAt(literal->firstSourceLocation());
  if (!token.fileId()) {
    diagnostics.reject(unit, source, "assembly literal requires source text");
  }
  auto spelling = unit.preprocessor()
                      ->source(token.fileId())
                      .substr(token.offset(), token.length());
  auto opening = spelling.find('(');
  if (!spelling.starts_with("R\"") || opening == std::string_view::npos ||
      spelling != literal->literal->value() ||
      literal->lastSourceLocation() != literal->firstSourceLocation().next()) {
    diagnostics.reject(unit, source,
                       "assembly requires one narrow raw string literal");
  }
  // The frontend has validated the raw delimiter; both delimiters have the
  // same length. Parse the original source slice to preserve exact locations.
  auto first = unit.tokenStartPosition(literal->firstSourceLocation());
  auto last = unit.tokenEndPosition(literal->firstSourceLocation());
  if (last.line > UINT16_MAX || last.column > UINT16_MAX ||
      first.column + opening + 1 > UINT16_MAX) {
    diagnostics.reject(unit, source,
                       "assembly range exceeds Loom's location representation");
  }
  const loom_source_range_t range = {
      .provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE,
      .filename = view(first.fileName),
      .source = view(unit.preprocessor()->source(token.fileId())),
      .start = token.offset() + opening + 1,
      .end = token.offset() + token.length() - opening,
      .start_line = first.line,
      .start_column = static_cast<uint32_t>(first.column + opening + 1),
      .end_line = last.line,
      .end_column = last.column,
  };
  std::string spelling_name = names.unique("__loom_assembly$");
  loom_string_id_t name;
  check(loom_module_intern_string(module, view(spelling_name), &name));
  loom_symbol_ref_t symbol = {};
  check(loom_module_add_symbol(module, name, &symbol.symbol_id));
  const loom_text_parse_options_t parse_options = {
      .diagnostic_sink = options.diagnostic_sink,
      .low_asm_environment = options.low_asm_environment,
  };
  loom_op_t* function = nullptr;
  check(loom_text_parse_low_assembly(range, view(contract_), symbol.symbol_id,
                                     module, &parse_options, &function));
  if (!function) {
    throw SourceRejected();
  }
  auto callable = loom_func_like_cast(module, function);
  auto* body = loom_func_like_body(callable);
  auto* entry = loom_region_entry_block(body);
  if (entry->arg_count != argument_count_ ||
      function->result_count !=
          ((loom_type_kind(result_type_) == LOOM_TYPE_NONE) ? 0 : 1)) {
    diagnostics.reject(unit, source,
                       "assembly register signature must match the C++ operand "
                       "and result counts");
  }
  const loom_source_id_t source_id =
      function->location == LOOM_LOCATION_UNKNOWN
          ? LOOM_SOURCE_ID_INVALID
          : loom_location_table_const_entry(&module->locations,
                                            function->location)
                ->file.source_id;
  fragments.record(callable, {
                                 .source_id = source_id,
                                 .source = range.source,
                                 .filename = range.filename,
                             });
  fragments_.emplace(literal, symbol);
  return symbol;
}

std::optional<Value> AssemblyIntrinsic::call(
    loom_symbol_ref_t fragment, std::span<const Value> arguments,
    loom_builder_t* builder, loom_location_id_t location) const {
  std::vector<loom_value_id_t> operands;
  operands.reserve(arguments.size());
  for (const auto& argument : arguments) {
    operands.push_back(argument.ssa());
  }
  loom_op_t* op;
  check(loom_low_invoke_build(
      builder, LOOM_LOW_INVOKE_BUILD_FLAG_HAS_INLINE_POLICY, 0,
      LOOM_INLINE_POLICY_INLINE, fragment, operands.data(), operands.size(),
      &result_type_, (loom_type_kind(result_type_) == LOOM_TYPE_NONE) ? 0 : 1,
      nullptr, 0, location, &op));
  if (loom_type_kind(result_type_) == LOOM_TYPE_NONE) {
    return std::nullopt;
  }
  return Value(loom_op_results(op)[0]);
}

bool AssemblyIntrinsic::equivalent(const AssemblyIntrinsic& other) const {
  return contract_ == other.contract_ &&
         argument_count_ == other.argument_count_ &&
         loom_type_equal(result_type_, other.result_type_);
}

}  // namespace loom::cxx_import
