// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/binding/assembly.h"

#include <cxx/ast.h>
#include <cxx/literals.h>
#include <cxx/names.h>
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
  std::vector<loom_type_t> transport_types;
  auto append_transport = [&](const cxx::Type* source_type) {
    types.append(source_type, owner, transport_types);
    for (loom_type_t type : transport_types) {
      if (loom_type_kind(type) != LOOM_TYPE_SCALAR &&
          loom_type_kind(type) != LOOM_TYPE_VECTOR &&
          loom_type_kind(type) != LOOM_TYPE_BUFFER &&
          !loom_type_is_dialect(type)) {
        diagnostics.reject(
            unit, owner,
            "assembly values require scalar, vector, buffer, or opaque "
            "dialect components");
      }
    }
  };
  size_t argument_count = 0;
  for (auto* parameter : std::span(parameters).subspan(1)) {
    append_transport(parameter);
    argument_count += transport_types.size();
    transport_types.clear();
  }
  auto* result = types.unqualified(signature->returnType());
  auto* selected_result = cxx::template_argument_type(arguments[1]);
  if (!selected_result || result != types.unqualified(selected_result)) {
    diagnostics.reject(
        unit, owner,
        "assembly return type must match its result type argument");
  }
  const Partition* result_partition = nullptr;
  if (result->kind() != cxx::TypeKind::kVoid) {
    result_partition = &types.partition(result, owner);
    append_transport(result);
  }
  return AssemblyIntrinsic(representation, argument_count, result_partition,
                           result_partition ? std::move(transport_types)
                                            : std::vector<loom_type_t>{});
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
  auto first = unit.tokenStartPosition(literal->firstSourceLocation());
  auto last = unit.tokenEndPosition(literal->firstSourceLocation());
  auto physical = diagnostics.contents().range(first, last);
  auto spelling =
      physical.source.substr(physical.start, physical.end - physical.start);
  auto opening = spelling.find('(');
  if (!spelling.starts_with("R\"") || opening == std::string_view::npos ||
      spelling != literal->literal->value() ||
      literal->lastSourceLocation() != literal->firstSourceLocation().next()) {
    diagnostics.reject(unit, source,
                       "assembly requires one narrow raw string literal");
  }
  // The frontend has validated the raw delimiter; both delimiters have the
  // same length. Parse the original source slice to preserve exact locations.
  if (opening >=
      (iree_host_size_t)LOOM_LOCATION_COORDINATE_MAX - first.column) {
    diagnostics.reject(unit, source,
                       "assembly range exceeds Loom's location representation");
  }
  const loom_source_range_t range = {
      .provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE,
      .filename = view(first.fileName),
      .source = view(physical.source),
      .start = physical.start + opening + 1,
      .end = physical.end - opening,
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
      function->result_count != result_types_.size()) {
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
    ValueArena& arena, loom_builder_t* builder,
    loom_location_id_t location) const {
  std::vector<loom_value_id_t> operands;
  operands.reserve(argument_count_);
  for (const auto& argument : arguments) {
    argument.append_to(operands);
  }
  loom_op_t* op;
  check(loom_low_invoke_build(
      builder, LOOM_LOW_INVOKE_BUILD_FLAG_HAS_INLINE_POLICY, 0,
      LOOM_INLINE_POLICY_INLINE, fragment, operands.data(), operands.size(),
      result_types_.data(), result_types_.size(), nullptr, 0, location, &op));
  if (!result_partition_) {
    return std::nullopt;
  }
  return arena.capture(
      *result_partition_,
      {loom_op_results(op), result_partition_->component_count});
}

bool AssemblyIntrinsic::equivalent(const AssemblyIntrinsic& other) const {
  if (contract_ != other.contract_ ||
      argument_count_ != other.argument_count_ ||
      result_partition_ != other.result_partition_ ||
      result_types_.size() != other.result_types_.size()) {
    return false;
  }
  for (size_t index = 0; index < result_types_.size(); ++index) {
    if (!loom_type_equal(result_types_[index], other.result_types_[index])) {
      return false;
    }
  }
  return true;
}

}  // namespace loom::cxx_import
