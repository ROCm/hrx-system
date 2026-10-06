// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_INTRINSICS_H_
#define LOOM_IMPORT_CXX_BINDING_INTRINSICS_H_

#include <cxx/attributes.h>
#include <cxx/symbols_fwd.h>

#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>

#include "loom/import/cxx/binding/assembly.h"
#include "loom/import/cxx/binding/atomic.h"
#include "loom/import/cxx/binding/check.h"
#include "loom/import/cxx/binding/decode.h"
#include "loom/import/cxx/binding/encoding.h"
#include "loom/import/cxx/binding/kernel.h"
#include "loom/import/cxx/binding/scalar_bindings.h"
#include "loom/import/cxx/binding/shaped.h"
#include "loom/import/cxx/binding/template_apply.h"
#include "loom/import/cxx/binding/view.h"
#include "loom/import/cxx/source/source.h"
#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {

class SymbolNames;
class Locations;

// Result of one operation binding that has already claimed a source call.
// Void intrinsics have no value; absence never means that the call was missed.
struct IntrinsicCallResult {
  // Emitted source value, absent for a handled void operation.
  std::optional<Value> value;
};

// Resolves annotated source declarations against operation contracts at
// admission. Calls consume the retained binding directly; typed builders
// consume the resulting trusted signature.
class Intrinsics {
 public:
  struct ScalarOperation {
    // Immutable generated binding for this source declaration.
    const loom_cxx_scalar_binding_t* scalar;
    // Explicit source permissions in addition to invocation permissions.
    uint8_t flags = 0;

    bool operator==(const ScalarOperation& other) const = default;
  };
  struct ScalarBinding {
    // Operation and permissions admitted independently of the concrete type.
    ScalarOperation operation;
    // Source-selected scalar representation within the admitted category.
    loom_type_t type;

    bool equivalent(const ScalarBinding& other) const {
      return operation == other.operation && loom_type_equal(type, other.type);
    }
  };
  using Binding =
      std::variant<ScalarBinding, ShapedIntrinsic, EncodingIntrinsic,
                   DecodeIntrinsic, ViewIntrinsic, AtomicIntrinsic,
                   FenceIntrinsic, SubgroupIntrinsic, BarrierIntrinsic,
                   AssemblyIntrinsic, TemplateApplyIntrinsic, CheckIntrinsic>;

  Intrinsics(cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
             Locations& locations, SymbolNames& names, loom_module_t* module)
      : unit_(unit),
        diagnostics_(diagnostics),
        types_(types),
        locations_(locations),
        names_(names),
        module_(module) {}

  // Admits raw attribute arguments before the frontend's string-only semantic
  // attribute map can erase unsupported arguments or duplicate bindings.
  void declaration(cxx::FunctionSymbol* function,
                   cxx::List<cxx::AttributeSpecifierAST*>* attributes,
                   cxx::AST* owner);

  // Returns an admitted check operation, or null for another declaration.
  // Check-body translation owns its source constants and emission.
  const CheckIntrinsic* check_binding(cxx::FunctionSymbol* function,
                                      cxx::AST* owner);

  // Resolves a concrete operation once for call admission. The returned binding
  // remains stable until this invocation ends, including across nested calls.
  // NULL means an ordinary source function rather than an owned operation.
  Binding* lookup(cxx::FunctionSymbol* function, cxx::AST* owner);

  // Emits an ordinary concrete operation using source-preserving values.
  // Assembly literals and check expectations are handled by their source
  // owners.
  IntrinsicCallResult call(const Binding& binding,
                           std::span<const Value> arguments, ValueArena& arena,
                           Storage& storage, cxx::AST* owner,
                           uint8_t math_flags, loom_builder_t* builder,
                           loom_location_id_t location);

 private:
  struct TemplateBinding {
    // Immutable source binding interned by the owning translation unit.
    const cxx::Attribute* attribute;
    // Type-independent semantics established at primary admission. Other
    // bindings resolve their complete contract at concrete specialization.
    std::variant<std::monostate, ScalarOperation, ShapedIntrinsic::Operation,
                 EncodingIntrinsic::Family>
        operation;
  };

  Binding resolve(cxx::FunctionSymbol* function,
                  const cxx::Attribute& attribute, cxx::AST* owner);
  ScalarOperation resolve_scalar_operation(
      const loom_cxx_scalar_binding_t* scalar, const cxx::Attribute& attribute,
      cxx::AST* owner);
  ScalarBinding resolve_scalar(ScalarOperation operation,
                               const cxx::FunctionType* signature,
                               cxx::AST* owner);
  TemplateApplyIntrinsic::Family* template_family(std::string_view spelling,
                                                  cxx::AST* owner);
  Binding* concrete_binding(cxx::FunctionSymbol* function, cxx::AST* owner);

  // Invocation-owned frontend supplying canonical semantic types.
  cxx::TranslationUnit& unit_;
  // Invocation-owned diagnostic boundary for source rejection.
  Diagnostics& diagnostics_;
  // Invocation-owned source type projection shared with ordinary translation.
  Types& types_;
  // Invocation-owned source provenance retained by family declarations.
  Locations& locations_;
  // Shared output namespace preventing external-family/callable collisions.
  SymbolNames& names_;
  // Invocation-owned output module interning static source specifications.
  loom_module_t* module_;
  // Validated bindings indexed by canonical semantic function symbol.
  std::unordered_map<cxx::FunctionSymbol*, Binding> bindings_;
  // Template source contracts indexed by admitted primary declaration.
  std::unordered_map<cxx::FunctionSymbol*, TemplateBinding> template_bindings_;
  // Distinct linked family names interned once in the output symbol table.
  std::unordered_map<std::string, TemplateApplyIntrinsic::Family>
      template_families_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_INTRINSICS_H_
