// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_CONTROL_ANALYSIS_H_
#define LOOM_IMPORT_CXX_CONTROL_ANALYSIS_H_

#include <cxx/ast_visitor.h>
#include <cxx/cxx_fwd.h>
#include <cxx/symbols_fwd.h>

#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "loom/import/cxx/value/types.h"

namespace loom::cxx_import {

// Retained ownership of a source assignment destination. A field or vector-lane
// write replaces a slice of its owning automatic binding; it is not a
// storage-backed store. Dynamic lane selection remains on the source expression
// and is evaluated once by translation.
struct Destination {
  // Owning local or parameter identity, borrowed from the source function.
  cxx::Symbol* binding;
  // First component of a nested member in the complete owning binding.
  size_t component_offset;
  // Member structure, or null when the destination is the entire binding.
  const Partition* member;
};

// One source lvalue's retained identity and write provenance at a call edge.
// The producer classifies the expression once; fixed-point propagation uses
// only these indexed facts.
struct CallArgument {
  // Source expression retained for diagnostics and parameter correspondence.
  cxx::ExpressionAST* source;
  // Automatic or reference bindings that become addressable when durable
  // identity is required. Conditional lvalues may contribute several.
  std::vector<cxx::Symbol*> identity_bindings;
  // Binding whose callable write effect reaches the enclosing function.
  cxx::Symbol* write_binding;
  // Whether the expression's identity is the current implicit object.
  bool current_receiver;
  // Whether the expression already denotes durable storage independently of
  // any caller fact, as with dereferences and reference-returning accessors.
  bool inherent_storage;
};

// One directly selected source call retained during the owning function walk.
// Callable propagation never traverses the frontend AST again.
struct DirectCall {
  // Concrete semantic callable selected by overload resolution.
  cxx::FunctionSymbol* function;
  // Call or overloaded operator expression owning diagnostics.
  cxx::AST* source;
  // Explicit argument facts in parameter order.
  std::vector<CallArgument> arguments;
  // Implicit object facts, absent for non-members and current-object calls.
  std::optional<CallArgument> receiver;
  // Whether the implicit object is the caller's own current object.
  bool current_receiver;
  // Whether source syntax already reaches the object through a pointer.
  bool storage_receiver;
};

// Monotone facts grown while propagating one retained call edge.
struct CallPropagation {
  bool caller_changed;
  bool callee_changed;
};

// Retained proof that a source loop has a stable, nonwrapping counted interval.
struct CountedLoop {
  struct Bound {
    // Resolved identifier supplying the loop-invariant bound.
    cxx::Symbol* binding;
    // Resolved source expression, including inserted value materializations.
    cxx::ExpressionAST* expression;
  };
  struct UnsignedRange {
    // Proven constant or stable runtime bound evaluated once by translation.
    std::variant<unsigned, Bound> upper;
    // Positive constant step in the source's unsigned-int width.
    unsigned step;
  };
  struct SignedRange {
    // Proven constant or stable runtime bound evaluated once by translation.
    std::variant<std::int32_t, Bound> upper;
    // Positive constant step in the source's signed-int width.
    unsigned step;
  };
  enum class AddressDomain {
    Index,
    Offset,
  };
  struct AddressRange {
    // Stable bound in the same native address domain as the induction value.
    Bound upper;
    // Signed logical index or unsigned physical offset comparison semantics.
    AddressDomain domain;
  };
  // Source binding replaced by the structured loop's induction argument.
  cxx::Symbol* induction;
  // Fixed-width C++ arithmetic or a target-selected native address interval.
  // Native address ranges have an implicit unit step: larger steps can wrap
  // after overshooting a bound whose target carrier width is not yet known.
  std::variant<UnsignedRange, SignedRange, AddressRange> range;
};

// Syntactic paths through a statement that take a particular exit. Other
// outcomes remain possible for None and Some; All takes that exit on every
// path.
enum class ExitFlow { None, Some, All };

// One resolved source label selecting a lexical switch clause. A missing key
// denotes the unique default label.
struct SwitchLabel {
  // Case or default statement retained for diagnostics and source locations.
  cxx::StatementAST* source;
  // Converted case value, or no value for the default label.
  std::optional<std::int64_t> key;
};

// Adjacent labels and the following lexical source segment. Statements appear
// exactly once even when several labels select the same clause.
struct SwitchClause {
  // Labels selecting this one source destination, in lexical order.
  std::vector<SwitchLabel> labels;
  // Statements through, but excluding, the next top-level switch label.
  std::vector<cxx::StatementAST*> statements;
  // Unique automatic bindings changed by this segment, in encounter order.
  std::vector<cxx::Symbol*> writes;
  // Whether some path reaches the next lexical clause or the switch tail.
  bool falls_through = false;
};

// Complete retained structure of one ordinary compound switch body.
struct SwitchLayout {
  // Lexical clauses, including adjacent labels grouped at one destination.
  std::vector<SwitchClause> clauses;
  // Whether one clause owns the source's unique default label.
  bool has_default = false;
  // Whether every selector destination admits one independent switch region.
  // Shared destinations and fallthrough require linear guarded translation.
  bool direct = true;
};

// One analysis owns control facts for an immutable source function body.
// Ordered writes include nested constructs and preserve source symbol identity.
// Constexpr branches visit their initializers, decision and source-selected
// arm. Exit outcomes aggregate each statement's already visited children, and
// counted-loop classification evaluates each header's constants once after its
// writes are complete. Queries only consume retained facts; they do not
// traverse source or output IR. The source unit and body outlive this object
// and every returned reference.
class ControlFlow final : private cxx::ASTVisitor {
 public:
  ControlFlow(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
              Types& types, cxx::StatementAST* body,
              cxx::FunctionSymbol* function = nullptr);
  ControlFlow(const ControlFlow&) = delete;
  ControlFlow& operator=(const ControlFlow&) = delete;

  // Unique bindings mutated under a structured statement, direct for-loop body,
  // or conditional value expression, in encounter order. Includes mutations in
  // conditions.
  std::span<cxx::Symbol* const> written(cxx::AST* owner) const;
  // Whether an evaluated address expression requires this binding's object
  // identity. The complete function is classified before translation begins.
  bool addressed(cxx::Symbol* binding) const;
  // Retained automatic-object destination, or no value for a memory access or
  // unsupported lvalue. A vector lane names its owning vector slice; whole
  // identifiers need no indexed projection record.
  std::optional<Destination> destination(cxx::ExpressionAST* expression) const;
  // Whether this member projects existing memory instead of an SSA record.
  // Object identity is propagated from visited children once during analysis.
  bool storage_backed(cxx::MemberExpressionAST* expression) const;
  // Null retains ordinary while semantics; a result permits scf.for lowering.
  const CountedLoop* counted(cxx::ForStatementAST* loop) const;
  // Retained lexical clauses for a supported runtime switch. Construction
  // rejects switch shapes that cannot provide this complete structure.
  const SwitchLayout& switch_layout(cxx::SwitchStatementAST* statement) const;
  // Retained return/fallthrough summary, including nested statements.
  ExitFlow returns(cxx::StatementAST* statement) const;
  // Continues escaping this statement to its enclosing loop. A nested loop
  // consumes its own continues rather than propagating them to its parent.
  ExitFlow continues(cxx::StatementAST* statement) const;
  // Breaks escaping this statement to its enclosing loop. A nested loop
  // consumes its own breaks rather than propagating them to its parent.
  ExitFlow breaks(cxx::StatementAST* statement) const;
  // Continue or break exits escaping this statement. This combined query
  // distinguishes a branch whose arms take different iteration exits from one
  // that can fall through.
  ExitFlow iteration_exits(cxx::StatementAST* statement) const;
  // Calls retained by the same source walk for sparse interprocedural effect
  // propagation before callable signatures are frozen.
  std::span<const DirectCall> calls() const { return calls_; }
  // Whether a reference parameter must carry durable storage identity.
  bool reference_requires_storage(cxx::Symbol* parameter) const;
  // Whether execution may modify the object named by a reference parameter.
  bool reference_written(cxx::Symbol* parameter) const;
  // Whether the implicit object itself must be addressable.
  bool receiver_requires_storage() const { return receiver_requires_storage_; }
  // Exact top-level current-object fields whose values may be replaced.
  std::span<cxx::FieldSymbol* const> receiver_writes() const {
    return receiver_writes_;
  }
  // Propagates reference and receiver facts in both directions across one
  // retained call edge.
  CallPropagation propagate_call(const DirectCall& call, ControlFlow& callee);
  // Normalizes exact receiver writes to declaration order after the monotone
  // propagation reaches a fixed point.
  void finalize_callable_effects();

 private:
  enum Path : unsigned {
    Fallthrough = 1,
    Return = 2,
    Continue = 4,
    Break = 8,
  };

  bool preVisit(cxx::AST* ast) override;
  void postVisit(cxx::AST* ast) override;
  void visit(cxx::CaseStatementAST* ast) override;
  void visit(cxx::DefaultStatementAST* ast) override;
  void visit(cxx::CompoundStatementAST* ast) override;
  void visit(cxx::IfStatementAST* ast) override;
  void visit(cxx::SwitchStatementAST* ast) override;
  void visit(cxx::AssignmentExpressionAST* ast) override;
  void visit(cxx::CompoundAssignmentExpressionAST* ast) override;
  void visit(cxx::PostIncrExpressionAST* ast) override;
  void visit(cxx::UnaryExpressionAST* ast) override;
  void visit(cxx::ReturnStatementAST* ast) override;
  void visit(cxx::CallExpressionAST* ast) override;
  void visit(cxx::SubscriptExpressionAST* ast) override;
  void visit(cxx::InitDeclaratorAST* ast) override;
  void visit(cxx::ConditionExpressionAST* ast) override;
  void visit(cxx::BracedInitListAST* ast) override;
  void visit(cxx::TypeConstructionAST* ast) override;
  void visit(cxx::BracedTypeConstructionAST* ast) override;
  static bool structured(cxx::AST* ast);
  void analyze_default_constructor(const cxx::Type* type,
                                   cxx::FunctionSymbol* constructor,
                                   cxx::AST* owner);
  std::optional<Destination> retain_destination(cxx::ExpressionAST* expression);
  void record(cxx::ExpressionAST* expression);
  void record(cxx::Symbol* binding);
  void record_implicit_object_call(cxx::CallExpressionAST* call,
                                   cxx::FunctionSymbol* callee);
  bool require_identity(cxx::ExpressionAST* expression);
  bool require_identity(const CallArgument& argument);
  bool argument_requires_storage(const CallArgument& argument) const;
  bool record_callable_write(cxx::ExpressionAST* expression);
  bool record_callable_write(cxx::Symbol* binding);
  bool require_reference_storage(cxx::Symbol* parameter);
  bool add_receiver_write(cxx::FieldSymbol* field);
  CallArgument retain_call_argument(cxx::ExpressionAST* expression);
  void collect_identity(cxx::ExpressionAST* expression, CallArgument& argument);
  void retain_call(cxx::CallExpressionAST* call, cxx::FunctionSymbol* function);
  void retain_call(cxx::SubscriptExpressionAST* call,
                   cxx::FunctionSymbol* function);
  void retain_reference_initializers(cxx::VariableSymbol* variable,
                                     cxx::ExpressionAST* initializer,
                                     cxx::AST* owner);
  std::optional<Destination> classify_destination(
      cxx::ExpressionAST* expression);
  bool classify_storage(cxx::ExpressionAST* expression) const;
  cxx::Symbol* object_owner(cxx::ExpressionAST* expression) const;
  std::optional<CountedLoop> classify(cxx::ForStatementAST* loop);
  std::optional<CountedLoop> classify_unsigned(
      cxx::ForStatementAST* loop, cxx::Symbol* induction,
      cxx::BinaryExpressionAST* condition);
  std::optional<CountedLoop> classify_signed(
      cxx::ForStatementAST* loop, cxx::Symbol* induction,
      cxx::BinaryExpressionAST* condition);
  std::optional<CountedLoop> classify_address(
      cxx::ForStatementAST* loop, cxx::Symbol* induction,
      cxx::BinaryExpressionAST* condition);
  unsigned paths(cxx::StatementAST* statement) const;
  unsigned sequence_paths(std::span<cxx::StatementAST* const> statements) const;
  unsigned classify_paths(cxx::StatementAST* statement) const;
  ExitFlow exits(cxx::StatementAST* statement, unsigned selected_paths) const;

  // Resolved source types and literal interpretation for loop admission.
  cxx::TranslationUnit& unit_;
  // Statement annotation admission during the existing source body walk.
  Diagnostics& diagnostics_;
  // Admitted source member partitions outlive all retained destination slices.
  Types& types_;
  // Source definition whose implicit object fields participate in SSA flow.
  cxx::FunctionSymbol* function_;
  // Active write-summary owners during construction only, including direct
  // for-loop bodies without a compound statement.
  std::vector<cxx::AST*> owners_;
  // Stable encounter order determines region argument/result order.
  std::unordered_map<cxx::AST*, std::vector<cxx::Symbol*>> writes_;
  // Automatic bindings whose addresses occur in evaluated source expressions.
  std::unordered_set<cxx::Symbol*> addressed_;
  // Reference locals retain their once-evaluated source destination shape for
  // effect and identity classification during this same walk.
  std::unordered_map<cxx::Symbol*, cxx::ExpressionAST*> reference_aliases_;
  // Reference parameters requiring buffer-plus-offset transport.
  std::unordered_set<cxx::Symbol*> storage_references_;
  // Reference parameters whose referents may be modified.
  std::unordered_set<cxx::Symbol*> written_references_;
  // Exact current-object field replacements in declaration encounter order.
  std::vector<cxx::FieldSymbol*> receiver_writes_;
  // Calls retained once for sparse fixed-point propagation.
  std::vector<DirectCall> calls_;
  // True when a direct or transitive use requires current-object identity.
  bool receiver_requires_storage_ = false;
  // Nested lvalue ownership and transitive component offsets computed once.
  std::unordered_map<cxx::ExpressionAST*, Destination> destinations_;
  // Memory record objects and member projections, including nested fields.
  std::unordered_set<cxx::ExpressionAST*> storage_expressions_;
  // Automatic roots retained through member projections and transparent
  // expressions. Complete address demand applies even to earlier field uses.
  std::unordered_map<cxx::ExpressionAST*, cxx::Symbol*> object_owners_;
  // Proven intervals retained after each source loop's children are visited.
  std::unordered_map<cxx::ForStatementAST*, CountedLoop> counted_;
  // Sparse path sets retain statements with function or iteration exits.
  std::unordered_map<cxx::StatementAST*, unsigned> paths_;
  struct SwitchFacts {
    // Public lexical structure consumed directly by translation.
    SwitchLayout layout;
    // Per-clause path sets retained for aggregate switch classification.
    std::vector<unsigned> clause_paths;
  };
  // Runtime switches retained during their one source-body traversal.
  std::unordered_map<cxx::SwitchStatementAST*, SwitchFacts> switches_;
  // Nested switches whose compound bodies are currently being traversed.
  std::vector<cxx::SwitchStatementAST*> switch_stack_;
  // Top-level case/default label currently admitted by the owning switch walk.
  cxx::StatementAST* active_switch_label_ = nullptr;
  // Canonical implicit constructor bodies visited once for this function.
  std::unordered_set<cxx::FunctionSymbol*> analyzed_constructors_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_CONTROL_ANALYSIS_H_
