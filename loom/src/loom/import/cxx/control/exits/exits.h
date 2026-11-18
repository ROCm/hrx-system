// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_CONTROL_EXITS_EXITS_H_
#define LOOM_IMPORT_CXX_CONTROL_EXITS_EXITS_H_

namespace loom::cxx_import {

// Fraction of the possible paths through a source statement taking an exit.
enum class ExitFlow { None, Some, All };

// Paths out of one statement and the callable exits needed to emit it. A
// direct return can yield an scf.if result. A return crossing a loop or
// switch and any path with no continuation require a function-level exit.
class StatementExits {
 public:
  enum Path : unsigned {
    Fallthrough = 1,
    Return = 2,
    Continue = 4,
    Break = 8,
    Diverge = 16,
  };

  constexpr StatementExits() = default;
  static constexpr StatementExits terminal(Path path) {
    return {path, path == Diverge};
  }
  static StatementExits branch(StatementExits then_exits,
                               StatementExits else_exits);
  // Appends one direct child in execution order. Even after a terminal
  // child, the lexical suffix count includes later callable-exit children.
  void append(StatementExits child);

  bool falls_through() const { return paths_ & Fallthrough; }
  bool reaches_latch() const { return paths_ & (Fallthrough | Continue); }
  bool escapes_region() const { return paths_ & (Return | Continue | Break); }
  ExitFlow returns() const { return flow(Return); }
  ExitFlow continues() const { return flow(Continue); }
  ExitFlow breaks() const { return flow(Break); }
  // Loops carrying a break flag track both ways of leaving the body. A
  // continue-only loop needs just the paths that transfer to its latch.
  ExitFlow iteration_exits(bool tracks_break = true) const {
    return flow(tracks_break ? Continue | Break : Continue);
  }
  bool requires_callable_exit() const { return callable_count_ != 0; }
  // On a compound, the number of direct children requiring a callable exit.
  unsigned callable_exit_count() const { return callable_count_; }

  enum class LoopTest { BeforeBody, AfterBody };
  // Consumes the body's breaks and continues. The condition has already been
  // checked in the source domain. An unknown test can fall through.
  StatementExits leave_loop(LoopTest test, bool always_true) const;

  bool operator==(const StatementExits&) const = default;

 private:
  friend class SwitchExits;
  constexpr StatementExits(unsigned paths, unsigned callable_count)
      : paths_(paths), callable_count_(callable_count) {}

  ExitFlow flow(unsigned selected) const {
    return !(paths_ & selected)   ? ExitFlow::None
           : (paths_ & ~selected) ? ExitFlow::Some
                                  : ExitFlow::All;
  }
  static StatementExits structured(unsigned paths) {
    return {paths, (paths & (Return | Diverge)) != 0};
  }

  // Set of possible source exits, including the lexical continuation.
  unsigned paths_ = Fallthrough;
  // Direct children are bounded by the translation unit's 32-bit token count.
  unsigned callable_count_ = 0;
};

// Each lexical switch clause is also a selector entry. Earlier clauses keep
// their escaping paths; only the final clause can fall through the switch.
class SwitchExits {
 public:
  void append_clause(StatementExits clause);
  StatementExits finish(bool has_default) const;

 private:
  // All escapes seen so far, plus fallthrough from the most recent clause.
  unsigned paths_ = StatementExits::Fallthrough;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_CONTROL_EXITS_EXITS_H_
