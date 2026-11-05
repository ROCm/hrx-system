// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_PREDICATE_H_
#define LOOMCXX_PREDICATE_H_

// Predicate expressions describe compile-time contracts. They are admitted
// only inside loom::where attributes and loom::assume calls and have no runtime
// implementation. The helper names match Loom's textual predicate vocabulary.
namespace loom {

// Declares integer truth, comparison, and conjunction contracts. Conditions
// are retained as Loom facts without runtime evaluation. Calls, mutation,
// volatile reads, and expressions without a retained scalar identity diagnose
// at import.
[[loom::assume]] void assume(bool condition);

namespace predicate {

// The first operand selects the predicate carrier. Later retained values use
// that same type, while pure constants may keep their ordinary literal types.
template <class T, class U>
[[loom::predicate("eq")]] bool eq(const T& left, const U& right);
template <class T, class U>
[[loom::predicate("ne")]] bool ne(const T& left, const U& right);
template <class T, class U>
[[loom::predicate("lt")]] bool lt(const T& left, const U& right);
template <class T, class U>
[[loom::predicate("le")]] bool le(const T& left, const U& right);
template <class T, class U>
[[loom::predicate("gt")]] bool gt(const T& left, const U& right);
template <class T, class U>
[[loom::predicate("ge")]] bool ge(const T& left, const U& right);
template <class T, class U>
[[loom::predicate("ult")]] bool ult(const T& left, const U& right);
template <class T, class U>
[[loom::predicate("ule")]] bool ule(const T& left, const U& right);
template <class T, class U>
[[loom::predicate("ugt")]] bool ugt(const T& left, const U& right);
template <class T, class U>
[[loom::predicate("uge")]] bool uge(const T& left, const U& right);
template <class T, class U>
[[loom::predicate("multiple_of")]] bool multiple_of(const T& value,
                                                    const U& divisor);
template <class T>
[[loom::predicate("power_of_two")]] bool power_of_two(const T& value);
template <class T, class Lower, class Upper>
[[loom::predicate("range")]] bool range(const T& value, const Lower& lower,
                                        const Upper& upper);
template <class T>
[[loom::predicate("not_nan")]] bool not_nan(const T& value);
template <class T>
[[loom::predicate("not_inf")]] bool not_inf(const T& value);
template <class T>
[[loom::predicate("finite")]] bool finite(const T& value);

// Returns the typed result identity of the function carrying the enclosing
// trailing loom::where attribute. A member access selects one scalar component
// of a returned record.
template <class T>
[[loom::predicate_result]] T result();

// Subject-implicit overloads constrain a loom::config declaration. The
// omitted first operand is the declared config value. Numeric arguments use
// that value's source type and retain ordinary C++ suffix requirements. Index
// configs accept integral and enum constants because their first operand is
// supplied by the enclosing declaration rather than template deduction.
template <class T>
[[loom::predicate("eq")]] bool eq(const T& value);
template <class T>
[[loom::predicate("ne")]] bool ne(const T& value);
template <class T>
[[loom::predicate("lt")]] bool lt(const T& value);
template <class T>
[[loom::predicate("le")]] bool le(const T& value);
template <class T>
[[loom::predicate("gt")]] bool gt(const T& value);
template <class T>
[[loom::predicate("ge")]] bool ge(const T& value);
template <class T>
[[loom::predicate("ult")]] bool ult(const T& value);
template <class T>
[[loom::predicate("ule")]] bool ule(const T& value);
template <class T>
[[loom::predicate("ugt")]] bool ugt(const T& value);
template <class T>
[[loom::predicate("uge")]] bool uge(const T& value);
template <class T>
[[loom::predicate("multiple_of")]] bool multiple_of(const T& divisor);
[[loom::predicate("power_of_two")]] bool power_of_two();
template <class T>
[[loom::predicate("range")]] bool range(const T& lower, const T& upper);
[[loom::predicate("not_nan")]] bool not_nan();
[[loom::predicate("not_inf")]] bool not_inf();
[[loom::predicate("finite")]] bool finite();

}  // namespace predicate

}  // namespace loom

#endif  // LOOMCXX_PREDICATE_H_
