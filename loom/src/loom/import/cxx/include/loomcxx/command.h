// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_COMMAND_H_
#define LOOMCXX_COMMAND_H_

#include <loomcxx/buffer.h>
#include <loomcxx/view.h>

// Declares a reusable command program. Parameters before [[loom::launch]] are
// specialization values; the marked parameter and every following parameter
// are launch bindings. Omitting the marker declares a specialization-only
// program.
#define LOOM_COMMAND_PROGRAM [[loom::command_program]]

namespace loom::command {

// Invokes a statically named command program with its source signature.
// Dependent view dimensions and encodings are projected into specializations;
// the view handle remains a launch binding.
template <auto Program, class... Args>
[[loom::op("command.launch")]] void launch(Args... args);

// Resolves one immutable named parameter as a typed view. Dynamic extent and
// encoding values, when present in the result type, precede the key pattern;
// index substitutions follow it in placeholder order.
template <class Shape, class T, class Encoding = void, class SourceT,
          class... Args>
[[loom::op("command.parameter")]] type::view<Shape, T, Encoding> parameter(
    type::buffer<SourceT> source, Args... args);

// Recovers the parameter-relative suffix of the binding containing source.
// The typed buffer preserves the view's immutable element contract.
template <class Shape, class T, class Encoding>
[[loom::op("command.parameter.base")]] type::buffer<T> base(
    type::view<Shape, T, Encoding> source);

}  // namespace loom::command

#endif  // LOOMCXX_COMMAND_H_
