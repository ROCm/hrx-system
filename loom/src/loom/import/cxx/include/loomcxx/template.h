// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_TEMPLATE_H_
#define LOOMCXX_TEMPLATE_H_

// Declares an ordinary C++ function whose calls apply the named link-selected
// Loom template family. The declaration has no C++ definition; its parameters
// and result define the semantic family signature at each call site.
#define LOOM_TEMPLATE_DECL(FAMILY) [[loom::op("template.apply", FAMILY)]]
// Defines one bodyful implementation of a LOOM_TEMPLATE_DECL family. Linking
// selects among providers using optional loom::target(object) and
// loom::priority(constant) attributes. Providers inherit the family's calling
// convention and are retained without becoming ordinary callable C++ symbols.
#define LOOM_TEMPLATE_DEF(FAMILY) [[loom::template_def(FAMILY)]]

namespace loom {
namespace detail {

template <class Signature>
struct template_result;

template <class Result, class... Parameters>
struct template_result<Result (*)(Parameters...)> {
  using type = Result;
};

template <class Result, class... Parameters>
struct template_result<Result (*)(Parameters...) noexcept> {
  using type = Result;
};

}  // namespace detail

// Applies the statically named LOOM_TEMPLATE_DECL family. The family determines
// the result type; deduced argument types are admitted by the Loom importer.
// Passing a family through a C++ template parameter does not select a provider.
template <auto Family, class... Args>
[[loom::op("template.apply")]]
typename detail::template_result<decltype(Family)>::type template_apply(
    Args... args);

}  // namespace loom

#endif  // LOOMCXX_TEMPLATE_H_
