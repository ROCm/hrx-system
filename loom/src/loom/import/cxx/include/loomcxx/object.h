// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_OBJECT_H_
#define LOOMCXX_OBJECT_H_

namespace loom {

// Compile-time-only method contract for one imported payload. Specializations
// declare bodyless imported methods; no protocol value exists at runtime.
template <class T>
class [[loom::protocol]] protocol;

// A borrowed handle to an incomplete payload declared with loom::import.
// Copying the handle preserves the borrow; it cannot extend payload lifetime.
// The handle exposes neither the payload's layout nor a native address.
template <class T>
class [[loom::type("ptr")]] ptr {
 public:
  protocol<T>* operator->() const;

 private:
  // Source representation retaining ordinary trivial-copy semantics.
  T* data_;
};

namespace detail {

// Physical source carrier for one owning managed-object reference. Public
// loom::ref<T> supplies its ordinary C++ lifetime around this semantic value.
template <class T>
class [[loom::type("ref")]] object_owner {
  T* data_;
};

template <class T>
[[loom::op("ref.null")]] object_owner<T> object_null();

template <class T>
[[loom::op("ref.duplicate")]] object_owner<T> object_duplicate(
    object_owner<T> value);

template <class T>
[[loom::op("ref.borrow")]] ptr<T> object_borrow(object_owner<T> value);

template <class T>
[[loom::op("ref.release")]] void object_release(object_owner<T> value);

template <class T>
[[loom::op("ref.compare.null")]] bool object_is_null(object_owner<T> value);

}  // namespace detail

// An owning handle to an incomplete payload declared with loom::import.
// Copies retain independent ownership, moves transfer it, and destruction
// promptly releases it. get() borrows the payload without exposing an address.
template <class T>
class ref {
 public:
  [[loom::inline]]
  ref()
      : owner_(detail::object_null<T>()) {}

  [[loom::inline]]
  ref(const ref& other)
      : owner_(detail::object_duplicate(other.owner_)) {}

  [[loom::inline]]
  ref(ref&& other)
      : owner_(other.owner_) {
    other.owner_ = detail::object_null<T>();
  }

  [[loom::inline]]
  ~ref() {
    detail::object_release(owner_);
  }

  [[loom::inline]]
  ref& operator=(ref other) {
    auto next = other.owner_;
    other.owner_ = detail::object_null<T>();
    auto previous = owner_;
    owner_ = next;
    detail::object_release(previous);
    return *this;
  }

  [[loom::inline]]
  explicit operator bool() const {
    return !detail::object_is_null(owner_);
  }

  [[loom::inline]]
  void reset() {
    auto previous = owner_;
    owner_ = detail::object_null<T>();
    detail::object_release(previous);
  }

  [[loom::inline]]
  ptr<T> get() const {
    return detail::object_borrow(owner_);
  }

  [[loom::inline]]
  ptr<T> operator->() const {
    return get();
  }

 private:
  detail::object_owner<T> owner_;
};

}  // namespace loom

#endif  // LOOMCXX_OBJECT_H_
