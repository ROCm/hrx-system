// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/view.h>

using Payload = loom::type::buffer<const unsigned char>;

static Payload make(unsigned value) {
  if (value == 11u) {
    return loom::buffer::root(loom::buffer::constant("\x0b"));
  }
  if (value == 12u) {
    return loom::buffer::root(loom::buffer::constant("\x0c"));
  }
  if (value == 22u) {
    return loom::buffer::root(loom::buffer::constant("\x16"));
  }
  return loom::buffer::root(loom::buffer::constant("\x21"));
}

static unsigned read(Payload value) {
  return loom::view::load(loom::buffer::view<1>(value, {}), 0u);
}

struct Cell {
  // Semantic payload remains a value; it cannot be materialized as C++ bytes.
  Payload payload;

  [[loom::noinline]] Cell& self() { return *this; }
  [[loom::noinline]] unsigned replace(Payload next) {
    const auto previous = read(payload);
    payload = next;
    return previous;
  }
};

[[loom::noinline]] static Cell& touch(Cell& value, unsigned& selections) {
  ++selections;
  return value;
}
[[loom::noinline]] static Cell& identity(Cell& value) { return value; }
[[loom::noinline]] static void replace(Cell& value, Payload next) {
  value.payload = next;
}

// A saved reference observes the selected binding's current contents after
// both replacements; a copied payload still observes its original contents.
unsigned selected_object(bool choose, unsigned observation) {
  Cell first{make(11u)}, second{make(12u)};
  unsigned selections = 0u;
  Cell& selected =
      choose ? touch(first, selections) : touch(second, selections);
  const Payload& watched = selected.payload;
  const auto snapshot = selected.payload;
  replace(selected, make(22u));
  choose = !choose;
  replace(identity(selected), make(33u));
  if (observation == 0u) {
    return read(first.payload);
  }
  if (observation == 1u) {
    return read(second.payload);
  }
  if (observation == 2u) {
    return read(watched);
  }
  if (observation == 3u) {
    return read(snapshot);
  }
  return selections;
}

[[loom::noinline]] static bool decide(bool choice, unsigned& decisions) {
  ++decisions;
  return choice;
}

unsigned selected_nested(bool outer, bool inner, unsigned observation) {
  Cell first{make(11u)}, second{make(12u)}, third{make(22u)};
  unsigned decisions = 0u;
  Cell& selected = outer ? (decide(inner, decisions) ? first : second) : third;
  const auto snapshot = selected.payload;
  replace(selected, make(33u));
  outer = !outer;
  inner = !inner;
  if (observation == 0u) {
    return read(first.payload);
  }
  if (observation == 1u) {
    return read(second.payload);
  }
  if (observation == 2u) {
    return read(third.payload);
  }
  if (observation == 3u) {
    return read(selected.payload);
  }
  if (observation == 4u) {
    return read(snapshot);
  }
  return decisions;
}

[[loom::noinline]] static Payload supply(Cell& value) {
  value.payload = make(22u);
  return make(33u);
}
[[loom::noinline]] static unsigned inspect_and_replace(Cell& value,
                                                       Payload next) {
  const auto previous = read(value.payload);
  value.payload = next;
  return previous;
}
[[loom::noinline]] static unsigned operator>>(Cell& value, Payload next) {
  return inspect_and_replace(value, next);
}

// The selected target is resolved before supply runs, but its value is read
// after the argument effects complete. Exercise free, member, and operator
// invocation with the same source observations.
unsigned selected_argument_order(bool choice, unsigned invocation,
                                 unsigned observation) {
  Cell first{make(11u)}, second{make(12u)};
  Cell& selected = choice ? first : second;
  unsigned previous = 0u;
  if (invocation == 0u) {
    previous = inspect_and_replace(selected, supply(selected));
  }
  if (invocation == 1u) {
    previous = selected.self().replace(supply(selected));
  }
  if (invocation == 2u) {
    previous = selected >> supply(selected);
  }
  if (observation == 0u) {
    return read(first.payload);
  }
  if (observation == 1u) {
    return read(second.payload);
  }
  if (observation == 2u) {
    return read(selected.payload);
  }
  return previous;
}

struct Pair {
  // First independently replaceable payload in the same source object.
  Payload first;
  // Second independently replaceable payload.
  Payload second;
};
[[loom::noinline]] static void assign(Payload& value, Payload next) {
  value = next;
}
unsigned selected_field(bool choice, unsigned observation) {
  Pair pair{make(11u), make(12u)};
  Payload& selected = choice ? pair.first : pair.second;
  const Payload& watched = selected;
  const auto snapshot = selected;
  assign(selected, make(33u));
  choice = !choice;
  if (observation == 0u) {
    return read(pair.first);
  }
  if (observation == 1u) {
    return read(pair.second);
  }
  if (observation == 2u) {
    return read(watched);
  }
  return read(snapshot);
}

// Repeated alternatives name the same field. A later unselected alternative
// must preserve the earlier update, and copied records remain independent.
unsigned repeated_field(unsigned choice) {
  Pair pair{make(11u), make(12u)};
  const auto snapshot = pair;
  Payload& selected = choice == 0u   ? pair.first
                      : choice == 1u ? pair.second
                      : choice == 2u ? pair.first
                                     : pair.second;
  assign(selected, make(22u));
  assign(selected, make(33u));
  return read(pair.first) + 100u * read(pair.second) +
         10000u * read(snapshot.first) + 1000000u * read(snapshot.second);
}

struct Partial {
  // A semantic member keeps the object in its source value representation.
  Payload payload;
  // First initialized candidate for the selected update.
  unsigned first;
  // Second initialized candidate for the selected update.
  unsigned second;
  // Unobserved state remains uninitialized through the update.
  unsigned unused;
};
unsigned partial_field(bool choice) {
  Partial value;
  value.payload = make(11u);
  value.first = 5u;
  value.second = 7u;
  unsigned& selected = choice ? value.first : value.second;
  selected += 3u;
  return 10u * value.first + value.second + read(value.payload);
}

struct Derived : Cell {
  // Neighboring field excluded from the selected base projection.
  unsigned marker;
};
unsigned selected_base(bool choice, unsigned observation) {
  Derived first{{make(11u)}, 101u}, second{{make(12u)}, 102u};
  Cell& selected =
      choice ? static_cast<Cell&>(first) : static_cast<Cell&>(second);
  const auto snapshot = selected.payload;
  replace(selected, make(33u));
  if (observation == 0u) {
    return read(first.payload);
  }
  if (observation == 1u) {
    return read(second.payload);
  }
  if (observation == 2u) {
    return first.marker;
  }
  if (observation == 3u) {
    return second.marker;
  }
  return read(snapshot);
}

[[loom::noinline]] static void transfer(Payload& output, const Payload& input) {
  output = input;
}
struct Envelope {
  // Field preceding the projected reference.
  unsigned prefix;
  // Nested field containing disjoint reference arguments.
  Pair pair;
  // Field following the projected reference.
  unsigned suffix;
  [[loom::noinline]] Envelope& self() { return *this; }
  [[loom::noinline]] void update(bool condition, Payload next) {
    if (condition) {
      self().pair.second = next;
    }
    transfer(self().pair.first, self().pair.second);
  }
};
unsigned returned_member(bool choice, unsigned observation) {
  Envelope value{100u, {make(11u), make(12u)}, 200u};
  value.update(choice, make(33u));
  if (observation == 0u) {
    return value.prefix;
  }
  if (observation == 1u) {
    return read(value.pair.first);
  }
  if (observation == 2u) {
    return read(value.pair.second);
  }
  return value.suffix;
}

struct Forwarded;
[[loom::noinline]] static void update(Forwarded& value);
[[loom::noinline]] static unsigned inspect(Forwarded& value);
struct Forwarded {
  // Payload changed by a free helper through the current receiver alias.
  Payload payload;
  [[loom::noinline]] Forwarded& self() { return *this; }
  [[loom::noinline]] unsigned observe() {
    const auto first = inspect(self());
    return first + inspect(*this);
  }
  [[loom::noinline]] void forward() { update(self()); }
};
[[loom::noinline]] static void update(Forwarded& value) {
  value.payload = make(33u);
}
[[loom::noinline]] static unsigned inspect(Forwarded& value) {
  return read(value.payload);
}
unsigned forwarded_receiver(bool mutate) {
  Forwarded value{make(11u)};
  if (mutate) {
    value.forward();
  }
  return value.observe();
}

[[loom::noinline]] static Cell& replace_pair(Cell& first, Cell& second) {
  first.payload = make(22u);
  second.payload = make(33u);
  return second;
}

// Independent selections share one ordinary call. Its reference result still
// names the second selected object after both caller bindings are restored.
unsigned independent_selections(bool choose_first, bool choose_second,
                                unsigned observation) {
  Cell first{make(11u)}, second{make(12u)};
  Cell third{make(22u)}, fourth{make(33u)};
  Cell& left = choose_first ? first : second;
  Cell& right = choose_second ? third : fourth;
  const auto snapshot = right.payload;
  Cell& returned = replace_pair(left, right);
  replace(returned, make(11u));
  if (observation == 0u) {
    return read(first.payload);
  }
  if (observation == 1u) {
    return read(second.payload);
  }
  if (observation == 2u) {
    return read(third.payload);
  }
  if (observation == 3u) {
    return read(fourth.payload);
  }
  if (observation == 4u) {
    return read(returned.payload);
  }
  return read(snapshot);
}

struct Observed {
  // Payload shared initially by distinct source objects.
  Payload payload;
  // Sum observed when each source object's lexical lifetime ends.
  unsigned* total;

  ~Observed() { *total += read(payload); }
};

// Equal component values do not merge two source identities or their cleanup.
unsigned selected_destruction(bool choose, unsigned observation) {
  unsigned total = 0u;
  unsigned snapshot = 0u;
  unsigned during_lifetime = 0u;
  {
    auto shared = make(11u);
    Observed first{shared, &total}, second{shared, &total};
    Observed& selected = choose ? first : second;
    const auto saved = selected.payload;
    assign(selected.payload, make(33u));
    snapshot = read(saved);
    during_lifetime = total;
  }
  if (observation == 0u) {
    return total;
  }
  return observation == 1u ? snapshot : during_lifetime;
}
