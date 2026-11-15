// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Seventeen serial transfer geometries exceed a tile's sixteen descriptors.
// Reentering three earlier shapes must restore their lengths and addresses.
#include <loomcxx/kernel.h>
#include <loomcxx/pipeline.h>
#include <loomcxx/target/aie2p.h>

namespace t = loom::type;
using Bytes = t::buffer<unsigned char>;
template <t::size_type Count>
using Stream = t::channel<t::tile<t::shape<Count>, int>>;

template <t::size_type Count>
[[loom::inline]] static Stream<Count> make_channel(t::pool memory) {
  auto storage = loom::buffer::root(
      loom::buffer::alloca<loom::memory_space::workgroup, 64>(
          t::offset{Count * sizeof(int)}, memory));
  return loom::channel::bind<loom::channel::discipline::fifo>(
      loom::buffer::view<1, Count>(loom::buffer::cast<int>(storage), {}),
      t::index{1});
}

template <t::size_type Count>
[[loom::inline]] static void copy_phase(Stream<32> channel, Bytes input,
                                        Bytes output, t::offset destination) {
  auto source = loom::buffer::view<Count>(loom::buffer::cast<int>(input),
                                          t::offset{(Count - 1) * 128}, {});
  auto write = loom::channel::reserve(channel);
  for (unsigned i = 0; i < 32; ++i) {
    loom::view::store(-9999, write.payload, t::index{i});
  }
  auto incoming_values = loom::view::subview<Count>(write.payload, {0}, {});
  auto incoming = loom::kernel::async::copy<
      loom::kernel::async::direction::global_to_workgroup,
      loom::cache::scope::workgroup, loom::cache::temporal::regular>(
      source, incoming_values);
  loom::kernel::async::wait<0>(loom::kernel::async::group(incoming));
  for (unsigned i = 0; i < static_cast<unsigned>(Count); ++i) {
    int value = loom::view::load(write.payload, t::index{i});
    loom::view::store(value + static_cast<int>(Count), write.payload,
                      t::index{i});
  }
  loom::channel::publish(write.write);
  auto read = loom::channel::acquire(channel);
  auto target =
      loom::buffer::view<32>(loom::buffer::cast<int>(output), destination, {});
  auto outgoing = loom::kernel::async::copy<
      loom::kernel::async::direction::workgroup_to_global,
      loom::cache::scope::workgroup, loom::cache::temporal::regular>(
      read.payload, target);
  loom::kernel::async::wait<0>(loom::kernel::async::group(outgoing));
  loom::channel::release(read.read);
}

constexpr loom::aie2p::target descriptor_device{.kind = "array"};

[[loom::pipeline("kernel"), loom::symbol("descriptor_reentry"), loom::retain,
  loom::target(descriptor_device)]]
void descriptor_reentry([[loom::launch]] Bytes input_arg, Bytes output_arg) {
  auto input = loom::buffer::assume::alignment<64>(
      loom::buffer::assume::noalias(input_arg));
  auto output = loom::buffer::assume::alignment<64>(
      loom::buffer::assume::noalias(output_arg));
  auto memory = loom::pipeline::memory<loom::memory_space::workgroup>(
      t::indices{t::index{0}, t::index{2}});
  [[loom::serial]] {
    [[loom::concurrent]] {
      auto one = make_channel<32>(memory);
      [[loom::strand(t::indices{t::index{0}, t::index{2}},
                     t::indices{t::index{1}, t::index{1}},
                     t::indices{t::index{1}, t::index{1}})]] {
        copy_phase<1>(one, input, output, t::offset{0});
      }
    }
    [[loom::concurrent]] {
      auto two = make_channel<32>(memory);
      [[loom::strand(t::indices{t::index{0}, t::index{2}},
                     t::indices{t::index{1}, t::index{1}},
                     t::indices{t::index{1}, t::index{1}})]] {
        copy_phase<2>(two, input, output, t::offset{128});
      }
    }
    [[loom::concurrent]] {
      auto three = make_channel<32>(memory);
      [[loom::strand(t::indices{t::index{0}, t::index{2}},
                     t::indices{t::index{1}, t::index{1}},
                     t::indices{t::index{1}, t::index{1}})]] {
        copy_phase<3>(three, input, output, t::offset{2 * 128});
      }
    }
    [[loom::concurrent]] {
      auto four = make_channel<32>(memory);
      [[loom::strand(t::indices{t::index{0}, t::index{2}},
                     t::indices{t::index{1}, t::index{1}},
                     t::indices{t::index{1}, t::index{1}})]] {
        copy_phase<4>(four, input, output, t::offset{3 * 128});
      }
    }
    [[loom::concurrent]] {
      auto five = make_channel<32>(memory);
      [[loom::strand(t::indices{t::index{0}, t::index{2}},
                     t::indices{t::index{1}, t::index{1}},
                     t::indices{t::index{1}, t::index{1}})]] {
        copy_phase<5>(five, input, output, t::offset{4 * 128});
      }
    }
    [[loom::concurrent]] {
      auto six = make_channel<32>(memory);
      [[loom::strand(t::indices{t::index{0}, t::index{2}},
                     t::indices{t::index{1}, t::index{1}},
                     t::indices{t::index{1}, t::index{1}})]] {
        copy_phase<6>(six, input, output, t::offset{5 * 128});
      }
    }
    [[loom::concurrent]] {
      auto seven = make_channel<32>(memory);
      [[loom::strand(t::indices{t::index{0}, t::index{2}},
                     t::indices{t::index{1}, t::index{1}},
                     t::indices{t::index{1}, t::index{1}})]] {
        copy_phase<7>(seven, input, output, t::offset{6 * 128});
      }
    }
    [[loom::concurrent]] {
      auto eight = make_channel<32>(memory);
      [[loom::strand(t::indices{t::index{0}, t::index{2}},
                     t::indices{t::index{1}, t::index{1}},
                     t::indices{t::index{1}, t::index{1}})]] {
        copy_phase<8>(eight, input, output, t::offset{7 * 128});
      }
    }
    [[loom::concurrent]] {
      auto nine = make_channel<32>(memory);
      [[loom::strand(t::indices{t::index{0}, t::index{2}},
                     t::indices{t::index{1}, t::index{1}},
                     t::indices{t::index{1}, t::index{1}})]] {
        copy_phase<9>(nine, input, output, t::offset{8 * 128});
      }
    }
    [[loom::concurrent]] {
      auto ten = make_channel<32>(memory);
      [[loom::strand(t::indices{t::index{0}, t::index{2}},
                     t::indices{t::index{1}, t::index{1}},
                     t::indices{t::index{1}, t::index{1}})]] {
        copy_phase<10>(ten, input, output, t::offset{9 * 128});
      }
    }
    [[loom::concurrent]] {
      auto eleven = make_channel<32>(memory);
      [[loom::strand(t::indices{t::index{0}, t::index{2}},
                     t::indices{t::index{1}, t::index{1}},
                     t::indices{t::index{1}, t::index{1}})]] {
        copy_phase<11>(eleven, input, output, t::offset{10 * 128});
      }
    }
    [[loom::concurrent]] {
      auto twelve = make_channel<32>(memory);
      [[loom::strand(t::indices{t::index{0}, t::index{2}},
                     t::indices{t::index{1}, t::index{1}},
                     t::indices{t::index{1}, t::index{1}})]] {
        copy_phase<12>(twelve, input, output, t::offset{11 * 128});
      }
    }
    [[loom::concurrent]] {
      auto thirteen = make_channel<32>(memory);
      [[loom::strand(t::indices{t::index{0}, t::index{2}},
                     t::indices{t::index{1}, t::index{1}},
                     t::indices{t::index{1}, t::index{1}})]] {
        copy_phase<13>(thirteen, input, output, t::offset{12 * 128});
      }
    }
    [[loom::concurrent]] {
      auto fourteen = make_channel<32>(memory);
      [[loom::strand(t::indices{t::index{0}, t::index{2}},
                     t::indices{t::index{1}, t::index{1}},
                     t::indices{t::index{1}, t::index{1}})]] {
        copy_phase<14>(fourteen, input, output, t::offset{13 * 128});
      }
    }
    [[loom::concurrent]] {
      auto fifteen = make_channel<32>(memory);
      [[loom::strand(t::indices{t::index{0}, t::index{2}},
                     t::indices{t::index{1}, t::index{1}},
                     t::indices{t::index{1}, t::index{1}})]] {
        copy_phase<15>(fifteen, input, output, t::offset{14 * 128});
      }
    }
    [[loom::concurrent]] {
      auto sixteen = make_channel<32>(memory);
      [[loom::strand(t::indices{t::index{0}, t::index{2}},
                     t::indices{t::index{1}, t::index{1}},
                     t::indices{t::index{1}, t::index{1}})]] {
        copy_phase<16>(sixteen, input, output, t::offset{15 * 128});
      }
    }
    [[loom::concurrent]] {
      auto seventeen = make_channel<32>(memory);
      [[loom::strand(t::indices{t::index{0}, t::index{2}},
                     t::indices{t::index{1}, t::index{1}},
                     t::indices{t::index{1}, t::index{1}})]] {
        copy_phase<17>(seventeen, input, output, t::offset{16 * 128});
      }
    }
    [[loom::concurrent]] {
      auto one = make_channel<32>(memory);
      [[loom::strand(t::indices{t::index{0}, t::index{2}},
                     t::indices{t::index{1}, t::index{1}},
                     t::indices{t::index{1}, t::index{1}})]] {
        copy_phase<1>(one, input, output, t::offset{17 * 128});
      }
    }
    [[loom::concurrent]] {
      auto nine = make_channel<32>(memory);
      [[loom::strand(t::indices{t::index{0}, t::index{2}},
                     t::indices{t::index{1}, t::index{1}},
                     t::indices{t::index{1}, t::index{1}})]] {
        copy_phase<9>(nine, input, output, t::offset{18 * 128});
      }
    }
    [[loom::concurrent]] {
      auto seventeen = make_channel<32>(memory);
      [[loom::strand(t::indices{t::index{0}, t::index{2}},
                     t::indices{t::index{1}, t::index{1}},
                     t::indices{t::index{1}, t::index{1}})]] {
        copy_phase<17>(seventeen, input, output, t::offset{19 * 128});
      }
    }
  }
}
