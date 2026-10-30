// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <loomcxx/buffer.h>
#include <loomcxx/low.h>

struct [[loom::representation("amd.xdna.aie2p.core")]] Aie2p {};
struct [[loom::type("aie2p.load_fifo")]] LoadFifo {};
using Buffer = loom::type::buffer<unsigned char>;

struct FifoState {
  // First hardware load FIFO's retained contents.
  LoadFifo first;
  // Second hardware load FIFO's retained contents.
  LoadFifo second;
};

struct Stream {
  // Next 64-byte input block.
  Buffer input;
  // Next 64-byte output block.
  Buffer output;
  // Hardware state carried between native fragments.
  FifoState fifos;
};

FifoState query_load_fifos();

[[loom::force_inline]] void subtract_blocks(Buffer input, Buffer output) {
  Stream stream{input, output, query_load_fifos()};
  for (int block = 0; block < 4; ++block) {
    stream = loom::low::assembly<Aie2p, Stream>(R"loom(
        (%input: reg<aie2p.ep>, %output: reg<aie2p.ep>, %fifo0: reg<aie2p.eldfiforeg>, %fifo1: reg<aie2p.eldfiforeg>) -> (reg<aie2p.ep>, reg<aie2p.ep>, reg<aie2p.eldfiforeg>, reg<aie2p.eldfiforeg>) {
          %pointer0 = copy %input : reg<aie2p.ep> -> reg<aie2p.eps>
          %position0 = mova.fifo.position 0
          %filled_pointer0, %filled_fifo0, %filled_position0 = vlda.fill.512 %pointer0, %fifo0, %position0
          %values0, %end_pointer0, %end_fifo0, %end_position0 = vlda.pop.512.i32x16 %filled_pointer0, %filled_fifo0, %filled_position0
          %half = mova.i32 256
          %half_modifier = mov.modifier %half
          %upper_origin = copy %input : reg<aie2p.ep> -> reg<aie2p.ep>
          %upper = padda.modifier %upper_origin, %half_modifier
          %pointer1 = copy %upper : reg<aie2p.ep> -> reg<aie2p.eps>
          %position1 = mova.fifo.position 0
          %filled_pointer1, %filled_fifo1, %filled_position1 = vldb.fill.512 %pointer1, %fifo1, %position1
          %values1, %end_pointer1, %end_fifo1, %end_position1 = vldb.pop.512.i32x16 %filled_pointer1, %filled_fifo1, %filled_position1
          %difference = vsub.32 %values0, %values1
          vst.512.i32x16 %difference, %output, 0
          %step = mova.i32 64
          %modifier = mov.modifier %step
          %next_input = padda.modifier %input, %modifier
          %next_output = padda.modifier %output, %modifier
          return %next_input, %next_output, %end_fifo0, %end_fifo1
        }
    )loom",
                                                stream);
  }
}
