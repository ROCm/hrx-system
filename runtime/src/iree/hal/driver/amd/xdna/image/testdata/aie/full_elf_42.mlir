// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module {
  aie.device(npu2_1col) {
    %logical_core = aie.logical_tile<CoreTile>(?, ?)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @objfifo_out(%logical_core, {%logical_shim_noc}, 2 : i32) : !aie.objectfifo<memref<1xi32>>
    %0 = aie.core(%logical_core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c1 step %c1_0 {
        %1 = aie.objectfifo.acquire @objfifo_out(Produce, 1) : memref<1xi32>
        %c0_1 = arith.constant 0 : index
        %c42_i32 = arith.constant 42 : i32
        memref.store %c42_i32, %1[%c0_1] : memref<1xi32>
        aie.objectfifo.release @objfifo_out(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<1xi32>) {
      aiex.npu.load_pdi {device_ref = @main}
      %1 = aiex.dma_configure_task_for @objfifo_out {
        aie.dma_bd(%arg0 : memref<1xi32> offset = 0 len = 1 sizes = [1, 1, 1, 1] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%1)
    }
  }
}
