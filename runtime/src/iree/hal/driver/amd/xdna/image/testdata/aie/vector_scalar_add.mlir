// Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module {
  aie.device(npu2_1col) {
    %logical_core = aie.logical_tile<CoreTile>(?, ?)
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_shim_noc_0 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in0(%logical_shim_noc, {%logical_core}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out(%logical_core, {%logical_shim_noc_0}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    %0 = aie.core(%logical_core) {
      scf.while : () -> () {
        %true = arith.constant true
        scf.condition(%true)
      } do {
        %c0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c1 = arith.constant 1 : index
        scf.for %arg0 = %c0 to %c32 step %c1 {
          %1 = aie.objectfifo.acquire @in0(Consume, 1) : memref<32xi32>
          %2 = aie.objectfifo.acquire @out(Produce, 1) : memref<32xi32>
          %c0_1 = arith.constant 0 : index
          %c32_2 = arith.constant 32 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg1 = %c0_1 to %c32_2 step %c1_3 {
            %3 = memref.load %1[%arg1] : memref<32xi32>
            %c1_i32 = arith.constant 1 : i32
            %4 = arith.addi %3, %c1_i32 : i32
            memref.store %4, %2[%arg1] : memref<32xi32>
          }
          aie.objectfifo.release @in0(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
        }
        scf.yield
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
      aiex.npu.load_pdi {device_ref = @main}
      %1 = aiex.dma_configure_task_for @in0 {
        aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 1024 sizes = [1, 1, 1, 1024] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<1024xi32> offset = 0 len = 1024 sizes = [1, 1, 1, 1024] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
    }
  }
}
