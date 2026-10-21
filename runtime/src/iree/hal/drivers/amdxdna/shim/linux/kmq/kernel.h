// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KERNEL_H
#define KERNEL_H

#include "bo.h"

namespace shim_xdna {

// Initializes the fixed ERT_START_NPU_PREEMPT_ELF payload. Kept separate from
// BO handling so the firmware packet ABI can be unit tested without a device.
void initialize_npu_elf_data(ert_npu_preempt_data* data,
                             uint64_t instruction_buffer,
                             uint32_t instruction_buffer_size);

// Returns whether |opcode| references an instruction BO that must be retained
// in the EXEC_CMD dependency list for the lifetime of the firmware command.
bool kernel_opcode_requires_instruction_bo_dependency(uint32_t opcode);

struct kernel {
  std::unique_ptr<bo> m_exec_buf_bo;
  ert_start_kernel_cmd* m_cmd_pkt = nullptr;
  size_t m_cmd_size = 0;
  uint32_t m_op = 0;
  uint32_t m_arg_cnt = 0;
  uint32_t m_reg_idx = 0;
  // START_DPU ert_dpu_data entries already written. The register map (opcode
  // uint64) begins after these, not after a single hardcoded entry.
  uint32_t m_dpu_count = 0;
  int m_init_errno = 0;
  std::vector<std::pair<std::string, uint64_t> > m_patching_args;
  std::vector<uint32_t> m_arg_reg_word_offsets;
  std::vector<uint32_t> m_arg_reg_word_counts;

  kernel(const pdev& p, uint32_t op);
  int init_errno() const;
  int reset();

  static void set_cu_idx(bo& bo_execbuf, cuidx_t cu_idx);
  void set_cu_idx(cuidx_t cu_idx);
  bo* get_exec_buf_bo() const;

  // `instruction_size` is the CERT-visible byte count (ctrlcode words). It
  // must not use the GEM BO size: GET_BO_INFO can round the allocation up,
  // and extra pages hang AIE4 START_DPU.
  int add_ctrl_bo(bo& bo_ctrl, size_t instruction_size);
  // Slices of `bo_ctrl`. count == 1 is chained=0 (KMD fill_direct_pkt).
  // count > 1 is XRT's per-column countdown (KMD fill_indirect_pkt).
  int add_dpu_columns(bo& bo_ctrl, size_t instruction_size,
                      const uint16_t* uc_index, const uint32_t* byte_offset,
                      const uint32_t* byte_size, uint32_t column_count);
  int add_arg_32(uint32_t val);
  int add_arg_64(uint64_t val);
  int update_arg_64(uint32_t arg_index, uint64_t val);
  int add_arg_bo(bo& bo_arg, const std::string& arg_name = "");
  // Like add_arg_bo but adds `offset` to the BO base before passing the
  // address to firmware. Required when a binding references a subview of a
  // larger BO at a non-zero offset.
  int add_arg_bo_at_offset(bo& bo_arg, uint64_t offset,
                           const std::string& arg_name = "");
  int inc_pkt_count(uint32_t n) const;
};
}  // namespace shim_xdna

#endif  // KERNEL_H
