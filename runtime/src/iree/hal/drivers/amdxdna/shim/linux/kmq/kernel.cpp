// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "kernel.h"

#include <cerrno>
#include <cstdint>
#include <cstring>

#include "amdxdna_accel.h"
#include "bo.h"
#include "device.h"
#include "host_queue.h"

#define MAX_EXEC_BO_SIZE 4096

namespace shim_xdna {
namespace {

int check_pkt_count_capacity(const kernel& k, uint32_t n) {
  if (!k.m_cmd_pkt) return k.m_init_errno ? k.m_init_errno : EINVAL;
  uint32_t next_count = k.m_cmd_pkt->count + n / sizeof(int32_t);
  if (k.m_cmd_size <
      sizeof(k.m_cmd_pkt->header) + next_count * sizeof(int32_t)) {
    return E2BIG;
  }
  return 0;
}

uint32_t* kernel_regmap(kernel& k) {
  if (k.m_op == ERT_START_DPU) {
    const uint32_t entries = k.m_dpu_count ? k.m_dpu_count : 1;
    return k.m_cmd_pkt->data + k.m_cmd_pkt->extra_cu_masks +
           entries * static_cast<uint32_t>(sizeof(amdxdna_cmd_start_dpu) /
                                           sizeof(uint32_t));
  }
  return get_ert_regmap_begin(k.m_cmd_pkt);
}

}  // namespace

void initialize_npu_elf_data(ert_npu_preempt_data* data,
                             uint64_t instruction_buffer,
                             uint32_t instruction_buffer_size) {
  std::memset(data, 0, sizeof(*data));
  data->instruction_buffer = instruction_buffer;
  data->instruction_buffer_size = instruction_buffer_size;
}

bool kernel_opcode_requires_instruction_bo_dependency(uint32_t opcode) {
  switch (opcode) {
    case ERT_START_NPU:
    case ERT_START_NPU_PREEMPT_ELF:
    case ERT_START_DPU:
      return true;
    default:
      return false;
  }
}

kernel::kernel(const pdev& p, uint32_t op) : m_op(op) {
  m_init_errno = bo::create(p, AMDXDNA_INVALID_CTX_HANDLE, MAX_EXEC_BO_SIZE,
                            AMDXDNA_BO_FLAGS_EXECBUF, &m_exec_buf_bo);
  if (m_init_errno) return;
  m_cmd_pkt = reinterpret_cast<ert_start_kernel_cmd*>(m_exec_buf_bo->map());
  if (!m_cmd_pkt) {
    m_init_errno = EINVAL;
    return;
  }
  m_cmd_size = m_exec_buf_bo->size();
  m_init_errno = reset();
}

int kernel::init_errno() const { return m_init_errno; }

int kernel::reset() {
  if (!m_cmd_pkt) return m_init_errno ? m_init_errno : EINVAL;
  std::memset(m_cmd_pkt, 0, m_cmd_size);
  m_arg_cnt = 0;
  m_reg_idx = 0;
  m_dpu_count = 0;
  m_patching_args.clear();
  m_arg_reg_word_offsets.clear();
  m_arg_reg_word_counts.clear();
  m_cmd_pkt->state = ERT_CMD_STATE_NEW;
  m_cmd_pkt->opcode = m_op;
  m_cmd_pkt->type = ERT_CU;
  // One word for cu mask
  return inc_pkt_count(sizeof(int32_t));
}

void kernel::set_cu_idx(bo& bo_execbuf, cuidx_t cu_idx) {
  ert_start_kernel_cmd* cmd_pkt =
      reinterpret_cast<ert_start_kernel_cmd*>(bo_execbuf.map());
  cmd_pkt->cu_mask = 0x1 << cu_idx.index;
}

void kernel::set_cu_idx(cuidx_t cu_idx) {
  m_cmd_pkt->cu_mask = 0x1 << cu_idx.index;
}

int kernel::add_ctrl_bo(bo& bo_ctrl, size_t instruction_size) {
  if (instruction_size == 0 || instruction_size > bo_ctrl.size() ||
      (instruction_size % sizeof(uint32_t)) != 0) {
    return EINVAL;
  }
  if (instruction_size > UINT32_MAX) return E2BIG;
  const uint32_t instr_bytes = static_cast<uint32_t>(instruction_size);
  ert_start_kernel_cmd* cmd_packet =
      reinterpret_cast<ert_start_kernel_cmd*>(m_exec_buf_bo->map());
  switch (m_op) {
    case ERT_START_CU:
      return 0;
    case ERT_START_NPU: {
      int err = inc_pkt_count(sizeof(ert_npu_data));
      if (err) return err;
      ert_npu_data* npu_data = get_ert_npu_data(cmd_packet);
      npu_data->instruction_buffer = bo_ctrl.get_paddr();
      npu_data->instruction_buffer_size = instr_bytes;
      npu_data->instruction_prop_count = 0;
      break;
    }
    case ERT_START_NPU_PREEMPT_ELF: {
      int err = inc_pkt_count(sizeof(ert_npu_preempt_data));
      if (err) return err;
      ert_npu_preempt_data* elf_data = get_ert_npu_elf_data(cmd_packet);
      if (!elf_data) return EINVAL;
      initialize_npu_elf_data(elf_data, bo_ctrl.get_paddr(), instr_bytes);
      break;
    }
    case ERT_START_DPU: {
      // One column (or a caller that has not split columns): KMD
      // fill_direct_pkt. Multi-column ELFs use add_dpu_columns.
      int err = inc_pkt_count(sizeof(amdxdna_cmd_start_dpu));
      if (err) return err;
      auto* dpu = reinterpret_cast<amdxdna_cmd_start_dpu*>(
          cmd_packet->data + cmd_packet->extra_cu_masks);
      dpu->dtrace_buffer = 0;
      dpu->instruction_buffer = bo_ctrl.get_paddr();
      dpu->instruction_buffer_size = instr_bytes;
      dpu->uc_index = 0;
      dpu->chained = 0;
      m_dpu_count = 1;
      break;
    }
    default:
      return EINVAL;
  }
  if (!kernel_opcode_requires_instruction_bo_dependency(m_op)) return 0;
  return m_exec_buf_bo->bind_at(kExecBoInstructionArgKey, bo_ctrl, 0,
                                instruction_size);
}

int kernel::add_dpu_columns(bo& bo_ctrl, size_t instruction_size,
                            const uint16_t* uc_index,
                            const uint32_t* byte_offset,
                            const uint32_t* byte_size, uint32_t column_count) {
  if (m_op != ERT_START_DPU) return EINVAL;
  if (column_count == 0 || column_count > HSA_MAX_LEVEL1_INDIRECT_ENTRIES) {
    return EINVAL;
  }
  if (!uc_index || !byte_offset || !byte_size) return EINVAL;
  if (instruction_size == 0 || instruction_size > bo_ctrl.size() ||
      (instruction_size % sizeof(uint32_t)) != 0) {
    return EINVAL;
  }
  if (instruction_size > UINT32_MAX) return E2BIG;
  if (m_dpu_count != 0) return EINVAL;
  const uint64_t base = bo_ctrl.get_paddr();
  ert_start_kernel_cmd* cmd_packet =
      reinterpret_cast<ert_start_kernel_cmd*>(m_exec_buf_bo->map());
  auto* dpu = reinterpret_cast<amdxdna_cmd_start_dpu*>(
      cmd_packet->data + cmd_packet->extra_cu_masks);
  for (uint32_t i = 0; i < column_count; ++i) {
    if (uc_index[i] >= HSA_MAX_LEVEL1_INDIRECT_ENTRIES) return EINVAL;
    const uint64_t end = (uint64_t)byte_offset[i] + (uint64_t)byte_size[i];
    if (byte_size[i] == 0 || (byte_size[i] % sizeof(uint32_t)) != 0 ||
        end > instruction_size) {
      return EINVAL;
    }
    int err = inc_pkt_count(sizeof(amdxdna_cmd_start_dpu));
    if (err) return err;
    dpu[i].dtrace_buffer = 0;
    dpu[i].instruction_buffer = base + byte_offset[i];
    dpu[i].instruction_buffer_size = byte_size[i];
    dpu[i].uc_index = uc_index[i];
    dpu[i].chained = static_cast<uint16_t>(column_count - 1 - i);
  }
  m_dpu_count = column_count;
  return m_exec_buf_bo->bind_at(kExecBoInstructionArgKey, bo_ctrl, 0,
                                instruction_size);
}

int kernel::add_arg_32(uint32_t val) {
  int err = inc_pkt_count(sizeof(val));
  if (err) return err;
  auto args = kernel_regmap(*this);
  m_arg_reg_word_offsets.push_back(m_reg_idx);
  m_arg_reg_word_counts.push_back(1);
  args[m_reg_idx++] = val;
  m_arg_cnt++;
  return 0;
}

int kernel::add_arg_64(uint64_t val) {
  int err = inc_pkt_count(sizeof(val));
  if (err) return err;
  auto args = kernel_regmap(*this);
  m_arg_reg_word_offsets.push_back(m_reg_idx);
  m_arg_reg_word_counts.push_back(2);
  args[m_reg_idx++] = val;
  args[m_reg_idx++] = val >> 32;
  m_arg_cnt++;
  return 0;
}

int kernel::update_arg_64(uint32_t arg_index, uint64_t val) {
  if (arg_index >= m_arg_reg_word_offsets.size() ||
      arg_index >= m_arg_reg_word_counts.size()) {
    return ERANGE;
  }
  if (m_arg_reg_word_counts[arg_index] != 2) {
    return EINVAL;
  }
  uint32_t word_index = m_arg_reg_word_offsets[arg_index];
  if (word_index + 1 >= m_reg_idx) return ERANGE;
  auto args = kernel_regmap(*this);
  args[word_index] = val;
  args[word_index + 1] = val >> 32;
  return 0;
}

int kernel::add_arg_bo(bo& bo_arg, const std::string& arg_name) {
  int err = check_pkt_count_capacity(*this, sizeof(uint64_t));
  if (err) return err;
  // Add to argument list for driver
  err = m_exec_buf_bo->bind_at(m_arg_cnt, bo_arg, 0, bo_arg.size());
  if (err) return err;
  // Add to argument list for control code patching
  if (arg_name.empty())
    m_patching_args.emplace_back(std::to_string(m_arg_cnt), bo_arg.get_paddr());
  else
    m_patching_args.emplace_back(arg_name, bo_arg.get_paddr());
  // Only increase m_arg_cnt now after it's used by code above.
  return add_arg_64(bo_arg.get_paddr());
}

int kernel::add_arg_bo_at_offset(bo& bo_arg, uint64_t offset,
                                 const std::string& arg_name) {
  if (offset > bo_arg.size()) return EINVAL;
  int err = check_pkt_count_capacity(*this, sizeof(uint64_t));
  if (err) return err;
  // Bind starting at `offset` so the driver tracks the slice this dispatch
  // actually touches. Size is capped to remaining BO bytes.
  err =
      m_exec_buf_bo->bind_at(m_arg_cnt, bo_arg, offset, bo_arg.size() - offset);
  if (err) return err;
  uint64_t paddr = bo_arg.get_paddr() + offset;
  if (arg_name.empty())
    m_patching_args.emplace_back(std::to_string(m_arg_cnt), paddr);
  else
    m_patching_args.emplace_back(arg_name, paddr);
  return add_arg_64(paddr);
}

int kernel::inc_pkt_count(uint32_t n) const {
  int err = check_pkt_count_capacity(*this, n);
  if (err) return err;
  m_cmd_pkt->count += n / sizeof(int32_t);
  return 0;
}

bo* kernel::get_exec_buf_bo() const { return m_exec_buf_bo.get(); }

}  // namespace shim_xdna
