// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "hwq.h"

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <limits>
#include <mutex>

#include "device.h"

#include "../../ert.h"
#include "bo.h"
#include "fence.h"
#include "host_queue.h"
#include "shim_debug.h"

namespace {

constexpr size_t kUmqSlots = 32;

void init_indirect_buf(volatile struct host_indirect_data* indirect_buf,
                       int size) {
  for (int i = 0; i < size; i++) {
    indirect_buf[i].header.type = HOST_QUEUE_PACKET_TYPE_VENDOR_SPECIFIC;
    indirect_buf[i].header.opcode = HOST_QUEUE_PACKET_EXEC_BUF;
    indirect_buf[i].header.count = sizeof(struct exec_buf);
    indirect_buf[i].header.distribute = 1;
    indirect_buf[i].header.indirect = 0;
  }
}

bool valid_queue_index(uint64_t read, uint64_t write, uint32_t capacity) {
  return (write >= read) && ((write - read) <= capacity);
}

void fill_direct_exec_buf(volatile struct host_queue_packet* pkt,
                          const struct amdxdna_cmd_start_dpu* dpu) {
  auto* data = const_cast<uint32_t*>(pkt->data);
  std::memset(data, 0, sizeof(struct exec_buf));
  volatile struct exec_buf* ebp =
      reinterpret_cast<volatile struct exec_buf*>(pkt->data);
  ebp->dpu_control_code_host_addr_low =
      static_cast<uint32_t>(dpu->instruction_buffer);
  ebp->dpu_control_code_host_addr_high =
      static_cast<uint32_t>(dpu->instruction_buffer >> 32);
  ebp->dtrace_buf_host_addr_low = 0;
  ebp->dtrace_buf_host_addr_high = 0;
  auto* hdr = &pkt->xrt_header;
  hdr->common_header.distribute = 0;
  hdr->common_header.indirect = 0;
  hdr->common_header.count = sizeof(struct exec_buf);
}

int RetryIoctl(int fd, unsigned long request, void* arg) {
  int ret = 0;
  do {
    ret = ::ioctl(fd, request, arg);
  } while (ret == -1 && errno == EINTR);
  return ret;
}

uint64_t abs_now_ns() {
  auto now = std::chrono::high_resolution_clock::now();
  auto now_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
  return now_ns.time_since_epoch().count();
}

ert_packet* get_chained_command_pkt(shim_xdna::bo* boh) {
  ert_packet* cmdpkt = reinterpret_cast<ert_packet*>(boh->map());
  return cmdpkt->opcode == ERT_CMD_CHAIN ? cmdpkt : nullptr;
}

bool ert_state_is_terminal(uint32_t state) {
  return state == ERT_CMD_STATE_COMPLETED || state == ERT_CMD_STATE_ERROR ||
         state == ERT_CMD_STATE_ABORT || state == ERT_CMD_STATE_TIMEOUT ||
         state == ERT_CMD_STATE_NORESPONSE;
}

// UMQ doorbell submit never registers a seq with EXEC_CMD, so WAIT_CMD /
// syncobj wait return EINVAL. CERT writes the ERT header through the mapped
// CMD BO (completion_signal); poll that instead. The CMD BO is not
// host-coherent, so invalidate the header before each read.
int wait_umq_completion(shim_xdna::bo* cmd, uint32_t timeout_ms) {
  const bool timed = timeout_ms != 0;
  const auto start = std::chrono::steady_clock::now();
  for (;;) {
    if (const int err =
            cmd->sync(shim_xdna::direction::device2host, sizeof(uint32_t), 0)) {
      return -err;
    }
    auto* pkt = reinterpret_cast<volatile ert_packet*>(cmd->map());
    if (pkt && ert_state_is_terminal(pkt->state)) return 1;
    if (timed) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
      if (elapsed >= static_cast<int64_t>(timeout_ms)) return 0;
    }
    usleep(1000);
  }
}

int wait_cmd(const shim_xdna::pdev& pdev, const shim_xdna::hw_ctx* ctx,
             shim_xdna::bo* cmd, uint32_t timeout_ms) {
  int ret = 1;
  auto id = cmd->get_cmd_id();
  uint32_t syncobj = ctx->m_syncobj;

  SHIM_DEBUG("Waiting for cmd (%ld)...", id);

  if (syncobj != AMDXDNA_INVALID_FENCE_HANDLE) {
    int64_t timeout = std::numeric_limits<int64_t>::max();
    if (timeout_ms) {
      timeout = timeout_ms;
      timeout *= 1000000;
      timeout += abs_now_ns();
    }
    drm_syncobj_timeline_wait wsobj = {
        .handles = reinterpret_cast<uintptr_t>(&syncobj),
        .points = reinterpret_cast<uintptr_t>(&id),
        .timeout_nsec = timeout,
        .count_handles = 1,
        .flags = 0,
    };
    if (RetryIoctl(pdev.m_dev_fd, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wsobj) ==
        -1) {
      if (errno == ETIME) {
        ret = 0;
      } else {
        return -errno;
      }
    }
  } else {
    amdxdna_drm_wait_cmd wcmd = {
        .hwctx = ctx->m_handle,
        .timeout = timeout_ms,
        .seq = id,
    };
    // Prefer the IOWR encoding shipped with older AIE2P DKMS. xdna-driver
    // 2.26 registers IOW; ENOTTY means retry the other encoding.
    static std::atomic<unsigned long> wait_cmd_ioctl{
        DRM_IOCTL_AMDXDNA_WAIT_CMD};
    unsigned long request = wait_cmd_ioctl.load(std::memory_order_relaxed);
    if (RetryIoctl(pdev.m_dev_fd, request, &wcmd) == -1) {
      if (errno == ENOTTY) {
        const unsigned long other =
            (request == DRM_IOCTL_AMDXDNA_WAIT_CMD)
                ? DRM_IOCTL_AMDXDNA_WAIT_CMD_WRITEONLY
                : DRM_IOCTL_AMDXDNA_WAIT_CMD;
        if (RetryIoctl(pdev.m_dev_fd, other, &wcmd) != -1) {
          wait_cmd_ioctl.store(other, std::memory_order_relaxed);
        } else if (errno == ETIME) {
          ret = 0;
        } else {
          return -errno;
        }
      } else if (errno == ETIME) {
        ret = 0;
      } else {
        return -errno;
      }
    }
  }

  return ret;
}

}  // namespace

namespace shim_xdna {

hw_q::hw_q(const device& device)
    : m_hwctx(nullptr),
      m_pdev(device.get_pdev()),
      m_queue_boh(AMDXDNA_INVALID_BO_HANDLE) {
  SHIM_DEBUG("Created KMQ HW queue");
}

void hw_q::bind_hwctx(const hw_ctx* ctx) {
  m_hwctx = ctx;
  SHIM_DEBUG("Bond HW queue to HW context %d", m_hwctx->m_handle);
}

void hw_q::unbind_hwctx() {
  if (!m_hwctx) return;
  SHIM_DEBUG("Unbond HW queue from HW context %d", m_hwctx->m_handle);
  m_hwctx = nullptr;
}

int hw_q::init_umq() {
  const size_t header_sz = sizeof(struct host_queue_header);
  const size_t queue_sz = sizeof(struct host_queue_packet) * kUmqSlots;
  const size_t indirect_sz = sizeof(struct host_indirect_data) *
                             HSA_MAX_LEVEL1_INDIRECT_ENTRIES * kUmqSlots;
  const size_t umq_sz = header_sz + queue_sz + indirect_sz;
  int err = bo::create(m_pdev, umq_sz, AMDXDNA_BO_CMD, &m_umq_bo);
  if (err) return err;
  void* buf = m_umq_bo->map();
  if (!buf) return EINVAL;
  std::memset(buf, 0, umq_sz);

  m_umq_hdr = reinterpret_cast<volatile struct host_queue_header*>(buf);
  m_umq_pkt = reinterpret_cast<volatile struct host_queue_packet*>(
      reinterpret_cast<uintptr_t>(buf) + header_sz);
  m_umq_indirect_buf = reinterpret_cast<volatile struct host_indirect_data*>(
      reinterpret_cast<uintptr_t>(buf) + header_sz + queue_sz);
  for (size_t i = 0; i < kUmqSlots; i++) {
    init_indirect_buf(
        &m_umq_indirect_buf[i * HSA_MAX_LEVEL1_INDIRECT_ENTRIES],
        HSA_MAX_LEVEL1_INDIRECT_ENTRIES);
  }
  m_umq_hdr->version.major = HOST_QUEUE_MAJOR_VERSION;
  m_umq_hdr->version.minor = HOST_QUEUE_MINOR_VERSION;
  m_umq_hdr->capacity = kUmqSlots;
  m_umq_hdr->data_address = m_umq_bo->get_paddr() + header_sz;
  m_indirect_paddr = m_umq_hdr->data_address + queue_sz;
  m_umq_slots = kUmqSlots;
  m_queue_boh = m_umq_bo->get_drm_bo_handle();
  // Firmware reads this BO through SVA. Publish the header before CREATE_HWCTX.
  if (int sync_err = m_umq_bo->sync(direction::host2device)) return sync_err;
  SHIM_DEBUG("Created UMQ host queue, size=%zu handle=%u", umq_sz, m_queue_boh);
  return 0;
}

int hw_q::map_doorbell(uint32_t doorbell_offset) {
  if (m_doorbell) {
    munmap(const_cast<uint32_t*>(m_doorbell),
           m_doorbell_map_size ? m_doorbell_map_size : sizeof(uint32_t));
    m_doorbell = nullptr;
    m_doorbell_map_size = 0;
  }
  if (doorbell_offset == AMDXDNA_INVALID_DOORBELL_OFFSET) {
    SHIM_DEBUG("UMQ doorbell not provided; using kernel-mode EXEC_CMD");
    return 0;
  }
  void* mapped = nullptr;
  size_t mapped_size = sizeof(uint32_t);
  int err = m_pdev.try_mmap(nullptr, mapped_size, PROT_WRITE, MAP_SHARED,
                            static_cast<off_t>(doorbell_offset), &mapped);
  if (err) {
    mapped_size = 4096;
    err = m_pdev.try_mmap(nullptr, mapped_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, static_cast<off_t>(doorbell_offset),
                          &mapped);
  }
  if (err) return err;
  m_doorbell = reinterpret_cast<volatile uint32_t*>(mapped);
  m_doorbell_map_size = mapped_size;
  SHIM_DEBUG("Mapped UMQ doorbell offset=0x%x -> %p", doorbell_offset,
             mapped);
  return 0;
}

int hw_q::get_next_umq_slot(uint32_t* out_slot) {
  if (!m_umq_hdr) return EINVAL;
  auto* h = m_umq_hdr;
  for (;;) {
    // read_index is firmware-owned and lives in the same non-coherent BO.
    if (int sync_err = m_umq_bo->sync(direction::device2host)) return sync_err;
    uint64_t wi = h->write_index;
    uint64_t ri = h->read_index;
    if (!valid_queue_index(ri, wi, h->capacity)) {
      usleep(100);
      wi = h->write_index;
      ri = h->read_index;
      if (!valid_queue_index(ri, wi, h->capacity)) return EINVAL;
    }
    if ((wi - ri) < h->capacity) {
      *out_slot = static_cast<uint32_t>(wi & (h->capacity - 1));
      return 0;
    }
    usleep(100);
  }
}

int hw_q::issue_umq_exec_buf(bo* cmd_bo) {
  auto* cmd = reinterpret_cast<ert_start_kernel_cmd*>(cmd_bo->map());
  if (!cmd || cmd->opcode != ERT_START_DPU) return EINVAL;
  auto* dpu = reinterpret_cast<amdxdna_cmd_start_dpu*>(
      cmd->data + cmd->extra_cu_masks);
  if (!dpu) return EINVAL;
  if (dpu->chained) return ENOTSUP;
  uint32_t slot_idx = 0;
  int err = get_next_umq_slot(&slot_idx);
  if (err) return err;
  fill_direct_exec_buf(&m_umq_pkt[slot_idx], dpu);
  auto* pkt = &m_umq_pkt[slot_idx];
  auto* hdr = &pkt->xrt_header;
  hdr->common_header.opcode = HOST_QUEUE_PACKET_EXEC_BUF;
  hdr->common_header.chain_flag = LAST_CMD;
  hdr->completion_signal =
      cmd_bo->get_paddr() + offsetof(ert_start_kernel_cmd, header);
  hdr->common_header.type = HOST_QUEUE_PACKET_TYPE_VENDOR_SPECIFIC;
  std::atomic_thread_fence(std::memory_order_seq_cst);
  uint64_t wi = m_umq_hdr->write_index;
  m_umq_hdr->write_index = wi + 1;
  // Packet and write_index must reach DRAM before the doorbell.
  if (int sync_err = m_umq_bo->sync(direction::host2device)) return sync_err;
  if (m_doorbell) *m_doorbell = 0;
  cmd_bo->set_cmd_id(wi);
  SHIM_DEBUG("Submitted UMQ exec_buf slot=%u seq=%lu", slot_idx,
             (unsigned long)wi);
  return 0;
}

int hw_q::wait_command(bo* cmd, uint32_t timeout_ms) const {
  if (poll_command(cmd)) return 1;
  if (m_umq_hdr && m_doorbell) return wait_umq_completion(cmd, timeout_ms);
  // AIE4 KMS wait returns EAGAIN while the ctx is reconnecting after TDR.
  for (int attempt = 0;; ++attempt) {
    const int rc = wait_cmd(m_pdev, m_hwctx, cmd, timeout_ms);
    if (rc >= 0) return rc;
    if (rc != -EAGAIN && rc != -EINTR) return rc;
    SHIM_DEBUG("WAIT_CMD retrying errno=%d seq=%llu attempt=%d", -rc,
               (unsigned long long)cmd->get_cmd_id(), attempt);
    if (attempt >= 30) return rc;
    sleep(1);
  }
}

int hw_q::submit_wait(const fence_handle* f, uint64_t* out_state) {
  return f->submit_wait(m_hwctx, out_state);
}

int hw_q::submit_wait(const std::vector<fence_handle*>& fences,
                      uint64_t* out_last_state) {
  return fence_handle::submit_wait(m_pdev, m_hwctx, fences, out_last_state);
}

int hw_q::submit_signal(const fence_handle* f, uint64_t* out_state) {
  return f->submit_signal(m_hwctx, out_state);
}

hw_q::~hw_q() {
  if (m_doorbell) {
    munmap(const_cast<uint32_t*>(m_doorbell),
           m_doorbell_map_size ? m_doorbell_map_size : sizeof(uint32_t));
    m_doorbell = nullptr;
    m_doorbell_map_size = 0;
  }
  SHIM_DEBUG("Destroying KMQ HW queue");
}

int hw_q::issue_command(bo* cmd_bo) {
  // Held across the ioctl so a timeout recovery, which takes the same mutex,
  // cannot destroy this context between the handle read and EXEC_CMD. Recovery
  // itself runs on the waiter, after this ioctl has returned.
  std::lock_guard<std::mutex> submit_gate(m_pdev.m_submit_mu);
  // Userspace doorbell is optional. Firmware may return
  // AMDXDNA_INVALID_DOORBELL_OFFSET, so CREATE_HWCTX still takes a host-queue
  // BO and submit uses kernel-mode EXEC_CMD. UMS emits only a direct packet
  // (chained==0). A per-column chain keeps using EXEC_CMD even when a
  // doorbell is mapped, so KMD fill_indirect_pkt names each microcontroller.
  // There is no userspace indirect encoder.
  if (m_umq_hdr && m_doorbell) {
    auto* cmd = reinterpret_cast<ert_start_kernel_cmd*>(cmd_bo->map());
    auto* dpu =
        (cmd && cmd->opcode == ERT_START_DPU)
            ? reinterpret_cast<amdxdna_cmd_start_dpu*>(cmd->data +
                                                       cmd->extra_cu_masks)
            : nullptr;
    if (dpu && dpu->chained == 0) {
      int err = issue_umq_exec_buf(cmd_bo);
      if (err) return err;
      m_exec_cmd_count.fetch_add(1, std::memory_order_relaxed);
      return 0;
    }
  }
  uint32_t arg_bo_hdls[kMaxArgBosPerCommand];
  uint32_t cmd_bo_hdl = cmd_bo->get_drm_bo_handle();
  uint32_t arg_count = 0;
  int err =
      cmd_bo->get_arg_bo_handles(arg_bo_hdls, kMaxArgBosPerCommand, &arg_count);
  if (err) return err;

  amdxdna_drm_exec_cmd ecmd = {
      .hwctx = m_hwctx->m_handle,
      .type = AMDXDNA_CMD_SUBMIT_EXEC_BUF,
      .cmd_handles = cmd_bo_hdl,
      .args = reinterpret_cast<uint64_t>(arg_bo_hdls),
      .cmd_count = 1,
      .arg_count = arg_count,
  };

  err = m_pdev.try_ioctl(DRM_IOCTL_AMDXDNA_EXEC_CMD, &ecmd);
  if (err) return err;
  m_exec_cmd_count.fetch_add(1, std::memory_order_relaxed);

  auto id = ecmd.seq;
  cmd_bo->set_cmd_id(id);
  SHIM_DEBUG("Submitted command (%ld)", id);
  return 0;
}

int poll_command(bo* cmd) {
  // Same non-coherent CMD BO as the blocking wait. Invalidate the header
  // before trusting CERT's state. A failed flush is "not ready".
  if (cmd->sync(direction::device2host, sizeof(uint32_t), 0)) return 0;
  ert_packet* cmdpkt = reinterpret_cast<ert_packet*>(cmd->map());
  // Only ERT_CMD_STATE_COMPLETED means "finished successfully". The ert state
  // enum is NOT monotonic past COMPLETED(4): SUBMITTED(7) is a pre-completion
  // state, and ERROR(5)/ABORT(6)/TIMEOUT(8)/NORESPONSE(9) are terminal
  // failures. A `>= COMPLETED` test wrongly reported all of those as done. Let
  // terminal-error states fall through to the syncobj wait (their fence is
  // signaled) so the caller's ert-state check surfaces the failure.
  //
  // AIE4 CERT writes COMPLETED as the entire header word, so opcode becomes
  // 0. AIE2P updates the state nibble and leaves the START_CU/START_NPU
  // opcode in place, so this check still reports those completions. A zero
  // opcode is a stale CERT write on a reused exec BO, not a new completion.
  if (cmdpkt->state == ERT_CMD_STATE_COMPLETED && cmdpkt->opcode != 0) {
    return 1;
  }
  return 0;
}

}  // namespace shim_xdna
