// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_PEER_DEVICE_FIXTURE_H_
#define AMDF_CTS_GPU_PEER_DEVICE_FIXTURE_H_

#include "libamdf/cts/gpu/gpu_device_fixture.h"

// Borrows exactly the two endpoints reserved by the runner. Endpoint IDs and
// native correlation keys do not establish physical topology or exclusivity;
// the runner supplies those facts through its inventory and resource lease.
class GpuPeerDeviceFixture : public GpuDeviceFixture {
 protected:
  void SetUp() override {
    if (!GetCtsDeviceCache().gpu_peer_endpoint_id().has_value()) {
      GTEST_SKIP() << "requires explicit primary and peer GPU endpoint IDs";
    }
    ASSERT_NO_FATAL_FAILURE(GpuDeviceFixture::SetUp());
    if (IsSkipped()) {
      return;
    }
    const auto status =
        GetCtsDeviceCache().GetGpuDevice(peer_endpoint_, &peer_device_);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      GTEST_SKIP() << "peer activation is unavailable for this native lifetime";
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(RecordGpuEndpointProperties(
        "amdf_gpu_peer", peer_endpoint_info_, peer_gpu_endpoint_info_));
  }

  // Cases select both native command families here, before either activation.
  // Failure leaves out_matches unchanged; success publishes the match result.
  virtual amdf_status_t MatchGpuPair(amdf_endpoint_t* primary,
                                     amdf_endpoint_t* peer, bool* out_matches) {
    (void)primary;
    (void)peer;
    *out_matches = true;
    return AMDF_STATUS_OK;
  }

  // Query endpoint selected by exact ID, retained by the shared cache.
  amdf_endpoint_t* peer_endpoint_ = nullptr;
  // Borrowed native device, released by the cache after all case resources.
  amdf_device_t* peer_device_ = nullptr;
  // Passive native correlation identity of the explicitly selected peer.
  amdf_endpoint_info_t peer_endpoint_info_ = {};
  // Passive compute target and geometry used by the peer's own commands.
  amdf_gpu_endpoint_info_t peer_gpu_endpoint_info_ = {};

 private:
  amdf_status_t MatchGpuEndpoint(amdf_endpoint_t* primary,
                                 bool* out_matches) final {
    const auto& peer_id = *GetCtsDeviceCache().gpu_peer_endpoint_id();
    amdf_status_t status =
        GetCtsDeviceCache().OpenEndpoint(peer_id, &peer_endpoint_);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    peer_endpoint_info_.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
    peer_endpoint_info_.structure_size = sizeof(peer_endpoint_info_);
    status = api_->endpoint_query_info(peer_endpoint_, &peer_endpoint_info_);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (peer_endpoint_info_.engine_kind != AMDF_ENGINE_KIND_GPU) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
    peer_gpu_endpoint_info_.type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO;
    peer_gpu_endpoint_info_.structure_size = sizeof(peer_gpu_endpoint_info_);
    status =
        gpu_api_->endpoint_query_info(peer_endpoint_, &peer_gpu_endpoint_info_);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    return MatchGpuPair(primary, peer_endpoint_, out_matches);
  }
};

#endif  // AMDF_CTS_GPU_PEER_DEVICE_FIXTURE_H_
