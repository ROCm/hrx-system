// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_D3D12_COMMAND_H_
#define AMDF_SRC_MEDIA_D3D12_COMMAND_H_

// DXVA consumes Windows scalar types and DEFINE_GUID without including them.
// clang-format off
#include <windows.h>
#include <dxva.h>
// clang-format on

#include "libamdf/src/media/d3d12/session.h"

namespace amdf::media::d3d12 {

struct Command {
  // Provider-neutral recording owner and immutable session association.
  amdf_media_command_t base;
  // One application-selected native recording allocator.
  ComPtr<ID3D12CommandAllocator> allocator;
  // Reusable native decode command list for a decode-session child.
  ComPtr<ID3D12VideoDecodeCommandList> decode;
  // Reusable native encode command list for an encode-session child.
  ComPtr<ID3D12VideoEncodeCommandList2> encode;
  // Type-erased executable identity retained by decode or encode above.
  ID3D12CommandList* executable;
  // One feedback query slot owned by this recording.
  ComPtr<ID3D12QueryHeap> query;
  // Private raw encoder metadata provisioned exactly during cold creation.
  ComPtr<ID3D12Resource> raw_metadata;
  // Explicit caller-owned feedback bytes, borrowed until command destruction.
  amdf_media_buffer_range_t feedback;
  // Feedback resource identity retained by the borrowed core memory.
  ID3D12Resource* feedback_resource;
  // Direct host view established cold over the same feedback allocation.
  const uint8_t* feedback_pointer;
  // Encode output range retained for post-retirement extent validation.
  amdf_media_buffer_range_t compressed;
  // Maximum complete slices provisioned in trailing CPU recording storage.
  uint32_t maximum_slice_count;
  // Maximum reference bindings admitted by this recording.
  uint32_t maximum_reference_count;
  // Maximum application-authored alias transitions admitted by preparation.
  uint32_t maximum_activation_count;
  // Slice controls in the owner's explicitly sized trailing allocation.
  void* slices;
  // Native transition storage in the same fixed trailing allocation.
  D3D12_RESOURCE_BARRIER* barriers;
  // Nonzero native prefix owned until checked retirement; zero before
  // acceptance.
  uint64_t submission;
  // Explicit operation represented by the current closed recording.
  amdf_media_command_operation_t operation;
  // True only after a complete native recording has closed successfully.
  bool prepared;
  // Whether the accepted prefix includes native decode execution.
  bool executed;
};

amdf_status_t AMDF_CALL
CommandCreate(amdf_media_session_t* session,
              const amdf_media_command_create_info_t* create_info,
              amdf_media_command_t** out_command);
amdf_status_t AMDF_CALL
CommandPrepare(amdf_media_command_t* command,
               const amdf_media_command_prepare_info_t* prepare_info);
amdf_status_t AMDF_CALL CommandQueryResult(
    amdf_media_command_t* command, amdf_media_command_result_info_t* out_info);
amdf_status_t AMDF_CALL CommandDestroy(amdf_media_command_t* command);

}  // namespace amdf::media::d3d12

#endif  // AMDF_SRC_MEDIA_D3D12_COMMAND_H_
