// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define CINTERFACE
#define CONST_VTABLE
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "libamdf/src/media/d3d12/queue_test_native.h"

#include <stddef.h>
#include <stdlib.h>

typedef struct native_objects_t {
  // Public controls and observations belonging to this test domain.
  amdf_media_queue_test_native_t state;
  // Device factory interface.
  ID3D12Device3 device;
  // Ordered native publication interface.
  ID3D12CommandQueue queue;
  // Independently observed native progress interface.
  ID3D12Fence fence;
  // Opaque prepared recording passed through ExecuteCommandLists.
  ID3D12VideoDecodeCommandList command;
} native_objects_t;

#define OWNER(object, member) \
  ((native_objects_t*)((char*)(object) - offsetof(native_objects_t, member)))

static ULONG STDMETHODCALLTYPE release_device(ID3D12Device3* self) {
  return --OWNER(self, device)->state.device_references;
}
static ULONG STDMETHODCALLTYPE release_queue(ID3D12CommandQueue* self) {
  return --OWNER(self, queue)->state.queue_references;
}
static ULONG STDMETHODCALLTYPE release_fence(ID3D12Fence* self) {
  return --OWNER(self, fence)->state.fence_references;
}
static ULONG STDMETHODCALLTYPE
release_command(ID3D12VideoDecodeCommandList* self) {
  return --OWNER(self, command)->state.command_references;
}

static HRESULT STDMETHODCALLTYPE
create_queue(ID3D12Device3* self, const D3D12_COMMAND_QUEUE_DESC* description,
             REFIID iid, void** out_queue) {
  (void)description;
  (void)iid;
  native_objects_t* objects = OWNER(self, device);
  ++objects->state.queue_references;
  *out_queue = &objects->queue;
  return S_OK;
}

static HRESULT STDMETHODCALLTYPE create_fence(ID3D12Device3* self,
                                              UINT64 initial,
                                              D3D12_FENCE_FLAGS flags,
                                              REFIID iid, void** out_fence) {
  (void)initial;
  (void)flags;
  (void)iid;
  native_objects_t* objects = OWNER(self, device);
  if (FAILED(objects->state.fence_create_status)) {
    return objects->state.fence_create_status;
  }
  ++objects->state.fence_references;
  *out_fence = &objects->fence;
  return S_OK;
}

static HRESULT STDMETHODCALLTYPE open_shared(ID3D12Device3* self, HANDLE handle,
                                             REFIID iid, void** out_object) {
  (void)handle;
  (void)iid;
  native_objects_t* objects = OWNER(self, device);
  ++objects->state.fence_references;
  *out_object = &objects->fence;
  return S_OK;
}

static HRESULT STDMETHODCALLTYPE removed_reason(ID3D12Device3* self) {
  amdf_media_queue_test_native_t* state = &OWNER(self, device)->state;
  ++state->removal_query_count;
  return state->removed_status;
}

static void STDMETHODCALLTYPE execute_commands(
    ID3D12CommandQueue* self, UINT count, ID3D12CommandList* const* commands) {
  (void)commands;
  OWNER(self, queue)->state.execute_count += count;
}

static HRESULT STDMETHODCALLTYPE signal_fence(ID3D12CommandQueue* self,
                                              ID3D12Fence* fence,
                                              UINT64 value) {
  (void)fence;
  amdf_media_queue_test_native_t* state = &OWNER(self, queue)->state;
  ++state->signal_count;
  state->signaled = value;
  return state->signal_status;
}

static HRESULT STDMETHODCALLTYPE wait_fence(ID3D12CommandQueue* self,
                                            ID3D12Fence* fence, UINT64 value) {
  (void)fence;
  (void)value;
  amdf_media_queue_test_native_t* state = &OWNER(self, queue)->state;
  ++state->wait_count;
  return state->wait_count == state->reject_wait ? E_FAIL : S_OK;
}

static UINT64 STDMETHODCALLTYPE completed_value(ID3D12Fence* self) {
  amdf_media_queue_test_native_t* state = &OWNER(self, fence)->state;
  ++state->completion_query_count;
  return state->completed;
}

static const ID3D12Device3Vtbl device_vtable = {
    .Release = release_device,
    .CreateCommandQueue = create_queue,
    .OpenSharedHandle = open_shared,
    .CreateFence = create_fence,
    .GetDeviceRemovedReason = removed_reason,
};
static const ID3D12CommandQueueVtbl queue_vtable = {
    .Release = release_queue,
    .ExecuteCommandLists = execute_commands,
    .Signal = signal_fence,
    .Wait = wait_fence,
};
static const ID3D12FenceVtbl fence_vtable = {
    .Release = release_fence,
    .GetCompletedValue = completed_value,
};
static const ID3D12VideoDecodeCommandListVtbl command_vtable = {
    .Release = release_command,
};

amdf_media_queue_test_native_t* amdf_media_queue_test_native_create(void) {
  native_objects_t* objects = calloc(1, sizeof(*objects));
  if (objects == NULL) {
    return NULL;
  }
  objects->device.lpVtbl = &device_vtable;
  objects->queue.lpVtbl = &queue_vtable;
  objects->fence.lpVtbl = &fence_vtable;
  objects->command.lpVtbl = &command_vtable;
  objects->state.device = &objects->device;
  objects->state.command = &objects->command;
  objects->state.device_references = objects->state.command_references = 1;
  return &objects->state;
}

void amdf_media_queue_test_native_destroy(
    amdf_media_queue_test_native_t* state) {
  free(state);
}
