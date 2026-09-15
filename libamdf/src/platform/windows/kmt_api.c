// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/windows/kmt_api.h"

#include <stddef.h>
#include <string.h>

amdf_status_t amdf_kmt_make_status(NTSTATUS status) {
  return amdf_make_status(AMDF_STATUS_DOMAIN_NTSTATUS, (uint32_t)status);
}

static void amdf_kmt_resolve_procedures(amdf_kmt_api_t* api) {
  api->enumerate_adapters = (PFND3DKMT_ENUMADAPTERS3)GetProcAddress(
      api->module, "D3DKMTEnumAdapters3");
  api->open_adapter_from_luid = (PFND3DKMT_OPENADAPTERFROMLUID)GetProcAddress(
      api->module, "D3DKMTOpenAdapterFromLuid");
  api->query_adapter_info = (PFND3DKMT_QUERYADAPTERINFO)GetProcAddress(
      api->module, "D3DKMTQueryAdapterInfo");
  api->create_device =
      (PFND3DKMT_CREATEDEVICE)GetProcAddress(api->module, "D3DKMTCreateDevice");
  api->destroy_device = (PFND3DKMT_DESTROYDEVICE)GetProcAddress(
      api->module, "D3DKMTDestroyDevice");
  api->get_device_state = (PFND3DKMT_GETDEVICESTATE)GetProcAddress(
      api->module, "D3DKMTGetDeviceState");
  api->create_paging_queue = (PFND3DKMT_CREATEPAGINGQUEUE)GetProcAddress(
      api->module, "D3DKMTCreatePagingQueue");
  api->destroy_paging_queue = (PFND3DKMT_DESTROYPAGINGQUEUE)GetProcAddress(
      api->module, "D3DKMTDestroyPagingQueue");
  api->create_context_virtual = (PFND3DKMT_CREATECONTEXTVIRTUAL)GetProcAddress(
      api->module, "D3DKMTCreateContextVirtual");
  api->destroy_context = (PFND3DKMT_DESTROYCONTEXT)GetProcAddress(
      api->module, "D3DKMTDestroyContext");
  api->create_allocation = (PFND3DKMT_CREATEALLOCATION2)GetProcAddress(
      api->module, "D3DKMTCreateAllocation2");
  api->destroy_allocation = (PFND3DKMT_DESTROYALLOCATION2)GetProcAddress(
      api->module, "D3DKMTDestroyAllocation2");
  api->reserve_gpu_virtual_address =
      (PFND3DKMT_RESERVEGPUVIRTUALADDRESS)GetProcAddress(
          api->module, "D3DKMTReserveGpuVirtualAddress");
  api->free_gpu_virtual_address =
      (PFND3DKMT_FREEGPUVIRTUALADDRESS)GetProcAddress(
          api->module, "D3DKMTFreeGpuVirtualAddress");
  api->map_gpu_virtual_address = (PFND3DKMT_MAPGPUVIRTUALADDRESS)GetProcAddress(
      api->module, "D3DKMTMapGpuVirtualAddress");
  api->make_resident =
      (PFND3DKMT_MAKERESIDENT)GetProcAddress(api->module, "D3DKMTMakeResident");
  api->evict = (PFND3DKMT_EVICT)GetProcAddress(api->module, "D3DKMTEvict");
  api->lock = (PFND3DKMT_LOCK2)GetProcAddress(api->module, "D3DKMTLock2");
  api->unlock = (PFND3DKMT_UNLOCK2)GetProcAddress(api->module, "D3DKMTUnlock2");
  api->invalidate_cache = (PFND3DKMT_INVALIDATECACHE)GetProcAddress(
      api->module, "D3DKMTInvalidateCache");
  api->wait_from_cpu =
      (PFND3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU)GetProcAddress(
          api->module, "D3DKMTWaitForSynchronizationObjectFromCpu");
  api->create_hardware_queue = (PFND3DKMT_CREATEHWQUEUE)GetProcAddress(
      api->module, "D3DKMTCreateHwQueue");
  api->destroy_hardware_queue = (PFND3DKMT_DESTROYHWQUEUE)GetProcAddress(
      api->module, "D3DKMTDestroyHwQueue");
  api->submit_command_to_hardware_queue =
      api->win32u_module == NULL
          ? NULL
          : (PFND3DKMT_SUBMITCOMMANDTOHWQUEUE)GetProcAddress(
                api->win32u_module, "NtGdiDdDDISubmitCommandToHwQueue");
  if (api->submit_command_to_hardware_queue == NULL) {
    api->submit_command_to_hardware_queue =
        (PFND3DKMT_SUBMITCOMMANDTOHWQUEUE)GetProcAddress(
            api->module, "D3DKMTSubmitCommandToHwQueue");
  }
  api->close_adapter =
      (PFND3DKMT_CLOSEADAPTER)GetProcAddress(api->module, "D3DKMTCloseAdapter");
}

bool amdf_kmt_api_supports_endpoint_discovery(const amdf_kmt_api_t* api) {
  return api->enumerate_adapters != NULL &&
         api->open_adapter_from_luid != NULL &&
         api->query_adapter_info != NULL && api->close_adapter != NULL;
}

bool amdf_kmt_api_supports_paging_devices(const amdf_kmt_api_t* api) {
  return api->create_device != NULL && api->destroy_device != NULL &&
         api->get_device_state != NULL && api->create_paging_queue != NULL &&
         api->destroy_paging_queue != NULL;
}

bool amdf_kmt_api_supports_device_contexts(const amdf_kmt_api_t* api) {
  return amdf_kmt_api_supports_paging_devices(api) &&
         api->create_context_virtual != NULL && api->destroy_context != NULL;
}

bool amdf_kmt_api_supports_memory(const amdf_kmt_api_t* api) {
  return api->create_allocation != NULL && api->destroy_allocation != NULL &&
         api->map_gpu_virtual_address != NULL && api->make_resident != NULL &&
         api->wait_from_cpu != NULL;
}

bool amdf_kmt_api_supports_gpu_memory(const amdf_kmt_api_t* api) {
  return amdf_kmt_api_supports_memory(api) &&
         api->reserve_gpu_virtual_address != NULL &&
         api->free_gpu_virtual_address != NULL && api->evict != NULL &&
         api->invalidate_cache != NULL;
}

bool amdf_kmt_api_supports_gpu_kernel_execution(const amdf_kmt_api_t* api) {
  return amdf_kmt_api_supports_device_contexts(api) &&
         api->wait_from_cpu != NULL && api->create_hardware_queue != NULL &&
         api->destroy_hardware_queue != NULL &&
         api->submit_command_to_hardware_queue != NULL;
}

bool amdf_kmt_api_supports_xdna_kernel_execution(const amdf_kmt_api_t* api) {
  return amdf_kmt_api_supports_device_contexts(api) &&
         amdf_kmt_api_supports_memory(api) && api->lock != NULL &&
         api->unlock != NULL && api->invalidate_cache != NULL &&
         api->create_hardware_queue != NULL &&
         api->destroy_hardware_queue != NULL &&
         api->submit_command_to_hardware_queue != NULL;
}

bool amdf_kmt_status_is_success_or_pending(NTSTATUS status) {
  const uint32_t code = (uint32_t)status;
  return code == 0 || code == 0x00000103u;
}

amdf_status_t amdf_kmt_wait_for_paging(
    const amdf_kmt_api_t* api, D3DKMT_HANDLE device,
    D3DKMT_HANDLE paging_sync_object,
    const volatile uint64_t* current_paging_fence, uint64_t target_value) {
  if (target_value == 0 ||
      (current_paging_fence != NULL && *current_paging_fence >= target_value)) {
    return AMDF_STATUS_OK;
  }
  D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait = {0};
  wait.hDevice = device;
  wait.ObjectCount = 1;
  wait.ObjectHandleArray = &paging_sync_object;
  wait.FenceValueArray = &target_value;
  return amdf_kmt_make_status(api->wait_from_cpu(&wait));
}

amdf_status_t amdf_kmt_api_initialize(amdf_kmt_api_t* out_api) {
  amdf_kmt_api_t api = {0};
  api.module = LoadLibraryExW(L"gdi32.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (api.module == NULL) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  api.win32u_module =
      LoadLibraryExW(L"win32u.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
  amdf_kmt_resolve_procedures(&api);
  *out_api = api;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_kmt_api_deinitialize(amdf_kmt_api_t* api) {
  if (api->win32u_module != NULL && !FreeLibrary(api->win32u_module)) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  api->win32u_module = NULL;
  if (api->module != NULL && !FreeLibrary(api->module)) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  memset(api, 0, sizeof(*api));
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_kmt_query_adapter_info(const amdf_kmt_api_t* api,
                                          D3DKMT_HANDLE adapter,
                                          KMTQUERYADAPTERINFOTYPE type,
                                          void* data, uint32_t data_size) {
  D3DKMT_QUERYADAPTERINFO query = {0};
  query.hAdapter = adapter;
  query.Type = type;
  query.pPrivateDriverData = data;
  query.PrivateDriverDataSize = data_size;
  return amdf_kmt_make_status(api->query_adapter_info(&query));
}
