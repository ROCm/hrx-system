// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/windows/host_locality.h"

// The property declarations become definitions after initguid.h.
// clang-format off
#include <initguid.h>
#include <devpkey.h>
// clang-format on

#include <limits.h>
#include <string.h>
#include <wchar.h>

#include "libamdf/src/allocator.h"

amdf_status_t amdf_windows_host_locality_api_initialize(
    amdf_windows_host_locality_api_t* out_api) {
  amdf_windows_host_locality_api_t api = {0};
  api.module =
      LoadLibraryExW(L"cfgmgr32.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (api.module == NULL) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  api.locate_device =
      (CONFIGRET(WINAPI*)(PDEVINST, DEVINSTID_W, ULONG))GetProcAddress(
          api.module, "CM_Locate_DevNodeW");
  api.get_property = (CONFIGRET(WINAPI*)(
      DEVINST, const DEVPROPKEY*, DEVPROPTYPE*, PBYTE, PULONG,
      ULONG))GetProcAddress(api.module, "CM_Get_DevNode_PropertyW");
  api.map_error = (DWORD(WINAPI*)(CONFIGRET, DWORD))GetProcAddress(
      api.module, "CM_MapCrToWin32Err");
  api.get_node_masks =
      (BOOL(WINAPI*)(USHORT, PGROUP_AFFINITY, USHORT, PUSHORT))GetProcAddress(
          GetModuleHandleW(L"kernel32.dll"), "GetNumaNodeProcessorMask2");
  *out_api = api;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_windows_host_locality_api_deinitialize(
    amdf_windows_host_locality_api_t* api) {
  if (api->module != NULL && !FreeLibrary(api->module)) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, GetLastError());
  }
  memset(api, 0, sizeof(*api));
  return AMDF_STATUS_OK;
}

// KMT returns the physical adapter's hardware key. Decode the returned key,
// then let Config Manager resolve its device identity; no registry values or
// machine-wide device enumeration participate in this query.
static amdf_status_t amdf_windows_pnp_device_instance(WCHAR* key,
                                                      WCHAR** out_instance) {
  static const WCHAR prefix[] = L"\\REGISTRY\\MACHINE\\SYSTEM\\";
  static const WCHAR enumerator[] = L"\\Enum\\";
  static const WCHAR suffix[] = L"\\Device Parameters";
  const size_t prefix_length = sizeof(prefix) / sizeof(WCHAR) - 1;
  const size_t enumerator_length = sizeof(enumerator) / sizeof(WCHAR) - 1;
  const size_t suffix_length = sizeof(suffix) / sizeof(WCHAR) - 1;
  if (_wcsnicmp(key, prefix, prefix_length) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  WCHAR* path = wcschr(key + prefix_length, L'\\');
  if (path == NULL || _wcsnicmp(path, enumerator, enumerator_length) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  path += enumerator_length;
  const size_t length = wcslen(path);
  if (length <= suffix_length ||
      _wcsicmp(path + length - suffix_length, suffix) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  path[length - suffix_length] = L'\0';
  *out_instance = path;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_windows_device_host_numa_node(
    const amdf_windows_host_locality_api_t* api, WCHAR* instance,
    uint32_t* out_numa_node) {
  DEVINST device;
  CONFIGRET result =
      api->locate_device(&device, instance, CM_LOCATE_DEVNODE_NORMAL);
  if (result != CR_SUCCESS) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32,
                            api->map_error(result, ERROR_GEN_FAILURE));
  }
  ULONG length = sizeof(uint32_t);
  uint32_t node = UINT32_MAX;
  DEVPROPTYPE type = 0;
  result = api->get_property(device, &DEVPKEY_Device_Numa_Node, &type,
                             (PBYTE)&node, &length, 0);
  if (result == CR_NO_SUCH_VALUE) {
    *out_numa_node = UINT32_MAX;
    return AMDF_STATUS_OK;
  }
  if (result != CR_SUCCESS) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32,
                            api->map_error(result, ERROR_GEN_FAILURE));
  }
  if (type != DEVPROP_TYPE_UINT32 || length != sizeof(node) ||
      (node != UINT32_MAX && node > USHRT_MAX)) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, ERROR_INVALID_DATA);
  }
  if (node == UINT32_MAX) {
    *out_numa_node = UINT32_MAX;
    return AMDF_STATUS_OK;
  }
  // The required count includes exactly the groups containing active CPUs.
  // A zero-capacity query therefore establishes eligibility without allocating
  // masks or mistaking a processor-group number for the NUMA identity.
  USHORT count = 0;
  if (!api->get_node_masks((USHORT)node, NULL, 0, &count)) {
    const DWORD error = GetLastError();
    if (error != ERROR_INSUFFICIENT_BUFFER) {
      return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, error);
    }
  }
  *out_numa_node = count != 0 ? node : UINT32_MAX;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_windows_query_host_locality(
    const amdf_windows_host_locality_api_t* api, const amdf_kmt_api_t* kmt,
    D3DKMT_HANDLE adapter, uint32_t physical_adapter_index,
    uint32_t* out_numa_node, amdf_allocator_t host_allocator) {
  if (api->locate_device == NULL || api->get_property == NULL ||
      api->map_error == NULL || api->get_node_masks == NULL) {
    *out_numa_node = UINT32_MAX;
    return AMDF_STATUS_OK;
  }
  UINT capacity = 0;
  D3DKMT_QUERY_PHYSICAL_ADAPTER_PNP_KEY query = {
      .PhysicalAdapterIndex = physical_adapter_index,
      .PnPKeyType = D3DKMT_PNP_KEY_HARDWARE,
      .pCchDest = &capacity,
  };
  amdf_status_t status = amdf_kmt_query_adapter_info(
      kmt, adapter, KMTQAITYPE_PHYSICALADAPTERPNPKEY, &query, sizeof(query));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (capacity == 0) {
    return amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, ERROR_INVALID_DATA);
  }
  WCHAR* key = NULL;
  status = amdf_calloc_array(host_allocator, capacity, sizeof(*key),
                             amdf_alignof(WCHAR), (void**)&key);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const UINT allocated_capacity = capacity;
  query.pDest = key;
  status = amdf_kmt_query_adapter_info(
      kmt, adapter, KMTQAITYPE_PHYSICALADAPTERPNPKEY, &query, sizeof(query));
  if (amdf_status_is_ok(status) &&
      (capacity > allocated_capacity ||
       wmemchr(key, L'\0', allocated_capacity) == NULL)) {
    status = amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, ERROR_INVALID_DATA);
  }
  WCHAR* instance = NULL;
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_pnp_device_instance(key, &instance);
  }
  uint32_t node = UINT32_MAX;
  if (amdf_status_is_ok(status)) {
    status = amdf_windows_device_host_numa_node(api, instance, &node);
  }
  amdf_free(host_allocator, key);
  if (amdf_status_is_ok(status)) {
    *out_numa_node = node;
  }
  return status;
}
