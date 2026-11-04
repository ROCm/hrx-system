// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// name, argument type, published /dev/dxg operation number
AMDF_LINUX_DXG_KMT_OPERATION(open_adapter_from_luid, D3DKMT_OPENADAPTERFROMLUID,
                             0x01)
AMDF_LINUX_DXG_KMT_OPERATION(create_device, D3DKMT_CREATEDEVICE, 0x02)
AMDF_LINUX_DXG_KMT_OPERATION(create_context_virtual,
                             D3DKMT_CREATECONTEXTVIRTUAL, 0x04)
AMDF_LINUX_DXG_KMT_OPERATION(destroy_context, D3DKMT_DESTROYCONTEXT, 0x05)
AMDF_LINUX_DXG_KMT_OPERATION(create_allocation, D3DKMT_CREATEALLOCATION, 0x06)
AMDF_LINUX_DXG_KMT_OPERATION(create_paging_queue, D3DKMT_CREATEPAGINGQUEUE,
                             0x07)
AMDF_LINUX_DXG_KMT_OPERATION(reserve_gpu_virtual_address,
                             D3DDDI_RESERVEGPUVIRTUALADDRESS, 0x08)
AMDF_LINUX_DXG_KMT_OPERATION(query_adapter_info, D3DKMT_QUERYADAPTERINFO, 0x09)
AMDF_LINUX_DXG_KMT_OPERATION(query_video_memory_info,
                             D3DKMT_QUERYVIDEOMEMORYINFO, 0x0A)
AMDF_LINUX_DXG_KMT_OPERATION(make_resident, D3DDDI_MAKERESIDENT, 0x0B)
AMDF_LINUX_DXG_KMT_OPERATION(map_gpu_virtual_address,
                             D3DDDI_MAPGPUVIRTUALADDRESS, 0x0C)
AMDF_LINUX_DXG_KMT_OPERATION(escape, D3DKMT_ESCAPE, 0x0D)
AMDF_LINUX_DXG_KMT_OPERATION(get_device_state, D3DKMT_GETDEVICESTATE, 0x0E)
AMDF_LINUX_DXG_KMT_OPERATION(destroy_allocation, D3DKMT_DESTROYALLOCATION2,
                             0x13)
AMDF_LINUX_DXG_KMT_OPERATION(enumerate_adapters, D3DKMT_ENUMADAPTERS2, 0x14)
AMDF_LINUX_DXG_KMT_OPERATION(close_adapter, D3DKMT_CLOSEADAPTER, 0x15)
AMDF_LINUX_DXG_KMT_OPERATION(create_hardware_queue, D3DKMT_CREATEHWQUEUE, 0x18)
AMDF_LINUX_DXG_KMT_OPERATION(destroy_device, D3DKMT_DESTROYDEVICE, 0x19)
AMDF_LINUX_DXG_KMT_OPERATION(destroy_hardware_queue, D3DKMT_DESTROYHWQUEUE,
                             0x1B)
AMDF_LINUX_DXG_KMT_OPERATION(destroy_paging_queue, D3DDDI_DESTROYPAGINGQUEUE,
                             0x1C)
AMDF_LINUX_DXG_KMT_OPERATION(free_gpu_virtual_address,
                             D3DKMT_FREEGPUVIRTUALADDRESS, 0x20)
AMDF_LINUX_DXG_KMT_OPERATION(invalidate_cache, D3DKMT_INVALIDATECACHE, 0x24)
AMDF_LINUX_DXG_KMT_OPERATION(lock, D3DKMT_LOCK2, 0x25)
AMDF_LINUX_DXG_KMT_OPERATION(submit_command_to_hardware_queue,
                             D3DKMT_SUBMITCOMMANDTOHWQUEUE, 0x34)
AMDF_LINUX_DXG_KMT_OPERATION(wait_on_hardware_queue,
                             D3DKMT_SUBMITWAITFORSYNCOBJECTSTOHWQUEUE, 0x36)
AMDF_LINUX_DXG_KMT_OPERATION(unlock, D3DKMT_UNLOCK2, 0x37)
AMDF_LINUX_DXG_KMT_OPERATION(update_gpu_virtual_address,
                             D3DKMT_UPDATEGPUVIRTUALADDRESS, 0x39)
AMDF_LINUX_DXG_KMT_OPERATION(wait_from_cpu,
                             D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU, 0x3A)
