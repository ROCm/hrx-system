// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/d3d12/image.h"

#include <cstring>

#include "amdf/gpu.h"
#include "libamdf/src/endpoint.h"
#include "libamdf/src/media/d3d12/configuration.h"
#include "libamdf/src/media/d3d12/memory.h"
#include "libamdf/src/media/d3d12/object.h"
#include "libamdf/src/structure.h"

namespace amdf::media::d3d12 {
namespace {

amdf_status_t ReadPlane(Device* device, Image* image,
                        ID3D12DescriptorHeap* heap, uint32_t layer,
                        uint32_t plane, uint64_t heap_address,
                        amdf_media_gfx11_plane_layout_t* out_plane) {
  const auto& description = image->requirements.description;
  const bool p010 = description.format == AMDF_MEDIA_PIXEL_FORMAT_P010;
  D3D12_SHADER_RESOURCE_VIEW_DESC view = {
      .Format = p010
                    ? (plane ? DXGI_FORMAT_R16G16_UNORM : DXGI_FORMAT_R16_UNORM)
                    : (plane ? DXGI_FORMAT_R8G8_UNORM : DXGI_FORMAT_R8_UNORM),
      .ViewDimension = description.array_layer_count == 1
                           ? D3D12_SRV_DIMENSION_TEXTURE2D
                           : D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
      .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
  };
  if (description.array_layer_count == 1) {
    view.Texture2D.MipLevels = 1;
    view.Texture2D.PlaneSlice = plane;
  } else {
    view.Texture2DArray.MipLevels = 1;
    view.Texture2DArray.FirstArraySlice = layer;
    view.Texture2DArray.ArraySize = 1;
    view.Texture2DArray.PlaneSlice = plane;
  }
  const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
  device->native->CreateShaderResourceView(image->resource.Get(), &view,
                                           handle);
  uint32_t words[8] = {};
  SIZE_T copied = 0;
  if (!ReadProcessMemory(GetCurrentProcess(),
                         reinterpret_cast<const void*>(handle.ptr), words,
                         sizeof(words), &copied) ||
      copied != sizeof(words)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  // This is a qualified AMD descriptor transport, not a portable interpretation
  // of D3D12_CPU_DESCRIPTOR_HANDLE. Reject every representation requiring
  // compression metadata or an unqualified address/XOR interpretation.
  const uint64_t address =
      (uint64_t(words[0]) | (uint64_t(words[1] & 255) << 32)) << 8;
  const uint32_t width = ((words[1] >> 30) | ((words[2] & 4095) << 2)) + 1;
  const uint32_t height = ((words[2] >> 14) & 16383) + 1;
  const uint32_t mode = (words[3] >> 20) & 31;
  const uint32_t format = (words[1] >> 20) & 255;
  const uint32_t expected_format =
      p010 ? (plane ? 23u : 7u) : (plane ? 14u : 1u);
  if (width != description.width >> plane ||
      height != description.height >> plane || format != expected_format ||
      ((words[1] >> 16) & 15) != 0 ||
      (words[3] & 4095) != (plane ? 0x22cu : 0x204u) ||
      ((words[3] >> 12) & 255) != 0 || ((words[3] >> 25) & 7) != 0 ||
      (mode != 10 && mode != 26) ||
      words[3] >> 28 != (description.array_layer_count == 1 ? 9u : 13u) ||
      words[4] != layer || ((words[6] >> 21) & 1) != 0 ||
      (words[6] >> 24) != 0 || words[7] != 0 || (address & 65535) != 0 ||
      address < heap_address) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const uint64_t offset = address - heap_address;
  const uint32_t block_bits = 16 - plane - uint32_t(p010);
  const uint32_t block_width = 1u << ((block_bits + 1) / 2);
  const uint32_t block_height = 1u << (block_bits / 2);
  const uint64_t plane_length =
      uint64_t((width + block_width - 1) / block_width) *
      ((height + block_height - 1) / block_height) * 65536;
  if (offset < image->byte_offset ||
      offset - image->byte_offset > image->requirements.byte_length ||
      plane_length >
          image->requirements.byte_length - (offset - image->byte_offset)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_media_gfx11_plane_layout_t result = {};
  std::memcpy(result.descriptor, words, sizeof(words));
  result.byte_offset = offset;
  result.swizzle_mode = mode;
  *out_plane = result;
  return AMDF_STATUS_OK;
}

amdf_status_t QualifyLayouts(
    Device* device, Image* image,
    const amdf_media_image_create_info_t* create_info) {
  if (create_info->layout_access_count == 0) {
    return AMDF_STATUS_OK;
  }
  const auto& description = image->requirements.description;
  if ((description.roles & (AMDF_MEDIA_IMAGE_ROLE_DECODE_OUTPUT |
                            AMDF_MEDIA_IMAGE_ROLE_ENCODE_INPUT)) == 0 ||
      device->native->GetDescriptorHandleIncrementSize(
          D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) != 32) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  D3D12_DESCRIPTOR_HEAP_DESC descriptor_heap = {
      .Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
      .NumDescriptors = 1,
      .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
  };
  ComPtr<ID3D12DescriptorHeap> descriptors;
  amdf_status_t status = Hresult(device->native->CreateDescriptorHeap(
      &descriptor_heap, IID_PPV_ARGS(&descriptors)));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const uint64_t heap_address =
      NativeMemory(image->memory, image->media_access_ordinal)
          ->resource->GetGPUVirtualAddress();
  auto* next_layout = reinterpret_cast<amdf_media_gfx11_image_layout_t*>(
      image->layouts + image->memory->info.access_count);
  const amdf_media_gfx11_image_layout_t* established = nullptr;
  for (uint32_t index = 0;
       amdf_status_is_ok(status) && index < create_info->layout_access_count;
       ++index) {
    const uint32_t ordinal = create_info->layout_access_ordinals[index];
    const auto& access = image->memory->accesses[ordinal];
    const auto* target =
        amdf_endpoint_get_cached_engine_info(access.device->endpoint);
    if (!amdf_device_is_extension(access.device, AMDF_EXTENSION_GPU) ||
        target == nullptr ||
        target->type != AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO ||
        reinterpret_cast<const amdf_gpu_endpoint_info_t*>(target)
                ->gfx_ip.major != 11 ||
        std::memcmp(
            &amdf_endpoint_get_cached_info(access.device->endpoint)->id,
            &amdf_endpoint_get_cached_info(device->base.base.endpoint)->id,
            sizeof(amdf_endpoint_id_t)) != 0 ||
        (access.info.address_kinds &
         (UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU)) == 0) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    } else if (image->layouts[ordinal] != nullptr) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    } else {
      image->layouts[ordinal] = next_layout;
      for (uint32_t layer = 0;
           amdf_status_is_ok(status) && layer < description.array_layer_count;
           ++layer) {
        auto& layout = next_layout[layer];
        layout.type = AMDF_STRUCTURE_TYPE_MEDIA_GFX11_IMAGE_LAYOUT;
        layout.structure_size = sizeof(layout);
        layout.plane_count = 2;
        layout.array_layer = layer;
        for (uint32_t plane = 0; amdf_status_is_ok(status) && plane < 2;
             ++plane) {
          if (established != nullptr) {
            layout.planes[plane] = established[layer].planes[plane];
          } else {
            status = ReadPlane(device, image, descriptors.Get(), layer, plane,
                               heap_address, &layout.planes[plane]);
          }
          if (amdf_status_is_ok(status)) {
            auto& native = layout.planes[plane];
            const uint64_t address =
                access.addresses[AMDF_MEMORY_ADDRESS_GPU] + native.byte_offset;
            if ((address & 255) != 0 || address >> 48 != 0) {
              status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
            } else {
              native.descriptor[0] = static_cast<uint32_t>(address >> 8);
              native.descriptor[1] = (native.descriptor[1] & ~255u) |
                                     static_cast<uint32_t>(address >> 40);
            }
          }
        }
      }
      if (established == nullptr) {
        established = next_layout;
      }
      next_layout += description.array_layer_count;
    }
  }
  return status;
}

}  // namespace

amdf_status_t AMDF_CALL ImageCreate(
    amdf_device_t* base, const amdf_media_image_create_info_t* create_info,
    amdf_media_image_t** out_image) {
  auto* device = reinterpret_cast<Device*>(base);
  amdf_memory_t* memory = create_info->memory;
  if (create_info->media_access_ordinal >= memory->info.access_count ||
      memory->accesses[create_info->media_access_ordinal].device != base ||
      create_info->layout_access_count > memory->info.access_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (NativeMemory(memory, create_info->media_access_ordinal)->profile !=
      kSharedHeap) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  for (uint32_t index = 0; index < create_info->layout_access_count; ++index) {
    if (create_info->layout_access_ordinals[index] >=
        memory->info.access_count) {
      return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    }
  }
  amdf_media_image_requirements_t requirements = {};
  amdf_status_t status = ImageRequirements(
      device, create_info->requirements.description, &requirements);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (create_info->requirements.byte_length != requirements.byte_length ||
      create_info->requirements.byte_offset_alignment !=
          requirements.byte_offset_alignment) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (create_info->byte_offset > memory->info.byte_length ||
      requirements.byte_length >
          memory->info.byte_length - create_info->byte_offset ||
      create_info->byte_offset % requirements.byte_offset_alignment != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const size_t access_bytes = size_t(memory->info.access_count) * sizeof(void*);
  const uint64_t layout_count = uint64_t(create_info->layout_access_count) *
                                requirements.description.array_layer_count;
  if (layout_count > (SIZE_MAX - sizeof(Image) - access_bytes) /
                         sizeof(amdf_media_gfx11_image_layout_t)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  void* storage = nullptr;
  status = amdf_calloc_with_trailing(
      base->host_allocator, sizeof(Image),
      access_bytes + layout_count * sizeof(amdf_media_gfx11_image_layout_t),
      alignof(Image), &storage);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  auto* image = new (storage) Image{};
  image->base.device = &device->base;
  image->memory = memory;
  image->media_access_ordinal = create_info->media_access_ordinal;
  image->byte_offset = create_info->byte_offset;
  image->requirements = requirements;
  image->layouts =
      reinterpret_cast<amdf_media_gfx11_image_layout_t**>(image + 1);
  const auto native_description = ImageDescription(requirements.description);
  status = Hresult(device->native->CreatePlacedResource(
      NativeMemory(memory, image->media_access_ordinal)->heap.Get(),
      image->byte_offset, &native_description, D3D12_RESOURCE_STATE_COMMON,
      nullptr, IID_PPV_ARGS(&image->resource)));
  if (amdf_status_is_ok(status)) {
    status = QualifyLayouts(device, image, create_info);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_device_register_child(base);
  }
  if (amdf_status_is_ok(status)) {
    *out_image = &image->base;
  } else {
    Delete(base->host_allocator, image);
  }
  return status;
}

amdf_status_t AMDF_CALL ImageQueryInfo(amdf_media_image_t* base,
                                       uint32_t access_ordinal,
                                       const amdf_media_image_query_t* query,
                                       amdf_media_image_info_t* out_info) {
  const auto* image = reinterpret_cast<const Image*>(base);
  if (access_ordinal >= image->memory->info.access_count ||
      query->array_layer >= image->requirements.description.array_layer_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  if (access_ordinal != image->media_access_ordinal &&
      image->layouts[access_ordinal] == nullptr) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (out_info->next != nullptr) {
    const amdf_status_t status = amdf_structure_validate_output(
        out_info->next, AMDF_STRUCTURE_TYPE_MEDIA_GFX11_IMAGE_LAYOUT,
        sizeof(amdf_media_gfx11_image_layout_t));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (image->layouts[access_ordinal] == nullptr) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    auto* layout =
        static_cast<amdf_media_gfx11_image_layout_t*>(out_info->next);
    const uint32_t structure_size = layout->structure_size;
    *layout = image->layouts[access_ordinal][query->array_layer];
    layout->structure_size = structure_size;
  }
  out_info->memory = image->memory;
  out_info->requirements = image->requirements;
  out_info->byte_offset = image->byte_offset;
  return AMDF_STATUS_OK;
}

void AMDF_CALL ImageDestroy(amdf_media_image_t* base) {
  auto* image = reinterpret_cast<Image*>(base);
  amdf_device_t* device = &base->device->base;
  Delete(device->host_allocator, image);
  amdf_device_unregister_child(device);
}

amdf_status_t ValidateFrame(Device* device, const amdf_media_frame_t* frame,
                            amdf_media_image_roles_t role, Image** out_image) {
  const amdf_status_t status = amdf_structure_validate_input(
      frame, AMDF_STRUCTURE_TYPE_MEDIA_FRAME, sizeof(*frame));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (frame->storage_kind != AMDF_MEDIA_FRAME_STORAGE_KIND_NATIVE_IMAGE) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  auto* image = reinterpret_cast<Image*>(frame->storage.image.image);
  if (image == nullptr || image->base.device != &device->base ||
      frame->storage.image.memory != image->memory ||
      frame->storage.image.byte_offset != image->byte_offset ||
      frame->storage.image.array_layer >=
          image->requirements.description.array_layer_count ||
      frame->format != image->requirements.description.format ||
      frame->width != image->requirements.description.width ||
      frame->height != image->requirements.description.height ||
      (image->requirements.description.roles & role) != role) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  *out_image = image;
  return AMDF_STATUS_OK;
}

}  // namespace amdf::media::d3d12
