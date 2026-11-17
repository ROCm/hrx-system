# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(iree_third_party_helpers)

function(iree_configure_miniaudio)
  if(TARGET iree::third_party::miniaudio)
    return()
  endif()
  # This narrow configuration is part of the runtime codec library's ABI. An
  # arbitrary installed miniaudio target may include device IO and workers.
  iree_dependency_require_pinned_source_allowed("miniaudio")
  iree_populate_locked_fetch_content(miniaudio _miniaudio_source_dir)
  add_library(iree_miniaudio_decoder STATIC
    "${_miniaudio_source_dir}/miniaudio.c")
  target_include_directories(iree_miniaudio_decoder SYSTEM PUBLIC
    "${_miniaudio_source_dir}")
  target_compile_definitions(iree_miniaudio_decoder PUBLIC
    MA_DR_MP3_FLOAT_OUTPUT
    MA_NO_DEVICE_IO MA_NO_ENCODING MA_NO_ENGINE MA_NO_GENERATION
    MA_NO_NODE_GRAPH MA_NO_RESOURCE_MANAGER MA_NO_RUNTIME_LINKING
    MA_NO_THREADING)
  foreach(_codec WAV MP3 FLAC)
    if(NOT IREE_CODEC_AUDIO_CODEC_${_codec})
      target_compile_definitions(iree_miniaudio_decoder PUBLIC MA_NO_${_codec})
    endif()
  endforeach()
  set_target_properties(iree_miniaudio_decoder PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  if(UNIX)
    target_link_libraries(iree_miniaudio_decoder PUBLIC m)
  endif()
  iree_add_alias_interface(iree::third_party::miniaudio iree_miniaudio_decoder)
endfunction()
