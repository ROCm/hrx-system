# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0

option(HRX_ENABLE_IPO
  "Enable interprocedural optimization for the whole HRX build." OFF)

function(hrx_configure_ipo)
  if(NOT HRX_ENABLE_IPO)
    return()
  endif()
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _hrx_ipo_supported OUTPUT _hrx_ipo_error)
  if(_hrx_ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE PARENT_SCOPE)
    message(STATUS "HRX: IPO/LTO enabled globally")
  else()
    message(WARNING "HRX: IPO/LTO not supported: ${_hrx_ipo_error}")
  endif()
endfunction()
