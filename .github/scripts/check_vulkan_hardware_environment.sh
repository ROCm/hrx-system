#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
  echo "Required Python interpreter python3 was not found." >&2
  exit 1
fi

python3 - <<'PY'
import ctypes
import os
import sys

VK_SUCCESS = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_MAX_PHYSICAL_DEVICE_NAME_SIZE = 256
VK_UUID_SIZE = 16

VK_PHYSICAL_DEVICE_TYPE_OTHER = 0
VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1
VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2
VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3
VK_PHYSICAL_DEVICE_TYPE_CPU = 4

DEVICE_TYPE_NAMES = {
    VK_PHYSICAL_DEVICE_TYPE_OTHER: "OTHER",
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: "INTEGRATED_GPU",
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: "DISCRETE_GPU",
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: "VIRTUAL_GPU",
    VK_PHYSICAL_DEVICE_TYPE_CPU: "CPU",
}


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.c_void_p),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


class VkPhysicalDeviceProperties(ctypes.Structure):
    _fields_ = [
        ("apiVersion", ctypes.c_uint32),
        ("driverVersion", ctypes.c_uint32),
        ("vendorID", ctypes.c_uint32),
        ("deviceID", ctypes.c_uint32),
        ("deviceType", ctypes.c_uint32),
        ("deviceName", ctypes.c_char * VK_MAX_PHYSICAL_DEVICE_NAME_SIZE),
        ("pipelineCacheUUID", ctypes.c_uint8 * VK_UUID_SIZE),
        ("_tail", ctypes.c_uint8 * 4096),
    ]


def check_environment_variable(name):
    value = os.environ.get(name)
    if not value:
        return
    lowered = value.lower()
    if "lavapipe" in lowered or "lvp" in lowered or "llvmpipe" in lowered:
        raise SystemExit(
            f"{name} selects a software Vulkan ICD ({value}); "
            "the CI Vulkan lane requires real hardware"
        )
    print(f"{name}={value}")


def check_software_forcing_environment():
    if os.environ.get("LIBGL_ALWAYS_SOFTWARE") in ("1", "true", "TRUE", "yes", "YES"):
        raise SystemExit(
            "LIBGL_ALWAYS_SOFTWARE requests software rendering; "
            "the CI Vulkan lane requires real hardware"
        )
    check_environment_variable("VK_ICD_FILENAMES")
    check_environment_variable("VK_DRIVER_FILES")


def api_version_string(version):
    variant = version >> 29
    major = (version >> 22) & 0x7F
    minor = (version >> 12) & 0x3FF
    patch = version & 0xFFF
    if variant:
        return f"{variant}:{major}.{minor}.{patch}"
    return f"{major}.{minor}.{patch}"


def vk_result_name(result):
    return "VK_SUCCESS" if result == VK_SUCCESS else f"VkResult {result}"


def main():
    check_software_forcing_environment()

    try:
        vulkan = ctypes.CDLL("libvulkan.so.1")
    except OSError as error:
        raise SystemExit(
            f"failed to load libvulkan.so.1: {error}; "
            "the Vulkan hardware runner must provide the Vulkan loader"
        ) from error

    vulkan.vkCreateInstance.argtypes = (
        ctypes.POINTER(VkInstanceCreateInfo),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    vulkan.vkCreateInstance.restype = ctypes.c_int32
    vulkan.vkEnumeratePhysicalDevices.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_void_p),
    )
    vulkan.vkEnumeratePhysicalDevices.restype = ctypes.c_int32
    vulkan.vkGetPhysicalDeviceProperties.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(VkPhysicalDeviceProperties),
    )
    vulkan.vkGetPhysicalDeviceProperties.restype = None
    vulkan.vkDestroyInstance.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    vulkan.vkDestroyInstance.restype = None

    create_info = VkInstanceCreateInfo(
        sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pNext=None,
        flags=0,
        pApplicationInfo=None,
        enabledLayerCount=0,
        ppEnabledLayerNames=None,
        enabledExtensionCount=0,
        ppEnabledExtensionNames=None,
    )
    instance = ctypes.c_void_p()
    result = vulkan.vkCreateInstance(
        ctypes.byref(create_info), None, ctypes.byref(instance)
    )
    if result != VK_SUCCESS:
        raise SystemExit(f"vkCreateInstance failed with {vk_result_name(result)}")

    try:
        device_count = ctypes.c_uint32()
        result = vulkan.vkEnumeratePhysicalDevices(
            instance, ctypes.byref(device_count), None
        )
        if result != VK_SUCCESS:
            raise SystemExit(
                f"vkEnumeratePhysicalDevices failed with {vk_result_name(result)}"
            )
        if device_count.value == 0:
            raise SystemExit("Vulkan loader found no physical devices")

        devices = (ctypes.c_void_p * device_count.value)()
        result = vulkan.vkEnumeratePhysicalDevices(
            instance, ctypes.byref(device_count), devices
        )
        if result != VK_SUCCESS:
            raise SystemExit(
                f"vkEnumeratePhysicalDevices returned {vk_result_name(result)}"
            )

        first_device_is_hardware = False
        found_hardware = False
        print(f"Vulkan physical devices: {device_count.value}")
        for index in range(device_count.value):
            properties = VkPhysicalDeviceProperties()
            vulkan.vkGetPhysicalDeviceProperties(
                devices[index], ctypes.byref(properties)
            )
            device_name = bytes(properties.deviceName).split(b"\0", 1)[0].decode(
                "utf-8", errors="replace"
            )
            device_type = DEVICE_TYPE_NAMES.get(
                properties.deviceType, f"UNKNOWN({properties.deviceType})"
            )
            print(
                f"[{index}] {device_name} "
                f"type={device_type} "
                f"vendor=0x{properties.vendorID:04x} "
                f"device=0x{properties.deviceID:04x} "
                f"api={api_version_string(properties.apiVersion)}"
            )
            is_hardware = False
            if properties.deviceType in (
                VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
                VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
            ):
                lowered_name = device_name.lower()
                if not any(
                    software in lowered_name
                    for software in ("lavapipe", "llvmpipe", "software rasterizer")
                ):
                    is_hardware = True
                    found_hardware = True
            if index == 0:
                first_device_is_hardware = is_hardware

        if not found_hardware:
            raise SystemExit(
                "Vulkan enumerated no integrated or discrete GPU devices; "
                "the CI Vulkan lane requires real hardware"
            )
        if not first_device_is_hardware:
            raise SystemExit(
                "Vulkan physical device 0 is not real hardware; the default "
                "CI test device must not accidentally select a software device"
            )
    finally:
        if instance.value:
            vulkan.vkDestroyInstance(instance, None)


if __name__ == "__main__":
    sys.exit(main())
PY
