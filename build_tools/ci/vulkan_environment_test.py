# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.ci import vulkan_environment


class VulkanEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_dir.cleanup)
        self.root = Path(self.temporary_dir.name)
        self.sysfs = self.root / "sys/class/drm"
        self.sysfs.mkdir(parents=True)
        self.dri = self.root / "dev/dri"
        self.dri.mkdir(parents=True)
        self.driver = self.root / "sys/bus/pci/drivers/amdgpu"
        self.driver.mkdir(parents=True)
        self.output = io.StringIO()
        redirect = contextlib.redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.environment = mock.patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def add_device(
        self,
        name="renderD128",
        address="0000:01:00.0",
        device_id=0x748F,
        vendor_id=0x1002,
        node_source="/dev/null",
    ) -> vulkan_environment.RenderDevice:
        pci_path = self.root / "sys/bus/pci/devices" / address
        pci_path.mkdir(parents=True)
        (pci_path / "vendor").write_text(f"0x{vendor_id:04x}\n")
        (pci_path / "device").write_text(f"0x{device_id:04x}\n")
        (pci_path / "driver").symlink_to(self.driver)
        entry = self.sysfs / name
        entry.mkdir()
        (entry / "device").symlink_to(pci_path)
        node_path = self.dri / name
        # Real character devices exercise open/fstat without requiring mknod or
        # GPU access. Only the external sysfs and Vulkan properties are fixtures.
        node_path.symlink_to(node_source)
        return vulkan_environment.RenderDevice(
            node_path, address, device_id, Path(node_source).stat().st_rdev
        )

    def profile(self, device: vulkan_environment.RenderDevice) -> dict:
        return {
            "capabilities": {
                "device": {
                    "properties": {
                        "VkPhysicalDeviceProperties": {
                            "deviceName": "AMD test device",
                            "deviceType": "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
                            "vendorID": 0x1002,
                            "deviceID": device.device_id,
                            "apiVersion": (1 << 22) | (3 << 12),
                        },
                        "VkPhysicalDeviceDrmPropertiesEXT": {
                            "hasRender": True,
                            "renderMajor": os.major(device.device_number),
                            "renderMinor": os.minor(device.device_number),
                        },
                    }
                }
            }
        }

    def test_discovers_devices_without_product_id_table(self):
        expected = self.add_device(device_id=0x748F)
        self.add_device("renderD129", "0000:02:00.0", vendor_id=0x8086)
        devices = vulkan_environment.discover_render_devices(self.sysfs, self.dri)
        self.assertEqual(devices, [expected])
        self.assertEqual(
            vulkan_environment.render_device_candidates(devices, None)[0].dri_prime,
            "pci-0000_01_00_0!",
        )
        self.assertIn("1002:748f", self.output.getvalue())
        self.assertIn("8086:748f", self.output.getvalue())

    def test_reports_unexposed_node_and_does_not_select_it(self):
        device = self.add_device()
        device.path.unlink()
        devices = vulkan_environment.discover_render_devices(self.sysfs, self.dri)
        self.assertEqual(devices, [])
        self.assertIn("unavailable", self.output.getvalue())
        with self.assertRaisesRegex(RuntimeError, "No accessible AMD"):
            vulkan_environment.render_device_candidates(devices, None)

    def test_allocated_node_must_be_accessible_without_substitution(self):
        first = self.add_device()
        allocated = self.add_device("renderD129", "0000:02:00.0")
        real_open = os.open

        def open_node(path, flags):
            if path == allocated.path:
                raise PermissionError("render node denied by device cgroup")
            return real_open(path, flags)

        with mock.patch.object(os, "open", side_effect=open_node):
            devices = vulkan_environment.discover_render_devices(self.sysfs, self.dri)
        self.assertEqual(devices, [first])
        with self.assertRaisesRegex(RuntimeError, "refusing to test a different GPU"):
            vulkan_environment.render_device_candidates(devices, str(allocated.path))
        self.assertIn("denied by device cgroup", self.output.getvalue())

    def test_allocation_distinguishes_identical_cards(self):
        first = self.add_device()
        second = self.add_device("renderD129", "0000:02:00.0", node_source="/dev/zero")
        devices = vulkan_environment.discover_render_devices(self.sysfs, self.dri)
        self.assertEqual(
            vulkan_environment.render_device_candidates(devices, None), devices
        )
        self.assertEqual(
            vulkan_environment.render_device_candidates(devices, str(second.path)),
            [second],
        )
        self.assertEqual(
            vulkan_environment.render_device_candidates(
                devices, f"{second.path},{first.path}"
            ),
            [second, first],
        )

    def test_allocation_list_rejects_unavailable_or_empty_members(self):
        device = self.add_device()
        devices = vulkan_environment.discover_render_devices(self.sysfs, self.dri)
        for allocation in (
            f"{device.path},{self.dri / 'renderD129'}",
            f"{self.dri / 'renderD129'},{device.path}",
            f"{device.path},",
        ):
            with self.subTest(allocation=allocation):
                with self.assertRaisesRegex(RuntimeError, "inaccessible AMD render"):
                    vulkan_environment.render_device_candidates(devices, allocation)

    def test_selects_discrete_gpu_within_allocation_list(self):
        self.add_device()
        integrated = self.add_device("renderD129", "0000:02:00.0")
        discrete = self.add_device(
            "renderD130", "0000:03:00.0", node_source="/dev/zero"
        )
        profiles = {
            device.dri_prime: self.profile(device) for device in [integrated, discrete]
        }
        profiles[integrated.dri_prime]["capabilities"]["device"]["properties"][
            "VkPhysicalDeviceProperties"
        ]["deviceType"] = "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU"
        environment_file = self.root / "github_env"
        os.environ["GITHUB_ENV"] = str(environment_file)
        os.environ["PARENT_GPU_DEVICES"] = f"{integrated.path},{discrete.path}"

        def run_vulkaninfo(command, *, env, check):
            Path(command[3]).write_text(json.dumps(profiles[env["DRI_PRIME"]]))

        with mock.patch.object(subprocess, "run", side_effect=run_vulkaninfo) as run:
            vulkan_environment.check_environment(self.sysfs, self.dri)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            environment_file.read_text(), f"DRI_PRIME={discrete.dri_prime}\n"
        )

    def test_discrete_gpu_preferred_unless_integrated_gpu_is_allocated(self):
        integrated = self.add_device()
        discrete = self.add_device(
            "renderD129", "0000:02:00.0", node_source="/dev/zero"
        )
        profiles = {
            device.dri_prime: self.profile(device) for device in [integrated, discrete]
        }
        profiles[integrated.dri_prime]["capabilities"]["device"]["properties"][
            "VkPhysicalDeviceProperties"
        ]["deviceType"] = "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU"
        environment_file = self.root / "github_env"
        os.environ["GITHUB_ENV"] = str(environment_file)

        def run_vulkaninfo(command, *, env, check):
            Path(command[3]).write_text(json.dumps(profiles[env["DRI_PRIME"]]))

        with mock.patch.object(subprocess, "run", side_effect=run_vulkaninfo):
            vulkan_environment.check_environment(self.sysfs, self.dri)
        self.assertEqual(
            environment_file.read_text(), f"DRI_PRIME={discrete.dri_prime}\n"
        )

        environment_file.write_text("")
        os.environ["PARENT_GPU_DEVICES"] = str(integrated.path)
        with mock.patch.object(subprocess, "run", side_effect=run_vulkaninfo) as run:
            vulkan_environment.check_environment(self.sysfs, self.dri)
        self.assertEqual(
            environment_file.read_text(), f"DRI_PRIME={integrated.dri_prime}\n"
        )
        self.assertEqual(run.call_count, 1)

        environment_file.write_text("")
        del os.environ["PARENT_GPU_DEVICES"]
        profiles[discrete.dri_prime]["capabilities"]["device"]["properties"][
            "VkPhysicalDeviceProperties"
        ]["deviceType"] = "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU"
        with mock.patch.object(subprocess, "run", side_effect=run_vulkaninfo):
            with self.assertRaisesRegex(RuntimeError, "without a discrete GPU"):
                vulkan_environment.check_environment(self.sysfs, self.dri)
        self.assertEqual(environment_file.read_text(), "")

    def test_vulkan_must_report_hardware_at_the_exact_render_node(self):
        selected = self.add_device()
        profile = self.profile(selected)
        vulkan_environment.validate_vulkan_device(profile, selected)

        cases = [
            (
                "VkPhysicalDeviceProperties",
                "deviceType",
                "VK_PHYSICAL_DEVICE_TYPE_CPU",
                "not real GPU",
            ),
            (
                "VkPhysicalDeviceProperties",
                "deviceID",
                0x7551,
                "selected AMD PCI device",
            ),
            (
                "VkPhysicalDeviceProperties",
                "vendorID",
                0x8086,
                "selected AMD PCI device",
            ),
            (
                "VkPhysicalDeviceDrmPropertiesEXT",
                "hasRender",
                False,
                "does not report a DRM",
            ),
            ("VkPhysicalDeviceDrmPropertiesEXT", "renderMinor", 129, "expected"),
        ]
        for structure, key, value, error in cases:
            with self.subTest(structure=structure, key=key):
                profile = self.profile(selected)
                profile["capabilities"]["device"]["properties"][structure][key] = value
                with self.assertRaisesRegex(RuntimeError, error):
                    vulkan_environment.validate_vulkan_device(profile, selected)
        profile = self.profile(selected)
        del profile["capabilities"]["device"]["properties"][
            "VkPhysicalDeviceDrmPropertiesEXT"
        ]
        with self.assertRaisesRegex(RuntimeError, "does not report a DRM"):
            vulkan_environment.validate_vulkan_device(profile, selected)

    def test_publishes_selection_only_after_native_validation(self):
        device = self.add_device()
        environment_file = self.root / "github_env"
        environment_file.write_text("OTHER=preserved\n")
        os.environ["GITHUB_ENV"] = str(environment_file)
        os.environ["PARENT_GPU_DEVICES"] = str(device.path)
        profile = self.profile(device)

        def run_vulkaninfo(command, *, env, check):
            self.assertEqual(command[:3], ["vulkaninfo", "--json=0", "--output"])
            self.assertEqual(env["DRI_PRIME"], "pci-0000_01_00_0!")
            self.assertTrue(check)
            self.assertEqual(environment_file.read_text(), "OTHER=preserved\n")
            Path(command[3]).write_text(json.dumps(profile))

        with mock.patch.object(subprocess, "run", side_effect=run_vulkaninfo):
            vulkan_environment.check_environment(self.sysfs, self.dri)
        self.assertEqual(
            environment_file.read_text(),
            "OTHER=preserved\nDRI_PRIME=pci-0000_01_00_0!\n",
        )

    def test_native_failure_does_not_publish_selection(self):
        self.add_device()
        environment_file = self.root / "github_env"
        os.environ["GITHUB_ENV"] = str(environment_file)
        with mock.patch.object(
            subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, "vulkaninfo"),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                vulkan_environment.check_environment(self.sysfs, self.dri)
        self.assertFalse(environment_file.exists())


if __name__ == "__main__":
    unittest.main()
