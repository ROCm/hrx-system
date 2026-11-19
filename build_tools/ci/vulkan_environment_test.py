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

from build_tools.ci import gpu_assignment, vulkan_environment


class VulkanEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_dir.cleanup)
        self.root = Path(self.temporary_dir.name)
        self.sysfs = self.root / "sys/class/drm"
        self.sysfs.mkdir(parents=True)
        self.dri = self.root / "dev/dri"
        self.dri.mkdir(parents=True)
        self.topology = self.root / "nodes"
        self.topology.mkdir()
        (self.topology.parent / "generation_id").write_text("1")
        self.enterContext(
            mock.patch.multiple(
                gpu_assignment,
                DRI=self.dri,
                DRM_CLASS=self.sysfs,
                KFD_NODES=self.topology,
            )
        )
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
        entry = self.sysfs / name
        entry.mkdir()
        (entry / "device").symlink_to(pci_path)
        node_path = self.dri / name
        # Real character devices exercise open/fstat without requiring mknod or
        # GPU access. Only the external sysfs and Vulkan properties are fixtures.
        node_path.symlink_to(node_source)
        number = node_path.stat().st_rdev
        (entry / "dev").write_text(f"{os.major(number)}:{os.minor(number)}")
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

    def test_denied_child_is_the_only_node_opened(self):
        sibling = self.add_device()
        child = self.add_device("renderD129", "0000:02:00.0", node_source="/dev/zero")
        os.environ.update(
            PARENT_GPU_DEVICES=f"{sibling.path},{child.path}",
            CHILD_GPU_DEVICES=str(child.path),
        )

        real_open = os.open

        def open_node(path, flags, *args, **kwargs):
            if Path(path).parent == self.dri:
                self.assertEqual(path, child.path, "opened an unallocated GPU")
                raise PermissionError("render node denied by device cgroup")
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(os, "open", side_effect=open_node):
            with self.assertRaisesRegex(PermissionError, "denied by device cgroup"):
                vulkan_environment.check_environment()

    def test_unassigned_job_fails_before_open_or_vulkaninfo(self):
        self.add_device()
        with (
            mock.patch.object(os, "open") as opened,
            mock.patch.object(subprocess, "run") as probe,
        ):
            with self.assertRaisesRegex(ValueError, "No runner GPU assignment"):
                vulkan_environment.check_environment()
            opened.assert_not_called()
            probe.assert_not_called()

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

        profile_output = self.root / "selected-device.json"
        with mock.patch.object(subprocess, "run", side_effect=run_vulkaninfo) as run:
            vulkan_environment.check_environment(profile_output)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            json.loads(profile_output.read_bytes()), profiles[discrete.dri_prime]
        )
        self.assertEqual(
            environment_file.read_text(),
            f"DRI_PRIME={discrete.dri_prime}\n"
            f"HAL_AMD_GPU_ASSIGNMENT={gpu_assignment.native_identity(discrete.device_number)}\n"
            f"AMDF_CTS_GPU_NATIVE_IDENTITY={gpu_assignment.native_identity(discrete.device_number)}\n",
        )

    def test_legacy_allocation_selects_integrated_sibling_on_same_host(self):
        self.add_device("renderD3", device_id=0x7550)
        integrated = self.add_device(
            "renderD5", "0000:02:00.0", node_source="/dev/zero"
        )
        for index, minor in enumerate((3, 5)):
            node = self.topology / str(index)
            node.mkdir()
            (node / "properties").write_text(
                f"simd_count 10\ndrm_render_minor {minor}\nunique_id 0\n"
            )
        os.environ["ROCR_VISIBLE_DEVICES"] = "1"
        environment_file = self.root / "github_env"
        os.environ["GITHUB_ENV"] = str(environment_file)
        profile = self.profile(integrated)
        profile["capabilities"]["device"]["properties"]["VkPhysicalDeviceProperties"][
            "deviceType"
        ] = "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU"

        def run_vulkaninfo(command, *, env, check):
            self.assertEqual(env["DRI_PRIME"], integrated.dri_prime)
            Path(command[3]).write_text(json.dumps(profile))

        with mock.patch.object(subprocess, "run", side_effect=run_vulkaninfo) as probe:
            vulkan_environment.check_environment()
            probe.assert_called_once()
        self.assertIn(
            f"DRI_PRIME={integrated.dri_prime}",
            environment_file.read_text().splitlines(),
        )

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
            vulkan_environment.check_environment()
        self.assertEqual(
            environment_file.read_text(),
            "OTHER=preserved\nDRI_PRIME=pci-0000_01_00_0!\n"
            f"HAL_AMD_GPU_ASSIGNMENT={gpu_assignment.native_identity(device.device_number)}\n"
            f"AMDF_CTS_GPU_NATIVE_IDENTITY={gpu_assignment.native_identity(device.device_number)}\n",
        )

    def test_native_failure_does_not_publish_selection(self):
        device = self.add_device()
        os.environ["PARENT_GPU_DEVICES"] = str(device.path)
        environment_file = self.root / "github_env"
        os.environ["GITHUB_ENV"] = str(environment_file)
        profile_output = self.root / "selected-device.json"
        with mock.patch.object(
            subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(1, "vulkaninfo"),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                vulkan_environment.check_environment(profile_output)
        self.assertFalse(environment_file.exists())
        self.assertFalse(profile_output.exists())


if __name__ == "__main__":
    unittest.main()
