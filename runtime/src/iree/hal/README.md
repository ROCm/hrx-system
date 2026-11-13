# IREE Hardware Abstraction Layer (HAL)

The IREE HAL expresses a low-level abstraction over modern compute APIs like
Vulkan (CPUs count too!). Each implementation of the HAL interface can:

* Enumerate and query devices and their capabilities
* Define executable code that runs on the device
* Allocate unified or discrete memory and provide cache control
* Organize work into sequences for deferred submission
* Provide explicit synchronization primitives for ordering submissions

The [public API](api.h) defines the caller contracts. The
[runtime component map](../../../README.md) places the HAL alongside the VM,
CPU task system, and asynchronous I/O. The
[native AMD guide](driver/amd/README.md) describes the shared GPU/NPU driver,
and [build configuration](../../../../BUILDING.md#runtime-libraries-and-drivers)
lists the available driver selections.

## Testing

See the [cts/ folder](./cts/) for the HAL Conformance Test Suite.
