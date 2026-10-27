# Device telemetry

The system-management controller supplies clock rates, temperatures, power,
activity, and related device statistics through native driver queries. These
observations describe physical regions and firmware sampling intervals. They
do not identify a dispatch, replace queue timestamps, or establish which
concurrent workload consumed the reported power.

Linux exposes individual `hwmon` attributes and versioned `gpu_metrics`
snapshots. Windows exposes PMLog sensors through native queries and a separate
shared-memory sampling service. Those transports have different ownership and
freshness contracts. A timestamp bracketing a host query describes the query's
duration; it does not date every sensor reading returned by that query.

## Linux sensor routes and physical meaning

The AMDGPU hwmon interface reports `freq1_input` and `freq2_input` in hertz,
power in microwatts, and temperature in millidegrees Celsius. The native
`read_sensor` implementation determines which firmware quantity reaches each
attribute. `smu_set_funcs` selects that producer by MP1 IP version, independently
of the shader compiler target. [Native attributes][hwmon]
[Producer selection][selection]

Several routes require more information than the attribute name:

| Native producer | Attribute or sensor | Actual source and consequence |
| --- | --- | --- |
| SMU14 APU, MP1 14.0.0, 14.0.1, 14.0.4, 14.0.5 | `freq1_input`, `freq2_input` | `GfxclkFrequency` and `MemclkFrequency` are filtered **target frequencies**. They do not establish measured operating rates. |
| The same SMU14 APU producer | `power1_average`, `power1_input` | Both reach `SocketPower`, the filtered APU+dGPU PPT/STAPM power quantity. It can span more than the APU package. |
| The same SMU14 APU producer | `EDGE_TEMP`, `HOTSPOT_TEMP` | These routes return GPU and SoC TSEN temperatures, respectively. The second route is not a GPU hotspot measurement merely because its sensor enum says hotspot. |
| Renoir, MP1 12.0.0 and 12.0.1 | `GFX_MCLK` / `freq2_input` | The source is `ClockFrequency[CLOCK_FCLK]`: a fabric clock, despite the memory-clock route's name. |
| SMU13 multi-die, MP1 13.0.6 and 13.0.12 | `GFX_SCLK` / `freq1_input` | The driver writes zero when its DPM capability is absent. The cited capability initialization requires firmware at least `0x00552f00` and `0x00561900`, respectively. This zero does not demonstrate an idle clock. |
| SMU13 multi-die | Hotspot temperature | `MaxSocketTemperature` has socket scope, distinct from the temperature of one graphics die. |

[SMU14 firmware meanings][smu14-fields] [SMU14 sensor mapping][smu14-sensors]
[Renoir mapping][renoir] [SMU13 firmware predicates][smu13-caps]
[SMU13 metric selection][smu13-metrics]

The driver publishes MP1 identity under
`ip_discovery/die/<die>/1/<instance>/{major,minor,revision}`; hardware ID 1 is
MP1. `fw_version/smc_fw_version` exposes the native firmware version in
hexadecimal. The SMU version check populates that field from the firmware's
`GetSmuVersion` response. A missing identity does not establish equivalence to
another producer. [IP identity][ip-identity] [Hardware IDs][hardware-ids]
[Firmware attribute][firmware-attribute] [Firmware response][firmware-response]

`power1_average` and `power1_input` can route to the same firmware quantity.
Their spelling alone supplies neither a filter constant nor independent
measurements. Likewise, package, board, GPU, CPU, NPU, and shared power-budget
readings can overlap. Adding them without their physical boundaries can count
the same consumption more than once.

## Linux combined snapshots

`gpu_metrics` begins with `metrics_table_header`: a 16-bit structure byte size,
an 8-bit format revision, and an 8-bit content revision. The format/content
pair selects a representation; the producer establishes field semantics and
availability. The sysfs callback requests a native snapshot, then copies at
most `PAGE_SIZE - 1` bytes. A declared structure larger than the returned
extent is a truncated record, not a partial successful decode.
[Header and layouts][layouts] [Sysfs read][snapshot-read]
[Bounded native copy][snapshot-copy]

### Fixed layouts

The fixed structures use native C layout. The following sizes follow the cited
definitions with 64-bit native alignment; they are not packed-field sums.
The header expressly discourages v1.0 and v2.0 because their field ordering is
not naturally aligned. The later layouts move fields and add arrays or nested
partition statistics, so one version's offsets cannot decode another version.
[Fixed structures][layouts]

| Format/content | Structure bytes |
| --- | --- |
| 1.0 / 1.1 / 1.2 / 1.3 | 80 / 96 / 104 / 120 |
| 1.4 / 1.5 | 288 / 360 |
| 1.6 / 1.7 / 1.8 | 1664 / 2208 / 3872 |
| 2.0 / 2.1 / 2.2 | 120 / 120 / 128 |
| 2.3 / 2.4 | 152 / 168 |
| 3.0 | 264 |

Widths, units, and unavailable values belong to individual fields and
producers. For example, v1.4 through v1.8 specify energy in exactly
`1 / 65536` joules. SMU14's v3.0 APU power fields use milliwatts, whereas
v1.4 socket-power fields use watts. A unit inferred from another layout can
therefore be wrong by three orders of magnitude. [Energy units][energy]
[APU fields][apu-layouts]

The native initializers commonly fill a table with `0xff` and overwrite
implemented fields. That convention does not make every zero a valid reading:
`aldebaran_get_gpu_metrics` explicitly writes zero power and energy on its
secondary die, where those measurements are unavailable. Its separate sensor
path returns an unsupported error for that power reading.
[Aldebaran snapshot][aldebaran] [Aldebaran power sensor][aldebaran-power]

### Tagged format 1.9

The header's `gpu_metrics_attr` contains a pointer, but it is not the sysfs wire
representation emitted by the cited SMU13 and SMU15 producers.
`DECLARE_SMU_METRICS_CLASS` constructs a packed object with this layout:

```text
offset 0:  metrics_table_header, 4 bytes
offset 4:  signed attribute count, 4 bytes
offset 8:  64-bit attribute tag
           inline scalar or array payload
           next 64-bit tag, immediately after that payload
           ...
```

There is no pointer or inter-record alignment padding in that serialization.
The producer writes native integers without byte swapping; the cited 64-bit
x86 representation is little-endian. Exported raw captures retain that byte
order alongside the source schema. The tag uses the low 32 bits as follows;
the cited encoder leaves the upper 32 bits zero. [Tag encoding][tags]
[Packed producer][packed]

| Tag bits | Meaning |
| --- | --- |
| 31:24 | Unit ID |
| 23:20 | Scalar type ID |
| 19:10 | Native attribute ID |
| 9:0 | Element count; one for a scalar |

Type IDs 0 through 7 select unsigned/signed 8-, 16-, 32-, and 64-bit integers,
in that order. Payload length is element count times scalar width. Unit IDs
0 through 10 select unspecified, MHz, Celsius, watts, ns, 10 ns, GT/s,
0.1 GT/s, GB/s, KB, and percentage. An unspecified unit retains no conversion
by itself, even when the attribute's name suggests energy or bandwidth.
[Type and unit IDs][tags]

The cited SMU13 producer emits 50 attributes in 1258 bytes; the SMU15 producer
emits 51 in 1176 bytes. Their field widths and array counts differ. A decoder
walks the encoded lengths rather than assuming one producer's offsets or
casting bytes to `gpu_metrics_attr`. Unknown attribute IDs can be skipped
when their type and extent are understood; an unknown scalar type does not
provide the width needed to find the next record. The complete count and
extent remain part of record validation. [SMU13 fields][smu13-tags]
[SMU15 fields][smu15-tags]

## Producer-dependent intervals and instance identity

Format 3.0 illustrates why representation and meaning are separate:

| Field family | SMU14 APU producer | SMU15 APU producer |
| --- | --- | --- |
| Clock rates | Copies firmware target-frequency fields. | Computes effective clock rates from accumulator differences and a sample-count difference. |
| Power | Supplies filtered domain powers; `SocketPower` and `ApuPower` have different physical scopes. | Computes interval averages; some output fields use different accumulator domains. |
| CPU core slots | Copies fixed firmware array indices. | Compacts cores with nonzero calculated frequency into consecutive output slots. A slot can refer to a different physical core on the next read. |
| Filter duration | Copies `FilterAlphaValue` in microseconds. | Does not populate that field with an interval width. |
| NPU fields | Supplies IPU/MPIPU clock, activity, power, and traffic fields. | Unwritten fields retain the initializer's unavailable representation. |

[SMU14 snapshot][smu14-snapshot] [SMU14 firmware definitions][smu14-fields]
[SMU15 accumulation and packing][smu15-snapshot]

The SMU15 path requires a nonzero accumulation-count difference before its
average calculation produces data. Its scalar hwmon cache has a different
initialization and publication path from the combined snapshot. A decoder
cannot treat an unpopulated cached average as a measured zero merely because
both interfaces eventually read the same controller.
[SMU15 scalar averages and snapshot][smu15-snapshot]

A native array slot establishes neither an OS CPU index nor a stable physical
core identity when the producer compacts active entries. The raw slot still
identifies the field within this one snapshot. Cross-snapshot per-core history
requires an identity mapping that this producer does not publish.

The driver's `system_clock_counter` is attached after collecting the table
using `ktime_get_boottime_ns()` on these producers. It is a driver read
timestamp in `CLOCK_BOOTTIME`, not a sensor integration endpoint and not
`CLOCK_MONOTONIC_RAW`. A separate `firmware_timestamp`, when present, retains
its own stated units and epoch. `smu_cmn_get_metrics_table` can reuse cached
data unless its caller requests a refresh. Query cadence therefore does not
establish sensor cadence or observation freshness. [Driver timestamps][smu14-snapshot]
[Native table cache][cache]

## Windows PMLog queries

`Wkmi::QueryPMLogData` accepts native adapter/device handles and returns sensor
values with per-sensor support. This query is distinct from starting a PMLog
shared-memory sampler. Its declaration contains no physical-adapter ordinal,
so a linked-adapter handle alone does not establish attribution to one of its
physical adapters. [WKMI declarations][wkmi]

The public ADL sensor definitions specify these useful quantities:

| Sensor ID | Native name | Unit |
| --- | --- | --- |
| 1 / 2 | `PMLOG_CLK_GFXCLK` / `PMLOG_CLK_MEMCLK` | MHz |
| 8 / 27 | `PMLOG_TEMPERATURE_EDGE` / `PMLOG_TEMPERATURE_HOTSPOT` | Celsius |
| 19 / 20 | `PMLOG_INFO_ACTIVITY_GFX` / `PMLOG_INFO_ACTIVITY_MEM` | Percentage |
| 23 | `PMLOG_ASIC_POWER` | Watts |

The supported flag is separate from the integer value. ADL's single-sensor
record declares that value as signed `int`; all-one bits can represent a
negative temperature and are not a universal unsupported sentinel. These
declarations do not supply an integration window or acquisition timestamp for
each value. [Sensor meanings][adl-sensors] [Signed record][adl-record]

## Sampling sequence and interpretation

1. Associate the native sensor source with the selected physical device and
   obtain the producer identity needed to interpret it.
2. Read the source's complete representation into host-owned storage while
   retaining a host-time bracket. Separate hwmon reads retain separate
   brackets; proximity does not turn them into one snapshot.
3. Validate the returned schema and extent, then interpret field availability,
   signedness, scale, region, and aggregation from that producer.
4. Preserve native timestamps independently of the host bracket. A source
   without acquisition-time information retains unknown age.
5. Store the values together with their interpretation and device identity.
   Repeat at the application's chosen cadence without inferring dispatch
   attribution from a device-wide sensor.

These synchronous native queries finish writing the supplied host result
before returning; their copied bytes have no continuing GPU writer. They can
block in driver/firmware access and can wake a runtime-suspended device. They
do not provide queue completion or retire workload-owned storage. The separate
[clock discussion](../observability.md#clock-domains-and-conversion) covers
timestamp rates, host correlation, and execution-point semantics.

[hwmon]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/pm/amdgpu_pm.c#L3376-L3565
[selection]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/pm/swsmu/amdgpu_smu.c#L769-L862
[smu14-fields]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/pm/swsmu/inc/pmfw_if/smu14_driver_if_v14_0_0.h#L180-L248
[smu14-sensors]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/pm/swsmu/smu14/smu_v14_0_0_ppt.c#L292-L402
[renoir]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/pm/swsmu/smu12/renoir_ppt.c#L1176-L1255
[smu13-caps]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/pm/swsmu/smu13/smu_v13_0_6_ppt.c#L308-L383
[smu13-metrics]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/pm/swsmu/smu13/smu_v13_0_6_ppt.c#L1251-L1330
[ip-identity]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/amdgpu/amdgpu_discovery.c
[hardware-ids]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/include/soc15_hw_ip.h#L25
[firmware-attribute]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/amdgpu/amdgpu_ucode.c#L774-L855
[firmware-response]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/pm/swsmu/smu_cmn.c#L1095-L1132
[layouts]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/include/kgd_pp_interface.h#L552-L1807
[snapshot-read]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/pm/amdgpu_pm.c#L1777-L1800
[snapshot-copy]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/pm/amdgpu_dpm.c#L1444-L1466
[energy]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/include/kgd_pp_interface.h#L967-L1020
[apu-layouts]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/include/kgd_pp_interface.h#L1424-L1807
[aldebaran]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/pm/swsmu/smu13/aldebaran_ppt.c#L1640-L1710
[aldebaran-power]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/pm/swsmu/smu13/aldebaran_ppt.c#L680-L688
[tags]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/include/kgd_pp_interface.h#L666-L715
[packed]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/pm/swsmu/smu_cmn.h#L270-L325
[smu13-tags]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/pm/swsmu/smu13/smu_v13_0_6_ppt.h
[smu15-tags]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/pm/swsmu/smu15/smu_v15_0_8_ppt.h
[smu14-snapshot]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/pm/swsmu/smu14/smu_v14_0_0_ppt.c#L598-L674
[smu15-snapshot]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/pm/swsmu/smu15/smu_v15_0_0_ppt.c#L427-L793
[cache]: https://github.com/torvalds/linux/blob/f307e9d20cc7bd6dd54a3fd8c2af767dd7ef2931/drivers/gpu/drm/amd/pm/swsmu/smu_cmn.c#L1223-L1257
[wkmi]: https://github.com/ROCm/rocm-systems/blob/6b7847ef03b76c004f57b8cf40cdef9621ee6f53/shared/amdgpu-windows-interop/wkmi/wkmi.h
[adl-sensors]: https://github.com/GPUOpen-LibrariesAndSDKs/display-library/blob/84c881cb2136f04287eb3af12de00a9e2867aa9e/include/adl_defines.h
[adl-record]: https://gpuopen-librariesandsdks.github.io/adl/structADLSingleSensorData.html
