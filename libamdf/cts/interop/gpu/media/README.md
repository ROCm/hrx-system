# Native media and GPU ownership witness

The H264, HEVC, VP9, and AV1 corpora exercise media and compute through runtime-loaded
libamdf tables. H264's twelve Main pictures include seven B pictures, three
simultaneously live short-term references, and display order different from
decode order. A second case repeats the stream through 32 pictures using four
recording owners, two decoded output images, four reconstructed reference
images, and fixed input, feedback, and GPU result backing. Profile-specific
constrained-baseline and High IDRs close the advertised decode matrix with
[independent exact-pixel oracles](../../../media/README.md#h264-profile-fixture-provenance).
The application supplies every reference, placement, recording slot, and native
queue dependency. HEVC adds Main/NV12 and Main10/P010, four live references,
five reconstructed slots, and nonuniform scaling lists. Each picture keeps its
codec slot until its last reference use; reordering the reference list never
renumbers that slot. The Main10 envelope also exercises compatible eight-bit
syntax with NV12 storage.

VP9's [shared corpus](../../../media/README.md#vp9-fixture-provenance) adds
eight-bit profile 0 and ten-bit profile 2, adaptive entropy contexts, segmentation
enable/disable, aliased logical references, and partial bottom reconstruction
blocks. Both precisions run 24 pictures through four recording owners and nine
application-owned reference slots. Exact GPU sample fingerprints distinguish
correct pixels from successful native feedback. Admission keeps coded dimensions
separate from reference padding and rejects unsupported precision without
silently converting the output.

AV1 uses the [shared independent fixtures](../../../media/README.md#av1-fixture-provenance)
at both sample precisions. Its baseline and extended 520×200 streams exercise
64/128-pixel superblocks, four tiles, super-resolution, restoration, and film
grain. Separate streams activate segmentation, quantization matrices, and nonidentity
global motion, and
128×128 lossless streams retain every source sample. Nine reference slots keep
reconstruction free of display grain while four recordings and two output slots
are reused through 24 pictures. The GPU hashes every decoded sample and computes
the same crop/color output used by the other video corpora.

The GPU publishes compressed bytes, waits natively for decode, reads the shared
NV12 or P010 output, and produces a moving 64x64 RGB crop. A decode that reuses
an output slot waits for that slot's last GPU consumer. Host observation occurs once per
batch, after GPU consumption and a guard capture. Retired feedback is queried
before the recording is prepared again. Commands remain immutable until their
last accepted use retires.

`reference_main.packet` is a pre-parsed, padded application input fixture, not a
new parser format in libamdf. Its little-endian header contains `H264REFS`, then
fifteen DWORDs: picture count, width, height, frame-number width minus four,
direct-8x8 flag, deblocking-control flag, constrained-intra flag, level, signed
initial QP/QS/chroma offsets, POC width minus four, maximum references, and
default L0/L1 counts minus one. Each picture has nine DWORDs (generation,
compressed extent, frame number, POC, reconstructed slot, display index,
reference flag, intra flag, reference count), three slot/frame-number/POC
triples, and its complete Annex B slice padded to 128 bytes.

The source stream was encoded with x264 from a synthetic moving image sequence.
The fixture's SHA-256 is
`06ce0cddfdad5a2756591db095553ee414b42fe041ea78d4d75bada4eccc7988`;
the original Annex B stream's SHA-256 is
`f6fd70addd5468fe3a43dfc9237a3979aa0b0140bc05c91a86864789cc1a92e5`.
FFmpeg's native H264 decoder agrees bit for bit with x264's reconstruction.
The decode-order NV12 oracle has SHA-256
`ef7063d29930c58e5f2f7e582378f574a3f81f7afe3071760e8fcfc20cb79099`.
The test retains one full-image fingerprint per picture, checks a generation
tag on every GPU output pixel, and independently evaluates crop and color
arithmetic on the host with a one-sample rounding bound. Native success feedback
alone cannot establish correct pixels or reference mappings.

`hevc8.packet` and `hevc10.packet` contain twelve x265-encoded pictures apiece,
with independent FFmpeg software reconstruction fingerprints. Their little-endian
header contains `H265CTSF`, sample precision, width, height, and picture count.
Each picture contains 24 DWORDs: POC, reconstructed slot, random-access flags,
slice-header RPS bit count, reference count, before/after set counts, compressed
extent, four slot/POC pairs, and four indices for each before/after set. A 64-bit
FNV-1a fingerprint of per-pixel Y/U/V samples and the padded Annex B slice follow.
Eight-bit samples occupy one byte each; P010 samples occupy two little-endian
bytes with the full ten-bit value in the high bits. Decode-order POCs are
0, 3, 2, 1, 6, 5, 4, 9, 8, 7, 11, and 10. The fixture SHA-256 values are
`e0edaf272b6a56ba450331f7e93f069d7c4455cd2d84f4cdc29adff9c7fd8f67`
and `044979a7f795fef2d31cbd634cc428866b5f75a86a03be47a7a8bd77271131a3`.
P010 checks retain every sample bit and require all six low storage bits to be
zero. No codec library participates in CTS execution or the media library.

The raw-address GPU consumer currently qualifies gfx1150's NV12/P010 64K_D_X
layout with zero pipe/bank XOR. Its source is rebuilt through Loom CXX and the
ordinary CTS kernel embedding rules. The test admits the exact returned layout before
dispatch; this witness does not establish other swizzles, gfx12, or hardware
image-instruction consumption of the returned descriptors.
Those require their corresponding execution witnesses.

The encode corpus runs H264 Main, HEVC Main/Main10, and AV1 Main at eight and
ten bits where the profile permits. Every row executes CQP, CBR, and VBR over
twelve GPU-produced pictures with four reusable recordings. Separate dependent-
picture cases prove that application-owned reconstruction is consumed instead
of replaying an intra picture. The first four compressed pictures are decoded
through an independent native session, consumed by the GPU, and compared with
the generated image. Output guards, the reported written extent, immutable
recordings, native queue dependencies, checked retirement, and teardown are
verified in the same process. Linux gfx1151 and gfx1201 use linear GPU input;
Windows gfx1150 uses the qualified 64K_D_X image layout.
