# Native media and GPU ownership witness

The H264 corpus exercises media and compute through the runtime-loaded libamdf
tables. Twelve pictures include seven B pictures, three simultaneously live
short-term references, and display order different from decode order. A second
case repeats the stream through 32 pictures using four recording owners, two
decoded output images, four reconstructed reference images, and fixed input,
feedback, and GPU result backing. The application supplies every reference,
placement, recording slot, and native queue dependency.

The GPU publishes compressed bytes, waits natively for decode, reads the shared
NV12 output, and produces a moving 64x64 RGB crop. A decode that reuses an output
slot waits for that slot's last GPU consumer. Host observation occurs once per
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

The raw-address GPU consumer currently qualifies gfx1150's NV12 64K_D_X layout
with zero pipe/bank XOR. Its source is rebuilt through Loom CXX and the ordinary
CTS kernel embedding rules. The test admits the exact returned layout before
dispatch; this witness does not establish other swizzles, gfx12, 10-bit codec
behavior, or hardware image-instruction consumption of the returned descriptors.
Those require their corresponding execution witnesses.
