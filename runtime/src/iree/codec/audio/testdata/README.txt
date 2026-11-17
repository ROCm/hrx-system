Deterministic two-tone audio: 1607 mono PCM16 frames at 16 kHz.
Frame i is round(16000*sin(2*pi*731*i/16000) +
8000*sin(2*pi*2137*i/16000)). Encoded with FFmpeg FLAC and
libmp3lame at 64 kbit/s. The .f32 files are independent FFmpeg
decodes in native source geometry, little-endian F32.
These small fixtures exercise codec framing and MP3 gapless trim;
resampling and multichannel conversion have generated WAV witnesses.
