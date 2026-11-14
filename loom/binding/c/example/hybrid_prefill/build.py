"""Build the experimental HAL adapter against headers matching the SDK ABI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--headers', required=True, type=Path,
                   help='publisher-pinned runtime/src directory (d01d1c1 for this SDK)')
    p.add_argument('--generated-headers', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--cxx', default='c++')
    p.add_argument('--cc', default='cc')
    p.add_argument('--sdk', type=Path, help='also build compile_npu against SDK libraries')
    p.add_argument('--loomc-headers', type=Path,
                   help='matching loom/binding/c/include directory; required with --sdk')
    a = p.parse_args()
    if a.sdk and not a.loomc_headers:
        p.error('--sdk requires --loomc-headers')
    source = Path(__file__).with_name('host_adapter.cc').resolve()
    output = a.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = output.parent / 'compiler-tmp'
    scratch.mkdir(exist_ok=True)
    cmd = [a.cxx, '-std=c++17', '-fPIC', '-shared', '-O3', '-g',
           '-DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl',
           '-I', str(a.headers.resolve()), '-I', str(a.generated_headers.resolve()),
           str(source), '-o', str(output), '-ldl', '-pthread']
    with Path(str(output)+'.build.log').open('w') as log:
        subprocess.run(cmd, env=dict(os.environ, TMPDIR=str(scratch)),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    Path(str(output)+'.build.json').write_text(json.dumps({
        'argv': cmd, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'binary_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'abi_warning': 'Header pin is a tested compatibility choice, not wheel source attestation.'
    }, indent=2))
    if a.sdk:
        capi = output.parent / 'compile_npu'
        cmd = [a.cc, '-std=c11', '-O2',
               '-DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl',
               '-I', str(a.loomc_headers.resolve()),
               '-I', str(a.headers.resolve()), '-I', str(a.generated_headers.resolve()),
               str(source.with_name('compile_npu.c')),
               '-L', str(a.sdk.resolve()), '-Wl,-rpath,'+str(a.sdk.resolve()),
               '-lloomc', '-liree', '-o', str(capi)]
        with Path(str(capi)+'.build.log').open('w') as log:
            subprocess.run(cmd, env=dict(os.environ, TMPDIR=str(scratch)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        Path(str(capi)+'.build.json').write_text(json.dumps({'argv': cmd,
            'binary_sha256': hashlib.sha256(capi.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
