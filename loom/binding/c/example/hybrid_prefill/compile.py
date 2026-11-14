"""Compile the aligned, packed signed-i8 whole-array GEMM package program."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def shape_config(m, n, k):
    if min(m, n, k) <= 0 or any(v % 128 for v in (m, n, k)):
        raise ValueError('M, N and K must be positive multiples of 128')
    if (m // 128) * (n // 128) % 16:
        raise ValueError('output must contain a multiple of 16 macrotiles')
    if k * 128 * 128 > 2**31 - 1:
        raise ValueError('full signed-i8 reduction can overflow i32')
    config = dict(steps=k // 128, tiles=m * n // 16384 // 16, nblocks=n // 128)
    if config['tiles'] > 65536 or config['nblocks'] > 2048:
        raise ValueError('shape exceeds the package program configuration bounds')
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m', type=int, default=1024)
    p.add_argument('--n', type=int, default=1024)
    p.add_argument('--k', type=int, default=5120)
    p.add_argument('--compiler', default='loom-compile')
    p.add_argument('--capi-runner', type=Path,
                   help='compile_npu executable linked to the SDK libloomc')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    try:
        config = shape_config(a.m, a.n, a.k)
    except ValueError as e:
        p.error(str(e))
    output = a.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = output.parent / 'compiler-tmp'
    scratch.mkdir(exist_ok=True)
    source = Path(__file__).with_name('whole_array_i8.loom')
    cmd = [a.compiler, str(source), '--root=prefill_i8',
           '--target=amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11',
           '--compile-report=summary', '--compile-report-output='+str(output)+'.json',
           '--output='+str(output)]
    cmd += [f'--config=whole_array.{key}={value}' for key, value in config.items()]
    if a.capi_runner:
        cmd = [str(a.capi_runner.resolve()), str(source),
               'amd.xdna.aie2p:amd.xdna.strix_halo.17f0_11', str(output),
               str(config['steps']), str(config['tiles']), str(config['nblocks'])]
    Path(str(output)+'.command.json').write_text(json.dumps({'argv':cmd,'shape':[a.m,a.n,a.k],
        'layouts':{'a':'[M/128,K/128,16,16,8,8]','b':'[N/128,K/128,16,16,8,8]',
                   'c':'[M/128,N/128,16,16,8,8]'}}, indent=2))
    with Path(str(output)+'.log').open('w') as log:
        subprocess.run(cmd, env=dict(os.environ, TMPDIR=str(scratch)), stdout=log,
                       stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
