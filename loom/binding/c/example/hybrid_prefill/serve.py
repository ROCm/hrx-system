"""Launch the experimental Qwen Q4_K prefill adapter with an existing Loom SDK."""
import argparse
import json
import math
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk',required=True,type=Path,help='directory containing loom-serve')
    p.add_argument('--model',required=True,type=Path)
    p.add_argument('--artifacts',required=True,type=Path)
    p.add_argument('--prefill-offload-device',choices=['xdna'])
    p.add_argument('--prefill-offload-root',choices=['g_gemm_q4_k_store_m256'],default='g_gemm_q4_k_store_m256')
    p.add_argument('--prefill-row-fraction',type=float,default=.15,help='fraction of output features, rounded to 512 features')
    p.add_argument('--precision',choices=['i8x2','i8'],default='i8x2')
    a,extra=p.parse_known_args()
    env=dict(os.environ)
    if a.prefill_offload_device:
        if not math.isfinite(a.prefill_row_fraction) or not 0<a.prefill_row_fraction<1:p.error('fraction must be finite and in (0,1)')
        n=int(17408*a.prefill_row_fraction/512+.5)*512
        if not 0<n<17408:p.error('aligned fraction leaves an empty branch')
        image=a.artifacts.resolve()/f'bank_{n}.bin'
        manifest=Path(str(image)+'.command.json')
        if not manifest.exists() or json.loads(manifest.read_text())['shape']!=[512,n,5120]:p.error('compile the matching bank image first with compile.py')
        adapter=a.artifacts.resolve()/'sdk_adapter.so'
        if not adapter.is_file():p.error('build the SDK adapter first')
        if not env.get('GPU_GRANT') or not env.get('NPU_GRANT'):p.error('acquire both hardware queues and export their grant IDs first')
        env.update(LOOM_HYBRID_FRACTION=str(a.prefill_row_fraction),LOOM_HYBRID_ROOT=a.prefill_offload_root,
                   LOOM_HYBRID_PRECISION=a.precision,LOOM_HYBRID_IMAGE=str(image),
                   LD_PRELOAD=str(adapter)+(':'+env['LD_PRELOAD'] if env.get('LD_PRELOAD') else ''))
    cmd=[str(a.sdk.resolve()/'loom-serve'),'--model='+str(a.model.resolve()),'--device=amdgpu',*extra]
    os.execve(cmd[0],cmd,env)


if __name__=='__main__':main()
