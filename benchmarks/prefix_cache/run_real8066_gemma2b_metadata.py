from pathlib import Path
import json, os, subprocess
root=Path(__file__).parent
model=Path('/home/gongji/0z5a/work/omni-cache-three-pr-e2e-20260927/models/gemma-2b-ranges')
assert (model/'.complete').exists()
for round_id, variant, cache in [(1,'base',0),(1,'base',1),(1,'textfast',0),(1,'textfast',1),(2,'textfast',1),(2,'textfast',0),(2,'base',1),(2,'base',0)]:
    name=f'real8066-gemma2b-metadata-p47-r{round_id}-{variant}-cache{cache}'
    src=Path('/home/gongji/0z5a/work/omni7902/review-8066/src') if variant=='base' else root/'src-8066-textfast'
    if (root/name/'complete').exists():
        print('SKIP completed',name,flush=True)
        continue
    assert not (root/name).exists()
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    selected=[list(map(int,line.split(','))) for line in usage.splitlines() if int(line.split(',')[0]) in (4,7)]
    assert len(selected)==2 and all(memory<100 and util==0 for _,memory,util in selected), selected
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='4,7',OMNI_CACHE_E2E_MILESTONE='8066',PYTHONPATH=str(src),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    command=['taskset','-c','32-39,56-63,96-103,120-127','/home/gongji/0z5a/bin/python','probe_e2e_metadata.py','--model',str(model),'--output',name,'--cache',str(cache),'--milestone','8066','--requests','24','--device','0,1','--dcp-world-size','2','--block-size','16','--attention-backend','FLASHINFER','--memory','0.4','--prompt-tokens','960','--max-tokens','32']
    print('START',name,flush=True)
    with (root/'logs'/f'{name}.log').open('w') as log:
        result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0 and (root/name/'complete').exists(), (name,result.returncode)
    print('COMPLETE',name,flush=True)
(root/'real8066-gemma2b-metadata-p47-all-complete').write_text('eight arms exited naturally\n')
