"""Re-verify the shipped envelope, using the canonical installed module."""
import math, os, subprocess, sys, time
import torch, vllm, vllm._custom_ops
from vllm_omni.quantization import fp8_online as fast

FP8 = torch.float8_e4m3fn

def sm_clock():
    try:
        return int(subprocess.run(["nvidia-smi","--query-gpu=clocks.sm","--format=csv,noheader,nounits","-i",str(torch.cuda.current_device())],capture_output=True,text=True,timeout=10).stdout.strip().splitlines()[0])
    except Exception:
        return -1

def wall(fn, iters=200, reps=7, warmup=60):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    out=[]
    for _ in range(reps):
        t0=time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter()-t0)/iters*1e6)
    return math.exp(sum(math.log(v) for v in out)/len(out))

def a1(x,o,s): getattr(torch.ops._C,"dynamic_per_token_scaled_fp8_quant")(o,x,s,None)
def p(x,o,s): fast.per_token(o,x,s,None)

print("module:", fast.__file__, "| max_rows:", fast.FASTPATH_MAX_ROWS)
ka=torch.randn(512,4096,device="cuda",dtype=torch.bfloat16)
ka_o=torch.empty_like(ka,dtype=FP8); ka_s=torch.empty((512,1),device="cuda",dtype=torch.float32)
for _ in range(3000): p(ka,ka_o,ka_s)
torch.cuda.synchronize()
print("clock", sm_clock(), "MHz")
print(f"{'M':>6}{'K':>6} {'A1 us':>9} {'P us':>9} {'A1/P':>7} {'path':>7}  exact")
rows=[]
for n in (3072,4096):
    for m in (1,4,16,64,128,256,512,1024,2048,4096):
        x=torch.randn(m,n,device="cuda",dtype=torch.bfloat16)*2
        o=torch.empty_like(x,dtype=FP8); s=torch.empty((m,1),device="cuda",dtype=torch.float32)
        a1(x,o,s); ro,rs=o.clone(),s.clone()
        p(x,o,s)
        ex=bool(torch.equal(ro.view(torch.uint8),o.view(torch.uint8)) and torch.equal(rs,s))
        ta=wall(lambda:a1(x,o,s)); tp=wall(lambda:p(x,o,s))
        for _ in range(2000): p(ka,ka_o,ka_s)
        torch.cuda.synchronize()
        path = "fast" if fast.supported(x) else "ref"
        rows.append((m,n,ta,tp,ta/tp,path,ex))
        print(f"{m:6d}{n:6d} {ta:9.2f} {tp:9.2f} {ta/tp:7.3f} {path:>7}  {'YES' if ex else 'NO'}")
import json
json.dump([{"m":r[0],"n":r[1],"a1_us":r[2],"p_us":r[3],"ratio":r[4],"path":r[5],"exact":r[6]} for r in rows],
          open("results/envelope.json","w"), indent=2)
print("wrote results/envelope.json")
