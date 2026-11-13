"""Measure synchronized training segments with CUDA events.

Capture forward/backward, gradient packing, collective exchange and updates
with reproducible seeds, model dimensions and per-rank provenance.
"""
import argparse, json, os, time, hashlib
from pathlib import Path
import torch
from torch import nn
import torch.distributed as dist

class Residual(nn.Module):
    def __init__(self, a, b, stride=1):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(a,b,3,stride,1,bias=False),nn.BatchNorm2d(b),nn.ReLU(),nn.Conv2d(b,b,3,1,1,bias=False),nn.BatchNorm2d(b))
        self.skip=nn.Identity() if a==b and stride==1 else nn.Sequential(nn.Conv2d(a,b,1,stride,bias=False),nn.BatchNorm2d(b))
    def forward(self,x):return torch.relu(self.body(x)+self.skip(x))

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(768,12,3072,dropout=0.,activation='gelu',batch_first=True,norm_first=True) for _ in range(12)])
        self.norm=nn.LayerNorm(768)
        self.register_buffer('mask',torch.triu(torch.full((512,512),float('-inf')),1))
    def forward(self,x):
        for layer in self.layers:x=layer(x,src_mask=self.mask)
        return self.norm(x)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['transformer','resnet18'],required=True);p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=16);a=p.parse_args()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
    dist.init_process_group('nccl');torch.manual_seed(130000+rank);torch.set_num_threads(2)
    if a.model=='transformer':model=Decoder();shape=(4,512,768)
    else:
        blocks=[nn.Conv2d(3,64,7,2,3,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(3,2,1)]
        for x,y,s in [(64,64,1),(64,64,1),(64,128,2),(128,128,1),(128,256,2),(256,256,1),(256,512,2),(512,512,1)]:blocks.append(Residual(x,y,s))
        model=nn.Sequential(*blocks,nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(512,1000));shape=(16,3,224,224)
    model=model.cuda();x=torch.randn(shape,device='cuda');optimizer=torch.optim.SGD(model.parameters(),lr=1e-3,momentum=.9)
    for t in model.parameters():dist.broadcast(t.data,src=0)
    params=list(model.parameters());buckets=[];current=[];size=0
    for param in params:
        if current and size+param.numel()*4>25*1024**2:buckets.append(current);current=[];size=0
        current.append(param);size+=param.numel()*4
    if current:buckets.append(current)
    records=[];seq=0;started=time.time()
    def timed(label,fn,measured,comm=False,nbytes=0):
        nonlocal seq
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();value=fn();end.record();end.synchronize()
        if measured:
            records.append({'index':len(records),'kind':'collective' if comm else 'compute','name':label,'duration_us':max(1,round(start.elapsed_time(end)*1000)),'bytes':nbytes,'type':0 if comm else None,'sequence':seq if comm else None})
            if comm:seq+=1
        return value
    for step in range(3+a.steps):
        measured=step>=3;optimizer.zero_grad(set_to_none=True)
        def compute():
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(x).float().square().mean()
            loss.backward()
        timed('forward_backward',compute,measured)
        for bucket in buckets:
            flat=timed('gradient_pack',lambda:torch.cat([p.grad.flatten() for p in bucket]),measured)
            timed('gradient_allreduce',lambda:dist.all_reduce(flat),measured,True,flat.numel()*flat.element_size())
            def unpack():
                flat.div_(dist.get_world_size());pos=0
                for p in bucket:p.grad.copy_(flat[pos:pos+p.numel()].view_as(p));pos+=p.numel()
            timed('gradient_unpack',unpack,measured)
        timed('sgd_update',optimizer.step,measured)
        if rank==0:print(a.model,'step',step,'recorded',measured,flush=True)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    payload={'model':a.model,'rank':rank,'world_size':dist.get_world_size(),'gpu':torch.cuda.get_device_name(local),'torch':torch.__version__,'collection_unix_s':started,'completed_unix_s':time.time(),'steps':a.steps,'warmup_steps':3,'input_shape':shape,'parameter_count':sum(p.numel() for p in params),'bucket_bytes':25*1024**2,'records':records,'scope':'Measured synchronous data-parallel training segments on eight A100 GPUs. Random weights and synthetic inputs; compute uses BF16 autocast, FP32 gradients. Training uses explicit synchronization between compute and collective segments. CUDA durations include GPU execution within each segment. Collective timings are archived but counterfactual network replay recomputes them.'}
    target=out/f'rank_{rank}.json'
    if target.exists():raise RuntimeError('Refuse overwrite capture')
    target.write_text(json.dumps(payload,indent=2)+'\n');dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
