from __future__ import annotations
import torch
from torch import nn

class LocalSpatialGATLayer(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1, adaptive_neighbors=4):
        super().__init__()
        if dim % heads: raise ValueError("dim must be divisible by heads")
        self.heads, self.head_dim = heads, dim // heads
        self.adaptive_neighbors = int(adaptive_neighbors)
        self.query=nn.Linear(dim,dim,bias=False); self.key=nn.Linear(dim,dim,bias=False)
        self.value=nn.Linear(dim,dim,bias=False); self.output=nn.Linear(dim,dim)
        self.dropout=nn.Dropout(dropout); self.norm1=nn.LayerNorm(dim); self.norm2=nn.LayerNorm(dim)
        self.ffn=nn.Sequential(nn.Linear(dim,2*dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*dim,dim))
    @staticmethod
    def build_adjacency(h,w,device):
        p=h*w; a=torch.eye(p,dtype=torch.bool,device=device)
        for y in range(h):
            for x in range(w):
                i=y*w+x
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        yy,xx=y+dy,x+dx
                        if (dx or dy) and 0<=yy<h and 0<=xx<w: a[i,yy*w+xx]=True
        return a
    def forward(self,nodes,adjacency):
        b,p,_=nodes.shape
        q=self.query(nodes).view(b,p,self.heads,self.head_dim)
        k=self.key(nodes).view(b,p,self.heads,self.head_dim)
        v=self.value(nodes).view(b,p,self.heads,self.head_dim)
        s=torch.einsum("bphd,bqhd->bpqh",q,k)/(self.head_dim**0.5)
        topk=min(max(1,self.adaptive_neighbors),p)
        idx=s.mean(-1).topk(topk,dim=-1).indices
        dynamic=torch.zeros(b,p,p,dtype=torch.bool,device=nodes.device)
        dynamic.scatter_(2,idx,True)
        s=s.masked_fill(~(adjacency.unsqueeze(0)|dynamic).unsqueeze(-1),torch.finfo(s.dtype).min)
        a=self.dropout(torch.softmax(s,dim=2))
        msg=torch.einsum("bpqh,bqhd->bphd",a,v).reshape(b,p,-1)
        x=self.norm1(nodes+self.dropout(self.output(msg)))
        return self.norm2(x+self.dropout(self.ffn(x)))

class TactileSpatialGraphBranch(nn.Module):
    """Variable sensor count; single-sensor input uses N=1."""
    def __init__(self,feature_dim=128,spatial_dim=128,dropout=0.1,graph_layers=2,adaptive_neighbors=4):
        super().__init__()
        if graph_layers<1: raise ValueError("graph_layers must be >= 1")
        self.spatial_dim=spatial_dim
        self.cnn=nn.Sequential(
            nn.Conv2d(3,32,3,2,1,bias=False),nn.GroupNorm(8,32),nn.GELU(),
            nn.Conv2d(32,64,3,2,1,bias=False),nn.GroupNorm(8,64),nn.GELU(),
            nn.Conv2d(64,spatial_dim,3,2,1,bias=False),nn.GroupNorm(8,spatial_dim),nn.GELU())
        self.graph_layers=nn.ModuleList([LocalSpatialGATLayer(spatial_dim,4,dropout,adaptive_neighbors) for _ in range(graph_layers)])
        self.node_attention=nn.Linear(spatial_dim,1)
        hidden=max(8,spatial_dim//2)
        self.temporal_attention=nn.Sequential(nn.Linear(spatial_dim,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,1))
        self.sensor_projection=nn.Sequential(nn.Linear(spatial_dim,feature_dim),nn.GELU(),nn.Dropout(dropout),nn.LayerNorm(feature_dim))
    def forward(self,rgb,rgb_mask,rgb_times=None,query_time=None):
        if rgb.ndim!=6 or rgb.shape[1]<1 or rgb.shape[3]!=3: raise ValueError("rgb must be [B,N,T,3,H,W]")
        b,n,t,c,h,w=rgb.shape
        if rgb_mask.shape!=(b,n,t): raise ValueError("rgb_mask must be [B,N,T]")
        valid=rgb_mask.bool()
        if query_time is not None:
            if rgb_times is None: raise ValueError("rgb_times required for online inference")
            times=torch.as_tensor(rgb_times,device=rgb.device,dtype=rgb.dtype)
            if times.shape==(b,t): times=times[:,None,:].expand(b,n,t)
            elif times.shape!=(b,n,t): raise ValueError("rgb_times must be [B,T] or [B,N,T]")
            q=torch.as_tensor(query_time,device=rgb.device,dtype=rgb.dtype)
            if q.ndim==0: q=q.expand(b,n)
            elif q.shape==(b,): q=q[:,None].expand(b,n)
            elif q.shape!=(b,n): raise ValueError("query_time must be scalar, [B] or [B,N]")
            valid=valid&(times<=q.unsqueeze(-1)+1e-9)
        flat=rgb.reshape(b*n*t,c,h,w); ids=valid.reshape(-1).nonzero(as_tuple=False).flatten()
        if ids.numel()==0: raise ValueError("No valid RGB frame at query_time")
        fmap=self.cnn(flat.index_select(0,ids)); _,_,mh,mw=fmap.shape
        nodes=fmap.flatten(2).transpose(1,2); adj=LocalSpatialGATLayer.build_adjacency(mh,mw,rgb.device)
        for layer in self.graph_layers: nodes=layer(nodes,adj)
        nw=torch.softmax(self.node_attention(nodes).squeeze(-1),dim=-1)
        frame=(nodes*nw.unsqueeze(-1)).sum(1)
        allf=rgb.new_zeros(b*n*t,self.spatial_dim).index_copy(0,ids,frame).reshape(b,n,t,self.spatial_dim)
        # Autocast may produce FP16 attention scores even though allf keeps
        # the input RGB dtype (normally FP32). Mask sentinels must match the
        # score dtype; using finfo(allf.dtype).min here can overflow when
        # scores is FP16. Perform this small temporal softmax in FP32 for
        # stable masking and normalization, including all-invalid sensor rows.
        scores = self.temporal_attention(allf).squeeze(-1).float()
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )
        per = (allf.float() * weights.unsqueeze(-1)).sum(dim=2)
        per=torch.where(valid.any(2).unsqueeze(-1),per,torch.zeros_like(per))
        per=self.sensor_projection(per)
        sm=valid.any(-1)
        episode=(per*sm.unsqueeze(-1)).sum(1)/sm.sum(1,keepdim=True).clamp_min(1)
        return episode,per
