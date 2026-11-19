from __future__ import annotations
from collections.abc import Mapping
import torch
from torch import nn
from .cte_encoders import CTEInputEncoder
from .cte_hgnn import CTEHGNN
from .spatial_graph import TactileSpatialGraphBranch

class SingleSensorSpatiotemporalFusion(nn.Module):
    """Two tokens: temporal CTE embedding and one spatial sensor embedding."""
    def __init__(self,dim,dropout=0.1,heads=4):
        super().__init__()
        if dim%heads: raise ValueError("feature_dim must be divisible by fusion_heads")
        self.temporal_projection=nn.Sequential(nn.LayerNorm(2*dim),nn.Linear(2*dim,dim),nn.GELU())
        self.position=nn.Parameter(torch.randn(1,2,dim)*0.02)
        layer=nn.TransformerEncoderLayer(d_model=dim,nhead=heads,dim_feedforward=2*dim,
            dropout=dropout,activation="gelu",batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=1)
        hidden=max(8,dim//2)
        self.pool=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.norm=nn.LayerNorm(dim)
    def forward(self,temporal,spatial):
        if spatial.ndim!=3 or spatial.shape[1]!=1: raise ValueError("Expected spatial [B,1,D]")
        tokens=torch.cat((self.temporal_projection(temporal).unsqueeze(1),spatial),dim=1)+self.position
        encoded=self.encoder(tokens); weights=torch.softmax(self.pool(encoded).squeeze(-1),dim=-1)
        return self.norm((encoded*weights.unsqueeze(-1)).sum(1)),weights

class STCTESingleSensorClassifier(nn.Module):
    def __init__(self,num_classes:int,config:Mapping):
        super().__init__()
        if num_classes<2: raise ValueError("At least two classes are required")
        dim=int(config.get("feature_dim",128)); sd=int(config.get("spatial_dim",dim)); drop=float(config.get("dropout",0.2))
        if sd!=dim: raise ValueError("spatial_dim must equal feature_dim")
        self.encoder=CTEInputEncoder(dim,str(config.get("backbone","resnet18")),bool(config.get("pretrained",False)))
        self.temporal=CTEHGNN(dim,num_classes,int(config.get("cte_num_atoms",8)),drop,float(config.get("cte_sigma_seconds",0.03)))
        self.spatial=TactileSpatialGraphBranch(dim,dim,drop,int(config.get("spatial_graph_layers",2)),int(config.get("spatial_adaptive_neighbors",4)))
        self.fusion=SingleSensorSpatiotemporalFusion(dim,drop,int(config.get("fusion_heads",4)))
        self.classifier=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Dropout(drop),nn.Linear(dim,num_classes))
    @staticmethod
    def _masks(batch):
        keys=("rgb","flow","rgb_mask","flow_mask","flow_dt","rgb_times","flow_start","flow_end","node_mask")
        missing=[k for k in keys if k not in batch]
        if missing: raise KeyError(f"Missing fields: {missing}")
        rm,fm,nm=batch["rgb_mask"].bool(),batch["flow_mask"].bool(),batch["node_mask"].bool()
        if rm.ndim!=3 or rm.shape[1]!=1: raise ValueError("rgb_mask must be [B,1,T]")
        if fm.ndim!=3 or fm.shape[:2]!=rm.shape[:2]: raise ValueError("flow_mask must be [B,1,Tf]")
        if nm.shape!=rm.shape[:2]: raise ValueError("node_mask must be [B,1]")
        return rm&nm.unsqueeze(-1),fm&nm.unsqueeze(-1)
    def forward_with_aux(self,batch,query_time=None):
        rm,fm=self._masks(batch)
        if batch["rgb"].shape[1]!=1 or batch["flow"].shape[1]!=1: raise ValueError("Single-sensor model requires N=1")
        enc={**batch,"rgb_mask":rm,"flow_mask":fm}
        rf,ff=self.encoder(enc)
        ht,taux=self.temporal(rf,ff,batch["rgb_times"],batch["flow_start"],batch["flow_end"],rm,fm,
            query_time=query_time,return_aux=True,return_features=True)
        _,spatial=self.spatial(batch["rgb"],rm,batch["rgb_times"],query_time)
        fused,weights=self.fusion(ht,spatial); logits=self.classifier(fused)
        aux={**taux,"fusion_token_weights":weights.detach(),
             "spatial_feature_norm":spatial.detach().norm(dim=-1).mean(),
             "temporal_feature_norm":ht.detach().norm(dim=-1).mean(),
             "fused_feature_norm":fused.detach().norm(dim=-1).mean()}
        return logits,aux
    def forward(self,batch): return self.forward_with_aux(batch)[0]
    def forward_online(self,batch,query_time):
        logits,aux=self.forward_with_aux(batch,query_time)
        probs=torch.softmax(logits,dim=-1)
        return {"logits":logits,"probabilities":probs,"prediction":probs.argmax(-1),"aux":aux}
