
from fastai.torch_imports import *
from fastai.core import *
from fastai.conv_learner import children
from .modules import SaveFeatures
import torchvision.models as models


class GramStyleLoss(nn.Module):
    def __init__(self, multiplier:float=1e4):
        super().__init__()
        m_vgg = vgg16(True).cuda().eval()  
        blocks = [i-1 for i,o in enumerate(children(m_vgg)) if isinstance(o,nn.MaxPool2d)]
        blocks, [m_vgg[i] for i in blocks]
        #vgg_layers = children(m_vgg)[:23]
        #m_vgg = nn.Sequential(*vgg_layers).cuda().eval()
        set_trainable(m_vgg, False) 
        self.m = m_vgg
        self.sfs = [SaveFeatures(children(m_vgg)[blocks[3]])]
        self.multiplier = multiplier


    def forward(self, input, target):
        self.m(VV(target.data))
        targ_styles = [V(o.features.clone()) for o in self.sfs]
        self.m(input)
        outs = [V(o.features) for o in self.sfs]
        losses = [self._gram_mse_loss(o, s) for o,s in zip(outs, targ_styles)]
        return sum(losses)

    def _gram(self, input):
        b,c,h,w = input.size()
        x = input.view(b*c, -1)
        return torch.mm(x, x.t())/input.numel()*self.multiplier

    def _gram_mse_loss(self, input, target): 
        return F.mse_loss(self._gram(input), self._gram(target))

class FeatureLoss(nn.Module):
    def __init__(self, block_wgts:[float]=[0.2,0.7,0.1], multiplier:float=1.0):
        super().__init__()
        m_vgg = vgg16(True)  
        blocks = [i-1 for i,o in enumerate(children(m_vgg)) if isinstance(o,nn.MaxPool2d)]
        layer_ids = blocks[:3]
        
        vgg_layers = children(m_vgg)[:23]
        m_vgg = nn.Sequential(*vgg_layers).cuda().eval()
        set_trainable(m_vgg, False)
        
        self.m,self.wgts = m_vgg,block_wgts
        self.sfs = [SaveFeatures(m_vgg[i]) for i in layer_ids]
        self.multiplier = multiplier

    def forward(self, input, target, sum_layers:bool=True):
        self.m(VV(target.data))
        res = [F.l1_loss(input,target)/100]
        targ_feat = [V(o.features.data.clone()) for o in self.sfs]
        self.m(input)
        res += [F.l1_loss(self._flatten(inp.features),self._flatten(targ))*wgt
               for inp,targ,wgt in zip(self.sfs, targ_feat, self.wgts)]
        if sum_layers: res = sum(res)
        return res*self.multiplier
    
    def _flatten(self, x:torch.Tensor): 
        return x.view(x.size(0), -1)
    
    def close(self):
        for o in self.sfs: o.remove()


