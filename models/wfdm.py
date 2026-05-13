import torch
import torch.nn as nn
import ultralytics.nn.tasks as tasks

# =========================================================
# WFDM MODULE (Must perfectly match Phase 1)
# =========================================================
class WFDM(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        hf_ch = channels * 3
        self.denoise = nn.Sequential(
            nn.Conv2d(hf_ch, hf_ch, 3, 1, 1, groups=hf_ch, bias=False),
            nn.Conv2d(hf_ch, hf_ch, 1, bias=False),
            nn.BatchNorm2d(hf_ch),
            nn.SiLU()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(hf_ch, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def dwt(self, x):
        if x.shape[2] % 2 != 0: x = nn.functional.pad(x, (0,0,0,1))
        if x.shape[3] % 2 != 0: x = nn.functional.pad(x, (0,1,0,0))
        A=x[:,:,0::2,0::2]; B=x[:,:,0::2,1::2]
        C=x[:,:,1::2,0::2]; D=x[:,:,1::2,1::2]
        LL=(A+B+C+D)*0.5; LH=(A+B-C-D)*0.5
        HL=(A-B+C-D)*0.5; HH=(A-B-C+D)*0.5
        return LL,LH,HL,HH

    def iwt(self, LL, LH, HL, HH):
        H,W=LL.shape[2],LL.shape[3]
        out=torch.zeros(LL.shape[0],LL.shape[1],
                        H*2,W*2,device=LL.device,dtype=LL.dtype)
        out[:,:,0::2,0::2]=(LL+LH+HL+HH)*0.5
        out[:,:,0::2,1::2]=(LL+LH-HL-HH)*0.5
        out[:,:,1::2,0::2]=(LL-LH+HL-HH)*0.5
        out[:,:,1::2,1::2]=(LL-LH-HL+HH)*0.5
        return out

    def forward(self, x):
        orig_h,orig_w=x.shape[2],x.shape[3]
        LL,LH,HL,HH=self.dwt(x)
        hf=torch.cat([LH,HL,HH],dim=1)
        mask=self.spatial_gate(hf)
        hf=hf+mask*(self.denoise(hf)-hf)
        c=x.shape[1]
        out=self.iwt(LL,hf[:,:c],hf[:,c:2*c],hf[:,2*c:])
        out=nn.functional.interpolate(out,size=(orig_h,orig_w),
                                      mode='bilinear',align_corners=False)
        return out

tasks.WFDM = WFDM

# =========================================================
# INJECTION FUNCTION
# =========================================================
def inject_wfdm(model):
    module=WFDM(channels=64)
    module.i=15
    module.f=-1
    module.type='WFDM'
    layers=list(model.model.model)
    
    # Must use insert(16) to match how Phase 1 was trained!
    layers.insert(16,module)
    model.model.model=nn.Sequential(*layers)
    print("✅ WFDM successfully injected into YOLO architecture")
    return model
