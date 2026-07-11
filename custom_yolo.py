import torch
import torch.nn as nn
from ultralytics.nn.model import DetectionModel
from ultralytics.nn.modules import Conv, C2f, Detect
from .pmse import PMSE, LightweightPMSE
from .scfpn import SCFPN
from .attention import CoordAttention

class CustomYOLOv8(DetectionModel):
    def __init__(self, cfg='yolov8s.yaml', ch=3, nc=10, use_lightweight_pmse=True, use_coord_attention=True, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        self.use_lightweight_pmse = use_lightweight_pmse
        self.use_coord_attention = use_coord_attention
        self._replace_modules()

    def _replace_modules(self):
        # Insert PMSE after certain C2f layers in backbone
        pmse_class = LightweightPMSE if self.use_lightweight_pmse else PMSE
        # Find indices of C2f layers (simplified: after layer 4, 6, 8)
        c2f_indices = [i for i, m in enumerate(self.model) if isinstance(m, C2f)]
        for idx in c2f_indices[1:3]:  # replace after second and third C2f
            in_ch = self.model[idx].cv2.conv.in_channels if hasattr(self.model[idx], 'cv2') else 256
            self.model.insert(idx+1, pmse_class(in_ch))
        # Replace neck with SCFPN (simplified: assume neck starts after backbone)
        neck_start = len(self.model) - 5  # approximate
        self.model = self.model[:neck_start] + nn.ModuleList([SCFPN()]) + self.model[neck_start+1:]
        if self.use_coord_attention:
            # Insert CoordAttention after each detection head
            for i, m in enumerate(self.model):
                if isinstance(m, Detect):
                    self.model.insert(i, CoordAttention(m.nc * m.no))
        # Re-initialize model weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def build_custom_model(nc=10, use_lightweight_pmse=True, use_coord_attention=True):
    from ultralytics import YOLO
    base = YOLO('yolov8s.pt')
    model = CustomYOLOv8(cfg='yolov8s.yaml', ch=3, nc=nc,
                         use_lightweight_pmse=use_lightweight_pmse,
                         use_coord_attention=use_coord_attention)
    # Copy pretrained weights where possible
    model.load_state_dict(base.model.state_dict(), strict=False)
    return model
