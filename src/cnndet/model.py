"""
model.py - CNN propia para detección (entrenada desde cero).

Diseño (estilo CenterNet: cada objeto se representa por su CENTRO):
  Backbone : stem + 4 etapas de bloques residuales propios (strides 4, 8, 16, 32)
  Neck     : pirámide top-down (FPN): fusiona los 4 niveles y devuelve un mapa a stride 4
  Cabezas  : hm  -> (nc) probabilidad de centro de cada clase
             wh  -> (2)  log(ancho), log(alto) de la caja, en celdas
             off -> (2)  desplazamiento subcelda del centro
No usa anclas ni pesos preentrenados.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1):
        super().__init__(nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.SiLU(inplace=True))


class ResBlock(nn.Module):
    """Dos convoluciones 3x3 + conexión residual (atajo 1x1 si cambian canales o resolución)."""
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = ConvBNAct(cin, cout, 3, stride)
        self.c2 = nn.Sequential(nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        nn.init.zeros_(self.c2[1].weight)   # el bloque parte como identidad: entrena más estable
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.c2(self.c1(x)) + self.skip(x))


def stage(cin, cout, n):
    return nn.Sequential(ResBlock(cin, cout, 2), *[ResBlock(cout, cout) for _ in range(n - 1)])


def head(cin, mid, cout, drop):
    return nn.Sequential(ConvBNAct(cin, mid, 3), nn.Dropout2d(drop) if drop > 0 else nn.Identity(),
                         nn.Conv2d(mid, cout, 1))


class CenterDet(nn.Module):
    def __init__(self, nc, width=1.0, depths=(2, 2, 3, 2), neck_ch=128, drop=0.1):
        super().__init__()
        w = [max(8, int(c * width)) for c in (32, 64, 128, 256, 512)]
        self.stem = nn.Sequential(ConvBNAct(3, w[0], 3, 2), ConvBNAct(w[0], w[0], 3, 1))  # stride 2
        self.s1, self.s2 = stage(w[0], w[1], depths[0]), stage(w[1], w[2], depths[1])      # 4, 8
        self.s3, self.s4 = stage(w[2], w[3], depths[2]), stage(w[3], w[4], depths[3])      # 16, 32
        self.lat = nn.ModuleList([nn.Conv2d(c, neck_ch, 1) for c in w[1:]])
        self.smooth = nn.ModuleList([ConvBNAct(neck_ch, neck_ch, 3) for _ in range(3)])
        self.hm, self.wh, self.off = head(neck_ch, 64, nc, drop), head(neck_ch, 64, 2, drop), head(neck_ch, 64, 2, drop)
        nn.init.constant_(self.hm[-1].bias, -4.6)    # al inicio P(centro)≈0.01: la pérdida inicial no se dispara

    def forward(self, x):
        c1 = self.s1(self.stem(x)); c2 = self.s2(c1); c3 = self.s3(c2); c4 = self.s4(c3)
        up = lambda t: F.interpolate(t, scale_factor=2, mode="nearest")
        p4 = self.lat[3](c4)
        p3 = self.smooth[0](self.lat[2](c3) + up(p4))
        p2 = self.smooth[1](self.lat[1](c2) + up(p3))
        p1 = self.smooth[2](self.lat[0](c1) + up(p2))   # stride 4
        return {"hm": self.hm(p1), "wh": self.wh(p1), "off": self.off(p1)}
