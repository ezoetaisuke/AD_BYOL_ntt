import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Conv2D （周波数×時間を画像とみなす） ----
class Conv2dAE(nn.Module):
    def __init__(self, in_ch=1, bottleneck_dim: int = 256):
        super().__init__()
        # Encoder: [B, 1, F, T] -> latent mean/logvar
        # 小規模・分かりやすさ優先のベースライン構成
        # 解像度は落ちるが、チャネル（観点）が増えるイメージ

        # バッチサイズが小さいときはbatchnormよりもgroupnormを使った方がよい
        def _gn(c):
            g = 32 if c >= 32 else max(1, c//4)
            return nn.GroupNorm(g, c)
        
        '''
        4段程度だと、最終マップが十分小さくなり、全結合（flat→μ,logvar）の入出力次元が現実的になる
        圧縮を深くしすぎると高周波の微細構造復元が難しくなる
        最終の特徴マップが 4〜16 ピクセル程度（各軸）に収まる深さが実務的に安定。
        ReLUを入れるのはモデルが複雑な境界を学習できるようにするため（ReLU無しだと単純な境界しか学習できない）
        '''
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=32, kernel_size=3, stride=2, padding=1),  _gn(32), nn.ReLU(inplace=True),  # /2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),     _gn(64),  nn.ReLU(inplace=True),   # /4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),    _gn(128), nn.ReLU(inplace=True),   # /8
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),   _gn(256), nn.ReLU(inplace=True),   # /16
        )
        
        # self.enc = nn.Sequential(
        #     nn.Conv2d(in_channels=in_ch, out_channels=32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),  # /2
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),   nn.BatchNorm2d(64),  nn.ReLU(inplace=True),   # /4
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),  nn.BatchNorm2d(128), nn.ReLU(inplace=True),   # /8
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),   # /16
        # )


        # 動的初期化する層（最初のforwardでサイズが決まる）
        self.enc_out_shape = None
        self.fc_enc = None
        self.fc_dec = None

        self.dec_body = None
        self.dec_head = None      

        self.z_dim = bottleneck_dim
        # 入力の空間サイズ（F, T）を保存し、デコード後に補正（補間）する
        self.input_shape = None

    def _init_latent_layers(self, enc_out_shape, device=None):
        """
        enc_out_shape: (C, F', T')
        device: この時点でのテンソルが載っているデバイス（h.device を渡す）
        ※ ここで生成する全レイヤを必ず device に移す（遅延初期化の落とし穴対策）
        """
        c, fh, fw = enc_out_shape
        flat_dim = c * fh * fw

        # 生成 → ただちに device へ
        self.fc_enc    = nn.Linear(flat_dim, self.z_dim).to(device)
        self.fc_dec = nn.Linear(self.z_dim, flat_dim).to(device)

        self.dec_body = nn.Sequential(
            nn.ConvTranspose2d(in_channels=c,   out_channels=128,   kernel_size=4,  stride=2,   padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64,    kernel_size=4,  stride=2,   padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64,  out_channels=32,    kernel_size=4,  stride=2,   padding=1), nn.ReLU(inplace=True),
        ).to(device)
        self.dec_head = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4,  stride=2,   padding=1).to(device)

    def encode(self, x):
        h = self.enc(x)  # [B, C, F', T']
        if self.enc_out_shape is None:
            self.enc_out_shape = h.shape[1:]  # (C,F',T')
            # ★ 遅延初期化：この時点のテンソル h の device（= x.device）に合わせる
            self._init_latent_layers(self.enc_out_shape, device=h.device)
        z = self.fc_enc(torch.flatten(h, 1))
        return z

    def decode(self, z):
        C, Fh, Fw = self.enc_out_shape
        h = self.fc_dec(z).view(-1, C, Fh, Fw)
        h = self.dec_body(h)
        x_hat = self.dec_head(h)
        if self.input_shape is not None:
            x_hat = F.interpolate(x_hat, size=self.input_shape, mode="bilinear", align_corners=False)
        return x_hat

    def forward(self, x: torch.Tensor):
        self.input_shape = x.shape[2:]
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def recon_loss(x: torch.Tensor, x_hat: torch.Tensor, loss_type: str = "mse", sigma2: float = 1.0) -> torch.Tensor:
    """AE reconstruction loss options: mse / l1 / gaussian_nll(fixed var)."""
    if loss_type == "mse":
        return F.mse_loss(x_hat, x, reduction="mean")
    elif loss_type == "l1":
        return F.l1_loss(x_hat, x, reduction="mean")
    elif loss_type == "gaussian_nll":
        return F.mse_loss(x_hat, x, reduction="mean") / (2.0 * float(sigma2))
    else:
        raise ValueError(f"Unsupported AE recon loss: {loss_type}")
