import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Relative Positional Bias (T5-style)
# =========================
# این ماژول برای هر لایه یک bias یادگرفتنی بر اساس فاصله‌ی نسبی بین دو
# موقعیت زمانی به امتیاز attention اضافه می‌کند. یک نمونه‌ی مشترک بین همه‌ی
# EncoderBlockها استفاده می‌شود (weight tying به سبک T5) تا پارامتر اضافه‌ی
# کمی به مدل تحمیل شود.
class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_len):
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        self.rel_bias = nn.Parameter(torch.zeros(2 * max_len - 1, num_heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

    def forward(self, seq_len):
        if seq_len > self.max_len:
            import warnings
            warnings.warn(
                f"seq_len ({seq_len}) > max_len ({self.max_len}); relative position "
                f"indices will saturate at the boundary instead of reflecting the "
                f"true distance. Re-instantiate KWT with a larger img_x if longer "
                f"sequences are expected."
            )

        device = self.rel_bias.device
        positions = torch.arange(seq_len, device=device)
        rel_pos = positions[None, :] - positions[:, None]
        rel_pos = rel_pos + (self.max_len - 1)
        rel_pos = rel_pos.clamp(0, 2 * self.max_len - 2)

        bias = self.rel_bias[rel_pos]        # (T, T, num_heads)
        bias = bias.permute(2, 0, 1)          # (num_heads, T, T)
        return bias


# =========================
# GQA Attention
# =========================

class MultiHeadAttentionGQA(nn.Module):
    def __init__(self, embed_dim, num_heads, gqa_groups=2, dropout=0.4):
        super().__init__()

        # ------------------------------------------------------------
        # اعتبارسنجی: چون gqa_groups و num_heads اکنون از خط فرمان
        # (train.py) قابل تنظیم هستند، یک ترکیب نامعتبر (مثلاً heads=8،
        # gqa_groups=3) باید همین‌جا با یک خطای واضح متوقف شود، نه با یک
        # RuntimeError مبهم از عمق forward() یا با نتایج بی‌صدا غلط.
        # ------------------------------------------------------------
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) باید بر num_heads ({num_heads}) بخش‌پذیر باشد."
            )
        if num_heads % gqa_groups != 0:
            raise ValueError(
                f"num_heads ({num_heads}) باید بر gqa_groups ({gqa_groups}) بخش‌پذیر باشد "
                f"(GQA نیاز دارد هر گروه دقیقاً به تعداد مساوی از headها اختصاص یابد)."
            )

        self.num_heads = num_heads
        self.gqa_groups = gqa_groups
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, gqa_groups * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, gqa_groups * self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rel_bias=None):
        """
        rel_bias: تنسور از پیش محاسبه‌شده به شکل (num_heads, T, T)، یا None.
        بر خلاف نسخه‌ی قبلی، این ماژول دیگر خودش RelativePositionBias را صدا
        نمی‌زند — چون آن ماژول بین همه‌ی ۱۲ بلوک مشترک است و T در کل مدل ثابت
        است، محاسبه‌ی bias یک‌بار در KWT.forward انجام می‌شود و همان تنسور
        آماده به هر ۱۲ بلوک پاس داده می‌شود (نه ۱۲ بار محاسبه‌ی تکراری).
        """
        B, T, C = x.shape
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        K = self.k_proj(x).view(B, T, self.gqa_groups, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.gqa_groups, self.head_dim).transpose(1, 2)

        K = K.repeat_interleave(self.num_heads // self.gqa_groups, dim=1)
        V = V.repeat_interleave(self.num_heads // self.gqa_groups, dim=1)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if rel_bias is not None:
            attn = attn + rel_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


# =========================
# Encoder Block
# =========================
class EncoderBlock(nn.Module):
    def __init__(self, dim, mlp_dim, heads, dropout, gqa_groups):
        super().__init__()
        self.attn = MultiHeadAttentionGQA(dim, heads, gqa_groups, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rel_bias=None):
        x = x + self.dropout(self.attn(self.norm1(x), rel_bias=rel_bias))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# =========================
# FULL KWT MODEL (TRAIN.PY COMPATIBLE)
# =========================
class KWT(nn.Module):
    def __init__(
        self,
        img_x,
        img_y,
        patch_x,
        patch_y,
        num_classes,
        dim=256,
        depth=9,
        heads=8,
        mlp_dim=512,
        channels=1,
        dropout=0.4,
        emb_dropout=0.1,
        gqa_groups=2,
        use_conv_encoder=False,
        use_relative_pos=False,
        share_layers=False,
        pool="mean",
        **kwargs
        # نکته: patch_x, patch_y, dim_head, use_layer_norm, use_residual,
        # use_attention_mask, weight_decay, warmup_epochs, device و مشابه آن‌ها
        # همچنان در این کلاس استفاده نمی‌شوند و صرفاً برای سازگاری با امضای
        # فراخوانی در train.py پذیرفته می‌شوند، بدون اثر روی معماری.
    ):
        super().__init__()

        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) باید بر heads ({heads}) بخش‌پذیر باشد.")
        if heads % gqa_groups != 0:
            raise ValueError(
                f"heads ({heads}) باید بر gqa_groups ({gqa_groups}) بخش‌پذیر باشد."
            )

        self.pool = pool
        self.use_relative_pos = use_relative_pos
        self.share_layers = share_layers

        # -------------------------
        # Front-end
        # -------------------------
        if use_conv_encoder:
            self.front = nn.Sequential(
                nn.Conv1d(img_y, dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                nn.Conv1d(dim, dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(dim),
                nn.GELU()
            )
        else:
            self.front = nn.Linear(img_y, dim)

        # -------------------------
        # Positional encoding (مطلق - همیشه فعال)
        # -------------------------
        self.pos = nn.Parameter(torch.randn(1, img_x, dim))
        self.emb_drop = nn.Dropout(emb_dropout)

        # -------------------------
        # Relative Positional Bias (اختیاری، weight-tied بین همه‌ی بلوک‌ها)
        # -------------------------
        self.rel_pos_bias = None
        if use_relative_pos:
            self.rel_pos_bias = RelativePositionBias(num_heads=heads, max_len=img_x)

        # -------------------------
        # Encoder stack
        # -------------------------
        # اگر share_layers=True (سبک ALBERT / Universal Transformer)، فقط
        # یک EncoderBlock ساخته می‌شود و همان یک شیء، depth بار در ModuleList
        # تکرار می‌شود — یعنی وزن‌ها بین همه‌ی لایه‌ها کاملاً مشترک هستند
        # (weight tying، دقیقاً همان الگویی که برای RelativePositionBias
        # استفاده کردیم). چون model.parameters()/named_modules() به‌صورت
        # پیش‌فرض بر اساس id تنسور/ماژول دیدوپلیکیت می‌کنند، count_parameters
        # این بلوک را فقط یک‌بار می‌شمارد — یعنی کاهش پارامتر واقعی و
        # قابل‌اندازه‌گیری است، نه یک مصنوع اندازه‌گیری.
        #
        # نکته‌ی مهم: FLOPs/MACs تغییری نمی‌کند، چون در forward همچنان
        # depth بار (یک‌بار به‌ازای هر لایه) forward واقعی اجرا می‌شود —
        # فقط تعداد وزن‌های *ذخیره‌شده* کم می‌شود، نه تعداد عملیات محاسباتی.
        if share_layers:
            shared_block = EncoderBlock(dim, mlp_dim, heads, dropout, gqa_groups)
            self.blocks = nn.ModuleList([shared_block for _ in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                EncoderBlock(dim, mlp_dim, heads, dropout, gqa_groups)
                for _ in range(depth)
            ])

        self.norm = nn.LayerNorm(dim)

        # -------------------------
        # Classifier
        # -------------------------
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        B, T, C = x.shape

        if isinstance(self.front, nn.Sequential):
            x = x.transpose(1, 2)
            x = self.front(x)
            x = x.transpose(1, 2)
        else:
            x = self.front(x)

        x = x + self.pos[:, :T, :]
        x = self.emb_drop(x)

        # bias نسبی فقط یک‌بار در کل forward pass محاسبه می‌شود (نه یک‌بار
        # به‌ازای هر بلوک)، چون T ثابت است و rel_pos_bias بین بلوک‌ها مشترک است.
        rel_bias_tensor = None
        if self.use_relative_pos and self.rel_pos_bias is not None:
            rel_bias_tensor = self.rel_pos_bias(T)

        for blk in self.blocks:
            x = blk(x, rel_bias=rel_bias_tensor)

        x = self.norm(x)

        if self.pool == "mean":
            x = x.mean(dim=1)
        else:
            x = x[:, 0]

        return self.fc(x)