import sys
import os
import time
import copy
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.quantization
import torch.nn.quantized.dynamic as nnqd
import torch.nn.utils.prune as prune
from tqdm import tqdm
from thop import profile, clever_format

from models.kwt import KWT
from utils.warmup_scheduler import GradualWarmupScheduler
from utils.label_smoothing import LabelSmoothingLoss
from datagen import Datagen


# =========================================================
# توابع صحیح شمارش پارامتر / sparsity / FLOPs برای مدل کوانتیزه‌شده
# =========================================================

def count_parameters(model):
    """شمارش استاندارد پارامترهای trainable (فقط برای مدل کوانتیزه‌نشده معتبر است)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_quant_aware(model):
    """
    شمارش درست تعداد پارامتر برای مدلی که ممکن است شامل لایه‌های
    torch.nn.quantized.dynamic.Linear باشد.

    نتیجه باید تقریباً برابر با تعداد پارامترهای مدل قبل از کوانتیزاسیون
    باشد، چون کوانتیزاسیون دینامیک تعداد پارامترها را عوض نمی‌کند — فقط
    bit-width ذخیره‌سازی آن‌ها را تغییر می‌دهد.
    """
    total = 0
    seen_ids = set()

    for p in model.parameters():
        if id(p) not in seen_ids:
            seen_ids.add(id(p))
            total += p.numel()

    for module in model.modules():
        if isinstance(module, nnqd.Linear):
            w = module.weight()
            total += w.numel()
            b = module.bias()
            if b is not None:
                total += b.numel()

    return total


def compute_real_sparsity(quantized_model):
    """
    درصد واقعی صفرهای موجود در وزن‌های لایه‌های Linear کوانتیزه‌شده را
    اندازه می‌گیرد. این عدد باید همیشه در کنار هر ادعای «sparsity-driven
    reduction» گزارش شود، نه فرض شود.
    """
    total_weights = 0
    zero_weights = 0

    for module in quantized_model.modules():
        if isinstance(module, nnqd.Linear):
            w_dequant = module.weight().dequantize()
            total_weights += w_dequant.numel()
            zero_weights += (w_dequant == 0).sum().item()

    if total_weights == 0:
        return 0.0
    return zero_weights / total_weights


def count_nonzero_dense(model):
    """
    تعداد پارامترهای «غیرصفر» را در تمام لایه‌های Linear/Conv1d (float،
    قبل از کوانتیزاسیون) می‌شمارد. برخلاف numel()، این عدد واقعاً با
    پرونینگ کم می‌شود چون وزن‌های هرس‌شده دقیقاً صفر هستند.
    توجه: این «تعداد پارامتر ذخیره‌شده» نیست (شکل تنسور عوض نمی‌شود)،
    بلکه «تعداد پارامتر غیرصفر» است — دو مفهوم متفاوت که باید در گزارش
    به‌وضوح از هم جدا بمانند.
    """
    total_nz, total_all = 0, 0
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            w = module.weight
            total_nz += (w != 0).sum().item()
            total_all += w.numel()
            if module.bias is not None:
                total_nz += (module.bias != 0).sum().item()
                total_all += module.bias.numel()
    return total_nz, total_all


def apply_pruning(model, amount):
    """
    پرونینگ L1 unstructured روی تمام لایه‌های Linear/Conv1d، سپس baking
    دائمی ماسک در وزن (prune.remove) تا مدل نهایی دیگر وابسته به
    reparametrization پرونینگ نباشد.
    """
    pruned = copy.deepcopy(model)
    pruned.to("cpu")
    for module in pruned.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return pruned


def sweep_pruning_amounts(base_model, val_generator, device,
                           amounts, max_acc_drop, get_likely_index_fn, number_of_correct_fn):
    """
    برای هر amount: مدل را هرس می‌کند، دقت validation را اندازه می‌گیرد،
    و کل جدول (amount -> nnz, sparsity%, val accuracy) را چاپ می‌کند.
    در پایان، بزرگ‌ترین amount ای را برمی‌گرداند که افت دقتش نسبت به مدل
    اصلی (هرس‌نشده) از max_acc_drop بیشتر نشده باشد.
    """

    def quick_val_accuracy(model):
        model.to(device)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in val_generator:
                data, target = data.float().to(device), target.to(device)
                output = model(data)
                pred = get_likely_index_fn(output)
                correct += number_of_correct_fn(pred, target)
                total += target.shape[-1]
        return correct / total

    baseline_acc = quick_val_accuracy(base_model)
    print(f"\nBaseline (unpruned) validation accuracy: {baseline_acc*100:.2f}%")

    print(f"\n{'Amount':>8s} {'NonZero':>12s} {'Total':>12s} {'Sparsity%':>10s} {'ValAcc%':>10s} {'AccDrop%':>10s}")
    results = []
    for amount in amounts:
        pruned = apply_pruning(base_model, amount)
        nz, total = count_nonzero_dense(pruned)
        acc = quick_val_accuracy(pruned)
        drop = (baseline_acc - acc) * 100
        sparsity_pct = 100 * (1 - nz / total)
        print(f"{amount:8.2f} {nz:12,d} {total:12,d} {sparsity_pct:10.2f} {acc*100:10.2f} {drop:10.2f}")
        results.append((amount, pruned, nz, total, acc, drop))

    acceptable = [r for r in results if r[5] <= max_acc_drop * 100]
    if not acceptable:
        print(f"\n⚠️ هیچ amount ای زیر آستانه‌ی افت دقت {max_acc_drop*100:.2f}% پیدا نشد. "
              f"کم‌ترین amount امتحان‌شده انتخاب می‌شود.")
        chosen = min(results, key=lambda r: r[0])
    else:
        chosen = max(acceptable, key=lambda r: r[0])

    print(f"\n✅ Chosen amount: {chosen[0]:.2f}  "
          f"(non-zero params: {chosen[2]:,} / {chosen[3]:,}, "
          f"val acc: {chosen[4]*100:.2f}%, acc drop: {chosen[5]:.2f}%)")
    return chosen[1], chosen


def get_model_size_kb(model):
    """
    حجم فایل روی دیسک (کیلوبایت) بر اساس torch.save(state_dict()).
    ⚠️ محدودیت شناخته‌شده: برای مدل‌هایی که یک ماژول کوانتیزه‌شده
    (nnqd.Linear) بین چند مسیر به‌اشتراک گذاشته شده (مثلاً share_layers=True)،
    این عدد را دست‌کم نگیرید اینکه معتبر است — چون _packed_params هر بار که
    state_dict() از یک مسیر دیگر بازدیدش می‌کند، وزن را از نو بازسازی می‌کند و
    یک تنسور فیزیکی تازه (نه رفرنس به همان تنسور قبلی) تولید می‌کند. نتیجه:
    وزن‌های به‌اشتراک‌گذاشته‌شده به‌جای یک‌بار، چندبار سریالایز می‌شوند و این
    عدد به‌طور مصنوعی بزرگ‌تر از حجم واقعی مدل می‌شود.
    برای مدل‌های بدون به‌اشتراک‌گذاری لایه (share_layers=False)، این عدد درست
    و قابل‌اعتماد است. برای هر مدلی با weight tying، از
    get_model_size_kb_dedup() استفاده کنید.
    """
    tmp_path = "._tmp_size_check.pth"
    torch.save(model.state_dict() if not isinstance(model, dict) else model, tmp_path)
    size_kb = os.path.getsize(tmp_path) / 1024
    os.remove(tmp_path)
    return size_kb


def get_model_size_kb_dedup(model):
    """
    حجم واقعی مدل (کیلوبایت) با دیدوپ صریح بر اساس id هر ماژول — درست برای
    هر دو حالت (کوانتیزه/غیرکوانتیزه، با/بدون weight tying).

    model.modules() به‌صورت پیش‌فرض بر اساس id ماژول دیدوپلیکیت می‌کند (دقیقاً
    همان مکانیزمی که model.parameters() هم استفاده می‌کند)، پس اگر یک
    EncoderBlock بین چند لایه مشترک باشد (share_layers=True)، اینجا هم فقط
    یک‌بار شمرده می‌شود — بدون وابستگی به رفتار سریالایزیشن torch.save که
    برای ماژول‌های کوانتیزه‌ی مشترک دچار باگ بازسازی تکراری وزن است.
    """
    total_bytes = 0
    seen_ids = set()

    for module in model.modules():
        if id(module) in seen_ids:
            continue
        seen_ids.add(id(module))

        if isinstance(module, nnqd.Linear):
            w = module.weight()  # تنسور کوانتیزه (معمولاً qint8 -> 1 بایت به‌ازای عنصر)
            total_bytes += w.numel() * w.element_size()
            total_bytes += 8  # overhead مقیاس/آفست per-tensor affine (~یک float32 + یک int32)
            b = module.bias()
            if b is not None:
                total_bytes += b.numel() * b.element_size()
        else:
            for p in module.parameters(recurse=False):
                total_bytes += p.numel() * p.element_size()
            for buf in module.buffers(recurse=False):
                total_bytes += buf.numel() * buf.element_size()

    return total_bytes / 1024


def calculate_complexity(model, dummy_shape=(1, 98, 40)):
    """FLOPs/MACs/Params معماری با thop — فقط برای مدل کوانتیزه‌نشده معتبر است."""
    dummy_input = torch.randn(*dummy_shape)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    macs = flops / 2
    flops_f, macs_f, params_f = clever_format([flops, macs, params], "%.3f")
    return flops_f, macs_f, params_f, flops, macs, params


def report_flops_correctly(flops_original, macs_original):
    """
    thop فقط hook برای انواع ماژول ثبت‌شده (nn.Linear، nn.Conv1d و ...) دارد؛
    torch.nn.quantized.dynamic.Linear نوع کاملاً متفاوتی است و thop آن را
    نمی‌شناسد (روی مدل کوانتیزه صدا زدنش عدد نزدیک صفر و غلط می‌دهد).
    چون کوانتیزاسیون تعداد عملیات را تغییر نمی‌دهد (فقط دقت عددی هر
    عملیات را از FP32 به INT8 می‌برد)، FLOPs/MACs معماری مدل اصلی را
    برای هر دو حالت گزارش می‌کنیم.
    """
    print(f"FLOPs (architecture-level, unchanged by quantization): {flops_original}")
    print(f"MACs  (architecture-level, unchanged by quantization): {macs_original}")


# =========================================================
# FAR / FRR روی مجموعه‌ی تست (utterance-level) — برای هر دو مدل قابل استفاده
# =========================================================

def evaluate_far_frr(model, test_loader, num_classes, label="model"):
    model.to("cpu")
    model.eval()

    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f"Building confusion matrix ({label})"):
            data = data.to("cpu").float()
            target = target.to("cpu").long()

            logits = model(data)
            preds = logits.argmax(dim=-1)

            for t, p in zip(target.view(-1), preds.view(-1)):
                t_i, p_i = int(t.item()), int(p.item())
                if 0 <= t_i < num_classes and 0 <= p_i < num_classes:
                    conf_mat[t_i, p_i] += 1

    total = conf_mat.sum()
    fars = np.zeros(num_classes, dtype=np.float64)
    frrs = np.zeros(num_classes, dtype=np.float64)

    for k in range(num_classes):
        tp = conf_mat[k, k]
        fn = conf_mat[k, :].sum() - tp
        fp = conf_mat[:, k].sum() - tp
        tn = total - tp - fn - fp

        denom_pos = tp + fn
        frrs[k] = fn / denom_pos if denom_pos > 0 else 0.0

        denom_neg = fp + tn
        fars[k] = fp / denom_neg if denom_neg > 0 else 0.0

    macro_far = float(fars.mean())
    macro_frr = float(frrs.mean())
    accuracy = float(np.trace(conf_mat)) / float(total) if total > 0 else 0.0

    print(f"\n🔎 FAR/FRR on test set ({label}):")
    print(f"Macro FAR: {macro_far:.6f}")
    print(f"Macro FRR: {macro_frr:.6f}")
    print(f"Overall accuracy: {accuracy:.6f}")

    return macro_far, macro_frr, accuracy


def measure_inference_time_real(model, real_data, num_runs=100):
    model.to("cpu")
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(real_data.to("cpu"))
    return (time.time() - start_time) / num_runs


def test_accuracy(model, data_loader):
    model.to("cpu")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Testing"):
            data, target = data.to("cpu"), target.to("cpu")
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    return correct / total


# =========================================================
# کوانتیزاسیون دینامیک
# =========================================================

def quantize_model(model, train_loader, num_batches=30):
    """کوانتیزاسیون دینامیک روی لایه‌های nn.Linear."""
    print(f"\nStarting dynamic quantization with {num_batches} calibration batches...")
    model = model.to("cpu")
    model.eval()

    qconfig = torch.quantization.QConfig(
        activation=torch.quantization.MinMaxObserver.with_args(
            quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine
        ),
        weight=torch.quantization.MinMaxObserver.with_args(
            quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        ),
    )

    print("Calibrating...")
    with tqdm(total=num_batches, desc="Calibration Progress") as pbar:
        for i, (inputs, _) in enumerate(train_loader):
            if i >= num_batches:
                break
            inputs = inputs.to("cpu")
            with torch.no_grad():
                model(inputs)
            pbar.update(1)

    print("Converting to quantized model...")
    model_quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear: qconfig}, dtype=torch.qint8, inplace=False
    )
    print("Quantization completed.")
    return model_quantized


# =========================================================
# Dataset wrapper
# =========================================================

class Dataset(torch.utils.data.Dataset):
    def __init__(self, part="train", steps=None, batch_size=512, version=1, preprocess="mfcc"):
        super().__init__()
        self.dataset = None
        self.datasetworkers = dict()
        self.batch_size = batch_size
        self.version = version
        self.preprocess = preprocess
        self.steps = steps
        self.part = part

    def __len__(self):
        if self.steps is not None:
            return self.steps
        worker_info = torch.utils.data.get_worker_info()
        if self.dataset is None:
            self.dataset = Datagen(batch_size=self.batch_size, version=self.version, preprocess=self.preprocess)
        if worker_info is not None:
            return self.datasetworkers[worker_info.id].dataLen(self.part)
        return self.dataset.dataLen(self.part)

    def __getitem__(self, index):
        worker_info = torch.utils.data.get_worker_info()
        dataset = self.dataset if worker_info is None else self.datasetworkers[worker_info.id]
        data, target = dataset.getData(self.part, index)
        target = target.astype(int)
        return data, target


def train_worker_init(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    wid = worker_info.id
    dataset.datasetworkers[wid] = Datagen(
        batch_size=dataset.batch_size, version=dataset.version, preprocess=dataset.preprocess
    )


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--lr", default=0.001, help="Learning rate of AdamW optimizer")
    parser.add_argument("--weight-decay", default=0.0005, help="Weight decay of AdamW optimizer")
    parser.add_argument("--warmup-epochs", default=10, type=int)
    # فلگ قدیمی --no-label-smooth منطق برعکس داشت (store_false → پیش‌فرض True →
    # در حالت پیش‌فرض cross-entropy ساده انتخاب می‌شد). حالا صریح و opt-in است؛
    # رفتار پیش‌فرض (بدون smoothing) همان رفتار عملی اجراهای قبلی است.
    parser.add_argument("--label-smooth", action="store_true",
                        help="Use label smoothing loss (default: plain cross-entropy)")
    parser.add_argument("--num-steps", default=23000, type=int)
    parser.add_argument("--version", default=1, type=int, choices=[1, 2, 3])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-workers", default=10, type=int)
    parser.add_argument("--calib-batches", default=30, type=int,
                        help="Number of calibration batches for dynamic quantization")
    parser.add_argument("--enable-pruning", action="store_true",
                        help="Apply L1 unstructured pruning (before quantization) with an accuracy-based criterion")
    parser.add_argument("--max-acc-drop", default=0.01, type=float,
                        help="Max acceptable validation accuracy drop (fraction, e.g. 0.01 = 1%%) when choosing the pruning amount. "
                             "The pruning amount is NOT chosen to hit any target parameter count.")
    parser.add_argument("--prune-amounts", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95",
                        help="Comma-separated candidate pruning amounts to sweep")

    # ------------------------------------------------------------------
    # ✅ اصلاح‌شده: ابعاد معماری اکنون واقعاً از خط فرمان قابل تنظیم‌اند.
    # قبلاً heads/dim/mlp_dim/layers/gqa_groups همه hardcode بودند (heads=8,
    # dim=48, mlp_dim=96, layers=12, gqa_groups=4) و آرگومان --num-heads
    # قدیمی هم اصلاً در ساخت مدل استفاده نمی‌شد. حالا هر ۵ تا از طریق
    # آرگومان قابل کنترل‌اند، پس برای بازتولید ستون‌های G8/G4/G2/G1 دیگر
    # نیازی به دست‌کاری دستی فایل نیست.
    # ------------------------------------------------------------------
    parser.add_argument("--gqa-groups", default=4, type=int,
                        help="Number of GQA groups (must evenly divide --heads)")
    parser.add_argument("--heads", default=8, type=int,
                        help="Number of attention heads (must evenly divide --dim)")
    parser.add_argument("--dim", default=48, type=int, help="Model embedding dimension")
    parser.add_argument("--mlp-dim", default=96, type=int, help="FFN hidden dimension")
    parser.add_argument("--layers", default=12, type=int, help="Number of encoder blocks (depth)")
    parser.add_argument("--share-layers", action="store_true",
                        help="ALBERT/Universal-Transformer style weight tying: reuse ONE EncoderBlock "
                             "instance for all --layers iterations instead of separate weights per layer. "
                             "Reduces stored parameters by roughly a factor of --layers; FLOPs/MACs are "
                             "unchanged since the same block still runs forward --layers times.")

    args = parser.parse_args()

    print("Running: python {}".format(sys.argv))

    batch_size = args.batch_size
    lr = float(args.lr)
    weight_decay = float(args.weight_decay)
    steps_per_epoch = 45 * (512 // batch_size)
    warmup_epochs = args.warmup_epochs
    epochs = 512
    num_steps = int(args.num_steps)
    version = args.version
    device = args.device

    if device == "cuda":
        num_workers = int(args.num_workers)
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    os.makedirs(os.path.join('results', args.experiment), exist_ok=True)

    num_classes = 12 if version != 3 else 35
    print("num_classes:", num_classes)

    # ---------------- model config ----------------
    # ✅ اصلاح‌شده: دیگر hardcode نیستند، از args می‌آیند.
    heads = args.heads
    dim = args.dim
    mlp_dim = args.mlp_dim
    layers = args.layers
    gqa_groups = args.gqa_groups
    img_x, img_y = 98, 40
    patch_x, patch_y = 1, 40

    print(f"Model config: dim={dim}, heads={heads}, mlp_dim={mlp_dim}, "
          f"layers={layers}, gqa_groups={gqa_groups}, share_layers={args.share_layers}")

    model = KWT(
        img_x=img_x, img_y=img_y, patch_x=patch_x, patch_y=patch_y,
        num_classes=num_classes, dim=dim, depth=layers, heads=heads,
        mlp_dim=mlp_dim, pool='mean', channels=1,
        dropout=0.1, emb_dropout=0.05,
        gqa_groups=gqa_groups, use_relative_pos=True, use_conv_encoder=True,
        share_layers=args.share_layers,
    )
    model.to(device)

    model_path = "model.pth"
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            print("⚠️ Checkpoint loading failed due to shape mismatch. Starting from scratch.\n", e)
    else:
        print("🔄 No pre-trained model found. Starting training from scratch.")

    original_params = count_parameters(model)
    print(f"\nOriginal Model Parameters (trainable, float32): {original_params:,}")

    # ------------------------------------------------------------------
    # بررسی صریح weight-tying در RelativePositionBias: چون همان ماژول به
    # همه‌ی ۱۲ بلوک پاس داده می‌شود، این چاپ تأیید می‌کند که model.parameters()
    # آن را فقط یک‌بار می‌شمارد (نه ۱۳ بار)، و اگر شک دارید، دستی هم تأیید کنید.
    # ------------------------------------------------------------------
    if model.rel_pos_bias is not None:
        tied_paths = [n for n, m in model.named_modules() if m is model.rel_pos_bias]
        rel_bias_numel = model.rel_pos_bias.rel_bias.numel()
        print(f"RelativePositionBias is tied across {len(tied_paths)} module paths "
              f"({rel_bias_numel:,} unique parameters, counted once in count_parameters()).")

    # ---------------- optimizer / scheduler ----------------
    paramlist = [
        param for name, param in model.named_parameters()
        if all(exclude not in name for exclude in ['bias', 'norm', 'g_feature', 'pos_embedding'])
    ]

    optimizer = optim.AdamW(paramlist, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=lr / 10
    )
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=cosine_scheduler
    )

    loss_fn = LabelSmoothingLoss(classes=num_classes, smoothing=0.1).to(device) \
        if args.label_smooth else F.cross_entropy

    # ---------------- data ----------------
    train_set = Dataset(part="train", steps=num_steps, batch_size=512, version=version, preprocess="mfcc")
    val_set = Dataset(part="val", batch_size=512, version=version, preprocess="mfcc")

    train_generator = torch.utils.data.DataLoader(
        train_set, num_workers=num_workers, batch_size=None,
        persistent_workers=True, worker_init_fn=train_worker_init, pin_memory=pin_memory
    )
    val_generator = torch.utils.data.DataLoader(val_set, num_workers=0, batch_size=None)
    next(iter(train_generator))  # init workers

    def get_likely_index(t):
        return t.argmax(dim=-1)

    def number_of_correct(pred, target):
        return pred.squeeze().eq(target).sum().item()

    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience, min_delta, counter = 15, 0.0005, 0

    def val(model, epoch, logfile=None):
        nonlocal best_val_acc, best_val_loss, counter
        model.eval()
        correct, count, val_count, val_loss = 0, 0, 0, 0.0
        for data, target in val_generator:
            data, target = data.float().to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            val_loss += loss_fn(output.squeeze(), target).item()
            count += 1
            val_count += target.shape[-1]
            pred = get_likely_index(output)
            correct += number_of_correct(pred, target)

        val_loss /= count
        val_accuracy = correct / val_count
        val_acc.append(val_accuracy)
        val_losses.append(val_loss)
        print(f"Epoch: {epoch}\tAccuracy: {correct}/{val_count} ({100.*val_accuracy:.2f}%)\tLoss: {val_loss}")
        if logfile is not None:
            logfile.write(f'{epoch},{train_losses[-1] if train_losses else 0},{val_loss},{val_accuracy}\n')
            logfile.flush()

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'val_acc': best_val_acc,
            }, os.path.join('results', args.experiment, 'best.pth'))

        stop_training = False
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            print(f"⏳ No improvement in val_loss. Counter: {counter}/{patience}")
            if counter >= patience:
                stop_training = True
        return stop_training

    # ---------------- training loop ----------------
    model.train()
    train_losses, val_losses, val_acc = [], [], []
    training_step = 1
    logfile = open(os.path.join('results', args.experiment, 'data.txt'), 'w')
    logfile.write('Epoch,Train Losses,Val Losses,Val Accuracy\n')

    for data, target in tqdm(train_generator):
        optimizer.zero_grad()
        data, target = data.float().to(device), target.to(device)

        output = model(data)
        loss = loss_fn(output.squeeze(), target)

        if torch.isnan(loss):
            print("❌ NaN detected in loss at step:", training_step)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())
        training_step += 1

        if training_step % steps_per_epoch == 0:
            for pg in optimizer.param_groups:
                print(f"📉 Epoch {training_step // steps_per_epoch} - LR: {pg['lr']}")
            stop_training = val(model, training_step // steps_per_epoch, logfile)
            model.train()
            scheduler.step()
            if stop_training:
                print("🛑 Early stopping triggered.")
                break

    logfile.close()

    torch.save({
        'epoch': training_step // steps_per_epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 'val_acc': val_acc,
    }, os.path.join('results', args.experiment, 'last.pth'))

    torch.save(model.state_dict(), "model.pth")
    print("✅ Model saved: model.pth")

    # =====================================================
    # پس از آموزش: کوانتیزاسیون + گزارش صادقانه‌ی همه‌ی معیارها
    # =====================================================
    torch.backends.quantized.engine = "fbgemm"
    print(f"Quantized Engine: {torch.backends.quantized.engine}")

    eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("model.pth", map_location=eval_device))
    model.eval()

    test_set = Dataset(part="test", batch_size=1, version=version, preprocess="mfcc")
    test_generator = torch.utils.data.DataLoader(test_set, num_workers=0, batch_size=None)
    real_sample = next(iter(test_generator))[0]

    # --- پرونینگ (اختیاری، قبل از کوانتیزاسیون، با معیار مستقل از عدد هدف) ---
    pruning_info = None
    model_for_quant = model
    if args.enable_pruning:
        nz_before, total_before = count_nonzero_dense(model)
        print(f"\nNon-zero dense parameters BEFORE pruning: {nz_before:,} / {total_before:,} "
              f"(sparsity: {100*(1 - nz_before/total_before):.4f}%)")

        amounts = [float(a) for a in args.prune_amounts.split(",")]
        pruned_model, chosen = sweep_pruning_amounts(
            model, val_generator, device, amounts, args.max_acc_drop,
            get_likely_index, number_of_correct,
        )
        chosen_amount, _, nz_after, total_after, val_acc_pruned, acc_drop = chosen

        # دقت تست مدل «فقط هرس‌شده» (float، قبل از کوانتیزاسیون) — بدون این عدد
        # نمی‌شود اثر pruning را از اثر کوانتیزاسیون جدا کرد.
        test_acc_pruned = test_accuracy(pruned_model, test_generator)

        pruning_info = {
            "amount": chosen_amount,
            "nonzero": nz_after,
            "total_dense": total_after,
            "val_acc": val_acc_pruned,
            "acc_drop_pct": acc_drop,
            "test_acc": test_acc_pruned,
            # پارامترهای غیرصفر «کل مدل»: هرس فقط وزن‌های Linear/Conv1d را صفر
            # می‌کند، پس تعداد صفرشده‌ها را از کل پارامترهای مدل کم می‌کنیم.
            "model_nonzero": original_params - (total_after - nz_after),
        }
        model_for_quant = pruned_model

    # --- کوانتیزاسیون (فقط یک‌بار، روی مدل هرس‌شده اگر پرونینگ فعال باشد) ---
    model_quantized = quantize_model(model_for_quant, train_generator, num_batches=args.calib_batches)
    model_quantized.to("cpu")
    torch.save(model_quantized, "model_quantized.pt")
    print("✅ Model quantized and saved: model_quantized.pt")

    # --- دقت روی مجموعه‌ی تست ---
    test_acc_original = test_accuracy(model, test_generator)
    test_acc_quantized = test_accuracy(model_quantized, test_generator)

    # --- FAR/FRR برای هر دو مدل ---
    far_orig, frr_orig, acc_orig_cm = evaluate_far_frr(model, test_generator, num_classes, label="original")
    far_quant, frr_quant, acc_quant_cm = evaluate_far_frr(model_quantized, test_generator, num_classes, label="quantized")

    # --- زمان استنتاج و speedup ---
    inference_time_original = measure_inference_time_real(model, real_sample)
    inference_time_quantized = measure_inference_time_real(model_quantized, real_sample)
    speedup = inference_time_original / inference_time_quantized

    # --- پارامترها (روش صحیح) ---
    quantized_params = count_parameters_quant_aware(model_quantized)
    real_sparsity = compute_real_sparsity(model_quantized)

    # --- حجم فایل روی دیسک (معیار اصلی فشرده‌سازی) ---
    # هر دو روش را محاسبه می‌کنیم: نسخه‌ی state_dict خام (که برای مدل‌های
    # با share_layers=True دچار باگ بازسازی تکراری وزن کوانتیزه‌شده است) و
    # نسخه‌ی دیدوپ‌شده (که برای هر دو حالت درست است). نسخه‌ی دیدوپ‌شده معیار
    # اصلی گزارش است.
    original_size_kb_raw = get_model_size_kb(model)
    quantized_size_kb_raw = get_model_size_kb(
        model_quantized.state_dict() if hasattr(model_quantized, 'state_dict') else model_quantized
    )
    original_size_kb = get_model_size_kb_dedup(model)
    quantized_size_kb = get_model_size_kb_dedup(model_quantized)
    size_reduction_pct = 100 * (1 - quantized_size_kb / original_size_kb)

    if args.share_layers:
        print(f"\n⚠️ share_layers=True: Comparing two model size measurement methods:")
        print(f"   Raw state_dict (may suffer from duplicate counting bug): "
            f"{original_size_kb_raw:.2f} KB -> {quantized_size_kb_raw:.2f} KB")
        print(f"   Deduplicated by module ID (valid, primary reporting metric):   "
            f"{original_size_kb:.2f} KB -> {quantized_size_kb:.2f} KB")

    # --- FLOPs/MACs (فقط معنادار برای معماری، نه برای دقت عددی) ---
    flops_o, macs_o, params_o, _, _, _ = calculate_complexity(model)
    report_flops_correctly(flops_o, macs_o)

    # =====================================================
    # خلاصه‌ی نهایی — همان چیزی که باید در Table 4/5/8/9 مقاله برود
    # =====================================================
    print("\n" + "=" * 60)
    print("FINAL SUMMARY (honest, quant-aware measurements)")
    print(f"Model config: dim={dim}, heads={heads}, mlp_dim={mlp_dim}, "
          f"layers={layers}, gqa_groups={gqa_groups}, share_layers={args.share_layers}")
    print("=" * 60)
    # وقتی pruning فعال است، ستون دوم در واقع «هرس‌شده + کوانتیزه» است چون
    # کوانتیزاسیون روی مدل هرس‌شده اعمال شده؛ برچسب باید همین را بگوید.
    quant_col_label = "Pruned+Quant" if pruning_info is not None else "Quantized"
    print(f"{'Metric':35s} {'Original':>15s} {quant_col_label:>15s}")
    print(f"{'Accuracy (confusion matrix)':35s} {acc_orig_cm:15.4f} {acc_quant_cm:15.4f}")
    print(f"{'Test accuracy':35s} {test_acc_original:15.4f} {test_acc_quantized:15.4f}")
    print(f"{'Macro FAR':35s} {far_orig:15.6f} {far_quant:15.6f}")
    print(f"{'Macro FRR':35s} {frr_orig:15.6f} {frr_quant:15.6f}")
    print(f"{'Parameters':35s} {original_params:15,d} {quantized_params:15,d}")
    print(f"{'File size (KB)':35s} {original_size_kb:15.2f} {quantized_size_kb:15.2f}")
    print(f"{'Inference time (sec/sample)':35s} {inference_time_original:15.6f} {inference_time_quantized:15.6f}")
    print("-" * 60)
    print(f"Speedup factor after quantization: {speedup:.2f}x")
    print(f"Real storage reduction (file size): {size_reduction_pct:.2f}%")
    print(f"Parameter count change: {original_params:,} -> {quantized_params:,} "
          f"({100*(1 - quantized_params/original_params):.2f}% — should be ~0%, quantization does not remove params)")
    if pruning_info is not None:
        print(f"Measured sparsity in quantized Linear weights: {real_sparsity*100:.4f}% "
              f"(reflects the L1 pruning applied BEFORE quantization, not quantization itself)")
    else:
        print(f"Measured sparsity in quantized Linear weights: {real_sparsity*100:.4f}% "
              f"(near-zero confirms quantization does NOT prune/zero weights)")
    print(f"FLOPs/MACs (architecture-level, identical before/after quantization): {flops_o} / {macs_o}")

    if pruning_info is not None:
        print("-" * 60)
        print("PRUNING (applied BEFORE quantization, amount chosen by accuracy criterion, "
              "NOT chosen to hit any target parameter count):")
        print(f"  Pruning amount selected: {pruning_info['amount']:.2f}")
        print(f"  Non-zero weights in prunable (Linear/Conv1d) layers: {pruning_info['nonzero']:,} "
              f"out of {pruning_info['total_dense']:,} "
              f"(sparsity: {100*(1 - pruning_info['nonzero']/pruning_info['total_dense']):.2f}%)")
        print(f"  Non-zero parameters in the WHOLE model: {pruning_info['model_nonzero']:,} "
              f"out of {original_params:,} total "
              f"(norms/biases/embeddings are not pruned)")
        print(f"  Validation accuracy after pruning: {pruning_info['val_acc']*100:.2f}% "
              f"(drop vs unpruned: {pruning_info['acc_drop_pct']:.2f} percentage points)")
        print(f"  TEST accuracy after pruning (float, before quantization): "
              f"{pruning_info['test_acc']*100:.2f}% "
              f"(vs unpruned original: {test_acc_original*100:.2f}%)")
        print("  NOTE: 'non-zero parameters' counts actual zeroed weights (real sparsity).")
        print("  It is NOT the same as 'total dense parameters' above, which stays the same")
        print("  shape/size regardless of pruning (unstructured pruning does not shrink tensors).")
        print("  Report BOTH numbers explicitly and label them accordingly in the paper —")
        print("  do not present the non-zero count as if it were the total stored parameter count.")

    print("=" * 60)
    print("NOTE FOR THE PAPER: report the (dense) parameter count as essentially unchanged")
    print("by quantization, and report storage compression via file size / bits-per-parameter.")
    if pruning_info is not None:
        print("If pruning is now part of the pipeline, update the paper text and reviewer")
        print("response accordingly: pruning is no longer 'future work', and the mechanism")
        print("behind any non-zero-parameter reduction is explicit L1 pruning, not quantization.")


if __name__ == '__main__':
    main()