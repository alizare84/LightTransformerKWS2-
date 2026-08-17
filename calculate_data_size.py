#!/usr/bin/env python3
"""
اسکریپت برای محاسبه تأثیر تغییرات پارامترهای feature extraction روی حجم داده‌ها
"""

import numpy as np

# پارامترهای فعلی (از datagen.py خطوط 44-51)
CURRENT_PARAMS = {
    'mel_upper_edge_hertz': 7600,
    'window_size_ms': 30.0,
    'window_stride_ms': 10.0,
    'mel_num_bins': 80,
    'dct_num_features': 40,
    'feature_type': 'mfcc_tf'
}

# مقادیر پیش‌فرض معمول (بر اساس استانداردهای رایج)
DEFAULT_PARAMS = {
    'mel_upper_edge_hertz': 4000,  # معمولاً 4000-8000 Hz
    'window_size_ms': 25.0,        # معمولاً 20-30 ms
    'window_stride_ms': 10.0,      # معمولاً 10 ms
    'mel_num_bins': 40,            # معمولاً 40-80
    'dct_num_features': 13,        # معمولاً 13 (MFCC استاندارد)
    'feature_type': 'mfcc_tf'
}

# پارامترهای صوتی
SAMPLE_RATE = 16000  # Hz
AUDIO_DURATION_SEC = 1.0  # فرض: هر نمونه صوتی 1 ثانیه است

def calculate_frames_per_sample(window_size_ms, window_stride_ms, audio_duration_sec, sample_rate):
    """
    محاسبه تعداد فریم‌های زمانی برای هر نمونه صوتی
    """
    # تعداد نمونه‌های صوتی
    num_samples = int(audio_duration_sec * sample_rate)
    
    # تعداد نمونه‌ها در هر پنجره
    window_size_samples = int((window_size_ms / 1000.0) * sample_rate)
    
    # تعداد نمونه‌ها در هر stride
    window_stride_samples = int((window_stride_ms / 1000.0) * sample_rate)
    
    # تعداد فریم‌ها
    num_frames = int((num_samples - window_size_samples) / window_stride_samples) + 1
    
    return num_frames

def calculate_features_per_sample(dct_num_features, num_frames):
    """
    محاسبه تعداد کل ویژگی‌ها برای هر نمونه صوتی
    """
    return num_frames * dct_num_features

def calculate_data_size(num_samples, features_per_sample, dtype_size=4):
    """
    محاسبه حجم داده به بایت
    dtype_size: اندازه هر float32 = 4 بایت
    """
    return num_samples * features_per_sample * dtype_size

def format_size(size_bytes):
    """فرمت کردن حجم به واحدهای خوانا"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

# محاسبه برای پیکربندی فعلی
current_frames = calculate_frames_per_sample(
    CURRENT_PARAMS['window_size_ms'],
    CURRENT_PARAMS['window_stride_ms'],
    AUDIO_DURATION_SEC,
    SAMPLE_RATE
)
current_features_per_sample = calculate_features_per_sample(
    CURRENT_PARAMS['dct_num_features'],
    current_frames
)

# محاسبه برای پیکربندی پیش‌فرض
default_frames = calculate_frames_per_sample(
    DEFAULT_PARAMS['window_size_ms'],
    DEFAULT_PARAMS['window_stride_ms'],
    AUDIO_DURATION_SEC,
    SAMPLE_RATE
)
default_features_per_sample = calculate_features_per_sample(
    DEFAULT_PARAMS['dct_num_features'],
    default_frames
)

# محاسبه برای 105 هزار نمونه (داده‌های شما)
YOUR_SAMPLES = 105000

# محاسبه برای مجموعه داده‌های مختلف
datasets = {
    'Your Dataset (105k samples)': YOUR_SAMPLES,
    'Training (typical ~80k samples)': 80000,
    'Validation (typical ~10k samples)': 10000,
    'Test (typical ~10k samples)': 10000,
    'Total (typical ~100k samples)': 100000
}

print("=" * 80)
print("تحلیل تأثیر تغییرات پارامترهای Feature Extraction روی حجم داده‌ها")
print("=" * 80)
print()

print("📊 پارامترهای فعلی:")
for key, value in CURRENT_PARAMS.items():
    print(f"  {key:25s} = {value}")
print()

print("📊 پارامترهای پیش‌فرض (معمول):")
for key, value in DEFAULT_PARAMS.items():
    print(f"  {key:25s} = {value}")
print()

print("=" * 80)
print("محاسبات برای هر نمونه صوتی (1 ثانیه):")
print("=" * 80)
print(f"تعداد فریم‌های زمانی (فعلی):     {current_frames}")
print(f"تعداد فریم‌های زمانی (پیش‌فرض):  {default_frames}")
print(f"ویژگی‌های هر فریم (فعلی):        {CURRENT_PARAMS['dct_num_features']}")
print(f"ویژگی‌های هر فریم (پیش‌فرض):     {DEFAULT_PARAMS['dct_num_features']}")
print()
print(f"کل ویژگی‌ها برای هر نمونه (فعلی):     {current_features_per_sample:,}")
print(f"کل ویژگی‌ها برای هر نمونه (پیش‌فرض):  {default_features_per_sample:,}")
print()

increase_factor = current_features_per_sample / default_features_per_sample
print(f"📈 ضریب افزایش: {increase_factor:.2f}x ({increase_factor * 100:.1f}%)")
print()

print("=" * 80)
print("حجم داده‌ها برای مجموعه‌های مختلف:")
print("=" * 80)
print(f"{'Dataset':<30} {'Default Size':<20} {'Current Size':<20} {'Increase':<15}")
print("-" * 80)

total_default = 0
total_current = 0

for dataset_name, num_samples in datasets.items():
    default_size = calculate_data_size(num_samples, default_features_per_sample)
    current_size = calculate_data_size(num_samples, current_features_per_sample)
    increase = current_size - default_size
    increase_pct = (increase / default_size) * 100
    
    total_default += default_size
    total_current += current_size
    
    print(f"{dataset_name:<30} {format_size(default_size):<20} {format_size(current_size):<20} {format_size(increase):<15} (+{increase_pct:.1f}%)")

print("-" * 80)
total_increase = total_current - total_default
total_increase_pct = (total_increase / total_default) * 100
print(f"{'TOTAL':<30} {format_size(total_default):<20} {format_size(total_current):<20} {format_size(total_increase):<15} (+{total_increase_pct:.1f}%)")
print()

print("=" * 80)
print("خلاصه:")
print("=" * 80)
print(f"• افزایش تعداد ویژگی‌ها: {increase_factor:.2f}x")
print(f"• افزایش حجم داده‌ها: {total_increase_pct:.1f}%")
print(f"• حجم اضافی برای 100k نمونه: {format_size(total_increase)}")
print()

# محاسبه دقیق برای 105 هزار نمونه
print("=" * 80)
print(f"📊 محاسبه دقیق برای {YOUR_SAMPLES:,} نمونه شما:")
print("=" * 80)
your_default_size = calculate_data_size(YOUR_SAMPLES, default_features_per_sample)
your_current_size = calculate_data_size(YOUR_SAMPLES, current_features_per_sample)
your_increase = your_current_size - your_default_size
your_increase_pct = (your_increase / your_default_size) * 100

print(f"حجم داده با پارامترهای پیش‌فرض:  {format_size(your_default_size)}")
print(f"حجم داده با پارامترهای فعلی:     {format_size(your_current_size)}")
print(f"افزایش حجم:                      {format_size(your_increase)} (+{your_increase_pct:.1f}%)")
print()
print(f"✅ تعداد نمونه‌ها: {YOUR_SAMPLES:,} (تغییر نکرده)")
print(f"📈 حجم داده‌ها: {format_size(your_default_size)} → {format_size(your_current_size)}")
print()

# محاسبه تأثیر mel_num_bins (اگر استفاده شود)
print("=" * 80)
print("نکات مهم:")
print("=" * 80)
print(f"• mel_num_bins از {DEFAULT_PARAMS['mel_num_bins']} به {CURRENT_PARAMS['mel_num_bins']} افزایش یافته")
print(f"  (افزایش {CURRENT_PARAMS['mel_num_bins'] / DEFAULT_PARAMS['mel_num_bins']:.2f}x)")
print(f"• dct_num_features از {DEFAULT_PARAMS['dct_num_features']} به {CURRENT_PARAMS['dct_num_features']} افزایش یافته")
print(f"  (افزایش {CURRENT_PARAMS['dct_num_features'] / DEFAULT_PARAMS['dct_num_features']:.2f}x)")
print(f"• window_size_ms از {DEFAULT_PARAMS['window_size_ms']}ms به {CURRENT_PARAMS['window_size_ms']}ms افزایش یافته")
print(f"  (افزایش {CURRENT_PARAMS['window_size_ms'] / DEFAULT_PARAMS['window_size_ms']:.2f}x)")
print(f"• mel_upper_edge_hertz از {DEFAULT_PARAMS['mel_upper_edge_hertz']}Hz به {CURRENT_PARAMS['mel_upper_edge_hertz']}Hz افزایش یافته")
print(f"  (افزایش {CURRENT_PARAMS['mel_upper_edge_hertz'] / DEFAULT_PARAMS['mel_upper_edge_hertz']:.2f}x)")
print()

