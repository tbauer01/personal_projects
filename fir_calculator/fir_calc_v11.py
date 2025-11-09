"""
FIR Filter Design Pipeline for Audio Room Correction
====================================================

This script designs linear-phase or minimum-phase FIR filters to correct measured speaker/room response
toward a target curve. It processes multi-channel systems with configurable smoothing,
compression, and high-frequency protection.

SIGNAL PROCESSING PIPELINE (in order applied):
----------------------------------------------

1. LOAD & ALIGN
   - Load measured frequency response (FRD) and target curve
   - Align target to measured response in lock region (lock_low to lock_high)
   - Compute raw scaling factor: target / measured (in linear domain)
   - Clip scaling to [scaling_clip_min, scaling_clip_max] to prevent extreme corrections

2. SMOOTHING (on scaling factor, before FIR design)
   - Convert to log-frequency grid for perceptually uniform smoothing
   - Apply two-level Gaussian smoothing:
     * sigma_lf: Heavy smoothing at low frequencies (more stable bass)
     * sigma_hf: Light smoothing at high frequencies (preserve detail)
   - Blend between smoothing levels from smooth_f_low → smooth_f_high
   - Purpose: Remove measurement noise and prevent FIR from chasing artifacts

3. COMPRESSION (optional, if compress=true) (broken) use HF taper
   - There is a known issue with this block in that enabling it leads to a cliff in the scaling
   - Apply power-law compression to scaling above compress_above_hz
   - scaling = scaling ^ compress_power (e.g., 0.5 = square root)
   - Purpose: Reduce correction strength at high frequencies (gentler on tweeters)

4. FIR GRID INTERPOLATION
   - Create log-spaced frequency grid (concentrates points at low frequencies)
   - Interpolate smoothed scaling to FIR design grid
   - Ensures low-frequency detail is preserved in final filter

5. HF TAPER (optional, if hf_taper_start is set)
   - Progressively blend scaling toward unity (1.0) from hf_taper_start → hf_taper_end
   - Uses raised-cosine window in log-frequency domain for smooth transition
   - Optional: Clamp deviation within [1/hf_max_dev, hf_max_dev] above taper start
   - Purpose: Prevent FIR from fighting room/speaker limitations at very high frequencies

6. FIR DESIGN
   - Use scipy.signal.firwin2 to design linear-phase FIR from frequency samples
   - Handle Type II constraint: force zero gain at Nyquist for even tap counts
   - Generates symmetric (linear phase) impulse response
   - Optionally convert to minimum phase using scipy.signal.minimum_phase

7. VALIDATION
   - Compute FIR frequency response via freqz
   - Compare actual vs intended gain at test frequencies
   - Compute predicted system response: measured × FIR

OUTPUT:
   - FIR coefficient files (.txt, .bin) for each channel
   - Diagnostic plots showing measured, target, predicted, and scaling curves
   - Console output with FIR accuracy metrics and group delay info
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin2, freqz, minimum_phase
from scipy.ndimage import gaussian_filter1d
import argparse
import json
import os
import shutil
from datetime import datetime


def parse_frd(path):
	freqs = []
	mag_db = []
	phase_deg = []
	with open(path, 'r') as f:
		for line in f:
			if line.strip() == '' or line.startswith('*') or line.startswith('#'):
				continue
			parts = line.strip().split()
			if len(parts) < 3:
				continue
			freqs.append(float(parts[0]))
			mag_db.append(float(parts[1]))
			phase_deg.append(float(parts[2]))
	freqs = np.array(freqs)
	mag_db = np.array(mag_db)
	phase_deg = np.array(phase_deg)
	H_f = 10**(mag_db/20) * np.exp(1j * np.deg2rad(phase_deg))
	return freqs, H_f


def parse_target_txt(path):
    """Parse a 2-column target file: freq(Hz) dB. Returns (freqs, mag_db)."""
    freqs = []
    mag_db = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip() == '' or line.startswith('*') or line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            freqs.append(float(parts[0]))
            mag_db.append(float(parts[1]))
    return np.array(freqs), np.array(mag_db)


def smooth_scaling_linear(freqs, s_lin,
                          f_low=200.0, f_high=8000.0,
                          sigma_lf=12, sigma_hf=3,
                          grid_points=None):
    """
    Smooth a linear-domain scaling curve s_lin(f) with heavy LF smoothing and light HF smoothing,
    blended across [f_low, f_high] on a log-frequency grid.

    - freqs: 1D array of Hz (monotonic increasing)
    - s_lin: 1D array of linear scaling (> 0)
    Returns: s_lin_smoothed sampled back at 'freqs'.
    """
    freqs = np.asarray(freqs)
    s_lin = np.asarray(s_lin)

    # Guard: need positive frequencies for log grid
    pos = freqs > 0
    if not np.any(pos):
        return s_lin.copy()

    fmin = max(float(np.min(freqs[pos])), 1.0)
    fmax = float(np.max(freqs))
    N = int(grid_points) if grid_points is not None else int(len(freqs))
    N = max(N, 16)

    f_log = np.logspace(np.log10(fmin), np.log10(fmax), N)

    # Interpolate to log grid
    s_log = np.interp(f_log, freqs, s_lin)

    # Two smoothing strengths
    s_lf = gaussian_filter1d(s_log, sigma=sigma_lf, mode='nearest')
    s_hf = gaussian_filter1d(s_log, sigma=sigma_hf, mode='nearest')

    # Blend mask b(f): 0 -> 1 from f_low to f_high using raised-cosine in log space
    b = np.zeros_like(f_log)
    below = f_log <= f_low
    above = f_log >= f_high
    mid = ~(below | above)
    b[below] = 0.0
    b[above] = 1.0
    if np.any(mid):
        t = (np.log10(f_log[mid]) - np.log10(f_low)) / (np.log10(f_high) - np.log10(f_low))
        b[mid] = 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))

    s_blend = (1.0 - b) * s_lf + b * s_hf

    # Resample back to original frequency sampling
    s_lin_smoothed = np.interp(freqs, f_log, s_blend)
    return s_lin_smoothed


def align_target_to_lock_region(meas_freqs, meas_mag_db, target_freqs, target_mag_db, lock_low, lock_high):
	target_interp = np.interp(meas_freqs, target_freqs, target_mag_db)
	mask = (meas_freqs >= lock_low) & (meas_freqs <= lock_high)
	offset = np.mean(meas_mag_db[mask] - target_interp[mask])
	aligned_target = target_interp + offset
	return aligned_target


def compute_fir(
    meas_freqs,
    meas_mag_db,
    target_mag_db,
    numtaps=512,
    sample_rate=48000,
    scaling_clip_min=0.1,
    scaling_clip_max=4.0,
    smooth_f_low=200.0,
    smooth_f_high=8000.0,
    smooth_sigma_lf=20,
    smooth_sigma_hf=16,
    compress=False,
    compress_above_hz=8000.0,
    compress_power=0.5,
    hf_taper_start=None,
    hf_taper_end=None,
    hf_max_dev=None,
    use_minimum_phase=False,
    maxtaps=None,
):
    # Minimum phase preserves tap count and magnitude with current settings
    original_numtaps = numtaps
    if use_minimum_phase:
        # Check maxtaps limit with original numtaps (no doubling needed)
        if maxtaps is not None and numtaps > maxtaps:
            raise SystemExit(f"ERROR: numtaps {numtaps} exceeds maxtaps limit {maxtaps}.")
        
        print(f"Minimum phase mode: using {numtaps} taps")
    
    meas_lin = 10**(meas_mag_db/20)
    target_lin = 10**(target_mag_db/20)
    scaling_factor = target_lin / np.maximum(meas_lin, 1e-12)
    scaling_factor = np.clip(scaling_factor, scaling_clip_min, scaling_clip_max)  # prevent wild swings

    # Smooth scaling in linear domain before interpolation to FIR grid
    smoothed_scaling_factor = smooth_scaling_linear(
        meas_freqs, scaling_factor,
        f_low=smooth_f_low,
        f_high=smooth_f_high,
        sigma_lf=smooth_sigma_lf,
        sigma_hf=smooth_sigma_hf,
        grid_points=None
    )


    # Create LOG-SPACED frequency grid from low freq to Nyquist to preserve LF detail
    fir_grid_N = numtaps * 4  # Fine grid for design
    eps = 1.0  # Start slightly above 0 Hz to enable log spacing
    fir_freqs = np.logspace(np.log10(eps), np.log10(sample_rate/2), fir_grid_N)
    # Prepend 0 Hz at the start for DC
    fir_freqs = np.concatenate([[0.0], fir_freqs])
    # Ensure last element is exactly Nyquist
    fir_freqs[-1] = sample_rate / 2
    fir_freqs_norm = fir_freqs / (sample_rate/2)  # Normalize to [0,1]

    # Optional: reduce HF influence by compressing towards unity above a threshold
    if compress:
        try:
            comp_freq = float(compress_above_hz)
            comp_pow = float(compress_power)
        except Exception:
            comp_freq = 8000.0
            comp_pow = 0.5
        hf_mask = meas_freqs >= comp_freq
        if np.any(hf_mask):
            smoothed_scaling_factor[hf_mask] = np.power(smoothed_scaling_factor[hf_mask], comp_pow)

    # Interpolate smoothed scaling to FIR grid
    fir_scaling = np.interp(fir_freqs, meas_freqs, smoothed_scaling_factor)

    # Optional high-frequency taper to unity and/or clamp deviation
    if hf_taper_start is not None and hf_taper_start > 0:
        start = float(hf_taper_start)
        end = float(hf_taper_end) if hf_taper_end is not None else (sample_rate / 2)
        end = max(end, start + 1.0)
        # Build raised-cosine blend b(f): 0 at <=start, 1 at >=end, in log-frequency domain
        b = np.zeros_like(fir_freqs)
        below = fir_freqs <= start
        above = fir_freqs >= end
        mid = ~(below | above)
        b[below] = 0.0
        b[above] = 1.0
        if np.any(mid):
            f_mid = np.maximum(fir_freqs[mid], 1.0)
            t = (np.log10(f_mid) - np.log10(start)) / (np.log10(end) - np.log10(start))
            t = np.clip(t, 0.0, 1.0)
            b[mid] = 0.5 - 0.5 * np.cos(np.pi * t)
        # Pull scaling progressively to unity
        fir_scaling = (1.0 - b) * fir_scaling + b * 1.0

        # Optionally limit remaining deviation above the taper start
        if hf_max_dev is not None and hf_max_dev > 1.0:
            mask = fir_freqs >= start
            lo = 1.0 / float(hf_max_dev)
            hi = float(hf_max_dev)
            fir_scaling[mask] = np.clip(fir_scaling[mask], lo, hi)

    # For even tap counts, ensure zero gain at Nyquist
    if numtaps % 2 == 0:
        fir_scaling[-1] = 0.0

    # Generate FIR
    fir = firwin2(numtaps, fir_freqs_norm, fir_scaling)
    
    # Convert to minimum phase if requested
    if use_minimum_phase:
        linear_taps = len(fir)
        
        # Pre-check linear phase filter for validity
        if np.any(np.isnan(fir)) or np.any(np.isinf(fir)):
            print(f"WARNING: Linear phase FIR has invalid coefficients, skipping minimum phase conversion")
        else:
            # Check target curve aggressiveness (heuristic for minimum phase stability)
            max_boost_db = np.max(target_mag_db)
            max_cut_db = np.min(target_mag_db)
            total_range_db = max_boost_db - max_cut_db
            
            # Try minimum phase conversion with error handling
            try:
                # Suppress specific numpy warnings during conversion
                with np.errstate(invalid='ignore'):
                    # Convert to minimum phase while preserving magnitude response
                    fir_minphase = minimum_phase(fir, method='homomorphic', half=False)
                
                # Check if conversion succeeded
                if np.any(np.isnan(fir_minphase)) or np.any(np.isinf(fir_minphase)):
                    print(f"WARNING: Minimum phase conversion failed (NaN/Inf), using linear phase")
                else:
                    # Success!
                    fir = fir_minphase
                    final_taps = len(fir)
                    
                    # Verify we're close to the target
                    if abs(final_taps - original_numtaps) > original_numtaps * 0.1:
                        print(f"WARNING: Final tap count {final_taps} differs significantly from target {original_numtaps}")
                        
            except Exception as e:
                print(f"WARNING: Minimum phase conversion failed with error: {e}")
                print(f"Using linear phase filter instead")
    
    # ========== DIAGNOSTIC: Check FIR accuracy vs intended scaling ==========
    # Check FIR coefficients for validity
    fir_stats = f"FIR stats: len={len(fir)}, min={np.min(fir):.6f}, max={np.max(fir):.6f}, nan_count={np.sum(np.isnan(fir))}"
    print(f"DEBUG: {fir_stats}")
    
    w_check, h_check = freqz(fir, worN=8192)
    w_check_hz = w_check * sample_rate / (2 * np.pi)  # Convert to Hz
    
    # Check for NaN in frequency response
    h_nan_count = np.sum(np.isnan(h_check))
    if h_nan_count > 0:
        print(f"DEBUG: Found {h_nan_count} NaN values in frequency response")
    
    actual_fir_db = 20*np.log10(np.maximum(np.abs(h_check), 1e-12))
    
    print(f"\n[FIR Diagnostic - {numtaps} taps @ {sample_rate/1000:.0f} kHz - {'Minimum' if use_minimum_phase else 'Linear'} Phase]")
    test_freqs = [100, 500, 1000, 2000, 3000, 5000, 7000, 10000, 15000]
    for f in test_freqs:
        if f > sample_rate / 2:
            continue
        idx_w = np.argmin(np.abs(w_check_hz - f))
        idx_fir = np.argmin(np.abs(fir_freqs - f))
        intended_db = 20*np.log10(np.maximum(fir_scaling[idx_fir], 1e-12))
        actual_db = actual_fir_db[idx_w]
        error_db = actual_db - intended_db
        print(f"  {f:5.0f} Hz: intended {intended_db:+6.2f} dB, actual {actual_db:+6.2f} dB, error {error_db:+6.2f} dB")
    print()
    # ========================================================================
    
    # Compute predicted response at measured frequencies
    # Use uniform grid then interpolate to measured freqs
    w_pred, h_pred = freqz(fir, worN=8192)
    w_pred_hz = w_pred * sample_rate / (2 * np.pi)
    
    # Interpolate FIR response to measured frequencies
    h_interp = np.interp(meas_freqs, w_pred_hz, h_pred)
    
    # Multiply measured magnitude by FIR response
    predicted_mag = meas_lin * np.abs(h_interp)
    predicted_mag_db = 20*np.log10(np.maximum(predicted_mag, 1e-12))

    return fir, scaling_factor, fir_scaling, fir_freqs, predicted_mag_db


def setup_output_dir(cfg, config_path):
    """Create timestamped output directory and copy input files."""
    # Base output dir is always 'output'
    base_output_dir = 'output'
    
    # Get prefix from config (default to 'fir' if not specified)
    prefix = cfg.get('prefix', 'fir')
    
    # Create timestamped subdirectory with prefix in name (newest first when sorted)
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    output_dir = os.path.join(base_output_dir, f"{timestamp}_{prefix}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy input files
    input_files = [
        config_path,  # config.json
        cfg['measured'],  # measured response
        cfg['target']    # target curve
    ]
    
    for file_path in input_files:
        if os.path.exists(file_path):
            shutil.copy2(file_path, output_dir)
            print(f"Copied {file_path} to {output_dir}")
        else:
            print(f"Warning: Could not copy {file_path} - file not found")
    
    return output_dir


def save_fir(fir, prefix):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    np.savetxt(f"{prefix}.txt", fir, fmt="%.8f")
    fir.astype(np.float32).tofile(f"{prefix}.bin")


def plot_all(freqs, meas_mag_db, target_db, predicted_db, scaling_factor, fir, prefix, fir_scaling=None, fir_freqs=None):
    plt.figure(figsize=(12,10))

    # 1️⃣ Measured vs Target vs Predicted
    plt.subplot(3,1,1)
    plt.semilogx(freqs, meas_mag_db, label="Measured")
    plt.semilogx(freqs, target_db, label="Target (aligned)")
    plt.semilogx(freqs, predicted_db, label="Predicted w/ FIR")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.ylim(0, 100)  # Set consistent y-axis limits
    plt.grid(True)
    plt.legend()
    plt.title("Measured, Target, and Predicted")

    # 2️⃣ Scaling Factor
    plt.subplot(3,1,2)
    plt.semilogx(freqs, scaling_factor, label="Scaling (measured freqs)")
    # Plot FIR-grid interpolated scaling if provided (skip DC/0 Hz point)
    if fir_scaling is not None and fir_freqs is not None:
        plt.semilogx(fir_freqs[1:], fir_scaling[1:], label="FIR scaling (FIR grid)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Scaling Factor")
    plt.xlim(freqs[0], freqs[-1])  # Match x-axis to measured data range
    plt.grid(True)
    plt.legend()
    plt.title("Scaling Factor vs Frequency")

    # 3️⃣ FIR Coefficients
    plt.subplot(3,1,3)
    plt.plot(fir)
    plt.xlabel("Tap Index")
    plt.ylabel("Amplitude")
    plt.title("FIR Coefficients")
    plt.tight_layout()

    plt.savefig(f"{prefix}_plots.png", dpi=200)
    plt.close()

# -----------------------------
# Main FIR generation routine
# -----------------------------
def generate_simple_fir(measured, target, cfg, crossover_freq=100.0):
    print("Generating FIR (using complex measured response)...")
    
    # Config parameters
    sample_rate = float(cfg.get('sample_rate', 48000.0))
    num_channels = int(cfg.get('num_channels', 4))
    numtaps = cfg.get('numtaps', [512]*num_channels)
    channel_names = cfg.get('channel_names', [f'Ch{i+1}' for i in range(num_channels)])
    prefix = cfg.get('prefix', 'global_eq')
    use_minimum_phase = cfg.get('minimum_phase', False)
    maxtaps = cfg.get('maxtaps', None)

    # Validate
    if len(numtaps) != num_channels:
        raise SystemExit(f"numtaps length {len(numtaps)} != num_channels {num_channels}")
    if len(channel_names) != num_channels:
        raise SystemExit(f"channel_names length {len(channel_names)} != num_channels {num_channels}")

    # Load target and measured
    target_freqs, target_mag_db = parse_target_txt(target)
    meas_freqs, H_meas = parse_frd(measured)
    
    # Convert target dB -> linear
    target_mag = 10**(target_mag_db/20)

    # Determine fine frequency grid for FIR design
    fir_grid_N = max(numtaps)*4
    fir_freqs = np.linspace(0, sample_rate/2, fir_grid_N)
    fir_mag = np.interp(fir_freqs, target_freqs, target_mag)

    predicted_mag_db = None
    group_delays_ms = []

    for ch_idx in range(num_channels):
        ch_name = channel_names[ch_idx]
        ch_original_taps = numtaps[ch_idx]
        ch_prefix = f"{prefix}_{ch_name}"

        # Minimum phase preserves tap count and magnitude
        ch_numtaps = ch_original_taps
        if use_minimum_phase:
            # Check maxtaps limit with original taps (no doubling needed)
            if maxtaps is not None and ch_numtaps > maxtaps:
                raise SystemExit(f"ERROR: Channel {ch_name} numtaps {ch_numtaps} exceeds maxtaps limit {maxtaps}.")
            
            print(f"Channel {ch_name}: Minimum phase mode using {ch_numtaps} taps")

        # Handle Type II FIR requirement for even taps
        fir_mag_copy = fir_mag.copy()
        if ch_numtaps % 2 == 0:
            fir_mag_copy[-1] = 0.0

        # Generate FIR (linear phase)
        fir = firwin2(ch_numtaps, fir_freqs/(sample_rate/2), fir_mag_copy)
        
        # Convert to minimum phase if requested
        if use_minimum_phase:
            linear_taps = len(fir)
            
            # Pre-check linear phase filter for validity
            if np.any(np.isnan(fir)) or np.any(np.isinf(fir)):
                print(f"WARNING: Channel {ch_name} linear phase FIR has invalid coefficients, skipping minimum phase conversion")
            else:
                # Check target curve aggressiveness (heuristic for minimum phase stability)
                max_boost_db = np.max(target_mag_db)
                max_cut_db = np.min(target_mag_db)
                total_range_db = max_boost_db - max_cut_db
                
                # Try minimum phase conversion with error handling
                try:
                    # Suppress specific numpy warnings during conversion
                    with np.errstate(invalid='ignore'):
                        # Convert to minimum phase while preserving magnitude response
                        fir_minphase = minimum_phase(fir, method='homomorphic', half=False)
                    
                    # Check if conversion succeeded
                    if np.any(np.isnan(fir_minphase)) or np.any(np.isinf(fir_minphase)):
                        print(f"WARNING: Channel {ch_name} minimum phase conversion failed (NaN/Inf), using linear phase")
                    else:
                        # Success!
                        fir = fir_minphase
                        final_taps = len(fir)
                        
                        # Verify we're close to the target
                        if abs(final_taps - ch_original_taps) > ch_original_taps * 0.1:
                            print(f"WARNING: Final tap count {final_taps} differs significantly from target {ch_original_taps}")
                            
                except Exception as e:
                    print(f"WARNING: Channel {ch_name} minimum phase conversion failed with error: {e}")
                    print(f"Using linear phase filter instead")
        
        save_fir(fir, ch_prefix)

        # Compute predicted response using complex measured FRD (first channel only)
        if ch_idx == 0:
            w, h = freqz(fir, worN=8192)        # older SciPy: w in radians/sample
            w_Hz = w * sample_rate / (2*np.pi)  # convert to Hz
            h_interp = np.interp(meas_freqs, w_Hz, h)
            predicted_H = H_meas * h_interp
            predicted_mag_db = 20*np.log10(np.maximum(np.abs(predicted_H), 1e-12))

        # Compute group delay (ms)
        if use_minimum_phase:
            # Minimum phase filters have much lower group delay, approximately half at low frequencies
            # but this varies with frequency. For simplicity, use a rough estimate.
            gd_ms = (ch_numtaps * 0.25) / sample_rate * 1000  # Rough estimate
            print(f"Channel {ch_name}: {ch_numtaps} taps (min-phase) -> approx group delay {gd_ms:.3f} ms")
        else:
            gd_ms = (ch_numtaps - 1) / (2*sample_rate) * 1000
            print(f"Channel {ch_name}: {ch_numtaps} taps (linear-phase) -> group delay {gd_ms:.3f} ms")
        group_delays_ms.append(gd_ms)

    # -----------------------------
    # Calculate delay adjustment for mains to align with longest FIR (subs)
    # -----------------------------
    max_delay = max(group_delays_ms)
    delay_adjust_ms = [max_delay - gd for gd in group_delays_ms]
    print("\nSuggested delay adjustment per channel to align at crossover:")
    for ch_name, d in zip(channel_names, delay_adjust_ms):
        print(f"  {ch_name}: {d:.3f} ms")

    # Plot measured vs predicted
    plt.figure(figsize=(12,6))
    plt.semilogx(meas_freqs, 20*np.log10(np.abs(H_meas)), label="Measured Input", color="k", linewidth=2)
    if predicted_mag_db is not None:
        plt.semilogx(meas_freqs, predicted_mag_db, label=f"Predicted (from {channel_names[0]})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Measured vs Predicted Response")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_predicted.png", dpi=200)
    # plt.show()


def generate_fitted_fir(measured, target, cfg):
    # Get config parameters
    lock_low = float(cfg.get('lock_low', 200.0))
    lock_high = float(cfg.get('lock_high', 400.0))
    sample_rate = float(cfg.get('sample_rate', 48000.0))
    num_channels = int(cfg.get('num_channels', 4))
    numtaps = cfg.get('numtaps', [1024] * num_channels)
    channel_names = cfg.get('channel_names', [f'Ch{i+1}' for i in range(num_channels)])
    prefix = cfg.get('prefix', 'global_eq')
    
    # Scaling clip parameters
    scaling_clip_min = float(cfg.get('scaling_clip_min', 0.1))
    scaling_clip_max = float(cfg.get('scaling_clip_max', 4.0))
    
    # Smoothing parameters
    smooth_f_low = float(cfg.get('smooth_f_low', 200.0))
    smooth_f_high = float(cfg.get('smooth_f_high', 8000.0))
    smooth_sigma_lf = float(cfg.get('smooth_sigma_lf', 20))
    smooth_sigma_hf = float(cfg.get('smooth_sigma_hf', 16))
    
    # Compression parameters
    compress = bool(cfg.get('compress', False))
    compress_above_hz = float(cfg.get('compress_above_hz', 8000.0))
    compress_power = float(cfg.get('compress_power', 0.5))
    
    # HF taper parameters
    hf_taper_enable = bool(cfg.get('hf_taper_enable', False))
    hf_taper_start = cfg.get('hf_taper_start', None) if hf_taper_enable else None
    hf_taper_end = cfg.get('hf_taper_end', None) if hf_taper_enable else None
    hf_max_dev = cfg.get('hf_max_dev', None) if hf_taper_enable else None
    
    # Phase type
    use_minimum_phase = bool(cfg.get('minimum_phase', False))
    
    # Maximum taps limit
    maxtaps = cfg.get('maxtaps', None)

    # Validate arrays match num_channels
    if len(numtaps) != num_channels:
        raise SystemExit(f"numtaps array length ({len(numtaps)}) must match num_channels ({num_channels})")
    if len(channel_names) != num_channels:
        raise SystemExit(f"channel_names array length ({len(channel_names)}) must match num_channels ({num_channels})")

    # Load measured FRD and target text file
    meas_freqs, H_meas = parse_frd(measured)
    target_freqs, target_mag_db = parse_target_txt(target)  # Target is already in dB

    # Validate data
    if len(meas_freqs) == 0:
        raise SystemExit(f"No data points found in measured FRD file: {measured}")
    if len(target_freqs) == 0:
        raise SystemExit(f"No data points found in target text file: {target}")

    print(f"Loaded {len(meas_freqs)} points from measured FRD")
    print(f"Loaded {len(target_freqs)} points from target curve")
    print(f"Target freq range: {target_freqs[0]:.1f} Hz to {target_freqs[-1]:.1f} Hz")
    print(f"Target magnitude range: {min(target_mag_db):.1f} dB to {max(target_mag_db):.1f} dB")

    # Convert measured to dB
    meas_mag_db = 20*np.log10(np.abs(H_meas))

    # Align target once for all channels
    aligned_target = align_target_to_lock_region(
        meas_freqs, meas_mag_db, target_freqs, target_mag_db, 
        lock_low, lock_high
    )

    # Process each channel and store results
    all_firs = []
    all_predicted_dbs = []
    all_scaling_factors = []
    all_fir_scalings = []
    all_fir_freqs = []
    group_delays_ms = []

    # Process each channel
    for ch_idx in range(num_channels):
        ch_name = channel_names[ch_idx]
        ch_numtaps = numtaps[ch_idx]
        ch_prefix = f"{prefix}_{ch_name}"

        # Compute FIR for this channel (now returns fir_scaling and fir_freqs too)
        fir, scaling_factor, fir_scaling, fir_freqs, predicted_db = compute_fir(
            meas_freqs,
            meas_mag_db,
            aligned_target,
            ch_numtaps,
            sample_rate,
            scaling_clip_min=scaling_clip_min,
            scaling_clip_max=scaling_clip_max,
            smooth_f_low=smooth_f_low,
            smooth_f_high=smooth_f_high,
            smooth_sigma_lf=smooth_sigma_lf,
            smooth_sigma_hf=smooth_sigma_hf,
            compress=compress,
            compress_above_hz=compress_above_hz,
            compress_power=compress_power,
            hf_taper_start=hf_taper_start,
            hf_taper_end=hf_taper_end,
            hf_max_dev=hf_max_dev,
            use_minimum_phase=use_minimum_phase,
            maxtaps=maxtaps,
        )

        # Save FIR and store results
        save_fir(fir, ch_prefix)
        all_firs.append(fir)
        all_predicted_dbs.append(predicted_db)
        all_scaling_factors.append(scaling_factor)
        all_fir_scalings.append(fir_scaling)
        all_fir_freqs.append(fir_freqs)

        # Compute group delay (ms)
        gd_ms = (ch_numtaps - 1) / (2*sample_rate) * 1000
        group_delays_ms.append(gd_ms)
        print(f"Channel {ch_name}: {ch_numtaps} taps -> group delay {gd_ms:.3f} ms")

    # Print delay adjustments
    max_delay = max(group_delays_ms)
    delay_adjust_ms = [max_delay - gd for gd in group_delays_ms]
    print("\nSuggested delay adjustment per channel to align:")
    for ch_name, d in zip(channel_names, delay_adjust_ms):
        print(f"  {ch_name}: {d:.3f} ms")


    # For each unique tap length, plot only the first channel with that tap length
    unique_taps = list(sorted(set(numtaps)))
    for tap_len in unique_taps:
        idx = next((i for i, n in enumerate(numtaps) if n == tap_len), None)
        if idx is not None:
            plot_all(
                meas_freqs,
                meas_mag_db,
                aligned_target,
                all_predicted_dbs[idx],
                all_scaling_factors[idx],
                all_firs[idx],
                f"{prefix}_taplen{tap_len}_{channel_names[idx]}",
                fir_scaling=all_fir_scalings[idx],
                fir_freqs=all_fir_freqs[idx]
            )


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Compute phase-aligned global EQ FIR from measured full system response.")
	parser.add_argument('--config', default=None, help='Path to JSON config file (defaults to config.json next to script)')
	parser.add_argument('--simple', action='store_true', help='Run in simple mode (behavior specified later)')
	args = parser.parse_args()

	# Resolve config path (default: config.json next to this script)
	if args.config:
		config_path = args.config
	else:
		config_path = os.path.join(os.path.dirname(__file__), 'config.json')

	if not os.path.exists(config_path):
		raise SystemExit(f"Config file not found: {config_path}")

	with open(config_path, 'r') as cf:
		try:
			cfg = json.load(cf)
		except json.JSONDecodeError as e:
			raise SystemExit(f"Error parsing config file '{config_path}': {e}")

	simple_mode = bool(args.simple)

	# Required fields
	measured = cfg.get('measured')
	target = cfg.get('target')
	if not measured:
		raise SystemExit("'measured' must be set in config.json")
	if not target:
		raise SystemExit("'target' must be set in config.json")

	# Setup output directory and copy input files
	output_dir = setup_output_dir(cfg, config_path)
	
	# Update prefix to use output directory
	if 'prefix' in cfg:
		cfg['prefix'] = os.path.join(output_dir, cfg['prefix'])
	else:
		cfg['prefix'] = os.path.join(output_dir, 'fir')

	if simple_mode:
		generate_simple_fir(measured, target, cfg)
	else:
		generate_fitted_fir(measured, target, cfg)
