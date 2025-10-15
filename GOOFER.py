import numpy as np
from numba import njit
import soundfile as sf
import os
import parselmouth

DSTORAGE = np.float16 # for files / big arrays
DCOMPUTE = np.float32 # for math
_W_CACHE = {}
_CACHE = {}

def get_cached_window(sr, n_fft):
    key = ("win", sr, n_fft)
    w = _CACHE.get(key)
    if w is None:
        w = np.hanning(n_fft).astype(np.float32) ** 0.5
        _CACHE[key] = w
    return w

def get_cached_freqs(sr, n_fft):
    key = ("freqs", sr, n_fft)
    f = _CACHE.get(key)
    if f is None:
        f = np.fft.rfftfreq(n_fft, 1.0 / sr).astype(np.float32).reshape(-1, 1)
        _CACHE[key] = f
    return f

def get_cached_boost(sr, n_fft):
    key = ("boost", sr, n_fft)
    b = _CACHE.get(key)
    if b is None:
        n_bins = n_fft // 2 + 1
        b = np.linspace(1, 100, n_bins, dtype=np.float32).reshape(-1, 1)
        _CACHE[key] = b
    return b

def get_cached_brightness(sr, n_fft):
    key = ("bright", sr, n_fft)
    curves = _CACHE.get(key)
    if curves is None:
        n_bins = n_fft // 2 + 1
        harm = create_brightness_curve(n_bins, sr, 2000, 3500, gain_db=3.0).astype(np.float32)
        brea = create_brightness_curve(n_bins, sr, 3500, 5000, gain_db=20.0).astype(np.float32)
        curves = (harm, brea)
        _CACHE[key] = curves
    return curves # (harm_curve, breath_curve)

def to_compute(x): return np.asarray(x, dtype=DCOMPUTE)

def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
def mel_to_hz(m):  return 700.0 * (10**(m / 2595.0) - 1.0)

def make_mel_knots(sr, n_fft, K):
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mel_min, mel_max = hz_to_mel(0.0), hz_to_mel(sr / 2.0)
    mel_knots = np.linspace(mel_min, mel_max, K, dtype=DCOMPUTE)
    hz_knots = mel_to_hz(mel_knots).astype(DCOMPUTE)
    return freqs.astype(DCOMPUTE), hz_knots

def precompute_interp_matrix(freqs_full, hz_knots):
    N = len(freqs_full); K = len(hz_knots)
    idx = np.searchsorted(hz_knots, freqs_full, side='right') - 1
    idx = np.clip(idx, 0, K - 2)
    x0 = hz_knots[idx]; x1 = hz_knots[idx + 1]
    w1 = (freqs_full - x0) / np.maximum(x1 - x0, 1e-12)
    w0 = 1.0 - w1
    W = np.zeros((N, K), dtype=DCOMPUTE)
    rows = np.arange(N)
    W[rows, idx]     = w0
    W[rows, idx + 1] = w1
    return W

def compress_env_to_knots(env_spec, sr, n_fft, eps=1e-2, K_start=32, K_step=16, K_max=192, smooth_sigma_bins=0.5):
    env = to_compute(env_spec)
    if smooth_sigma_bins > 0:
        env = gaussian_filter1d(env, sigma=smooth_sigma_bins, axis=0)
    log_env = np.log(np.maximum(env, 1e-8)).astype(DCOMPUTE)

    n_bins, T = log_env.shape
    freqs = np.fft.rfftfreq(n_fft, 1.0/sr).astype(DCOMPUTE)

    bin_resolution = sr / n_fft
    check_idx = np.linspace(0, T-1, min(256, T), dtype=int)
    env_check = env[:, check_idx]

    best = None
    K_values = np.arange(K_start, K_max + 1, K_step)
    for K in K_values:
        _, hz_knots = make_mel_knots(sr, n_fft, K)
        bin_idx = np.clip(np.round(hz_knots / bin_resolution).astype(int), 0, n_bins-1)
        knot_vals_log = log_env[bin_idx, :]

        W = precompute_interp_matrix(freqs, hz_knots)
        recon_log = W @ knot_vals_log[:, check_idx]
        rel_err = np.max(
            np.abs(np.exp(recon_log) - env_check) / (env_check + 1e-8)
        )

        if rel_err < eps:
            best = {
                "mode": "knots",
                "knot_vals_log": knot_vals_log.astype(DSTORAGE),
                "hz_knots": hz_knots.astype(DCOMPUTE),
                "n_bins": int(n_bins),
                "n_fft": int(n_fft),
                "sr": int(sr),
            }
            break

    if best is None:
        # fall back to max K
        _, hz_knots = make_mel_knots(sr, n_fft, K_max)
        bin_idx = np.clip(np.round(hz_knots / bin_resolution).astype(int), 0, n_bins-1)
        knot_vals_log = log_env[bin_idx, :]
        best = {
            "mode": "knots",
            "knot_vals_log": knot_vals_log.astype(DSTORAGE),
            "hz_knots": hz_knots.astype(DCOMPUTE),
            "n_bins": int(n_bins),
            "n_fft": int(n_fft),
            "sr": int(sr),
        }
    return best

def decode_env_from_knots(env_pack):
    assert env_pack["mode"] == "knots"
    knot_vals_log = np.asarray(env_pack["knot_vals_log"]).astype(DCOMPUTE)
    hz_knots = np.asarray(env_pack["hz_knots"]).astype(DCOMPUTE)
    n_fft = int(env_pack["n_fft"])
    sr = int(env_pack["sr"])
    n_bins = int(env_pack["n_bins"])

    key = (sr, n_fft, hz_knots.shape[0])
    W = _W_CACHE.get(key)
    if W is None:
        freqs = np.fft.rfftfreq(n_fft, 1.0/sr).astype(DCOMPUTE)
        W = precompute_interp_matrix(freqs, hz_knots).astype(DCOMPUTE)
        _W_CACHE[key] = W

    log_env = W @ knot_vals_log
    env = np.exp(log_env).astype(DCOMPUTE)
    if env.shape[0] != n_bins:
        env = env[:n_bins, :]
    return env

def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))

def interp1d(x, y, kind='linear', fill_value='extrapolate'):
    if kind != 'linear':
        raise ValueError("Only 'linear' interpolation is supported.") # im lazy, duh

    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) == 0:
        raise ValueError("x cannot be empty")
    
    if len(x) == 1:
        x0 = x[0]
        y0 = y[0]
        
        def interpolator(x_new):
            x_new = np.asarray(x_new)
            
            if fill_value == 'extrapolate':
                return np.full_like(x_new, y0, dtype=y.dtype)
            else:
                try:
                    fill_val = float(fill_value)
                    result = np.full_like(x_new, fill_val)
                    mask = np.isclose(x_new, x0)
                    result[mask] = y0
                    return result
                except (TypeError, ValueError):
                    raise ValueError("fill_value must be 'extrapolate' or a number")
        
        return interpolator

    slope_left = (y[1] - y[0]) / (x[1] - x[0] + 1e-10)
    slope_right = (y[-1] - y[-2]) / (x[-1] - x[-2] + 1e-10)

    def interpolator(x_new):
        x_new = np.asarray(x_new)
        
        if fill_value != 'extrapolate':
            try:
                fill_val = float(fill_value)

                within_bounds = (x_new >= x[0]) & (x_new <= x[-1])
                y_new = np.empty_like(x_new)
                
                if np.any(within_bounds):
                    y_new[within_bounds] = np.interp(x_new[within_bounds], x, y)
                
                y_new[~within_bounds] = fill_val
                return y_new
                
            except (TypeError, ValueError):
                raise ValueError("fill_value must be 'extrapolate' or a number")
        
        within_bounds = (x_new >= x[0]) & (x_new <= x[-1])
        y_new = np.interp(x_new, x, y)
        
        left_mask = x_new < x[0]
        if np.any(left_mask):
            y_new[left_mask] = y[0] + slope_left * (x_new[left_mask] - x[0])
            
        right_mask = x_new > x[-1]
        if np.any(right_mask):
            y_new[right_mask] = y[-1] + slope_right * (x_new[right_mask] - x[-1])
            
        return y_new

    return interpolator

def gaussian_filter1d(input_array, sigma, axis=-1, truncate=4.0):
    arr = np.asarray(input_array)
    if arr.size == 0 or arr.shape[axis] == 0 or sigma <= 0.0:
        return arr.copy()

    # calculate kernel
    radius = int(truncate * sigma + 0.5)
    if radius <= 0:
        return arr.copy()
    t = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()

    arr_moved = np.moveaxis(arr, axis, -1)

    padded = np.pad(arr_moved,
                    [(0, 0)] * (arr_moved.ndim - 1) + [(radius, radius)],
                    mode='reflect')

    out_moved = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='valid'), -1, padded)
    return np.moveaxis(out_moved, -1, axis) # gotta move axis back where it came from

def gaussian_filter(input_array, sigma):
    arr = np.asarray(input_array)
    if arr.ndim != 2:
        raise ValueError("gaussian_filter expects a 2D array.")

    # handle empties safely
    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return arr.copy()

    # allow (s0, s1) or single float
    if isinstance(sigma, (list, tuple)):
        if len(sigma) != 2:
            raise ValueError("sigma must be a float or a 2-tuple for 2D arrays.")
        s0, s1 = (max(float(s), 0.0) for s in sigma)
    else:
        s0 = s1 = max(float(sigma), 0.0)

    out = arr
    if s0 > 0.0:
        out = gaussian_filter1d(out, s0, axis=0)
    if s1 > 0.0:
        out = gaussian_filter1d(out, s1, axis=1)
    return out

def save_features(path, features, f0_interp, voicing_mask, formants, sr, y_len):
    with open(path, 'wb') as f:
        if isinstance(features, dict) and features.get("mode") == "knots":
            np.savez_compressed(
                f,
                mode=np.array(['knots']),
                knot_vals_log=features["knot_vals_log"],      # fp16
                hz_knots=features["hz_knots"],                # fp32
                n_bins=np.array([features["n_bins"]], dtype=np.int32),
                n_fft=np.array([features["n_fft"]], dtype=np.int32),
                env_sr=np.array([features["sr"]], dtype=np.int32),

                f0_interp=f0_interp.astype(DSTORAGE),
                voicing_mask=voicing_mask.astype(DSTORAGE),
                formants=formants,
                sr=np.array([sr], dtype=np.int32),
                y_len=np.array([y_len], dtype=np.int64),
            )
        else:
            env_spec = np.asarray(features, dtype=DSTORAGE)
            np.savez_compressed(
                f,
                mode=np.array(['full']),
                env_spec=env_spec,
                f0_interp=f0_interp.astype(DSTORAGE),
                voicing_mask=voicing_mask.astype(DSTORAGE),
                formants=formants,
                sr=np.array([sr], dtype=np.int32),
                y_len=np.array([y_len], dtype=np.int64),
                n_fft=np.array([env_spec.shape[0]*2-2], dtype=np.int32)
            )

def load_features(path):
    data = np.load(path, allow_pickle=True)
    mode = str(data["mode"][0])
    if mode == "knots":
        env_pack = {
            "mode": "knots",
            "knot_vals_log": data["knot_vals_log"],
            "hz_knots": data["hz_knots"],
            "n_bins": int(data["n_bins"][0]),
            "n_fft": int(data["n_fft"][0]),
            "sr": int(data["env_sr"][0]),
        }
        env_spec = env_pack # decoded later
    else:
        env_spec = to_compute(data["env_spec"])
    f0_interp = to_compute(data["f0_interp"])
    voicing_mask = to_compute(data["voicing_mask"])
    formants = data["formants"].item()
    sr = int(data["sr"][0])
    y_len = int(data["y_len"][0])
    return env_spec, f0_interp, voicing_mask, formants, sr, y_len

def f0_estimate(snd, fr_duration, f0_min=75, f0_max=950, voice_thresh=0.63, sil_thresh=0.01, voice_cost=0.01):

    #snd = parselmouth.Sound(snd, sr)
    pitch = snd.to_pitch(method=parselmouth.Sound.ToPitchMethod.AC,
                        time_step=fr_duration,
                        pitch_floor=f0_min,
                        pitch_ceiling=f0_max,
                        #voicing_threshold=voice_thresh,
                        #silence_threshold=sil_thresh,
                        #voiced_unvoiced_cost=voice_cost
                        )

    return pitch.selected_array['frequency']

def stft(x, n_fft=2048, hop_length=512, window=None):
    if window is None:
        window = np.hanning(n_fft) ** 0.5
    x = np.asarray(x, dtype=np.float32)
    pad = n_fft // 2
    x_padded = np.pad(x, pad, mode='reflect') if len(x) >= 2 else np.pad(x, pad, mode='edge')
    if len(x_padded) < n_fft:
        x_padded = np.pad(x_padded, (0, n_fft - len(x_padded)), mode='edge')
    num_frames = max(1, 1 + (len(x_padded) - n_fft) // hop_length)
    frames = np.lib.stride_tricks.as_strided(
        x_padded,
        shape=(n_fft, num_frames),
        strides=(x_padded.strides[0], hop_length * x_padded.strides[0])
    ).copy()
    frames *= window[:, None]
    return np.fft.rfft(frames, axis=0)

@njit
def _overlap_add(frames, window, hop_length, expected_len):
    n_frames = frames.shape[1]
    n_fft = frames.shape[0]
    y = np.zeros(expected_len, dtype=np.float32)
    win_sum = np.zeros(expected_len, dtype=np.float32)
    
    for i in range(n_frames):
        start = i * hop_length
        for j in range(n_fft):
            idx = start + j
            val = frames[j, i] * window[j]
            y[idx] += val
            win_sum[idx] += window[j] * window[j]
    
    for i in range(expected_len):
        if win_sum[i] > 1e-9:
            y[i] /= win_sum[i]
    return y

def istft(S, hop_length=512, window=None, length=None):
    n_fft = (S.shape[0] - 1) * 2
    if window is None:
        window = np.hanning(n_fft).astype(np.float32) ** 0.5
    else:
        window = np.asarray(window, dtype=np.float32)
    
    S = np.asarray(S, dtype=np.complex64)
    frames = np.fft.irfft(S, axis=0, n=n_fft).astype(np.float32)
    
    pad = n_fft // 2
    expected_len = n_fft + hop_length * (frames.shape[1] - 1)
    
    y = _overlap_add(frames, window, hop_length, expected_len)
    y = y[pad: expected_len - pad]
    
    if length is not None:
        if y.shape[0] < length:
            y = np.pad(y, (0, length - y.shape[0]), mode='constant')
        else:
            y = y[:length]
    return y

@njit
def fix_f0_gaps(f0_array, max_gap=4):
    f0_fixed = f0_array.copy()
    i = 0
    n = len(f0_fixed)
    while i < n:
        if f0_fixed[i] == 0.0:
            start = i
            while i < n and f0_fixed[i] == 0.0:
                i += 1
            end = i
            gap_len = end - start
            if start > 0 and end < n and gap_len <= max_gap:
                left_val = f0_fixed[start - 1]
                right_val = f0_fixed[end]
                for j in range(gap_len):
                    ratio = (j + 1) / (gap_len + 1)
                    f0_fixed[start + j] = left_val * (1 - ratio) + right_val * ratio
        else:
            i += 1
    return f0_fixed

def lf_model_pulse(T, Ra=0.01, Rg=1.47, Rk=0.34, sr=44100, smoothing=False):
    # Added ARX glottal pulse
    # s(t) = G(t) * i(t) + e(g) - Glottal Response * Impulse + Exogenous Residual
    T0_samples = int(round(sr * T)) # amount of samples for one period
    if T0_samples <= 3:  # just defining a minimum amount of pulses
        T0_samples = 3
    t = np.linspace(0, T, T0_samples, endpoint=False, dtype=np.float32)
    
    # LF model parameters (normalized to period T)
    Ta = Ra * T  # open phase
    Te = T  # entire period
    
    # Calculate Tp and Tc based on Rg and Rk
    Tp = Ta  # peak time (end of opening phase)
    Tc = Tp + Rk * (Te - Tp)  # return phase

    pulse = np.zeros(T0_samples, dtype=np.float32)

    mask1 = t < Tp # open phase rise
    if np.any(mask1):
        pulse[mask1] = np.sin(np.pi * t[mask1] / (2 * Tp)) ** 2

    mask2 = (t >= Tp) & (t < Tc) # return phase decay
    if np.any(mask2):
        tau = (t[mask2] - Tp) / (Tc - Tp)
        pulse[mask2] = np.exp(-Rg * tau) * np.cos(np.pi * tau / 2)

    if smoothing:
        pulse = _smooth_arx_pulse(pulse, T0_samples)

    max_val = np.max(np.abs(pulse))
    if max_val > 0:
        pulse /= max_val

    return pulse

@njit(fastmath=True, cache=True)
def pulse_train_numba(f0_interp, sr, Ra=0.02, Rg=1.7, Rk=0.8):
    f0 = f0_interp.astype(np.float32)
    N = f0.size
    pulse = np.zeros(N, dtype=np.float32)

    total_phase = 0.0
    next_k = 1.0
    last_valid_f0 = 160.0

    cache_T0 = np.zeros(5, dtype=np.int64)
    cache_len = 0
    cache_bank = np.zeros((5, 8192), dtype=np.float32)

    for i in range(N):
        f0i = f0[i]
        if f0i > 1e-6:
            last_valid_f0 = f0i
        total_phase += f0i / sr

        while total_phase >= next_k:
            T = 1.0 / max(last_valid_f0, 1e-6)
            T0 = int(round(sr * T))
            if T0 < 3:
                T0 = 3
            if T0 > 8192:
                T0 = 8192

            found = -1
            for c in range(cache_len):
                if cache_T0[c] == T0:
                    found = c
                    break

            if found == -1:
                buf = np.zeros(T0, dtype=np.float32)
                Ta = Ra * T; Te = T; Tp = Ta; Tc = Tp + Rk * (Te - Tp)
                j = 0
                while j < T0:
                    ti = (j * T) / T0
                    if ti < Tp:
                        buf[j] = np.sin(np.pi * ti / (2.0 * Tp + 1e-12)) ** 2
                    elif ti < Tc:
                        tau = (ti - Tp) / (Tc - Tp + 1e-12)
                        buf[j] = np.exp(-Rg * tau) * np.cos(np.pi * tau / 2.0)
                    else:
                        buf[j] = 0.0
                    j += 1
                m = 0.0
                for j in range(T0):
                    a = abs(buf[j])
                    if a > m:
                        m = a
                if m > 0.0:
                    for j in range(T0):
                        buf[j] /= m

                if cache_len < 5:
                    cache_T0[cache_len] = T0
                    for j in range(T0):
                        cache_bank[cache_len, j] = buf[j]
                    found = cache_len
                    cache_len += 1
                else:
                    cache_T0[0] = T0
                    for j in range(T0):
                        cache_bank[0, j] = buf[j]
                    found = 0

            T0_used = cache_T0[found]
            end = i + T0_used
            if end > N:
                end = N

            j = i; k = 0
            while j < end:
                pulse[j] += cache_bank[found, k]
                j += 1; k += 1

            next_k += 1.0

    return pulse

def smooth_mask_ds(mask, sigma=100, ds=4):
    if ds > 1:
        short = mask[::ds].astype(np.float32)
    else:
        short = mask.astype(np.float32)
    sig_short = max(1.0, sigma / max(1, ds))
    short_s = gaussian_filter1d(short, sigma=sig_short)
    if ds > 1:
        x_old = np.linspace(0.0, 1.0, num=short_s.size, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, num=mask.size, dtype=np.float32)
        f = interp1d(x_old, short_s, kind='linear', fill_value='extrapolate')
        return f(x_new).astype(np.float32)
    else:
        return short_s.astype(np.float32)

def _smooth_arx_pulse(pulse, T0_samples):
    smoothed = pulse.copy() # copy pulse
    
    if len(pulse) > 5:
        sigma = max(1, T0_samples // 20)  # adaptive smoothing
        if sigma > 0:
            smoothed = gaussian_filter1d(pulse, sigma=sigma) # gaussian smoothing
    
    closed_phase_start = int(T0_samples * 0.7)
    if closed_phase_start < len(smoothed):
        smoothed[closed_phase_start:] = 0.0
    
    return smoothed

def create_brightness_curve(n_bins, sr, start_hz=4000, end_hz=4500, gain_db=6.0):
    freqs = np.linspace(0, sr / 2, n_bins)
    gain = np.ones_like(freqs)

    # Gain curve: slowly rises
    start_idx = np.searchsorted(freqs, start_hz)
    end_idx = np.searchsorted(freqs, end_hz)
    rise = np.linspace(0, 1, end_idx - start_idx)
    gain[start_idx:end_idx] = 1 + rise * (10**(gain_db / 20) - 1)
    gain[end_idx:] = 10**(gain_db / 20)
    return gain[:, None]

def stretch_feature(feature, stretch, kind='linear'):
    if stretch == 1.0:
        return feature.copy()

    target_len = int(feature.shape[-1] * stretch)

    if feature.ndim == 1:
        x_old = np.linspace(0, 1, len(feature))
        x_new = np.linspace(0, 1, target_len)
        interp_func = interp1d(x_old, feature, kind=kind, fill_value='extrapolate')
        return interp_func(x_new)
    elif feature.ndim == 2:
        x_old = np.linspace(0, 1, feature.shape[1])
        x_new = np.linspace(0, 1, target_len)
        return np.stack([
            interp1d(x_old, row, kind=kind, fill_value='extrapolate')(x_new)
            for row in feature
        ], axis=0)
    else:
        raise ValueError('Only 1D or 2D features are supported.')

def shift_formants(env, shift_ratio, sr):
    n_bins, n_frames = env.shape
    freqs = np.linspace(0, sr / 2, n_bins)
    warped_freqs = np.clip(freqs / shift_ratio, 0, sr / 2)

    interp_env = np.zeros_like(env)
    for t in range(n_frames):
        interp_func = interp1d(freqs, env[:, t], kind='linear', fill_value='extrapolate')
        interp_env[:, t] = interp_func(warped_freqs)
    return interp_env

def match_env_frames(env, target_frames):
    if env.shape[1] > target_frames:
        return env[:, :target_frames]
    elif env.shape[1] < target_frames:
        pad_width = target_frames - env.shape[1]
        return np.pad(env, ((0, 0), (0, pad_width)), mode='edge')
    return env


def create_volume_jitter(length, sr, speed=6.0, strength=0.1, seed=None, vibrato=False):
    #acts more like a vibrato now
    if seed is not None:
        np.random.seed(seed)
    t = np.arange(length) / sr
    if vibrato:
        phase = np.random.uniform(0, 2*np.pi) if seed is not None else 0 # making vibrato sinusoid
        noise = np.sin(2 * np.pi * speed * t + phase)

        # add fade
        fade_samples = int(0.1 * sr)
        if fade_samples < length:
            fade_in = np.linspace(0, 1, fade_samples)
            noise[:fade_samples] *= fade_in
    else:
        noise = np.random.randn(len(t))
        noise = gaussian_filter1d(noise, sigma=sr / (speed * 6))
        noise /= np.max(np.abs(noise) + 1e-6)
        
    envelope = 1.0 + noise * strength
    
    return np.clip(envelope, 0.5, 1.5) if vibrato else envelope
    #return envelope

def apply_f0_jitter(f0_array, sr, speed=40.0, strength=0.04, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(0, len(f0_array) / sr, num=len(f0_array))
    noise = np.random.randn(len(t))
    noise = gaussian_filter1d(noise, sigma=sr / (speed * 6))
    noise /= np.max(np.abs(noise) + 1e-6)
    jitter = 1.0 + noise * strength
    return jitter

def _detect_pulse_events(f0_interp, sr, subharm_semitones, voicing_mask, last_f0_init=160.0):
    n = len(f0_interp)
    last_f0 = last_f0_init
    ratios = np.empty(len(subharm_semitones), dtype=np.float64)
    for idx in range(len(subharm_semitones)):
        ratios[idx] = 2.0 ** (subharm_semitones[idx] / 12.0)

    phase_tracker = np.zeros(len(ratios), dtype=np.float64)
    events = []

    for i in range(n):
        f0 = f0_interp[i]
        if voicing_mask[i] <= 0 or f0 <= 0:
            continue
        last_f0 = f0

        for j in range(len(ratios)):
            ratio = ratios[j]
            sub_f0 = last_f0 * ratio
            if sub_f0 < 1e-2:
                continue
            phase_tracker[j] += sub_f0 / sr
            if phase_tracker[j] >= 1.0:
                events.append((i, sub_f0, ratio))
                phase_tracker[j] -= 1.0

    return events

def add_subharms(f0_interp, sr, subharm_weight=0.5, subharm_semitones=-12, voicing_mask=None):
    f0_interp = np.asarray(f0_interp, dtype=np.float64)
    if voicing_mask is None:
        voicing_mask = (f0_interp > 0).astype(np.float64)
    else:
        voicing_mask = np.asarray(voicing_mask, dtype=np.float64)

    if not isinstance(subharm_semitones, (list, tuple, np.ndarray)):
        subharm_semitones = [subharm_semitones]
    subharm_semitones = np.array(subharm_semitones, dtype=np.float64)

    events = _detect_pulse_events(f0_interp, sr, subharm_semitones, voicing_mask)

    sub_pulse = np.zeros_like(f0_interp, dtype=np.float64)
    pulse_cache = {}

    for i, sub_f0, ratio in events:
        T = 1.0 / sub_f0
        cache_key = f'{sub_f0:.2f}_sub{ratio:.3f}'  # (T, Ra, Rg, Rk, sr)

        if cache_key not in pulse_cache:
            pulse_cache[cache_key] = lf_model_pulse(
                T, Ra=0.02, Rg=1.7, Rk=1, sr=sr, smoothing=False
            ).astype(np.float64)

        lf_pulse = pulse_cache[cache_key]
        start = i
        end = min(len(sub_pulse), start + len(lf_pulse))
        sub_pulse[start:end] += lf_pulse[:end - start]

    sub_pulse *= voicing_mask
    max_val = np.max(np.abs(sub_pulse))
    if max_val > 1e-6:
        sub_pulse /= max_val
    sub_pulse *= subharm_weight

    return sub_pulse

def add_multiple_subharms(f0_interp, sr, semitone_list=[-12, 12], weights=None, voicing_mask=None):
    if weights is None:
        weights = [1.0 / len(semitone_list)] * len(semitone_list)
    
    total = np.zeros_like(f0_interp)
    for semi, weight in zip(semitone_list, weights):
        total += add_subharms(f0_interp, sr, voicing_mask=voicing_mask,
                              subharm_weight=weight, subharm_semitones=semi)
    return total

def apply_subharm_vibrato(f0_interp, sr, vibrato_rate=6.0, vibrato_depth=0.1, vibrato_delay=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t = np.arange(len(f0_interp)) / sr
    
    phase = np.random.uniform(0, 2*np.pi) if seed else 0 # create sinusoid
    vibrato = np.sin(2 * np.pi * vibrato_rate * t + phase)# * subharm_weight
    
    fade_in_samples = int(vibrato_delay * sr) # fade
    fade_in = np.linspace(0, 1, fade_in_samples)
    if len(fade_in) < len(vibrato):
        vibrato[:fade_in_samples] *= fade_in
    
    # voiced only
    voiced = f0_interp > 0
    modulated_f0 = f0_interp.copy()
    modulated_f0[voiced] = modulated_f0[voiced] * (1 + vibrato[voiced] * vibrato_depth)
    
    return modulated_f0

def extract_formants(y, sr, hop_length, max_formants=5, target_frames=None):
    snd = parselmouth.Sound(y, sr)
    formant_obj = snd.to_formant_burg(time_step=hop_length / sr, max_number_of_formants=max_formants)
    num_frames = formant_obj.get_number_of_frames()
    formant_tracks = {i: [] for i in range(1, max_formants + 1)}

    for frame_idx in range(num_frames):
        t = formant_obj.get_time_from_frame_number(frame_idx + 1)
        for formant_num in range(1, max_formants + 1):
            try:
                f = formant_obj.get_value_at_time(formant_num, t)
                formant_tracks[formant_num].append(f if f is not None else 0.0)
            except:
                formant_tracks[formant_num].append(0.0)

    # Pad to match env_spec frames
    if target_frames is not None:
        for i in formant_tracks:
            diff = target_frames - len(formant_tracks[i])
            if diff > 0:
                formant_tracks[i].extend([0.0] * diff)
            else:
                formant_tracks[i] = formant_tracks[i][:target_frames]

    return formant_tracks

def transpose_formants(formant_tracks, shift_ratios):
    """
    old_formant_tracks: dictionary of formant tracks, 
    where keys are formant numbers and values are arrays of formant values
    """
    transposed = {}
    for i, track in formant_tracks.items():
        ratio = shift_ratios.get(i, 1.0)
        transposed[i] = np.array(track) * ratio
    return transposed

def transpose_formants_array(formant_array, shift_ratios):
    """
    formant_array: (4, T) array, where row 0=F1, 1=F2, 2=F3, 3=F4
    shift_ratios: list or array of 4 ratios [r1, r2, r3, r4]
    Returns: (4, T) array, transposed formants
    """
    shift_ratios = np.asarray(shift_ratios, dtype=np.float64)  # shape (4,)
    return formant_array * shift_ratios[:, None]  # broadcasting: (4,1) * (4,T)

#@njit
def _interp_warp_env_by_formants(x, y, x_new):

    x = np.asarray(x)
    y = np.asarray(y)


    slope_left = (y[1] - y[0]) / (x[1] - x[0] + 1e-10)
    slope_right = (y[-1] - y[-2]) / (x[-1] - x[-2] + 1e-10)


    x_new = np.asarray(x_new)
    
    y_new = np.interp(x_new, x, y)
    
    left_mask = x_new < x[0]
    if np.any(left_mask):
        y_new[left_mask] = y[0] + slope_left * (x_new[left_mask] - x[0])
        
    right_mask = x_new > x[-1]
    if np.any(right_mask):
        y_new[right_mask] = y[-1] + slope_right * (x_new[right_mask] - x[-1])
        
    return y_new
   
#@njit
def warp_env_by_formants(env, orig_formants, shifted_formants, sr):
    n_bins, n_frames = env.shape
    freqs = np.linspace(0.0, sr / 2.0, n_bins)
    warped_env = np.zeros_like(env)

    max_pts = 6
    src_buf = np.empty(max_pts, dtype=np.float64)
    dst_buf = np.empty(max_pts, dtype=np.float64)

    for t in range(n_frames):
        k = 0
        src_buf[k] = 0.0
        dst_buf[k] = 0.0
        k += 1

        for i in range(1, 5):
            f_orig = orig_formants[i - 1, t]
            f_shifted = shifted_formants[i - 1, t]
            if f_orig > 50.0 and f_orig < sr / 2.0 and f_shifted > 50.0:
                src_buf[k] = f_orig
                dst_buf[k] = f_shifted
                k += 1

        src_buf[k] = sr / 2.0
        dst_buf[k] = sr / 2.0
        k += 1

        src_pts = src_buf[:k]
        dst_pts = dst_buf[:k]

        warped_freqs = _interp_warp_env_by_formants(dst_pts, src_pts, freqs)
        env_col = env[:, t]
        warped_col = _interp_warp_env_by_formants(freqs, env_col, warped_freqs)
        warped_env[:, t] = warped_col

    return warped_env

@njit
def one_pole_highpass(x, sr, fc):
    if fc <= 0: 
        return np.zeros_like(x)
    rc = 1.0 / (2.0 * np.pi * fc)
    a = rc / (rc + 1.0 / sr)
    y = np.zeros_like(x, dtype=np.float32)
    prev_x = 0.0
    prev_y = 0.0
    for i in range(len(x)):
        xn = float(x[i])
        yn = a * (prev_y + xn - prev_x)
        y[i] = yn
        prev_x = xn
        prev_y = yn
    return y

def make_smooth_noise(length, sr, smooth_ms=120.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = np.random.randn(length).astype(np.float32)
    sigma = max(1.0, (smooth_ms * 0.001 * sr) / 6.0)
    return gaussian_filter1d(n, sigma=sigma)

def apply_vocal_roughness(y, f0_interp, voicing_mask, sr,
                          k_list=(2, 3, 4),
                          h_list=None,
                          alpha=0.6,
                          hp_fc=300.0,
                          noise_amp=0.6,
                          noise_smooth_ms=120.0,
                          alpha_slew_ms=120.0):
    y = np.asarray(y, dtype=np.float32)
    f0 = np.asarray(f0_interp, dtype=np.float32)
    vmask = np.asarray(voicing_mask, dtype=np.float32)

    N = len(y)
    if h_list is None:
        h_list = [0.45, 0.28, 0.18][:len(k_list)]
        if len(h_list) < len(k_list):
            extra = len(k_list) - len(h_list)
            h_list += [h_list[-1] * 0.6**i for i in range(1, extra+1)]

    mod_sum = np.zeros(N, dtype=np.float32)

    for idx, (k, hk) in enumerate(zip(k_list, h_list)):
        nz = make_smooth_noise(N, sr, noise_smooth_ms, seed=(1337 + idx))
        f_mod = (f0 / float(k)) * (1.0 + noise_amp * nz)
        f_mod = np.maximum(f_mod, 0.0) * vmask
        phase = 2.0 * np.pi * np.cumsum(f_mod) / float(sr)
        mod_sum += hk * np.cos(phase).astype(np.float32)

    y_mod = y * (1.0 + mod_sum)
    y_sub = y_mod - y

    y_sub_hp = one_pole_highpass(y_sub, sr, hp_fc)

    alpha_track = alpha * vmask
    sigma = max(1.0, (alpha_slew_ms * 0.001 * sr) / 6.0)
    alpha_slewed = gaussian_filter1d(alpha_track, sigma=sigma).astype(np.float32)

    return y + alpha_slewed * y_sub_hp

def extract_features(y, sr, n_fft=1024, hop_length=256,
                     f0_min=75, f0_max=600, f0_merge_range=2):
    window = get_cached_window(sr, n_fft)
    voicing_threshold = f0_min
    S_orig = stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    mag = np.abs(S_orig) + 1e-8
    env_spec = gaussian_filter1d(mag, sigma=2.0, axis=0)
    #blurred = gaussian_filter1d(env_spec, sigma=1.0, axis=1)
    #alpha = 1.0
    #env_spec = env_spec + alpha * (env_spec - blurred)

    n_frames = env_spec.shape[1]
    formants = extract_formants(y, sr, hop_length, target_frames=n_frames)

    snd = parselmouth.Sound(y, sr)
    frame_duration = hop_length / sr
    pitch = f0_estimate(snd=snd, fr_duration=frame_duration)
    f0_track = np.nan_to_num(pitch)
    f0_track = fix_f0_gaps(f0_track, f0_merge_range)

    times_f0 = np.linspace(0, len(y)/sr, num=len(f0_track))
    interp_func = interp1d(times_f0, f0_track, kind='linear', fill_value=0)
    times_samples = np.linspace(0, len(y)/sr, num=len(y))
    f0_interp = interp_func(times_samples)
    f0_interp = np.clip(f0_interp, 1e-5, 2000)

    voicing_mask = (f0_interp > voicing_threshold).astype(float)

    env_knots = compress_env_to_knots(env_spec, sr=sr, n_fft=n_fft, eps=1e-2, K_start=32, K_step=16, K_max=192)
    return env_spec, f0_interp, voicing_mask, formants, env_knots

def synthesize(env_spec, f0_interp, voicing_mask,
               y, sr, n_fft=1024, hop_length=256, glottal_smoothing=False,
               stretch_factor=1.0, start_sec=None, end_sec=None,
               apply_brightness=True, normalize=1.0,
               uv_strength=0.75, breath_strength=0.1, noise_transition_smoothness=100,
               pitch_shift=1.0, formant_shift=1.0,
               f0_jitter=False, f0_jitter_speed=100, f0_jitter_strength=1.5,
               volume_jitter=False, volume_vibrato=False, volume_jitter_speed=150, volume_jitter_strength_harm=50, volume_jitter_strength_breath=100,
               add_subharm=False, subharm_semitones=-12, subharm_weight=0.5, subharm_vibrato=False,
               cut_subharm_below_f0=True, subharm_vibrato_rate=6.0, subharm_vibrato_depth=0.1, subharm_f0_jitter=0, subharm_vibrato_delay=0.1,
               F1_shift=1.0, F2_shift=1.0, F3_shift=1.0, F4_shift=1.0, formants=None,
               roughness_on=False, rough_k_list=(2,3,4), rough_h_list=None, rough_alpha=0.6,
               rough_hp_fc=320.0, rough_noise_amp=0.6, rough_noise_smooth_ms=120.0, rough_alpha_slew_ms=120.0):
    window = get_cached_window(sr, n_fft)

    if isinstance(env_spec, dict) and env_spec.get("mode") == "knots":
        env_spec = decode_env_from_knots(env_spec)
    env_spec = to_compute(env_spec)
    f0_interp = to_compute(f0_interp)
    voicing_mask = to_compute(voicing_mask)
    y = to_compute(y)

    env_spec4breathiness = gaussian_filter1d(env_spec, sigma=1.75, axis=0)

    f0_interp *= pitch_shift

    n_frames = env_spec.shape[1]

    formants_array = np.stack([
        np.asarray(formants[i], dtype=np.float64) for i in (1, 2, 3, 4)
    ], axis=0)  # shape: (4, n_frames)

    if any(shift != 1.0 for shift in [F1_shift, F2_shift, F3_shift, F4_shift]):
        shift_ratios = [F1_shift, F2_shift, F3_shift, F4_shift]
        
        shifted_formants_array = transpose_formants_array(formants_array, shift_ratios)
        
        env_spec = warp_env_by_formants(
            env_spec, 
            formants_array, 
            shifted_formants_array, 
            sr
        )

    if formant_shift != 1.0:
        env_spec = shift_formants(env_spec, formant_shift, sr)

    if stretch_factor != 1.0:
        if start_sec is not None and end_sec is not None:
            start_idx = int(start_sec * sr)
            end_idx = int(end_sec * sr)

            # 1D arrays
            f0_interp = np.concatenate([
                f0_interp[:start_idx],
                stretch_feature(f0_interp[start_idx:end_idx], stretch_factor),
                f0_interp[end_idx:]
            ])

            voicing_mask = np.concatenate([
                voicing_mask[:start_idx],
                stretch_feature(voicing_mask[start_idx:end_idx], stretch_factor, kind='linear'),
                voicing_mask[end_idx:]
            ])

            # 2D arrays: convert to frame index
            start_frame = int((start_sec * sr) / hop_length)
            end_frame = int((end_sec * sr) / hop_length)

            env_spec = np.concatenate([
                env_spec[:, :start_frame],
                stretch_feature(env_spec[:, start_frame:end_frame], stretch_factor),
                env_spec[:, end_frame:]
            ], axis=1)

            env_spec4breathiness = np.concatenate([
                env_spec4breathiness[:, :start_frame],
                stretch_feature(env_spec4breathiness[:, start_frame:end_frame], stretch_factor),
                env_spec4breathiness[:, end_frame:]
            ], axis=1)
        else:
            f0_interp = stretch_feature(f0_interp, stretch_factor)
            env_spec = stretch_feature(env_spec, stretch_factor)
            voicing_mask = stretch_feature(voicing_mask, stretch_factor, kind='linear')
            env_spec4breathiness = stretch_feature(env_spec4breathiness, stretch_factor)

        new_len = len(f0_interp)

        if stretch_factor > 1.0:
            pad_len = new_len - len(y)
            if len(y) == 0:
                y = np.zeros(new_len, dtype=np.float32)
            else:
                y = np.pad(y, (0, pad_len), mode='edge')
        else:
            y = y[:new_len]

    if f0_jitter:
        f0_jitter = apply_f0_jitter(f0_interp, sr, speed=f0_jitter_speed, strength=f0_jitter_strength)
        f0_interp *= 1.0 + ((f0_jitter - 1.0) * voicing_mask)

    # ARX-LF glottal pulse train generator
    pulse = pulse_train_numba(f0_interp.astype(np.float32), sr, Ra=0.02, Rg=1.7, Rk=0.8).astype(np.float32)

    if add_subharm:
        f0_for_subharms = f0_interp
        if subharm_f0_jitter > 0.0:
            subharm_jitter = apply_f0_jitter(f0_for_subharms, sr, speed=f0_jitter_speed, strength=subharm_f0_jitter)
            f0_for_subharms *= 1.0 + ((subharm_jitter - 1.0) * voicing_mask)

        if subharm_vibrato:
            f0_for_subharms = apply_subharm_vibrato(
                f0_for_subharms, sr,
                vibrato_rate=subharm_vibrato_rate,
                vibrato_depth=subharm_vibrato_depth,
                vibrato_delay=subharm_vibrato_delay
            )
        else:
            f0_for_subharms = f0_for_subharms
            
        sub_pulse = add_subharms(
            f0_for_subharms, sr, voicing_mask=voicing_mask,
            subharm_weight=subharm_weight,
            subharm_semitones=subharm_semitones
        )
        pulse += sub_pulse

    S_harm = stft(pulse, n_fft=n_fft, hop_length=hop_length, window=window)

    # dynamic highpass based on F0 for better breathiness ig
    freqs = get_cached_freqs(sr, n_fft)
    n_frames_harm = S_harm.shape[1]
    f0_env_frames = f0_interp[::hop_length]
    f0_env_frames = np.pad(f0_env_frames, (0, n_frames_harm - len(f0_env_frames)), mode='edge')
    f0_env_frames = f0_env_frames[:n_frames_harm]
    cutoff = f0_env_frames.reshape(1, -1)

    # Smooth sigmoid mask per frame
    sharpness = 5
    highpass_mask = 1.0 / (1.0 + np.exp(-np.clip((freqs - cutoff) / sharpness, -60, 60)))

    if cut_subharm_below_f0:
        S_harm *= highpass_mask
    if env_spec.shape[1] > S_harm.shape[1]:
        env_spec = env_spec[:, :S_harm.shape[1]]
    elif env_spec.shape[1] < S_harm.shape[1]:
        pad_width = S_harm.shape[1] - env_spec.shape[1]
        env_spec = np.pad(env_spec, ((0, 0), (0, pad_width)), mode='edge')
    #log_time('    STFT')
    mag_harm = np.max(np.abs(S_harm) + 1e-8)
    freq_bins = S_harm.shape[0]
    boost_curve = get_cached_boost(sr, n_fft)
    bright_harm, bright_breath = get_cached_brightness(sr, n_fft)

    env_spec_4harm = env_spec

    S_harm = (S_harm / mag_harm) * env_spec_4harm
    S_harm *= boost_curve

    if apply_brightness:
        voiced_frames = voicing_mask[::hop_length]
        if voiced_frames.size < S_harm.shape[1]:
            voiced_frames = np.pad(voiced_frames, (0, S_harm.shape[1] - voiced_frames.size), mode='edge')
        else:
            voiced_frames = voiced_frames[:S_harm.shape[1]]

        harm_voiced = S_harm[:, :voiced_frames.size].copy()
        mask_v = (voiced_frames > 0)
        if np.any(mask_v):
            cols = np.nonzero(mask_v)[0]
            harm_voiced[:, cols] *= bright_harm
            harm_voiced[:, cols] = gaussian_filter(harm_voiced[:, cols], sigma=(0.5, 0))
        S_harm[:, :voiced_frames.size] = harm_voiced

    harmonic = istft(S_harm, hop_length=hop_length, window=window, length=len(y))

    env_noise = match_env_frames(env_spec4breathiness, S_harm.shape[1]).astype(np.float32)
    n_bins, T_frames = env_noise.shape

    rng = np.random.default_rng()
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(n_bins, T_frames)).astype(np.float32)
    U = np.cos(phi) + 1j * np.sin(phi)

    # Unvoiced: use full-band envelope; Breath: apply high-pass mask
    S_uv = U * env_noise
    S_breath = (U * env_noise) * highpass_mask

    if apply_brightness:
        voiced_frames = voicing_mask[::hop_length]
        if voiced_frames.size < S_breath.shape[1]:
            voiced_frames = np.pad(voiced_frames, (0, S_breath.shape[1] - voiced_frames.size), mode='edge')
        else:
            voiced_frames = voiced_frames[:S_breath.shape[1]]

        breath_voiced = S_breath[:, :voiced_frames.size].copy()
        mask_v = (voiced_frames > 0)
        if np.any(mask_v):
            cols = np.nonzero(mask_v)[0]
            breath_voiced[:, cols] *= bright_breath
            breath_voiced[:, cols] = gaussian_filter(breath_voiced[:, cols], sigma=(0.5, 0))

        S_breath[:, :voiced_frames.size] = breath_voiced

    aper_breath = istft(S_breath, hop_length=hop_length, window=window, length=len(y))
    aper_uv = istft(S_uv, hop_length=hop_length, window=window, length=len(y))

    # Gain Control (Breathiness vs Unvoiced)
    voicing_mask_smooth = smooth_mask_ds(voicing_mask, sigma=noise_transition_smoothness, ds=4)
    breathy_aper = aper_breath * voicing_mask_smooth * breath_strength
    noisy_aper = aper_uv * (1.0 - voicing_mask_smooth) * uv_strength
    aper_uv = noisy_aper
    aper_bre = breathy_aper

    if volume_jitter:
    # the volume jitter thing
        harmonic_jitter = create_volume_jitter(len(harmonic), sr, speed=volume_jitter_speed, strength=volume_jitter_strength_harm, vibrato=volume_vibrato)
        breathy_jitter = create_volume_jitter(len(aper_bre), sr, speed=volume_jitter_speed, strength=volume_jitter_strength_breath, vibrato=volume_vibrato)
        voicing_jitter_mask = gaussian_filter1d(voicing_mask, sigma=20)
        harmonic *= 1.0 + (harmonic_jitter - 1.0) * voicing_jitter_mask
        aper_bre *= 1.0 + (breathy_jitter - 1.0) * voicing_jitter_mask

    combined = harmonic + aper_uv + aper_bre

    if roughness_on:
        harmonic_rough = apply_vocal_roughness(
            harmonic, f0_interp, voicing_mask, sr,
            k_list=rough_k_list,
            h_list=rough_h_list,
            alpha=rough_alpha,
            hp_fc=rough_hp_fc,
            noise_amp=rough_noise_amp,
            noise_smooth_ms=rough_noise_smooth_ms,
            alpha_slew_ms=rough_alpha_slew_ms
        )
        combined = harmonic_rough + aper_uv + aper_bre

    norm_amt = float(np.clip(normalize, 0.0, 1.0))

    peak = float(np.max(np.abs(combined)) + 1e-12)
    gain_peak = 1.0 / peak

    gain = gain_peak ** norm_amt

    harmonic *= gain
    aper_uv  *= gain
    aper_bre *= gain
    reconstruct = combined * gain

    return reconstruct, harmonic, aper_uv, aper_bre

if __name__ == "__main__":

    _ = pulse_train_numba(np.zeros(16, dtype=np.float32), 44100) #warmup for benchmark

    input_file = '_input.wav'

    stretch_factor = 1.0

    pitch_shift = 1.0
    formant_shift = 1.0

    F1 = 1.0
    F2 = 1.0
    F3 = 1.0
    F4 = 1.0

    volume_jitter = False #only on voiced
    volume_vibrato= False
    volume_jitter_speed=128
    volume_jitter_strength_harm = 60
    volume_jitter_strength_breath = 10
    
    subharm_vibrato = False
    subharm_vibrato_rate=75#36
    subharm_vibrato_depth=3
    subharm_vibrato_delay=0.01

    add_subharm = False
    subharm_weight=1.5 #3.0
    subharm_semitones=1.5 # or like a list for multiple subharms lets say [-12, 12]
    cut_subharm_below_f0=False
    subharm_f0_jitter=0

    f0_jitter = False

    input_name = os.path.splitext(input_file)[0]
    y, sr = sf.read(input_file)
    if y.ndim > 1:
        y = y.mean(axis=1)

    n_fft = 2048
    hop_length = n_fft // 4

    import time
    
    start_time = time.time()
    
    env_spec, f0_interp, voicing_mask, formants, env_knots = extract_features(y, sr, n_fft=n_fft, hop_length=hop_length)

    reconstruct, harmonic, aper_uv, aper_bre= synthesize(
        env_spec, f0_interp, voicing_mask, y, sr,
        n_fft=n_fft, hop_length=hop_length, stretch_factor=stretch_factor,
        pitch_shift=pitch_shift, formant_shift=formant_shift,
        formants=formants, F1_shift=F1, F2_shift=F2, F3_shift=F3, F4_shift=F4,
        f0_jitter=f0_jitter, volume_jitter=volume_jitter,
        add_subharm=add_subharm, subharm_weight=subharm_weight,
        subharm_vibrato=subharm_vibrato, subharm_vibrato_rate=subharm_vibrato_rate,
        subharm_semitones=subharm_semitones, cut_subharm_below_f0=cut_subharm_below_f0,
        subharm_vibrato_depth=subharm_vibrato_depth, subharm_f0_jitter=subharm_f0_jitter,
        subharm_vibrato_delay=subharm_vibrato_delay, volume_vibrato=volume_vibrato,
        volume_jitter_speed=volume_jitter_speed, volume_jitter_strength_harm=volume_jitter_strength_harm,
        volume_jitter_strength_breath=volume_jitter_strength_breath,
        roughness_on=False,
        rough_k_list=(2.5,4),
        rough_h_list=[0.1, 0.4, 0.6, 2],
        rough_alpha=2,
        rough_hp_fc=280.0,
        rough_noise_amp=0.6,
        rough_noise_smooth_ms=120.0,
        rough_alpha_slew_ms=120.0
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    
    reconstruct_wav = f'{input_name}_reconstruct.wav'
    harmonic_wav = f'{input_name}_harmonic.wav'
    breathiness = f'{input_name}_breathiness.wav'
    unvoiced = f'{input_name}_unvoiced.wav'
    sf.write(reconstruct_wav, reconstruct, sr)
    sf.write(harmonic_wav, harmonic, sr)
    sf.write(breathiness, aper_bre, sr)
    sf.write(unvoiced, aper_uv, sr)
    print(f'Reconstructed audio saved: {reconstruct_wav}')

    save_feature = False
    if save_feature:
        env_spec = env_spec.astype(np.float16)
        f0_interp = f0_interp.astype(np.float16)
        voicing_mask = voicing_mask.astype(np.float16)

        np.savez_compressed(
            f'{input_name}_features.npz',
            env_spec=env_spec,
            f0_interp=f0_interp,
            voicing_mask=voicing_mask,
            formants=formants,
            sr=np.array([sr]),
            y_len=np.array([len(y)]) 
        )
        print(f'Saved feature set: {input_name}_features.npz')

        #data = np.load("pjs001_singing_seg001_features.npz", allow_pickle=True)

        #env_spec = data["env_spec"]
        #f0_interp = data["f0_interp"]
        #voicing_mask = data["voicing_mask"]
        #formants = data["formants"].item()
        #sr = int(data["sr"][0])
        #y_len = int(data["y_len"][0])
