import numpy as np
import soundfile as sf
import os
import parselmouth

def interp1d(x, y, kind='linear', fill_value='extrapolate'):
    if kind != 'linear':
        raise ValueError("Only 'linear' interpolation is supported.") # im lazy, duh

    x = np.asarray(x)
    y = np.asarray(y)
    
    def interpolator(x_new):
        x_new = np.asarray(x_new)
        y_new = np.zeros_like(x_new, dtype=y.dtype)
        
        # find indices for interpolation
        indices = np.searchsorted(x, x_new, side='left')
        indices = np.clip(indices, 1, len(x) - 1)
        # identify points within bounds
        within_bounds = (x_new >= x[0]) & (x_new <= x[-1])

        # linear interpolation for points within bounds
        if np.any(within_bounds):
            x0 = x[indices[within_bounds] - 1]
            x1 = x[indices[within_bounds]]
            y0 = y[indices[within_bounds] - 1]
            y1 = y[indices[within_bounds]]
            # we hate division by zero
            denom = x1 - x0
            denom = np.where(denom == 0, 1e-10, denom)
            # linear interpolation
            slope = (y1 - y0) / denom
            y_new[within_bounds] = y0 + slope * (x_new[within_bounds] - x0)
        
        # handle points outside bounds based on fill_value
        if fill_value != 'extrapolate':
            # if fill_value is a number likr lets say 0), set out-of-bounds values to it
            try:
                fill_val = float(fill_value)
                y_new[~within_bounds] = fill_val
            except (TypeError, ValueError):
                raise ValueError("fill_value must be 'extrapolate' or a number")
        else:
            # extrapolate for points outside bounds
            # left side: use slope from first two points
            left_mask = x_new < x[0]
            if np.any(left_mask):
                slope_left = (y[1] - y[0]) / (x[1] - x[0] + 1e-10)
                y_new[left_mask] = y[0] + slope_left * (x_new[left_mask] - x[0])
            # right side: use slope from last two points
            right_mask = x_new > x[-1]
            if np.any(right_mask):
                slope_right = (y[-1] - y[-2]) / (x[-1] - x[-2] + 1e-10)
                y_new[right_mask] = y[-1] + slope_right * (x_new[right_mask] - x[-1])
            # left side, right side- ha wo tsu k ida shi te papa pa
        return y_new
    return interpolator

def gaussian_filter1d(input_array, sigma, axis=-1, truncate=4.0):
    # clamp to something reasonable
    sigma = max(float(sigma), 1e-6)
    # calculate kernel
    radius = int(truncate * sigma + 0.5)
    t = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()

    # for easier handling...
    input_array = np.moveaxis(np.asarray(input_array), axis, -1)
    
    # pad reflectively to avoid crusty edges
    padded = np.pad(input_array, 
                    [(0, 0)] * (input_array.ndim - 1) + [(radius, radius)],
                    mode='reflect')

    # apply convolution manually for each 1D slice along last axis
    output = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='valid'), -1, padded)

    return np.moveaxis(output, -1, axis) # gotta move axis back where it came from

def gaussian_filter(input_array, sigma):
    input_array = np.asarray(input_array)
    if input_array.ndim != 2:
        raise ValueError("gaussian_filter expects a 2D array.")
    
    sigma = tuple(max(s, 1e-6) for s in sigma)
    
    output = gaussian_filter1d(input_array, sigma[0], axis=0)
    output = gaussian_filter1d(output, sigma[1], axis=1)
    return output

def load_features(path):
    data = np.load(path, allow_pickle=True)
    return (
        data["env_spec"],
        data["f0_interp"],
        data["voicing_mask"],
        data["formants"].item(),
        int(data["sr"][0]),
        int(data["y_len"][0])
    )

def save_features(path, env_spec, f0_interp, voicing_mask, formants, sr, y_len):
    with open(path, 'wb') as goofy_ahh:
        np.savez_compressed(
            goofy_ahh,
            env_spec=env_spec,
            f0_interp=f0_interp,
            voicing_mask=voicing_mask,
            formants=formants,
            sr=np.array([sr]),
            y_len=np.array([y_len]),
        )

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
        window = np.hanning(n_fft)
    pad = n_fft // 2
    x_padded = np.pad(x, pad, mode='reflect')
    num_frames = 1 + (len(x_padded) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        x_padded,
        shape=(n_fft, num_frames),
        strides=(x_padded.strides[0], hop_length * x_padded.strides[0])
    ).copy()
    frames *= window[:, None]
    return np.fft.rfft(frames, axis=0)

def istft(stft_matrix, hop_length=512, window=None, length=None):
    n_fft = (stft_matrix.shape[0] - 1) * 2
    if window is None:
        window = np.hanning(n_fft)
    frames = np.fft.irfft(stft_matrix, axis=0)
    pad = n_fft // 2
    expected_len = n_fft + hop_length * (frames.shape[1] - 1)
    y = np.zeros(expected_len)
    win_sum = np.zeros(expected_len)
    for i in range(frames.shape[1]):
        start = i * hop_length
        y[start:start+n_fft] += frames[:, i] * window
        win_sum[start:start+n_fft] += window**2
    nonzero = win_sum > 1e-8
    y[nonzero] /= win_sum[nonzero]
    y = y[pad:pad + (length or (len(y)-2*pad))]
    return y

def fix_f0_gaps(f0_array, max_gap=4):
    f0_fixed = f0_array.copy()
    i = 0
    while i < len(f0_fixed):
        if f0_fixed[i] == 0:
            start = i
            while i < len(f0_fixed) and f0_fixed[i] == 0:
                i += 1
            end = i
            if start > 0 and end < len(f0_fixed) and (end - start) <= max_gap:
                interp = np.linspace(f0_fixed[start - 1], f0_fixed[end], end - start + 2)[1:-1]
                f0_fixed[start:end] = interp
        else:
            i += 1
    return f0_fixed

def lf_model_pulse(T, Ra=0.01, Rg=1.47, Rk=0.34, sr=44100):
    t = np.linspace(0, T, int(sr * T), endpoint=False)
    Ee = 1.0
    Ta = Ra * T
    Tp = T / (2 * Rg)
    Te = Tp + Rk * (T - Tp)

    Ug = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < Te:
            Ug[i] = Ee * np.exp(-np.pi * (ti / Tp)**2)
        elif ti < T:
            Ug[i] = -Ee * ((1 - (ti - Te) / (T - Te))**2)
    return Ug / (np.max(np.abs(Ug)) + 1e-6)

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

def add_subharms(f0_interp, sr, subharm_weight=0.5, subharm_octaves=1):
    sub_pulse = np.zeros_like(f0_interp)
    phase = 0.0
    last_f0 = 160.0  # fallback
    pulse_cache = {} 
    
    for i in range(len(f0_interp)):
        f0 = f0_interp[i]
        if f0 > 0:
            last_f0 = f0
        T = (2 ** subharm_octaves) / last_f0  # 1 or 2 octaves below
        phase += f0_interp[i] / (sr * (2 ** subharm_octaves))
        
        if phase >= 1.0:
            key = f'{last_f0:.2f}_sub{subharm_octaves}oct'
            if key not in pulse_cache:
                pulse_cache[key] = lf_model_pulse(T, Ra=0.02, Rg=1.7, Rk=1, sr=sr)
            lf_pulse = pulse_cache[key]
            start = i
            end = min(len(sub_pulse), start + len(lf_pulse))
            sub_pulse[start:end] += lf_pulse[:end - start]
            phase -= 1.0
    
    sub_pulse /= np.max(np.abs(sub_pulse) + 1e-6)
    return sub_pulse * subharm_weight

def apply_subharm_vibrato(f0_interp, sr, vibrato_rate=6.0, vibrato_depth=0.1, vibrato_delay=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t = np.arange(len(f0_interp)) / sr
    
    phase = np.random.uniform(0, 2*np.pi) if seed else 0 # create sinusoid
    vibrato = np.sin(2 * np.pi * vibrato_rate * t + phase) * subharm_weight
    
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
    transposed = {}
    for i, track in formant_tracks.items():
        ratio = shift_ratios.get(i, 1.0)
        transposed[i] = np.array(track) * ratio
    return transposed

def warp_env_by_formants(env, orig_formants, shifted_formants, sr):
    n_bins, n_frames = env.shape
    freqs = np.linspace(0, sr / 2, n_bins)
    warped_env = np.zeros_like(env)

    for t in range(n_frames):
        freq_map_src = []
        freq_map_dst = []

        for i in range(1, 5):
            f_orig = orig_formants.get(i, [0]*n_frames)[t]
            f_shifted = shifted_formants.get(i, [0]*n_frames)[t]
            if f_orig > 50 and f_orig < sr / 2 and f_shifted > 50:
                freq_map_src.append(f_orig)
                freq_map_dst.append(f_shifted)

        freq_map_src = [0.0] + freq_map_src + [sr / 2]
        freq_map_dst = [0.0] + freq_map_dst + [sr / 2]

        warp_func = interp1d(freq_map_dst, freq_map_src, kind='linear', fill_value='extrapolate')
        warped_freqs = warp_func(freqs)
        interp_func = interp1d(freqs, env[:, t], kind='linear', fill_value='extrapolate')
        warped_env[:, t] = interp_func(warped_freqs)

    return warped_env

def generate_noise(noise_type, length, sr):
    if noise_type == 'white':
        noise = np.random.randn(length)
    elif noise_type == 'pink':
        # 1/f filter in the freq-domain
        X = np.fft.rfft(np.random.randn(length))
        freqs = np.fft.rfftfreq(length, 1/sr)
        # scale by 1/sqrt(f) (pink)
        X /= np.maximum(freqs, 1.0)**0.5
        noise = np.fft.irfft(X, n=length)
    elif noise_type == 'brown':
        # integrate white noise (1/f²)
        wn = np.random.randn(length)
        noise = np.cumsum(wn)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    return noise / (np.max(np.abs(noise)) + 1e-8)

def extract_features(y, sr, n_fft=1024, hop_length=256,
                     f0_min=75, f0_max=600, f0_merge_range=2):
    window = np.hanning(n_fft)
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

    return env_spec, f0_interp, voicing_mask, formants

def synthesize(env_spec, f0_interp, voicing_mask,
               y, sr, n_fft=1024, hop_length=256,
               stretch_factor=1.0, start_sec=None, end_sec=None,
               apply_brightness=True, normalize=True, noise_type='white',
               uv_strength=0.5, breath_strength=0.0375, noise_transition_smoothness=100,
               pitch_shift=1.0, formant_shift=1.0,
               f0_jitter=False, f0_jitter_speed=100, f0_jitter_strength=1.5,
               volume_jitter=False, volume_vibrato=False, volume_jitter_speed=150, volume_jitter_strength_harm=50, volume_jitter_strength_breath=100,
               add_subharm=False, subharm_weight=0.5, subharm_vibrato=False, subharm_vibrato_rate=6.0, subharm_vibrato_depth=0.1,
               subharm_vibrato_delay=0.1,
               F1_shift=1.0, F2_shift=1.0, F3_shift=1.0, F4_shift=1.0,
               formants=None):
    window = np.hanning(n_fft)

    env_spec4breathiness = gaussian_filter1d(env_spec, sigma=1.75, axis=0)

    f0_interp *= pitch_shift

    n_frames = env_spec.shape[1]

    if any(shift != 1.0 for shift in [F1_shift, F2_shift, F3_shift, F4_shift]):
        ind_formant_ratios = {1: F1_shift, 2: F2_shift, 3: F3_shift, 4: F4_shift}
        ind_formant_shifted = transpose_formants(formants, ind_formant_ratios)
        env_spec = warp_env_by_formants(env_spec, formants, ind_formant_shifted, sr)

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

    # LF glottal pulse train generator: generate LF pulse train from F0
    pulse = np.zeros_like(f0_interp)
    phase = 0.0
    last_f0 = 160.0 # fallback
    pulse_cache = {}

    for i in range(len(f0_interp)):
        f0 = f0_interp[i]
        if f0 > 0:
            last_f0 = f0
        T = 1.0 / last_f0
        phase += f0_interp[i] / sr

        if phase >= 1.0:
            if last_f0 not in pulse_cache:
                pulse_cache[last_f0] = lf_model_pulse(T, Ra=0.02, Rg=1.7, Rk=1, sr=sr)
            lf_pulse = pulse_cache[last_f0]
            start = i
            end = min(len(pulse), start + len(lf_pulse))
            pulse[start:end] += lf_pulse[:end - start]
            phase -= 1.0
            
    if add_subharm:
        if subharm_vibrato:
            f0_for_subharms = apply_subharm_vibrato(
                f0_interp, sr,
                vibrato_rate=subharm_vibrato_rate,
                vibrato_depth=subharm_vibrato_depth,
                vibrato_delay=subharm_vibrato_delay,
            )
        else:
            f0_for_subharms = f0_interp
            
        sub_pulse = add_subharms(
            f0_for_subharms, sr,
            subharm_weight=subharm_weight,
            subharm_octaves=1
        )
        pulse += sub_pulse

    S_harm = stft(pulse, n_fft=n_fft, hop_length=hop_length, window=window)
    if env_spec.shape[1] > S_harm.shape[1]:
        env_spec = env_spec[:, :S_harm.shape[1]]
    elif env_spec.shape[1] < S_harm.shape[1]:
        pad_width = S_harm.shape[1] - env_spec.shape[1]
        env_spec = np.pad(env_spec, ((0, 0), (0, pad_width)), mode='edge')
    #log_time('    STFT')
    mag_harm = np.max(np.abs(S_harm) + 1e-8)
    freq_bins = S_harm.shape[0]
    boost_curve = np.linspace(1, 100, freq_bins).reshape(-1, 1)

    env_spec_4harm = env_spec

    S_harm = (S_harm / mag_harm) * env_spec_4harm
    S_harm = S_harm * boost_curve

    if apply_brightness:
        brightness_curve = create_brightness_curve(S_harm.shape[0], sr, 3500, 5000, gain_db=3.0)
        voiced_frames = voicing_mask[::hop_length]
        if voiced_frames.size < S_harm.shape[1]:
            voiced_frames = np.pad(voiced_frames, (0, S_harm.shape[1] - voiced_frames.size), mode='edge')
        else:
            voiced_frames = voiced_frames[:S_harm.shape[1]]

        harm_voiced = S_harm[:, :voiced_frames.size].copy()
        harm_voiced[:, voiced_frames > 0] *= brightness_curve
        harm_voiced[:, voiced_frames > 0] = gaussian_filter(harm_voiced[:, voiced_frames > 0], sigma=(0.5, 0))
        S_harm[:, :voiced_frames.size] = harm_voiced
    harmonic = istft(S_harm, hop_length=hop_length, window=window, length=len(y))

    raw_noise = generate_noise(noise_type, len(y), sr)
    S_noise = stft(raw_noise, n_fft=n_fft, hop_length=hop_length, window=window)

    # dynamic highpass based on F0 for better breathiness ig
    freqs = np.fft.rfftfreq(n_fft, 1 / sr).reshape(-1, 1)  # (n_freq, 1)
    f0_env_frames = f0_interp[::hop_length]
    f0_env_frames = np.pad(f0_env_frames, (0, S_noise.shape[1] - len(f0_env_frames)), mode='edge')
    cutoff = f0_env_frames.reshape(1, -1)  # (1, n_frames)

    # Smooth sigmoid mask per frame
    highpass_mask = (freqs >= cutoff)

    # Apply to noise
    S_noise_filtered = S_noise * highpass_mask

    env_noise = match_env_frames(env_spec4breathiness, S_noise.shape[1])

    mag_noise = np.abs(S_noise) + 1e-8
    S_uv = (S_noise / mag_noise) * env_noise

    mag_noise_breath = np.abs(S_noise_filtered) + 1e-8
    S_breath = (S_noise_filtered  / mag_noise_breath) * env_noise

    if apply_brightness:
        brightness_curve = create_brightness_curve(S_breath.shape[0], sr, 3500, 5000, gain_db=20.0)
        voiced_frames = voicing_mask[::hop_length]
        if voiced_frames.size < S_breath.shape[1]:
            voiced_frames = np.pad(voiced_frames, (0, S_breath.shape[1] - voiced_frames.size), mode='edge')
        else:
            voiced_frames = voiced_frames[:S_breath.shape[1]]

        breath_voiced = S_breath[:, :voiced_frames.size].copy()
        breath_voiced[:, voiced_frames > 0] *= brightness_curve
        breath_voiced[:, voiced_frames > 0] = gaussian_filter(breath_voiced[:, voiced_frames > 0], sigma=(0.5, 0))
        S_breath[:, :voiced_frames.size] = breath_voiced

    aper_breath = istft(S_breath, hop_length=hop_length, window=window, length=len(y))
    aper_uv = istft(S_uv, hop_length=hop_length, window=window, length=len(y))

    # Gain Control (Breathiness vs Unvoiced)
    voicing_mask_smooth = gaussian_filter1d(voicing_mask, sigma=noise_transition_smoothness)
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

    if normalize:
        peak = np.max(np.abs(combined) + 1e-6)
        gain = 1.0 / peak
        harmonic *= gain
        aper_uv *= gain
        aper_bre *= gain
        reconstruct = combined * gain
    else:
        reconstruct = combined

    return reconstruct, harmonic, aper_uv, aper_bre

if __name__ == "__main__":

    input_file = 'pjs001_singing_seg001.wav'

    noise_type = 'white'  #'white' or 'brown' or 'pink'

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
    subharm_vibrato_rate=36
    subharm_vibrato_depth=1.5
    subharm_vibrato_delay=0.01

    add_subharm = False
    subharm_weight=3.0

    f0_jitter = False

    input_name = os.path.splitext(input_file)[0]
    y, sr = sf.read(input_file)
    if y.ndim > 1:
        y = y.mean(axis=1)


    n_fft = 2048 // 2
    hop_length = 512 // 2

    env_spec, f0_interp, voicing_mask, formants = extract_features(y, sr, n_fft=n_fft, hop_length=hop_length)

    reconstruct, harmonic, aper_uv, aper_bre= synthesize(
        env_spec, f0_interp, voicing_mask, y, sr,
        n_fft=n_fft, hop_length=hop_length,
        noise_type=noise_type, stretch_factor=stretch_factor,
        pitch_shift=pitch_shift, formant_shift=formant_shift,
        formants=formants, F1_shift=F1, F2_shift=F2, F3_shift=F3, F4_shift=F4,
        f0_jitter=f0_jitter, volume_jitter=volume_jitter,
        add_subharm=add_subharm, subharm_weight=subharm_weight,
        subharm_vibrato=subharm_vibrato, subharm_vibrato_rate=subharm_vibrato_rate,
        subharm_vibrato_depth=subharm_vibrato_depth, 
        subharm_vibrato_delay=subharm_vibrato_delay, volume_vibrato=volume_vibrato,
        volume_jitter_speed=volume_jitter_speed, volume_jitter_strength_harm=volume_jitter_strength_harm,
        volume_jitter_strength_breath=volume_jitter_strength_breath)


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
