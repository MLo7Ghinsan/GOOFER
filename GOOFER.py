import numpy as np
import soundfile as sf
import os
import parselmouth
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import medfilt
import time
start_time = time.time()

test = False

# params
noise_type = 'white'  #'white' or 'brown' or 'pink'
noise_transition_smoothness = 100 # Smoothness when transitioning from unvoiced to breathiness, was 40 originally
uv_strength = 0.5 # Unvoiced noise level
breath_strength = 0.0375 # Breathiness in voiced speech
voicing_threshold = 25 # Hz (above this = voiced)

start_sec = None #0.3
end_sec = None #0.4
stretch_factor = 1.0

pitch_shift = 1.0

formant_shift = 1.0

individual_formant_shift = False
F1 = 1.0
F2 = 1.0
F3 = 1.0
F4 = 1.0

volume_jitter = False #only on voiced
volume_jitter_speed = 150
volume_jitter_strength_harm = 50
volume_jitter_strength_breath = 100

add_subharm = False
subharm_weight = 0.5

f0_jitter = False
f0_jitter_speed = 100
f0_jitter_strength =1.5

# Im using small af n_fft and hop_length cus bigger is questionable
n_fft = 2048 // 4
hop_length = 512 // 8
window = np.hanning(n_fft)

f0_merge_range = 10

apply_brightness = True
normalize = True
save_features_wav = True

input_file = 'toy.wav' # test file lmao
input_file2 = None
morph_strength = 0.0 # 0.0 = only first audio, 1.0 = only second audio

input_name = os.path.splitext(input_file)[0]
y, sr = sf.read(input_file)
if y.ndim > 1:
    y = y.mean(axis=1)

second_audio_available = False
if input_file2 and os.path.isfile(input_file2) and os.path.getsize(input_file2) > 0:
    y2, sr2 = sf.read(input_file2)
    if y2.ndim > 1:
        y2 = y2.mean(axis=1)
    if sr2 != sr:
        raise ValueError('Second audio must have the same sample rate as the first')
    second_audio_available = True
else:
    y2 = None

# tools
current_time = time.time()
def log_time(label):
    global current_time
    new_time = time.time()
    print(f'{label} took: {new_time - current_time:.2f} s')
    current_time = new_time

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

def fix_f0_gaps(f0_array, max_gap=10):
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
        return interp1d(x_old, feature, kind=kind, fill_value='extrapolate')(x_new)
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

def create_volume_jitter(length, sr, speed=30.0, strength=0.5, seed=None):
    #noise modulato... Band-limit it (like dirty LFO)
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(0, length / sr, num=length)
    noise = np.random.randn(len(t))
    noise = gaussian_filter1d(noise, sigma=sr / (speed * 6))
    noise /= np.max(np.abs(noise) + 1e-6)
    envelope = 1.0 + noise * strength
    return envelope

def apply_f0_jitter(f0_array, sr, speed=40.0, strength=0.04, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(0, len(f0_array) / sr, num=len(f0_array))
    noise = np.random.randn(len(t))
    noise = gaussian_filter1d(noise, sigma=sr / (speed * 6))
    noise /= np.max(np.abs(noise) + 1e-6)
    jitter = 1.0 + noise * strength
    return jitter

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

        warp_func = interp1d(freq_map_dst, freq_map_src, kind='linear', bounds_error=False, fill_value='extrapolate')
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
        raise ValueError(f'Unknown noise type: {noise_type}')
    return noise / (np.max(np.abs(noise)) + 1e-8)

print('Spectral Envelope Estimation:')
# Spectral envelope
S_orig = stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
log_time('    STFT')
mag = np.abs(S_orig) + 1e-8
env_spec = mag

if individual_formant_shift:
    n_frames = env_spec.shape[1]
    formants = extract_formants(y, sr, hop_length, target_frames=n_frames)
    ind_formant_ratios = {1: F1, 2: F2, 3: F3, 4: F4}
    ind_formant_shifted = transpose_formants(formants, ind_formant_ratios)
    env_spec = warp_env_by_formants(env_spec, formants, ind_formant_shifted, sr)

env_spec4breathiness = gaussian_filter1d(env_spec, sigma=1.75, axis=0)

if formant_shift != 1.0:
    env_spec = shift_formants(env_spec, formant_shift, sr)

# envelope morphing with second audio (if any)
if second_audio_available:
    S_morph = stft(y2, n_fft=n_fft, hop_length=hop_length, window=window)
    mag_morph = np.abs(S_morph) + 1e-8
    env_morph = mag_morph
    env_morph4breath = gaussian_filter1d(env_morph, sigma=1.75, axis=0)

    env_morph = match_env_frames(env_morph, env_spec.shape[1])
    env_morph4breath = match_env_frames(env_morph4breath, env_spec4breathiness.shape[1])

    env_spec = env_spec * (1.0 - morph_strength) + env_morph * morph_strength
    env_spec4breathiness = (env_spec4breathiness * (1.0 - morph_strength) + env_morph4breath * morph_strength)
    log_time('    Morph envelope')

log_time('    Envelope filter')

print('F0 Estimation:')
# F0 estimation
snd = parselmouth.Sound(y, sr)
frame_duration = hop_length / sr
pitch = snd.to_pitch(time_step=frame_duration, pitch_floor=75, pitch_ceiling=600) # pitch floor could be voicing_threshold
f0_track = np.nan_to_num(pitch.selected_array['frequency'])
f0_track = fix_f0_gaps(f0_track, f0_merge_range)

# interpolate F0 to sample rate
times_f0 = np.linspace(0, len(y)/sr, num=len(f0_track))
interp_func = interp1d(times_f0, f0_track, kind='linear', fill_value=0, bounds_error=False)
times_samples = np.linspace(0, len(y)/sr, num=len(y))
f0_interp = interp_func(times_samples)
f0_interp = np.clip(f0_interp, 1e-5, 2000)
f0_interp *= pitch_shift # f0 shifting test
voicing_mask = (f0_interp > voicing_threshold).astype(float)

# Stretch if used
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
            stretch_feature(voicing_mask[start_idx:end_idx], stretch_factor, kind='nearest'),
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
        voicing_mask = stretch_feature(voicing_mask, stretch_factor, kind='nearest')
        env_spec4breathiness = stretch_feature(env_spec4breathiness, stretch_factor)

    new_len = len(f0_interp)
    if stretch_factor > 1.0:
        y = np.pad(y, (0, new_len - len(y)), mode='edge')

    else:
        y = y[:new_len]

if test:
    ### test synthesis f0 edit
    constant_f0 = 120
    vibrato_rate = 5.5     # Hz
    vibrato_depth = 20   # Hz
    vibrato_delay = 1.0    # seconds before vibrato starts
    vibrato_ramp_time = 7  # seconds to reach full depth

    t = np.linspace(0, len(f0_interp) / sr, num=len(f0_interp))
    voiced_mask = f0_interp > voicing_threshold
    f0_interp[voiced_mask] = constant_f0
    vibrato_envelope = np.zeros_like(t)
    ramp_start = vibrato_delay
    ramp_end = ramp_start + vibrato_ramp_time
    ramp_mask = (t >= ramp_start) & (t <= ramp_end)
    vibrato_envelope[ramp_mask] = (t[ramp_mask] - ramp_start) / (ramp_end - ramp_start)
    vibrato_envelope[t > ramp_end] = 1.0
    vibrato_envelope = vibrato_envelope**2  # makes the wobble glide in instead of snap
    f0_interp += np.sin(2 * np.pi * vibrato_rate * t) * vibrato_depth * vibrato_envelope
    ###

if f0_jitter:
    f0_jitter = apply_f0_jitter(f0_interp, sr, speed=f0_jitter_speed, strength=f0_jitter_strength)
    f0_interp *= 1.0 + ((f0_jitter - 1.0) * voicing_mask)

log_time('    PYin')

print('LF Pulse Train:')
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
            # low Rg makes gritty harmonics sacrificing dynamics
            pulse_cache[last_f0] = lf_model_pulse(T, Ra=0.02, Rg=1.7, Rk=1, sr=sr) #Ra=0.02, Rg=1.7, Rk=0
        lf_pulse = pulse_cache[last_f0]
        start = i
        end = min(len(pulse), start + len(lf_pulse))
        pulse[start:end] += lf_pulse[:end - start]
        phase -= 1.0

if add_subharm:
    sub_pulse = np.zeros_like(f0_interp)
    phase = 0.0
    for i in range(len(f0_interp)):
        f0 = f0_interp[i]
        if f0 > 0:
            last_f0 = f0
        T = 2.0 / last_f0 # 1 octave below
        phase += f0_interp[i] / (sr * 2.0)

        if phase >= 1.0:
            key = f'{last_f0:.2f}_sub1oct'
            if key not in pulse_cache:
                pulse_cache[key] = lf_model_pulse(T, Ra=0.02, Rg=1.7, Rk=1, sr=sr)
            lf_pulse = pulse_cache[key]
            start = i
            end = min(len(sub_pulse), start + len(lf_pulse))
            sub_pulse[start:end] += lf_pulse[:end - start]
            phase -= 1.0

    sub_pulse /= np.max(np.abs(sub_pulse) + 1e-6)
    pulse += subharm_weight * sub_pulse

pulse /= np.max(np.abs(pulse) + 1e-6)

log_time('    Generation')

# Harmonic synth
print('Harmonic Synthesis:')
S_harm = stft(pulse, n_fft=n_fft, hop_length=hop_length, window=window)
if env_spec.shape[1] > S_harm.shape[1]:
    env_spec = env_spec[:, :S_harm.shape[1]]
elif env_spec.shape[1] < S_harm.shape[1]:
    pad_width = S_harm.shape[1] - env_spec.shape[1]
    env_spec = np.pad(env_spec, ((0, 0), (0, pad_width)), mode='edge')
log_time('    STFT')
mag_harm = np.max(np.abs(S_harm) + 1e-8)
freq_bins = S_harm.shape[0]
boost_curve = np.linspace(1, 100, freq_bins).reshape(-1, 1)
# lower f0 reproduce original signal, this is an attempt to fix
if pitch_shift < 1.0:
    env_spec_4harm = gaussian_filter1d(env_spec, sigma=1.65, axis=0)
else:
    env_spec_4harm = env_spec

S_harm = (S_harm / mag_harm) * env_spec_4harm
S_harm = S_harm * boost_curve

if apply_brightness:
    brightness_curve = create_brightness_curve(S_harm.shape[0], sr, 3500, 5000, gain_db=6.0)
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

log_time('    ISTFT')

print('Aperiodic Synthesis:')
raw_noise = generate_noise(noise_type, len(y), sr)
S_noise = stft(raw_noise, n_fft=n_fft, hop_length=hop_length, window=window)

log_time('    STFT (breath + uv)')

env_noise = match_env_frames(env_spec4breathiness, S_noise.shape[1])

mag_noise = np.abs(S_noise) + 1e-8
S_breath = (S_noise / mag_noise) * env_noise
S_uv = (S_noise / mag_noise) * env_noise

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

    #uv_unvoiced = S_uv[:, :voiced_frames.size].copy()
    #uv_unvoiced[:, voiced_frames <= 0] *= brightness_curve
    #uv_unvoiced[:, voiced_frames <= 0] = gaussian_filter(uv_unvoiced[:, voiced_frames <= 0], sigma=(0.5, 0))
    #S_uv[:, :voiced_frames.size] = uv_unvoiced

aper_breath = istft(S_breath, hop_length=hop_length, window=window, length=len(y))
aper_uv = istft(S_uv, hop_length=hop_length, window=window, length=len(y))

log_time('    ISTFT (breath + uv)')

# Gain Control (Breathiness vs Unvoiced)
voicing_mask_smooth = gaussian_filter1d(voicing_mask, sigma=noise_transition_smoothness)
breathy_aper = aper_breath * voicing_mask_smooth * breath_strength
noisy_aper = aper_uv * (1.0 - voicing_mask_smooth) * uv_strength
aper_uv = noisy_aper
aper_bre = breathy_aper

if volume_jitter:
# the volume jitter thing
    harmonic_jitter = create_volume_jitter(len(harmonic), sr, speed=volume_jitter_speed, strength=volume_jitter_strength_harm)
    breathy_jitter = create_volume_jitter(len(aper_bre), sr, speed=volume_jitter_speed, strength=volume_jitter_strength_breath)
    voicing_jitter_mask = gaussian_filter1d(voicing_mask, sigma=20)
    harmonic *= 1.0 + (harmonic_jitter - 1.0) * voicing_jitter_mask
    aper_bre *= 1.0 + (breathy_jitter - 1.0) * voicing_jitter_mask

# sLAy!!!
harmonic_wav, breath_wav, unvoiced_wav, reconstruct_wav = f'{input_name}_harmonics.wav', f'{input_name}_breathiness.wav', f'{input_name}_unvoiced.wav', f'{input_name}_reconstruct.wav'

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

if save_features_wav:
    sf.write(harmonic_wav, harmonic, sr)
    sf.write(breath_wav, aper_bre, sr)
    sf.write(unvoiced_wav, aper_uv, sr)
    print(f'Files saved: {harmonic_wav} + {breath_wav} + {unvoiced_wav}')

sf.write(reconstruct_wav, reconstruct, sr)
print(f'Reconstructed audio saved: {reconstruct_wav}')
