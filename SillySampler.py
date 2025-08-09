import sys, os, logging, re, traceback, multiprocessing
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

import GOOFER as gf

logging.basicConfig(format='%(message)s', level=logging.INFO)

# --- UTAU pitch & flags parsing --------------------------------------------
notes = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}
note_re = re.compile(r'([A-G]#?)(-?\d+)')
flag_re = re.compile(r'([A-Za-z]{1,2})([+-]?\d+)?')

def parse_flags(flag_string):
    flags = {}
    for key, val in flag_re.findall(flag_string.replace('/', '')):
        flags[key] = int(val) if val else None
    return flags

def to_uint6(c):
    o=ord(c)
    if o>=97: return o-71
    if o>=65: return o-65
    if o>=48: return o+4
    if o==43: return 62
    if o==47: return 63
    raise ValueError(f"Bad b64 '{c}'")

def to_int12(p):
    v=(to_uint6(p[0])<<6)|to_uint6(p[1])
    return v-4096 if (v&0x800) else v

def to_int12_stream(s):
    return [to_int12(s[i:i+2]) for i in range(0,len(s),2)]

def pitch_string_to_cents(x):
    parts = x.split('#')
    out = []
    for i in range(0, len(parts), 2):
        chunk = parts[i:i+2]
        if len(chunk) == 2:
            ps, run = chunk
            out += to_int12_stream(ps)
            out += [out[-1]] * int(run)
        else:
            out += to_int12_stream(chunk[0])
    arr = np.array(out, dtype=np.float32)
    return np.zeros_like(arr) if arr.size == 0 or np.all(arr == arr[0]) else np.concatenate([arr, [0.0]])

def note_to_midi(n):
    m = note_re.match(n)
    if not m: raise ValueError(f"Bad note '{n}'")
    nm, octv = m.groups()
    return (int(octv)+1)*12 + notes[nm]

def midi_to_hz(m):
    return 440.0 * 2**((m-69)/12)

def dynamic_butter_filter(signal, f0, sr, cutoff_factor, order=4, btype='lowpass'):
    out = np.zeros_like(signal)

    if np.any(f0 > 0):
        f0_smooth = np.convolve(f0, np.ones(5) / 5, mode='same')
    else:
        f0_smooth = f0.copy()

    if btype == 'lowpass':
        zi = [0.0] * order
        def apply_filter(x, cutoff, prev):
            alpha = (2 * np.pi * cutoff) / (2 * np.pi * cutoff + sr)
            y = np.empty_like(x)
            y_prev = prev
            for i, xn in enumerate(x):
                y_prev = y_prev + alpha * (xn - y_prev)
                y[i] = y_prev
            return y, y_prev
    else:
        zi = [(0.0, None)] * order
        def apply_filter(x, cutoff, prev_state):
            prev_y, prev_x = prev_state
            alpha = sr / (2 * np.pi * cutoff + sr)
            y = np.empty_like(x)
            if prev_x is None:
                prev_x = x[0]
            for i, xn in enumerate(x):
                prev_y = alpha * (prev_y + xn - prev_x)
                y[i] = prev_y
                prev_x = xn
            return y, (prev_y, prev_x)

    idx = 0
    while idx < len(signal):
        seg_f0 = f0_smooth[idx:idx+512]
        local_f0 = np.mean(seg_f0[seg_f0 > 0]) if np.any(seg_f0 > 0) else 0

        if local_f0 > 0:
            min_mult = 1.0
            max_mult = 20.0
            norm = np.clip((local_f0 - 16) / (4186 - 16), 0.0, 1.0)
            multiplier = max_mult - norm * (max_mult - min_mult)
            period = sr / local_f0
            seg_size = int(period * multiplier)
        else:
            seg_size = 256

        seg_size = max(64, min(seg_size, 2048))

        overlap_size = seg_size // 2
        fade_in = np.linspace(0, 1, overlap_size)
        fade_out = np.linspace(1, 0, overlap_size)

        end = min(idx + seg_size, len(signal))
        seg_signal = signal[idx:end]
        seg_f0_vals = f0_smooth[idx:end]
        mean_f0 = np.mean(seg_f0_vals[seg_f0_vals > 0]) if np.any(seg_f0_vals > 0) else 0

        if np.isnan(mean_f0) or mean_f0 <= 0:
            filtered = seg_signal
        else:
            fc = mean_f0 * cutoff_factor
            nyq = 0.5 * sr
            if btype == 'lowpass':
                normal_fc = min(fc / nyq, 0.99)
                cutoff = normal_fc * nyq
                filtered = seg_signal
                for j in range(order):
                    filtered, zi[j] = apply_filter(filtered, cutoff, zi[j])
            else:
                normal_fc = max(fc / nyq, 20 / nyq)
                cutoff = normal_fc * nyq
                filtered = seg_signal
                for j in range(order):
                    filtered, zi[j] = apply_filter(filtered, cutoff, zi[j])

        if idx > 0:
            cross_len = min(overlap_size, len(filtered), len(out) - idx)
            out[idx:idx+cross_len] = (
                out[idx:idx+cross_len] * fade_out[:cross_len] +
                filtered[:cross_len] * fade_in[:cross_len]
            )
            if cross_len < len(filtered):
                out[idx+cross_len:idx+len(filtered)] = filtered[cross_len:]
        else:
            out[idx:end] = filtered

        idx += seg_size - overlap_size

    return out

def stretch_prefix_1d(x, pre_len, factor):
    n = len(x)
    if pre_len <= 1 or n <= 1 or abs(factor - 1.0) < 1e-6:
        return x
    pre_new = max(1, int(round(pre_len * factor)))
    n_new = pre_new + (n - pre_len)
    idx_new = np.arange(n_new, dtype=np.float64)
    old_pos = np.where(idx_new < pre_new,
                       idx_new / factor,
                       (idx_new - pre_new) + pre_len)
    f = gf.interp1d(np.arange(n, dtype=np.float64), x, kind='linear', fill_value='extrapolate')
    return f(old_pos)

def stretch_prefix_2d_frames(M, pre_len, factor):
    n = M.shape[1]
    if pre_len <= 1 or n <= 1 or abs(factor - 1.0) < 1e-6:
        return M
    pre_new = max(1, int(round(pre_len * factor)))
    n_new = pre_new + (n - pre_len)
    idx_old = np.arange(n, dtype=np.float64)
    idx_new = np.arange(n_new, dtype=np.float64)
    old_pos = np.where(idx_new < pre_new,
                       idx_new / factor,
                       (idx_new - pre_new) + pre_len)
    rows = []
    for row in M:
        f = gf.interp1d(idx_old, row, kind='linear', fill_value='extrapolate')
        rows.append(f(old_pos))
    return np.stack(rows, axis=0)

def stretch_prefix_formant_track(track, pre_len, factor):
    arr = np.asarray(track, dtype=np.float64)
    arr_w = stretch_prefix_1d(arr, pre_len, factor)
    return arr_w

def is_audio_file(file):
    return file.suffix.lower() in ['.wav', '.flac', '.aiff', '.aif', '.mp3']

def process_file(audio_file):
    feat_file = audio_file.with_name(f"{audio_file.stem}_features.goofy")
    if feat_file.exists():
        logging.info(f"[SKIP] {feat_file.name} already exists")
        return
    try:
        logging.info(f"[EXTRACT] {audio_file}")
        y, sr = sf.read(audio_file)
        if y.ndim > 1:
            y = y.mean(axis=1)
        env, f0i, vmask, forms = gf.extract_features(y, sr)
        ylen = len(y)
        gf.save_features(feat_file, env, f0i, vmask, forms, sr, ylen)
    except Exception as e:
        logging.error(f"[ERROR] Failed to extract {audio_file.name}: {str(e)}")

def extract_features_recursive(input_path):
    input_path = Path(input_path)
    all_files = input_path.rglob('*') if input_path.is_dir() else [input_path]
    audio_files = [f for f in all_files if f.is_file() and is_audio_file(f)]

    num_threads = multiprocessing.cpu_count()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_file, audio_files)

    logging.info(f"[DONE] Extracted features from {len(audio_files)} files using {num_threads} threads.")

class GooferResampler:
    def __init__(
        self,
        in_file, out_file,
        pitch, velocity,
        flags='',
        offset=0, length=1000, consonant=0, cutoff=0,
        volume=100, modulation=0, tempo='!120', pitch_string='AA'
    ):
        self.in_file    = Path(in_file)
        self.out_file   = Path(out_file)
        self.pitch_m    = note_to_midi(pitch)
        self.velocity   = float(velocity)
        self.flags      = parse_flags(flags)
        self.offset     = float(offset) / 1000.0
        self.length     = float(length) / 1000.0
        self.consonant  = float(consonant) / 1000.0
        self.cutoff     = float(cutoff) / 1000.0
        self.volume     = float(volume) / 100.0
        self.modulation = float(modulation) / 100.0
        self.tempo      = float(tempo.lstrip('!'))
        self.bend       = pitch_string_to_cents(pitch_string)

        # gender flag
        self.formant_shift = 1.0 + (self.flags.get('g', 0) / 200.0)

        # formants band flag
        self.F1_shift = 1.0 + (self.flags.get('fa', 0) / 100.0)
        self.F2_shift = 1.0 + (self.flags.get('fb', 0) / 100.0)
        self.F3_shift = 1.0 + (self.flags.get('fc', 0) / 100.0)
        self.F4_shift = 1.0 + (self.flags.get('fd', 0) / 100.0)

        # roughness/harshness flag
        sh_val = self.flags.get('sh', None)
        self.f0_jitter = sh_val is not None and sh_val > 0
        self.f0_jitter_strength = (sh_val or 0) / 50.0
        sr_val = self.flags.get('sr', None)
        self.volume_jitter = sr_val is not None and sr_val > 0
        self.volume_jitter_strength = (sr_val or 0) / 50.0

        # breathiness flag
        self.breathiness_mix = (self.flags.get('B', 0) + 100) / 100.0

        # unvoiced flag
        self.unvoiced_mix = (self.flags.get('U', 0) + 100) / 100.0

        # voicing flag
        self.harmonic_mix = np.clip(self.flags.get('V', 100), 0, 100) / 100.0

        # stretch flag
        loop_flag = next((k for k in self.flags if k.lower() == 'l'), None)
        if loop_flag:
            lval = self.flags[loop_flag]
            if lval == 0:
                self.loop_mode = 'concat'
            elif lval == 1:
                self.loop_mode = 'avg'
            elif lval == 2:
                self.loop_mode = 'stretch'
            else:
                self.loop_mode = 'concat' # default slay
        else:
            self.loop_mode = 'concat' # default if no L flag (bad lmao)

        # tension flag
        self.tension = self.flags.get('st', 0) / 100.0

        # growl flag
        sg_val = self.flags.get('sg', 0)
        self.subharm_weight = (sg_val / 100.0) * 1.5
        self.add_subharm = sg_val > 0

        self.render()

    def render(self):
        feat = self.in_file.with_name(f'{self.in_file.stem}_features.goofy')
        if feat.exists():
            logging.info('Loading cached features')
            env, f0i, vmask, forms, sr, ylen = gf.load_features(feat)
        else:
            logging.info('Extracting features')
            y, sr = sf.read(self.in_file)
            if y.ndim > 1:
                y = y.mean(axis=1)
            env, f0i, vmask, forms = gf.extract_features(y, sr)
            ylen = len(y)
            gf.save_features(feat, env, f0i, vmask, forms, sr, ylen)

        features = (env, f0i, vmask, forms, sr, ylen)
        self.resample(features)

    def resample(self, features):
        env_spec, f0_interp, voicing_mask, forms, sr, ylen = features

        hop_length = 256

        start_sample     = int(self.offset * sr)
        consonant_sample = start_sample + int(self.consonant * sr)
        sample_length_sec = ylen / sr

        if self.cutoff < 0:
            end_sec = self.offset - self.cutoff
        else:
            end_sec = sample_length_sec - self.cutoff

        end_sample = int(end_sec * sr)

        #debug
        #logging.info(f"Sample length: {sample_length_sec:.3f}s")
        #logging.info(f"Offset: {self.offset:.3f}s: Start sample: {start_sample}")
        #logging.info(f"Cutoff: {self.cutoff:.3f}s: End sample: {end_sample}")

        logging.info('Interpolating features')
        # cut frame indices
        start_frame     = start_sample // hop_length
        consonant_frame = consonant_sample // hop_length
        end_frame       = end_sample // hop_length

        #debug
        #logging.info(f"Start frame: {start_frame} ({start_sample / sr:.3f}s)")
        #logging.info(f"Consonant frame: {consonant_frame} ({consonant_sample / sr:.3f}s)")
        #logging.info(f"End frame: {end_frame} ({end_sample / sr:.3f}s)")

        env_pre  = env_spec[:, start_frame:consonant_frame]
        f0_pre   = f0_interp[start_sample:consonant_sample]
        mask_pre = voicing_mask[start_sample:consonant_sample]

        env_tail = env_spec[:, consonant_frame:end_frame]
        f0_tail  = f0_interp[consonant_sample:end_sample]
        mask_tail= voicing_mask[consonant_sample:end_sample]

        desired_tail_samples = int(self.length * sr)

        # Loop (tile) envelope frames for sustain... 
        tail_frames = env_tail.shape[1]
        desired_tail_frames = int(np.ceil(self.length * sr / hop_length))

        if tail_frames >= desired_tail_frames:
            env_tail_looped = env_tail[:, :desired_tail_frames]
        else:
            reps = desired_tail_frames // tail_frames
            rem  = desired_tail_frames % tail_frames

            if self.loop_mode == 'stretch':
                if tail_frames == 0:
                    env_tail_looped = np.zeros((env_spec.shape[0], desired_tail_frames), dtype=np.float32)
                else:
                    env_tail_looped = gf.stretch_feature(env_tail, desired_tail_frames / tail_frames)

            elif tail_frames >= desired_tail_frames:
                env_tail_looped = env_tail[:, :desired_tail_frames]

            else:
                if self.loop_mode == 'avg':
                    loop_tile = (env_tail + env_tail[:, ::-1]) / 2.0
                    parts = [loop_tile] * reps
                    if rem:
                        parts.append(loop_tile[:, :rem])
                    env_tail_looped = np.concatenate(parts, axis=1)

                else: # "concat" mode
                    full_loop = [env_tail.copy()]
                    for _ in range(reps - 1):
                        prev = full_loop[-1]

                        max_fade = min(8, tail_frames // 2)
                        fade_in  = np.linspace(0, 1, max_fade)[None, :]
                        fade_out = np.linspace(1, 0, max_fade)[None, :]

                        A = prev[:, -max_fade:]
                        B = env_tail[:, :max_fade]
                        crossfaded = A * fade_out + B * fade_in

                        chunk = np.concatenate([
                            prev[:, :-max_fade],
                            crossfaded,
                            env_tail[:, max_fade:]
                        ], axis=1)

                        full_loop[-1] = chunk
                        full_loop.append(env_tail.copy())

                    if rem:
                        last_chunk = env_tail[:, :rem]
                        prev = full_loop[-1]
                        max_fade = min(8, rem // 2)
                        if max_fade > 0:
                            fade_in  = np.linspace(0, 1, max_fade)[None, :]
                            fade_out = np.linspace(1, 0, max_fade)[None, :]
                            A = prev[:, -max_fade:]
                            B = last_chunk[:, :max_fade]
                            crossfaded = A * fade_out + B * fade_in
                            chunk = np.concatenate([
                                prev[:, :-max_fade],
                                crossfaded,
                                last_chunk[:, max_fade:]
                            ], axis=1)
                        else:
                            chunk = np.concatenate([prev, last_chunk], axis=1)

                        full_loop[-1] = chunk

                    env_tail_looped = np.concatenate(full_loop, axis=1)

        # loop f0 and voicing mask
        tail_len = len(f0_tail)
        if tail_len >= desired_tail_samples:
            f0_tail_looped   = f0_tail[:desired_tail_samples]
            mask_tail_looped = mask_tail[:desired_tail_samples]
        else:
            reps_samp = desired_tail_samples // tail_len
            rem_samp  = desired_tail_samples % tail_len
            parts_f0   = [f0_tail] * reps_samp
            parts_mask = [mask_tail] * reps_samp
            if rem_samp:
                parts_f0.append(f0_tail[:rem_samp])
                parts_mask.append(mask_tail[:rem_samp])
            f0_tail_looped   = np.concatenate(parts_f0)
            mask_tail_looped = np.concatenate(parts_mask)

        formants_pre = {k: v[start_frame:consonant_frame] for k, v in forms.items()}
        formants_tail = {k: v[consonant_frame:end_frame] for k, v in forms.items()}

        formants_tail_looped = {}
        for k in forms:
            track = formants_tail[k]
            if self.loop_mode == 'stretch':
                formants_tail_looped[k] = gf.stretch_feature(np.array(track), desired_tail_frames / len(track)).tolist()
            else:
                reps = desired_tail_frames // len(track)
                rem = desired_tail_frames % len(track)
                formants_tail_looped[k] = (track * reps + track[:rem])

        formants_new = {
            k: np.concatenate([formants_pre[k], formants_tail_looped[k]])
            for k in forms
        }

        # concatenate pre and looped tail
        env_new  = np.concatenate([env_pre, env_tail_looped], axis=1)
        f0_new   = np.concatenate([f0_pre, f0_tail_looped])
        mask_new = np.concatenate([mask_pre, mask_tail_looped])

        target_frames = env_new.shape[1]
        for k in formants_new:
            f = formants_new[k]
            if len(f) < target_frames:
                pad = target_frames - len(f)
                formants_new[k] = np.pad(f, (0, pad), mode='edge')
            elif len(f) > target_frames:
                formants_new[k] = f[:target_frames]

        # convel shits
        vel_factor = float(2.0 ** (1.0 - (self.velocity / 100.0)))
        #vel_factor = float(np.clip(vel_factor, 0.33, 3.0)) this was for a test lmao idt ppl will need this

        pre_frames  = env_pre.shape[1]
        pre_samples = len(f0_pre)

        if abs(vel_factor - 1.0) > 1e-6 and pre_frames > 1 and pre_samples > 1:
            env_new = stretch_prefix_2d_frames(env_new, pre_frames, vel_factor)

            new_target_frames = env_new.shape[1]
            formants_new_warped = {}
            for k, track in formants_new.items():
                formants_new_warped[k] = stretch_prefix_formant_track(track, pre_frames, vel_factor)
                f = formants_new_warped[k]
                if len(f) < new_target_frames:
                    f = np.pad(f, (0, new_target_frames - len(f)), mode='edge')
                elif len(f) > new_target_frames:
                    f = f[:new_target_frames]
                formants_new_warped[k] = f
            formants_new = formants_new_warped

            f0_new   = stretch_prefix_1d(f0_new,   pre_samples, vel_factor)
            mask_new = stretch_prefix_1d(mask_new, pre_samples, vel_factor)

        # thank you straycat
        n_total   = len(f0_new)
        t_samples = np.arange(n_total) / sr

        # self.bend is cents; /100 = semitones; + self.pitch = absolute MIDI curve
        pitch_semi = self.bend.astype(np.float64) / 100.0 + self.pitch_m

        # pitch offset flag
        t_cents = self.flags.get('t', 0)
        if t_cents:
            pitch_semi = pitch_semi + (t_cents / 100.0)

        # tick times: one tick = 1/96 quarter‐note; quarter-note = 60/tempo seconds
        # so tick_dt = 60/(tempo*96)
        tick_dt    = 60.0 / (self.tempo * 96.0)
        t_pitch    = np.arange(len(pitch_semi)) * tick_dt

        pitch_interp = gf.interp1d(t_pitch, pitch_semi, kind='linear', fill_value='extrapolate')
        t_clamped = np.clip(t_samples, t_pitch[0], t_pitch[-1])
        midi_curve = pitch_interp(t_clamped)
        f0_new = mask_new * midi_to_hz(midi_curve)

        # dummy y-length (goofer doesnt care)
        y_len_new = np.empty(mask_new.shape, dtype=np.bool_)

        # synthesis

        logging.info('Synthesizing')
        _, harmonic, aper_uv, aper_bre = gf.synthesize(
            env_new,
            f0_new,
            mask_new,
            y_len_new,
            sr,
            formant_shift=self.formant_shift,
            formants=formants_new,
            F1_shift=self.F1_shift,
            F2_shift=self.F2_shift,
            F3_shift=self.F3_shift,
            F4_shift=self.F4_shift,
            f0_jitter=self.f0_jitter,
            f0_jitter_strength=self.f0_jitter_strength,
            volume_jitter=self.volume_jitter,
            volume_jitter_strength_harm=self.volume_jitter_strength,
            volume_jitter_strength_breath=self.volume_jitter_strength * 2,
            add_subharm=self.add_subharm,
            subharm_weight=self.subharm_weight,
            subharm_semitones=12,
            subharm_vibrato=True,
            subharm_vibrato_rate=75,
            subharm_vibrato_depth=3,
            subharm_vibrato_delay=0.01,
            cut_subharm_below_f0=False,
            subharm_f0_jitter=0,
        )

        # apply tension if not zero
        if self.tension != 0:
            abs_ten = abs(self.tension)
            lp_factor = 2 - abs_ten
            hp_factor = abs_ten
            if self.tension < 0:
                abs_ten = abs(self.tension)
                order = int(np.round(1 + (abs_ten * 4)))
                order = np.clip(order, 1, 6)
                lp_factor = 2.0 - abs_ten * 0.75
                rms_before = np.sqrt(np.mean(harmonic**2) + 1e-9)
                harmonic = dynamic_butter_filter(harmonic, f0_new, sr, lp_factor, order=order, btype='lowpass')
                rms_after = np.sqrt(np.mean(harmonic**2) + 1e-9)
                if rms_after > 0:
                    harmonic *= rms_before / rms_after
                aper_bre = dynamic_butter_filter(aper_bre, f0_new, sr, abs_ten, order=4, btype='highpass')
            else:
                highpassed = dynamic_butter_filter(harmonic, f0_new, sr, hp_factor * 4, order=4, btype='highpass')
                boosted = highpassed * (1.0 + abs_ten * 20)
                harmonic += boosted
                aper_bre = dynamic_butter_filter(aper_bre, f0_new, sr, lp_factor / 0.5, order=6, btype='lowpass')
                aper_bre *= 1.0 - abs_ten

        # apply volume and write output
        breath_scaled  = aper_bre * self.breathiness_mix
        uv_scaled      = aper_uv * self.unvoiced_mix
        harmonic_scaled= harmonic * self.harmonic_mix

        voiced_mix = harmonic_scaled + breath_scaled

        target_peak = 0.98

        if self.tension != 0:
            max_amplitude = np.max(np.abs(voiced_mix))
            if max_amplitude > 0:
                voiced_mix = voiced_mix / max_amplitude * target_peak

        custom_mix = voiced_mix + uv_scaled

        max_amplitude = np.max(np.abs(custom_mix))
        if max_amplitude > 0:
            custom_mix = custom_mix / max_amplitude * target_peak

        out = custom_mix * self.volume

        logging.info(f'Writing {self.out_file}')
        sf.write(self.out_file, out, sr)

def split_arguments(input_string):
    otherargs = input_string.split(' ')[-11:]
    file_path_strings = ' '.join(input_string.split(' ')[:-11])
    parts = re.findall(r'([^\s]+\.wav)', file_path_strings)
    if len(parts) < 2:
        raise ValueError('Missing .wav file paths in POST string')
    first_file, second_file = parts[:2]
    return [first_file, second_file] + otherargs

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer): pass

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        post_data_string = post_data.decode('utf-8')
        try:
            args = split_arguments(post_data_string)
            GooferResampler(*args)
        except Exception as e:
            trcbk = traceback.format_exc()
            self.send_response(500)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(f'An error occurred.\n{trcbk}'.encode('utf-8'))
            return
        self.send_response(200)
        self.end_headers()

def run(server_class=ThreadedHTTPServer, handler_class=RequestHandler, port=8572):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting HTTP server on port {port}...')
    httpd.serve_forever()

version = 'v1.4'
help_string = (
    'Usage:\n'
    '  SillySampler.py in.wav out.wav pitch velocity flags\n'
    '           offset(ms) length(ms) consonant(ms) cutoff(ms)\n'
    '           volume(%) modulation(%) !tempo pitch_string\n\n'
    'Example:\n'
    '  SillySampler.py in.wav out.wav C4 100 g0 0 1000 0 700 100 0 !120 AA'
)

if __name__ == '__main__':
    logging.info(f'SillySampler {version}')
    if len(sys.argv) == 1:
        try:
            run()
        except Exception as e:
            if isinstance(e, TypeError):
                logging.info(help_string)
            else:
                raise e
    else:
        args = sys.argv[1:]
        logging.info(f'Args: {args} (count={len(args)})')
        try:
            if len(args) == 1:
                # SLAY: Folder mode!!! :D Just extract features, nothing else.
                input_path = Path(args[0])
                if input_path.exists():
                    logging.info(f'Scanning folder: {input_path}')
                    extract_features_recursive(input_path)
                    logging.info(f'Done extracting features.')
                    input("Press Enter to exit... ")
                    sys.exit(0)
                else:
                    raise FileNotFoundError(f"Folder or file not found: {input_path}")
            if len(args) < 13:
                raise TypeError(f'Expected 13 arguments but got {len(args)}')
            GooferResampler(*args)
        except TypeError as e:
            logging.error('Argument parsing failed: %s', str(e))
            logging.error(help_string)
            sys.exit(1)
        except Exception:
            logging.exception('Failed to render')
            sys.exit(1)
