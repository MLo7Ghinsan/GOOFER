import sys, os, logging, re, traceback
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import numpy as np
import soundfile as sf
import scipy.interpolate as interp
from scipy.interpolate import Akima1DInterpolator

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

        self.formant_shift = 1.0 + (self.flags.get('g', 0) / 200.0)
        self.F1_shift = 1.0 + (self.flags.get('fa', 0) / 100.0)
        self.F2_shift = 1.0 + (self.flags.get('fb', 0) / 100.0)
        self.F3_shift = 1.0 + (self.flags.get('fc', 0) / 100.0)
        self.F4_shift = 1.0 + (self.flags.get('fd', 0) / 100.0)

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

        self.render()

    def render(self):
        feat = self.in_file.with_name(f'{self.in_file.stem}_features.npz')
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
                
        # thank you straycat
        n_total   = len(f0_new)
        t_samples = np.arange(n_total) / sr

        # self.bend is cents; /100 = semitones; + self.pitch = absolute MIDI curve
        pitch_semi = self.bend.astype(np.float64) / 100.0 + self.pitch_m

        # tick times: one tick = 1/96 quarter‐note; quarter-note = 60/tempo seconds
        # so tick_dt = 60/(tempo*96)
        tick_dt    = 60.0 / (self.tempo * 96.0)
        t_pitch    = np.arange(len(pitch_semi)) * tick_dt

        pitch_interp = Akima1DInterpolator(t_pitch, pitch_semi)
        t_clamped = np.clip(t_samples, t_pitch[0], t_pitch[-1])
        midi_curve = pitch_interp(t_clamped)
        f0_new = mask_new * midi_to_hz(midi_curve)

        # dummy y-length (goofer doesnt care)
        y_len_new = np.empty(f0_new.shape, dtype=np.bool_)

        # synthesis
        logging.info('Synthesizing')
        reconstruct, harmonic, aper_uv, aper_bre = gf.synthesize(
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
        )

        # apply volume and write output
        out = reconstruct * self.volume
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

version = 'v1.0'
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

