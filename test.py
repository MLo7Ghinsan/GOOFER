import soundfile as sf
import os
import numpy as np
import GOOFER as gf

_ = gf.pulse_train_numba(np.zeros(16, dtype=np.float32), 44100) #warmup for benchmark

input_file = 'test.wav'

noise_type = 'white'  #'white' or 'brown' or 'pink'

stretch_factor = 1.0

pitch_shift = 1.5

formant_shift = 1.0

input_name = os.path.splitext(input_file)[0]
y, sr = sf.read(input_file)
if y.ndim > 1:
    y = y.mean(axis=1)
import time

start_time = time.time()
t0 = time.time()
env_spec, f0_interp, voicing_mask, formants, _ = gf.extract_features(y, sr)
t1 = time.time()
print(f"Feature extraction took: {t1 - t0:.3f} seconds")

#for i in range(20):
#    reconstruct, harmonic, aper_uv, aper_bre= gf.synthesize(
#        env_spec, f0_interp, voicing_mask, y, sr,
#        stretch_factor=stretch_factor,
#        pitch_shift=pitch_shift, formant_shift=formant_shift,
#        formants=formants, normalize=0)

t2 = time.time()
reconstruct, harmonic, aper_uv, aper_bre= gf.synthesize(
    env_spec, f0_interp, voicing_mask, y, sr,
    stretch_factor=stretch_factor,
    pitch_shift=pitch_shift, formant_shift=formant_shift,
    formants=formants, normalize=1, breath_strength=0.05, uv_strength=0.4)
t3 = time.time()
print(f"Synthesis took: {t3 - t2:.3f} seconds")

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
reconstruct_wav = f'{input_name}_reconstruct_numba.wav'
sf.write(reconstruct_wav, reconstruct, sr)
print(f'Reconstructed audio saved: {reconstruct_wav}')
