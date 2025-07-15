import soundfile as sf
import os
import numpy as np
import GOOFER as gf


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

add_subharm = False

f0_jitter = False

input_name = os.path.splitext(input_file)[0]
y, sr = sf.read(input_file)
if y.ndim > 1:
    y = y.mean(axis=1)

env_spec, f0_interp, voicing_mask, formants = gf.extract_features(y, sr)

reconstruct, harmonic, aper_uv, aper_bre= gf.synthesize(
    env_spec, f0_interp, voicing_mask, y, sr,
    noise_type=noise_type, stretch_factor=stretch_factor,
    pitch_shift=pitch_shift, formant_shift=formant_shift,
    formants=formants, F1_shift=F1, F2_shift=F2, F3_shift=F3, F4_shift=F4,
    f0_jitter=f0_jitter, volume_jitter=volume_jitter, add_subharm=add_subharm)

reconstruct_wav = f'{input_name}_reconstruct.wav'
sf.write(reconstruct_wav, reconstruct, sr)
print(f'Reconstructed audio saved: {reconstruct_wav}')

save_feature = True
feature_stat = 'half'
if save_feature:
    if feature_stat == 'half':
        env_spec = env_spec.astype(np.float16)
        f0_interp = f0_interp.astype(np.float16)
        voicing_mask = voicing_mask.astype(np.float16)
    else:
        env_spec = env_spec.astype(np.float32)
        f0_interp = f0_interp.astype(np.float32)
        voicing_mask = voicing_mask.astype(np.float32)

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

env_spec, f0_interp, voicing_mask, formants, sr, y_len = gf.load_features("pjs001_singing_seg001_features.npz")

reconstruct, harmonic, aper_uv, aper_bre= gf.synthesize(
    env_spec, f0_interp, voicing_mask, np.zeros(y_len, dtype=np.float32), sr,
    noise_type=noise_type, stretch_factor=stretch_factor,
    pitch_shift=pitch_shift, formant_shift=formant_shift,
    formants=formants, F1_shift=F1, F2_shift=F2, F3_shift=F3, F4_shift=F4,
    f0_jitter=f0_jitter, volume_jitter=volume_jitter, add_subharm=add_subharm)

reconstruct_wav = f'{input_name}_fureatures_reconstruct.wav'
sf.write(reconstruct_wav, reconstruct, sr)
print(f'Reconstructed audio saved: {reconstruct_wav}')
