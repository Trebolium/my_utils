import argparse, os, pdb, librosa, pickle, torch, sys
import numpy as np
import soundfile as sf
from gen_spect import audio_to_mel
from torch.backends import cudnn
sys.path.insert(1, '/homes/bdoc3/my_data/autovc_data')
from hparams import hparams
from synthesis import build_model
from synthesis import wavegen
import soundfile as sf

parser = argparse.ArgumentParser()

parser.add_argument('--src_dir', type=str, default='wav2mel2wvnt_src_dir')
parser.add_argument('--dst_dir', type=str, default='wav2mel2wvnt.py_dst_dir')
parser.add_argument('--which_cuda', type=int, default=0)
parser.add_argument('--percent', type=float, default=1.0)
config = parser.parse_args()

device = torch.device(f'cuda:{config.which_cuda}' if torch.cuda.is_available() else 'cpu')
wn_model = build_model().to(device)
checkpoint = torch.load("/homes/bdoc3/my_data/autovc_data/checkpoint_step001000000_ema.pth")
wn_model.load_state_dict(checkpoint["state_dict"])
wn_model.to(device)

wav_root = config.src_dir
_, _, file_list = next(os.walk(wav_root))
mel_list = []
for file_path in file_list:
    if not file_path.startswith('.'):
        if file_path.endswith('.wav'):
            audio, rate = sf.read(os.path.join(wav_root, file_path))
            print('making spmel for', file_path)
            mel = audio_to_mel(audio, rate)
            print('making wvnt audio for', file_path)
            breakpoint = int(mel.shape[0]*config.percent)
            waveform = wavegen(wn_model, config.which_cuda, c=mel[:breakpoint])
            new_file_name = os.path.basename(file_path)[:-4] +'_wvnt' +'.wav'
            sf.write(os.path.join(config.dst_dir, new_file_name), waveform, samplerate=16000)

print('done')
