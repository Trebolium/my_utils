import librosa, torchaudio
import soundfile as sf
import os, pdb, pickle
import numpy as np

mel_list = pickle.load(open('./louderBreathy.pkl','rb'))

for mel, file_name in mel_list:
    mel_rot = np.rot90(mel)
    stft = librosa.feature.inverse.mel_to_stft(mel_rot, sr=16000)
    audio = librosa.griffinlim(stft)
    sf.write('./louderBreathy/' +file_name[:-4] +'_LibGrifLim' +'.wav', audio, samplerate=16000)

    inverter = torchaudio.transforms.InverseMelScale(513, 80)
    t_stft = inverter(torch.from_numpy(mel_rot.copy()).float())
    griflimmer = torchaudio.transforms.GriffinLim(n_fft=1024, hop_length=256)
    t_audio = griflimmer(t_stft)
    print('mel: ', mel.shape, 'stft: ', stft.shape, 'audio: ', audio.shape, 't_stft: ', t_stft.shape, 't_audio: ', t_audio.shape)
    sf.write('./louderBreathy/' +file_name[:-4] +'_torchGrifLim' +'.wav', t_audio, samplerate=16000)

