import sys, librosa, os, pdb
if os.path.abspath('.../my_utils') not in sys.path: sys.path.insert(1, os.path.abspath('.../my_utils'))
from audio.editor import desilence_concat_audio
from my_os import recursive_file_retrieval
import pyworld as pw
from pydub import AudioSegment
from tqdm import tqdm
import soundfile as sf

src_ds = sys.argv[1]
dst_ds = sys.argv[2]
f_ext = sys.argv[3]
sr = int(sys.argv[4])

_, all_files = recursive_file_retrieval(src_ds)
pdb.set_trace()
all_audio_files = [file for file in all_files if file.endswith(f_ext)]
if not os.path.exists(dst_ds):
    os.mkdir(dst_ds)

for file in tqdm(all_audio_files):
    dst_fpath = os.path.join(dst_ds, os.path.basename(file[:-4] +'.wav'))
    if os.path.exists(dst_fpath):
        continue
    y, _ = librosa.load(file, sr=sr)
    # y = AudioSegment.from_file(file)
    y = desilence_concat_audio(y, sr)
    print(f'{file} processed')
    # AudioSegment.from_file("never_gonna_give_you_up.mp4", "mp4")
    sf.write(dst_fpath, y, samplerate=sr)
    # librosa.output.write_wav(dst_fpath, y, sr)

