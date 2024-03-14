from pyannote.audio import Pipeline
  
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

import torch
pipeline.to(torch.device("cuda"))

audio_path = "b.wav"
sample_rate = 16000

name = audio_path[:-4]

# inference on the whole file
diarization = pipeline(audio_path)



import librosa
import soundfile as sf

y, sr = librosa.load(audio_path, sr=sample_rate)

print(f"size: {y.size}")

duration = librosa.get_duration(y=y, sr=sample_rate)
print(f"Audio Duration: {duration}")

NUM_SPEAKERS = 0

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start}s stop={turn.end}s speaker_{speaker}")
    if int(speaker[8:]) > NUM_SPEAKERS:
        NUM_SPEAKERS = int(speaker[8:])
    if round(turn.end * sr) > duration:
        duration = round(turn.end * sr)

for i in range(0, NUM_SPEAKERS + 1):
    ny = [0.0 for j in range(round(sr*duration))]

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if int(speaker[8:]) == i:
            # start = round(seg['start'] * sr)
            # end = round(seg['end'] * sr)
            start = round(turn.start * sr)
            end = round(turn.end * sr)
            for j in range(start, end):
                ny[j] = y[j]

    sf.write(f"output\\{name}\\{name}_label{i}.wav", ny, sr, format='WAV')

# # inference on an excerpt
# from pyannote.core import Segment
# excerpt = Segment(start=2.0, end=5.0)

# from pyannote.audio import Audio
# waveform, sample_rate = Audio().crop("file.wav", excerpt)
# pipeline({"waveform": waveform, "sample_rate": sample_rate})