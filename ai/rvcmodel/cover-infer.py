from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pytube import YouTube
from pydub import AudioSegment
from starlette.requests import Request

import myinfer as mif
import numpy as np

import os
import librosa
import requests

cur_dir = os.getcwd()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


cover_path = f"{cur_dir}/cover"
model_path = f"{cur_dir}/model"


class Request(BaseModel):
    youtubeURL: str
    userCode: int
    modelCode: int
    coverCode: str
    pitch: int


def create_cover(request):
    coverCode = request.coverCode

    download(request)
    vocals, bass, drums, other = split(request)
    inferred = infer(request, vocals)
    output = mixing(request, inferred, bass, drums, other)

    # file = open(output, 'rb')
    # upload = {'file': file}

    # response = requests.post(f"https://j10e201.p.ssafy.io/api/cover/{coverCode}", files=upload)


# AI 커버 제작 요청
@app.post('/rvc/cover')
def cover(request: Request, background_tasks: BackgroundTasks):
    background_tasks.add_task(create_cover, request)
    return {"status": 200, "message": "Process accepted"}
    

# 유튜브 음원 다운로드
def download(request: Request):
    youtubeURL = request.youtubeURL
    userCode = request.userCode
    coverCode = request.coverCode

    yt = YouTube(youtubeURL)

    print(f'start download youtube : {yt.title}')

    filename = f"{coverCode}.wav"
    path = f"{cover_path}/{userCode}"

    filter = yt.streams.filter(adaptive=True, file_extension="mp4", only_audio=True)
    filter.first().download(filename=filename,
                            output_path=f"{path}/{coverCode}/origin")


# 음원에서 보컬, MR 분리
def split(request: Request):
    userCode = request.userCode
    coverCode = request.coverCode

    print('start split vocal & MR')

    audio_path = f"{cover_path}/{userCode}/{coverCode}/origin/{coverCode}.wav"
    # spl = f"python3 -m spleeter separate -p spleeter:2stems -o {path} {audio_path}"
    spl = f"spleeter separate -p spleeter:4stems -o {cover_path}/{userCode} {audio_path}"
    os.system(spl)

    vocals = f"{cover_path}/{userCode}/{coverCode}/vocals.wav"
    bass = f"{cover_path}/{userCode}/{coverCode}/bass.wav"
    drums = f"{cover_path}/{userCode}/{coverCode}/drums.wav"
    other = f"{cover_path}/{userCode}/{coverCode}/other.wav"

    return vocals, bass, drums, other


# 음원 추론(목소리 변환)
def infer(request: Request, vocal: str):
    userCode = request.userCode
    coverCode = request.coverCode
    modelCode = request.modelCode

    pth = f"{model_path}/{modelCode}/pth.pth"
    # index = f"{model_path}/{modelCode}/index.index"
    index = None

    output = f"{cover_path}/{userCode}/{coverCode}/inferred.wav"

    print("start infer")

    inf = mif.infer(f0up_key=request.pitch,
                    input_path=vocal,
                    index_path=index,
                    f0method="rmvpe",
                    opt_path=output,
                    model_path=pth,
                    index_rate=0,
                    device="cuda:0",
                    is_half=True)
    
    inf.run()

    inferred = f"{cover_path}/{userCode}/{coverCode}/inferred.wav"
    return inferred


# MR + 변환된 음성 믹싱
def mixing(request: Request, inferred: str, bass: str, drums: str, other: str):
    userCode = request.userCode
    coverCode = request.coverCode
    pitch = request.pitch

    output = f"{cover_path}/{userCode}/{coverCode}/{coverCode}.mp3"

    print("start mixing")

    drums_audio = AudioSegment.from_wav(drums)
    bass_audio = AudioSegment.from_wav(bass)
    other_audio = AudioSegment.from_wav(other)
    infer_audio = AudioSegment.from_wav(inferred)

    if pitch != 0:
        bass_audio = pitch_shift(bass_audio, pitch)
        other_audio = pitch_shift(other_audio, pitch)

    mixed_audio = drums_audio.overlay(bass_audio).overlay(other_audio).overlay(infer_audio)

    print(f"write result {output}")

    mixed_audio.export(output, format="mp3")

    return output


## 피치 조절
def pitch_shift(sound, n_steps):
    y = np.frombuffer(sound._data, dtype=np.int32).astype(np.float64)/2**15
    y = librosa.effects.pitch_shift(y, sound.frame_rate, n_steps=n_steps)
    a  = AudioSegment(np.array(y * (1<<15), dtype=np.int32).tobytes(), frame_rate = sound.frame_rate, sample_width=4, channels = 1)
    return a


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app, host='0.0.0.0', port=7866)