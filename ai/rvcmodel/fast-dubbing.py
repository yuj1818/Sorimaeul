from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

import os, shutil
import requests
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cur_dir = os.getcwd()
root_path = f"{cur_dir}/sorimaeul-data/dub"


class Request(BaseModel):
    userCode: int
    dubCode: int
    videoSourceCode: int
    inferredVoice: List[int]
    dubName: str


# 더빙 영상 제작
def create_dubbing(request: Request):
    userCode = request.userCode
    dubCode = request.dubCode
    videoSourceCode = request.videoSourceCode
    inferredVoice = request.inferredVoice
    dubName = request.dubName

    try:
        video_file = f"{root_path}/source_{videoSourceCode}/origin/video/video.mp4"
        audio_origin_path = f"{root_path}/source_{videoSourceCode}/origin/audio"
        audio_recorded_path = f"{root_path}/source_{videoSourceCode}/user_{userCode}/Unconverted"
        audio_inferred_path = f"{root_path}/source_{videoSourceCode}/user_{userCode}/Converted"

        output_path = f"{root_path}/source_{videoSourceCode}/dub_{dubCode}/dub_{dubCode}.mp4"

        logger.info(f"Start create video: {root_path}/source_{videoSourceCode}/dub_{dubCode}/dub_{dubCode}.mp4")

        video = VideoFileClip(video_file)
        audio = video.audio

        combine_audio = [audio]

        for idx, flag in enumerate(inferredVoice):
            if idx == 0:
                pass

            if flag == 0:
                add_audio = f"{audio_origin_path}/{idx}.wav"
            elif flag == 1:
                add_audio = f"{audio_recorded_path}/{idx}.wav"
            elif flag == 2:
                add_audio = f"{audio_inferred_path}/{idx}.wav"
            
            combine_audio.append(AudioFileClip(add_audio))
            
        audio = CompositeAudioClip(combine_audio)

        video = video.set_audio(audio)

        video.write_videofile(output_path, codec='libx264', audio_codec='aac')
        logger.info(f"Write video {output_path}")

        response = requests.post("https://j10e201.p.ssafy.io/api/dub/save",
                                 json={"dubCode":dubCode, "path":output_path})
        response.raise_for_status()
        
        logger.info(f"Response status {response.status_code}")
        msg = f'더빙 영상 "{dubName}" 생성이 완료되었습니다.'

    except Exception as e:
        logger.info(f"Error occured: {e}")
        msg = f'더빙 영상 "{dubName}" 생성에 실패했습니다.'
    
    finally:
        sendNotification(userCode, dubCode, msg)


# 알림 전송
def sendNotification(userCode, targetCode, msg):
    logger.info("Send notification")
    try:
        response = requests.post(f"https://j10e201.p.ssafy.io/api/sse/notify",
                                 json={"userCode":userCode,
                                       "name":"dubbing",
                                       "data": {
                                           "targetCode":targetCode,
                                           "content":msg
                                           }})
        logger.info(f"Response status {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.info(f"Error occurred: {e}")


@app.get("/create-dubbing")
def create(request, background_tasks: BackgroundTasks):
    background_tasks.add_task(create_dubbing, request)
    return {"status": 200, "message": "Process accepted"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app, host='0.0.0.0', port=7000)
