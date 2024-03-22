from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from pydantic import BaseModel
from Dubbing_Video_Creator import Infer

import os, shutil
import requests
import logging

logger = logging.getLogger(__name__)

cur_dir = os.getcwd()
dubbing_path = f"{cur_dir}/dubbing"

app = FastAPI()

origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferRequest(BaseModel):
    userCode: int
    modelCode: int
    pitch: int
    file: UploadFile


class DubbingRequest(BaseModel):
    userCode: int


def create_dubbing(request):
    return


# 음성 추론 요청
@app.post("/rvc/infer/{userCode}/{dubCode}/{voiceIndex}/{modelCode}/{pitch}")
async def infer(userCode: int, dubCode: int, voiceIndex: str, modelCode: int, pitch: int, file: UploadFile):
    file_path = f"{dubbing_path}/{userCode}/{dubCode}/{voiceIndex}/{modelCode}"
    file_name = "voice_origin.wav"

    try:
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        logger.info(f"File write : {file_path}/{file_name}")
        content = await file.read()
        with open(os.path.join(file_path, file_name), "wb+") as fp:
            fp.write(content)

        # 음성 추론 시작
        inf = Infer.infer(pitch, modelCode, file_path, file_name)

        # 파일 응답
        return FileResponse(inf, media_type="audio/wav", filename=file_name)
    except Exception as e:
        logger.info(f"Error occured: {e}")
        return {"status": 500,
                "message": "음성 추론에 실패했습니다."}


# 더빙 영상 제작 요청
@app.post("/rvc/dubbing")
def dubbing(request: DubbingRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(create_dubbing, request)
    return {"status": 200, "message": "Process accepted"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app, host='0.0.0.0', port=7868)