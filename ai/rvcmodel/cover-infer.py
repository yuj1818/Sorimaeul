from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.requests import Request
from AI_Cover_Creator import AI_Cover

import os, shutil
import requests
import logging

logger = logging.getLogger(__name__)

cur_dir = os.getcwd()
cover_path = f"{cur_dir}/cover"

app = FastAPI()

origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Request(BaseModel):
    youtubeURL: str
    userCode: int
    modelCode: int
    coverCode: int
    coverName: str
    pitch: int


def create_cover(request):
    userCode = request.userCode
    coverCode = request.coverCode
    coverName = request.coverName

    output = AI_Cover(request).create()

    file = open(output, 'rb')
    upload = {'file': file}

    # 커버 업로드
    response = requests.post(f"https://j10e201.p.ssafy.io/api/cover/{coverCode}", files=upload)
    logger.info(response)

    # 알림 전송
    response = requests.post(f"https://j10e201.p.ssafy.io/api/sse/notify",
                             data={"userCode":userCode,
                                   "data":f'AI 커버 "{coverName}" 생성 완료!'})
    logger.info(response)

    # 폴더 삭제
    shutil.rmtree(f"{cover_path}/{userCode}/{coverCode}")
    logger.info(f"Remove {cover_path}/{userCode}/{coverCode}")


# AI 커버 제작 요청
@app.post('/rvc/cover')
def cover(request: Request, background_tasks: BackgroundTasks):
    background_tasks.add_task(create_cover, request)
    return {"status": 200, "message": "Process accepted"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app, host='0.0.0.0', port=7866)