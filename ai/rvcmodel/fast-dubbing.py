from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

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


root_path = "/home/ubuntu/sorimaeul-data/dub"


class Request(BaseModel):
    userCode: int
    dubCode: int
    videoSourceCode: int
    inferredVoice: List[bool]


# 더빙 영상 제작
def create_dubbing(request):
    
    return


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
