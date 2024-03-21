from flask import Flask, request
import os
import onetrain as ot

#from flasktrain import train1key ??? 
# 내 목표 : 같은 디렉토리에 있는 train1key 함수를 post 요청을 보내 실행 시켜야 함.

app = Flask(__name__)


# 테스트용
# @app.route('/hello', methods=['GET'])
# def hello():
#     return 'Hello world!'


# input 값 정리
# exp_dir1 : 만들어지는 목소리 모델명 ex) gosegu, jeongmin (입력받아야함)
# sr2 :  "48k" 고정 (목표 샘플링률)
# if_f0_3 : True 고정 (이 설정 체크 안하면 음 높낮이 X)
# trainset_dir4 : 훈련할 음악파일이 포함된 경로 ex) /home/j-j10e201/jeongmin
# spk_id5 :  0 고정  (화자 ID 지정하는 변수인데 0으로 둬도 문제 X)
# np7 : 16 고정 (할당되는 CPU 수)
# f0method8 : harvest 고정 (음성모델 학습기법 선택, harvest가 전반적으로 무난)
# save_epoch10 : 5 고정 (저장 빈도)
# total_epoch11 : 100 고정 (전체 epoch 수)
# batch_size12 : 8 고정 (그래픽 카드 별 batch size)
# if_save_latest13 : "Yes" 고정 (是로 설정해야 불필요한 파일 저장되지 않음)
# pretrained_G14 : assets/pretrained_v2/f0G40k.pth 고정 (미리 훈련된 G모델 경로)
# pretrained_D15 : assets/pretrained_v2/f0D40k.pth 고정 (미리 훈련된 D 모델 경로)
# gpus16 : 0 고정   (사용할 그래픽 카드 설정, 우리 서버 기준 0 박혀 있음)
# if_cache_gpu17 : "No" 고정  (Yes 설정하면 오류 발생 위험있음)
# if_save_every_weights18 : "No 고정" (저장시 마다 모델 저장 여부 결정, Yes 설정하면 용량 낭비)
# version19 :  "v2" 고정   (모델 버전 선택)
# gpus_rmvpe : "0" 고정 (harvest 버전 사용하면 신경 쓸 필요 없는 값)
# train 함수 실행

@app.route('/training', methods=['POST'])
def training():
    
    # 여기서 요청에 따라 다른 값 주면 됨
    ot.train1key("jeongmingod",
                       "48k",
                       True,
                       "/home/j-j10e201/jeongmin",
                       0,
                       16,
                       "harvest",
                       5,
                       100,
                       8,
                       "Yes",
                       "assets/pretrained_v2/f0G40k.pth",
                       "assets/pretrained_v2/f0D40k.pth",
                       0,
                       "No",
                       "No",
                       "v2",
                       0);
    
    return 'Success' 


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)





# @app.route('/train_start_all', methods=['POST'])
# def train_start_all():
#     # 여기서 POST 요청의 데이터를 가져와서 train1key 함수에 필요한 매개변수로 전달합니다.
#     exp_dir1 = request.form.get('exp_dir1')
#     sr2 = request.form.get('sr2')
#     if_f0_3 = request.form.get('if_f0_3')
#     trainset_dir4 = request.form.get('trainset_dir4')
#     spk_id5 = request.form.get('spk_id5')
#     np7 = request.form.get('np7')
#     f0method8 = request.form.get('f0method8')
#     save_epoch10 = request.form.get('save_epoch10')
#     total_epoch11 = request.form.get('total_epoch11')
#     batch_size12 = request.form.get('batch_size12')
#     if_save_latest13 = request.form.get('if_save_latest13')
#     pretrained_G14 = request.form.get('pretrained_G14')
#     pretrained_D15 = request.form.get('pretrained_D15')
#     gpus16 = request.form.get('gpus16')
#     if_cache_gpu17 = request.form.get('if_cache_gpu17')
#     if_save_every_weights18 = request.form.get('if_save_every_weights18')
#     version19 = request.form.get('version19')
#     gpus_rmvpe = request.form.get('gpus_rmvpe')
    
#     # train1key 함수 호출하여 실행합니다.
#     result = train1key(
#         exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, np7, f0method8, save_epoch10,
#         total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15,
#         gpus16, if_cache_gpu17, if_save_every_weights18, version19, gpus_rmvpe
#     )
    
#     return result  # train1key 함수의 실행 결과를 반환합니다.