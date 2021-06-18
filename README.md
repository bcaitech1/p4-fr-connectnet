### 부스트캠프 AI Tech - Team Project 수식 인식기

![logo](https://user-images.githubusercontent.com/24247768/122545863-4c966400-d069-11eb-9da3-bbb8427b057b.png)

|Task|Date|Team|
|---|---|---|
|수식 이미지를 latex 포맷의 텍스트로 변환하는 문제|2021.05.24 ~ 2021.06.15|5조 ConnectNet|

> `P stage 4 대회 진행 과정과 결과를 기록한 Team Git repo 입니다. 대회 특성상 수정 혹은 삭제된 부분이 존재 합니다`


---

#### ✔️ OCR Task
   * Rank : 7
   * LB: 0.5639


### 📋 Table of content

[Team 소개](#Team)<br>
[Gound Rule](#rule)<br>
[실험노트](https://docs.google.com/spreadsheets/d/1v_ZMKii5nt6VgrtCVA-bue42jWa5wpcJWnHMoFL9OUE)<br>
[설치 및 실행](#Install) <br>
[ocRec 수식 인식 프로그램] (https://github.com/bcaitech1/p4-fr-connectnet/tree/main/ocrec)

#### [수식 인식기](#ocr)

1.1 [대회 전략](#strategy)<br>
1.2 [Model](#model)<br>
1.3 [Ensemble](#ensemble)<br>
1.4 [실패한 부분](#fail)<br>

---



### 🌏Team - ConnectNet <a name = 'Team'></a>

* 김현우 [![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://github.com/philgineer/)](https://github.com/akorea)
* 배철환 [![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://github.com/philgineer/)](https://github.com/bcc0830)
* 서준배 [![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://github.com/philgineer/)](https://github.com/deokisys)
[![Blog Badge](http://img.shields.io/badge/Blog-51a9fe?style=flat&link=https://philgineer.com/)](https://deokisys.github.io/)
* 윤준호 [![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://github.com/philgineer/)](https://github.com/philgineer) [![Blog Badge](http://img.shields.io/badge/Blog-51a9fe?style=flat&link=https://philgineer.com/)](https://philgineer.com/)
* 임기홍 [![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://github.com/philgineer/)](https://github.com/GihongYim)
* 조호성 [![Github Badge](https://img.shields.io/badge/-Github-161c22?style=flat&logo=github&link=https://github.com/philgineer/)](https://github.com/chohoseong)

#### Ground rule <a name = 'rule'></a>

1. 공유 코드 작성 시
   * `작성자 표시`
   * `함수` 단위 작성
   * `merge` 시  .py 형식 사용
   
2. 모델 로그 기록
   * `Wandb` 사용

3. 회의 기록 작성
   * [wiki](https://github.com/bcaitech1/p4-fr-connectnet/wiki)
4. 자료 공유 및 연락 - Slack

5. Code review

   * Git에 code review 요청, 
   * Peer session 활용

6. 결과 제출 시

   * 실험 내용 결과 공유 및 기록 


### 설치 및 실행 <a name = 'Install'></a>

* 소스  다운로드 
```shell
git clone https://github.com/bcaitech1/p4-fr-connectnet.git
```

* 설치 
```shell
pip install -r requirements.txt
```


* 학습
* SATRN(LB:0.7888)
    ```shell
    python ./train.py --c config/SATRN.yaml
    ```
* Aster(LB:0.7917)
    ```shell
    python ./train.py --c config/Attention.yaml
    ```

* 평가
  * [trained model 다운로드](https://drive.google.com/drive/folders/1oFh8gIGQ81mEiRwYPa_s8ML_rcVVK8RJ?usp=sharing)
```shell
python ./inference.py --checkpoint aster.pth
```


### 🔍수식인식 <a name = 'ocr'></a>

#### 1. 대회 전략 <a name = 'strategy'></a>

### 1.1 **Task 분석과 접근법 도출**

- 유사한 task인 Scene Text Recognition을 참조하여 SOTA 논문 분석 및 리뷰
- 동일한 task인 논문을 참고해 베이스라인 아키텍처 수정 방향 논의

### 1.2 **다양한 실험을 통해 성능 향상 시도**

- **하이퍼파라미터 튜닝**
    - SATRN의  hidden dimension, filter dimension증가
        - 0.01의 성능 향상
- **모델 앙상블**
- **Penalty 추가**
    - \frac{1}} 처럼 괄호가 맞지 않지 않는 경우가 발생
    - 그러나, 이미 토큰단위에서 loss를 계산하기 때문에 2차적으로 stack을 이용해 1 - (짝이 맞는 괄호 쌍 / 전체 괄호 쌍)을 더해줌으로서 일종의 penalty 부여

        $Loss = 0.8CE + 0.2 ParenPenalty$

- **데이터셋 추가**
    - Aida Dataset을 추가적으로 학습(100000+100000)
    - 더 데이터를 추가하려했으나 서버용량 때문에 추가못함 10만개 == 약 12GB
    - 학습 시간이 오래 걸림
- **Data Augmentation**
    - Image Binarization

        데이터의 noise가 매우 심하여 최대한 숫자와 배경만 남기는 Adaptive Threshold를 통해 노이즈를 감소

        ![_2021-06-16_18 12 31](https://user-images.githubusercontent.com/24247768/122544424-c4fc2580-d067-11eb-8ee9-72a2cbc3305f.png)

        Original

        ![Untitled](https://user-images.githubusercontent.com/24247768/122544456-c9284300-d067-11eb-8dbf-e0d0c0fe0805.png)


        Binarization

    - Random Rotation / Affine(Shear)

        다양한 각도에서 촬영된 다양한 필체로 쓰인 손글씨 데이터를 잘 아우르는 분포를 학습할 수 있도록 다양한 변환을 적용

        ![_2021-06-16_18 12 43](https://user-images.githubusercontent.com/24247768/122544431-c62d5280-d067-11eb-98c0-58bbced6c389.png)

        Rotation

        ![_2021-06-16_18 12 51](https://user-images.githubusercontent.com/24247768/122544434-c6c5e900-d067-11eb-8baa-fdda7f492ff8.png)

        Shear

    - ColorJitter (Bright / Contrast)

        ![_2021-06-16_18 13 00](https://user-images.githubusercontent.com/24247768/122544439-c6c5e900-d067-11eb-905b-bcaf5b9ffb39.png)

        Random Brightness

        ![_2021-06-16_18 13 07](https://user-images.githubusercontent.com/24247768/122544443-c75e7f80-d067-11eb-9740-e37c7ee3acdb.png)

        Random Contrast

- **Outlier Correction**

    ![Untitled 1](https://user-images.githubusercontent.com/24247768/122544444-c7f71600-d067-11eb-938d-be6073ad09e1.png)


    - 가로/세로 가 0.75 보다 작은 경우 이미지의 내부의 글자가 세로로 출력되어 있음

    ![Untitled 2](https://user-images.githubusercontent.com/24247768/122544447-c7f71600-d067-11eb-97d5-4dc5479f7ea5.png)


    - 데이터의 종횡비 (가로 / 세로)가 0.75이하인 경우, 즉 세로가 지나치게 긴 이미지들은 일괄적으로 시계방향 90도 회전 → 0.051 상승
        - ex) 종횡비가 0.75 이상, 0.8이하인 데이터

            ![Untitled 3](https://user-images.githubusercontent.com/24247768/122544449-c88fac80-d067-11eb-85c6-c6e791775649.png)


- Vertical Image는 못맞추는걸로...

    ![Untitled 4](https://user-images.githubusercontent.com/24247768/122544450-c88fac80-d067-11eb-9ccc-e8ae67404622.png)


- Model

     [Paperswithcode](https://paperswithcode.com/sota/object-detection-on-coco) 사이트를 참고해 Scene Text Recognition의 SOTA 모델들을 선택해 테스트 진행

    - ScreenShot

        ![Untitled 5](https://user-images.githubusercontent.com/24247768/122544454-c9284300-d067-11eb-82f1-3720c7b78f8d.png)



#### 2. Model <a name = 'model'></a>

**1) SATRN**

- DenseNet / Transformer (encoder + decoder)
    - LB: 0.7888
    - optimizer : Adam (learning_rate = 5e-4)
    - loss:  CrossEntropyLoss
    - hyperparameters : batch : 16, epochs : 50
    - image_size: (128, 256)
    - 추가로 시도한 것
        - Dense layer depth 증가
        - 다양한 Augmentation 적용
        - positionalencoding2D 을 adaptive2DpositionEncoder로 개선
        - hidden dimension, filter dimension 증가

**2) Aster**

- CNN / Bi-LSTM / LSTM
    - LB : 0.7917
    - loss : CrossEntropy
    - optimizer : Adam (learning_rate = 5e-4)
    - hyperparameters : batch : 32, epochs : 50
    - image_size: (80, 320)
    - 추가로 시도한 것
        - Deformable conv layer

            주어진 데이터셋에는 기울어진 수식들이 많이 들어있었음.

            기존의 논문에서는 STN을 통과하여 이미지를 정렬시킴 → 연산량이 많다 

            마지막 3 block에서 conv layer를 Deformable conv layer로 바꾸어 성능 향상을 봄.

**3) CSTR**

- Naive CNN / CBAM & SADM / Multiple Linear
    - LB : None
    - Valid Acc : 0.28 ~ 0.31
    - optimizer : AdaDelta (learning_rate = 0.0 ~ 1.0 CosineAnnealingWarmUp)
    - loss : LabelSmooth (ratio = 0.1)
    - hyperparameter : batch 100, epochs : 50
    - image_size : (48, 192)
    - 추가로 시도한 것
        - 실험초반 오버피팅 이슈 발생 → dropout(p = 0.1), weight_decay (1e-3) 설정
        - 이후 오버피팅은 일어나지 않았으나 성능 이슈 발생
        - CNN Layer의 dim을 2배씩 늘려 전체적 파라미터를 2배로 size up → 실패



#### 3. Ensemble <a name = 'ensemble'></a>

- 서로 다른 방향성을 가진 transformer기반의 Satrn과 attention기반 모델을 앙상블
    - SATRN, Attention
- 멀티스케일 러닝의 효과를 보기위하 각 모델마다 다른 크기의 이미지입력을 사용
- 서로 다른 seed를 이용하여 서로 다른 train셋으로 학습하는 효과를 이용
- 앙상블1 : SATRN(128,384), SATRN(128,256), Aster(80, 320)
    - 싱글 모델 보다 성능이 떨어짐 (LB : 0.74)
    - 128, 128 이미지로 동일하게 inference 하여 성능이 떨어졌음
    - 앙상블1 의 문제점을 보완하기 위해 앙상블 2를 시도
- 앙상블2 : SATRN(128,384), SATRN(128,256), Aster(80, 320) (TTA적용)
    - 입력 이미지를 학습시 입력한 이미지 크기에 맞게 inference
    - 메모리 폭파 → with torch.no_grad() 넣지 않아 발생한 문제
- 앙상블3 : SATRN (128, 256) + Aster(80, 320)
    - 서버 문제 때문에 LB 점수를 알 수 없음

<br>

#### 4. 잘되지 않았던 것 <a name = 'fail'></a>

- 앙상블
    - 제출방식에 있어 기존방식과 달라 예외를 완벽히 잡지 못함
    - 앙상블 1은 모델의 성능이 하락
    - 앙상블2와 앙상블3은 부스트캠프의 서버의 용량문제로 성능 측정 불가
- 새로운 모델 구현
    - SRN, CSTR 구현 시도
    - Efficientnetv2 FPN Backbone 구현 시도
    - 기존 Attention 베이스라인에서 GRU 에러 fix 후 구현 → LSTM과 큰 차이 없음
- Beam search
    - RNN, LSTM에 있어 각각 token 단위에서 예측하는 걸 보완해 top k개의 후보를 고려하는 알고리즘인데, 모든 word가 한 번에 입력되고 예측하는 transformer의 경우 성능 향상에 도움이 될지 미지수임
    - 같은 이유로 RNN, LSTM에서와 달리 transformer에서는 token 선택 시 사전 확률 - 사후 확률이 달라지기 때문에 구현 난이도가 상당하고, 한다고 하더라도 연산량이 큰 폭으로 증가함
- 시각화
    - attention map을 시각화 하려했지만 못함
- 데이터셋 추가
    - Aida dataset을 추가하여 하였으나 점수가 별로 안올라서 포기
    - Im2Latex를 추가하려고 하였으나 서버용량이 부족하여 추가하지 못함


<br>



# Reference

### Paper
- [Translating Math Formula Images to LaTeX Sequences Using Deep
Neural Networks with Sequence-level Training](https://arxiv.org/pdf/1908.11415.pdf)
- [Image to Latex](http://cs231n.stanford.edu/reports/2017/pdfs/815.pdf)
- [Pattern Generation Strategies for Improving Recognition of Handwritten
Mathematical Expressions](https://arxiv.org/pdf/1901.06763.pdf)
- [On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w34/Lee_On_Recognizing_Texts_of_Arbitrary_Shapes_With_2D_Self-Attention_CVPRW_2020_paper.pdf)


### GIt
- [LSTM](https://github.com/harvardnlp/im2markup)
- [Aster Pytorch ](https://github.com/ayumiymk/aster.pytorch)
- [SRN paddlepaddle](https://www.paddlepaddle.org.cn/hub/scene/ocr)
- [im2markup](https://github.com/harvardnlp/im2markup/)
- [satrn](https://github.com/clovaai/SATRN)
- [satrnPytorch,CSTR](https://github.com/Media-Smart/vedastr)

### Site & Kaggle
- [Detecting Mathematical Expressions in Scientific
Document Images Using a U-Net Trained
on a Diverse Dataset](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8861044)
- [Painfree LaTeX with Optical Character Recognition and Machine Learning (CS229)](http://cs229.stanford.edu/proj2016/poster/ChangGuptaZhang-PainfreeLatexWithOpticalCharacterRecognitionAndMachineLearning.pdf)
- [Translating Math Formula Images to LaTeX Sequences Using Deep
Neural Networks with Sequence-level Training](https://arxiv.org/pdf/1908.11415.pdf)
- [OCR API for Math and Science](https://mathpix.com/ocr)
- [콴다(QANDA)앱](https://play.google.com/store/apps/details?id=com.mathpresso.qanda&hl=ko&gl=US)
- [Math survey](https://www.cs.rit.edu/~rlaz/files/mathSurvey.pdf)
- [“책을찍다”에 사용되는 Image Segmentation 기술 (전처리 관련)](https://medium.com/team-red/%EC%B1%85%EC%9D%84%EC%B0%8D%EB%8B%A4-%EC%97%90-%EC%82%AC%EC%9A%A9%EB%90%98%EB%8A%94-%EC%98%81%EC%83%81%EB%B6%84%ED%95%A0-image-segmentation-%EA%B8%B0%EC%88%A0-aa5c8f36f8ab)
-  [Translating Math Formula Images to LaTeX Sequences Using Deep Neural Networks with Sequence-level Training](https://paperswithcode.com/paper/translating-mathematical-formula-images-to)
- [Image to Latex](http://cs231n.stanford.edu/reports/2017/pdfs/815.pdf)
* [수식 인식기 논문 ]( https://arxiv.org/pdf/1908.11415.pdf)
* [수식 인식기 코드]( https://paperswithcode.com/paper/translating-mathematical-formula-images-to)
* [Open CV 필터](https://bkshin.tistory.com/entry/OpenCV-18-%EA%B2%BD%EA%B3%84-%EA%B2%80%EC%B6%9C-%EB%AF%B8%EB%B6%84-%ED%95%84%ED%84%B0-%EB%A1%9C%EB%B2%84%EC%B8%A0-%EA%B5%90%EC%B0%A8-%ED%95%84%ED%84%B0-%ED%94%84%EB%A6%AC%EC%9C%97-%ED%95%84%ED%84%B0-%EC%86%8C%EB%B2%A8-%ED%95%84%ED%84%B0-%EC%83%A4%EB%A5%B4-%ED%95%84%ED%84%B0-%EB%9D%BC%ED%94%8C%EB%9D%BC%EC%8B%9C%EC%95%88-%ED%95%84%ED%84%B0-%EC%BA%90%EB%8B%88-%EC%97%A3%EC%A7%80)
- [캐글-im2latex](https://www.kaggle.com/shahrukhkhan/im2latex100k?select=formula_images_processed)
- [SOTA-textrecognition](https://paperswithcode.com/sota/scene-text-recognition-on-icdar2013)





