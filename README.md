### 부스트캠프 AI Tech - Team Project 수식 인식기

|Task|Date|Team|
|---|---|---|
|수식 이미지를 latex 포맷의 텍스트로 변환하는 문제|2021.05.24 ~ 2021.06.15|5조 ConnectNet|

> `P stage 4 대회 진행 과정과 결과를 기록한 Team Git repo 입니다. 대회 특성상 수정 혹은 삭제된 부분이 존재 합니다`

---



### 📋 Table of content

[Team 소개](#Team)<br>
[Gound Rule](#rule)<br>
[실험노트](https://docs.google.com/spreadsheets/d/1v_ZMKii5nt6VgrtCVA-bue42jWa5wpcJWnHMoFL9OUE)
[설치 및 실행](#Install) <br>

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

* 설치 
```shell
pip install -r requirements.txt
```

* 소스  다운로드 
```shell
git clone https://github.com/bcaitech1/p4-fr-connectnet.git
```

* 학습
```shell
python ./train.py --c config/SATRN.yaml
```

* 평가
```shell
python ./inference.py --checkpoint ./log/satrn/checkpoints/0050.pth
```

# Reference

### Paper
- [Translating Math Formula Images to LaTeX Sequences Using Deep
Neural Networks with Sequence-level Training](https://arxiv.org/pdf/1908.11415.pdf)
- [Image to Latex](http://cs231n.stanford.edu/reports/2017/pdfs/815.pdf)
- [Pattern Generation Strategies for Improving Recognition of Handwritten
Mathematical Expressions](https://arxiv.org/pdf/1901.06763.pdf)
- [On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w34/Lee_On_Recognizing_Texts_of_Arbitrary_Shapes_With_2D_Self-Attention_CVPRW_2020_paper.pdf)


### GIt


### Site & Kaggle






