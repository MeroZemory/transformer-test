# Transformer 프로젝트

이 프로젝트는 PyTorch를 사용하여 구현한 Transformer 모델의 예제입니다. 이 모델은 자연어 처리(NLP) 및 기타 시퀀스-투-시퀀스 작업에 활용할 수 있는 인코더-디코더 구조를 포함하고 있습니다.

## 주요 기능
- **Positional Encoding**: 시퀀스의 위치 정보를 반영하는 인코딩 제공
- **Multi-Head Attention**: 병렬 어텐션 헤드를 통한 어텐션 메커니즘
- **Feed Forward Network**: 각 레이어의 정보를 처리하는 피드포워드 네트워크
- **Transformer Encoder & Decoder**: 인코더와 디코더 레이어로 구성된 Transformer 모델

## 설치 및 사용법
1. **가상환경 생성 및 활성화**  
   ```bash
   python -m venv .venv
   # 활성화 방법:
   # PowerShell: .\.venv\Scripts\Activate.ps1
   # CMD: .venv\Scripts\activate.bat
   # Git Bash: source .venv/Scripts/activate
   ```

2. **필수 패키지 설치**  
   ```bash
   pip install -r requirements.txt
   ```

3. **모델 실행**  
   ```bash
   python transformer.py
   ```  
   실행 시, 예제 출력(예: `torch.Size([2, 10, 1000])`)을 확인할 수 있습니다.

## Lint 및 코드 스타일
- `flake8`를 사용하여 코드 스타일 검사를 수행합니다.  
  ```bash
  flake8
  ```

## 추가 작업 및 확장
- 데이터를 사용하여 Transformer 모델을 학습 및 튜닝할 수 있습니다.
- 코드 개선, 새로운 기능 추가, 또는 다른 NLP 태스크에 맞게 모델을 확장할 수 있습니다.

## 문의
프로젝트 관련 문의는 저장소의 Issue 탭을 통해 남겨주세요. 