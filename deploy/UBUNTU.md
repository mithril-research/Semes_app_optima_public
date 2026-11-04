# OPTIMA Ubuntu 서버 실행

Ubuntu 22.04/24.04, Python 3.10–3.12를 기준으로 합니다. GPU 없이 CPU로 검사와
학습을 실행할 수 있습니다. 아래 명령은 저장소 최상위 폴더에서 실행합니다.

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libgomp1 fonts-nanum
bash src/app/run_ubuntu.sh setup
bash src/app/run_ubuntu.sh
```

다른 PC의 브라우저에서 `http://서버IP:8000`으로 접속합니다. 서버에 GUI나
Windows DLL은 필요하지 않습니다. 브라우저를 닫아도 서버는 계속 실행됩니다.
종료는 Ctrl+C입니다. 기본 바인딩은 `0.0.0.0`이며 포트를 바꿀 수 있습니다.

```bash
PORT=8010 bash src/app/run_ubuntu.sh
# 로컬 접속만 허용할 때
IO_HOST=127.0.0.1 bash src/app/run_ubuntu.sh
```

현재 앱에는 사용자 인증이 없으므로 사내망/VPN에서 사용하고, 외부 공개 시에는
인증과 HTTPS를 제공하는 리버스 프록시 뒤에 배치합니다. 방화벽은 필요한 접속망에만
해당 포트를 허용합니다. 여러 Uvicorn worker를 실행하지 마세요. 검사 작업과 감시 상태가
프로세스 메모리에 있으므로 서버 한 개로 실행합니다.

## 데이터와 모델

기본 설정·데이터·로그·학습 결과는 `src/` 아래에 저장됩니다. 배포 코드와 분리하려면:

```bash
IO_STATE_DIR=/srv/optima-data bash src/app/run_ubuntu.sh
```

실행 사용자에게 해당 폴더의 쓰기 권한이 필요합니다. 기존 설정을 새 위치로 자동 이전하지는
않습니다. 기존 모델을 설정 탭에서 등록하거나 `IO_ARTIFACTS=/절대경로/모델폴더`로 지정합니다.
모델 파일은 저장소에 포함되지 않으므로 검사 전에 학습된 모델을 별도로 배치해야 합니다.

설정 화면의 챔버 경로는 **서버의 Linux 경로**입니다. 기본값은 입력 폴더 아래
`DSV1`~`DSV6`이며 `D:\...` 같은 Windows 경로는 서버에 마운트한 경로로 바꿉니다.
브라우저의 폴더 탐색 기능 또는 직접 입력을 사용합니다. 클라이언트 PC의 파일은
서버에서 바로 읽을 수 없으므로 서버로 복사하거나 공유 폴더를 마운트합니다.

## 재부팅 후 자동 실행(systemd)

저장소를 `/opt/Semes_app_optima_public`에 놓고 위 setup을 먼저 실행합니다.
다른 경로를 사용한다면 `deploy/optima.service`의 WorkingDirectory와 ExecStart를 수정합니다.

```bash
sudo useradd --system --home-dir /var/lib/optima --shell /usr/sbin/nologin optima
sudo cp deploy/optima.service /etc/systemd/system/optima.service
sudo systemctl daemon-reload
sudo systemctl enable --now optima
sudo systemctl status optima
sudo journalctl -u optima -f
```

`optima` 계정은 코드·가상환경·모델에 읽기/실행 권한, 입력 데이터에 필요한 읽기/쓰기
권한이 있어야 합니다. `/var/lib/optima`는 systemd가 생성하며 설정·작업 결과가 유지됩니다.
추가 환경변수는 `/etc/optima.env`에 `PORT=8010` 같은 형식으로 작성하고 서비스를 재시작합니다.

## 선택: NVIDIA GPU

드라이버/CUDA 환경이 준비된 서버에서 `bash src/app/run_ubuntu.sh setup gpu`로 설치합니다.
이 경로는 CUDA 12.8 PyTorch와 GPU ONNX Runtime을 사용합니다. ONNX CUDA 실행에는
호환 CUDA/cuDNN 라이브러리도 필요하며 GPU 환경은 별도 검증해야 합니다.
CPU로 되돌릴 때는 `setup cpu`를 실행합니다. 두 ONNX Runtime 패키지는 동시에 설치하지 않습니다.

설치 Python을 지정하려면 `PYTHON=python3.12`, 가상환경을 따로 두려면
`OPTIMA_VENV=/절대경로/venv`를 사용합니다. 후자는 설치와 실행 모두 동일하게 지정합니다.

## 인터넷·pip가 없는 서버: 오프라인 설치

인터넷이 되는 Linux PC에서 대상 서버와 같은 Python 부버전(3.10/3.11/3.12),
CPU 아키텍처 및 Ubuntu 버전으로 준비합니다. Windows에서 설치한 패키지나 가상환경을
그대로 복사하면 Linux에서 사용할 수 없습니다. 준비 PC에는 pip가 필요하지만 서버에는
pip가 없어도 됩니다. 서버에는 해당 버전의 Python 표준 라이브러리(venv 포함)와
libgomp1 같은 시스템 공유 라이브러리가 있어야 합니다. 이 번들은 Python 자체나 OS
패키지를 포함하지 않습니다. 최소 OS 설치라면 시스템 패키지는 별도로 반입해야 합니다.

인터넷 PC에서:

```bash
python3 src/app/ubuntu_offline.py prepare /tmp/optima-packages
```

CPU PyTorch, ONNX Runtime, 웹 서버, 학습 패키지와 pip 자체를 내려받으며,
버전 고정 목록과 SHA-256 검증 목록을 생성합니다. 출력 폴더는 비어 있어야 합니다.
앱 소스와 `optima-packages` 폴더 전체를 USB/SCP 등으로 서버에 복사합니다.

서버의 앱 최상위 폴더에서:

```bash
bash src/app/run_ubuntu.sh setup-offline /반입경로/optima-packages
bash src/app/run_ubuntu.sh
```

pip가 없는 새 가상환경을 만들고, 반입한 pip wheel을 이용해 로컬 파일만 설치합니다.
온라인 인덱스와 패키지 빌드는 사용하지 않습니다. 설치 후 의존성과 CPU 연산을 검사합니다.
기존 가상환경은 덮어쓰지 않으므로 이미 venv가 있다면 새 경로를 지정하세요.

```bash
OPTIMA_VENV=/srv/optima-venv bash src/app/run_ubuntu.sh setup-offline /반입경로/optima-packages
OPTIMA_VENV=/srv/optima-venv bash src/app/run_ubuntu.sh
```

Python 부버전이나 아키텍처가 다르거나 파일이 누락·손상되면 설치 전에 중단합니다.
준비된 패키지의 호환 대상을 `manifest.json`에서 확인할 수 있습니다.
