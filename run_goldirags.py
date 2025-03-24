#!/usr/bin/env python
"""GoldiRAGs 파이프라인 실행 스크립트"""

import os
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가하여 goldirags 모듈을 임포트할 수 있게 함
sys.path.insert(0, str(Path(__file__).resolve().parent))

from goldirags.main import main

if __name__ == "__main__":
    # GoldiRAGs 파이프라인 실행
    main() 