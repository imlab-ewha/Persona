"""
condition.py
------------
질의(Query) 분석을 통해 관련 있는 페르소나 속성(Attribute)을 선정하는 모듈입니다.
실제 DB 접속을 통해 컬럼 존재 여부를 확인하는 로직이 포함되어 있습니다.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sshtunnel import SSHTunnelForwarder
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
# DB 설정
SSH_HOST = os.getenv("SSH_HOST")
SSH_PORT = int(os.getenv("SSH_PORT", 4040))
SSH_USER = os.getenv("SSH_USER")
SSH_PASSWORD = os.getenv("SSH_PASSWORD")

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_USER = os.getenv("DB_USER", "pdp")
DB_PASSWORD = os.getenv("DB_PASSWORD", "1234")
DB_NAME = os.getenv("DB_NAME", "persona")

tunnel = SSHTunnelForwarder(
    (SSH_HOST, SSH_PORT),
    ssh_username=SSH_USER,
    ssh_password=SSH_PASSWORD,
    remote_bind_address=(DB_HOST, DB_PORT)
)
tunnel.start()

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@127.0.0.1:{tunnel.local_bind_port}/{DB_NAME}"
)

def get_relevant_attributes(query: str) -> list:
    """
    입력된 질의와 관련된 속성을 추출하기 전, 실제 DB의 컬럼 구성을 확인합니다.
    최종적으로는 빈 리스트를 반환합니다.
    """
    try:

            # 💡 실제 DB의 컬럼 리스트 확인
            inspector = inspect(engine)
            columns_info = inspector.get_columns('persona_profile_test', schema='public')
            db_columns = [col['name'] for col in columns_info]
            
            print(f"  [DB 확인] persona_profile_test 테이블에서 {len(db_columns)}개의 컬럼을 확인했습니다.")
            # print(f"  [컬럼 목록] {db_columns}") # 필요 시 주석 해제하여 확인 가능

            # TODO: 나중에 여기서 query와 db_columns를 매칭하는 로직이 들어갈 예정입니다.
            
    except Exception as e:
        print(f"  [오류] DB 컬럼 확인 중 에러 발생: {e}")

    # 요청에 따라 빈 리스트 반환
    return []