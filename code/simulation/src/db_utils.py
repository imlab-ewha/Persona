import os
import streamlit as st
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder
from dotenv import load_dotenv

def get_db_engine():
    # 1. 클라우드 환경인지 확인 (st.secrets 접근 시 발생하는 에러 차단)
    is_cloud = False
    try:
        # secrets.toml 파일이 있고, 그 안에 키가 하나라도 있는지 확인
        if st.secrets and len(st.secrets.keys()) > 0:
            if "SSH_HOST" in st.secrets:
                is_cloud = True
    except Exception:
        # 파일이 없으면 StreamlitSecretNotFoundError가 발생하며 이쪽으로 넘어옵니다.
        is_cloud = False

    if is_cloud:
        # --- [Cloud 모드] SSH 터널링 접속 ---
        st.info("☁️ Streamlit Cloud 모드로 접속 중...")
        tunnel = SSHTunnelForwarder(
            (st.secrets["SSH_HOST"], int(st.secrets["SSH_PORT"])),
            ssh_username=st.secrets["SSH_USER"],
            ssh_password=st.secrets["SSH_PASSWORD"],
            remote_bind_address=(st.secrets["DB_HOST"], int(st.secrets["DB_PORT"]))
        )
        tunnel.start()
        
        db_url = (
            f"postgresql+psycopg2://{st.secrets['DB_USER']}:{st.secrets['DB_PASSWORD']}"
            f"@127.0.0.1:{tunnel.local_bind_port}/{st.secrets['DB_NAME']}"
        )
        return create_engine(db_url), tunnel
    
    else:
        # --- [Local 모드] 서버 터미널에서 .env로 접속 ---
        st.info("🏠 로컬/서버 터미널 모드로 접속 중...")
        
        # 보현님이 알려주신 경로 반영: .../simulation/src/db_utils.py -> .../simulation/.env
        current_dir = os.path.dirname(os.path.abspath(__file__)) # src 폴더
        parent_dir = os.path.dirname(current_dir)               # simulation 폴더
        env_path = os.path.join(parent_dir, ".env")
        
        load_dotenv(env_path)
        
        db_url = (
            f"postgresql+psycopg2://{os.getenv('DB_USER')}@{os.getenv('DB_HOST')}:"
            f"{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        )
        return create_engine(db_url), None