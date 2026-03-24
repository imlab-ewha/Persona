# db_utils.py
import os
import streamlit as st
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder

def get_db_engine():
    # 1. 스트림릿 클라우드(Secrets 설정이 있을 때)
    if "SSH_HOST" in st.secrets:
        tunnel = SSHTunnelForwarder(
            (st.secrets["SSH_HOST"], int(st.secrets["SSH_PORT"])),
            ssh_username=st.secrets["SSH_USER"],
            ssh_password=st.secrets["SSH_PASSWORD"],
            remote_bind_address=(st.secrets["DB_HOST"], int(st.secrets["DB_PORT"]))
        )
        tunnel.start()
        
        db_url = f"postgresql+psycopg2://{st.secrets['DB_USER']}:{st.secrets['DB_PASSWORD']}@127.0.0.1:{tunnel.local_bind_port}/{st.secrets['DB_NAME']}"
        return create_engine(db_url), tunnel
    
    # 2. 로컬 서버 환경 (기존 .env 사용)
    else:
        from dotenv import load_dotenv
        load_dotenv()
        db_url = f"postgresql+psycopg2://{os.getenv('DB_USER')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        return create_engine(db_url), None