import ast
import os
import json
import time
import pandas as pd
from tqdm import tqdm

from openai import OpenAI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# --- LLM 출력 스키마 정의 (JSON 형태 강제) ---
class SurveyOutput(BaseModel):
    keyword: str
    description: List[str]

# --- 기본 설정 ---
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# OpenAI API 키 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 경로 설정
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(_BASE, "prompts")
PROMPT_FILE = os.path.join(PROMPTS_DIR, "history_promp.txt")

# 입력 CSV 및 출력 JSON 경로
CSV_FILE_PATH = "/home/imlab/Persona/data/processed2/persona_profile_survey.csv"
OUTPUT_JSON_DIR = "/home/imlab/Persona/data/processed2"
OUTPUT_JSON_PATH = os.path.join(OUTPUT_JSON_DIR, "survey.json")


# --- 함수 1: 프롬프트 생성 및 LLM 결과만 반환 ---
def get_llm_mapping(question: str, options: list):
    """
    프롬프트를 읽어 LLM을 호출하고, LLM의 원본 출력 객체(keyword, description 포함)를 그대로 반환합니다.
    """
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # 프롬프트 포매팅 (.replace 사용으로 JSON 중괄호 충돌 방지)
    options_str = json.dumps(options, ensure_ascii=False)
    prompt = prompt_template.replace("{question}", question).replace("{options}", options_str)

    try:
        # system 구조 없이, user 역할 하나에 프롬프트를 통째로 전달
        response = client.beta.chat.completions.parse(
            model="gpt-5.4-nano", 
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format=SurveyOutput
        )
        
        # LLM이 파싱한 형태 그대로 리턴 (SurveyOutput 객체)
        return response.choices[0].message.parsed
        
    except Exception as e:
        print(f"\n[API 오류] '{question}' 처리 중 에러 발생: {e}")
        return None


# --- 함수 2: CSV 순회, 매핑, 실시간 JSON 저장 ---
def process_survey_csv():
    if not os.path.exists(OUTPUT_JSON_DIR):
        os.makedirs(OUTPUT_JSON_DIR)

    print(f"데이터 로드 중: {CSV_FILE_PATH}")
    try:
        # low_memory=False 를 추가하여 DtypeWarning 방지
        df = pd.read_csv(CSV_FILE_PATH, low_memory=False)
    except Exception as e:
        print(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    # 전체 컬럼 가져오기
    columns = df.columns.tolist()
    
    # persona_id 컬럼 제외
    if "persona_id" in columns:
        columns.remove("persona_id")

    print(f"총 {len(columns)}개의 컬럼(질문)을 순회하며 작업을 진행합니다.")

    # 전체 컬럼에 대해 순회 시작
    for col in tqdm(columns, desc="컬럼 처리 중"):
        # 해당 컬럼의 모든 고유 값 추출 (결측치 제외)
        unique_rows = df[col].dropna().unique().tolist()
        
        all_unique_items = set()
        
        # 리스트 평탄화 및 unique 값 추출
        for raw_val in unique_rows:
            val = str(raw_val).strip()
            if not val or val.lower() == 'nan':
                continue
            
            # CSV에서는 리스트도 "[...]" 형태의 문자열로 저장됨
            if val.startswith('[') and val.endswith(']'):
                try:
                    parsed_list = ast.literal_eval(val)
                    if isinstance(parsed_list, list):
                        for item in parsed_list:
                            if str(item).strip():
                                all_unique_items.add(str(item).strip())
                    else:
                        all_unique_items.add(val)
                except (ValueError, SyntaxError):
                    # 따옴표가 없거나 형식이 깨진 배열 문자열 처리
                    clean_val = val[1:-1]
                    items = [x.strip() for x in clean_val.split(',')]
                    for item in items:
                        if item:
                            all_unique_items.add(item)
            else:
                all_unique_items.add(val)

        options = list(all_unique_items)

        if not options:
            print(f"\n[알림] '{col}' 컬럼에 유효한 데이터가 없습니다. 건너뜁니다.")
            continue

        # 1. LLM 아웃풋 객체 받아오기
        llm_result = get_llm_mapping(question=col, options=options)
        
        if not llm_result:
            continue

        # 2. 원본 options와 LLM의 description을 직접 매핑
        descriptions = llm_result.description
        keyword = llm_result.keyword # LLM이 생성한 키워드 추출
        
        if len(options) != len(descriptions):
            print(f"\n[경고] '{col}'의 원본 옵션 개수({len(options)})와 LLM 결과 개수({len(descriptions)})가 다릅니다.")
        
        # 🌟 description 안에 들어갈 매핑 딕셔너리 생성
        description_dict = {}
        for original_val, desc_val in zip(options, descriptions):
            description_dict[original_val] = desc_val

        # 🌟 최종적으로 저장될 딕셔너리 구조 조립
        mapped_dict = {
            "keyword": keyword,
            "description": description_dict
        }

        # 3. 실시간 JSON 적재
        if os.path.exists(OUTPUT_JSON_PATH):
            with open(OUTPUT_JSON_PATH, "r", encoding="utf-8") as f:
                try:
                    survey_data = json.load(f)
                except json.JSONDecodeError:
                    survey_data = {}
        else:
            survey_data = {}

        # {"질문명": {"keyword": "...", "description": {"옵션1": "...", "옵션2": "..."}}} 형태로 저장
        survey_data[col] = mapped_dict

        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(survey_data, f, ensure_ascii=False, indent=4)
        
        # API Rate Limit 보호
        time.sleep(1)

    print(f"\n✅ 전체 작업이 완료되었습니다. 결과 파일: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    process_survey_csv()

# import ast
# import os
# import json
# import time
# import pandas as pd
# from tqdm import tqdm

# from openai import OpenAI
# from pydantic import BaseModel
# from typing import List
# from dotenv import load_dotenv

# # --- LLM 출력 스키마 정의 (JSON 형태 강제) ---
# class SurveyOutput(BaseModel):
#     keyword: str
#     description: List[str]

# # --- 기본 설정 ---
# load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# # OpenAI API 키 설정
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # 경로 설정
# _BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# PROMPTS_DIR = os.path.join(_BASE, "prompts")
# PROMPT_FILE = os.path.join(PROMPTS_DIR, "history_promp.txt")

# # 입력 CSV 및 출력 JSON 경로
# CSV_FILE_PATH = "/home/imlab/Persona/data/processed2/persona_profile_survey.csv"
# OUTPUT_JSON_DIR = "/home/imlab/Persona/data/processed2"
# OUTPUT_JSON_PATH = os.path.join(OUTPUT_JSON_DIR, "survey.json")

# # --- 🎯 테스트를 원하는 특정 컬럼 지정 ---
# TEST_COLUMNS = [
#     "다음은 평소 스마트폰 기기 활용도와 관련한 항목입니다. 잘 읽고 일치하는 보기를 선택해주시기 바랍니다._1)나는 평소 스마트폰 기기의 다양한 기능에 대한 관심이 많다",
#     "귀하께서 현 거주 주택에서 경험한 ‘층간소음’은 일상 생활에 어느 정도 영향을 미치는 편이십니까?"
# ]


# # --- 함수 1: 프롬프트 생성 및 LLM 결과만 반환 ---
# def get_llm_mapping(question: str, options: list):
#     """
#     프롬프트를 읽어 LLM을 호출하고, LLM의 원본 출력 객체(keyword, description 포함)를 그대로 반환합니다.
#     """
#     with open(PROMPT_FILE, "r", encoding="utf-8") as f:
#         prompt_template = f.read()

#     # 프롬프트 포매팅 (.replace 사용으로 JSON 중괄호 충돌 방지)
#     options_str = json.dumps(options, ensure_ascii=False)
#     prompt = prompt_template.replace("{question}", question).replace("{options}", options_str)

#     try:
#         # system 구조 없이, user 역할 하나에 프롬프트를 통째로 전달
#         response = client.beta.chat.completions.parse(
#             model="gpt-5.4-nano", 
#             messages=[
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.0,
#             response_format=SurveyOutput
#         )
        
#         # LLM이 파싱한 형태 그대로 리턴 (SurveyOutput 객체)
#         return response.choices[0].message.parsed
        
#     except Exception as e:
#         print(f"\n[API 오류] '{question}' 처리 중 에러 발생: {e}")
#         return None


# # --- 함수 2: CSV 순회, 매핑, 실시간 JSON 저장 ---
# def process_survey_csv():
#     if not os.path.exists(OUTPUT_JSON_DIR):
#         os.makedirs(OUTPUT_JSON_DIR)

#     print(f"데이터 로드 중: {CSV_FILE_PATH}")
#     try:
#         # low_memory=False 를 추가하여 DtypeWarning 방지
#         df = pd.read_csv(CSV_FILE_PATH, low_memory=False)
#     except Exception as e:
#         print(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
#         return

#     columns = df.columns.tolist()
    
#     # 지정한 테스트 컬럼들만 필터링 (실제 CSV에 존재하는지 확인)
#     columns_to_process = [col for col in TEST_COLUMNS if col in columns]

#     if not columns_to_process:
#         print("⚠️ 테스트하려고 지정한 컬럼이 CSV 파일 내에 존재하지 않습니다. 컬럼명을 다시 확인해주세요.")
#         return

#     print(f"총 {len(columns_to_process)}개의 컬럼(질문)을 순회하며 테스트를 진행합니다.")

#     for col in tqdm(columns_to_process, desc="컬럼 처리 중"):
#         # 해당 컬럼의 모든 고유 값 추출 (결측치 제외)
#         unique_rows = df[col].dropna().unique().tolist()
        
#         all_unique_items = set()
        
#         # 리스트 평탄화 및 unique 값 추출
#         for raw_val in unique_rows:
#             val = str(raw_val).strip()
#             if not val or val.lower() == 'nan':
#                 continue
            
#             # CSV에서는 리스트도 "[...]" 형태의 문자열로 저장됨
#             if val.startswith('[') and val.endswith(']'):
#                 try:
#                     parsed_list = ast.literal_eval(val)
#                     if isinstance(parsed_list, list):
#                         for item in parsed_list:
#                             if str(item).strip():
#                                 all_unique_items.add(str(item).strip())
#                     else:
#                         all_unique_items.add(val)
#                 except (ValueError, SyntaxError):
#                     # 따옴표가 없거나 형식이 깨진 배열 문자열 처리
#                     clean_val = val[1:-1]
#                     items = [x.strip() for x in clean_val.split(',')]
#                     for item in items:
#                         if item:
#                             all_unique_items.add(item)
#             else:
#                 all_unique_items.add(val)

#         options = list(all_unique_items)

#         if not options:
#             print(f"\n[알림] '{col}' 컬럼에 유효한 데이터가 없습니다.")
#             continue

#         # 1. LLM 아웃풋 객체 받아오기
#         llm_result = get_llm_mapping(question=col, options=options)
        
#         if not llm_result:
#             continue

#         # 2. 원본 options와 LLM의 description을 직접 매핑
#         descriptions = llm_result.description
#         keyword = llm_result.keyword # LLM이 생성한 키워드 추출
        
#         if len(options) != len(descriptions):
#             print(f"\n[경고] '{col}'의 원본 옵션 개수({len(options)})와 LLM 결과 개수({len(descriptions)})가 다릅니다.")
        
#         # 🌟 description 안에 들어갈 매핑 딕셔너리 생성
#         description_dict = {}
#         for original_val, desc_val in zip(options, descriptions):
#             description_dict[original_val] = desc_val

#         # 🌟 최종적으로 저장될 딕셔너리 구조 조립
#         mapped_dict = {
#             "keyword": keyword,
#             "description": description_dict
#         }

#         # 3. 실시간 JSON 적재
#         if os.path.exists(OUTPUT_JSON_PATH):
#             with open(OUTPUT_JSON_PATH, "r", encoding="utf-8") as f:
#                 try:
#                     survey_data = json.load(f)
#                 except json.JSONDecodeError:
#                     survey_data = {}
#         else:
#             survey_data = {}

#         # {"질문명": {"keyword": "...", "description": {"옵션1": "...", "옵션2": "..."}}} 형태로 저장
#         survey_data[col] = mapped_dict

#         with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
#             json.dump(survey_data, f, ensure_ascii=False, indent=4)
        
#         # API Rate Limit 보호
#         time.sleep(1)

#     print(f"\n✅ 테스트 작업이 완료되었습니다. 결과 파일: {OUTPUT_JSON_PATH}")


# if __name__ == "__main__":
#     process_survey_csv()