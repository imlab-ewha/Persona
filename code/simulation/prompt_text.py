import json
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values
from openai import OpenAI
import anthropic


# =========================
# Page config
# =========================
st.set_page_config(page_title="Prompt TXT + JSON Tester", layout="wide")


# =========================
# Constants
# =========================
DEFAULT_ENV_PATH = "/home/imlab/Persona/code/simulation/.env"

OPENAI_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
]

ANTHROPIC_MODELS = [
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-haiku-4-5-20251001",
]


# =========================
# Helpers
# =========================
def load_env_file(env_path_str: str) -> dict:
    env_path = Path(env_path_str).expanduser().resolve()

    if not env_path.exists():
        raise FileNotFoundError(f".env file not found: {env_path}")

    env_vars = dotenv_values(env_path)
    return env_vars


def load_prompt_template(prompt_txt_path: str) -> str:
    path = Path(prompt_txt_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"TXT file not found: {path}")

    if path.suffix.lower() != ".txt":
        raise ValueError("Prompt file must be a .txt file.")

    return path.read_text(encoding="utf-8")


def parse_input_json(raw_text: str) -> dict:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON input:\n{e}")

    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object.")

    if "question" not in data:
        raise ValueError("Input JSON must contain 'question'.")

    if "options" not in data:
        raise ValueError("Input JSON must contain 'options'.")

    if not isinstance(data["question"], str):
        raise ValueError("'question' must be a string.")

    if not isinstance(data["options"], list):
        raise ValueError("'options' must be a list.")

    if not all(isinstance(x, str) for x in data["options"]):
        raise ValueError("Every item in 'options' must be a string.")

    return data


def render_prompt(prompt_template: str, input_data: dict) -> str:
    question = input_data["question"]
    options = input_data["options"]

    options_text = json.dumps(options, ensure_ascii=False, indent=2)

    rendered = prompt_template.replace("{question}", question)
    rendered = rendered.replace("{options}", options_text)

    return rendered


def call_openai(
    api_key: str,
    model_name: str,
    rendered_prompt: str,
    reasoning_effort: str,
    max_output_tokens: int,
) -> str:
    client = OpenAI(api_key=api_key)

    # GPT-5 계열은 Responses API 사용
    response = client.responses.create(
        model=model_name,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
        input=[
            {
                "role": "system",
                "content": "Follow the instructions exactly. Return only the requested output.",
            },
            {
                "role": "user",
                "content": rendered_prompt,
            },
        ],
    )

    return (response.output_text or "").strip()


def extract_anthropic_text(response) -> str:
    parts = []
    for block in response.content:
        # 일반 텍스트 블록만 합치기
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def call_anthropic(
    api_key: str,
    model_name: str,
    rendered_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        system="Follow the instructions exactly. Return only the requested output.",
        messages=[
            {
                "role": "user",
                "content": rendered_prompt,
            }
        ],
    )

    return extract_anthropic_text(response)


# =========================
# UI
# =========================
st.title("Prompt TXT + JSON Tester")
st.caption("TXT 경로 + JSON input을 넣고, OpenAI 또는 Claude 모델로 바로 테스트할 수 있어.")

left, right = st.columns([1, 1])

with left:
    provider = st.selectbox(
        "Provider",
        options=["OpenAI", "Anthropic"],
        index=0,
    )

    if provider == "OpenAI":
        model_name = st.selectbox("Model", OPENAI_MODELS, index=0)
        reasoning_effort = st.selectbox(
            "Reasoning effort (OpenAI only)",
            options=["minimal", "low", "medium", "high"],
            index=1,
        )
    else:
        model_name = st.selectbox("Model", ANTHROPIC_MODELS, index=0)
        reasoning_effort = None

    env_path = st.text_input(
        ".env path",
        value=DEFAULT_ENV_PATH,
        placeholder="/home/imlab/Persona/code/simulation/.env",
    )

    prompt_txt_path = st.text_input(
        "Prompt TXT path",
        value="",
        placeholder="/path/to/prompt_template.txt",
    )

    input_json_text = st.text_area(
        "Input JSON",
        value=json.dumps(
            {
                "question": "당신은 임영웅에 대해 어떻게 생각하십니까?",
                "options": [
                    "좋다고 생각함",
                    "별로라고 생각함",
                    "잘 모르겠음"
                ]
            },
            ensure_ascii=False,
            indent=2
        ),
        height=260,
    )

    if provider == "OpenAI":
        max_output_tokens = st.number_input(
            "Max output tokens",
            min_value=256,
            max_value=16384,
            value=2048,
            step=256,
        )
        anthropic_temperature = None
    else:
        max_output_tokens = st.number_input(
            "Max tokens",
            min_value=256,
            max_value=8192,
            value=2048,
            step=256,
        )
        anthropic_temperature = st.slider(
            "Temperature (Anthropic only)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
        )

    run_button = st.button("Run", type="primary")

with right:
    st.markdown("### Required prompt placeholders")
    st.code(
        "question:\n{question}\n\noptions:\n{options}",
        language="text"
    )

    st.markdown("### Example input JSON")
    st.code(
        json.dumps(
            {
                "question": "당신은 임영웅에 대해 어떻게 생각하십니까?",
                "options": [
                    "좋다고 생각함",
                    "별로라고 생각함",
                    "잘 모르겠음"
                ]
            },
            ensure_ascii=False,
            indent=2
        ),
        language="json"
    )

    st.markdown("### Expected .env keys")
    st.code(
        "OPENAI_API_KEY=your_openai_key\n"
        "ANTHROPIC_API_KEY=your_anthropic_key\n",
        language="bash"
    )

# =========================
# Run
# =========================
if run_button:
    try:
        if not prompt_txt_path.strip():
            st.error("Prompt TXT path를 입력해줘.")
            st.stop()

        # 입력 파싱
        input_data = parse_input_json(input_json_text)

        # 프롬프트 로드
        prompt_template = load_prompt_template(prompt_txt_path)

        # env 로드
        env_vars = load_env_file(env_path)

        # 프롬프트 렌더링
        rendered_prompt = render_prompt(prompt_template, input_data)

        st.success("Prompt TXT + JSON + .env 로드 완료")

        with st.expander("Rendered Prompt Preview", expanded=False):
            st.text(rendered_prompt)

        # 모델 호출
        if provider == "OpenAI":
            api_key = env_vars.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in the .env file.")

            output_text = call_openai(
                api_key=api_key,
                model_name=model_name,
                rendered_prompt=rendered_prompt,
                reasoning_effort=reasoning_effort,
                max_output_tokens=int(max_output_tokens),
            )

        else:
            api_key = env_vars.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in the .env file.")

            output_text = call_anthropic(
                api_key=api_key,
                model_name=model_name,
                rendered_prompt=rendered_prompt,
                max_tokens=int(max_output_tokens),
                temperature=float(anthropic_temperature),
            )

        st.markdown("## Output (raw text)")
        st.text_area("LLM Output", value=output_text, height=300)

        # JSON이면 보기 좋게 표시
        try:
            parsed = json.loads(output_text)
            st.markdown("## Output (parsed JSON)")
            st.json(parsed)
        except Exception:
            st.caption("출력이 valid JSON이 아니라서 raw text만 표시했어.")

    except Exception as e:
        st.error(str(e))