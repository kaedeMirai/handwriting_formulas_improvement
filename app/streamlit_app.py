import re
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
import streamlit as st
from PIL import Image

from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

BASE_MODEL_VL = "Qwen/Qwen3-VL-2B-Instruct"
ADAPTER_PATH = "checkpoints/qwen3vl_2b_latex_lora_ocr_math"
PROMPT = "Convert the formula in the image to LaTeX. Return only LaTeX."


@st.cache_resource
def load_model(adapter_path: str):
    """
    Загружает processor, base model и LoRa adapter.
    """

    processor = AutoProcessor.from_pretrained(
        adapter_path,
        trust_remote_code=True,
    )

    # processor.tokenizer.padding_side = "right"

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_VL,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    base_model.config.pad_token_id = processor.tokenizer.pad_token_id

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
    )

    model.eval()

    return processor, model


def predict_latex(
    image: Image.Image, processor, model, max_new_tokens: int = 384
) -> str:
    """
    Делает inference: image -> LaTeX.
    """

    image = image.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    input_token_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_token_len:]

    prediction = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )[0]

    return prediction


def main():
    st.set_page_config(
        page_title="handwriter",
        layout="wide",
    )

    st.title("Рукописная формула в LaTeX")

    with st.sidebar:
        st.header("Настройки модели")

        adapter_path = st.text_input(
            "Путь к LoRA-адаптеру",
            value=ADAPTER_PATH,
        )

        max_new_tokens = st.slider(
            "Максимум новых токенов",
            min_value=64,
            max_value=512,
            value=384,
            step=32,
        )

    uploaded_file = st.file_uploader(
        "Загрузите изображение формулы",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if uploaded_file is None:
        st.info("Загрузите изображение, чтобы запустить inference.")
        return

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Входное изображение")
        st.image(image, use_container_width=True)

    with st.spinner("Загружаю модель и генерирую LaTeX..."):
        processor, model = load_model(adapter_path)
        predicted_latex = predict_latex(
            image=image,
            processor=processor,
            model=model,
            max_new_tokens=max_new_tokens,
        )

    with col2:
        st.subheader("Предсказанный LaTeX")
        st.code(predicted_latex, language="latex")

        st.subheader("Отрисованный результат")

        try:
            st.latex(predicted_latex)
        except Exception as error:
            st.error("Не удалось отрисовать LaTeX.")
            st.exception(error)


if __name__ == "__main__":
    main()
