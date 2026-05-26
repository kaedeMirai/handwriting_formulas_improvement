import sys
import re
import torch
from PIL import Image

from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
ADAPTER_PATH = "checkpoints/qwen3vl_2b_latex_lora_masked_20k"

PROMPT = "Convert the formula in the image to LaTeX. Return only LaTeX."


def main(image_path: str):
    image = Image.open(image_path).convert("RGB")

    processor = AutoProcessor.from_pretrained(ADAPTER_PATH)

    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
    )

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    base_model.config.pad_token_id = processor.tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

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
            max_new_tokens=384,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    input_token_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_token_len:]

    raw_prediction = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )[0]

    print("\nRaw prediction: ")
    print(raw_prediction)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/infer.py path/to/image.png")
        sys.exit(1)

    main(sys.argv[1])
