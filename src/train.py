import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

MODEL_VL = "Qwen/Qwen3-VL-2B-Instruct"
DATASET_OCR = "linxy/LaTeX_OCR"
OUTPUT_DIR = "checkpoints/qwen3vl_2b_latex_lora_masked_20k"
TRAIN_SAMPLES = 20000
PROMPT = "Convert the formula in the image to LaTeX. Return only LaTeX."


class DataCollator:
    """
    DataCollator готовит batch для masked SFT.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        images = []
        full_texts = []
        prompt_texts = []

        for example in examples:
            image = example["image"].convert("RGB")
            latex = example["text"]
            prompt_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ]
            full_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": PROMPT},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": latex},
                    ],
                },
            ]

            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            full_text = self.processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            images.append(image)
            prompt_texts.append(prompt_text)
            full_texts.append(full_text)

        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()

        for i, prompt_text in enumerate(prompt_texts):
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=[images[i]],
                padding=False,
                return_tensors="pt",
            )

            prompt_len = prompt_inputs["input_ids"].shape[1]

            labels[i, :prompt_len] = -100

        pad_token_id = self.processor.tokenizer.pad_token_id

        if pad_token_id is not None:
            labels[batch["input_ids"] == pad_token_id] = -100

        batch["labels"] = labels

        return batch


def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_OCR)

    train_dataset = dataset["train"]

    train_dataset = train_dataset.shuffle(seed=42).select(range(TRAIN_SAMPLES))

    print(f"Train samples: {len(train_dataset)}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_VL,
        trust_remote_code=True,
    )

    processor.tokenizer.padding_side = "right"

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_VL,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    data_collator = DataCollator(processor)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        warmup_ratio=0.03,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving adapter and processor...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print(f"Training finished. Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
