# Handwriting [✎] Formulas Improvement

Тестовый проект для задачи:

```text
изображение рукописной формулы -> LaTeX
```

В проекте есть пайплайн для обучения, оценки и инференса, а также Streamlit-приложение для демонстрации результата.

Лучшая модель: `Qwen/Qwen3-VL-2B-Instruct`, дообученная через LoRA masked SFT на 20,000 примерах из `linxy/LaTeX_OCR`.

**Лучший результат на test subset `linxy/LaTeX_OCR`, 70 примеров:**

| Модель | Train samples | LoRA rank | CER ↓ | Exact Match ↑ |
| --- | ---: | ---: | ---: | ---: |
| `Qwen3-VL-2B-Instruct` + LoRA masked SFT | 20,000 | 32 | **0.0602** | **0.4571** |

## Установка через uv

Установить `uv` (если нет)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Создать окружение и установить зависимости из `pyproject.toml`:

```bash
uv sync
```

## Установка через pip на linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Если нужен CUDA-вариант PyTorch, установите сборку под вашу версию CUDA перед установкой остальных зависимостей или вместе с ними.

## Обучение

```bash
make train
# uv run python src/train.py
```

Параметры финального запуска:

| Параметр | Значение |
| --- | --- |
| Base model | `Qwen/Qwen3-VL-2B-Instruct` |
| Dataset | `linxy/LaTeX_OCR` |
| Train samples | 20000 |
| Fine-tuning | LoRA masked SFT |
| LoRA dropout | 0.05 |
| Learning rate | `5e-5` |
| Epochs | 1 |
| Precision | fp16 |
| Final train loss | 0.06596 |
| Checkpoint | `checkpoints/qwen3vl_2b_latex_lora_masked_20k` |

В обучении промт, указатель изображения и пад токен имеют значение -100

## Evaluation

Запуск evaluation:

```bash
make eval
# uv run python src/evaluate.py
```

Evaluation проводится на `linxy/LaTeX_OCR` test subset из 70 примеров.

Метрики: CER и Exact Match после нормализации LaTeX.

| Метрика | Смысл |
| --- | --- |
| CER | CER между нормализованным prediction и target LaTeX |
| Exact Match | полное совпадение после лёгкой нормализации LaTeX |

Результат сохранятеся в json и указывается как ouput_json:

```text
eval_results_qwen3vl_20k.json
```

## Важная правка decoding

Сначала обучал модель `HuggingFaceTB/SmolVLM-256M-Instruct`, далее перешёл к обучению `Qwen/Qwen3-VL-2B-Instruct` - 100/5000/20000. Модель показывала постепенное улучшение. Разница между 5000 и 20000 минимальна. Тогда я скорректировал параметры для инференса, что в разы улучшило качество, из за чего я снова прогнал полученные модели.
Изначальные параметры генерации:

```text
max_new_tokens = 64 / 128
repetition_penalty = 1.15 / 1.2
no_repeat_ngram_size = 3
```

## Результаты

| Setup | Model | Train samples | Decoding | CER | Exact Match |
| --- | --- | ---: | --- | ---: | ---: |
| Baseline SFT | `SmolVLM-256M-Instruct` | 5,000 | initial | 0.4244 | 0.0143 |
| SFT | `Qwen3-VL-2B-Instruct` | 100 | initial | 0.3529 | 0.0000 |
| SFT | `Qwen3-VL-2B-Instruct` | 5,000 | initial | 0.2654 | 0.0143 |
| SFT | `Qwen3-VL-2B-Instruct` | 20,000 | initial | 0.2643 | 0.0000 |
| SFT | `Qwen3-VL-2B-Instruct` | 100 | corrected | 0.1058 | 0.3143 |
| SFT | `Qwen3-VL-2B-Instruct` | 5,000 | corrected | 0.0644 | 0.4286 |
| SFT | `Qwen3-VL-2B-Instruct` | 20,000 | corrected | **0.0602** | **0.4571** |

Переход от 100 к 5,000 train examples дал большой прирост качества. Переход от 5,000 к 20,000 examples и увеличение LoRA rank с 16 до 32 дали меньший дополнительный выигрыш.
(Zero-shot, one-shot и SFT на `linxy/LaTeX_OCR + deepcopy/MathWriting-human` не запускались.)

## Inference

CLI inference для одного изображения:

```bash
make train
# uv run python src/infer.py test_data/sample_formula.png
```

Скрипт загружает:

```text
Base model: Qwen/Qwen3-VL-2B-Instruct
LoRA adapter: checkpoints/qwen3vl_2b_latex_lora_masked_20k
```

## Streamlit-приложение

Запуск:

```bash
make streamlit
# uv run streamlit run app/streamlit_app.py
```
Можно указать в интерфейсе следующее:

- подать на вход `png`, `jpg`, `jpeg`, `webp`;
- выбирать путь к LoRA adapter;
- отрисовывать формулу через `st.latex`.

Пример интерфейса:
![alt text](additional_readme/image-1.png)
![alt text](additional_readme/image-2.png)
