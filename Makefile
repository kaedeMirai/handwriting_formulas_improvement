streamlit:
	uv run streamlit run app/streamlit_app.py

eval:
	uv run python src/evaluate.py

train:
	uv run python src/train.py

infer:
	uv run python src/infer.py test_data/image.png