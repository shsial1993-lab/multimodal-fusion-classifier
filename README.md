# Multimodal Fusion Classifier

A compact late-fusion model for combining image embeddings, text embeddings, and
tabular features in one prediction head. The design mirrors production multimodal
pipelines while keeping each encoder replaceable by a pretrained backbone.

## What it demonstrates

- Independent modality encoders
- Feature normalization before fusion
- Missing-modality masking for resilient inference
- A reproducible synthetic forward-pass demo

## Run

~~~bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m src.demo
~~~

The synthetic inputs are placeholders. Add modality-specific validation and a
leakage-safe split before training on real data.

## Structure

- src/model.py — late-fusion classifier
- src/demo.py — three-modality smoke test

## License

Apache-2.0
