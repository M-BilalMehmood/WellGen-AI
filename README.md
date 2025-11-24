# WellGen AI - Personalized Wellness Coach

A Gen AI course project that builds custom models for personalized diet planning and body visualization.

## Project Structure
```
wellgen-ai/
├── data/               # Training datasets
├── models/             # Saved model checkpoints
├── notebooks/          # Jupyter notebooks for training
├── src/               # Source code
│   ├── train_text.py      # Text model fine-tuning
│   ├── train_image.py     # Image model fine-tuning
│   ├── inference.py       # Model inference pipeline
│   └── app.py            # Streamlit frontend
├── requirements.txt
└── README.md
```

## Models
1. **Text Generation**: Fine-tuned Llama/Mistral for diet plan generation
2. **Image Generation**: Stable Diffusion/ControlNet for body part visualizations

## Setup
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Train text model (run in Colab for GPU)
python src/train_text.py

# Train image model
python src/train_image.py

# Run inference app
streamlit run src/app.py
```

## Training Data Sources
- Synthetic diet plans generated using prompt engineering
- Public nutrition databases (USDA, MyFitnessPal)
- Medical body part images (open medical datasets)
