# WellGen AI - Project Structure

## 📁 Organized Workspace Structure

```
wellgen-ai/
├── .env                        # Environment variables (API keys)
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── STRUCTURE.md                # This file
│
├── src/                        # Source code directory
│   ├── __init__.py
│   ├── start.py                # 🚀 MAIN ENTRY POINT - Run this!
│   │
│   ├── text_gen/               # Text generation module
│   │   ├── __init__.py
│   │   ├── wellgen_rag.py      # Core RAG application logic
│   │   └── rag_system.py       # Vector database & retrieval system
│   │
│   ├── image_gen/              # Image generation module (future feature)
│   │   ├── __init__.py
│   │   ├── download_food_dataset.py
│   │   ├── generate_images.py
│   │   ├── setup_image_model.py
│   │   └── train_image_model.py
│   │
│   └── utils/                  # Data processing utilities
│       ├── __init__.py
│       ├── convert_kaggle_to_rag.py  # Convert Kaggle data to RAG format
│       ├── count_data.py             # Data counting utility
│       └── download_chatbot_data.py  # Download chatbot datasets
│
├── knowledge_base/             # Processed nutrition knowledge
│   ├── diet_guidelines.json    # Diet guidelines
│   ├── kaggle_nutrition.json   # Main nutrition knowledge base (750+ docs)
│   └── medical_nutrition.json  # Medical nutrition data
│
├── data/                       # Raw Kaggle datasets (large files)
│   ├── training_data.json      # ~270 MB
│   ├── training_data_filtered.json
│   ├── raw/                    # Raw data files
│   └── zips/                   # Compressed datasets
│
└── model/                      # Model artifacts (local models if needed)
    ├── config.json
    ├── model.safetensors        # ~990 MB
    ├── tokenizer files
    └── ...
```

## 🎯 Key Files

### Core Application Files (In Use)
- **`src/start.py`** - Main entry point for the application
- **`src/text_gen/wellgen_rag.py`** - Core RAG logic with Groq API integration
- **`src/text_gen/rag_system.py`** - FAISS vector database and retrieval
- **`knowledge_base/kaggle_nutrition.json`** - Nutrition knowledge base
- **`requirements.txt`** - Python dependencies
- **`.env`** - API keys (GROQ_API_KEY)

### Utility Files (Data Processing)
- **`src/utils/convert_kaggle_to_rag.py`** - Converts raw Kaggle data to RAG format
- **`src/utils/count_data.py`** - Counts data entries
- **`src/utils/download_chatbot_data.py`** - Downloads chatbot training data

### Image Generation (Future Feature)
- All files in `src/image_gen/` - For future food image generation features

## 🚀 How to Run

```bash
# From project root directory
python src/start.py
```

## 📦 Module Organization

### `src/text_gen/`
Contains all text generation and RAG-related code:
- RAG system implementation
- Groq API integration
- Diet plan generation
- Conversational AI

### `src/image_gen/`
Contains image generation utilities (not currently used in main app):
- Food dataset management
- Image generation models
- Training scripts

### `src/utils/`
Contains data processing and utility scripts:
- Data conversion tools
- Download scripts
- Data analysis utilities

## 🔧 Dependencies

All dependencies are managed in `requirements.txt`:
- `python-dotenv` - Environment variable management
- `groq` - Groq API client
- `sentence-transformers` - Embedding models
- `faiss-cpu` - Vector similarity search
- `torch` - PyTorch (for embeddings)
- `transformers` - Hugging Face transformers

## 📝 Notes

- **Data folder** contains large raw datasets (~500+ MB) - kept for reference
- **Model folder** contains model artifacts (~1 GB) - kept for potential local inference
- **Knowledge base** is the processed, production-ready data used by the app
- All Python packages have `__init__.py` for proper module imports
