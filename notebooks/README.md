# Financial Sentiment Analysis Notebooks

This directory contains Jupyter notebooks for processing financial tweets, performing sentiment analysis, and training machine learning models for the financial sentiment analysis project.

## Notebook Execution Order

The notebooks should be executed in the following order:

1. `00b_ner_stock_identification.ipynb` - Processes raw tweets with Named Entity Recognition and identifies verified stock symbols
2. `00_data_labeling_with_resume.ipynb` - Labels tweets using Mistral AI API with resume capability
3. `00c_data_labeling_with_stocks.ipynb` - (Optional) Labels tweets with verified stock symbols 
4. `00_data_labeling.ipynb` - (Optional) Original data labeling without stock focus
5. `01_data_preparation.ipynb` - Prepares data for model training
6. `02a_gemma3_training_lora.ipynb` - Trains the Gemma 3 model with LoRA fine-tuning
7. `02b_finbert_training.ipynb` - Trains the FinBERT model (supports both 3-class and 5-class variants)

## Notebook Descriptions

### 00b_ner_stock_identification.ipynb
- Processes raw tweet CSV files using BERT-based Named Entity Recognition (NER)
- Identifies entity types and their values within tweets
- Extracts potential stock symbols (patterns like $XXX)
- Verifies extracted symbols against real stocks using yfinance
- Outputs processed data with NER results and verified stock symbols
- Creates both a full dataset and a filtered dataset of tweets with verified stock symbols

### 00_data_labeling_with_resume.ipynb
- Primary data labeling notebook with resume capability
- Uses Mistral AI API for 5-class sentiment labeling
- Implements KeyManager for handling multiple API keys with rotation
- Supports resuming from previous runs with progress tracking
- Processes data from Hugging Face's stock_market_tweets dataset
- Uses ThreadPoolExecutor for parallel processing
- Periodically saves progress to allow resuming after interruptions

### 00_data_labeling_with_resume.py
- Python script version of the resume-capable data labeling
- Provides command-line interface for batch processing
- Supports the same features as the notebook version

### 00c_data_labeling_with_stocks.ipynb
- Loads the preprocessed data with verified stock symbols
- Labels the sentiment of each tweet
- Focuses the sentiment analysis on the specific stocks mentioned
- Filters out non-relevant tweets
- Creates datasets for stock-specific sentiment analysis and model training

### 00_data_labeling.ipynb
- Original data labeling notebook without stock symbol focus
- Can be used if a broader sentiment analysis is needed

### 00_data_labeling.py
- Python script version of the data labeling process
- Provides command-line interface for batch processing

### 01_data_preparation.ipynb
- Prepares labeled data for model training
- Performs preprocessing and filtering
- Creates train/validation/test splits
- Handles data format conversion for different models

### 02a_gemma3_training_lora.ipynb
- Implements Gemma 3 model with LoRA fine-tuning
- Uses 8-bit quantization for efficient training
- Configures LoRA parameters (r=8, alpha=16)
- Targets q_proj and v_proj modules
- Implements multi-metric evaluation and early stopping
- Saves the trained adapter for efficient deployment

### 02b_finbert_training.ipynb
- Implements FinBERT model training
- Fine-tunes on financial tweet data
- Includes evaluation metrics
- Saves the trained model for both 3-class and 5-class implementations
- Handles different dataset column names with appropriate mapping

### finbert_results/
- Contains evaluation results from FinBERT models
- Includes confusion matrices, performance metrics
- Stores model comparison data

## Output Files

The notebooks will generate various output files in the `../data/` directory:

- `tweets_with_ner_and_stocks.csv` - All tweets with NER and stock symbol information
- `tweets_with_verified_stocks.csv` - Filtered dataset with only tweets mentioning verified stock symbols
- `labeled_stock_tweets.csv` - Main labeled dataset from the resume-capable approach
- `labeled_stock_tweets_progress.json` - Progress tracking for resume capability
- `stock_tweets_labeled.csv` - Tweets with verified stock symbols and their sentiment labels
- `stock_tweets_for_training.csv` - Final dataset for model training, excluding non-relevant tweets
- `stock_tweets_by_symbol.csv` - Expanded dataset for analysis by individual stock symbol

## Model Artifacts

Trained models and their artifacts are saved in the `../models/` directory:

### Gemma 3 Model (`../models/gemma3/`)
- `gemma3_lora_adapter_best/` - Best performing LoRA adapter weights
- `gemma3_lora_adapter_final/` - Final trained LoRA adapter weights
- `metrics.csv` - Model evaluation metrics
- `training_history.csv` - Training progress history

### FinBERT Model (`../models/finbert/`)
- `model.safetensors` - Model weights in safetensors format
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer configuration
- `vocab.txt` - Vocabulary file
- `metrics.csv` - Model evaluation metrics

## Required Dependencies

To run these notebooks, you'll need the following Python packages:
- pandas
- numpy
- polars
- transformers
- torch
- peft
- accelerate
- bitsandbytes (for 8-bit quantization)
- yfinance
- requests
- tqdm
- huggingface_hub
- tensorboard (for training visualization)

You can install them using:
```
pip install pandas numpy polars transformers torch peft accelerate bitsandbytes yfinance requests tqdm huggingface_hub tensorboard
```

Additionally, you'll need:
- Mistral AI API keys (set as environment variables MISTRAL_API_KEY, MISTRAL_API_KEY_1, etc.)
- Internet access for yfinance stock symbol verification and Hugging Face datasets
- Sufficient disk space for the model weights and output files 

## API Key Setup

For the 00_data_labeling_with_resume.ipynb notebook, you'll need to set up Mistral AI API keys:

1. Set your primary API key as `MISTRAL_API_KEY` environment variable
2. For additional keys, use `MISTRAL_API_KEY_1`, `MISTRAL_API_KEY_2`, etc.
3. The system will automatically rotate between keys if rate limits are encountered 