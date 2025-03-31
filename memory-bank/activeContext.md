# Active Development Context

## Project Overview
PARIS: Personalized AI-Advisor for Robo Investment Strategies - A comprehensive financial sentiment analysis system using Google's Gemma 3, Gamma 3 and FinBERT models with a modern web interface.

## Key Features
- 🤖 Multi-model sentiment analysis (Gemma 3, Gamma 3, and FinBERT)
- 📊 Interactive performance visualization dashboard
- 🔄 Real-time sentiment prediction
- 🎯 5-class sentiment classification (STRONGLY_POSITIVE, POSITIVE, NEUTRAL, NEGATIVE, STRONGLY_NEGATIVE)
- 📈 Comprehensive model evaluation metrics
- 🚀 Modern web interface with TypeScript and Tailwind CSS
- 🐳 Containerized development environment
- ⚡ Resume-capable data labeling for large datasets

## Current Implementation Status

### 1. Machine Learning Pipeline
- ✅ NER processing with bert-base-ner to identify entity types
- ✅ Stock symbol detection and verification using yfinance
- ✅ Data labeling with Mistral AI API (5-class classification) with Hugging Face's stock_market_tweets dataset
- ✅ Resume-capable data labeling with 00_data_labeling_with_resume.ipynb
- ✅ Gamma 3 model with LoRA fine-tuning and early stopping
- ✅ FinBERT model implementation for 3-class sentiment
- ✅ FinBERT model implementation for 5-class sentiment
- ✅ Multi-metric evaluation system
- ✅ Gemma 3 model implementation with LoRA fine-tuning

### 2. Web Application
- ✅ FastAPI backend with model serving
- ✅ Next.js frontend with TypeScript
- ✅ Real-time visualization components
- ✅ Docker development environment

### 3. Development Environment
- ✅ Docker Compose setup with Makefile commands
- ✅ Comprehensive environment variables in .env
- ✅ Development tools container for enhanced development
- ✅ Hot reloading for both frontend and backend

### 4. Sentiment Classes
1. STRONGLY_POSITIVE
2. POSITIVE
3. NEUTRAL
4. NEGATIVE
5. STRONGLY_NEGATIVE

### 5. Model Training Features
- LoRA configuration (r=8, alpha=16)
- Multi-metric early stopping
- Comprehensive evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Cohen's Kappa
  - Matthews Correlation Coefficient
  - ROC-AUC

### 6. Data Processing Enhancement
- Entity recognition using BERT-based NER model
- Stock symbol extraction and verification
- Focus on stock-specific sentiment analysis
- Direct access to Hugging Face's stock_market_tweets dataset with Polars
- Resume-capable data labeling with Mistral AI API 

### 7. Development Workflow
- Docker-centric development using `make` commands
- Container-based services with proper resource allocation
- Jupyter notebook integration via `make jupyter`
- Testing via `make test-backend` and `make test-frontend`

## Project Structure
```
.
├── data/                  # Data storage
│   ├── models/           # Trained model files
│   └── tweets/           # Raw and processed tweets
├── example/              # Example projects
│   └── CS6520-KOLs-Opinions-Sentiment-Classification-main/
├── logs/                 # Log files
│   └── gemma3_training/  # Gemma 3 training logs
├── memory-bank/          # Project memory for Cline
├── models/               # Trained model artifacts
│   ├── gemma3/          # Gemma 3 model files
│   │   ├── gemma3_lora_adapter_best/    # Best performing LoRA adapter
│   │   ├── gemma3_lora_adapter_final/   # Final trained LoRA adapter
│   │   ├── metrics.csv                  # Model evaluation metrics
│   │   └── training_history.csv         # Training progress history
│   └── finbert/         # FinBERT model files
│       ├── model.safetensors           # Model weights
│       ├── config.json                 # Model configuration
│       ├── tokenizer.json              # Tokenizer configuration
│       ├── vocab.txt                   # Vocabulary file
│       └── metrics.csv                 # Model evaluation metrics
├── notebooks/            # Jupyter notebooks
│   ├── 00_data_labeling.ipynb                # Full data labeling
│   ├── 00_data_labeling.py                   # Python script version of data labeling
│   ├── 00_data_labeling_with_resume.ipynb    # Resume-capable data labeling with Mistral AI
│   ├── 00_data_labeling_with_resume.py       # Python script for resume-capable data labeling
│   ├── 00b_ner_stock_identification.ipynb    # NER and stock symbol detection
│   ├── 00c_data_labeling_with_stocks.ipynb   # Stock-specific sentiment labeling
│   ├── 01_data_preparation.ipynb             # Data preparation for model training
│   ├── 02a_gemma3_training_lora.ipynb        # Gemma 3 training with LoRA
│   ├── 02b_finbert_training.ipynb            # FinBERT training
│   └── finbert_results/                      # FinBERT model results
├── scraper/              # Twitter scraping tools
│   ├── twitter_api.py               # Twitter API integration
│   ├── test_crawler.py              # Crawler testing
│   ├── TWITTER_API_README.md        # Documentation
│   ├── crawler_config/              # Configuration files
│   ├── validators/                  # Validation tools
│   └── data/                        # Scraped data
├── src/                  # Core Python modules
│   ├── data_processor.py           # Data processing utilities
│   └── evaluation/                 # Evaluation tools
├── tmp/                  # Temporary files
├── web/                  # Web application
│   ├── backend/         # FastAPI backend
│   ├── frontend/        # Next.js frontend
│   ├── docker-compose.yml         # Container configuration for web
│   └── README.md                  # Web-specific documentation
├── .clinerules           # Cline rules file
├── .env                  # Environment variables
├── .env.example          # Example environment variables
├── .gitignore            # Git ignore rules
├── .gitmessage           # Git commit message template
├── Dockerfile            # Main Dockerfile
├── LICENSE               # Project license
├── Makefile              # Development commands
├── README.md             # Main project documentation
└── requirements.txt      # Python dependencies
```

## Available Make Commands
```bash
make help      # Show all available commands
make up        # Start all services
make down      # Stop all services
make build     # Build/rebuild services
make dev       # Start with development tools
make clean     # Remove all containers and volumes
make logs      # View service logs
make jupyter   # Start Jupyter notebook server
make test-backend  # Run backend tests
make test-frontend # Run frontend tests
```

## Development Environment
The following services are available:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Jupyter Notebooks: http://localhost:8888

## Current Working Branches
- main: Primary development branch
- feature/model-training: Model training implementations
- feature/web-interface: Web application development

## Notebook Execution Order
The notebooks should be executed in the following order:
1. `00b_ner_stock_identification.ipynb` - Processes raw tweets with NER and identifies stock symbols
2. `00_data_labeling_with_resume.ipynb` - Labels tweets using Mistral AI API with resume capability 
3. `00c_data_labeling_with_stocks.ipynb` - (Optional) Labels tweets with verified stock symbols
4. `00_data_labeling.ipynb` - (Optional) Original data labeling without stock focus
5. `01_data_preparation.ipynb` - Prepares data for model training
6. `02a_gemma3_training_lora.ipynb` - Trains the Gemma 3 model with LoRA fine-tuning
7. `02b_finbert_training.ipynb` - Trains the FinBERT model (supports both 3-class and 5-class variants)

## Recent Updates
- Added project name: PARIS (Personalized AI-Advisor for Robo Investment Strategies)
- Created comprehensive main README.md with project overview, setup instructions, and documentation
- Added detailed information about common issues and their fixes
- Added documentation for Gemma 3 AttributeError fixes
- Added documentation for FinBERT column mismatch fixes
- Created detailed README.md in notebooks directory documenting notebook execution order and descriptions
- Organized notebooks in a clear execution sequence for reproducible results
- Added comprehensive documentation for dependencies and required API keys
- Documented output files generated by notebooks and their locations
- Documented model artifacts structure in models directory
- Improved Gemma 3 model training with 8-bit quantization for efficient training
- Standardized LoRA parameters (r=8, alpha=16) targeting q_proj and v_proj modules 
- Implemented multi-metric evaluation and early stopping consistently across models
- Configured efficient model saving formats (safetensors for FinBERT models)
- Added proper versioning for both 3-class and 5-class FinBERT implementations
- Completed data labeling of the stock_market_tweets dataset using the resume-capable approach
- Implemented Gemma 3 model with LoRA fine-tuning in 02a_gemma3_training_lora.ipynb
- Extended FinBERT model to support 5-class sentiment classification
- Added label mapping saving for FinBERT 5-class model to ensure consistent inference
- Modified model output directory to separate 3-class and 5-class models
- Improved 5-class FinBERT model evaluation with proper confusion matrix visualization
- Replaced OpenRouter Gemini 2.5 Pro API with Mistral AI API for sentiment labeling
- Implemented resume-capable data labeling with the new notebook 00_data_labeling_with_resume.ipynb
- Added KeyManager class for handling multiple Mistral API keys with rotation
- Added parallel processing with ThreadPoolExecutor for better performance
- Restructured data labeling workflow to support resuming from previous runs
- Added 5-class sentiment labeling using Mistral AI API
- Added stock symbol detection and verification step
- Added Named Entity Recognition (NER) preprocessing
- Modified data labeling to focus on stock-specific tweets
- Changed from Gemma 3 12B to more efficient Gemma 3 4B model
- Restructured development environment to be Docker-centric
- Replaced setup.sh with comprehensive Makefile
- Added development tools container
- Improved build system and cleanup procedures

## Known Issues and Fixes

### Gemma 3 AttributeError Fix
When working with Gemma 3 models, you may encounter:
```
AttributeError: 'Gemma3Config' object has no attribute 'hidden_size'
```

Solution: Modify the code to check for multiple attribute names:
```python
if hasattr(base_model.config, "model_dim"):
    hidden_dim = base_model.config.model_dim
elif hasattr(base_model.config, "hidden_size"):
    hidden_dim = base_model.config.hidden_size
else:
    # Default value for Gemma 3 models (4B model typically uses 4096)
    hidden_dim = 4096
```

### FinBERT Column Mismatch Fix
The FinBERT training notebook may expect columns `text` and `label`, but the dataset has `description` and `sentiment` columns.

Solutions:
1. Use the standalone Python script `finbert_training.py`
2. Modify the notebook to map columns appropriately:
```python
df['text'] = df['description']
sentiment_map = {
    'NEGATIVE': 0,
    'NEUTRAL': 1,
    'POSITIVE': 2
}
df['label'] = df['sentiment'].map(sentiment_map)
```
3. Use the automated fix script `fix_finbert_notebook.py`

## Next Steps
1. Complete model evaluation and comparison between Gemma 3, Gamma 3, and FinBERT models
2. Compare performance between 3-class and 5-class FinBERT models
3. Implement model versioning system
4. Add model performance comparison visualization
5. Implement real-time sentiment analysis API
6. Add automated testing suite

## Technical Decisions
1. Using 5-class sentiment labeling for more focused sentiment analysis
2. Maintaining both 3-class and 5-class FinBERT models to compare performance and granularity
3. Using Mistral AI API instead of OpenRouter for better reliability in sentiment classification
4. Using Polars instead of Pandas for more efficient dataframe operations
5. Using BERT-based NER to identify entities in financial tweets
6. Focusing ML training on tweets with verified stock symbols
7. Using LoRA for Gamma 3 and Gemma 3 to reduce training resources
8. Implementing resume-capable data labeling to handle large datasets reliably
9. Multi-metric early stopping for better model quality
10. Docker-based development for consistency
11. TypeScript + Tailwind for modern frontend
12. Makefile for standardized commands

## Dependencies
```
Python: 3.9+
Node.js: 18+
Docker & Docker Compose
Frameworks:
- FastAPI
- Next.js
- Transformers
- PyTorch
- Chart.js
- Polars
Additional:
- BERT-based NER
- yfinance
- Mistral AI API
- huggingface_hub
- peft
- accelerate
- bitsandbytes
- tensorboard
```

## API Endpoints
- GET `/metrics` - Model performance metrics
- GET `/confusion_matrices` - Confusion matrices
- GET `/sample_predictions` - Sample predictions
- GET `/performance_comparison` - Model comparison

## Key Project Files
- `notebooks/00b_ner_stock_identification.ipynb` - NER and stock symbol detection
- `notebooks/00_data_labeling_with_resume.ipynb` - Enhanced data labeling using Mistral AI API with resume capability
- `notebooks/00c_data_labeling_with_stocks.ipynb` - Labels tweets with verified stock symbols
- `notebooks/01_data_preparation.ipynb` - Prepares data for model training
- `notebooks/02a_gemma3_training_lora.ipynb` - Gemma 3 model training with LoRA
- `notebooks/02b_finbert_training.ipynb` - FinBERT model training (supports both 3-class and 5-class)
- `web/backend/` - FastAPI backend implementation
- `web/frontend/` - Next.js frontend with TypeScript
- `Makefile` - Development commands and workflow
- `docker-compose.yml` - Container configuration
- `finbert_training.py` - Standalone script for FinBERT training
- `fix_finbert_notebook.py` - Script to fix column mismatch in FinBERT notebook

## Key Output Files
- `data/tweets_with_ner_and_stocks.csv` - All tweets with NER and stock symbol information
- `data/tweets_with_verified_stocks.csv` - Filtered dataset with only tweets mentioning verified stock symbols
- `data/labeled_stock_tweets.csv` - Main labeled dataset from the resume-capable approach
- `data/labeled_stock_tweets_progress.json` - Progress tracking for resume capability
- `data/stock_tweets_labeled.csv` - Tweets with verified stock symbols and their sentiment labels
- `data/stock_tweets_for_training.csv` - Final dataset for model training
- `data/stock_tweets_by_symbol.csv` - Expanded dataset for analysis by individual stock symbol

## Model Artifacts Structure
### Gemma 3 Model (`models/gemma3/`)
- `gemma3_lora_adapter_best/` - Best performing LoRA adapter weights
- `gemma3_lora_adapter_final/` - Final trained LoRA adapter weights
- `metrics.csv` - Model evaluation metrics
- `training_history.csv` - Training progress history

### FinBERT Model (`models/finbert/` and `models/finbert_5labels/`)
- `model.safetensors` - Model weights in safetensors format
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer configuration
- `vocab.txt` - Vocabulary file
- `metrics.csv` - Model evaluation metrics

## Current Focus
- Comparing performance across all implemented models (FinBERT, Gamma 3, Gemma 3)
- Implementing and evaluating financial sentiment analysis models
- Building a web interface for sentiment prediction
- Supporting both 3-class and 5-class sentiment classification
- Developing robust data labeling and model training pipelines

## Active Decisions
- Using separate directories for 3-class and 5-class models to maintain both versions
- Saving label mapping alongside model weights for consistent inference
- Standardizing training parameters across models for fair comparison
- Using parameter-efficient fine-tuning (LoRA) for large foundation models
- Implementing early stopping with multiple metrics for better model quality
- Using Docker and Make-based workflow for consistent development environment

## Technical Considerations
- GPU memory constraints for larger models
- Model performance trade-offs between accuracy and inference speed
- API rate limitations for data labeling
- Containerization for reproducible development and deployment
- Model versioning and compatibility
- Frontend state management for complex model interactions
- Multiple API keys management for distributed labeling

## Design Considerations
- User interface for comparing model predictions
- Visualization of confidence scores
- Interactive exploration of sentiment analysis results
- Clear presentation of model strengths and limitations
- User experience for model selection and configuration
- Dashboard layout for effective information presentation

## Current Issues
- Balancing model size and performance
- Ensuring consistent label mapping across model versions
- Managing training resources efficiently
- Handling API rate limitations for large-scale labeling
- Ensuring reproducibility of model training across environments
- Comparing 3-class and 5-class models fairly despite different class distributions
