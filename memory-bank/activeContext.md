# Active Development Context

## Project Overview
Financial sentiment analysis system using Google's Gemma 3, Gamma 3 and FinBERT models with a modern web interface.

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

## Current Working Branches
- main: Primary development branch
- feature/model-training: Model training implementations
- feature/web-interface: Web application development

## Recent Updates
- Train models using the newly labeled data
- Complete data labeling of the stock_market_tweets dataset using the resume-capable approach
- Implemented Gemma 3 model with LoRA fine-tuning in 02a_gemma3_training_lora.ipynb
- Extended FinBERT model to support 5-class sentiment classification (STRONGLY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, STRONGLY_POSITIVE)
- Added label mapping saving for FinBERT 5-class model to ensure consistent inference
- Modified model output directory to separate 3-class and 5-class models
- Improved 5-class FinBERT model evaluation with proper confusion matrix visualization
- Replaced OpenRouter Gemini 2.5 Pro API with Mistral AI API for sentiment labeling
- Implemented resume-capable data labeling with the new notebook 00_data_labeling_with_resume.ipynb
- Deprecated 01_data_preparation.ipynb in favor of 00_data_labeling_with_resume.ipynb
- Added KeyManager class for handling multiple Mistral API keys with rotation
- Added parallel processing with ThreadPoolExecutor for better performance
- Restructured data labeling workflow to support resuming from previous runs
- Added 5-class sentiment labeling using Mistral AI API
- Added stock symbol detection and verification step
- Added Named Entity Recognition (NER) preprocessing
- Modified data labeling to focus on stock-specific tweets
- Added Gemma 3 model implementation with LoRA fine-tuning
- Changed from Gemma 3 12B to more efficient Gemma 3 4B model
- Restructured development environment to be Docker-centric
- Replaced setup.sh with comprehensive Makefile
- Added development tools container
- Improved build system and cleanup procedures

## Next Steps
1. Complete model evaluation and comparison between Gemma 3 Finetune, and FinBERT models
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
```

## API Endpoints
- GET `/metrics` - Model performance metrics
- GET `/confusion_matrices` - Confusion matrices
- GET `/sample_predictions` - Sample predictions
- GET `/performance_comparison` - Model comparison

## Key Project Files
- `notebooks/00_data_labeling_with_resume.ipynb` - Enhanced data labeling using Mistral AI API with resume capability
- `notebooks/00b_ner_stock_identification.ipynb` - NER and stock symbol detection
- `notebooks/02a_gemma3_training_lora.ipynb` - Gemma 3 model training with LoRA
- `notebooks/02b_finbert_training.ipynb` - FinBERT model training (3-class)
- `notebooks/02b_finbert_training_5labels.ipynb` - FinBERT model training (5-class)
- `web/backend/` - FastAPI backend implementation
- `web/frontend/` - Next.js frontend with TypeScript
- `Makefile` - Development commands and workflow
- `docker-compose.yml` - Container configuration

## Current Focus
- Comparing performance across all implemented models (FinBERT, Gamma 3)
- Implementing and evaluating financial sentiment analysis models
- Building a web interface for sentiment prediction
- Supporting both 3-class and 5-class sentiment classification
- Developing robust data labeling and model training pipelines

## Recent Changes
- Train models using the newly labeled data
- Complete data labeling of the stock_market_tweets dataset using the resume-capable approach
- Added Gemma 3 implementation with LoRA fine-tuning in 02a_gemma3_training_lora.ipynb
- Modified FinBERT training to support 5-class sentiment classification
- Updated the label mapping to: STRONGLY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, STRONGLY_POSITIVE
- Changed model output directory to '../models/finbert_5labels' to avoid overwriting the 3-class model
- Used 'description' column instead of 'text' for model input
- Added code to save the label mapping for consistent inference
- Updated model initialization and confusion matrix to handle 5 classes
- Configured model training parameters (batch_size=16, learning_rate=1e-5, early_stopping_patience=3)
- Maintained backward compatibility with existing 3-class model implementation
- Developed improved data labeling workflow with resume capability

## Next Steps
- Complete evaluation of 5-class FinBERT model performance
- Compare results between 3-class and 5-class models
- Integrate 5-class model into the web interface
- Update API to support both 3-class and 5-class predictions
- Consider fine-tuning Gemma 3 with 5-class classification
- Develop visualization components for model comparison
- Implement model versioning system for tracking different configurations

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
