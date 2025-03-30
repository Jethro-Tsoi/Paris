# Financial Sentiment Analysis with LLMs

This project implements a comprehensive financial sentiment analysis system using Google's Gamma 3, Gemma 3, and FinBERT models, with a modern web interface for visualization and comparison.

## Features

- 🤖 Multi-model sentiment analysis (Gamma 3, Gemma 3, and FinBERT)
- 📊 Interactive performance visualization dashboard
- 🔄 Real-time sentiment prediction
- 🎯 5-class sentiment classification (STRONGLY_POSITIVE, POSITIVE, NEUTRAL, NEGATIVE, STRONGLY_NEGATIVE)
- 📈 Comprehensive model evaluation metrics
- 🚀 Modern web interface with TypeScript and Tailwind CSS
- 🐳 Containerized development environment
- ⚡ Resume-capable data labeling for large datasets

## Prerequisites

- Docker and Docker Compose
- Make (optional, but recommended)
- Mistral AI API keys

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-sentiment.git
cd financial-sentiment
```

2. Set up the environment:
```bash
make setup
```

3. Configure your API keys in `.env`:
```bash
# Copy the example file
cp .env.example .env

# Edit with your Mistral API keys
nano .env
```

4. Start the development environment:
```bash
make up
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

The following services will be available:

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Jupyter Notebooks: http://localhost:8888

## Project Structure

```
.
├── data/                  # Data storage
│   ├── models/           # Trained model files
│   └── tweets/           # Raw and processed tweets
├── notebooks/            # Jupyter notebooks
│   ├── 00b_ner_stock_identification.ipynb  # NER and stock symbol detection
│   ├── 00_data_labeling_with_resume.ipynb  # Resume-capable data labeling with Mistral AI
│   ├── 00c_data_labeling_with_stocks.ipynb # Stock-specific sentiment labeling
│   ├── 00_data_labeling.ipynb              # Original data labeling (optional)
│   ├── 02a_gamma3_training_lora.ipynb      # Gamma 3 training
│   ├── 02b_finbert_training.ipynb          # FinBERT training
│   └── 02b_gemma3_training_lora.ipynb      # Gemma 3 training
├── web/                  # Web application
│   ├── backend/         # FastAPI backend
│   └── frontend/        # Next.js frontend
├── Makefile             # Development commands
└── docker-compose.yml   # Container configuration
```

## Model Training

### Data Labeling

1. Start Jupyter server:
```bash
make jupyter
```

2. Open `notebooks/00_data_labeling_with_resume.ipynb`

3. Configure your Mistral API keys in environment variables:
```
MISTRAL_API_KEY=your_primary_key
MISTRAL_API_KEY_1=your_second_key
MISTRAL_API_KEY_2=your_third_key
```

4. The notebook supports resuming labeling from previous runs using checkpoint files.

### Training Models

1. Gamma 3 with LoRA:
- Open `notebooks/02a_gamma3_training_lora.ipynb`
- Features:
  - LoRA fine-tuning (r=8, alpha=16)
  - Multi-metric early stopping
  - Comprehensive evaluation

2. Gemma 3 with LoRA:
- Open `notebooks/02b_gemma3_training_lora.ipynb`
- Features:
  - 8-bit quantization
  - LoRA fine-tuning
  - Gradient clipping
  - Multi-metric monitoring

3. FinBERT:
- Open `notebooks/02b_finbert_training.ipynb`
- Features:
  - Native fine-tuning
  - Early stopping
  - Performance metrics

## Development

### Running Tests

```bash
# Backend tests
make test-backend

# Frontend tests
make test-frontend
```

### Development Tools

Start development environment with additional tools:
```bash
make dev
```

This includes:
- Hot reloading
- Development containers
- Debugging tools
- Live code updates

### Making Changes

1. Frontend:
- Edit files in `web/frontend/src/`
- Changes reflect instantly with hot reloading

2. Backend:
- Edit files in `web/backend/`
- Auto-reloads with code changes

3. Notebooks:
- Edit in Jupyter interface
- Auto-saves enabled

## API Key Management

The project uses a custom KeyManager class to handle multiple Mistral AI API keys with automatic rotation on rate limits:

1. Set your API keys in the `.env` file:
```
MISTRAL_API_KEY=your_primary_key
MISTRAL_API_KEY_1=your_second_key
```

2. The system will automatically:
- Rotate keys when rate limits are reached
- Track when rate-limited keys become available again
- Implement waiting strategies when all keys are limited

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Monitoring

View service logs:
```bash
make logs
```

Monitor specific service:
```bash
make logs service=backend  # or frontend
```

## Cleanup

Remove all containers and volumes:
```bash
make clean
```

## License

MIT License - see LICENSE file for details

## Need Help?

1. Run `make help` to see all available commands
2. Check the API documentation at http://localhost:8000/docs
3. Visit the frontend at http://localhost:3000

# Fixing Gemma 3 Model AttributeError

When working with Google's Gemma 3 models, you may encounter this error:

```
AttributeError: 'Gemma3Config' object has no attribute 'hidden_size'
```

## Solution

Gemma 3 models use different attribute names in their configuration compared to traditional transformer models. To fix this error, modify your code to check for multiple possible attribute names:

```python
class GemmaForSequenceClassification(nn.Module):
    def __init__(self, base_model, num_labels=5):
        nn.Module.__init__(self)
        self.base_model = base_model
        self.num_labels = num_labels
        
        # For Gemma 3 models, try different attribute names for the hidden dimension
        if hasattr(base_model.config, "model_dim"):
            hidden_dim = base_model.config.model_dim
        elif hasattr(base_model.config, "hidden_size"):
            hidden_dim = base_model.config.hidden_size
        else:
            # Default value for Gemma 3 models
            # 4B model typically uses 4096, 12B uses 8192
            print("Warning: Could not find hidden dimension. Using default of 4096.")
            hidden_dim = 4096
            
        print(f"Using hidden dimension: {hidden_dim}")
        self.classifier = nn.Linear(hidden_dim, num_labels)
        
    # ... rest of the class remains the same
```

This fix allows your code to work with Gemma 3 models by:
1. First checking for `model_dim` (used by Gemma 3)
2. Falling back to `hidden_size` (used by many other models)
3. Using a reasonable default value if neither attribute is found

## More Information

The correct hidden dimension values for Gemma 3 models are:
- 1B model: ~2048
- 4B model: ~4096
- 12B model: ~8192
- 27B model: ~12288

These values may vary slightly based on the exact model architecture.

# FinBERT Column Mismatch Fix

This repository contains a fix for the column mismatch in the FinBERT training notebook.

## Issue

The original notebook `notebooks/02b_finbert_training.ipynb` attempts to use columns `text` and `label`, but the dataset has `description` and `sentiment` columns instead.

## Solutions

There are two ways to fix this:

### Option 1: Use the standalone Python script

The file `finbert_training.py` contains a standalone version of the FinBERT training code that automatically:
- Maps the `description` column to `text`
- Maps the `sentiment` column to `label` with the appropriate conversion
- Handles missing values

Run the script with:
```bash
python finbert_training.py
```

### Option 2: Modify the notebook directly

In the notebook, change the data loading function to:

```python
def load_labeled_data(file_path):
    """Load Gemini-labeled dataset"""
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} labeled tweets")
    
    # Map columns to expected format
    # Use 'description' as 'text'
    df['text'] = df['description']
    
    # Convert 'sentiment' to numeric 'label'
    sentiment_map = {
        'NEGATIVE': 0,
        'NEUTRAL': 1,
        'POSITIVE': 2
    }
    df['label'] = df['sentiment'].map(sentiment_map)
    
    # Drop rows with missing values in text or label
    df = df.dropna(subset=['text', 'label'])
    print(f"Final dataset shape after cleaning: {df.shape}")
    
    return df
```

### Option 3: Use the automated fix script

The `fix_finbert_notebook.py` script can automatically create a fixed version of the notebook:

```bash
python fix_finbert_notebook.py
```
