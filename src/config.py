# Model sizes to test (parameters in millions)
MODEL_SIZES = [1, 10, 100]  # Test 3 different model sizes to see scaling effects

# Training configuration
DATASET_NAME = "wikitext"              # The dataset name on Hugging Face
DATASET_CONFIG = "wikitext-2-raw-v1"   # Specific version - WikiText-2 is perfect size for our experiment
MAX_TOKENS = 100_000_000               # How much text to train on (100 million words/tokens)
EVAL_TOKENS = 1_000_000                # Smaller amount for testing model performance

# Training hyperparameters - these control how the model learns
LEARNING_RATE = 1e-3    # How big steps the model takes when learning (0.001)
BATCH_SIZE = 32         # How many examples to process at once
MAX_EPOCHS = 10         # How many times to go through the entire dataset
SAVE_STEPS = 1000       # Save model every 1000 training steps
DROPOUT = 0.1		# Regularisation

# Paths - where to save things
OUTPUT_DIR = "results"  # Folder for plots and analysis
MODEL_DIR = "models"    # Folder for trained models

# Weights & Biases - for tracking experiments
WANDB_PROJECT = "scaling-laws-analysis"  # Your project name on W&B dashboard