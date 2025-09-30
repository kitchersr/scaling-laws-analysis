"""
TRANSFORMER MODEL FOR LANGUAGE MODELING
========================================

This file implements a transformer model from scratch for next-token prediction.

EXECUTION FLOW (when you use the model):
1. You call: model(input_ids)
2. TransformerModel.forward() is called automatically
3. It creates embeddings (word + position)
4. Passes through multiple TransformerBlocks
5. Each block does: Attention → FeedForward → Output
6. Final prediction: which word comes next?

FILE STRUCTURE:
- MultiHeadAttention: The core attention mechanism
- FeedForward: Simple neural network layer
- TransformerBlock: Combines attention + feedforward
- TransformerModel: The complete model (uses everything above)
- Helper functions: Configuration and utilities

Read the classes top-to-bottom to understand the pieces,
but remember execution flows: TransformerModel → TransformerBlock → MultiHeadAttention
"""

import torch
import torch.nn as nn
import math


# ============================================================================
# PART 1: MULTI-HEAD ATTENTION
# ============================================================================
# This is the core "magic" of transformers - learning which words to pay
# attention to when processing each word.

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    The attention mechanism asks: "When processing word X, which other words
    should I pay attention to?" It learns this through Query, Key, Value.
    
    Multi-head means we do this multiple times in parallel, allowing the model
    to learn different types of relationships (e.g., subject-verb, adjective-noun).
    """
    
    def __init__(self, d_model, n_heads):
        """
        Initialize the attention mechanism.
        
        Args:
            d_model: Dimension of embeddings (e.g., 128, 256, 512)
            n_heads: Number of attention heads (e.g., 8)
        
        What this does:
        - Stores the model dimensions
        - Calculates dimensions per head (d_k = d_model / n_heads)
        - Creates random weight matrices for Q, K, V, O (learned during training)
        """
        super().__init__()
        
        # Store dimensions
        self.d_model = d_model          # Total embedding dimension (e.g., 128)
        self.n_heads = n_heads          # Number of attention heads (e.g., 8)
        self.d_k = d_model // n_heads   # Dimension per head (e.g., 128/8 = 16)
        
        # Create learnable transformation matrices
        # These start as random numbers and learn during training
        # nn.Linear creates a matrix that transforms input dimension → output dimension
        
        self.w_q = nn.Linear(d_model, d_model)  # Query transformation (128→128)
        # w_q learns: "Given a word, what should it look for?"
        
        self.w_k = nn.Linear(d_model, d_model)  # Key transformation (128→128)
        # w_k learns: "Given a word, what information does it offer?"
        
        self.w_v = nn.Linear(d_model, d_model)  # Value transformation (128→128)
        # w_v learns: "Given a word, what information should it provide?"
        
        self.w_o = nn.Linear(d_model, d_model)  # Output transformation (128→128)
        # w_o learns: "How to combine information from all heads?"
        
    def forward(self, x, mask=None):
        """
        Process input through attention mechanism.
        
        Args:
            x: Input embeddings, shape (batch_size, seq_len, d_model)
               Example: (1, 3, 128) = 1 sentence, 3 words, 128 dimensions each
            mask: Optional mask to prevent attending to future words
        
        Returns:
            Attention output: same shape as input (batch_size, seq_len, d_model)
        
        What this does:
        1. Transform x into Q, K, V for all words at once
        2. Split into multiple heads
        3. Compute attention scores (which words relate to which)
        4. Apply mask (prevent looking at future words)
        5. Compute weighted combination of Values
        6. Combine all heads and return
        """
        
        # Get dimensions from input
        # If x has shape (1, 3, 128), this extracts: batch_size=1, seq_len=3, d_model=128
        batch_size, seq_len, d_model = x.size()
        
        # STEP 1: Create Q, K, V by transforming input embeddings
        # --------------------------------------------------------
        # For sentence "the cat sat" with embeddings x:
        # x = [[0.2, -0.1, 0.6, ...],  ← "the" (128 numbers)
        #      [0.8, 0.7, 0.3, ...],   ← "cat" (128 numbers)
        #      [0.4, -0.4, 0.7, ...]]  ← "sat" (128 numbers)
        
        # Apply learned transformations
        Q = self.w_q(x)  # Query: what each word is looking for
        K = self.w_k(x)  # Key: what each word has to offer
        V = self.w_v(x)  # Value: actual information each word provides
        
        # Q, K, V each have shape (batch_size, seq_len, d_model) = (1, 3, 128)
        
        # STEP 2: Split into multiple heads
        # ----------------------------------
        # Why? Different heads can learn different relationships
        # Head 1 might learn subject-verb, Head 2 might learn adjective-noun, etc.
        
        # Reshape to split dimensions across heads
        # .view() reshapes: (1, 3, 128) → (1, 3, 8, 16)
        #   batch_size=1, seq_len=3, n_heads=8, d_k=16 per head
        # .transpose(1, 2) swaps dimensions: (1, 3, 8, 16) → (1, 8, 3, 16)
        #   Now shape is: (batch_size, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # STEP 3: Compute attention scores
        # ---------------------------------
        # For each word's Query, compute similarity with all Keys
        # torch.matmul(Q, K.transpose(-2, -1)) computes all dot products at once
        
        # K.transpose(-2, -1) flips last two dimensions for matrix multiplication
        # scores will have shape (batch_size, n_heads, seq_len, seq_len)
        # Example for 3 words: (1, 8, 3, 3) = for each head, 3x3 matrix of scores
        #   [[Q_the·K_the, Q_the·K_cat, Q_the·K_sat],
        #    [Q_cat·K_the, Q_cat·K_cat, Q_cat·K_sat],
        #    [Q_sat·K_the, Q_sat·K_cat, Q_sat·K_sat]]
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Why divide by sqrt(d_k)?
        # Scaling factor to keep numbers stable (prevents very large values)
        # Without this, softmax can become too peaked (all attention on one word)
        
        # STEP 4: Apply causal mask
        # --------------------------
        # For language modeling, words can only attend to past words, not future
        # mask looks like: [[1, 0, 0],   ← "the" can only see "the"
        #                   [1, 1, 0],   ← "cat" can see "the" and "cat"
        #                   [1, 1, 1]]   ← "sat" can see all words
        
        if mask is not None:
            # Replace positions where mask=0 with large negative number
            # -1e9 = -1,000,000,000 (effectively negative infinity)
            # After softmax, these become ~0 probability
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # STEP 5: Convert scores to probabilities with softmax
        # -----------------------------------------------------
        # Softmax converts any numbers to probabilities (sum to 1.0)
        # Example: scores = [0.1, 0.8, -1e9] → attn_weights = [0.3, 0.7, 0.0]
        
        attn_weights = torch.softmax(scores, dim=-1)
        # Shape: (batch_size, n_heads, seq_len, seq_len)
        # attn_weights[0, 0, 2, :] = how much "sat" (word 2) attends to each word
        
        # STEP 6: Apply attention to Values
        # ----------------------------------
        # Compute weighted sum: for each word, take weighted combination of all Values
        # torch.matmul(attn_weights, V) does this for all words at once
        
        attn_output = torch.matmul(attn_weights, V)
        # Shape: (batch_size, n_heads, seq_len, d_k)
        
        # Example for "sat":
        # sat_output = 0.1×V_the + 0.6×V_cat + 0.3×V_sat
        # Now "sat" representation includes information about "cat"!
        
        # STEP 7: Concatenate heads back together
        # ----------------------------------------
        # We split into 8 heads, now combine them back
        # .transpose(1, 2): (1, 8, 3, 16) → (1, 3, 8, 16)
        # .contiguous(): ensures memory is laid out properly for next operation
        # .view(): (1, 3, 8, 16) → (1, 3, 128) flatten last two dimensions
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # STEP 8: Final output transformation
        # ------------------------------------
        # Apply learned transformation to combined heads
        # This allows the model to learn how to best use the multi-head attention
        
        return self.w_o(attn_output)


# ============================================================================
# PART 2: FEED-FORWARD NETWORK
# ============================================================================
# After attention, we pass through a simple neural network.
# This adds non-linearity and gives the model more expressive power.

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    This is a simple two-layer neural network applied to each word independently.
    Structure: x → expand (d_model → d_ff) → activation → compress (d_ff → d_model)
    
    Think of it as: take each word's representation, expand it to think deeply,
    then compress back to original size with richer understanding.
    """
    
    def __init__(self, d_model, d_ff):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Input/output dimension (e.g., 128)
            d_ff: Hidden dimension (typically 4×d_model, e.g., 512)
        """
        super().__init__()
        
        # Two linear transformations with activation in between
        self.linear1 = nn.Linear(d_model, d_ff)      # Expand: 128 → 512
        self.linear2 = nn.Linear(d_ff, d_model)      # Compress: 512 → 128
        self.gelu = nn.GELU()                        # Activation function (smooth version of ReLU)
        
    def forward(self, x):
        """
        Process input through feed-forward network.
        
        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor, same shape as input
        
        What this does:
        1. Expand dimensions: transform each word from 128 → 512 dimensions
        2. Apply GELU activation (adds non-linearity)
        3. Compress back: 512 → 128 dimensions
        
        This happens independently for each word (position-wise).
        """
        # x: (batch_size, seq_len, 128)
        # → linear1 → (batch_size, seq_len, 512)
        # → gelu → (batch_size, seq_len, 512)
        # → linear2 → (batch_size, seq_len, 128)
        
        return self.linear2(self.gelu(self.linear1(x)))


# ============================================================================
# PART 3: TRANSFORMER BLOCK
# ============================================================================
# Combines attention + feed-forward with residual connections and layer norm.
# This is one "layer" of the transformer. We'll stack multiple blocks.

class TransformerBlock(nn.Module):
    """
    Single transformer block: Attention → FeedForward with residual connections.
    
    Structure:
    1. Multi-head attention
    2. Add & Normalize (residual connection + layer normalization)
    3. Feed-forward network
    4. Add & Normalize (residual connection + layer normalization)
    
    Residual connections (x + output) help with training deep networks.
    Layer normalization stabilizes the values.
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension (e.g., 128)
            n_heads: Number of attention heads (e.g., 8)
            d_ff: Feed-forward hidden dimension (e.g., 512)
            dropout: Dropout probability for regularization (e.g., 0.1 = 10% dropout)
        """
        super().__init__()
        
        # Create sub-layers
        self.attention = MultiHeadAttention(d_model, n_heads)  # Attention mechanism
        self.feed_forward = FeedForward(d_model, d_ff)         # Feed-forward network
        
        # Layer normalization: normalizes values to have mean=0, std=1
        # Helps with training stability
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout: randomly sets some values to 0 during training
        # Prevents overfitting by forcing model to be robust
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Process input through one transformer block.
        
        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)
            mask: Optional causal mask for attention
        
        Returns:
            Output tensor, same shape as input but with richer representations
        
        What this does:
        1. Pass through attention (words gather context from other words)
        2. Add original input + apply dropout + normalize
        3. Pass through feed-forward (each word processed independently)
        4. Add previous output + apply dropout + normalize
        """
        
        # SUB-LAYER 1: Self-attention with residual connection
        # -----------------------------------------------------
        # Call attention mechanism (this calls MultiHeadAttention.forward())
        # Remember: attention(x, mask) is shorthand for attention.forward(x, mask)
        attn_output = self.attention(x, mask)
        
        # Residual connection: add original input to attention output
        # Why? Helps gradient flow during training (combats vanishing gradients)
        # Apply dropout for regularization, then normalize
        x = self.norm1(x + self.dropout(attn_output))
        
        # SUB-LAYER 2: Feed-forward with residual connection
        # ---------------------------------------------------
        # Call feed-forward network (this calls FeedForward.forward())
        ff_output = self.feed_forward(x)
        
        # Residual connection again: add previous output to feed-forward output
        # Apply dropout for regularization, then normalize
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


# ============================================================================
# PART 4: COMPLETE TRANSFORMER MODEL
# ============================================================================
# This brings everything together: embeddings → transformer blocks → prediction

class TransformerModel(nn.Module):
    """
    Complete transformer model for language modeling (next-token prediction).
    
    Architecture:
    1. Embedding layer (convert word indices to vectors)
    2. Position embedding (add positional information)
    3. Stack of N transformer blocks (attention + feed-forward)
    4. Final layer norm
    5. Output projection (predict next word)
    
    This is called the "decoder-only" transformer architecture, used by GPT models.
    """
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len, dropout=0.1):
        """
        Initialize the complete transformer model.
        
        Args:
            vocab_size: Size of vocabulary (e.g., 50,000 words)
            d_model: Model dimension (e.g., 128, 256, 512)
            n_heads: Number of attention heads (e.g., 8)
            n_layers: Number of transformer blocks to stack (e.g., 6, 12)
            max_seq_len: Maximum sequence length (e.g., 512 words)
            dropout: Dropout probability (e.g., 0.1)
        """
        super().__init__()
        
        # Store dimensions
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # EMBEDDINGS: Convert word indices to vectors
        # --------------------------------------------
        # Token embedding: lookup table mapping word index → vector
        # vocab_size rows (one per word), d_model columns (vector dimension)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Example: "cat" (index 1) → [0.8, 0.7, 0.3, ...] (128 numbers)
        
        # Position embedding: lookup table mapping position → vector
        # max_seq_len rows (one per position), d_model columns
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        # Example: position 0 → [0.1, 0.2, -0.1, ...] (128 numbers)
        
        # TRANSFORMER BLOCKS: Stack multiple layers
        # ------------------------------------------
        # Create a list of n_layers transformer blocks
        # Each block has: attention + feed-forward + residual connections
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_model * 4,  # Feed-forward hidden size is typically 4× model size
                dropout=dropout
            )
            for _ in range(n_layers)  # Repeat n_layers times
        ])
        
        # OUTPUT LAYERS: Convert final representations to word predictions
        # -----------------------------------------------------------------
        # Final layer normalization for stability
        self.ln_f = nn.LayerNorm(d_model)
        
        # Language model head: projects d_model dimensions → vocab_size dimensions
        # Output is scores for each word in vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # WEIGHT INITIALIZATION
        # ---------------------
        # Initialize all weights using custom initialization function
        # This gives better starting values than default random initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """
        Initialize weights for model components.
        
        Args:
            module: A component of the model (Linear, Embedding, etc.)
        
        What this does:
        - For Linear layers: initialize weights from normal distribution (mean=0, std=0.02)
        - For Linear layers with bias: initialize bias to zeros
        - For Embedding layers: initialize from normal distribution (mean=0, std=0.02)
        
        Why? Proper initialization helps the model train faster and more stably.
        """
        if isinstance(module, nn.Linear):
            # Initialize weight matrix with small random values
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # If this layer has bias, initialize to zeros
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Initialize embedding table with small random values
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, input_ids, labels=None):
        """
        Forward pass: process input through entire model.
        
        This is THE MAIN FUNCTION that gets called when you use the model.
        When you do: output = model(input_ids)
        Python automatically calls: output = model.forward(input_ids)
        
        Args:
            input_ids: Tensor of word indices, shape (batch_size, seq_len)
                      Example: [[0, 1, 2]] = one sentence with 3 words
            labels: Optional tensor of target word indices for training
                   Same shape as input_ids
        
        Returns:
            Dictionary with:
            - "logits": Predicted scores for each word, shape (batch_size, seq_len, vocab_size)
            - "loss": Training loss (if labels provided), otherwise None
        
        EXECUTION FLOW:
        1. Create embeddings (word + position)
        2. Create causal mask
        3. Pass through all transformer blocks sequentially
        4. Apply final layer norm
        5. Project to vocabulary size (get word predictions)
        6. Calculate loss if training
        """
        
        # Get dimensions from input
        # input_ids has shape (batch_size, seq_len)
        # Example: (1, 3) = 1 sentence, 3 words
        batch_size, seq_len = input_ids.size()
        
        # STEP 1: Create causal mask
        # ---------------------------
        # For language modeling, each position can only attend to previous positions
        # torch.tril creates lower triangular matrix (1s below diagonal, 0s above)
        # Example for 3 words:
        # [[1, 0, 0],
        #  [1, 1, 0],
        #  [1, 1, 1]]
        mask = torch.tril(torch.ones(seq_len, seq_len))
        # Add two dimensions for batch and heads: (seq_len, seq_len) → (1, 1, seq_len, seq_len)
        mask = mask.unsqueeze(0).unsqueeze(0)
        # Move to same device as input (CPU or GPU)
        mask = mask.to(input_ids.device)
        
        # STEP 2: Create embeddings
        # --------------------------
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        # torch.arange(0, seq_len) creates: [0, 1, 2] for seq_len=3
        # .unsqueeze(0) adds batch dimension: [0, 1, 2] → [[0, 1, 2]]
        # .to(input_ids.device) moves to same device as input
        positions = torch.arange(0, seq_len).unsqueeze(0).to(input_ids.device)
        
        # Look up embeddings
        # token_embedding(input_ids): converts word indices to vectors
        #   input_ids [0, 1, 2] → [[vec_the, vec_cat, vec_sat]]
        # position_embedding(positions): converts position indices to vectors
        #   positions [0, 1, 2] → [[vec_pos0, vec_pos1, vec_pos2]]
        # Add them together element-wise
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # x now has shape (batch_size, seq_len, d_model)
        # Example: (1, 3, 128) = 1 sentence, 3 words, 128-dim vectors
        
        # STEP 3: Pass through transformer blocks
        # ----------------------------------------
        # Process through each transformer block sequentially
        # Each block enhances the representations with attention + feed-forward
        for block in self.blocks:
            # block(x, mask) calls TransformerBlock.forward(x, mask)
            # Input: x with current representations
            # Output: x with enhanced representations
            # We keep reassigning to x, so it gets progressively richer
            x = block(x, mask)
        
        # After all blocks, x has shape (batch_size, seq_len, d_model)
        # But now each word's vector contains rich contextual information
        
        # STEP 4: Final layer normalization
        # ----------------------------------
        # Normalize the final representations for stability
        x = self.ln_f(x)
        
        # STEP 5: Project to vocabulary size
        # -----------------------------------
        # Transform from d_model dimensions to vocab_size dimensions
        # This gives us a score for each possible next word
        logits = self.lm_head(x)
        # logits has shape (batch_size, seq_len, vocab_size)
        # Example: (1, 3, 50000) = for each of 3 positions, scores for 50,000 words
        
        # STEP 6: Calculate loss (if training)
        # -------------------------------------
        loss = None
        if labels is not None:
            # For language modeling, we predict the NEXT token
            # Input: "the cat sat" → Labels: "cat sat <end>"
            # So we need to shift: compare prediction at position i with label at position i+1
            
            # Shift logits: remove last position (no next word to predict after last word)
            # [..., :-1, :] means: take all except last position
            # .contiguous() ensures memory is laid out correctly
            shift_logits = logits[..., :-1, :].contiguous()
            
            # Shift labels: remove first position (no previous word to predict for first word)
            # [..., 1:] means: take all except first position
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            # This measures how well our predictions match the true next words
            loss_fn = nn.CrossEntropyLoss()
            # Flatten to 2D: (batch×seq_len, vocab_size) and (batch×seq_len)
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),  # (batch*seq_len, vocab_size)
                shift_labels.view(-1)                           # (batch*seq_len)
            )
        
        # Return predictions and loss
        return {"loss": loss, "logits": logits}


# ============================================================================
# PART 5: HELPER FUNCTIONS
# ============================================================================
# Utility functions for creating models and analyzing them

def get_model_config(target_params_millions):
    """
    Get model configuration for target parameter count.
    
    Args:
        target_params_millions: Target model size in millions of parameters
                               Options: 1, 10, or 100
    
    Returns:
        Dictionary with model configuration:
        - d_model: embedding dimension
        - n_heads: number of attention heads
        - n_layers: number of transformer blocks
    
    This function provides pre-calculated configurations that approximately
    hit the target parameter counts. These are rough estimates.
    
    Usage:
        config = get_model_config(10)  # Get config for ~10M parameter model
        model = TransformerModel(vocab_size=50000, **config, max_seq_len=512)
    """
    configs = {
        1: {
            "d_model": 128,    # Small embedding size
            "n_heads": 8,      # 8 attention heads
            "n_layers": 6      # 6 transformer blocks
        },    # Results in ~1M parameters
        
        10: {
            "d_model": 256,    # Medium embedding size
            "n_heads": 8,      # 8 attention heads
            "n_layers": 8      # 8 transformer blocks
        },   # Results in ~10M parameters
        
        100: {
            "d_model": 512,    # Large embedding size
            "n_heads": 8,      # 8 attention heads
            "n_layers": 12     # 12 transformer blocks
        }  # Results in ~100M parameters
    }
    return configs[target_params_millions]


def count_parameters(model):
    """
    Count the total number of trainable parameters in the model.
    
    Args:
        model: A PyTorch model (TransformerModel instance)
    
    Returns:
        Integer: total number of trainable parameters
    
    What this does:
    - Iterates through all parameters (weights and biases) in the model
    - For each parameter, counts the total number of values (.numel())
    - Only counts parameters that are trainable (requires_grad=True)
    - Sums them all up
    
    Usage:
        model = TransformerModel(...)
        num_params = count_parameters(model)
        print(f"Model has {num_params:,} parameters")
        # Output: Model has 1,234,567 parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# USAGE EXAMPLE (Not executed, just for reference)
# ============================================================================
"""
# Step 1: Get configuration for desired model size
config = get_model_config(1)  # 1M parameter model

# Step 2: Create the model
model = TransformerModel(
    vocab_size=50000,              # 50k word vocabulary
    d_model=config["d_model"],     # 128 dimensions
    n_heads=config["n_heads"],     # 8 heads
    n_layers=config["n_layers"],   # 6 layers
    max_seq_len=512,               # Max 512 words
    dropout=0.1                    # 10% dropout
)

# Step 3: Check parameter count
num_params = count_parameters(model)
print(f"Model has {num_params:,} parameters")

# Step 4: Use the model
input_ids = torch.tensor([[0, 1, 2]])  # "the cat sat"
output = model(input_ids)              # Calls model.forward(input_ids)
logits = output["logits"]              # Get predictions
# logits shape: (1, 3, 50000) = scores for each next word
"""

def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)