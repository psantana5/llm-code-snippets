import torch
import torch.nn as nn
import math

# Add a helper function for verbose printing
def verbose_print(message, enabled=True):
    if enabled:
        print(f"[VERBOSE] {message}")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, verbose=False):
        super(PositionalEncoding, self).__init__()
        self.verbose = verbose
        
        verbose_print(f"Initializing PositionalEncoding with d_model={d_model}, max_seq_length={max_seq_length}", self.verbose)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        verbose_print("Positional encoding matrix created with sine and cosine patterns", self.verbose)
        
    def forward(self, x):
        verbose_print(f"PositionalEncoding: Adding positional information to input tensor of shape {x.shape}", self.verbose)
        # Add positional encoding to input
        result = x + self.pe[:, :x.size(1), :]
        verbose_print(f"PositionalEncoding: Output tensor shape: {result.shape}", self.verbose)
        return result

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, verbose=False):
        super(MultiHeadAttention, self).__init__()
        self.verbose = verbose
        
        verbose_print(f"Initializing MultiHeadAttention with d_model={d_model}, num_heads={num_heads}", self.verbose)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        verbose_print(f"Each attention head will have dimension d_k={self.d_k}", self.verbose)
        
        # Linear projections for Q, K, V, and output
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        verbose_print(f"MultiHeadAttention: Processing input tensors Q:{q.shape}, K:{k.shape}, V:{v.shape}", self.verbose)
        
        # Linear projections and reshape for multi-head attention
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        verbose_print(f"After projection and reshaping for {self.num_heads} heads: Q:{q.shape}, K:{k.shape}, V:{v.shape}", self.verbose)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        verbose_print(f"Attention scores calculated with shape {scores.shape}", self.verbose)
        
        # Apply mask if provided
        if mask is not None:
            verbose_print(f"Applying attention mask with shape {mask.shape}", self.verbose)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        verbose_print(f"Attention weights calculated with shape {attn_weights.shape}", self.verbose)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)
        verbose_print(f"Applied attention weights to values, output shape: {output.shape}", self.verbose)
        
        # Reshape and apply final linear projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(output)
        
        verbose_print(f"MultiHeadAttention: Final output shape: {output.shape}", self.verbose)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, verbose=False):
        super(FeedForward, self).__init__()
        self.verbose = verbose
        
        verbose_print(f"Initializing FeedForward with d_model={d_model}, d_ff={d_ff}", self.verbose)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        verbose_print(f"FeedForward: Input shape: {x.shape}", self.verbose)
        
        # First linear layer
        ff1 = self.linear1(x)
        verbose_print(f"After first linear layer: {ff1.shape}", self.verbose)
        
        # ReLU activation
        relu_out = self.relu(ff1)
        verbose_print(f"After ReLU activation: {relu_out.shape}", self.verbose)
        
        # Second linear layer
        output = self.linear2(relu_out)
        verbose_print(f"FeedForward: Output shape: {output.shape}", self.verbose)
        
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, verbose=False):
        super(EncoderLayer, self).__init__()
        self.verbose = verbose
        
        verbose_print(f"Initializing EncoderLayer with d_model={d_model}, num_heads={num_heads}, d_ff={d_ff}, dropout={dropout}", self.verbose)
        self.self_attn = MultiHeadAttention(d_model, num_heads, verbose)
        self.feed_forward = FeedForward(d_model, d_ff, verbose)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        verbose_print(f"EncoderLayer: Processing input with shape {x.shape}", self.verbose)
        
        # Self-attention with residual connection and layer normalization
        verbose_print("EncoderLayer: Performing self-attention", self.verbose)
        attn_output = self.self_attn(x, x, x, mask)
        
        verbose_print("EncoderLayer: Adding residual connection and applying layer normalization", self.verbose)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer normalization
        verbose_print("EncoderLayer: Applying feed-forward network", self.verbose)
        ff_output = self.feed_forward(x)
        
        verbose_print("EncoderLayer: Adding second residual connection and applying layer normalization", self.verbose)
        x = self.norm2(x + self.dropout(ff_output))
        
        verbose_print(f"EncoderLayer: Output shape: {x.shape}", self.verbose)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, verbose=False):
        super(DecoderLayer, self).__init__()
        self.verbose = verbose
        
        verbose_print(f"Initializing DecoderLayer with d_model={d_model}, num_heads={num_heads}, d_ff={d_ff}, dropout={dropout}", self.verbose)
        self.self_attn = MultiHeadAttention(d_model, num_heads, verbose)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, verbose)
        self.feed_forward = FeedForward(d_model, d_ff, verbose)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        verbose_print(f"DecoderLayer: Processing input with shape {x.shape}", self.verbose)
        
        # Self-attention with residual connection and layer normalization
        verbose_print("DecoderLayer: Performing masked self-attention", self.verbose)
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        
        verbose_print("DecoderLayer: Adding residual connection and applying layer normalization", self.verbose)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention with residual connection and layer normalization
        verbose_print("DecoderLayer: Performing cross-attention with encoder output", self.verbose)
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        
        verbose_print("DecoderLayer: Adding residual connection and applying layer normalization", self.verbose)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward with residual connection and layer normalization
        verbose_print("DecoderLayer: Applying feed-forward network", self.verbose)
        ff_output = self.feed_forward(x)
        
        verbose_print("DecoderLayer: Adding final residual connection and applying layer normalization", self.verbose)
        x = self.norm3(x + self.dropout(ff_output))
        
        verbose_print(f"DecoderLayer: Output shape: {x.shape}", self.verbose)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1, verbose=False):
        super(Transformer, self).__init__()
        self.verbose = verbose
        
        verbose_print(f"""
        Initializing Transformer with:
        - Source vocabulary size: {src_vocab_size}
        - Target vocabulary size: {tgt_vocab_size}
        - Model dimension (d_model): {d_model}
        - Number of attention heads: {num_heads}
        - Number of encoder layers: {num_encoder_layers}
        - Number of decoder layers: {num_decoder_layers}
        - Feed-forward dimension: {d_ff}
        - Dropout rate: {dropout}
        """, self.verbose)
        
        # Embeddings and positional encoding
        verbose_print("Creating embedding layers for source and target", self.verbose)
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, verbose=verbose)
        
        # Encoder and decoder layers
        verbose_print(f"Creating {num_encoder_layers} encoder layers", self.verbose)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, verbose) 
            for _ in range(num_encoder_layers)
        ])
        
        verbose_print(f"Creating {num_decoder_layers} decoder layers", self.verbose)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, verbose) 
            for _ in range(num_decoder_layers)
        ])
        
        # Final linear layer for output prediction
        verbose_print("Creating output projection layer", self.verbose)
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
           Unmasked positions are filled with float(0.0).
        """
        verbose_print(f"Generating square subsequent mask of size {sz}x{sz}", self.verbose)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        verbose_print("Mask created to prevent looking at future tokens in the sequence", self.verbose)
        return mask
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        verbose_print(f"Transformer forward pass: src shape={src.shape}, tgt shape={tgt.shape}", self.verbose)
        
        # Embed source and target sequences and add positional encoding
        verbose_print("Converting input tokens to embeddings and adding positional encoding", self.verbose)
        src = self.dropout(self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model)))
        tgt = self.dropout(self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))
        
        verbose_print(f"After embedding and positional encoding: src shape={src.shape}, tgt shape={tgt.shape}", self.verbose)
        
        # Create masks if not provided
        if tgt_mask is None:
            verbose_print("No target mask provided, generating causal mask automatically", self.verbose)
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            
        # Pass through encoder layers
        verbose_print("Starting encoder processing", self.verbose)
        enc_output = src
        for i, enc_layer in enumerate(self.encoder_layers):
            verbose_print(f"Processing encoder layer {i+1}/{len(self.encoder_layers)}", self.verbose)
            enc_output = enc_layer(enc_output, src_mask)
        
        verbose_print(f"Encoder output shape: {enc_output.shape}", self.verbose)
            
        # Pass through decoder layers
        verbose_print("Starting decoder processing", self.verbose)
        dec_output = tgt
        for i, dec_layer in enumerate(self.decoder_layers):
            verbose_print(f"Processing decoder layer {i+1}/{len(self.decoder_layers)}", self.verbose)
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        
        verbose_print(f"Decoder output shape: {dec_output.shape}", self.verbose)
            
        # Final linear layer to get logits
        verbose_print("Applying final linear projection to get output logits", self.verbose)
        output = self.output_linear(dec_output)
        
        verbose_print(f"Final output shape: {output.shape}", self.verbose)
        return output

# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("TRANSFORMER MODEL DEMONSTRATION")
    print("=" * 80)
    print("This program demonstrates a simple transformer model implementation.")
    print("We'll create a model, feed it some random data, and show the output shapes.")
    print("=" * 80)
    
    # Enable verbose mode
    verbose = True
    
    # Define model parameters
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    
    print("\nSTEP 1: Creating the transformer model")
    print("-" * 50)
    # Create model
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        verbose=verbose
    )
    
    print("\nSTEP 2: Creating sample input data")
    print("-" * 50)
    print(f"Creating source tensor with batch size 2, sequence length 10")
    print(f"Each value is a token ID between 1 and {src_vocab_size-1}")
    # Create sample input
    src = torch.randint(1, src_vocab_size, (2, 10))  # Batch size 2, sequence length 10
    
    print(f"\nCreating target tensor with batch size 2, sequence length 8")
    print(f"Each value is a token ID between 1 and {tgt_vocab_size-1}")
    tgt = torch.randint(1, tgt_vocab_size, (2, 8))   # Batch size 2, sequence length 8
    
    print("\nSTEP 3: Performing forward pass through the model")
    print("-" * 50)
    print("This will run the input through all encoder and decoder layers")
    # Forward pass
    output = model(src, tgt)
    
    print("\nSTEP 4: Summary of shapes")
    print("-" * 50)
    print(f"Input shape: {src.shape} (batch_size, src_sequence_length)")
    print(f"Target shape: {tgt.shape} (batch_size, tgt_sequence_length)")
    print(f"Output shape: {output.shape} (batch_size, tgt_sequence_length, tgt_vocab_size)")
    print("\nThe output contains logits (scores) for each token in the vocabulary")
    print("for each position in the target sequence.")
    
    print("\nSTEP 5: What would happen next in a real application")
    print("-" * 50)
    print("1. Apply softmax to convert logits to probabilities")
    print("2. Select the token with highest probability for each position")
    print("3. Convert token IDs back to words/characters")
    print("4. Return the generated sequence as the model's output")
    
    print("\nEnd of demonstration")
    print("=" * 80)