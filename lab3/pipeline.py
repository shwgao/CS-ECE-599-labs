import torch
import torch.nn as nn
from torch.distributed.pipelining import ScheduleGPipe
from torch.distributed.pipelining import PipelineStage
import os

# Class to hold model configuration parameters
class ModelArgs:
    def __init__(self):
        self.n_layers = 12  # Number of transformer layers
        self.hidden_size = 768  # Size of the hidden layer
        self.num_heads = 12  # Number of attention heads
        self.vocab_size = 32000  # Vocabulary size for embeddings
        self.max_seq_length = 512  # Maximum sequence length

# Transformer model definition
class Transformer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        
        # Token embedding layer
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.hidden_size)
        # Layer normalization
        self.norm = nn.LayerNorm(model_args.hidden_size)
        
        # Dictionary to hold transformer layers
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(model_args)
        
        # Output layer to map hidden states to vocabulary size
        self.output = nn.Linear(model_args.hidden_size, model_args.vocab_size)
        
        # Precompute frequency table for rotary embeddings
        self.freqs_cis = self._precompute_freqs_cis(model_args.hidden_size // model_args.num_heads, model_args.max_seq_length)

    def _precompute_freqs_cis(self, head_dim, seq_length):
        # Precompute frequency table for rotary embeddings
        freqs = torch.tensor(
            [1.0 / (10000.0 ** (i / head_dim)) for i in range(0, head_dim, 2)]
        )
        t = torch.arange(seq_length, dtype=torch.float)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def forward(self, tokens: torch.Tensor):
        # Embed tokens if embedding layer is present
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        # Pass through each transformer layer
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)

        # Apply layer normalization if present
        h = self.norm(h) if self.norm else h
        # Map to output vocabulary size
        output = self.output(h).float() if self.output else h
        return output

# Transformer block definition
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            args.hidden_size, 
            args.num_heads,
            batch_first=True
        )
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(args.hidden_size, 4 * args.hidden_size),
            nn.ReLU(),
            nn.Linear(4 * args.hidden_size, args.hidden_size)
        )
        # Layer normalization layers
        self.norm1 = nn.LayerNorm(args.hidden_size)
        self.norm2 = nn.LayerNorm(args.hidden_size)

    def forward(self, x, freqs_cis):
        # Ensure input has correct shape
        if x.size(-1) != self.attention.embed_dim:
            raise ValueError(f"Expected input dimension {self.attention.embed_dim}, but got {x.size(-1)}")
        
        # Apply multi-head attention
        attn_output, _ = self.attention(x, x, x)
        # Add & normalize
        x = self.norm1(x + attn_output)
        # Apply feed-forward network
        ff_output = self.feed_forward(x)
        # Add & normalize
        x = self.norm2(x + ff_output)
        return x

def main():
    # Initialize distributed environment
    torch.distributed.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    
    # Set device for the current process
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    
    # Pipeline configuration
    num_stages = 2  # Number of pipeline stages
    n_microbatches = 4  # Number of microbatches
    batch_size = 32  # Batch size
    seq_length = 512  # Sequence length
    num_iterations = 5  # Number of iterations to run the pipeline
    
    # Model configuration
    model_args = ModelArgs()
    
    # Example input for the pipeline stage
    example_input_microbatch = (
        torch.zeros(batch_size // n_microbatches, seq_length, dtype=torch.long),
    )
    
    # Determine the stage index for the current rank
    stage_index = rank % num_stages
    
    # Initialize model on 'meta' device for memory efficiency
    with torch.device('meta'):
        model = Transformer(model_args)
        
        if stage_index == 0:
            # First stage keeps embedding and first half of layers
            for layer_id in range(model_args.n_layers // 2, model_args.n_layers):
                del model.layers[str(layer_id)]
            model.norm = None
            model.output = None
            print(f"Stage {stage_index}: Initialized with embedding and first half of layers.")
        
        elif stage_index == 1:
            # Second stage keeps second half of layers and output
            model.tok_embeddings = None
            for layer_id in range(0, model_args.n_layers // 2):
                del model.layers[str(layer_id)]
            # Ensure the input to this stage has the correct dimension
            example_input_microbatch = (
                torch.zeros(batch_size // n_microbatches, seq_length, model_args.hidden_size, dtype=torch.float),
            )
            print(f"Stage {stage_index}: Initialized with second half of layers and output.")
    
        # Move model to the specified device
        model = model.to_empty(device=device)
        
        # Create a pipeline stage
        stage = PipelineStage(
            model,
            stage_index,
            num_stages,
            device,
            input_args=example_input_microbatch,
        )
    
    # Create schedule for pipeline execution
    schedule = ScheduleGPipe(stage, n_microbatches)
    
    # Run the pipeline for a specified number of iterations
    for iteration in range(num_iterations):
        print(f"Rank {rank}: Starting iteration {iteration + 1}")
        
        # Generate sample input data
        if rank == 0:
            x = torch.randint(0, model_args.vocab_size, (batch_size, seq_length), device=device)
            print(f"Rank {rank}: Starting pipeline with input data.")
            output = schedule.step(x)
            print(output)
            print(f"Rank {rank}: Completed pipeline step with output.")
        else:
            print(f"Rank {rank}: Waiting for input data.")
            output = schedule.step()
            print(output.shape)
            print(f"Rank {rank}: Completed pipeline step.")
        
        torch.distributed.barrier()
        print(f"Rank {rank}: Finished iteration {iteration + 1}")

if __name__ == "__main__":
    main()