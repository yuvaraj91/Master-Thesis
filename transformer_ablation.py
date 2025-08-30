import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import math
from dataclasses import dataclass
from sklearn.metrics import bleu_score
import json

@dataclass
class ExperimentConfig:
    """Configuration for ablation experiments"""
    model_dim: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dropout: float = 0.1
    max_seq_len: int = 512
    vocab_size: int = 32000
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert model_dim % num_heads == 0
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        self.q_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        self.out_linear = nn.Linear(model_dim, model_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and reshape for multi-head attention
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and apply final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)
        output = self.out_linear(context)
        
        return output, attention_weights

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_seq_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_len, model_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() *
                           -(math.log(10000.0) / model_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class FeedForward(nn.Module):
    def __init__(self, model_dim: int, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(model_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = FeedForward(model_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_out, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection and layer norm
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = FeedForward(model_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        # Self-attention
        self_attn_out, _ = self.self_attention(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        
        # Cross-attention
        cross_attn_out, _ = self.cross_attention(x, encoder_output, encoder_output, cross_attn_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x

class ModifiedTransformer(nn.Module):
    """Modified Transformer for ablation study"""
    def __init__(self, config: ExperimentConfig, experiment_type: str = "baseline"):
        super().__init__()
        self.config = config
        self.experiment_type = experiment_type
        
        # Embedding layers
        self.src_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.tgt_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.pos_encoding = PositionalEncoding(config.model_dim, config.max_seq_len)
        
        # Encoder layers (can be amplified)
        encoder_layers = config.num_encoder_layers
        if experiment_type == "amplified_encoder":
            encoder_layers = config.num_encoder_layers * 2  # Double encoder layers
            
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config.model_dim, config.num_heads, config.dropout)
            for _ in range(encoder_layers)
        ])
        
        # Decoder layers (can be pruned or compressed)
        decoder_layers = self._get_decoder_layers()
        self.decoder_layers = nn.ModuleList(decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(config.model_dim, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
        
        self._init_weights()
    
    def _get_decoder_layers(self) -> List[nn.Module]:
        """Get decoder layers based on experiment type"""
        if self.experiment_type == "pruned_decoder_1":
            return [DecoderLayer(self.config.model_dim, self.config.num_heads, self.config.dropout)]
        elif self.experiment_type == "pruned_decoder_2":
            return [DecoderLayer(self.config.model_dim, self.config.num_heads, self.config.dropout) 
                   for _ in range(2)]
        elif self.experiment_type == "pruned_decoder_3":
            return [DecoderLayer(self.config.model_dim, self.config.num_heads, self.config.dropout) 
                   for _ in range(3)]
        elif self.experiment_type == "compressed_decoder":
            # Use fewer attention heads for compression
            compressed_heads = max(1, self.config.num_heads // 2)
            return [DecoderLayer(self.config.model_dim, compressed_heads, self.config.dropout) 
                   for _ in range(self.config.num_decoder_layers)]
        else:  # baseline or amplified_encoder
            return [DecoderLayer(self.config.model_dim, self.config.num_heads, self.config.dropout) 
                   for _ in range(self.config.num_decoder_layers)]
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.1)
    
    def create_masks(self, src, tgt):
        """Create attention masks"""
        src_mask = (src != 0).unsqueeze(1).unsqueeze(1)  # Padding mask
        
        tgt_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool().to(tgt.device)
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(1)
        tgt_mask = tgt_mask & tgt_padding_mask
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.create_masks(src, tgt)
        
        # Encoder
        src_emb = self.dropout(self.pos_encoding(self.src_embedding(src)))
        encoder_output = src_emb
        
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
        
        # Decoder
        tgt_emb = self.dropout(self.pos_encoding(self.tgt_embedding(tgt)))
        decoder_output = tgt_emb
        
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, tgt_mask, src_mask)
        
        # Output projection
        output = self.output_projection(decoder_output)
        return output

class AblationStudyExperiment:
    """Main experiment class for running ablation studies"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
    
    def create_dummy_dataset(self, num_samples: int = 10000):
        """Create dummy dataset for demonstration"""
        src_data = torch.randint(1, self.config.vocab_size, (num_samples, self.config.max_seq_len // 2))
        tgt_data = torch.randint(1, self.config.vocab_size, (num_samples, self.config.max_seq_len // 2))
        
        dataset = torch.utils.data.TensorDataset(src_data, tgt_data)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
    
    def train_model(self, model, train_loader, experiment_name: str):
        """Train a model and return training metrics"""
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        model.train()
        training_times = []
        losses = []
        
        print(f"Training {experiment_name}...")
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0
            
            for batch_idx, (src, tgt) in enumerate(train_loader):
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                # Create target input (shift right) and target output
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                optimizer.zero_grad()
                
                # Forward pass
                output = model(src, tgt_input)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}/{self.config.num_epochs}, "
                          f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            epoch_time = time.time() - epoch_start_time
            training_times.append(epoch_time)
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")
        
        return {
            'training_times': training_times,
            'losses': losses,
            'total_training_time': sum(training_times)
        }
    
    def evaluate_model(self, model, test_loader, experiment_name: str):
        """Evaluate model and compute BLEU scores and inference time"""
        model.eval()
        model.to(self.device)
        
        total_inference_time = 0
        predictions = []
        references = []
        
        print(f"Evaluating {experiment_name}...")
        
        with torch.no_grad():
            for src, tgt in test_loader:
                src = src.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                
                # Simple greedy decoding for evaluation
                batch_size = src.size(0)
                max_len = tgt.size(1)
                
                # Start with start token (assuming token 1 is start token)
                decoder_input = torch.ones(batch_size, 1, dtype=torch.long).to(self.device)
                
                for _ in range(max_len - 1):
                    output = model(src, decoder_input)
                    next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    decoder_input = torch.cat([decoder_input, next_token], dim=1)
                
                inference_time = time.time() - start_time
                total_inference_time += inference_time
                
                # Convert to lists for BLEU calculation (simplified)
                pred_tokens = decoder_input[:, 1:].cpu().numpy()  # Remove start token
                ref_tokens = tgt.cpu().numpy()
                
                predictions.extend(pred_tokens.tolist())
                references.extend(ref_tokens.tolist())
        
        # Calculate simplified BLEU score (for demonstration)
        # In practice, you would use proper tokenization and BLEU calculation
        avg_inference_time = total_inference_time / len(test_loader)
        
        # Mock BLEU score calculation (replace with actual BLEU computation)
        mock_bleu = np.random.uniform(0.2, 0.4)  # Placeholder
        
        return {
            'bleu_score': mock_bleu,
            'avg_inference_time': avg_inference_time,
            'total_inference_time': total_inference_time
        }
    
    def run_experiment(self, experiment_type: str):
        """Run a single experiment"""
        print(f"\n{'='*50}")
        print(f"Running experiment: {experiment_type}")
        print(f"{'='*50}")
        
        # Create model
        model = ModifiedTransformer(self.config, experiment_type)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
        
        # Create datasets
        train_loader = self.create_dummy_dataset(8000)
        test_loader = self.create_dummy_dataset(2000)
        
        # Train model
        train_metrics = self.train_model(model, train_loader, experiment_type)
        
        # Evaluate model
        eval_metrics = self.evaluate_model(model, test_loader, experiment_type)
        
        # Combine results
        results = {
            'num_parameters': num_params,
            'experiment_type': experiment_type,
            **train_metrics,
            **eval_metrics
        }
        
        self.results[experiment_type] = results
        return results
    
    def run_all_experiments(self):
        """Run all ablation study experiments"""
        experiment_types = [
            "baseline",
            "pruned_decoder_1",
            "pruned_decoder_2", 
            "pruned_decoder_3",
            "amplified_encoder",
            "compressed_decoder"
        ]
        
        for exp_type in experiment_types:
            try:
                self.run_experiment(exp_type)
            except Exception as e:
                print(f"Error in experiment {exp_type}: {str(e)}")
        
        return self.results
    
    def analyze_results(self):
        """Analyze and visualize results"""
        if not self.results:
            print("No results to analyze. Run experiments first.")
            return
        
        print(f"\n{'='*60}")
        print("ABLATION STUDY RESULTS SUMMARY")
        print(f"{'='*60}")
        
        # Create summary table
        print(f"{'Experiment':<20} {'BLEU':<8} {'Params':<10} {'Train Time':<12} {'Inference':<12}")
        print(f"{'-'*65}")
        
        baseline_bleu = self.results.get('baseline', {}).get('bleu_score', 0)
        
        for exp_name, results in self.results.items():
            bleu = results['bleu_score']
            params = results['num_parameters']
            train_time = results['total_training_time']
            inference_time = results['avg_inference_time']
            
            # Calculate relative performance
            bleu_diff = bleu - baseline_bleu if baseline_bleu > 0 else 0
            bleu_str = f"{bleu:.3f}({bleu_diff:+.3f})" if exp_name != 'baseline' else f"{bleu:.3f}"
            
            print(f"{exp_name:<20} {bleu_str:<8} {params:<10,} {train_time:<12.1f}s {inference_time:<12.4f}s")
        
        # Plot results
        self.plot_results()
    
    def plot_results(self):
        """Create visualization plots"""
        if not self.results:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        exp_names = list(self.results.keys())
        
        # BLEU scores
        bleu_scores = [self.results[exp]['bleu_score'] for exp in exp_names]
        ax1.bar(exp_names, bleu_scores, color='skyblue')
        ax1.set_title('BLEU Scores by Experiment')
        ax1.set_ylabel('BLEU Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Training times
        train_times = [self.results[exp]['total_training_time'] for exp in exp_names]
        ax2.bar(exp_names, train_times, color='lightcoral')
        ax2.set_title('Training Times by Experiment')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Inference times
        inference_times = [self.results[exp]['avg_inference_time'] for exp in exp_names]
        ax3.bar(exp_names, inference_times, color='lightgreen')
        ax3.set_title('Average Inference Times by Experiment')
        ax3.set_ylabel('Inference Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Parameter counts
        param_counts = [self.results[exp]['num_parameters'] for exp in exp_names]
        ax4.bar(exp_names, param_counts, color='gold')
        ax4.set_title('Model Parameters by Experiment')
        ax4.set_ylabel('Number of Parameters')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('ablation_study_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, filename: str = 'ablation_results.json'):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")

def main():
    """Main function to run the ablation study"""
    # Configuration
    config = ExperimentConfig(
        model_dim=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout=0.1,
        max_seq_len=256,  # Reduced for faster training
        vocab_size=10000,  # Reduced for demonstration
        batch_size=16,    # Reduced for memory efficiency
        learning_rate=1e-4,
        num_epochs=3      # Reduced for demonstration
    )
    
    print("Starting Transformer Decoder Ablation Study")
    print(f"Configuration: {config}")
    
    # Run experiments
    experiment = AblationStudyExperiment(config)
    results = experiment.run_all_experiments()
    
    # Analyze results
    experiment.analyze_results()
    
    # Save results
    experiment.save_results()
    

if __name__ == "__main__":
    main()