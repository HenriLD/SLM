import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import os
import re
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model path
model_path = "HuggingFaceTB/SmolLM2-135M" # or HenriLD/smolrx-135M-PM05A15FW80 for continual pre-training
output_dir = "probability_lens_heatmap"
os.makedirs(output_dir, exist_ok=True)

class ProbabilityLensAnalyzer:
    def __init__(self, model_path):
        """Initialize with a model path."""
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.num_layers = self.model.config.num_hidden_layers
        
        # Register hooks to capture hidden states
        self.hidden_states = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture layer activations."""
        for name, module in self.model.named_modules():
            if 'mlp' in name:
                layer_num = int(name.split('.')[2]) if '.' in name else -1
                if layer_num >= 0:
                    module.register_forward_hook(
                        lambda m, i, o, layer=layer_num: self._save_hidden_state(layer, o)
                    )
    
    def _save_hidden_state(self, layer, output):
        """Save hidden states from a specific layer."""
        self.hidden_states[layer] = output.detach()
    
    def _clean_token_text(self, token_text):
        """Clean token text for better display."""
        # Remove any special characters or prefixes for display purposes
        cleaned = token_text.replace('Ġ', '')  # Remove byte-level BPE prefix
        cleaned = cleaned.replace('▁', '')  # Remove sentencepiece prefix
        cleaned = re.sub(r'^G', '', cleaned)  # Remove leading 'G' if present
        return cleaned
            
    def analyze_text(self, text, save_dir=None):
        """Analyze token representations across layers using probability instead of raw logits."""
        # Create save directory if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        token_ids = inputs['input_ids'][0]
        raw_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        # Clean token display for visualization
        tokens = [self._clean_token_text(t) for t in raw_tokens]
        
        # Forward pass with no gradient computation
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get the unembedding matrix (transpose of the embedding matrix)
        if hasattr(self.model, 'transformer'):
            unembedding = self.model.transformer.wte.weight
        else:
            unembedding = self.model.lm_head.weight
            
        # Collect token predictions at each layer
        layer_predictions = {}
        
        for layer in sorted(self.hidden_states.keys()):
            # Project hidden states back to vocabulary space
            hidden_states = self.hidden_states[layer]
            logits = torch.matmul(hidden_states, unembedding.transpose(0, 1))
            
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Get top tokens for each position
            top_k = 5
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
            
            # Store results for each token position
            position_data = []
            
            for pos in range(len(tokens)):
                # Make sure to detach tensors before converting to numpy
                indices = top_indices[0, pos].detach().cpu().numpy()
                probabilities = top_probs[0, pos].detach().cpu().numpy()
                
                # Convert indices to tokens and clean them
                top_tokens_raw = [self.tokenizer.decode(idx) for idx in indices]
                top_tokens_text = [t.strip() for t in top_tokens_raw]  # Clean whitespace
                
                position_data.append({
                    'position': pos,
                    'input_token': tokens[pos],
                    'top1_token': top_tokens_text[0],
                    'top1_prob': probabilities[0],
                    'top_tokens': top_tokens_text,
                    'top_probs': probabilities
                })
            
            layer_predictions[layer] = position_data
        
        # Create heatmap visualizations
        if save_dir:
            self._create_heatmap_visualizations(tokens, layer_predictions, save_dir)
        
        return {
            'tokens': tokens,
            'raw_tokens': raw_tokens,
            'layer_predictions': layer_predictions
        }
    
    def _create_heatmap_visualizations(self, tokens, layer_predictions, save_dir):
        """Create heatmap-style visualizations using probabilities."""
        # 1. Prepare data for token prediction heatmap
        layers = sorted(layer_predictions.keys())
        token_positions = range(len(tokens))
        
        # Create matrices for top predicted tokens and their probabilities
        token_matrix = np.zeros((len(layers), len(tokens)), dtype=object)
        prob_matrix = np.zeros((len(layers), len(tokens)))
        
        for i, layer in enumerate(layers):
            for j, pos in enumerate(token_positions):
                token_matrix[i, j] = layer_predictions[layer][pos]['top1_token']
                prob_matrix[i, j] = layer_predictions[layer][pos]['top1_prob']
        
        # 2. Create a better heatmap visualization with probability values
        fig, axes = plt.subplots(3, 1, figsize=(max(12, len(tokens) * 0.8), 20), 
                               gridspec_kw={'height_ratios': [4, 4, 2]})
        
        # Plot 1: Probability values heatmap
        # Use viridis colormap with normalization to make differences more visible
        min_prob, max_prob = np.min(prob_matrix), np.max(prob_matrix)
        # Adjust color range to make variations more visible
        # If the range is too small, artificially create more contrast
        if max_prob - min_prob < 0.05:
            vmin, vmax = min_prob, min_prob + 0.1  # Artificially increase range
        else:
            vmin, vmax = min_prob, max_prob
            
        im1 = axes[0].imshow(prob_matrix, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        axes[0].set_title('Token Prediction Probabilities Across Layers')
        axes[0].set_xlabel('Input Position')
        axes[0].set_ylabel('Layer')
        axes[0].set_xticks(range(len(tokens)))
        axes[0].set_xticklabels(tokens, rotation=90)
        axes[0].set_yticks(range(len(layers)))
        axes[0].set_yticklabels(layers)
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label('Probability')
        
        # Add probability values as text annotations
        for i in range(len(layers)):
            for j in range(len(tokens)):
                prob_value = prob_matrix[i, j]
                # Format probability with appropriate precision
                prob_text = f"{prob_value:.2f}"
                axes[0].text(j, i, prob_text, ha="center", va="center", 
                           color="white" if prob_value > (vmin + vmax)/2 else "black", 
                           fontsize=8)
        
        # Plot 2: Token predictions with probability heatmap
        im2 = axes[1].imshow(prob_matrix, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        for i in range(len(layers)):
            for j in range(len(tokens)):
                # Truncate token text for display if needed
                token_text = token_matrix[i, j]
                if len(token_text) > 6:
                    token_text = token_text[:6] + "..."
                
                axes[1].text(j, i, token_text, ha="center", va="center", 
                           color="white" if prob_matrix[i, j] > (vmin + vmax)/2 else "black", 
                           fontsize=8)
        axes[1].set_title('Top Token Predictions Across Layers')
        axes[1].set_xlabel('Input Position')
        axes[1].set_ylabel('Layer')
        axes[1].set_xticks(range(len(tokens)))
        axes[1].set_xticklabels(tokens, rotation=90)
        axes[1].set_yticks(range(len(layers)))
        axes[1].set_yticklabels(layers)
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_label('Probability')
        
        # Plot 3: Agreement with input token heatmap
        # Create agreement matrix
        agreement_matrix = np.zeros((len(layers), len(tokens)))
        for i, layer in enumerate(layers):
            for j, pos in enumerate(token_positions):
                input_token = tokens[j].lower()
                predictions = layer_predictions[layer][j]['top_tokens']
                probabilities = layer_predictions[layer][j]['top_probs']
                
                # Calculate agreement score
                agreement_score = 0
                for k, (pred, prob) in enumerate(zip(predictions, probabilities)):
                    if input_token in pred.lower():
                        agreement_score = prob  # Use actual probability value
                        break
                
                agreement_matrix[i, j] = agreement_score
        
        # Use a clear colormap for agreement
        cmap = LinearSegmentedColormap.from_list('agreement', ['white', 'red'], N=256)
        im3 = axes[2].imshow(agreement_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=max(0.5, np.max(agreement_matrix)))
        axes[2].set_title('Agreement with Input Token (Probability of Original Token)')
        axes[2].set_xlabel('Input Position')
        axes[2].set_ylabel('Layer')
        axes[2].set_xticks(range(len(tokens)))
        axes[2].set_xticklabels(tokens, rotation=90)
        axes[2].set_yticks(range(len(layers)))
        axes[2].set_yticklabels(layers)
        cbar3 = plt.colorbar(im3, ax=axes[2])
        cbar3.set_label('Probability of Original Token')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/probability_lens_detailed_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a compact single heatmap with tokens and probabilities
        plt.figure(figsize=(max(12, len(tokens) * 1.0), max(10, len(layers) * 0.4)))
        
        # Use a custom colormap to maximize visibility of probability differences
        custom_cmap = LinearSegmentedColormap.from_list(
            'custom_viridis', 
            plt.cm.viridis(np.linspace(0.2, 1, 256))  # Skip the darkest part of viridis
        )
        
        im = plt.imshow(prob_matrix, cmap=custom_cmap, aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(im, label='Probability')
        
        # Add token and probability annotations
        for i in range(len(layers)):
            for j in range(len(tokens)):
                token_text = token_matrix[i, j]
                if len(token_text) > 6:
                    token_text = token_text[:6]
                
                prob_value = prob_matrix[i, j]
                text = f"{token_text}\n{prob_value:.2f}"
                
                plt.text(j, i, text, ha="center", va="center", 
                       color="white" if prob_value > (vmin + vmax)/2 else "black", 
                       fontsize=8)
        
        plt.title('Token Predictions with Probabilities Across Layers')
        plt.xlabel('Input Position')
        plt.ylabel('Layer')
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.yticks(range(len(layers)), layers)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/probability_lens_compact_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    # Sample texts for analysis
    texts = [
        "The patient presented with acute myocardial infarction and was administered aspirin.",
        "Recurrent neural networks have been largely replaced by transformer architectures.",
        "The BRCA1 gene mutation is associated with increased risk of breast cancer."
    ]
    
    # Create analyzer
    print("Creating Probability Lens analyzer...")
    analyzer = ProbabilityLensAnalyzer(model_path)
    
    # Analyze each text
    for i, text in enumerate(texts):
        text_id = f"text_{i+1}"
        print(f"\nAnalyzing {text_id}: {text}")
        
        # Run analysis
        print("Running probability lens analysis...")
        save_dir = os.path.join(output_dir, text_id)
        results = analyzer.analyze_text(text, save_dir=save_dir)
    
    print("\nAnalysis complete! Check the output directories for probability heatmap visualizations.")

if __name__ == "__main__":
    main()