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
from typing import List, Dict, Tuple, Optional, Union

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model path
model_path = "HuggingFaceTB/SmolLM2-135M" #HenriLD/smolrx-135M-PM05A15FW80 for continual pre-training
output_dir = "attention_lens_visualizations"
os.makedirs(output_dir, exist_ok=True)

class AttentionLensAnalyzer:
    """
    Implementation of the Attention Lens technique for analyzing attention patterns
    in transformer-based language models as mentioned in the project proposal.
    """
    def __init__(self, model_path: str):
        """Initialize with a model path."""
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Get model configuration
        self.config = self.model.config
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        
        # Register hooks to capture attention patterns
        self.attention_maps = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention matrices for each layer and head."""
        for name, module in self.model.named_modules():
            # Look for attention modules in the model
            if 'attn' in name and 'attn.q_proj' not in name and 'attn.k_proj' not in name and 'attn.v_proj' not in name:
                # Extract layer number from the module name
                layer_match = re.search(r'h\.(\d+)\.', name)
                if layer_match:
                    layer_num = int(layer_match.group(1))
                    # Register a forward hook to capture attention scores
                    module.register_forward_hook(
                        lambda m, i, o, layer=layer_num: self._save_attention_map(layer, o)
                    )
    
    def _save_attention_map(self, layer: int, output):
        """Save attention maps from a specific layer."""
        # Different models may have different output structures for attention
        if isinstance(output, tuple):
            # Some models return attention weights as part of a tuple
            if len(output) > 1 and hasattr(output[1], 'get'):
                attn_weights = output[1].get('attention_scores')
                if attn_weights is not None:
                    self.attention_maps[layer] = attn_weights.detach()
            # If direct access doesn't work, try accessing the attention weights from the output directly
            elif len(output) > 1 and len(output) >= 3:  # Some models return attention as third element
                self.attention_maps[layer] = output[2].detach()
        else:
            # For models where the output is the attention weights directly
            self.attention_maps[layer] = output.detach()
    
    def _clean_token_text(self, token_text: str) -> str:
        """Clean token text for better display."""
        # Remove special characters or prefixes for display purposes
        cleaned = token_text.replace('Ġ', '')  # Remove byte-level BPE prefix
        cleaned = cleaned.replace('▁', '')  # Remove sentencepiece prefix
        cleaned = re.sub(r'^G', '', cleaned)  # Remove leading 'G' if present
        return cleaned

    def analyze_text(self, text: str, save_dir: Optional[str] = None) -> Dict:
        """
        Analyze attention patterns for input text across all layers and heads.
        
        Args:
            text: Input text to analyze
            save_dir: Directory to save visualizations (optional)
            
        Returns:
            Dictionary containing analysis results
        """
        # Create save directory if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        token_ids = inputs['input_ids'][0]
        raw_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        # Clean token display for visualization
        tokens = [self._clean_token_text(t) for t in raw_tokens]
        
        # Clear previous attention maps
        self.attention_maps = {}
        
        # Forward pass with output_attentions=True to capture attention weights
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Access attention weights if they weren't captured by hooks
        if not self.attention_maps and hasattr(outputs, 'attentions') and outputs.attentions is not None:
            # Some models provide attentions directly in the output
            for layer_idx, layer_attention in enumerate(outputs.attentions):
                self.attention_maps[layer_idx] = layer_attention.detach()
        
        # Analyze attention patterns if available
        if self.attention_maps:
            # Create visualizations if save_dir is provided
            if save_dir:
                self._create_attention_visualizations(tokens, self.attention_maps, save_dir)
            
            return {
                'tokens': tokens,
                'raw_tokens': raw_tokens,
                'attention_maps': self.attention_maps
            }
        else:
            print("Attention weights not captured. The model may not expose attention weights.")
            return {
                'tokens': tokens,
                'raw_tokens': raw_tokens,
                'attention_maps': {}
            }

    def _create_attention_visualizations(self, tokens: List[str], attention_maps: Dict, save_dir: str):
        """
        Create visualizations for attention patterns.
        
        Args:
            tokens: List of tokens
            attention_maps: Dictionary of attention weights by layer
            save_dir: Directory to save visualizations
        """
        # Create directory for visualizations if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # For each layer where attention was captured
        for layer_idx, attention in attention_maps.items():
            # Create a figure for this layer with subplots for each attention head
            if attention.dim() == 4:  # shape: [batch, heads, seq_len, seq_len]
                num_heads = attention.size(1)
                
                # Create grid of subplots - determine rows and columns
                num_rows = int(np.ceil(np.sqrt(num_heads)))
                num_cols = int(np.ceil(num_heads / num_rows))
                
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3))
                
                # Flatten axes if multiple rows and columns
                if num_rows > 1 or num_cols > 1:
                    axes = axes.flatten()
                else:
                    axes = [axes]  # Make it iterable if there's just one subplot
                
                # For each attention head
                for head_idx in range(num_heads):
                    if head_idx < len(axes):  # Make sure we have enough subplots
                        ax = axes[head_idx]
                        
                        # Get attention matrix for this head (first batch item)
                        attn_matrix = attention[0, head_idx].cpu().numpy()
                        
                        # Create heatmap
                        sns.heatmap(
                            attn_matrix, 
                            ax=ax,
                            cmap="viridis",
                            xticklabels=tokens,
                            yticklabels=tokens,
                            vmin=0.0, 
                            vmax=np.max(attn_matrix)
                        )
                        
                        # Set title and labels
                        ax.set_title(f"Head {head_idx}")
                        ax.set_xlabel("Attended to")
                        ax.set_ylabel("Attended from")
                        
                        # Rotate tick labels for better readability
                        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                        plt.setp(ax.get_yticklabels(), rotation=0)
                
                # Hide any unused subplots
                for head_idx in range(num_heads, len(axes)):
                    axes[head_idx].axis('off')
                
                # Add overall title
                plt.suptitle(f"Layer {layer_idx} Attention Patterns", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
                
                # Save figure
                plt.savefig(f"{save_dir}/layer_{layer_idx}_attention_heads.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                # Also create an aggregated attention visualization for this layer
                plt.figure(figsize=(10, 8))
                
                # Average attention across all heads
                avg_attention = attention[0].mean(dim=0).cpu().numpy()
                
                # Create heatmap of average attention
                sns.heatmap(
                    avg_attention,
                    cmap="viridis",
                    xticklabels=tokens,
                    yticklabels=tokens,
                    vmin=0.0,
                    vmax=np.max(avg_attention)
                )
                
                plt.title(f"Layer {layer_idx} - Average Attention Across All Heads")
                plt.xlabel("Attended to")
                plt.ylabel("Attended from")
                
                # Rotate tick labels for better readability
                plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
                plt.yticks(rotation=0)
                
                plt.tight_layout()
                plt.savefig(f"{save_dir}/layer_{layer_idx}_average_attention.png", dpi=300, bbox_inches='tight')
                plt.close()
            else:
                print(f"Unexpected attention tensor shape: {attention.shape} for layer {layer_idx}")
        
        # Create a summary visualization showing the dominant attention pattern for key tokens
        self._create_attention_summary(tokens, attention_maps, save_dir)
    
    def _create_attention_summary(self, tokens: List[str], attention_maps: Dict, save_dir: str):
        """
        Create a summary visualization showing how attention evolves across layers.
        
        Args:
            tokens: List of tokens
            attention_maps: Dictionary of attention weights by layer
            save_dir: Directory to save visualizations
        """
        # Only create this visualization if we have attention maps from multiple layers
        if len(attention_maps) <= 1:
            return
        
        # Select a subset of tokens to analyze if the sequence is too long
        max_tokens_to_display = 10
        token_indices = list(range(min(len(tokens), max_tokens_to_display)))
        
        # Prepare data structure for plotting
        layers = sorted(attention_maps.keys())
        token_attention_evolution = {idx: [] for idx in token_indices}
        
        # For each layer, calculate the attention focus for selected tokens
        for layer_idx in layers:
            attention = attention_maps[layer_idx]
            
            # Calculate average attention across heads for this layer
            if attention.dim() == 4:  # [batch, heads, seq_len, seq_len]
                avg_attn = attention[0].mean(dim=0).cpu().numpy()  # Average across heads
                
                # For each token we're tracking
                for token_idx in token_indices:
                    if token_idx < avg_attn.shape[0]:
                        # Get which token this token attends to the most
                        max_attended_idx = np.argmax(avg_attn[token_idx])
                        token_attention_evolution[token_idx].append(max_attended_idx)
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        
        # For each token we're tracking
        for token_idx in token_indices:
            if token_idx < len(tokens) and token_attention_evolution[token_idx]:
                # Plot how this token's attention focus changes across layers
                plt.plot(
                    layers, 
                    token_attention_evolution[token_idx],
                    marker='o',
                    linestyle='-',
                    label=f"Token: {tokens[token_idx]}"
                )
        
        plt.title("Evolution of Maximum Attention Focus Across Layers")
        plt.xlabel("Layer")
        plt.ylabel("Token Index with Maximum Attention")
        
        # Add a secondary y-axis with token labels
        ax = plt.gca()
        sec_ax = ax.secondary_yaxis('right')
        
        # Set tick positions and labels for the secondary axis
        token_positions = list(range(len(tokens)))
        sec_ax.set_yticks(token_positions)
        sec_ax.set_yticklabels([tokens[i] if i < len(tokens) else "" for i in token_positions])
        
        plt.xticks(layers)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/attention_evolution_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create attention flow diagram
        self._create_attention_flow_diagram(tokens, attention_maps, save_dir)
    
    def _create_attention_flow_diagram(self, tokens: List[str], attention_maps: Dict, save_dir: str):
        """
        Create a diagram showing the flow of attention through tokens across layers.
        
        Args:
            tokens: List of tokens
            attention_maps: Dictionary of attention weights by layer
            save_dir: Directory to save visualizations
        """
        # Only create if we have multiple layers
        if len(attention_maps) <= 1:
            return
        
        # Limit the number of tokens to display
        max_tokens = min(len(tokens), 20)  # Limit to 20 tokens for readability
        tokens_to_display = tokens[:max_tokens]
        
        # Prepare the figure
        fig_height = max(10, max_tokens * 0.5)  # Scale figure height based on number of tokens
        plt.figure(figsize=(15, fig_height))
        
        # Get sorted layers
        layers = sorted(attention_maps.keys())
        
        # For each layer, we'll create a column in our visualization
        for layer_idx, layer in enumerate(layers):
            attention = attention_maps[layer]
            
            # Calculate mean attention across heads
            if attention.dim() == 4:  # [batch, heads, seq_len, seq_len]
                avg_attn = attention[0].mean(dim=0).cpu().numpy()[:max_tokens, :max_tokens]
                
                # Plot attention weights as heatmap for this layer
                plt.subplot(1, len(layers), layer_idx + 1)
                sns.heatmap(
                    avg_attn,
                    cmap="Blues",
                    xticklabels=tokens_to_display,
                    yticklabels=tokens_to_display,
                    vmin=0.0,
                    vmax=np.max(avg_attn)
                )
                
                plt.title(f"Layer {layer}")
                
                # Rotate tick labels
                plt.xticks(rotation=90)
                plt.yticks(rotation=0)
                
                if layer_idx == 0:
                    plt.ylabel("From token")
                
                plt.xlabel("To token")
        
        plt.suptitle("Attention Flow Across Layers", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        plt.savefig(f"{save_dir}/attention_flow_across_layers.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a more detailed token-wise attention flow visualization
        self._create_token_attention_flow(tokens, attention_maps, save_dir)
    
    def _create_token_attention_flow(self, tokens: List[str], attention_maps: Dict, save_dir: str):
        """
        Create visualizations for how specific tokens attend to others across layers.
        
        Args:
            tokens: List of tokens
            attention_maps: Dictionary of attention weights by layer
            save_dir: Directory to save visualizations
        """
        # Choose a subset of interesting tokens to track (e.g., first 5 tokens)
        tokens_to_track = min(5, len(tokens))
        
        # For each token we're tracking
        for token_idx in range(tokens_to_track):
            if token_idx >= len(tokens):
                continue
                
            # Create a figure showing how this token attends to others across layers
            plt.figure(figsize=(14, 8))
            
            # Get sorted layers
            layers = sorted(attention_maps.keys())
            
            # Create a matrix to hold attention values across layers
            attn_values = np.zeros((len(layers), min(len(tokens), 20)))
            
            # Populate the matrix
            for i, layer in enumerate(layers):
                attention = attention_maps[layer]
                
                if attention.dim() == 4:  # [batch, heads, seq_len, seq_len]
                    # Average across heads for this layer
                    avg_attn = attention[0].mean(dim=0).cpu().numpy()
                    
                    # Get attention pattern for our target token
                    if token_idx < avg_attn.shape[0]:
                        attn_pattern = avg_attn[token_idx, :min(len(tokens), 20)]
                        attn_values[i] = attn_pattern
            
            # Create the heatmap
            sns.heatmap(
                attn_values,
                cmap="YlOrRd",
                xticklabels=tokens[:min(len(tokens), 20)],
                yticklabels=layers,
                vmin=0.0,
                vmax=np.max(attn_values)
            )
            
            plt.title(f"How token '{tokens[token_idx]}' attends to other tokens across layers")
            plt.xlabel("Attended to")
            plt.ylabel("Layer")
            
            # Rotate x-tick labels for better readability
            plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/token_{token_idx}_attention_flow.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def analyze_model_comparison(self, text: str, other_model_path: str, save_dir: Optional[str] = None) -> Dict:
        """
        Compare attention patterns between this model and another model.
        Useful for analyzing how fine-tuning affects attention mechanisms.
        
        Args:
            text: Input text to analyze
            other_model_path: Path to the other model for comparison
            save_dir: Directory to save visualizations
            
        Returns:
            Dictionary with comparison results
        """
        # Create save directory if specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        print(f"Comparing attention patterns with model: {other_model_path}")
        
        # Create an analyzer for the other model
        other_analyzer = AttentionLensAnalyzer(other_model_path)
        
        # Analyze the same text with both models
        base_results = self.analyze_text(text)
        other_results = other_analyzer.analyze_text(text)
        
        # Compare the attention patterns if available
        if base_results['attention_maps'] and other_results['attention_maps']:
            if save_dir:
                self._create_comparison_visualizations(
                    base_results['tokens'],
                    base_results['attention_maps'],
                    other_results['attention_maps'],
                    save_dir
                )
            
            return {
                'tokens': base_results['tokens'],
                'base_attention': base_results['attention_maps'],
                'other_attention': other_results['attention_maps']
            }
        else:
            print("Could not compare attention patterns. Attention maps not available for one or both models.")
            return {
                'tokens': base_results['tokens'],
                'base_attention': base_results['attention_maps'],
                'other_attention': other_results['attention_maps']
            }
    
    def _create_comparison_visualizations(self, tokens: List[str], base_attn: Dict, other_attn: Dict, save_dir: str):
        """
        Create visualizations comparing attention patterns between two models.
        
        Args:
            tokens: List of tokens
            base_attn: Attention maps from the base model
            other_attn: Attention maps from the other model (e.g., fine-tuned)
            save_dir: Directory to save visualizations
        """
        # Find common layers between the two models
        common_layers = sorted(set(base_attn.keys()) & set(other_attn.keys()))
        
        if not common_layers:
            print("No common layers found for comparison.")
            return
        
        # For each common layer
        for layer in common_layers:
            # Get attention for this layer from both models
            base_attention = base_attn[layer]
            other_attention = other_attn[layer]
            
            # Create figures to compare average attention across heads
            if base_attention.dim() == 4 and other_attention.dim() == 4:
                # Average across heads
                base_avg = base_attention[0].mean(dim=0).cpu().numpy()
                other_avg = other_attention[0].mean(dim=0).cpu().numpy()
                
                # Create comparison figure
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                
                # Determine max value for consistent color scaling
                vmax = max(np.max(base_avg), np.max(other_avg))
                
                # Plot base model attention
                sns.heatmap(
                    base_avg,
                    ax=ax1,
                    cmap="viridis",
                    xticklabels=tokens,
                    yticklabels=tokens,
                    vmin=0.0,
                    vmax=vmax
                )
                ax1.set_title("Base Model")
                ax1.set_xlabel("Attended to")
                ax1.set_ylabel("Attended from")
                plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                # Plot other model attention
                sns.heatmap(
                    other_avg,
                    ax=ax2,
                    cmap="viridis",
                    xticklabels=tokens,
                    yticklabels=tokens,
                    vmin=0.0,
                    vmax=vmax
                )
                ax2.set_title("Other Model")
                ax2.set_xlabel("Attended to")
                plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                # Plot difference
                difference = other_avg - base_avg
                
                # Use a divergent colormap for the difference
                sns.heatmap(
                    difference,
                    ax=ax3,
                    cmap="coolwarm",
                    xticklabels=tokens,
                    yticklabels=tokens,
                    center=0,
                    vmin=-max(abs(difference.min()), abs(difference.max())),
                    vmax=max(abs(difference.min()), abs(difference.max()))
                )
                ax3.set_title("Difference (Other - Base)")
                ax3.set_xlabel("Attended to")
                plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                # Add overall title
                plt.suptitle(f"Layer {layer} Attention Comparison", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
                
                # Save figure
                plt.savefig(f"{save_dir}/layer_{layer}_attention_comparison.png", dpi=300, bbox_inches='tight')
                plt.close()


def main():
    # Sample texts for analysis
    texts = [
        "The patient presented with acute myocardial infarction and was administered aspirin.",
        "Recurrent neural networks have been largely replaced by transformer architectures.",
        "The BRCA1 gene mutation is associated with increased risk of breast cancer."
    ]
    
    # Create analyzer
    print("Creating Attention Lens analyzer...")
    analyzer = AttentionLensAnalyzer(model_path)
    
    # Analyze each text
    for i, text in enumerate(texts):
        text_id = f"text_{i+1}"
        print(f"\nAnalyzing {text_id}: {text}")
        
        # Run analysis
        print("Running attention lens analysis...")
        save_dir = os.path.join(output_dir, text_id)
        results = analyzer.analyze_text(text, save_dir=save_dir)
        
        if not results['attention_maps']:
            print("No attention maps were captured. The model may not expose attention weights.")
    
    # Optional: If you have a fine-tuned version of the model, you can compare them
    # fine_tuned_model_path = "path/to/fine_tuned_model"
    # analyzer.analyze_model_comparison(texts[0], fine_tuned_model_path, save_dir=os.path.join(output_dir, "comparison"))
    
    print("\nAnalysis complete! Check the output directories for attention visualizations.")

if __name__ == "__main__":
    main()