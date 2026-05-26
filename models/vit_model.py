"""
vit_model.py — Vision Transformer Gesture Classifier (Proposed)
================================================================
ViT-based classifiers using the timm library for hand gesture recognition.
Supports pretrained ViT-Base and ViT-Small with fine-tuning.
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("Please install timm: pip install timm")


class GestureViT(nn.Module):
    """
    Vision Transformer for gesture classification using timm.
    
    Default: vit_base_patch16_224 (pretrained on ImageNet)
    
    Architecture:
        Input Image (3 × 224 × 224)
            ↓
        Patch Embedding: 16×16 patches → 196 tokens
            ↓
        [CLS] Token + Positional Encoding
            ↓
        Transformer Encoder × L layers
            ↓
        [CLS] Token → Classification Head → num_classes
    
    Input: (B, 3, 224, 224)
    Output: (B, num_classes)
    """
    
    def __init__(self, model_name="vit_base_patch16_224", num_classes=7,
                 pretrained=True, dropout=0.1):
        super(GestureViT, self).__init__()
        
        self.model_name = model_name
        
        # Create ViT model using timm
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )
    
    def forward(self, x):
        return self.vit(x)
    
    def freeze_backbone(self):
        """
        Phase 1: Freeze all layers except the classification head.
        Only the final linear layer will be trained.
        """
        for name, param in self.vit.named_parameters():
            if "head" not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """
        Phase 2: Unfreeze all layers for end-to-end fine-tuning.
        """
        for param in self.vit.parameters():
            param.requires_grad = True
    
    def get_attention_maps(self, x):
        """
        Extract attention maps from the last transformer block.
        Useful for visualization and interpretability.
        
        Args:
            x: Input tensor (B, 3, 224, 224)
        
        Returns:
            attention: Attention weights from last block
        """
        # Register hook to capture attention weights
        attention_weights = []
        
        def hook_fn(module, input, output):
            # output is (attn_output, attn_weights) for some implementations
            attention_weights.append(output)
        
        # Get the last attention block
        last_block = self.vit.blocks[-1].attn
        hook = last_block.attn_drop.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            _ = self.vit(x)
        
        hook.remove()
        return attention_weights


def build_vit_model(model_name="vit_base_patch16_224", num_classes=7,
                    pretrained=True, dropout=0.1):
    """
    Factory function to build ViT model.
    
    Args:
        model_name: timm model name
            - "vit_base_patch16_224"  (~86M params, higher accuracy)
            - "vit_small_patch16_224" (~22M params, faster)
            - "vit_tiny_patch16_224"  (~5.7M params, fastest)
        num_classes: Number of gesture classes
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate for classification head
    
    Returns:
        nn.Module: The ViT model
    """
    return GestureViT(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )


if __name__ == "__main__":
    # Quick test
    print("Testing ViT model...")
    model = GestureViT(
        model_name="vit_base_patch16_224",
        num_classes=7,
        pretrained=False,  # Skip download for testing
    )
    
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"ViT-Base: input={x.shape} → output={out.shape}")
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable:,}")
    
    # Test freezing
    model.freeze_backbone()
    trainable_frozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nAfter freeze_backbone():")
    print(f"Trainable Parameters: {trainable_frozen:,} ({trainable_frozen/1e6:.2f}M)")
    
    model.unfreeze_backbone()
    trainable_unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nAfter unfreeze_backbone():")
    print(f"Trainable Parameters: {trainable_unfrozen:,} ({trainable_unfrozen/1e6:.2f}M)")
