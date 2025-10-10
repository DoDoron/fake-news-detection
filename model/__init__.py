"""Collection of baseline models."""
from .bow_mlp import BoWMLP
from .cnn_text import CNNText
from .bilstm import BiLSTM
from .tiny_transformer import TinyTransformer

MODEL_REGISTRY = {
    "bow_mlp": BoWMLP,
    "cnn_text": CNNText,
    "bilstm": BiLSTM,
    "tiny_transformer": TinyTransformer,
}

__all__ = ["BoWMLP", "CNNText", "BiLSTM", "TinyTransformer", "MODEL_REGISTRY"]
