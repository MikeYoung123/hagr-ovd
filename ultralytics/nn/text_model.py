# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from ultralytics.utils import checks
from ultralytics.utils.torch_utils import smart_inference_mode

try:
    import clip
except ImportError:
    checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
    import clip


class TextModel(nn.Module):
    """Abstract base class for text encoding models.

    This class defines the interface for text encoding models used in vision-language tasks. Subclasses must implement
    the tokenize and encode_text methods to provide text tokenization and encoding functionality.

    Methods:
        tokenize: Convert input texts to tokens for model processing.
        encode_text: Encode tokenized texts into normalized feature vectors.
    """

    def __init__(self):
        """Initialize the TextModel base class."""
        super().__init__()

    @abstractmethod
    def tokenize(self, texts):
        """Convert input texts to tokens for model processing."""
        pass

    @abstractmethod
    def encode_text(self, texts, dtype):
        """Encode tokenized texts into normalized feature vectors."""
        pass

    def split_tokens(self, tokens, batch: int | None):
        """Split tokenized text into batches while preserving tokenizer output structure."""
        if isinstance(tokens, torch.Tensor):
            batch = batch or len(tokens)
            return tokens.split(batch)
        if isinstance(tokens, Mapping):
            length = next(iter(tokens.values())).shape[0]
            batch = batch or length
            return ({k: v[i : i + batch] for k, v in tokens.items()} for i in range(0, length, batch))
        raise TypeError(f"Unsupported token type {type(tokens)} for text batching.")


class CLIP(TextModel):
    """Implements OpenAI's CLIP (Contrastive Language-Image Pre-training) text encoder.

    This class provides a text encoder based on OpenAI's CLIP model, which can convert text into feature vectors that
    are aligned with corresponding image features in a shared embedding space.

    Attributes:
        model (clip.model.CLIP): The loaded CLIP model.
        image_preprocess (callable): Preprocessing transform for images.
        device (torch.device): Device where the model is loaded.

    Methods:
        tokenize: Convert input texts to CLIP tokens.
        encode_text: Encode tokenized texts into normalized feature vectors.

    Examples:
        >>> import torch
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> clip_model = CLIP(size="ViT-B/32", device=device)
        >>> tokens = clip_model.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> text_features = clip_model.encode_text(tokens)
        >>> print(text_features.shape)
    """

    def __init__(self, size: str, device: torch.device) -> None:
        """Initialize the CLIP text encoder.

        This class implements the TextModel interface using OpenAI's CLIP model for text encoding. It loads a
        pre-trained CLIP model of the specified size and prepares it for text encoding tasks.

        Args:
            size (str): Model size identifier (e.g., 'ViT-B/32').
            device (torch.device): Device to load the model on.
        """
        super().__init__()
        self.model, self.image_preprocess = clip.load(size, device=device)
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts: str | list[str], truncate: bool = True) -> torch.Tensor:
        """Convert input texts to CLIP tokens.

        Args:
            texts (str | list[str]): Input text or list of texts to tokenize.
            truncate (bool, optional): Whether to trim texts that exceed CLIP's context length. Defaults to True to
                avoid RuntimeError from overly long inputs while still allowing explicit opt-out.

        Returns:
            (torch.Tensor): Tokenized text tensor with shape (batch_size, context_length) ready for model processing.

        Examples:
            >>> model = CLIP("ViT-B/32", device="cpu")
            >>> tokens = model.tokenize("a photo of a cat")
            >>> print(tokens.shape)  # torch.Size([1, 77])
            >>> strict_tokens = model.tokenize("a photo of a cat", truncate=False)  # Enforce strict length checks
            >>> print(strict_tokens.shape)  # Same shape/content as tokens since prompt less than 77 tokens
        """
        return clip.tokenize(texts, truncate=truncate).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Encode tokenized texts into normalized feature vectors.

        This method processes tokenized text inputs through the CLIP model to generate feature vectors, which are then
        normalized to unit length. These normalized vectors can be used for text-image similarity comparisons.

        Args:
            texts (torch.Tensor): Tokenized text inputs, typically created using the tokenize() method.
            dtype (torch.dtype, optional): Data type for output features.

        Returns:
            (torch.Tensor): Normalized text feature vectors with unit length (L2 norm = 1).

        Examples:
            >>> clip_model = CLIP("ViT-B/32", device="cuda")
            >>> tokens = clip_model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = clip_model.encode_text(tokens)
            >>> features.shape
            torch.Size([2, 512])
        """
        txt_feats = self.model.encode_text(texts).to(dtype)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        return txt_feats

    @smart_inference_mode()
    def encode_image(self, image: Image.Image | torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Encode images into normalized feature vectors.

        This method processes image inputs through the CLIP model to generate feature vectors, which are then
        normalized to unit length. These normalized vectors can be used for text-image similarity comparisons.

        Args:
            image (PIL.Image | torch.Tensor): Image input as a PIL Image or preprocessed tensor. If a PIL Image is
                provided, it will be converted to a tensor using the model's image preprocessing function.
            dtype (torch.dtype, optional): Data type for output features.

        Returns:
            (torch.Tensor): Normalized image feature vectors with unit length (L2 norm = 1).

        Examples:
            >>> from ultralytics.nn.text_model import CLIP
            >>> from PIL import Image
            >>> clip_model = CLIP("ViT-B/32", device="cuda")
            >>> image = Image.open("path/to/image.jpg")
            >>> image_tensor = clip_model.image_preprocess(image).unsqueeze(0).to("cuda")
            >>> features = clip_model.encode_image(image_tensor)
            >>> features.shape
            torch.Size([1, 512])
        """
        if isinstance(image, Image.Image):
            image = self.image_preprocess(image).unsqueeze(0).to(self.device)
        img_feats = self.model.encode_image(image).to(dtype)
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        return img_feats


class SigLIP(TextModel):
    """Implements a Hugging Face SigLIP/SigLIP2 text encoder."""

    def __init__(self, size: str, device: torch.device) -> None:
        """Initialize the SigLIP text encoder.

        Args:
            size (str): Hugging Face model id or local model directory.
            device (torch.device): Device to load the model on.
        """
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            checks.check_requirements(["transformers>=4.49.0,<5.0.0", "safetensors>=0.4.3"])
            from transformers import AutoModel, AutoTokenizer

        super().__init__()
        local_files_only = Path(size).exists()
        self.model = AutoModel.from_pretrained(size, local_files_only=local_files_only).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(size, local_files_only=local_files_only)
        text_config = getattr(self.model.config, "text_config", None)
        self.max_length = getattr(text_config, "max_position_embeddings", None)
        if not self.max_length or self.max_length > 100000:
            self.max_length = 64
        self.device = device
        self.eval()

    def tokenize(self, texts: str | list[str], truncate: bool = True) -> dict[str, torch.Tensor]:
        """Convert input texts to SigLIP tokens."""
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=truncate,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in tokens.items()}

    @staticmethod
    def _extract_features(outputs):
        """Extract pooled text features from tensor, dict, tuple, or HF model-output objects."""
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, Mapping):
            for key in ("text_embeds", "pooler_output", "last_hidden_state"):
                value = outputs.get(key)
                if isinstance(value, torch.Tensor):
                    return value[:, 0] if key == "last_hidden_state" and value.ndim == 3 else value
        for attr in ("text_embeds", "pooler_output", "last_hidden_state"):
            value = getattr(outputs, attr, None)
            if isinstance(value, torch.Tensor):
                return value[:, 0] if attr == "last_hidden_state" and value.ndim == 3 else value
        if isinstance(outputs, (tuple, list)):
            for value in outputs:
                if isinstance(value, torch.Tensor):
                    return value[:, 0] if value.ndim == 3 else value
        raise TypeError(f"Unsupported SigLIP text output type: {type(outputs)}")

    @smart_inference_mode()
    def encode_text(self, texts: Mapping[str, torch.Tensor] | torch.Tensor, dtype: torch.dtype = torch.float32):
        """Encode tokenized texts into normalized feature vectors."""
        inputs = {"input_ids": texts} if isinstance(texts, torch.Tensor) else dict(texts)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        if hasattr(self.model, "get_text_features"):
            txt_feats = self._extract_features(self.model.get_text_features(**inputs))
        else:
            text_model = self.model.text_model if hasattr(self.model, "text_model") else self.model
            txt_feats = self._extract_features(text_model(**inputs))
            if hasattr(self.model, "text_projection"):
                txt_feats = self.model.text_projection(txt_feats)
        txt_feats = txt_feats.to(dtype)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        return txt_feats


class MobileCLIP(TextModel):
    """Implement Apple's MobileCLIP text encoder for efficient text encoding.

    This class implements the TextModel interface using Apple's MobileCLIP model, providing efficient text encoding
    capabilities for vision-language tasks with reduced computational requirements compared to standard CLIP models.

    Attributes:
        model (mobileclip.model.MobileCLIP): The loaded MobileCLIP model.
        tokenizer (callable): Tokenizer function for processing text inputs.
        device (torch.device): Device where the model is loaded.
        config_size_map (dict): Mapping from size identifiers to model configuration names.

    Methods:
        tokenize: Convert input texts to MobileCLIP tokens.
        encode_text: Encode tokenized texts into normalized feature vectors.

    Examples:
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> text_encoder = MobileCLIP(size="s0", device=device)
        >>> tokens = text_encoder.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> features = text_encoder.encode_text(tokens)
    """

    config_size_map = {"s0": "s0", "s1": "s1", "s2": "s2", "b": "b", "blt": "b"}

    def __init__(self, size: str, device: torch.device) -> None:
        """Initialize the MobileCLIP text encoder.

        This class implements the TextModel interface using Apple's MobileCLIP model for efficient text encoding.

        Args:
            size (str): Model size identifier (e.g., 's0', 's1', 's2', 'b', 'blt').
            device (torch.device): Device to load the model on.
        """
        try:
            import mobileclip
        except ImportError:
            # Ultralytics fork preferred since Apple MobileCLIP repo has incorrect version of torchvision
            checks.check_requirements("git+https://github.com/ultralytics/mobileclip.git")
            import mobileclip

        super().__init__()
        config = self.config_size_map[size]
        file = f"mobileclip_{size}.pt"
        if not Path(file).is_file():
            from ultralytics import download

            download(f"https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/{file}")
        self.model = mobileclip.create_model_and_transforms(f"mobileclip_{config}", pretrained=file, device=device)[0]
        self.tokenizer = mobileclip.get_tokenizer(f"mobileclip_{config}")
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        """Convert input texts to MobileCLIP tokens.

        Args:
            texts (list[str]): List of text strings to tokenize.

        Returns:
            (torch.Tensor): Tokenized text inputs with shape (batch_size, sequence_length).

        Examples:
            >>> model = MobileCLIP("s0", "cpu")
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
        """
        return self.tokenizer(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Encode tokenized texts into normalized feature vectors.

        Args:
            texts (torch.Tensor): Tokenized text inputs.
            dtype (torch.dtype, optional): Data type for output features.

        Returns:
            (torch.Tensor): Normalized text feature vectors with L2 normalization applied.

        Examples:
            >>> model = MobileCLIP("s0", device="cpu")
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = model.encode_text(tokens)
            >>> features.shape
            torch.Size([2, 512])  # Actual dimension depends on model size
        """
        text_features = self.model.encode_text(texts).to(dtype)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features


class MobileCLIPTS(TextModel):
    """Load a TorchScript traced version of MobileCLIP.

    This class implements the TextModel interface using Apple's MobileCLIP model in TorchScript format, providing
    efficient text encoding capabilities for vision-language tasks with optimized inference performance.

    Attributes:
        encoder (torch.jit.ScriptModule): The loaded TorchScript MobileCLIP text encoder.
        tokenizer (callable): Tokenizer function for processing text inputs.
        device (torch.device): Device where the model is loaded.

    Methods:
        tokenize: Convert input texts to MobileCLIP tokens.
        encode_text: Encode tokenized texts into normalized feature vectors.

    Examples:
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> text_encoder = MobileCLIPTS(device=device)
        >>> tokens = text_encoder.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> features = text_encoder.encode_text(tokens)
    """

    def __init__(self, device: torch.device, weight: str = "mobileclip_blt.ts"):
        """Initialize the MobileCLIP TorchScript text encoder.

        This class implements the TextModel interface using Apple's MobileCLIP model in TorchScript format for efficient
        text encoding with optimized inference performance.

        Args:
            device (torch.device): Device to load the model on.
            weight (str): Path to the TorchScript model weights.
        """
        super().__init__()
        from ultralytics.utils.downloads import attempt_download_asset

        self.encoder = torch.jit.load(attempt_download_asset(weight), map_location=device)
        self.tokenizer = clip.clip.tokenize
        self.device = device

    def tokenize(self, texts: list[str], truncate: bool = True) -> torch.Tensor:
        """Convert input texts to MobileCLIP tokens.

        Args:
            texts (list[str]): List of text strings to tokenize.
            truncate (bool, optional): Whether to trim texts that exceed the tokenizer context length. Defaults to True,
                matching CLIP's behavior to prevent runtime failures on long captions.

        Returns:
            (torch.Tensor): Tokenized text inputs with shape (batch_size, sequence_length).

        Examples:
            >>> model = MobileCLIPTS(device=torch.device("cpu"))
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> strict_tokens = model.tokenize(
            ...     ["a very long caption"], truncate=False
            ... )  # RuntimeError if exceeds 77-token
        """
        return self.tokenizer(texts, truncate=truncate).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Encode tokenized texts into normalized feature vectors.

        Args:
            texts (torch.Tensor): Tokenized text inputs.
            dtype (torch.dtype, optional): Data type for output features.

        Returns:
            (torch.Tensor): Normalized text feature vectors with L2 normalization applied.

        Examples:
            >>> model = MobileCLIPTS(device="cpu")
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = model.encode_text(tokens)
            >>> features.shape
            torch.Size([2, 512])  # Actual dimension depends on model size
        """
        # NOTE: no need to do normalization here as it's embedded in the torchscript model
        return self.encoder(texts).to(dtype)


def build_text_model(variant: str, device: torch.device = None) -> TextModel:
    """Build a text encoding model based on the specified variant.

    Args:
        variant (str): Model variant in format "base:size" (e.g., "clip:ViT-B/32" or "siglip:path").
        device (torch.device, optional): Device to load the model on.

    Returns:
        (TextModel): Instantiated text encoding model.

    Examples:
        >>> model = build_text_model("clip:ViT-B/32", device=torch.device("cuda"))
        >>> model = build_text_model("mobileclip:s0", device=torch.device("cpu"))
        >>> model = build_text_model("siglip:siglip2-base-patch16-512", device=torch.device("cuda"))
    """
    base, size = variant.split(":", 1)
    if base == "clip":
        return CLIP(size, device)
    elif base in {"siglip", "siglip2"}:
        return SigLIP(size, device)
    elif base == "mobileclip":
        return MobileCLIPTS(device)
    elif base == "mobileclip2":
        return MobileCLIPTS(device, weight="mobileclip2_b.ts")
    else:
        raise ValueError(
            f"Unrecognized base model '{base}'. Supported models are 'clip', 'siglip', 'siglip2', "
            "'mobileclip', 'mobileclip2'."
        )

