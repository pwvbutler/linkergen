import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Nach0Encoder(nn.Module):
    """
    Wraps the frozen nach0 T5 encoder. Used only as a fixed text-embedding
    step in train.py/sample.py, never fine-tuned, so its parameters never
    enter LinkerGen's optimizer or checkpoint.
    """
    def __init__(
        self,
        model_name: str = "insilicomedicine/nach0_base",
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        full_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.encoder = full_model.encoder
        del full_model

        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @property
    def hidden_size(self) -> int:
        return self.encoder.config.hidden_size

    def forward(
        self,
        descriptions: list[str],
        device: torch.device,
        max_length: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            hidden_states: (B, S, hidden_size)
            attention_mask: (B, S), used to mask padding in cross-attention
        """
        enc = self.tokenizer(
            descriptions,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = self.encoder(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
        return out.last_hidden_state, enc["attention_mask"]
