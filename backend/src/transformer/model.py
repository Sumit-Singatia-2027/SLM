"""Model loading with LoRA adapter support"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os
from pathlib import Path

# BUG FIX: the old default here was the Windows-only literal ".\models\lora_adapter".
# Two problems with that:
#   1. It's an *unescaped* string literal -- "\m" and "\l" aren't valid Python
#      escapes, so Python keeps the backslashes as literal characters (and
#      newer Python versions even emit a SyntaxWarning about it).
#   2. Even ignoring that, backslash is not a path separator on Linux/macOS
#      (including Colab), so os.path.exists() on that literal string returns
#      False there -- the adapter silently fails to load and the code quietly
#      falls back to the base model with no error, no crash, just worse math
#      performance nobody notices.
# Fix: build the path with pathlib (OS-independent separators) and anchor it
# to this file's location rather than the current working directory, so it
# resolves correctly no matter where the caller's script/notebook is run from.
_BACKEND_ROOT = Path(__file__).resolve().parents[2]  # backend/src/transformer -> backend/
DEFAULT_LORA_ADAPTER_PATH = str(_BACKEND_ROOT / "models" / "lora_adapter")


class MathTransformerModel:
    """Wrapper for the Qwen2.5-Math model with LoRA"""
    
    def __init__(
        self, 
        base_model_id="Qwen/Qwen2.5-Math-1.5B-Instruct",
        lora_adapter_path=DEFAULT_LORA_ADAPTER_PATH,  # Path to your fine-tuned LoRA weights
        device=None
    ):
        self.base_model_id = base_model_id
        self.lora_adapter_path = lora_adapter_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔄 Loading base model: {base_model_id}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_id, 
            trust_remote_code=True,
            padding_side="left"
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Load LoRA adapter if provided
        if lora_adapter_path and os.path.exists(lora_adapter_path):
            print(f"🔄 Loading LoRA adapter from: {lora_adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.base_model,
                lora_adapter_path,
                torch_dtype=torch.bfloat16
            )
            print("✅ LoRA adapter loaded successfully!")
        else:
            self.model = self.base_model
            if lora_adapter_path:
                print(f"⚠️  LoRA path not found: {lora_adapter_path}")
                print("📌 Using base model without LoRA")
        
        self.model.eval()  # Set to evaluation mode
        print(f"✅ Model loaded on {self.device}")
    
    def get_model(self):
        """Return the model (with LoRA if loaded)"""
        return self.model
    
    def get_tokenizer(self):
        """Return the tokenizer"""
        return self.tokenizer
    
    def get_eos_token_id(self):
        """Get the end-of-sequence token ID"""
        return self.tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    def merge_and_save(self, output_path):
        """Merge LoRA weights with base model and save"""
        if isinstance(self.model, PeftModel):
            print(f"🔄 Merging LoRA weights with base model...")
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
            print(f"✅ Merged model saved to: {output_path}")
        else:
            print("⚠️  No LoRA adapter to merge")