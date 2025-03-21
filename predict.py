# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from snac import SNAC
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_CACHE = "orpheus-3b-0.1-ft"
SNAC_MODEL = "snac_24khz"
MODEL_URL = "https://weights.replicate.delivery/default/canopylabs/orpheus-3b-0.1-ft/model.tar"
SNAC_URL = "https://weights.replicate.delivery/default/hubertsiuzdak/snac_24khz/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"
        
        # Download the weights if they don't exist
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # Download the weights if they don't exist
        if not os.path.exists(SNAC_MODEL):
            download_weights(SNAC_URL, SNAC_MODEL)

        # Load SNAC model
        print("Loading SNAC model...")
        self.snac_model = SNAC.from_pretrained(SNAC_MODEL)
        self.snac_model = self.snac_model.to(self.device)

        # Load Orpheus model
        print(f"Loading model...")
        
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_CACHE, torch_dtype=torch.bfloat16)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE)
        print(f"Models loaded successfully to {self.device}")

    def process_prompt(self, prompt, voice):
        prompt = f"{voice}: {prompt}"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        
        start_token = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)
        
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
        attention_mask = torch.ones_like(modified_input_ids)
        
        return modified_input_ids.to(self.device), attention_mask.to(self.device)

    def parse_output(self, generated_ids):
        token_to_find = 128257
        token_to_remove = 128258
        
        token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

        if len(token_indices[1]) > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
        else:
            cropped_tensor = generated_ids

        processed_rows = []
        for row in cropped_tensor:
            masked_row = row[row != token_to_remove]
            processed_rows.append(masked_row)

        code_lists = []
        for row in processed_rows:
            row_length = row.size(0)
            new_length = (row_length // 7) * 7
            trimmed_row = row[:new_length]
            trimmed_row = [t - 128266 for t in trimmed_row]
            code_lists.append(trimmed_row)
            
        return code_lists[0]

    def redistribute_codes(self, code_list):
        device = next(self.snac_model.parameters()).device
        
        layer_1 = []
        layer_2 = []
        layer_3 = []
        for i in range((len(code_list)+1)//7):
            layer_1.append(code_list[7*i])
            layer_2.append(code_list[7*i+1]-4096)
            layer_3.append(code_list[7*i+2]-(2*4096))
            layer_3.append(code_list[7*i+3]-(3*4096))
            layer_2.append(code_list[7*i+4]-(4*4096))
            layer_3.append(code_list[7*i+5]-(5*4096))
            layer_3.append(code_list[7*i+6]-(6*4096))
            
        codes = [
            torch.tensor(layer_1, device=device).unsqueeze(0),
            torch.tensor(layer_2, device=device).unsqueeze(0),
            torch.tensor(layer_3, device=device).unsqueeze(0)
        ]
        
        audio_hat = self.snac_model.decode(codes)
        return audio_hat.detach().squeeze().cpu().numpy()

    def predict(
        self,
        text: str = Input(description="Text to convert to speech"),
        voice: str = Input(description="Voice to use", choices=["tara", "dan", "josh", "emma"], default="tara"),
        temperature: float = Input(description="Temperature for generation", default=0.6, ge=0.1, le=1.5),
        top_p: float = Input(description="Top P for nucleus sampling", default=0.95, ge=0.1, le=1.0),
        repetition_penalty: float = Input(description="Repetition penalty", default=1.1, ge=1.0, le=2.0),
        max_new_tokens: int = Input(description="Maximum number of tokens to generate", default=1200, ge=100, le=20000)
    ) -> Path:
        """Generate speech from text using Orpheus TTS model"""
        
        # Process the input text
        input_ids, attention_mask = self.process_prompt(text, voice)
        
        # Generate speech tokens
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
                eos_token_id=128258,
            )
        
        # Process the generated tokens
        code_list = self.parse_output(generated_ids)
        
        # Convert to audio
        audio_samples = self.redistribute_codes(code_list)
        
        # Save the audio to a WAV file
        output_path = Path("/tmp/output.wav")
        sf.write(output_path, audio_samples, 24000)
        
        return output_path
