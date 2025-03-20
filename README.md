# Orpheus TTS - Text-to-Speech Model

This is a [Cog](https://github.com/replicate/cog) model that provides high-quality text-to-speech synthesis using the Orpheus model with SNAC audio generation.

## Model Description

Orpheus is a text-to-speech model that generates natural-sounding speech from text input. It supports multiple voices and provides various parameters to control the generation process.

## Features

- Multiple voice options (tara, dan, josh, emma)
- High-quality 24kHz audio output
- Adjustable generation parameters
- Fast inference using CUDA acceleration

## Requirements

- GPU with CUDA 12.1 support
- Python 3.11
- Required Python packages (installed automatically):
  - torch==2.4.0
  - transformers==4.49.0
  - snac==1.2.1
  - soundfile==0.13.1
  - and other dependencies

## Input Parameters

| Parameter | Type | Description | Default | Range |
|-----------|------|-------------|---------|--------|
| text | string | Text to convert to speech | Required | - |
| voice | string | Voice to use (choices: tara, dan, josh, emma) | tara | - |
| temperature | float | Temperature for generation | 0.6 | 0.1 - 1.5 |
| top_p | float | Top P for nucleus sampling | 0.95 | 0.1 - 1.0 |
| repetition_penalty | float | Repetition penalty | 1.1 | 1.0 - 2.0 |
| max_new_tokens | integer | Maximum number of tokens to generate | 1200 | 100 - 2000 |

## Output

The model outputs a WAV audio file (24kHz sample rate) containing the generated speech.

## Example Usage

```python
from cog import BasePredictor, Input, Path

# Initialize the predictor
predictor = Predictor()

# Generate speech
output_path = predictor.predict(
    text="Hello, how are you today?",
    voice="tara",
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.1,
    max_new_tokens=1200
)
```

## Model Architecture

The model uses a two-stage architecture:
1. Text-to-token generation using a causal language model (Orpheus)
2. Audio synthesis using SNAC (Speech Neural Audio Codec)

## License

Please refer to the model license for usage terms and conditions.

## Acknowledgments

- SNAC model for high-quality audio synthesis
- Cog for model packaging and deployment 