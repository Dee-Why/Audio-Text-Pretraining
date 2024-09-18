import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa

# Set device and model parameters
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v3"

# Load the model and processor
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Set up the pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)

# Load and resample the audio to 16 kHz
file_path = 'exp/recording2.m4a'  # Path to your .m4a file
audio, sampling_rate = librosa.load(file_path, sr=16000)  # Resample to 16 kHz

# Ensure the audio is in a format the pipeline can process
audio_input = {"array": audio, "sampling_rate": 16000}

# Use the pipeline for speech recognition
result = pipe(audio_input)
print(result["text"])

