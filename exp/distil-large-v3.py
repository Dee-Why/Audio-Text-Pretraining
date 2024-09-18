import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)

# dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

# sample = dataset[0]["audio"]

import librosa

# Path to your .m4a file
file_path = 'exp/recording2.m4a'  # Update this path to your m4a file

# Load the audio file
audio, sampling_rate = librosa.load(file_path, sr=None)

# Print results
print(f'Audio data: {audio}')
print(f'Sampling rate: {sampling_rate}')


result = pipe(audio)
print(result["text"])

