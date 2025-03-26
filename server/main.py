from fastapi import FastAPI, File, UploadFile, Form

from transformers import SeamlessM4TModel, SeamlessM4TProcessor, SeamlessM4TForTextToText, AutoProcessor
import torch
import torchaudio
import io
from saveAudio import save_audio 


app = FastAPI(title="SeamlessM4T API", description="Translate Text & Speech Seamlessly", version="1.0")




text_processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
text_model = SeamlessM4TForTextToText.from_pretrained("facebook/seamless-m4t-v2-large")

speech_processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
speech_model = SeamlessM4TModel.from_pretrained("facebook/seamless-m4t-v2-large")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_model.to(device)
speech_model.to(device)


language_codes = {

    "English": "eng",

    "French": "fra",

    "Spanish": "spa",

    "German": "deu",

    "Chinese": "zho",

    "Hindi": "hin",

    "Japanese": "jpn",

    "Korean": "kor",

    "Arabic": "arb",

}


@app.get("/")

async def home():

    return {"message": "Welcome to SeamlessM4T Translation API"}

@app.post("/translate-text")

async def translate_text(text: str = Form(...), target_language: str = Form(...)):

    if target_language not in language_codes:

        return {"error": "Invalid language. Choose from: " + ", ".join(language_codes.keys())}


    inputs = text_processor(text=text, src_lang="eng", return_tensors="pt").to(device)


    with torch.no_grad():

        translated_tokens = text_model.generate(**inputs, tgt_lang=language_codes[target_language])


    translated_text = text_processor.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    return {"translated_text": translated_text}


@app.post("/speech-to-text")

async def speech_to_text(audio: UploadFile = File(...), target_language: str = Form(...)):

    if target_language not in language_codes:

        return {"error": "Invalid language. Choose from: " + ", ".join(language_codes.keys())}


    try:

        audio_data = await audio.read()

        audio_stream = io.BytesIO(audio_data)

        waveform, original_sample_rate = torchaudio.load(audio_stream) #renamed original_sample_rate


        print(f"Waveform shape: {waveform.shape}, dtype: {waveform.dtype}, sample rate: {original_sample_rate}")


        if original_sample_rate != 16000:

            waveform = torchaudio.functional.resample(waveform, orig_freq=original_sample_rate, new_freq=16000)

            sample_rate = 16000 #update sample rate

            print(f"Waveform shape after resample: {waveform.shape}, dtype: {waveform.dtype}, sample rate: {sample_rate}") #added print

        else:

            sample_rate = original_sample_rate


        if waveform.shape[0] > 1:

            waveform = waveform.mean(dim=0, keepdim=True)


        input_features = speech_processor(audios=waveform.squeeze(), sampling_rate=16000, return_tensors="pt").to(device)

        print(f"Input features shape: {input_features['input_features'].shape}")

        print(f"Input features: {input_features['input_features']}")


        with torch.no_grad():

            generated_ids = speech_model.generate(input_features=input_features["input_features"], tgt_lang=language_codes[target_language])


        if isinstance(generated_ids, tuple):

            generated_ids = generated_ids[0]


        generated_ids = generated_ids.to("cpu").to(torch.long)

        print(f"Generated IDs: {generated_ids}")


        transcribed_text = speech_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"transcribed text: {transcribed_text}")


        return {"transcribed_text": transcribed_text}


    except Exception as e:

        return {"error": str(e)}




@app.post("/text-to-speech")

async def text_to_speech(text: str = Form(...)):

    tgt_lang = "eng"

    inputs = text_processor(text=text, src_lang="eng", tgt_lang=tgt_lang, return_tensors="pt").to(device)


    with torch.no_grad():

        speech_output = text_model.generate(**inputs)


    if isinstance(speech_output, tuple):

        speech_output = speech_output[0]


    speech_audio_np = speech_output.cpu().numpy()


    output_filename = "speech_output.wav"

    save_audio(speech_audio_np, 16000, output_filename)


    return {"audio_data": speech_output.tolist()}




@app.post("/speech-to-speech")

async def speech_to_speech(audio: UploadFile = File(...), target_language: str = Form(...)):

    if target_language not in language_codes:

        return {"error": "Invalid language. Choose from: " + ", ".join(language_codes.keys())}


    audio_data = await audio.read()

    audio_tensor, sample_rate = torchaudio.load(io.BytesIO(audio_data))


    if sample_rate != 16000:

        audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=sample_rate, new_freq=16000)


    if audio_tensor.shape[0] > 1:

        audio_tensor = audio_tensor.mean(dim=0, keepdim=True)


    inputs = speech_processor(audios=audio_tensor.squeeze(), sampling_rate=16000, return_tensors="pt").to(device)


    with torch.no_grad():

        translated_audio = speech_model.generate(

            input_features=inputs["input_features"],

            tgt_lang=language_codes[target_language], 

        )


    if isinstance(translated_audio, tuple):

        translated_audio = translated_audio[0]


    translated_audio_np = translated_audio.cpu().numpy()


    output_filename = "translated_output.wav"

    save_audio(translated_audio_np, 16000, output_filename)


    return {"audio_data": translated_audio.tolist()}


@app.post("/test-upload")

async def test_upload(audio: UploadFile = File(...)):

    return {"filename": audio.filename}