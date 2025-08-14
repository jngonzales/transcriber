from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
import torch
import os
from pyannote.audio import Pipeline

# Step 1: Initialize Flask App and CORS
app = Flask(__name__)

# --- CRUCIAL CHANGE FOR NGROK TESTING ---
# Because ngrok's URL changes, we temporarily allow all origins.
# For a more permanent solution with a static domain (paid ngrok, cloud hosting, or fixed IP with port forwarding),
# you would revert this to specify the exact allowed domain(s) for security.
CORS(app)
# --- END CRUCIAL CHANGE ---

# Step 2: Determine device and model size for Whisper
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model_size = "large-v3"

# Initialize diarization pipeline as None by default
diarization_pipeline = None

# Step 3: Load the Whisper model
print(f"Loading Whisper model '{whisper_model_size}' on device '{device}'...")
whisper_model = WhisperModel(whisper_model_size, device=device, compute_type="float16" if device == "cuda" else "int8")
print("Whisper model loaded successfully.")

# Step 4: Define the transcription endpoint
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    print("Received a request on /transcribe")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Get diarization flag from frontend
    diarize = request.form.get('diarize', 'false').lower() == 'true'
    print(f"Diarization requested: {diarize}")

    audio_path = None

    try:
        # Save the uploaded file temporarily
        audio_filename = file.filename if file.filename else "uploaded_audio"
        audio_path = os.path.join(os.getcwd(), f"temp_{os.getpid()}_{audio_filename}")
        file.save(audio_path)
        print(f"Temporary audio saved to: {audio_path}")

        # --- Diarization Model Loading (Lazy Load / Check on Demand) ---
        global diarization_pipeline
        if diarize and diarization_pipeline is None:
            hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if not hf_token:
                raise ValueError("HUGGING_FACE_HUB_TOKEN environment variable is not set. Speaker diarization requires it. Please set it and accept the pyannote model terms on Hugging Face.")
            
            print("Attempting to load Speaker Diarization model (pyannote.audio)...")
            try:
                diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization@3.1", # Keeping 3.1 as it's the latest
                    use_auth_token=hf_token
                )
                if device == "cuda":
                    diarization_pipeline.to(torch.device("cuda"))
                print("Speaker Diarization model loaded successfully.")
            except Exception as e:
                diarization_pipeline = None
                raise Exception(f"Failed to load diarization model. Did you accept its terms on Hugging Face (e.g., pyannote/speaker-diarization-3.1) and set the HUGGING_FACE_HUB_TOKEN environment variable? Error: {e}")

        # --- Transcription Step ---
        segments_generator, info = whisper_model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=True,
            temperature=0
        )
        
        all_whisper_words = []
        for segment in segments_generator:
            for word in segment.words:
                all_whisper_words.append({
                    "text": word.word.strip(),
                    "start": word.start,
                    "end": word.end,
                    "probability": word.probability
                })
        
        full_text_from_whisper = "".join(word["text"] + " " for word in all_whisper_words).strip()

        word_chunks_for_frontend = []
        final_transcript_text = ""

        if diarize and diarization_pipeline:
            print("Performing speaker diarization...")
            diarization_results = diarization_pipeline(audio_path)

            current_speaker = None
            current_speaker_words = []
            
            def add_speaker_segment():
                nonlocal final_transcript_text
                if current_speaker_words:
                    segment_text = " ".join(w["text"] for w in current_speaker_words).strip()
                    if segment_text:
                        if final_transcript_text and not final_transcript_text.endswith('\n'):
                            final_transcript_text += "\n"
                        final_transcript_text += f"{current_speaker}: {segment_text}\n"
                    current_speaker_words.clear()

            for speech_segment, speaker_label in diarization_results.itertracks(yield_label=True):
                if current_speaker is not None and speaker_label != current_speaker:
                    add_speaker_segment()
                
                current_speaker = speaker_label
                
                for word_idx, word in enumerate(all_whisper_words):
                    if word["start"] >= speech_segment.start and word["end"] <= speech_segment.end:
                        current_speaker_words.append(word)
                        word_chunks_for_frontend.append({
                            "text": word["text"],
                            "timestamp": [word["start"], word["end"]],
                            "score": word["probability"],
                            "speaker": speaker_label
                        })
                    elif word["start"] > speech_segment.end:
                        pass
            
            add_speaker_segment()

            print("Diarization complete.")
        else:
            final_transcript_text = full_text_from_whisper
            for word in all_whisper_words:
                word_chunks_for_frontend.append({
                    "text": word["text"],
                    "timestamp": [word["start"], word["end"]],
                    "score": word["probability"],
                    "speaker": None
                })
            print("Standard transcription (no diarization) complete.")

        print("Transcription successful. Sending response.")
        return jsonify({
            'text': final_transcript_text.strip(),
            'chunks': word_chunks_for_frontend,
            'language': info.language
        })

    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"Temporary audio file deleted: {audio_path}")

# Step 6: Start the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)