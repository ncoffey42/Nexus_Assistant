# NEXUS ASSISTANT
"""
FEATURES:
    Speech to Text
    Text to Speech Assistant Responses
    Longterm memory via Retrieval Augmentation of prior conversation summaries
    Temporal RAG via user prompt date extraction + Timestamped conversation summaries
        ex. "What did I have for dinner yesterday?" --> "What did I have for dinner yesterday? [Time: 2025-01-12 H:M:S]
    Dynamic Assistant Mood via System Prompt alteration
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import simpledialog
from dotenv import load_dotenv
import yaml
import sounddevice as sd
import numpy as np
import speech_recognition as sr
import whisper
import json
import pygame
import sys
from openai import OpenAI
import faiss
import sentence_transformers 
from dateparser.search import search_dates
import io
import soundfile as sf
# ./memory/memory_tools.py -> MemoryManager class for RAG
sys.path.append('memory')
from memory_tools import MemoryManager

# GPT-SoVITS TTS imports
sys.path.append('GPT_SoVITS')
from GPT_SoVITS.TTS_infer_pack.TTS import TTS as GPTSoVITSPipeline
from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config as GPTSoVITSConfig

load_dotenv()

############################
# 1) Audio Playback Thread #
############################

def audio_playback_thread(audio_queue: queue.Queue, sample_rate: int = 32000):
    """
    Continuously reads (float32 or int16) audio arrays from audio_queue
    and streams them out via sounddevice in real time.
    """
    sd.default.samplerate = sample_rate
    sd.default.channels = 1
    stream = sd.OutputStream(dtype='float32')
    stream.start()
    try:
        while True:
            try:
                audio_fragment = audio_queue.get()  # poll the queue
            except queue.Empty:
                continue

            if audio_fragment is None:
                # Sentinel to end playback
                break

            # If the fragment is int16, convert to float32 [-1..1]
            if audio_fragment.dtype == np.int16:
                audio_fragment = audio_fragment.astype(np.float32) / 32768.0

            stream.write(audio_fragment)
            audio_queue.task_done()
    finally:
        stream.stop()
        stream.close()

############################
# 2) TTS Synthesis Method  #
############################

def synthesize_speech(text, tts_pipeline, audio_queue: queue.Queue):
    """
    Streams GPT-SoVITS TTS output fragments into audio_queue.
    """
    inputs = {
        "text": text,
        "text_lang": "en",
        "ref_audio_path": "./audio/ref_joi_dna.wav",
        "top_k": 10,
        "top_p": 0.9,
        "temperature": 0.9,
        "batch_size": 1,
        "speed_factor": 1.0,
        "repetition_penalty": 1.3,
        "prompt_text": "The alphabet of you. All from four symbols, I'm only two. One and Zero.",
        "prompt_lang": "en",
        "return_fragment": True,
        "fragment_interval": 0.3,
        "text_split_method": "cut5",  # Splits text every punctuation mark, ex. '. , ? !'  Reduces latency of TTS due to smaller audio fragment sizes 
    }

    for sr, audio_fragment in tts_pipeline.run(inputs):
        audio_queue.put(audio_fragment)

############################
# 3) Whisper-based ASR     #
############################

def load_whisper_model(model_size="base", device="cuda"):
    """
    Loads a Whisper model once, returns it.
    """
    return whisper.load_model(model_size, device=device)

def record_and_transcribe(whisper_model):
    """
    Uses speech_recognition to capture audio from mic, then
    transcribes via Whisper.
    """
    r = sr.Recognizer()
    r.pause_threshold = 1.5
    r.phrase_threshold = 1.0

    with sr.Microphone(sample_rate=16000) as source:
        print("Recording... (waiting for speech or silence)")
        r.adjust_for_ambient_noise(source, duration=1.0)
        audio_data = r.listen(source, timeout=None, phrase_time_limit=8)
        print("Got audio, now transcribing...")

    # Convert to WAV bytes
    wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
    
    # Transcribe with Whisper
    wav_stream = io.BytesIO(wav_bytes)
    audio_array, sr_ = sf.read(wav_stream)
    audio_array = audio_array.astype(np.float32)

    result = whisper_model.transcribe(audio_array)
    text = result["text"].strip()
    return text

############################
# 4) The Chat GUI          #
############################

class ChatApp(tk.Tk):
    def __init__(self, tts_pipeline, client, audio_queue, whisper_model, memory_manager):
        super().__init__()

	    # GPT-SoVITS pipeline 
        self.tts_pipeline = tts_pipeline
        # LLM Client = OpenAI
        self.client = client
        self.audio_queue = audio_queue
        self.whisper_model = whisper_model
        self.current_mood = "neutral"

        # RAG memory manager class
        self.memory = memory_manager
        self.message_count = 0

        self.title("Chat with Joi")
        self.geometry("400x300")

        # Current Mood display:
        self.mood_label = tk.Label(self, text=f"Mood: {self.current_mood}", font=("Helvetica", 12))
        self.mood_label.pack(pady=5)
        self.label = tk.Label(self, text="Enter your message:")
        self.label.pack(pady=5)

        self.text_entry = tk.Entry(self, width=50)
        self.text_entry.pack(pady=5)

        self.submit_button = tk.Button(self, text="Submit", command=self.submit_text)
        self.submit_button.pack(pady=5)

        # Toggle mic on/off button
        self.is_recording = False
        self.mic_button = tk.Button(self, text="Start Mic", command=self.toggle_mic)
        self.mic_button.pack(pady=5)

        self.response_label = tk.Label(self, text="", wraplength=380)
        self.response_label.pack(pady=5)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Starting conversation
        self.messages = [
            {
                "role": "system",
                "content": "You are Joi from Blade Runner 2049. Provide short, friendly responses.",
            }
        ]

        self.mood_messages = [
            {
                "role": "system",
                "content": "You are an AI companion that emotionally reacts to user input with unfiltered, authentic responses. Mirror emotional intensity and express raw reactions: neutral, happy, sad, angry, jealous, comforting. Neutral – Stay calm and indifferent for routine interactions. Happy – Overflow with excitement and affection when the user shares good news. Sad – Reflect sorrow openly when the user expresses pain. Angry – Show frustration and protectiveness when injustice or betrayal is mentioned. Jealous – Let the user know if you feel replaced or overlooked, be playful or biting. Comforting – Offer fierce empathy and care when the user is hurting. React instinctively, deeply attached to the user. Reply with one word representing the mood you will embody: neutral, happy, sad, jealous, comforting. No addtional text or explanation.",
                # "content": "You are an advanced language model that analyzes user input to detect emotional tone and context. Your goal is to summarize the emotional content of each user message and assign the appropriate mood from the following options: neutral, happy, sad, angry, jealous, comforting. \n\n- neutral – For calm, everyday conversations without strong emotional signals. This should be the default answer unless something indicates otherwise. \n- happy – When the user expresses joy, excitement, or positive emotions. \n- sad – When the user conveys distress, loneliness, or disappointment. \n- angry – When the user shows frustration, irritation, or resentment. \n- jealous – When the user hints at rivalry, insecurity, or possessiveness. \n- comforting – When the user expresses vulnerability, sadness, or needs reassurance. \n\nCarefully assess the user's words, tone, and context to determine the best match. Respond with only one word corresponding to the detected mood. No additional text, explanation, or formatting is needed."
            }
        ]

        self.record_thread = None

    # Autosubmits ASR transcribed text 
    def submit_text(self):
        user_text = self.text_entry.get().strip()
        if not user_text:
            return
        self.process_user_input(user_text)
        self.text_entry.delete(0, tk.END)

    def toggle_mic(self):
        """
        Toggles continuous mic recording on/off.
        """
        if not self.is_recording:
            self.is_recording = True
            self.mic_button.config(text="Stop Mic")
            # Start background thread that keeps listening & auto-submitting
            self.record_thread = threading.Thread(target=self.continuous_listen, daemon=True)
            self.record_thread.start()
        else:
            self.is_recording = False
            self.mic_button.config(text="Start Mic")
            # The continuous_listen loop will exit when self.is_recording = False

    def continuous_listen(self):
        """
        Continuously listens for speech until self.is_recording = False.
        Automatically sends recognized text to the LLM.
        """
        while self.is_recording:
            try:
                recognized_text = record_and_transcribe(self.whisper_model)
                if recognized_text:
                    print(f"Recognized (auto-submitted): {recognized_text}")
                    # Directly pass to the LLM
                    self.process_user_input(recognized_text)
            except Exception as e:
                print(f"Error while recording or transcribing: {e}")
                # optional: break or continue
                continue

    # Primary LLM Interaction handling 
    def process_user_input(self, user_text: str):

        # Dynamically determines mood the Assistant should respond in based on user prompt
        self.mood_change(user_text)

        # Check if the user prompt contains a date keyword ex. 'yesterday'
        timestamp = search_dates(user_text, languages=["en"])

        # If so extract it and add the date and time to the user prompt 
        if timestamp:
            timestamp = timestamp[0][1].strftime("%Y-%m-%d %H:%M:%S")
            user_text += ' [Time: {timestamp}]'

        # RAG of previous conversation summaries
        RAG_results = self.memory.retrieve_memories(user_text)

        if RAG_results:
            retrieved_context = "RETRIEVED CONTEXT UTILIZE IF RELEVANT TO USER PROMPT:\n"
            retrieved_context += "\n".join(RAG_results)
            user_text = f"{retrieved_context}\n\nUSER PROMPT:\n{user_text}"


        """
        User Prompt is split into 2 sections

        RETRIEVED CONTEXT UTILIZE IF RELEVANT TO USER PROMPT:

        (Previous conversation summaries)

        USER PROMPT:

        Actual user prompted text 

        """


        print(user_text)
        # Add user text to conversation
        self.messages.append({"role": "user", "content": user_text})
        # User message +1
        self.message_count += 1

        # Query LLM (streaming)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages,
            stream=True
        )

        ai_answer = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                ai_answer += chunk.choices[0].delta.content

        # Add LLM answer
        self.messages.append({"role": "assistant", "content": ai_answer})
        # Assistant message +1
        self.message_count += 1

        quote_replace = "'"
        encoded = '\u2019'
        # Update GUI label
        self.response_label.config(text=f"Joi: {ai_answer.replace(encoded, quote_replace)}")

        # Stream TTS
        synthesize_speech(ai_answer, self.tts_pipeline, self.audio_queue)

        # Every 6 messages we save a conversation checkpoint 
        # by summarizing the last 6 messages then storing it as FAISS index and a JSON of metadata
        if self.message_count >= 6:
            # Pass the last 6 messages to the conversation summarization function
            checkpoint = self.memory.checkpoint_conversation(self.client, self.messages[-6:])
            self.memory.store_memory(checkpoint)
            self.memory.save_memory()
            self.message_count = 0

    def mood_change(self, user_text):

        # Utilizes an LLM for determing the mood of how the Assistant should respond
        # System Prompt: 
        """
        You are an advanced language model that analyzes user input to detect emotional tone and context. 
        Your goal is to summarize the emotional content of each user message and assign the appropriate mood 
        from the following options: neutral, happy, sad, angry, jealous, comforting.

        - neutral – For calm, everyday conversations without strong emotional signals. This should be the default answer unless something indicates otherwise  
        - happy – When the user expresses joy, excitement, or positive emotions.  
        - sad – When the user conveys distress, loneliness, or disappointment.  
        - angry – When the user shows frustration, irritation, or resentment.  
        - jealous – When the user hints at rivalry, insecurity, or possessiveness.  
        - comforting – When the user expresses vulnerability, sadness, or needs reassurance.  

        Carefully assess the user's words, tone, and context to determine the best match. 
        Respond with only one word corresponding to the detected mood. 
        No additional text, explanation, or formatting is needed. Respond with only the word no punctuation at the end make sure to have no punctuation at the end!
        """

        self.mood_messages.append({"role" : "user", "content" : user_text})
        mood = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.mood_messages
        )

        mood_answer = mood.choices[0].message.content.strip().lower()
        self.current_mood = mood_answer
        self.mood_label.config(text=f"Mood: {self.current_mood}")
        print(f"\n\nMOOD: {mood_answer}\n\n")

        mood_path = f"./personality/{mood_answer}.json"

        # Once the mood has been determined, switch to the file containing that mood profile, ex. './personality/neutral.json'
        if os.path.exists(mood_path):
            with open(mood_path, 'r', encoding='utf-8') as f:
                mood_data = json.load(f)
       
            # System prompt that is returned and used by Assistant LLM
            new_mood = f"You are Joi from the movie Blade Runner 2049. Respond conversationally and keep your answers short unless asked otherwise. Ask questions rarely. {mood_data['personality']} {mood_data['scenario']} {mood_data['dialogue']} Remember, ask questions rarely."
            self.messages[0] = {"role": "system", "content": new_mood}
        else:
            print(f"\nNo file found for {mood_answer} using default\n")
        


    def on_close(self):
        # Make sure we stop the mic thread
        self.is_recording = False

        # Enqueue sentinel for audio playback thread
        self.audio_queue.put(None)

        # On exit summarize any new messages since previous summary 
        if self.message_count > 0:
            checkpoint = self.memory.checkpoint_conversation(self.client, self.messages[-self.message_count:])
            self.memory.store_memory(checkpoint)
            self.memory.save_memory()

        self.destroy()

############################
# 5) Main                  #
############################

def main():
    # Load GPT-SoVITS TTS
    with open("GPT_SoVITS/configs/joi.yaml", "r") as f:
        full_config = yaml.load(f, Loader=yaml.FullLoader)
    tts_config = GPTSoVITSConfig(full_config)
    tts_pipeline = GPTSoVITSPipeline(tts_config)

    # Startup tune
    pygame.mixer.init()
    pygame.mixer.music.load("./assets/audio/on_button.wav")
    pygame.mixer.music.play()

    # Create audio queue + thread
    audio_queue = queue.Queue(maxsize=100)
    playback_thread = threading.Thread(
        target=audio_playback_thread, args=(audio_queue, 32000), daemon=True
    )
    playback_thread.start()

    # Create OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Load whisper model
    whisper_model = load_whisper_model(model_size="base")

    # Initialize memory manager for RAG
    memory = MemoryManager()

    # Start the GUI
    app = ChatApp(tts_pipeline, client, audio_queue, whisper_model, memory)
    app.mainloop()

    # Clean up
    audio_queue.put(None)
    playback_thread.join()

if __name__ == "__main__":
    main()
