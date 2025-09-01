"""Hume speech-to-speech service implementation.
This module provides a speech-to-speech LLM service using Hume EVI 3.
"""

import asyncio
import base64
import datetime
import os
import numpy as np
from hume import MicrophoneInterface, Stream
from hume.client import AsyncHumeClient
from hume.empathic_voice.chat.socket_client import ChatConnectOptions
from hume.empathic_voice.chat.types import SubscribeEvent
import sounddevice as sd

def extract_top_n_emotions(emotion_scores: dict, n: int) -> dict:
    sorted_emotions = sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)
    top_n_emotions = {emotion: score for emotion, score in sorted_emotions[:n]}
    return top_n_emotions

def print_emotions(emotion_scores: dict) -> None:
    print(' | '.join([f"{emotion} ({score:.2f})" for emotion, score in emotion_scores.items()]))

def log(text: str) -> None:
    now = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%H:%M:%S")
    print(f"[{now}] {text}")


class HumeEVIService:
	"""
    Wrapper for Hume Empathic Voice Interface (EVI, speech-to-speech).
	"""

	def __init__(self, HUME_API_KEY, HUME_CONFIG_ID, allow_user_interrupt=False):
		self.HUME_API_KEY = HUME_API_KEY
		self.HUME_CONFIG_ID = HUME_CONFIG_ID
		self.allow_user_interrupt = allow_user_interrupt
		self.client = AsyncHumeClient(api_key=self.HUME_API_KEY)

	async def on_message(self, message: SubscribeEvent):
		if message.type == "chat_metadata":
			log(f"<{message.type}> Chat ID: {message.chat_id}, Chat Group ID: {message.chat_group_id}")
   
		elif message.type == "user_message" or message.type == "assistant_message":
			log(f"{message.message.role}: {message.message.content}")
			print_emotions(extract_top_n_emotions(dict(message.models.prosody and message.models.prosody.scores or {}), 3))
   
		elif message.type == "audio_output":
			await self.stream.put(base64.b64decode(message.data.encode("utf-8")))
			audio_bytes = base64.b64decode(message.data.encode("utf-8"))
			audio_array = np.frombuffer(audio_bytes, dtype=np.int16)  # Hume sends PCM16
			sd.play(audio_array, samplerate=16000)  # match Hume’s default sample rate
   
		elif message.type == "error":
			raise RuntimeError(f"Received error message from Hume websocket ({message.code}): {message.message}")
   
		else:
			log(f"<{message.type}>")

	async def run(self):
		self.stream = Stream.new()
    
		async with self.client.empathic_voice.chat.connect_with_callbacks(
			options=ChatConnectOptions(config_id=self.HUME_CONFIG_ID),
			on_open=lambda: print("WebSocket connection opened."),
			on_message=self.on_message,
			on_close=lambda: print("WebSocket connection closed."),
			on_error=lambda err: print(f"Error: {err}")
		) as socket:
			await MicrophoneInterface.start(
				socket,
				allow_user_interrupt=self.allow_user_interrupt,
				byte_stream=self.stream
        	)