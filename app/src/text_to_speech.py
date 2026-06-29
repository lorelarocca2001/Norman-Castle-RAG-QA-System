# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:11:15 2026

@author: loren
"""
import re
import asyncio
from uuid import uuid4

import edge_tts

from src.config import OUTPUT_DIR


EDGE_TTS_VOICE = "it-IT-IsabellaNeural"
EDGE_TTS_RATE = "+0%"
EDGE_TTS_VOLUME = "+0%"


def clean_text_for_speech(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"`+", "", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"[-•]\s+", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


async def synthesize_speech_async(text):
    clean_text = clean_text_for_speech(text)

    output_path = OUTPUT_DIR / f"answer_{uuid4().hex}.mp3"

    communicate = edge_tts.Communicate(
        text=clean_text,
        voice=EDGE_TTS_VOICE,
        rate=EDGE_TTS_RATE,
        volume=EDGE_TTS_VOLUME
    )

    await communicate.save(str(output_path))

    return output_path


def synthesize_speech(text):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(synthesize_speech_async(text))

    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(synthesize_speech_async(text))
    finally:
        new_loop.close()