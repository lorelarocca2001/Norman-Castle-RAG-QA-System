# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:10:55 2026

@author: loren
"""

from src.loader import load_whisper_model


def transcribe_audio(
    audio_path,
    whisper_model_size="small",
    whisper_language="it"
):
    whisper_model = load_whisper_model(whisper_model_size)

    segments, info = whisper_model.transcribe(
        audio_path,
        language=whisper_language,
        task="transcribe",
        beam_size=1,
        best_of=1,
        vad_filter=True,
        condition_on_previous_text=False,
        temperature=0.0
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    )

    return text.strip()