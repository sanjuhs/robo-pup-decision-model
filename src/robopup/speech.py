"""Offline, finite-vocabulary command transcription. Audio is never persisted."""

import io
import wave

from pocketsphinx import Decoder

VOICE_COMMANDS = (
    "stop",
    "stay",
    "go to the ball",
    "go behind the ball",
    "fetch the ball",
    "bring the ball home",
    "follow me",
    "follow the beacon",
    "push the crate",
    "push the box",
    "walk forward",
    "run forward",
    "jump",
    "hop",
    "rest",
    "take a break",
    "drop the ball",
    "release the ball",
)


def transcribe_wav(content: bytes):
    if len(content) > 300_000:
        raise ValueError("Record eight seconds or less")
    try:
        with wave.open(io.BytesIO(content), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
                raise ValueError("Expected mono 16 kHz, 16-bit PCM WAV")
            if not 1600 <= audio.getnframes() <= 128000:
                raise ValueError("Record between 0.1 and eight seconds")
            raw = audio.readframes(audio.getnframes())
    except (wave.Error, EOFError) as error:
        raise ValueError("Invalid WAV recording") from error
    decoder = Decoder(samprate=16000, lm=None, loglevel="ERROR")
    grammar = (
        "#JSGF V1.0; grammar commands; public <command> = [please] ("
        + " | ".join(VOICE_COMMANDS)
        + ") [please];"
    )
    decoder.set_jsgf_string("commands", grammar)
    decoder.activate_search("commands")
    decoder.start_utt()
    decoder.process_raw(raw, False, True)
    decoder.end_utt()
    hypothesis = decoder.hyp()
    if hypothesis is None or not hypothesis.hypstr.strip():
        raise ValueError("No command recognized; try again or type it")
    return {
        "text": hypothesis.hypstr,
        "engine": "PocketSphinx offline English command grammar",
        "duration_s": len(raw) / 32000,
        "requires_send": True,
    }
