"""Model transcription."""

import logging
from collections.abc import AsyncIterable

from .const import Settings, TranscribingError
from .models import Model, ModelType
from .transcribe_coqui_stt import transcribe_coqui_stt
from .transcribe_kaldi import transcribe_kaldi

_LOGGER = logging.getLogger(__name__)


async def transcribe(
    model: Model, settings: Settings, audio_stream: AsyncIterable[bytes]
) -> str:
    """Transcribe text from an audio stream."""
    if model.type == ModelType.KALDI:
        return await transcribe_kaldi(model, settings, audio_stream)

    if model.type == ModelType.COQUI_STT:
        return await transcribe_coqui_stt(model, settings, audio_stream)

    raise TranscribingError(f"Unexpected model type for {model.id}: {model.type}")
