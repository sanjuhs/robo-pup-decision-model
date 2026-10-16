from pathlib import Path

from fastapi.testclient import TestClient

from robopup.server import app
from robopup.speech import transcribe_wav


AUDIO = Path(__file__).parent / "assets/fetch-ball.wav"


def test_real_offline_recognizer_on_synthetic_fixture():
    result = transcribe_wav(AUDIO.read_bytes())
    assert result["text"] == "fetch the ball"
    assert result["requires_send"] is True
    assert 1 < result["duration_s"] < 2


def test_audio_endpoint_rejects_invalid_and_cross_origin_input():
    with TestClient(app) as client:
        assert client.post("/api/transcribe", content=b"bad").status_code == 400
        assert client.post("/api/transcribe", content=b"x" * 300001).status_code == 413
        assert (
            client.post(
                "/api/transcribe",
                content=AUDIO.read_bytes(),
                headers={"origin": "https://unrelated.example"},
            ).status_code
            == 403
        )
        result = client.post("/api/transcribe", content=AUDIO.read_bytes())
        assert result.status_code == 200 and result.json()["text"] == "fetch the ball"
