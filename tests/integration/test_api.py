from fastapi.testclient import TestClient

from voice_twin.api.server import app


def test_health():
    client=TestClient(app)
    assert client.get("/health").json()["ok"] is True
