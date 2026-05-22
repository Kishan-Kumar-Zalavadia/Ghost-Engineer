from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

from backend.main import app


@pytest.mark.asyncio
async def test_health_check_db_up():
    with patch("backend.main.database.ping", new_callable=AsyncMock, return_value=True), \
         patch("backend.main.database.connect", new_callable=AsyncMock), \
         patch("backend.main.database.disconnect", new_callable=AsyncMock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"


@pytest.mark.asyncio
async def test_health_check_db_down():
    with patch("backend.main.database.ping", new_callable=AsyncMock, return_value=False), \
         patch("backend.main.database.connect", new_callable=AsyncMock), \
         patch("backend.main.database.disconnect", new_callable=AsyncMock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["database"] == "unreachable"
