"""The public request-serving API: model calls and model discovery."""

from fastapi import APIRouter

data_plane_router = APIRouter(tags=["data-plane"])

