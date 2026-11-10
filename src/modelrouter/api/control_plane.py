"""The authenticated administrative API: identity, tenancy, and configuration."""

from fastapi import APIRouter

control_plane_router = APIRouter(tags=["control-plane"])

