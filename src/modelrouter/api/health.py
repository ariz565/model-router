"""Liveness and readiness routes shared by every deployment plane."""

from fastapi import APIRouter

health_router = APIRouter(tags=["health"])

