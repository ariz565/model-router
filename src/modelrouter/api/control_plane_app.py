"""ASGI entrypoint for private ModelRouter control-plane deployments."""

from modelrouter.server import create_control_plane_app

app = create_control_plane_app()

