"""ASGI entrypoint for public ModelRouter gateway deployments."""

from modelrouter.server import create_gateway_app

app = create_gateway_app()

