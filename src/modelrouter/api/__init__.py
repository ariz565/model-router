"""Explicit HTTP-plane composition for ModelRouter deployments.

The route modules own deployment boundaries; :mod:`modelrouter.server` owns
the endpoint handlers and service lifecycle. Keeping those concerns separate
lets the same handlers run in a unified local app or in isolated production
gateway, control-plane, and worker processes.
"""

