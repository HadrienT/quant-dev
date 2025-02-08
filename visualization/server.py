from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class MonitoringMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.excluded_paths = [
            "/_stcore/",
            "/_stcore/host-config",
            "/_stcore/health",
            "/_stcore/stream",
        ]

    async def dispatch(self, request, call_next):
        path = request.url.path

        should_block = any(
            path.startswith(excluded) for excluded in self.excluded_paths
        )

        if should_block:

            return Response(
                status_code=204, headers={"X-Request-Blocked": "true"}  # No Content
            )

        response = await call_next(request)
        return response


def run(main_script_path: str) -> None:
    """Run the Streamlit app with custom server configuration."""
    import streamlit.web.bootstrap as bootstrap
    from streamlit.web.server import Server

    class CustomServer(Server):
        def __init__(self, main_script_path, is_hello):
            super().__init__(main_script_path, is_hello)

        def _configure_app(self, app: FastAPI):
            app.add_middleware(MonitoringMiddleware)
            super()._configure_app(app)

    # Override the default Server class
    bootstrap.Server = CustomServer

    # Run the app
    flag_options = {}

    bootstrap.run(
        main_script_path=main_script_path,
        flag_options=flag_options,
        is_hello=False,
        args=[],
    )


if __name__ == "__main__":
    app_script = "app.py"
    run(app_script)
