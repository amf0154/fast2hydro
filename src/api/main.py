import fastapi

from .routers import holtwinter_router

app = fastapi.FastAPI()

app.include_router(holtwinter_router.router, prefix="/holtwinter", tags=["Holt-Winter Model"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
