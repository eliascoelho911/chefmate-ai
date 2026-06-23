from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.startup import init_dependencies
from app.api.recipes import router as recipes_router
from app.middleware.security import APIKeyMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_dependencies()
    yield


app = FastAPI(title="Chefmate AI", lifespan=lifespan)

# Enforce API key on all requests (healthcheck is exempt)
app.add_middleware(APIKeyMiddleware)

app.include_router(recipes_router, prefix="/recipes")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "Welcome to the Chefmate AI API!"}
