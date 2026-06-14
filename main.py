from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.startup import init_dependencies
from app.api.chat import router as chat_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_dependencies()
    yield

app = FastAPI(title="Chefmate AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/chat")

@app.get("/")
def root():
    return {"message": "Welcome to the Chefmate AI API!"}