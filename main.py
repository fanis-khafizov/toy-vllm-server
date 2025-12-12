from __future__ import annotations

import asyncio
import inspect

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

MODEL_PATH = "/data/users/fanis/models/Qwen3-32B/"


class LLMEngineSingleton:
    _engine: AsyncLLMEngine | None = None
    _lock: asyncio.Lock | None = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    async def init(cls) -> AsyncLLMEngine:
        if cls._engine is not None:
            return cls._engine

        async with cls._get_lock():
            if cls._engine is None:
                engine_args = AsyncEngineArgs(
                    model=MODEL_PATH,
                    tensor_parallel_size=1,
                    dtype="auto",
                    trust_remote_code=True,
                )
                cls._engine = AsyncLLMEngine.from_engine_args(engine_args)

        return cls._engine

    @classmethod
    def get(cls) -> AsyncLLMEngine:
        if cls._engine is None:
            raise RuntimeError("LLM engine is not initialized")
        return cls._engine

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._engine is not None

    @classmethod
    async def shutdown(cls) -> None:
        engine = cls._engine
        cls._engine = None
        if engine is None:
            return

        shutdown_fn = getattr(engine, "shutdown", None)
        if callable(shutdown_fn):
            if inspect.iscoroutinefunction(shutdown_fn):
                await shutdown_fn()
            else:
                shutdown_fn()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация и очистка ресурсов"""
    # Загрузка модели при запуске
    print("Загрузка модели...")
    await LLMEngineSingleton.init()
    print("Модель загружена!")
    
    yield
    
    # Очистка при завершении
    print("Завершение работы...")
    await LLMEngineSingleton.shutdown()

app = FastAPI(
    title="vLLM Server",
    description="Простой сервер для генерации текста с помощью vLLM",
    version="0.1.0",
    lifespan=lifespan
)

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

class PromptResponse(BaseModel):
    answer: str
    prompt: str

@app.get("/")
async def root():
    return {
        "message": "vLLM Server работает",
        "endpoints": {
            "health": "/health",
            "generate": "/generate (POST)"
        }
    }

@app.get("/health")
async def health():
    """Проверка состояния сервера"""
    return {
        "status": "healthy",
        "model_loaded": LLMEngineSingleton.is_initialized(),
    }

@app.post("/generate", response_model=PromptResponse)
async def generate(request: PromptRequest):
    """Генерация ответа на основе prompt"""
    if not LLMEngineSingleton.is_initialized():
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    try:
        llm_engine = LLMEngineSingleton.get()
        # Параметры генерации
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens
        )
        
        # Асинхронная генерация
        request_id = f"request-{id(request)}"
        results_generator = llm_engine.generate(
            request.prompt,
            sampling_params,
            request_id
        )
        
        # Получение финального результата
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        # Извлечение результата
        generated_text = final_output.outputs[0].text
        
        return PromptResponse(
            answer=generated_text,
            prompt=request.prompt
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)