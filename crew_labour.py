import os
from dotenv import load_dotenv
import requests
from common import create_agents, create_tasks, build_crew
from datetime import datetime

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY не найден в .env")

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["OPENAI_API_BASE"] = openai_api_base

print("🔍 Определение доступного LLM...")


def check_ollama_available():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


use_ollama = check_ollama_available()

if use_ollama:
    print("✅ Обнаружен Ollama — используется локальная модель")
    from crewai.llms.base_llm import BaseLLM


    class OllamaCustomLLM(BaseLLM):
        def __init__(self, model="deepseek-r1:8b", base_url="http://localhost:11434"):
            super().__init__(model=model)
            self.base_url = base_url

        def call(self, messages, **kwargs):
            prompt = messages[-1].get("content", "") if isinstance(messages, list) else str(messages)
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=600
                )
                response.raise_for_status()
                return response.json().get("response", "")
            except Exception as e:
                raise RuntimeError(f"Ошибка Ollama: {e}")


    llm = OllamaCustomLLM()
else:
    print("ℹ️ Ollama не найден — используется OpenAI API")
    llm = None  # CrewAI возьмет OpenAI из окружения

# Создаем агентов с нужным LLM
researcher, analyst = create_agents(llm)
research_task, report_task = create_tasks(researcher, analyst)
crew = build_crew(researcher, analyst, research_task, report_task)

print(f"🚀 Запуск анализа востребованных ИТ-навыков через {'Ollama' if use_ollama else 'OpenAI'}...")

try:
    result = crew.kickoff()

    print("==================================================")
    print("РЕЗУЛЬТАТ АНАЛИЗА:")
    print("==================================================")
    print(result)

    # Сохраняем в файл
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Формируем имя файла с текущей датой и временем
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(RESULTS_DIR, f"result_{timestamp}.md")

    try:
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(str(result))
        print(f"✅ Результат сохранён в файл: {result_file}")
    except Exception as e:
        print(f"❌ Ошибка при сохранении результата: {e}")

except Exception as e:
    print(f"❌ Ошибка выполнения задачи: {e}")
