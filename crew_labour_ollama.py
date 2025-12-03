import requests
from crewai.llms.base_llm import BaseLLM
from common import create_agents, create_tasks, build_crew


class OllamaCustomLLM(BaseLLM):
    def __init__(self, model="deepseek-r1:8b", base_url="http://localhost:11434"):
        super().__init__(model=model)
        self.base_url = base_url

    def call(self, messages, **kwargs):
        prompt = messages[-1].get("content", "") if isinstance(messages, list) else str(messages)
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json().get("response", "")


def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200
    except:
        return False


if not check_ollama():
    raise RuntimeError("Ollama недоступен. Запустите Ollama и убедитесь, что модель загружена.")

llm = OllamaCustomLLM()

researcher, analyst = create_agents(llm)
research_task, report_task = create_tasks(researcher, analyst)
crew = build_crew(researcher, analyst, research_task, report_task)

print("🚀 Запуск анализа через локальную модель Ollama...")
try:
    result = crew.kickoff()
    print(result)
except Exception as e:
    print(f"❌ Ошибка выполнения задачи: {e}")
