from crewai import Agent, Task, Crew


def create_agents(llm=None):
    researcher = Agent(
        role="Аналитик рынка труда",
        goal="Найти 5 наиболее востребованных навыков в ИТ-отрасли за 2025–2026 годы",
        backstory="Ты эксперт по анализу рынка труда, специализирующийся на сборе актуальных данных.",
        verbose=True,
        llm=llm
    )

    analyst = Agent(
        role="Автор отчётов",
        goal="Составить краткий отчёт о востребованных навыках в ИТ на основе данных",
        backstory="Ты опытный писатель, умеющий обобщать данные в лаконичные отчёты.",
        verbose=True,
        llm=llm
    )

    return researcher, analyst


def create_tasks(researcher, analyst):
    research_task = Task(
        description="На основе твоих знаний о рынке труда ИТ-отрасли, найди 5 наиболее востребованных навыков за 2025–2026 годы. Укажи краткое описание каждого навыка.",
        expected_output="Список из 5 навыков с описанием (1–2 предложения на навык).",
        agent=researcher
    )

    report_task = Task(
        description="На основе данных от Исследователя составь отчёт на 100–150 слов о востребованных навыках в ИТ-отрасли.",
        expected_output="Отчёт в формате markdown (100–150 слов).",
        agent=analyst
    )

    return research_task, report_task


def build_crew(researcher, analyst, research_task, report_task, verbose=True):
    return Crew(
        agents=[researcher, analyst],
        tasks=[research_task, report_task],
        verbose=verbose,
        telemetry_enabled=False
    )
