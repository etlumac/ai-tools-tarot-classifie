import random
import time
from locust import HttpUser, task, between, events

TAROT_QUERIES = [
    "Бывший ко мне вернется?",
    "Что меня ждет в карьере в этом месяце?",
    "Стоит ли начинать новые отношения?",
    "Как улучшить здоровье?",
    "Почему я чувствую одиночество?",
    "Выпала карта Колесница, что это значит?",
    "Стоит ли менять работу?",
    "Когда я встречу свою любовь?",
    "Как наладить отношения с партнером?",
    "Что мне делать дальше?",
    "Будет ли повышение зарплаты?",
    "Почему всё идет не по плану?",
    "Как найти своё предназначение?",
    "Стоит ли доверять этому человеку?",
    "Что ждет меня в любви?",
]


class TarotClassifierUser(HttpUser):
    wait_time = between(0.5, 2.0)

    request_timeout = 30.0

    @task
    def predict_single(self):
        """Основной сценарий: один текстовый запрос"""
        text = random.choice(TAROT_QUERIES)

        with self.client.post(
                "/predict",
                json={"text": text},
                headers={"Content-Type": "application/json"},
                catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "class_id" in data and "probabilities" in data:
                    if len(data["probabilities"]) == 5:
                        response.success()
                    else:
                        response.failure(f"Неверное количество классов: {len(data['probabilities'])}")
                else:
                    response.failure(f"Неверная структура ответа: {data.keys()}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")



@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    stats = environment.stats
    print(f"\nИтоги:")
    print(f"   Всего запросов: {stats.total.num_requests}")
    print(f"   Ошибок: {stats.total.num_failures}")
    print(f"   Среднее время: {stats.total.avg_response_time:.0f} мс")
    print(f"   95-й перцентиль: {stats.total.get_response_time_percentile(0.95):.0f} мс")
    print(f"   RPS: {stats.total.current_rps:.2f}")