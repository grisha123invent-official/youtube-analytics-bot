# Бот-аналитик для YouTube каналов

Телеграм-бот для анализа YouTube-каналов, который помогает создателям контента оптимизировать свою стратегию и генерировать идеи для нового контента.

## Возможности

- Анализ статистики видео (просмотры, лайки, комментарии)
- Отдельный анализ обычных видео и Shorts
- Определение лучших дней для публикации
- Анализ тегов и тем контента
- Кластеризация контента для выявления популярных тем
- Генерация плана контента с использованием GPT-4o и o1-mini
- Генерация идей для новых видео и Shorts
- Оценка пользовательских идей для видео по шкале от 1 до 10
- Анализ трендов в выбранной нише с использованием Perplexity API
- Интеграция с Telegram для удобного использования

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/username/youtube-analytics-bot.git
cd youtube-analytics-bot
```

2. Создайте и активируйте виртуальное окружение:
```bash
python -m venv .venv
# Для Windows
.venv\Scripts\activate
# Для Linux/Mac
source .venv/bin/activate
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Создайте файл .env и заполните его необходимыми API-ключами:
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
YOUTUBE_API_KEY=your_youtube_api_key
OPENAI_API_KEY=your_openai_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key
```

## Использование

1. Запустите бота:
```bash
python main.py
```

2. Откройте Telegram и найдите своего бота
3. Отправьте команду /start для начала работы
4. Отправьте ссылку на YouTube-канал для анализа
5. Используйте кнопки для генерации идей для видео или Shorts
6. Отправьте боту свою идею для оценки

## Функции бота

- **Анализ канала**: Отправьте ссылку на канал для получения полного анализа
- **Генерация идей для видео**: Получите уникальные идеи для новых видео
- **Генерация идей для Shorts**: Получите идеи для коротких вертикальных видео
- **Оценка идей**: Отправьте свою идею, чтобы получить оценку и рекомендации по улучшению
- **Анализ трендов**: Узнайте о текущих трендах в выбранной нише

## Технологии

- Python 3.8+
- Telebot (PyTelegramBotAPI)
- YouTube Data API v3
- OpenAI API (GPT-4o и o1-mini)
- Perplexity API (для расширенного анализа трендов с моделью sonar-pro)
- Pandas и NumPy для анализа данных
- Scikit-learn для кластеризации контента
- Многопоточность для асинхронной генерации контента

## Требования

См. файл requirements.txt для полного списка зависимостей. 