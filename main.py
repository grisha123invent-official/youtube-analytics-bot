import os
import json
import telebot
from telebot import types
import threading
import time
from datetime import datetime, timedelta
import googleapiclient.discovery
import googleapiclient.errors
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройки API ключей
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")


# Класс для анализа YouTube-каналов
class YouTubeAnalyticsBot:
    def __init__(self, youtube_api_key, openai_api_key, perplexity_api_key=None):
        """
        Инициализация бота с ключами API.

        Args:
            youtube_api_key (str): API-ключ YouTube
            openai_api_key (str): API-ключ OpenAI для GPT
            perplexity_api_key (str, optional): API-ключ Perplexity
        """
        self.youtube_api_key = youtube_api_key
        self.openai_api_key = openai_api_key
        self.perplexity_api_key = perplexity_api_key

        # Инициализация клиента YouTube API
        self.youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=youtube_api_key
        )

    def _is_shorts(self, video_info):
        """
        Определяет, является ли видео Shorts.
        
        Args:
            video_info (dict): Информация о видео
            
        Returns:
            bool: True если это Shorts, False в противном случае
        """
        # Проверка по нескольким критериям:
        # 1. Вертикальный формат (соотношение сторон)
        # 2. Короткая продолжительность (до 60 секунд)
        # 3. Хештег #shorts в названии или описании
        
        if video_info.get("duration_seconds", 0) <= 60:
            if (video_info.get("width", 0) < video_info.get("height", 0)) or \
               "#shorts" in video_info.get("title", "").lower() or \
               "#shorts" in video_info.get("description", "").lower():
                return True
        
        return False

    def get_channel_videos(self, channel_id, max_results=50):
        """
        Получение списка видео с канала.

        Args:
            channel_id (str): ID YouTube-канала
            max_results (int): Максимальное количество получаемых видео

        Returns:
            list: Список словарей с информацией о видео
        """
        # Сначала получаем ID загруженных видео
        uploads_id = self._get_uploads_playlist_id(channel_id)

        if not uploads_id:
            return {"error": f"Не удалось найти канал с ID: {channel_id}"}

        # Затем получаем видео из плейлиста загрузок
        videos = []
        request = self.youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads_id,
            maxResults=max_results
        )

        while request:
            response = request.execute()

            for item in response.get("items", []):
                video_id = item["contentDetails"]["videoId"]
                video_title = item["snippet"]["title"]
                published_at = item["snippet"]["publishedAt"]

                # Получаем дополнительную информацию о видео
                video_info = self._get_video_details(video_id)

                if video_info:
                    is_shorts = self._is_shorts(video_info)
                    videos.append({
                        "id": video_id,
                        "title": video_title,
                        "published_at": published_at,
                        "views": video_info.get("views", 0),
                        "likes": video_info.get("likes", 0),
                        "comments": video_info.get("comments", 0),
                        "description": video_info.get("description", ""),
                        "tags": video_info.get("tags", []),
                        "duration_seconds": video_info.get("duration_seconds", 0),
                        "width": video_info.get("width", 0),
                        "height": video_info.get("height", 0),
                        "is_shorts": is_shorts
                    })

            # Проверяем, есть ли следующая страница результатов
            request = self.youtube.playlistItems().list_next(request, response)

        return videos

    def _get_uploads_playlist_id(self, channel_id):
        """
        Получение ID плейлиста загрузок канала.

        Args:
            channel_id (str): ID YouTube-канала

        Returns:
            str: ID плейлиста загрузок
        """
        try:
            request = self.youtube.channels().list(
                part="contentDetails",
                id=channel_id
            )
            response = request.execute()

            if "items" in response and len(response["items"]) > 0:
                return response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

            return None
        except Exception as e:
            print(f"Ошибка при получении ID плейлиста: {e}")
            return None

    def _get_video_details(self, video_id):
        """
        Получение детальной информации о видео.

        Args:
            video_id (str): ID видео

        Returns:
            dict: Словарь с информацией о видео
        """
        try:
            request = self.youtube.videos().list(
                part="statistics,snippet,contentDetails",
                id=video_id
            )
            response = request.execute()

            if "items" in response and len(response["items"]) > 0:
                item = response["items"][0]
                
                # Получение длительности видео в ISO 8601 формате (PT#M#S)
                duration_iso = item["contentDetails"].get("duration", "PT0S")
                duration_seconds = self._parse_duration(duration_iso)
                
                # Получение соотношения сторон и размеров
                width = 0
                height = 0
                if "maxres" in item["snippet"].get("thumbnails", {}):
                    width = item["snippet"]["thumbnails"]["maxres"].get("width", 0)
                    height = item["snippet"]["thumbnails"]["maxres"].get("height", 0)
                
                return {
                    "views": int(item["statistics"].get("viewCount", 0)),
                    "likes": int(item["statistics"].get("likeCount", 0)),
                    "comments": int(item["statistics"].get("commentCount", 0)),
                    "description": item["snippet"].get("description", ""),
                    "tags": item["snippet"].get("tags", []),
                    "duration_seconds": duration_seconds,
                    "width": width,
                    "height": height,
                    "title": item["snippet"].get("title", "")
                }

            return {}
        except Exception as e:
            print(f"Ошибка при получении деталей видео: {e}")
            return {}
            
    def _parse_duration(self, duration_iso):
        """
        Парсинг ISO 8601 формата длительности видео в секунды.
        
        Args:
            duration_iso (str): Длительность в формате ISO 8601 (например, PT1H2M3S)
            
        Returns:
            int: Длительность в секундах
        """
        duration_seconds = 0
        
        # Удаляем префикс PT
        if duration_iso.startswith("PT"):
            duration_iso = duration_iso[2:]
        
        # Поиск часов, минут и секунд
        hours = 0
        minutes = 0
        seconds = 0
        
        h_pos = duration_iso.find("H")
        if h_pos != -1:
            hours = int(duration_iso[:h_pos])
            duration_iso = duration_iso[h_pos+1:]
            
        m_pos = duration_iso.find("M")
        if m_pos != -1:
            minutes = int(duration_iso[:m_pos])
            duration_iso = duration_iso[m_pos+1:]
            
        s_pos = duration_iso.find("S")
        if s_pos != -1:
            seconds = int(duration_iso[:s_pos])
            
        # Рассчитываем общую длительность в секундах
        duration_seconds = hours * 3600 + minutes * 60 + seconds
        
        return duration_seconds

    def _get_channel_info(self, channel_id):
        """
        Получение основной информации о канале.
        
        Args:
            channel_id (str): ID YouTube-канала
            
        Returns:
            dict: Информация о канале (название, описание, статистика)
        """
        try:
            request = self.youtube.channels().list(
                part="snippet,statistics,contentDetails",
                id=channel_id
            )
            response = request.execute()
            
            if "items" in response and len(response["items"]) > 0:
                channel = response["items"][0]
                
                return {
                    "id": channel_id,
                    "title": channel["snippet"].get("title", ""),
                    "description": channel["snippet"].get("description", ""),
                    "published_at": channel["snippet"].get("publishedAt", ""),
                    "subscriber_count": int(channel["statistics"].get("subscriberCount", 0)),
                    "video_count": int(channel["statistics"].get("videoCount", 0)),
                    "view_count": int(channel["statistics"].get("viewCount", 0)),
                    "thumbnail": channel["snippet"].get("thumbnails", {}).get("high", {}).get("url", "")
                }
            
            return None
        except Exception as e:
            print(f"Ошибка при получении информации о канале: {e}")
            return None

    def get_last_week_videos(self, channel_id, max_results=50):
        """
        Получение видео за последнюю неделю.

        Args:
            channel_id (str): ID YouTube-канала
            max_results (int): Максимальное количество получаемых видео

        Returns:
            list: Список словарей с информацией о видео за последнюю неделю
        """
        videos = self.get_channel_videos(channel_id, max_results)

        if isinstance(videos, dict) and "error" in videos:
            return videos

        # Фильтруем видео, опубликованные за последнюю неделю
        one_week_ago = datetime.now() - timedelta(days=7)

        last_week_videos = []
        for video in videos:
            published_at = datetime.strptime(
                video["published_at"], "%Y-%m-%dT%H:%M:%SZ"
            )
            if published_at >= one_week_ago:
                last_week_videos.append(video)

        return last_week_videos

    def analyze_videos(self, videos):
        """
        Анализ видео и их показателей.

        Args:
            videos (list): Список словарей с информацией о видео

        Returns:
            dict: Результаты анализа
        """
        if isinstance(videos, dict) and "error" in videos:
            return videos

        if not videos:
            return {"error": "Нет видео для анализа"}

        # Создаем DataFrame для анализа
        df = pd.DataFrame(videos)

        # Конвертируем строки даты в datetime
        df["published_at"] = pd.to_datetime(df["published_at"])
        
        # Разделяем на обычные видео и shorts
        shorts_df = df[df["is_shorts"] == True] if "is_shorts" in df.columns else pd.DataFrame()
        regular_df = df[df["is_shorts"] == False] if "is_shorts" in df.columns else df

        # Рассчитываем общую статистику
        stats = {
            "total_videos": len(df),
            "total_shorts": len(shorts_df),
            "total_regular_videos": len(regular_df),
            "avg_views": df["views"].mean(),
            "avg_likes": df["likes"].mean(),
            "avg_comments": df["comments"].mean(),
            "engagement_rate": (df["likes"].sum() + df["comments"].sum()) / max(df["views"].sum(), 1) * 100,
            "best_video": df.loc[df["views"].idxmax()]["title"] if len(df) > 0 else "Нет данных",
            "best_video_views": df["views"].max() if len(df) > 0 else 0,
            "best_days": self._get_best_publishing_days(df)
        }
        
        # Статистика по Shorts
        if len(shorts_df) > 0:
            stats["shorts_stats"] = {
                "avg_views": shorts_df["views"].mean(),
                "avg_likes": shorts_df["likes"].mean(),
                "avg_comments": shorts_df["comments"].mean(),
                "engagement_rate": (shorts_df["likes"].sum() + shorts_df["comments"].sum()) / max(shorts_df["views"].sum(), 1) * 100,
                "best_shorts": shorts_df.loc[shorts_df["views"].idxmax()]["title"] if len(shorts_df) > 0 else "Нет данных",
                "best_shorts_views": shorts_df["views"].max() if len(shorts_df) > 0 else 0
            }
        else:
            stats["shorts_stats"] = {"message": "Нет Shorts для анализа"}
            
        # Статистика по обычным видео
        if len(regular_df) > 0:
            stats["regular_videos_stats"] = {
                "avg_views": regular_df["views"].mean(),
                "avg_likes": regular_df["likes"].mean(),
                "avg_comments": regular_df["comments"].mean(),
                "engagement_rate": (regular_df["likes"].sum() + regular_df["comments"].sum()) / max(regular_df["views"].sum(), 1) * 100,
                "best_regular_video": regular_df.loc[regular_df["views"].idxmax()]["title"] if len(regular_df) > 0 else "Нет данных",
                "best_regular_video_views": regular_df["views"].max() if len(regular_df) > 0 else 0
            }
        else:
            stats["regular_videos_stats"] = {"message": "Нет обычных видео для анализа"}

        # Анализируем теги и темы
        popular_tags = self._analyze_tags(df)
        topics = self._cluster_content(df)
        
        # Генерируем рекомендации по улучшению статистики
        recommendations = self._generate_improvement_recommendations(df, shorts_df, regular_df, stats, popular_tags)

        return {
            "stats": stats,
            "popular_tags": popular_tags,
            "topic_clusters": topics,
            "recommendations": recommendations
        }
        
    def _generate_improvement_recommendations(self, all_df, shorts_df, regular_df, stats, popular_tags):
        """
        Генерирует рекомендации для улучшения показателей канала на основе анализа данных.
        
        Args:
            all_df (DataFrame): Данные по всем видео
            shorts_df (DataFrame): Данные по Shorts
            regular_df (DataFrame): Данные по обычным видео
            stats (dict): Статистика канала
            popular_tags (list): Популярные теги
            
        Returns:
            dict: Рекомендации по улучшению показателей
        """
        recommendations = {}
        
        # Рекомендации по расписанию публикаций
        if stats["best_days"]:
            recommendations["publishing_schedule"] = {
                "message": f"Рекомендуемые дни для публикации: {', '.join(stats['best_days'][:3])}",
                "explanation": "Публикация в эти дни показывает наилучшую вовлеченность аудитории."
            }
        
        # Рекомендации по формату контента
        if len(shorts_df) > 0 and len(regular_df) > 0:
            shorts_engagement = (shorts_df["likes"].sum() + shorts_df["comments"].sum()) / max(shorts_df["views"].sum(), 1) * 100
            regular_engagement = (regular_df["likes"].sum() + regular_df["comments"].sum()) / max(regular_df["views"].sum(), 1) * 100
            
            if shorts_engagement > regular_engagement:
                recommendations["content_format"] = {
                    "message": "Shorts показывают лучшую вовлеченность аудитории",
                    "explanation": f"Коэффициент вовлеченности Shorts ({shorts_engagement:.2f}%) выше, чем у обычных видео ({regular_engagement:.2f}%). Рекомендуется увеличить количество Shorts."
                }
            else:
                recommendations["content_format"] = {
                    "message": "Обычные видео показывают лучшую вовлеченность аудитории",
                    "explanation": f"Коэффициент вовлеченности обычных видео ({regular_engagement:.2f}%) выше, чем у Shorts ({shorts_engagement:.2f}%). Рекомендуется сосредоточиться на качественных полноформатных видео."
                }
        elif len(shorts_df) == 0 and len(regular_df) > 0:
            recommendations["content_format"] = {
                "message": "Рекомендуется добавить Shorts в стратегию контента",
                "explanation": "Shorts могут привлечь новую аудиторию и увеличить охват канала."
            }
        elif len(shorts_df) > 0 and len(regular_df) == 0:
            recommendations["content_format"] = {
                "message": "Рекомендуется добавить обычные видео в стратегию контента",
                "explanation": "Длинные видео могут увеличить время просмотра и улучшить монетизацию."
            }
        
        # Рекомендации по использованию тегов
        if popular_tags and len(popular_tags) > 0:
            recommendations["tags"] = {
                "message": f"Наиболее эффективные теги: {', '.join(popular_tags[:5])}",
                "explanation": "Использование этих тегов может улучшить видимость ваших видео в рекомендациях."
            }
        
        # Рекомендации по улучшению вовлеченности
        if all_df["comments"].mean() < 10:
            recommendations["engagement"] = {
                "message": "Низкое количество комментариев",
                "explanation": "Рекомендуется стимулировать обсуждение в комментариях, задавая вопросы аудитории и реагируя на комментарии."
            }
            
        # Общие рекомендации по улучшению контента
        recommendations["general"] = [
            {
                "title": "Оптимизация заголовков и миниатюр",
                "description": "Используйте яркие, привлекающие внимание миниатюры и информативные заголовки с ключевыми словами."
            },
            {
                "title": "Последовательность публикаций",
                "description": "Поддерживайте регулярный график публикаций для удержания аудитории."
            },
            {
                "title": "Перекрестные ссылки",
                "description": "Ссылайтесь на свои предыдущие видео для увеличения просмотров и времени просмотра."
            },
            {
                "title": "Призывы к действию",
                "description": "Включайте призывы к действию (подписка, лайк, комментарий) в видео для повышения вовлеченности."
            },
            {
                "title": "Анализ конкурентов",
                "description": "Изучайте успешные видео конкурентов в вашей нише для определения эффективных стратегий."
            }
        ]
        
        return recommendations

    def _get_best_publishing_days(self, df):
        """
        Определение лучших дней для публикации.

        Args:
            df (DataFrame): DataFrame с данными о видео

        Returns:
            list: Список дней недели, отсортированных по просмотрам
        """
        if len(df) == 0:
            return []

        # Добавляем день недели публикации
        df["day_of_week"] = df["published_at"].dt.day_name()

        # Группируем по дням недели и считаем среднее число просмотров
        day_stats = df.groupby("day_of_week")["views"].mean().reset_index()

        # Сортируем дни по среднему числу просмотров (по убыванию)
        day_stats = day_stats.sort_values("views", ascending=False)

        return day_stats["day_of_week"].tolist()

    def _analyze_tags(self, df):
        """
        Анализ тегов для выявления популярных.

        Args:
            df (DataFrame): DataFrame с данными о видео

        Returns:
            list: Список тегов (без частоты)
        """
        # Собираем все теги
        all_tags = []
        for tags in df["tags"]:
            if tags:
                all_tags.extend(tags)

        # Считаем частоту тегов
        tag_counter = Counter(all_tags)

        # Получаем 10 самых популярных тегов и возвращаем только имена тегов, без частоты
        popular_tags = [tag for tag, _ in tag_counter.most_common(10)]

        return popular_tags

    def _cluster_content(self, df, n_clusters=5):
        """
        Кластеризация контента на основе заголовков и описаний.

        Args:
            df (DataFrame): DataFrame с данными о видео
            n_clusters (int): Количество кластеров

        Returns:
            list: Список тем с ключевыми словами
        """
        if len(df) == 0:
            return []

        # Объединяем заголовки и описания для анализа
        df["content"] = df["title"] + " " + df["description"]

        # Используем TF-IDF для извлечения признаков
        tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words="english",
            min_df=2
        )

        # Если у нас мало видео, уменьшаем количество кластеров
        if len(df) < n_clusters:
            n_clusters = max(2, len(df) // 2)

        try:
            # Преобразуем тексты в матрицу TF-IDF
            tfidf_matrix = tfidf.fit_transform(df["content"])

            # Кластеризуем контент
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df["cluster"] = kmeans.fit_predict(tfidf_matrix)

            # Для каждого кластера извлекаем ключевые слова
            topics = []
            feature_names = tfidf.get_feature_names_out()

            for i in range(n_clusters):
                # Получаем индексы видео в кластере
                cluster_videos = df[df["cluster"] == i]

                if len(cluster_videos) == 0:
                    continue

                # Вычисляем средние значения TF-IDF для кластера
                cluster_center = kmeans.cluster_centers_[i]

                # Получаем топ-5 ключевых слов
                top_indices = cluster_center.argsort()[-5:][::-1]
                top_keywords = [feature_names[idx] for idx in top_indices]

                # Вычисляем средние показатели для кластера
                avg_views = cluster_videos["views"].mean()

                topics.append({
                    "id": i,
                    "keywords": top_keywords,
                    "avg_views": avg_views,
                    "video_count": len(cluster_videos),
                    "example_titles": cluster_videos["title"].tolist()[:3]
                })

            # Сортируем темы по средним просмотрам
            topics.sort(key=lambda x: x["avg_views"], reverse=True)

            return topics

        except Exception as e:
            print(f"Ошибка кластеризации: {e}")
            return []

    def analyze_with_perplexity(self, niche, keywords):
        """
        Анализ трендов ниши с использованием GPT-4o.

        Args:
            niche (str): Тематика канала
            keywords (list): Список ключевых слов для анализа

        Returns:
            str: Результаты анализа трендов
        """
        try:
            # Если нет ключа API или параметров, возвращаем пустой результат
            if not self.openai_api_key or not niche or not keywords:
                return "Анализ трендов недоступен (не указан API-ключ или параметры)"

            # Формируем запрос на основе тематики и ключевых слов
            prompt = f"""Ты эксперт по анализу трендов на YouTube. Проанализируй текущие тренды и популярные темы в нише "{niche}" на YouTube.

Вот ключевые слова, которые уже используются на канале: {', '.join(keywords[:15])}

Пожалуйста, предоставь следующую информацию:
1. Топ-5 трендовых тем в нише "{niche}" на YouTube в настоящее время
2. Ключевые слова и хэштеги, которые стоит использовать для продвижения контента
3. Форматы видео, которые наиболее популярны у аудитории в этой нише сейчас
4. Рекомендации по оптимальной длительности видео для этой ниши
5. Идеи для коллабораций или интеграции популярных трендов

Основывайся на текущих трендах и предпочтениях аудитории в 2023-2024 годах.
"""

            # Отправляем запрос
            client = OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Ошибка при анализе трендов: {e}")
            return f"Не удалось выполнить анализ трендов: {str(e)}"

    def analyze_with_gpt(self, videos, analysis):
        """
        Анализ данных с использованием GPT для получения рекомендаций.

        Args:
            videos (list): Список видео для анализа
            analysis (dict): Результаты предварительного анализа

        Returns:
            str: Текстовые рекомендации от GPT
        """
        try:
            # Если нет API-ключа, не выполняем анализ
            if not self.openai_api_key:
                return "Анализ с помощью GPT недоступен (не указан API-ключ)"

            # Получаем статистику для формирования промпта
            stats = analysis["stats"]
            popular_tags = analysis["popular_tags"]
            
            # Создаем список с заголовками видео
            video_titles = [video["title"] for video in videos[:10]]
            
            # Формируем запрос
            prompt = f"""Ты - эксперт по анализу YouTube-каналов и стратегии контента. Проанализируй следующие данные о YouTube-канале и предложи конкретные рекомендации для улучшения показателей.

Статистика канала за последнюю неделю:
- Всего видео: {stats['total_videos']}
- Обычных видео: {stats.get('total_regular_videos', 'Нет данных')}
- Shorts: {stats.get('total_shorts', 'Нет данных')}
- Средние просмотры: {stats['avg_views']:.0f}
- Средние лайки: {stats['avg_likes']:.0f}
- Средние комментарии: {stats['avg_comments']:.0f}
- Коэффициент вовлеченности: {stats['engagement_rate']:.2f}%
- Лучшие дни для публикации: {', '.join(stats['best_days'][:3]) if stats['best_days'] else 'Недостаточно данных'}

Последние видео на канале:
{chr(10).join([f"- {title}" for title in video_titles])}

Популярные теги: {', '.join(popular_tags) if popular_tags else 'Нет данных'}

Предложи конкретный план улучшения канала, включая:
1. Рекомендации по форматам контента (Shorts vs обычные видео)
2. Оптимальное расписание публикаций
3. Темы для новых видео (минимум 5 конкретных идей)
4. Советы по улучшению заголовков и описаний
5. Стратегии увеличения вовлеченности аудитории

Давай только конкретные, практические советы без общих фраз.
"""

            # Отправляем запрос
            client = OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Ошибка при анализе с GPT: {e}")
            return "Не удалось выполнить анализ с GPT. Пожалуйста, попробуйте позже."

    def generate_content_plan(self, channel_id, niche, status_callback=None):
        """
        Генерация контент-плана на основе анализа канала.

        Args:
            channel_id (str): ID YouTube-канала
            niche (str): Ниша канала для лучшего анализа
            status_callback (function, optional): Функция обратного вызова для обновления статуса

        Returns:
            dict: Результаты анализа и рекомендации
        """
        try:
            if status_callback:
                status_callback("Получение информации о канале...")

            # Получаем информацию о канале
            channel_info = self._get_channel_info(channel_id)
            channel_title = channel_info.get("title") if channel_info else None

            if status_callback:
                status_callback("Анализ последних видео...")

            # Получаем и анализируем видео
            videos = self.get_last_week_videos(channel_id, max_results=30)
            
            if isinstance(videos, dict) and "error" in videos:
                return videos

            analysis_results = self.analyze_videos(videos)
            
            if isinstance(analysis_results, dict) and "error" in analysis_results:
                return analysis_results

            # Ключевые слова из тегов
            keywords = analysis_results.get("popular_tags", [])[:10]

            if status_callback:
                status_callback("Генерация рекомендаций с помощью ИИ...")

            # Получаем инсайты от GPT
            gpt_insights = self.get_gpt_insights(analysis_results, channel_title)

            # Объединяем все результаты
            content_plan = {
                **analysis_results,
                "channel_info": channel_info,
                "gpt_insights": gpt_insights
            }

            return content_plan

        except Exception as e:
            print(f"Ошибка при создании контент-плана: {e}")
            return {"error": f"Не удалось создать контент-план: {str(e)}"}

    def get_gpt_insights(self, analysis_data, channel_title=None):
        """
        Получение аналитики и рекомендаций от GPT.

        Args:
            analysis_data (dict): Результаты предварительного анализа
            channel_title (str, optional): Название канала

        Returns:
            str: Текстовые рекомендации от GPT
        """
        try:
            # Создаем клиента OpenAI
            client = OpenAI(api_key=self.openai_api_key)

            # Формируем базовый промпт
            channel_info = f"канала '{channel_title}'" if channel_title else "YouTube канала"
            
            # Базовая статистика
            stats = analysis_data.get("stats", {})
            
            # Формируем информацию о статистике для обычных видео и Shorts
            shorts_info = ""
            if "shorts_stats" in stats and isinstance(stats["shorts_stats"], dict) and "message" not in stats["shorts_stats"]:
                shorts = stats["shorts_stats"]
                shorts_info = f"""
Статистика Shorts:
- Количество: {stats.get('total_shorts', 0)}
- Средние просмотры: {shorts.get('avg_views', 0):.2f}
- Средние лайки: {shorts.get('avg_likes', 0):.2f}
- Средние комментарии: {shorts.get('avg_comments', 0):.2f}
- Коэффициент вовлеченности: {shorts.get('engagement_rate', 0):.2f}%
- Лучший Shorts: "{shorts.get('best_shorts', 'Н/Д')}"
"""
            
            regular_videos_info = ""
            if "regular_videos_stats" in stats and isinstance(stats["regular_videos_stats"], dict) and "message" not in stats["regular_videos_stats"]:
                regular = stats["regular_videos_stats"]
                regular_videos_info = f"""
Статистика обычных видео:
- Количество: {stats.get('total_regular_videos', 0)}
- Средние просмотры: {regular.get('avg_views', 0):.2f}
- Средние лайки: {regular.get('avg_likes', 0):.2f}
- Средние комментарии: {regular.get('avg_comments', 0):.2f}
- Коэффициент вовлеченности: {regular.get('engagement_rate', 0):.2f}%
- Лучшее видео: "{regular.get('best_regular_video', 'Н/Д')}"
"""
            
            # Рекомендации
            recommendations_info = ""
            if "recommendations" in analysis_data:
                rec = analysis_data["recommendations"]
                
                publishing_info = ""
                if "publishing_schedule" in rec:
                    publishing_info = f"- {rec['publishing_schedule']['message']}\n"
                
                content_format_info = ""
                if "content_format" in rec:
                    content_format_info = f"- {rec['content_format']['message']}\n"
                
                tags_info = ""
                if "tags" in rec:
                    tags_info = f"- {rec['tags']['message']}\n"
                
                engagement_info = ""
                if "engagement" in rec:
                    engagement_info = f"- {rec['engagement']['message']}\n"
                
                recommendations_info = f"""
Основные рекомендации:
{publishing_info}{content_format_info}{tags_info}{engagement_info}
"""

            # Формируем основной промпт с учетом всех данных
            prompt = f"""Ты - эксперт по аналитике YouTube и стратегии развития каналов. Проанализируй статистику {channel_info} и предложи план развития контента.

Общая статистика канала:
- Всего видео: {stats.get('total_videos', 0)}
- Средние просмотры: {stats.get('avg_views', 0):.2f}
- Средние лайки: {stats.get('avg_likes', 0):.2f}
- Средние комментарии: {stats.get('avg_comments', 0):.2f}
- Коэффициент вовлеченности: {stats.get('engagement_rate', 0):.2f}%
- Лучшее видео: "{stats.get('best_video', 'Н/Д')}"
- Лучшие дни публикации: {', '.join(stats['best_days'][:3]) if stats['best_days'] else 'Недостаточно данных'}

{shorts_info}
{regular_videos_info}
{recommendations_info}

Популярные теги: {', '.join([str(tag) for tag in analysis_data.get('popular_tags', [])][:10]) if analysis_data.get('popular_tags') else 'Нет данных'}

Популярные темы: {', '.join([', '.join(topic.get('keywords', ['Без названия'])[:2]) for topic in analysis_data.get('topic_clusters', [])[:5]]) if analysis_data.get('topic_clusters') else 'Нет данных'}

Задачи:
1. Проанализируй эффективность разных форматов контента (обычные видео и Shorts) и дай конкретные рекомендации по каждому формату
2. Оцени, какие темы и теги показывают лучшие результаты
3. Предложи 5-7 конкретных идей для новых видео, которые могли бы хорошо работать на этом канале
4. Разработай недельный план публикаций с учетом лучших дней и форматов
5. Предложи 3-5 конкретных способов улучшить вовлеченность аудитории

Сосредоточься на практических рекомендациях, которые можно сразу применить.
"""

            # Отправляем запрос
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Ты - эксперт по анализу YouTube-каналов и стратегии контента."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Ошибка при получении рекомендаций от GPT: {e}")
            return "Не удалось получить рекомендации от ИИ. Пожалуйста, попробуйте позже."
            
    def generate_content_ideas(self, channel_info, stats, popular_tags, topic_clusters, existing_titles=None, content_type=None):
        """
        Генерирует идеи для нового контента с использованием модели GPT-4o.
        
        Args:
            channel_info (dict): Информация о канале
            stats (dict): Статистика канала
            popular_tags (list): Популярные теги канала
            topic_clusters (list): Кластеры тем канала
            existing_titles (list, optional): Список существующих заголовков видео
            content_type (str, optional): Тип контента - 'video' или 'shorts'
            
        Returns:
            list: Список идей для контента
        """
        try:
            # Создаем клиента OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            # Определяем тип контента
            content_type_text = ""
            if content_type == "video":
                content_type_text = "обычных видео (не Shorts)"
            elif content_type == "shorts":
                content_type_text = "коротких вертикальных видео Shorts"
            else:
                content_type_text = "видео и Shorts"
                
            # Формируем список существующих заголовков для исключения
            existing_content = ""
            if existing_titles and len(existing_titles) > 0:
                existing_content = "Вот список существующих видео на канале, не повторяй эти темы:\n"
                for i, title in enumerate(existing_titles[:20], 1):
                    existing_content += f"{i}. {title}\n"
            
            # Получаем информацию о канале
            channel_title = channel_info.get("title", "канала") if channel_info else "канала"
                    
            # Формируем запрос на основе статистики канала
            prompt = f"""Ты - эксперт по YouTube-контенту, который генерирует креативные и привлекательные идеи для видео и Shorts. Сгенерируй 15 конкретных и оригинальных идей для {content_type_text} для YouTube-канала "{channel_title}".

Популярные теги на канале: {', '.join(popular_tags) if popular_tags else 'Нет данных'}

Учитывай следующую статистику для создания релевантного контента:
- Лучшее видео на канале: "{stats.get('best_video', 'Нет данных')}"
- Лучшие дни для публикации: {', '.join(stats.get('best_days', [])[:3]) if stats.get('best_days') else 'Нет данных'}

{existing_content}

Каждая идея должна содержать:
1. Привлекательный заголовок
2. Краткое описание (1-2 предложения)
3. Ключевые моменты (3-5 пунктов)

ВАЖНО: 
- Предлагай только новые, оригинальные темы, которые отличаются от уже существующих
- Предлагай актуальные тренды и темы с высоким потенциалом просмотров
- Учитывай специфику формата {content_type_text}
- Формат каждой идеи: "ЗАГОЛОВОК: описание. Ключевые моменты: • пункт1 • пункт2 • пункт3"
- Выдавай ТОЛЬКО список идей БЕЗ дополнительной информации, вступлений и заключений
"""

            # Отправляем запрос к модели GPT-4o
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Ты - эксперт по YouTube-контенту, который генерирует креативные и привлекательные идеи для видео и Shorts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=2000
            )

            # Обрабатываем ответ
            content_ideas_text = response.choices[0].message.content.strip()
            
            # Разбиваем текст на отдельные идеи
            content_ideas = []
            
            # Разделяем по номерам (1., 2., и т.д.) или по новым строкам, если нет номеров
            if any(f"{i}." in content_ideas_text for i in range(1, 16)):
                # Если есть нумерация, разделяем по ней
                import re
                ideas_raw = re.split(r'\d+\.', content_ideas_text)
                # Удаляем пустые элементы
                ideas_raw = [idea.strip() for idea in ideas_raw if idea.strip()]
            else:
                # Иначе разделяем по новым строкам
                ideas_raw = [idea.strip() for idea in content_ideas_text.split('\n\n') if idea.strip()]
            
            # Обрабатываем каждую идею
            for idea in ideas_raw:
                if ":" in idea:
                    parts = idea.split(":", 1)
                    title = parts[0].strip()
                    description = parts[1].strip()
                    
                    # Ищем ключевые моменты
                    key_points = []
                    if "Ключевые моменты" in description:
                        desc_parts = description.split("Ключевые моменты", 1)
                        description = desc_parts[0].strip()
                        
                        # Извлекаем пункты
                        points_text = desc_parts[1].strip().lstrip(":").strip()
                        # Разделяем по маркерам списка или по новым строкам
                        if "•" in points_text:
                            key_points = [point.strip() for point in points_text.split("•") if point.strip()]
                        elif "-" in points_text:
                            key_points = [point.strip() for point in points_text.split("-") if point.strip()]
                        else:
                            key_points = [point.strip() for point in points_text.split("\n") if point.strip()]
                    
                    content_ideas.append({
                        "title": title,
                        "description": description,
                        "key_points": key_points
                    })
                else:
                    # Если нет четкого разделения на заголовок и описание
                    content_ideas.append({
                        "title": idea,
                        "description": "",
                        "key_points": []
                    })
            
            return content_ideas[:15]  # Возвращаем максимум 15 идей

        except Exception as e:
            print(f"Ошибка при генерации идей контента: {e}")
            return [{"title": "Не удалось сгенерировать идеи", "description": str(e), "key_points": []}]
            
    def evaluate_content_idea(self, idea, channel_info, stats, popular_tags, existing_titles=None):
        """
        Оценивает идею контента по шкале от 1 до 10.
        
        Args:
            idea (str): Идея для оценки
            channel_info (dict): Информация о канале
            stats (dict): Статистика канала
            popular_tags (list): Популярные теги
            existing_titles (list, optional): Список существующих заголовков
            
        Returns:
            dict: Результат оценки идеи
        """
        try:
            # Создаем клиента OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            # Формируем список существующих заголовков для проверки оригинальности
            existing_content = ""
            if existing_titles and len(existing_titles) > 0:
                existing_content = "Заголовки существующих видео на канале:\n"
                for i, title in enumerate(existing_titles[:10], 1):
                    existing_content += f"{i}. {title}\n"
            
            # Получаем информацию о канале
            channel_title = channel_info.get("title", "канала") if channel_info else "канала"
            
            # Формируем запрос
            prompt = f"""Ты - эксперт по YouTube-контенту, который оценивает идеи для видео по шкале от 1 до 10. Оцени следующую идею для YouTube-видео по шкале от 1 до 10 для канала "{channel_title}":

"{idea}"

{existing_content}

Популярные теги на канале: {', '.join(popular_tags) if popular_tags else 'Нет данных'}
Лучшее видео на канале: "{stats.get('best_video', 'Нет данных')}"

Оцени идею по следующим критериям:
1. Оригинальность (не повторяет существующие видео)
2. Потенциал просмотров и вовлеченности
3. Соответствие тематике канала
4. Актуальность и трендовость

Дай оценку по шкале от 1 до 10, где 1 - очень плохо, 10 - превосходно.
Также предложи 2-3 способа улучшить эту идею или альтернативные формулировки.

Формат ответа:
Оценка: X/10
Комментарий: (краткое обоснование оценки)
Рекомендации по улучшению:
• Рекомендация 1
• Рекомендация 2
• Рекомендация 3
"""

            # Отправляем запрос
            response = client.chat.completions.create(
                model="gpt-4o",  # Используем модель GPT-4o
                messages=[
                    {"role": "system", "content": "Ты - эксперт по YouTube-контенту, который оценивает идеи для видео по шкале от 1 до 10."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=500
            )

            # Обрабатываем ответ
            evaluation_text = response.choices[0].message.content.strip()
            
            # Извлекаем оценку
            rating = 0
            if "Оценка:" in evaluation_text:
                rating_text = evaluation_text.split("Оценка:")[1].split("\n")[0].strip()
                try:
                    if "/" in rating_text:
                        rating = int(rating_text.split("/")[0].strip())
                    else:
                        # Находим число в тексте
                        import re
                        numbers = re.findall(r'\d+', rating_text)
                        if numbers:
                            rating = int(numbers[0])
                except:
                    rating = 0
            
            # Извлекаем комментарий
            comment = ""
            if "Комментарий:" in evaluation_text:
                comment = evaluation_text.split("Комментарий:")[1].split("Рекомендации")[0].strip()
            
            # Извлекаем рекомендации
            recommendations = []
            if "Рекомендации" in evaluation_text:
                recommendations_text = evaluation_text.split("Рекомендации")[1].strip()
                # Разделяем по маркерам списка или по новым строкам
                if "•" in recommendations_text:
                    recommendations = [rec.strip() for rec in recommendations_text.split("•") if rec.strip()]
                elif "-" in recommendations_text:
                    recommendations = [rec.strip() for rec in recommendations_text.split("-") if rec.strip()]
                else:
                    recommendations = [rec.strip() for rec in recommendations_text.split("\n") if rec.strip()]
            
            return {
                "rating": rating,
                "comment": comment,
                "recommendations": recommendations,
                "raw_evaluation": evaluation_text
            }

        except Exception as e:
            print(f"Ошибка при оценке идеи контента: {e}")
            return {"rating": 0, "comment": str(e), "recommendations": [], "raw_evaluation": str(e)}


# Класс для Telegram-бота
class YouTubeTelegramBot:
    def __init__(self, token, youtube_api_key, openai_api_key, perplexity_api_key=None):
        """
        Инициализация Telegram-бота.

        Args:
            token (str): Токен Telegram-бота
            youtube_api_key (str): API-ключ YouTube
            openai_api_key (str): API-ключ OpenAI
            perplexity_api_key (str, optional): API-ключ Perplexity
        """
        self.bot = telebot.TeleBot(token)
        self.youtube_analyzer = YouTubeAnalyticsBot(youtube_api_key, openai_api_key, perplexity_api_key)

        # Словари для хранения состояния пользователей
        self.user_states = {}
        self.user_data = {}

        # Регистрация обработчиков сообщений
        self.register_handlers()

    def register_handlers(self):
        """Регистрация обработчиков сообщений"""

        @self.bot.message_handler(commands=['start'])
        def start_handler(message):
            self.send_welcome(message)

        @self.bot.message_handler(commands=['help'])
        def help_handler(message):
            self.send_help(message)

        @self.bot.message_handler(content_types=['text'])
        def text_handler(message):
            self.process_text_message(message)

    def send_welcome(self, message):
        """Отправка приветственного сообщения"""
        user_id = message.from_user.id
        self.user_states[user_id] = 'main'

        # Создаем клавиатуру
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(types.KeyboardButton("📊 Анализ YouTube-канала"))
        markup.add(types.KeyboardButton("🎬 Идеи для видео"), types.KeyboardButton("📱 Идеи для Shorts"))
        markup.add(types.KeyboardButton("💡 Оценить идею"), types.KeyboardButton("ℹ️ Справка"))

        self.bot.send_message(
            message.chat.id,
            f"👋 Здравствуйте, {message.from_user.first_name}!\n\n"
            "Я бот для анализа YouTube-каналов и создания контент-планов.\n\n"
            "С моей помощью вы можете:\n"
            "• Анализировать статистику видео\n"
            "• Получать идеи для видео и Shorts\n"
            "• Оценивать свои идеи и получать рекомендации\n\n"
            "Выберите действие в меню.",
            reply_markup=markup
        )

    def send_help(self, message):
        """Отправка справочного сообщения"""
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(types.KeyboardButton("📊 Анализ YouTube-канала"))
        markup.add(types.KeyboardButton("🎬 Идеи для видео"), types.KeyboardButton("📱 Идеи для Shorts"))
        markup.add(types.KeyboardButton("💡 Оценить идею"), types.KeyboardButton("ℹ️ Справка"))

        self.bot.send_message(
            message.chat.id,
            "📚 *Справка по боту*\n\n"
            "1️⃣ *Анализ канала* — анализ статистики и рекомендации\n"
            "2️⃣ *Идеи для видео* — 15 идей для обычных видео\n"
            "3️⃣ *Идеи для Shorts* — 15 идей для коротких вертикальных видео\n"
            "4️⃣ *Оценить идею* — оценка вашей идеи от 1 до 10\n",
            parse_mode="Markdown",
            reply_markup=markup
        )

    def process_text_message(self, message):
        """Обработка текстовых сообщений"""
        user_id = message.from_user.id
        state = self.user_states.get(user_id, 'main')

        # Обработка кнопки возврата в главное меню
        if message.text == "⬅️ Назад" or message.text == "ℹ️ Справка":
            if message.text == "ℹ️ Справка":
                self.send_help(message)

            self.user_states[user_id] = 'main'
            if user_id in self.user_data:
                self.user_data.pop(user_id, None)

            # Восстанавливаем главное меню
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(types.KeyboardButton("📊 Анализ YouTube-канала"))
            markup.add(types.KeyboardButton("🎬 Идеи для видео"), types.KeyboardButton("📱 Идеи для Shorts"))
            markup.add(types.KeyboardButton("💡 Оценить идею"), types.KeyboardButton("ℹ️ Справка"))

            if message.text == "⬅️ Назад":
                self.bot.send_message(
                    message.chat.id,
                    "Вы вернулись в главное меню.",
                    reply_markup=markup
                )
            return

        # Обработка кнопки запуска анализа
        if message.text == "📊 Анализ YouTube-канала":
            self.user_states[user_id] = 'waiting_for_channel'
            self.user_data[user_id] = {}

            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(types.KeyboardButton("⬅️ Назад"))

            self.bot.send_message(
                message.chat.id,
                "Введите ID канала YouTube (начинается с 'UC') или полный URL канала.",
                reply_markup=markup
            )
            return
            
        # Обработка кнопки идей для видео
        if message.text == "🎬 Идеи для видео":
            if user_id in self.user_data and 'channel_id' in self.user_data[user_id]:
                # Если канал уже был проанализирован, генерируем идеи сразу
                self.generate_video_ideas(message)
            else:
                # Если канал еще не задан, запрашиваем его
                self.user_states[user_id] = 'waiting_for_channel_video_ideas'
                self.user_data[user_id] = {}

                markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                markup.add(types.KeyboardButton("⬅️ Назад"))

                self.bot.send_message(
                    message.chat.id,
                    "Введите ID канала YouTube (начинается с 'UC') или полный URL канала для генерации идей для видео.",
                    reply_markup=markup
                )
            return
            
        # Обработка кнопки идей для Shorts
        if message.text == "📱 Идеи для Shorts":
            if user_id in self.user_data and 'channel_id' in self.user_data[user_id]:
                # Если канал уже был проанализирован, генерируем идеи сразу
                self.generate_shorts_ideas(message)
            else:
                # Если канал еще не задан, запрашиваем его
                self.user_states[user_id] = 'waiting_for_channel_shorts_ideas'
                self.user_data[user_id] = {}

                markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                markup.add(types.KeyboardButton("⬅️ Назад"))

                self.bot.send_message(
                    message.chat.id,
                    "Введите ID канала YouTube (начинается с 'UC') или полный URL канала для генерации идей для Shorts.",
                    reply_markup=markup
                )
            return
            
        # Обработка кнопки оценки идеи
        if message.text == "💡 Оценить идею":
            if user_id in self.user_data and 'channel_id' in self.user_data[user_id]:
                # Если канал уже был проанализирован, запрашиваем идею для оценки
                self.user_states[user_id] = 'waiting_for_idea_to_evaluate'
                
                markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                markup.add(types.KeyboardButton("⬅️ Назад"))
                
                self.bot.send_message(
                    message.chat.id,
                    "Введите идею для видео, которую нужно оценить по шкале от 1 до 10.",
                    reply_markup=markup
                )
            else:
                # Если канал еще не задан, запрашиваем его
                self.user_states[user_id] = 'waiting_for_channel_idea_evaluation'
                self.user_data[user_id] = {}
                
                markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                markup.add(types.KeyboardButton("⬅️ Назад"))
                
                self.bot.send_message(
                    message.chat.id,
                    "Введите ID канала YouTube (начинается с 'UC') или полный URL канала для оценки идеи.",
                    reply_markup=markup
                )
            return

        # Обработка ввода канала для анализа
        if state == 'waiting_for_channel':
            channel_input = message.text.strip()
            channel_id = self.extract_channel_id(channel_input)

            if not channel_id:
                self.bot.send_message(
                    message.chat.id,
                    "⚠️ Не удалось распознать ID канала. Пожалуйста, введите ID в формате 'UCxxxxxxxxxxxxxxxx' "
                    "или полный URL канала."
                )
                return

            # Сохраняем ID канала
            self.user_data[user_id]['channel_id'] = channel_id
            self.user_states[user_id] = 'waiting_for_niche'

            self.bot.send_message(
                message.chat.id,
                f"✅ ID канала принят: {channel_id}\n\n"
                "Теперь введите тематику канала для более точного анализа трендов.\n"
                "Например: 'технологии', 'кулинария', 'игры', 'образование' и т.д."
            )
            return
            
        # Обработка ввода канала для генерации идей видео
        if state == 'waiting_for_channel_video_ideas':
            channel_input = message.text.strip()
            channel_id = self.extract_channel_id(channel_input)

            if not channel_id:
                self.bot.send_message(
                    message.chat.id,
                    "⚠️ Не удалось распознать ID канала. Пожалуйста, введите ID в формате 'UCxxxxxxxxxxxxxxxx' "
                    "или полный URL канала."
                )
                return

            # Сохраняем ID канала и запускаем генерацию идей для видео
            self.user_data[user_id] = {'channel_id': channel_id}
            self.generate_video_ideas(message)
            return
            
        # Обработка ввода канала для генерации идей Shorts
        if state == 'waiting_for_channel_shorts_ideas':
            channel_input = message.text.strip()
            channel_id = self.extract_channel_id(channel_input)

            if not channel_id:
                self.bot.send_message(
                    message.chat.id,
                    "⚠️ Не удалось распознать ID канала. Пожалуйста, введите ID в формате 'UCxxxxxxxxxxxxxxxx' "
                    "или полный URL канала."
                )
                return

            # Сохраняем ID канала и запускаем генерацию идей для Shorts
            self.user_data[user_id] = {'channel_id': channel_id}
            self.generate_shorts_ideas(message)
            return
            
        # Обработка ввода канала для оценки идеи
        if state == 'waiting_for_channel_idea_evaluation':
            channel_input = message.text.strip()
            channel_id = self.extract_channel_id(channel_input)

            if not channel_id:
                self.bot.send_message(
                    message.chat.id,
                    "⚠️ Не удалось распознать ID канала. Пожалуйста, введите ID в формате 'UCxxxxxxxxxxxxxxxx' "
                    "или полный URL канала."
                )
                return

            # Сохраняем ID канала и переходим к запросу идеи для оценки
            self.user_data[user_id] = {'channel_id': channel_id}
            self.user_states[user_id] = 'waiting_for_idea_to_evaluate'
            
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(types.KeyboardButton("⬅️ Назад"))
            
            self.bot.send_message(
                message.chat.id,
                "Введите идею для видео, которую нужно оценить по шкале от 1 до 10.",
                reply_markup=markup
            )
            return
            
        # Обработка ввода идеи для оценки
        if state == 'waiting_for_idea_to_evaluate':
            idea = message.text.strip()
            
            if not idea:
                self.bot.send_message(
                    message.chat.id,
                    "⚠️ Идея не может быть пустой. Пожалуйста, введите идею для оценки."
                )
                return
                
            # Сохраняем идею и запускаем оценку
            self.user_data[user_id]['idea_to_evaluate'] = idea
            self.evaluate_user_idea(message)
            return

        # Обработка ввода тематики канала
        if state == 'waiting_for_niche':
            niche = message.text.strip()

            if not niche:
                self.bot.send_message(
                    message.chat.id,
                    "⚠️ Тематика не может быть пустой. Пожалуйста, введите тематику канала."
                )
                return

            # Сохраняем тематику канала
            self.user_data[user_id]['niche'] = niche

            # Запускаем анализ в отдельном потоке
            self.start_analysis(message, user_id)
            return
            
    def generate_video_ideas(self, message):
        """
        Генерирует идеи для видео на основе анализа канала.
        
        Args:
            message (Message): Сообщение пользователя
        """
        user_id = message.from_user.id
        channel_id = self.user_data[user_id]['channel_id']
        
        # Отправляем сообщение о начале генерации
        status_msg = self.bot.send_message(
            message.chat.id,
            "🎬 Генерирую идеи для видео. Это может занять некоторое время..."
        )
        
        # Запускаем генерацию идей в отдельном потоке
        def ideas_thread():
            try:
                # Получаем информацию о канале
                channel_info = self.youtube_analyzer._get_channel_info(channel_id)
                
                # Получаем и анализируем видео
                videos = self.youtube_analyzer.get_last_week_videos(channel_id, max_results=30)
                
                if isinstance(videos, dict) and "error" in videos:
                    self.bot.edit_message_text(
                        f"❌ Ошибка при получении видео: {videos['error']}",
                        message.chat.id,
                        status_msg.message_id
                    )
                    return
                
                # Получаем заголовки существующих видео
                existing_titles = [video['title'] for video in videos]
                
                # Анализируем видео
                analysis_results = self.youtube_analyzer.analyze_videos(videos)
                
                if isinstance(analysis_results, dict) and "error" in analysis_results:
                    self.bot.edit_message_text(
                        f"❌ Ошибка при анализе видео: {analysis_results['error']}",
                        message.chat.id,
                        status_msg.message_id
                    )
                    return
                
                # Генерируем идеи для видео
                content_ideas = self.youtube_analyzer.generate_content_ideas(
                    channel_info, 
                    analysis_results["stats"],
                    analysis_results["popular_tags"],
                    analysis_results["topic_clusters"],
                    existing_titles,
                    "video"
                )
                
                # Отправляем результаты
                self.send_content_ideas(message.chat.id, content_ideas, "видео")
                
            except Exception as e:
                self.bot.send_message(
                    message.chat.id,
                    f"❌ Произошла ошибка при генерации идей: {str(e)}"
                )
                
        # Запускаем поток
        threading.Thread(target=ideas_thread).start()
        
    def generate_shorts_ideas(self, message):
        """
        Генерирует идеи для Shorts на основе анализа канала.
        
        Args:
            message (Message): Сообщение пользователя
        """
        user_id = message.from_user.id
        channel_id = self.user_data[user_id]['channel_id']
        
        # Отправляем сообщение о начале генерации
        status_msg = self.bot.send_message(
            message.chat.id,
            "📱 Генерирую идеи для Shorts. Это может занять некоторое время..."
        )
        
        # Запускаем генерацию идей в отдельном потоке
        def ideas_thread():
            try:
                # Получаем информацию о канале
                channel_info = self.youtube_analyzer._get_channel_info(channel_id)
                
                # Получаем и анализируем видео
                videos = self.youtube_analyzer.get_last_week_videos(channel_id, max_results=30)
                
                if isinstance(videos, dict) and "error" in videos:
                    self.bot.edit_message_text(
                        f"❌ Ошибка при получении видео: {videos['error']}",
                        message.chat.id,
                        status_msg.message_id
                    )
                    return
                
                # Получаем заголовки существующих видео
                existing_titles = [video['title'] for video in videos]
                
                # Анализируем видео
                analysis_results = self.youtube_analyzer.analyze_videos(videos)
                
                if isinstance(analysis_results, dict) and "error" in analysis_results:
                    self.bot.edit_message_text(
                        f"❌ Ошибка при анализе видео: {analysis_results['error']}",
                        message.chat.id,
                        status_msg.message_id
                    )
                    return
                
                # Генерируем идеи для Shorts
                content_ideas = self.youtube_analyzer.generate_content_ideas(
                    channel_info, 
                    analysis_results["stats"],
                    analysis_results["popular_tags"],
                    analysis_results["topic_clusters"],
                    existing_titles,
                    "shorts"
                )
                
                # Отправляем результаты
                self.send_content_ideas(message.chat.id, content_ideas, "Shorts")
                
            except Exception as e:
                self.bot.send_message(
                    message.chat.id,
                    f"❌ Произошла ошибка при генерации идей: {str(e)}"
                )
                
        # Запускаем поток
        threading.Thread(target=ideas_thread).start()
        
    def send_content_ideas(self, chat_id, content_ideas, content_type):
        """
        Отправляет сгенерированные идеи пользователю.
        
        Args:
            chat_id (int): ID чата
            content_ideas (list): Список идей для контента
            content_type (str): Тип контента ('видео' или 'Shorts')
        """
        try:
            # Отправляем заголовок
            self.bot.send_message(
                chat_id,
                f"🚀 *15 идей для {content_type}*\n\n",
                parse_mode="Markdown"
            )
            
            # Форматируем и отправляем каждую идею отдельным сообщением
            for i, idea in enumerate(content_ideas, 1):
                idea_message = f"*{i}. {idea['title']}*\n\n"
                
                if idea['description']:
                    idea_message += f"{idea['description']}\n\n"
                
                if idea['key_points'] and len(idea['key_points']) > 0:
                    idea_message += "*Ключевые моменты:*\n"
                    for point in idea['key_points']:
                        idea_message += f"• {point}\n"
                
                # Отправляем идею
                self.bot.send_message(
                    chat_id,
                    idea_message,
                    parse_mode="Markdown"
                )
            
            # Отправляем итоговое сообщение
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(types.KeyboardButton("📊 Анализ YouTube-канала"))
            markup.add(types.KeyboardButton("🎬 Идеи для видео"), types.KeyboardButton("📱 Идеи для Shorts"))
            markup.add(types.KeyboardButton("💡 Оценить идею"), types.KeyboardButton("ℹ️ Справка"))
            
            self.bot.send_message(
                chat_id,
                f"✅ Готово! Сгенерировано 15 идей для {content_type}.\n\n"
                "Вы можете выбрать другой тип контента или оценить свою идею.",
                reply_markup=markup
            )
            
        except Exception as e:
            self.bot.send_message(
                chat_id,
                f"❌ Ошибка при отправке идей: {str(e)}"
            )
            
    def evaluate_user_idea(self, message):
        """
        Оценивает идею пользователя по шкале от 1 до 10.
        
        Args:
            message (Message): Сообщение пользователя
        """
        user_id = message.from_user.id
        channel_id = self.user_data[user_id]['channel_id']
        idea = self.user_data[user_id]['idea_to_evaluate']
        
        # Отправляем сообщение о начале оценки
        status_msg = self.bot.send_message(
            message.chat.id,
            "🔍 Оцениваю вашу идею. Это может занять некоторое время..."
        )
        
        # Запускаем оценку идеи в отдельном потоке
        def evaluation_thread():
            try:
                # Получаем информацию о канале
                channel_info = self.youtube_analyzer._get_channel_info(channel_id)
                
                # Получаем и анализируем видео
                videos = self.youtube_analyzer.get_last_week_videos(channel_id, max_results=30)
                
                if isinstance(videos, dict) and "error" in videos:
                    self.bot.edit_message_text(
                        f"❌ Ошибка при получении видео: {videos['error']}",
                        message.chat.id,
                        status_msg.message_id
                    )
                    return
                
                # Получаем заголовки существующих видео
                existing_titles = [video['title'] for video in videos]
                
                # Анализируем видео
                analysis_results = self.youtube_analyzer.analyze_videos(videos)
                
                if isinstance(analysis_results, dict) and "error" in analysis_results:
                    self.bot.edit_message_text(
                        f"❌ Ошибка при анализе видео: {analysis_results['error']}",
                        message.chat.id,
                        status_msg.message_id
                    )
                    return
                
                # Оцениваем идею
                evaluation = self.youtube_analyzer.evaluate_content_idea(
                    idea,
                    channel_info,
                    analysis_results["stats"],
                    analysis_results["popular_tags"],
                    existing_titles
                )
                
                # Отправляем результаты оценки
                self.send_idea_evaluation(message.chat.id, idea, evaluation)
                
            except Exception as e:
                self.bot.send_message(
                    message.chat.id,
                    f"❌ Произошла ошибка при оценке идеи: {str(e)}"
                )
                
        # Запускаем поток
        threading.Thread(target=evaluation_thread).start()
        
    def send_idea_evaluation(self, chat_id, idea, evaluation):
        """
        Отправляет результаты оценки идеи пользователю.
        
        Args:
            chat_id (int): ID чата
            idea (str): Идея пользователя
            evaluation (dict): Результаты оценки
        """
        try:
            # Формируем рейтинг в виде звездочек
            rating = evaluation['rating']
            stars = "⭐" * rating + "☆" * (10 - rating)
            
            # Формируем сообщение с оценкой
            evaluation_message = f"*Оценка вашей идеи:*\n\n"
            evaluation_message += f"*\"{idea}\"*\n\n"
            evaluation_message += f"*Рейтинг:* {rating}/10 {stars}\n\n"
            
            if evaluation['comment']:
                evaluation_message += f"*Комментарий:*\n{evaluation['comment']}\n\n"
            
            if evaluation['recommendations'] and len(evaluation['recommendations']) > 0:
                evaluation_message += "*Рекомендации по улучшению:*\n"
                for rec in evaluation['recommendations']:
                    evaluation_message += f"• {rec}\n"
            
            # Отправляем оценку
            self.bot.send_message(
                chat_id,
                evaluation_message,
                parse_mode="Markdown"
            )
            
            # Отправляем итоговое сообщение
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(types.KeyboardButton("📊 Анализ YouTube-канала"))
            markup.add(types.KeyboardButton("🎬 Идеи для видео"), types.KeyboardButton("📱 Идеи для Shorts"))
            markup.add(types.KeyboardButton("💡 Оценить идею"), types.KeyboardButton("ℹ️ Справка"))
            
            self.bot.send_message(
                chat_id,
                "✅ Готово! Вы можете оценить другую идею или выбрать другую функцию.",
                reply_markup=markup
            )
            
        except Exception as e:
            self.bot.send_message(
                chat_id,
                f"❌ Ошибка при отправке оценки: {str(e)}"
            )

    def extract_channel_id(self, text):
        """
        Извлечение ID канала из текста.

        Args:
            text (str): URL или ID канала

        Returns:
            str: ID канала или None, если не удалось извлечь
        """
        text = text.strip()

        # Проверяем, является ли введенный текст уже ID каналом
        if text.startswith('UC') and len(text) > 10:
            return text

        # Извлекаем ID из URL типа youtube.com/channel/UC...
        if '/channel/' in text:
            try:
                channel_id = text.split('/channel/')[1].split('/')[0].split('?')[0]
                if channel_id.startswith('UC'):
                    return channel_id
            except:
                pass

        # Извлекаем идентификатор из URL типа youtube.com/@username
        if '/@' in text:
            try:
                # Здесь мы возвращаем сам URL для последующего получения ID
                return f"channel_url:{text}"
            except:
                pass

        return None

    def send_status_message(self, chat_id, text):
        """
        Отправка сообщения о статусе анализа.

        Args:
            chat_id (int): ID чата
            text (str): Текст сообщения
        """
        try:
            self.bot.send_message(chat_id, text)
        except Exception as e:
            print(f"Ошибка при отправке статуса: {e}")

    def start_analysis(self, message, user_id):
        """
        Запуск анализа канала в отдельном потоке.

        Args:
            message (Message): Сообщение пользователя
            user_id (int): ID пользователя
        """
        # Получаем данные для анализа
        channel_id = self.user_data[user_id]['channel_id']
        niche = self.user_data[user_id]['niche']

        # Проверяем, является ли channel_id URL-адресом
        if channel_id.startswith('channel_url:'):
            url = channel_id.replace('channel_url:', '')
            status_msg = self.bot.send_message(
                message.chat.id,
                f"🔍 Получение ID канала из URL: {url}..."
            )

            # TODO: Добавить функцию получения ID канала из URL
            # Пока просто сообщаем об ошибке
            self.bot.edit_message_text(
                "⚠️ Извините, получение ID канала из URL-адреса временно недоступно. "
                "Пожалуйста, введите ID канала напрямую (начинается с 'UC').",
                message.chat.id,
                status_msg.message_id
            )
            return

        # Отправляем начальное сообщение о статусе
        status_msg = self.bot.send_message(
            message.chat.id,
            "🚀 Начинаем анализ канала. Это может занять некоторое время..."
        )

        # Создаем функцию для обновления статуса
        def update_status(text):
            try:
                self.bot.edit_message_text(
                    text,
                    message.chat.id,
                    status_msg.message_id
                )
            except Exception as e:
                print(f"Ошибка при обновлении статуса: {e}")

        # Запускаем анализ в отдельном потоке
        def analysis_thread():
            try:
                # Выполняем анализ
                content_plan = self.youtube_analyzer.generate_content_plan(
                    channel_id,
                    niche,
                    update_status
                )

                # Проверяем результат
                if "error" in content_plan:
                    self.bot.send_message(
                        message.chat.id,
                        f"❌ Ошибка при анализе канала: {content_plan['error']}"
                    )
                    return

                # Отправляем результаты анализа
                self.send_analysis_results(message.chat.id, content_plan)

            except Exception as e:
                self.bot.send_message(
                    message.chat.id,
                    f"❌ Произошла ошибка при анализе: {str(e)}"
                )

        # Запускаем поток
        threading.Thread(target=analysis_thread).start()

    def send_analysis_results(self, chat_id, content_plan):
        """
        Отправка упрощенных результатов анализа пользователю (только статистика и контент-план).

        Args:
            chat_id (int): ID чата
            content_plan (dict): Результаты анализа
        """
        try:
            # Отправляем основную информацию о канале одним сообщением
            if "channel_info" in content_plan and content_plan["channel_info"]:
                channel = content_plan["channel_info"]
                stats = content_plan["stats"]
                
                # Общая информация о канале и статистика
                main_message = (
                    f"📺 *Информация о канале {channel['title']}*\n\n"
                    f"📈 Всего видео за неделю: {stats['total_videos']}\n"
                    f"🎬 Обычных видео: {stats['total_regular_videos']}\n"
                    f"📱 Shorts: {stats['total_shorts']}\n"
                    f"👁️ Средние просмотры: {stats['avg_views']:.0f}\n"
                    f"💬 Средние комментарии: {stats['avg_comments']:.0f}\n"
                    f"🔄 Вовлеченность: {stats['engagement_rate']:.2f}%\n"
                    f"🏆 Лучшее видео: \"{stats['best_video']}\"\n"
                    f"📅 Лучшие дни: {', '.join(stats['best_days'][:3]) if stats['best_days'] else 'Недостаточно данных'}"
                )
                
                self.bot.send_message(chat_id, main_message, parse_mode="Markdown")
            
            # Отправляем рекомендации по улучшению кратко
            if "recommendations" in content_plan:
                recommendations = content_plan["recommendations"]
                
                # Только самые важные рекомендации
                rec_message = "📈 *Рекомендации по улучшению показателей*\n\n"
                
                # Добавляем рекомендации по расписанию и формату контента
                if "publishing_schedule" in recommendations:
                    rec_message += f"• {recommendations['publishing_schedule']['message']}\n"
                
                if "content_format" in recommendations:
                    rec_message += f"• {recommendations['content_format']['message']}\n"
                
                if "tags" in recommendations:
                    rec_message += f"• {recommendations['tags']['message']}\n"
                
                if "general" in recommendations:
                    # Добавляем только первые 3 общие рекомендации
                    for idx, rec in enumerate(recommendations["general"][:3], 1):
                        rec_message += f"• {rec['title']}\n"
                
                self.bot.send_message(chat_id, rec_message, parse_mode="Markdown")
            
            # Генерируем идеи для видео и shorts
            status_msg = self.bot.send_message(
                chat_id,
                "🚀 Генерирую идеи для контента...",
                parse_mode="Markdown"
            )
            
            # Запускаем генерацию идей в отдельном потоке
            def generate_ideas_thread():
                try:
                    # Получаем заголовки существующих видео
                    existing_titles = []
                    if "channel_info" in content_plan:
                        channel_info = content_plan["channel_info"]
                        videos = self.youtube_analyzer.get_last_week_videos(channel_info["id"], max_results=30)
                        if not isinstance(videos, dict) and videos:
                            existing_titles = [video['title'] for video in videos]
                    
                    # Генерируем идеи для видео и для Shorts
                    video_ideas = self.youtube_analyzer.generate_content_ideas(
                        content_plan.get("channel_info"),
                        content_plan["stats"],
                        content_plan.get("popular_tags", []),
                        content_plan.get("topic_clusters", []),
                        existing_titles,
                        "video"
                    )
                    
                    shorts_ideas = self.youtube_analyzer.generate_content_ideas(
                        content_plan.get("channel_info"),
                        content_plan["stats"],
                        content_plan.get("popular_tags", []),
                        content_plan.get("topic_clusters", []),
                        existing_titles,
                        "shorts"
                    )
                    
                    # Обновляем статусное сообщение
                    self.bot.edit_message_text(
                        "✅ Идеи сгенерированы!",
                        chat_id,
                        status_msg.message_id
                    )
                    
                    # Отправляем только заголовки идей для видео
                    video_message = "🎬 *15 идей для видео:*\n\n"
                    for i, idea in enumerate(video_ideas[:15], 1):
                        video_message += f"{i}. {idea['title']}\n"
                    
                    self.bot.send_message(chat_id, video_message, parse_mode="Markdown")
                    
                    # Отправляем только заголовки идей для Shorts
                    shorts_message = "📱 *15 идей для Shorts:*\n\n"
                    for i, idea in enumerate(shorts_ideas[:15], 1):
                        shorts_message += f"{i}. {idea['title']}\n"
                    
                    self.bot.send_message(chat_id, shorts_message, parse_mode="Markdown")
                    
                    # Сохраняем идеи для последующего просмотра
                    user_id = None
                    for uid, data in self.user_data.items():
                        if "channel_id" in data and data["channel_id"] == content_plan.get("channel_info", {}).get("id"):
                            user_id = uid
                            break
                    
                    if user_id:
                        self.user_data[user_id]["video_ideas"] = video_ideas
                        self.user_data[user_id]["shorts_ideas"] = shorts_ideas
                except Exception as e:
                    self.bot.send_message(
                        chat_id,
                        f"❌ Ошибка при генерации идей: {str(e)}"
                    )
            
            # Запускаем поток для генерации идей
            threading.Thread(target=generate_ideas_thread).start()
            
            # Отправляем итоговое сообщение с кнопками
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.add(types.KeyboardButton("📊 Анализ YouTube-канала"))
            markup.add(types.KeyboardButton("🎬 Идеи для видео"), types.KeyboardButton("📱 Идеи для Shorts"))
            markup.add(types.KeyboardButton("💡 Оценить идею"), types.KeyboardButton("ℹ️ Справка"))
            
            self.bot.send_message(
                chat_id,
                "✅ Анализ завершен! Выберите дальнейшее действие:",
                reply_markup=markup
            )

        except Exception as e:
            self.bot.send_message(
                chat_id,
                f"❌ Ошибка при отправке результатов анализа: {str(e)}"
            )

    def run(self):
        """
        Запуск бота с обработкой ошибок.
        """
        print("Бот запущен. Нажмите Ctrl+C для остановки.")
        try:
            self.bot.polling(none_stop=True)
        except Exception as e:
            print(f"Ошибка при работе бота: {e}")
            time.sleep(15)
            self.run()  # Перезапуск при ошибке

# Запуск приложения
if __name__ == "__main__":
    try:
        # Инициализация бота с API-ключами
        telegram_bot = YouTubeTelegramBot(
            token=TELEGRAM_BOT_TOKEN,
            youtube_api_key=YOUTUBE_API_KEY,
            openai_api_key=OPENAI_API_KEY,
            perplexity_api_key=PERPLEXITY_API_KEY
        )

        # Запуск бота
        telegram_bot.run()
    except Exception as e:
        print(f"Ошибка при запуске бота: {e}")