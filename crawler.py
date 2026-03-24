import json
import os
import time
import random
import logging
from collections import deque

import requests

MINIMUM_RATING = None  # Set to an integer (e.g., 850) to enforce a minimum rating
MAXIMUM_RATING = None  # Set to an integer (e.g., 1800) to enforce a maximum rating
TARGET_GAME_COUNT = 10000
OUTPUT_FILE_PATH = os.path.join("data", "raw_games.json")
STATE_FILE_PATH = os.path.join("data", "crawler_state.json")
MAXIMUM_ARCHIVES_PER_PLAYER = 10
API_BASE_URL = "https://api.chess.com/pub"
REQUEST_HEADERS = {
    "User-Agent": "ChessCrawler/1.0 (student project)"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def fetch_api_response(url: str) -> dict | None:
    time.sleep(0.2)
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    except requests.RequestException as exception:
        logger.error("Request failed for %s: %s", url, exception)
        return None

    if response.status_code == 200:
        return response.json()

    if response.status_code == 429:
        logger.warning("Rate limited, sleeping 10 seconds")
        time.sleep(10)
        return None

    if response.status_code != 404:
        logger.warning("HTTP %d for %s", response.status_code, url)

    return None


def load_existing_games() -> list[dict]:
    if not os.path.exists(OUTPUT_FILE_PATH):
        return []

    try:
        with open(OUTPUT_FILE_PATH, "r", encoding="utf-8") as file:
            games = json.load(file)
        logger.info("Resumed with %d existing games.", len(games))
        return games
    except (json.JSONDecodeError, IOError) as exception:
        logger.error("Could not load existing games: %s", exception)
        return []


def save_games_to_file(games: list[dict]) -> None:
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    temporary_path = OUTPUT_FILE_PATH + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(games, file, ensure_ascii=False)
    os.replace(temporary_path, OUTPUT_FILE_PATH)


def load_crawler_state() -> tuple[set[str], deque[str]]:
    if not os.path.exists(STATE_FILE_PATH):
        return set(), deque()
    try:
        with open(STATE_FILE_PATH, "r", encoding="utf-8") as file:
            state = json.load(file)
        visited = set(state.get("visited_players", []))
        queue = deque(state.get("player_queue", []))
        logger.info("Resumed BFS state with %d visited players and %d in queue.", len(visited), len(queue))
        return visited, queue
    except Exception as exception:
        logger.error("Could not load crawler state: %s", exception)
        return set(), deque()


def save_crawler_state(visited_players: set[str], player_queue: deque[str]) -> None:
    os.makedirs(os.path.dirname(STATE_FILE_PATH), exist_ok=True)
    state = {
        "visited_players": list(visited_players),
        "player_queue": list(player_queue)
    }
    temporary_path = STATE_FILE_PATH + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False)
    os.replace(temporary_path, STATE_FILE_PATH)


def fetch_archive_urls(username: str) -> list[str]:
    response = fetch_api_response(f"{API_BASE_URL}/player/{username}/games/archives")
    if response is None:
        return []

    archive_urls = response.get("archives", [])
    most_recent_first = list(reversed(archive_urls))
    return most_recent_first[:MAXIMUM_ARCHIVES_PER_PLAYER]


def is_rating_in_range(rating: int | None) -> bool:
    if rating is None:
        return False
    if MINIMUM_RATING is not None and rating < MINIMUM_RATING:
        return False
    if MAXIMUM_RATING is not None and rating > MAXIMUM_RATING:
        return False
    return True


def fetch_player_rapid_rating(username: str) -> int | None:
    response = fetch_api_response(f"{API_BASE_URL}/player/{username}/stats")
    if response is None:
        return None

    rapid_stats = response.get("chess_rapid", {})
    last_rating = rapid_stats.get("last", {})
    return last_rating.get("rating")


def extract_game_record(raw_game: dict) -> dict | None:
    if raw_game.get("rules", "chess") != "chess":
        return None

    white_player = raw_game.get("white", {})
    black_player = raw_game.get("black", {})
    white_rating = white_player.get("rating")
    black_rating = black_player.get("rating")

    if not is_rating_in_range(white_rating):
        return None
    if not is_rating_in_range(black_rating):
        return None

    pgn_text = raw_game.get("pgn")
    if not pgn_text:
        return None

    return {
        "white": white_player.get("username", ""),
        "black": black_player.get("username", ""),
        "white_rating": white_rating,
        "black_rating": black_rating,
        "pgn": pgn_text,
        "time_class": raw_game.get("time_class", ""),
        "url": raw_game.get("url", ""),
        "end_time": raw_game.get("end_time", 0),
    }


def discover_seed_players(required_count: int = 10) -> list[str]:
    logger.info("Discovering seed players from Chess.com API")

    country_codes = ["US", "GB", "IN", "DE", "FR", "BR", "CA", "PH", "ES", "MX"]
    candidate_usernames: list[str] = []

    for country_code in country_codes:
        response = fetch_api_response(f"{API_BASE_URL}/country/{country_code}/players")
        if response and "players" in response:
            players = response["players"]
            logger.info("  %s: %d players listed", country_code, len(players))
            candidate_usernames.extend(players)
        if len(candidate_usernames) >= 5000:
            break

    if not candidate_usernames:
        logger.error("Could not fetch any country player lists.")
        return []

    #getting rid of bias
    random.shuffle(candidate_usernames)
    verified_seeds: list[str] = []

    for username in candidate_usernames:
        if len(verified_seeds) >= required_count:
            break

        rating = fetch_player_rapid_rating(username)
        if rating is None:
            continue
        if not is_rating_in_range(rating):
            continue

        verified_seeds.append(username)
        logger.info("  Seed found: %s (rating %d)", username, rating)

    if not verified_seeds:
        logger.warning("No seeds found in target range, using raw candidates.")
        verified_seeds = candidate_usernames[:required_count]

    logger.info("Discovered %d seed players.", len(verified_seeds))
    return verified_seeds


def process_player_archives(username: str, seen_game_urls: set[str]) -> list[dict]:
    archive_urls = fetch_archive_urls(username)
    collected_games: list[dict] = []

    for archive_url in archive_urls:
        response = fetch_api_response(archive_url)
        if response is None:
            continue

        for raw_game in response.get("games", []):
            record = extract_game_record(raw_game)
            if record is None:
                continue
            if record["url"] in seen_game_urls:
                continue

            seen_game_urls.add(record["url"])
            collected_games.append(record)

    return collected_games


def collect_opponent_usernames(game_records: list[dict], visited_players: set[str]) -> list[str]:
    new_usernames: set[str] = set()

    for record in game_records:
        white_username = record["white"].lower()
        black_username = record["black"].lower()

        if white_username not in visited_players:
            new_usernames.add(white_username)
        if black_username not in visited_players:
            new_usernames.add(black_username)

    return list(new_usernames)


def crawl() -> None:
    all_games = load_existing_games()
    seen_game_urls: set[str] = {game["url"] for game in all_games}
    
    visited_players, player_queue = load_crawler_state()

    if not player_queue:
        seed_players = discover_seed_players(required_count=10)
        if not seed_players:
            logger.error("No seed players found. Cannot start crawl.")
            return

        for player_name in seed_players:
            player_queue.append(player_name.lower())

    logger.info("Starting crawl — target: %d games", TARGET_GAME_COUNT)

    while player_queue and len(all_games) < TARGET_GAME_COUNT:
        current_username = player_queue.popleft()

        if current_username in visited_players:
            continue
        visited_players.add(current_username)

        logger.info(
            "Processing player: %s  (queue: %d, games: %d/%d)",
            current_username, len(player_queue), len(all_games), TARGET_GAME_COUNT,
        )

        new_games = process_player_archives(current_username, seen_game_urls)
        
        # We always want to collect their opponents to keep the BFS queue alive
        opponent_usernames = collect_opponent_usernames(new_games, visited_players)
        for opponent in opponent_usernames:
            player_queue.append(opponent)

        if new_games:
            all_games.extend(new_games)
            save_games_to_file(all_games)
            logger.info("  Saved %d new games (total: %d)", len(new_games), len(all_games))
            
        # Save state so if it crashes, we remember who we visited and who is in queue
        save_crawler_state(visited_players, player_queue)

    logger.info("Crawl finished. Total games collected: %d", len(all_games))


if __name__ == "__main__":
    crawl()
