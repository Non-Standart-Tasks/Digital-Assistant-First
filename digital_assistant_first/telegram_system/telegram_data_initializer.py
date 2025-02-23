from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from digital_assistant_first.telegram_system.telegram_collector import TelegramCollector
from digital_assistant_first.utils.load_utils import load_config_yaml
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")
phone_number = os.getenv("TELEGRAM_PHONE_NUMBER")

config = load_config_yaml()

async def update_telegram_messages():
    collector = TelegramCollector(api_id=api_id, api_hash=api_hash, phone_number=phone_number)
    messages = await collector.collect_messages(config.get('collect_messages', 100))
    collector.save_messages(messages, "data/telegram_messages.json")

def run_update():
    asyncio.run(update_telegram_messages())


class TelegramManager:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.start_scheduler()

    def start_scheduler(self):
        """Starts the scheduler with an update every hour"""
        self.scheduler.add_job(
            run_update,
            'interval',
            minutes=20,
            id='update_telegram_messages_status',
            next_run_time=datetime.now()
        )
        self.scheduler.start()

    def stop_scheduler(self):
        """Stops the scheduler"""
        self.scheduler.shutdown()