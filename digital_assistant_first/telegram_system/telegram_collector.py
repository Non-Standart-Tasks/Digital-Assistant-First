import os
import json
from telethon import TelegramClient
from typing import List, Dict


def ensure_data_directory():
    os.makedirs("data", exist_ok=True)

class TelegramCollector:
    def __init__(self, api_id: str, api_hash: str, phone_number: str):
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone_number = phone_number
        self.categories = {
            'restaurants': ['newinmoscow', 'yanrestoran',
                            'yanrestoranspb', 'doing_spb', 'night2daymoscow',
                            'night2day_spb'],
            'events': ['dubaiafisha', 'planrestoran'],
            'travel': ['ranarod', 'samokatus', 'dearpassengers'],

        }

    async def collect_messages(self, limit: int = 100) -> Dict[str, List[Dict]]:
        async with TelegramClient('session', self.api_id, self.api_hash) as client:
            await client.start(phone=self.phone_number)
            
            categorized_messages = {category: [] for category in self.categories.keys()}
            
            for category, channels in self.categories.items():
                for channel in channels:
                    try:
                        chat_info = await client.get_entity(channel)
                        messages = await client.get_messages(entity=chat_info, limit=limit)
                        
                        for message in messages:
                            if message.text and not hasattr(message.media, 'video'):
                                message_data = {
                                    "text": message.text,
                                    "link": f"https://t.me/{channel}/{message.id}",
                                    "date": message.date.isoformat(),
                                    "channel": channel,
                                    "category": category
                                }
                                categorized_messages[category].append(message_data)
                    except Exception as e:
                        print(f"Error collecting messages from {channel}: {e}")

            return categorized_messages

    def save_messages(self, messages: Dict[str, List[Dict]], output_file: str):
        ensure_data_directory()
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)