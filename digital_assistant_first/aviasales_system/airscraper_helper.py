import json
from typing import Optional, List, Dict, Tuple
from datetime import date
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
import time
import json
import hashlib
import os
import logging
from datetime import datetime
from itertools import chain
from concurrent.futures import ThreadPoolExecutor
from digital_assistant_first.aviasales_system.aviasales_travelpayouts_helper import TravelPayoutsHelper
import joblib
import numpy as np
# from digital_assistant_first.utils.aviasales_economy_helper import (
#     AviasalesIATAConverter,
#     AviasalesEconomyHelper,
# )

load_dotenv()

HEADERS = {
    "x-rapidapi-host": "sky-scrapper.p.rapidapi.com",
    "x-rapidapi-key": "88a3fe1cffmshfdd45650e17f575p146a66jsn944467eaa8c2"
}

TICKETS_QUERY_URL = "https://sky-scrapper.p.rapidapi.com/api/v2/flights/searchFlightsWebComplete"

# # TO REMOVE
import yaml


def load_config_yaml(config_file="config.yaml"):
    """Загрузить конфигурацию из YAML-файла."""
    with open(config_file, "r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)
    return config_yaml


config = load_config_yaml("config.yaml")
logger = 1
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.system(f'export OPENAI_API_KEY="{OPENAI_API_KEY}"')
model = ChatOpenAI(model="gpt-4o-mini")
# ### TO REMOVE

class AirscraperHelper:
    def __init__(self, logger, aviasales_json):
        self.logger = logger
        self.aviasales_json = aviasales_json

    def _query_cities(self):
        origin = self.aviasales_json["origin"]
        destination = self.aviasales_json["destination"]

        def get_airport_data(query):
            resp = requests.get(
                f"https://sky-scrapper.p.rapidapi.com/api/v1/flights/searchAirport?query={query}&locale=en-US",
                headers=HEADERS
            )
            return resp
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            orig_resp_future = executor.submit(get_airport_data, origin)
            dest_resp_future = executor.submit(get_airport_data, destination)
            orig_resp = orig_resp_future.result()
            dest_resp = dest_resp_future.result()
        
        if orig_resp.status_code != 200 or dest_resp.status_code != 200:
            self.logger.error(f"Error querying cities: {orig_resp.status_code} or {dest_resp.status_code}")
            return None
        
        orig_data = orig_resp.json()["data"][0]
        dest_data = dest_resp.json()["data"][0]

        orig_sky_id = orig_data["skyId"]
        dest_sky_id = dest_data["skyId"]

        orig_entity_id = orig_data["entityId"]
        dest_entity_id = dest_data["entityId"]
        
        return {"orig_sky_id": orig_sky_id, "dest_sky_id": dest_sky_id, "orig_entity_id": orig_entity_id, "dest_entity_id": dest_entity_id}
    
    def _query_tickets(self, airscraper_iata_data):

        tries = 0

        while tries < 3:
            params = {
                "originSkyId": airscraper_iata_data["orig_sky_id"],
                "destinationSkyId": airscraper_iata_data["dest_sky_id"],
                "originEntityId": airscraper_iata_data["orig_entity_id"],
                "destinationEntityId": airscraper_iata_data["dest_entity_id"],
                "date": self.aviasales_json["to"],
                "cabinClass": "business" if self.aviasales_json["travel_class"] == "C" else "economy",
                "adults": 1,
                "sortBy": "best",
                "limit": 10,
                "currency": "USD",
                "market": "en-US",
                "countryCode": "US"
            }
            if self.aviasales_json["back"]:
                params["returnDate"] = self.aviasales_json["back"]
            if self.aviasales_json["adults"] > 1:
                params["adults"] = self.aviasales_json["adults"]
            if self.aviasales_json["children"] > 0:
                params["childrens"] = self.aviasales_json["children"]
            if self.aviasales_json["infants"] > 0:
                params["infants"] = self.aviasales_json["infants"]

            response = requests.get(TICKETS_QUERY_URL, headers=HEADERS, params=params).json()
            if response["data"]["context"]["status"] == "complete":
                return response["data"]["itineraries"]
            else:
                tries += 1
                print(f"retrying... {tries}/3")
                time.sleep(10)
        return None
    

    def parse_itineraries(self, data):
        def fmt_ts(ts):
            return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S")

        results = []
        for entry in data:
            # 1) price
            price = entry.get("price", {}).get("raw") or entry.get("price", {}).get("amount")

            # 2) general departure/arrival
            leg_deps = [leg["departure"] for leg in entry.get("legs", [])]
            leg_arrs = [leg["arrival"]   for leg in entry.get("legs", [])]
            general_departure_ts = fmt_ts(min(leg_deps)) if leg_deps else None
            general_arrival_ts   = fmt_ts(max(leg_arrs)) if leg_arrs else None

            # 3) per‑segment legs info & collect airlines
            legs_info = []
            airlines = set()
            for leg in entry.get("legs", []):
                for seg in leg.get("segments", []):
                    # collect each marketing carrier name
                    mkt = seg.get("marketingCarrier", {})
                    if mkt.get("name"):
                        airlines.add(mkt["name"])

                    legs_info.append({
                        "origin_city":           seg["origin"]["parent"]["name"],
                        "origin_airport_name":   seg["origin"]["name"],
                        "origin_airport_iata":   seg["origin"]["flightPlaceId"],
                        "destination_city":      seg["destination"]["parent"]["name"],
                        "destination_airport_name": seg["destination"]["name"],
                        "destination_airport_iata": seg["destination"]["flightPlaceId"],
                        "duration":              seg.get("durationInMinutes"),
                        "departure_ts":          fmt_ts(seg["departure"]),
                        "arrival_ts":            fmt_ts(seg["arrival"]),
                        "airline":               mkt.get("name")
                    })

            # 4) urls as { price: url }
            urls = {}
            for opt in entry.get("pricingOptions", [])[:2]:
                amt = opt.get("price", {}).get("amount")
                items = opt.get("items", [])
                if items and items[0].get("url") and amt is not None:
                    urls[amt] = "https://skyscanner.com" + items[0]["url"]

            # 5) stops count
            stops_count = sum(leg.get("stopCount", 0) for leg in entry.get("legs", []))

            # 6) farePolicy flags
            fare = entry.get("farePolicy", {})

            results.append({
                "price":                   price,
                "general_departure_ts":    general_departure_ts,
                "general_arrival_ts":      general_arrival_ts,
                "legs":                    legs_info,
                "urls":                    urls,
                "stops_count":             stops_count,
                "isChangeAllowed":         fare.get("isChangeAllowed", False),
                "isPartiallyChangeable":   fare.get("isPartiallyChangeable", False),
                "isCancellationAllowed":   fare.get("isCancellationAllowed", False),
                "isPartiallyRefundable":   fare.get("isPartiallyRefundable", False),
                "airlines":                list(airlines)
            })

        return results
    
if __name__ == "__main__":
    aviasales_json = {
        "origin": "BAK",
        "destination": "TNR",
        "adults": 2,
        "children": 1,
        "infants": 1,
        "handbag": True,
        "baggage": False,
        "to": "2025-05-22",
        "back": False,
        "travel_class": "Y",
        "airlines": [],
        "blacklist_airlines": [],
        "max_stops": -1,
        "to_hour_range": "",
        "from_hour_range": "",
    }
    airscraper_helper = AirscraperHelper(logger, aviasales_json)
    cities_mapping = airscraper_helper._query_cities()
    tickets_data = airscraper_helper._query_tickets(cities_mapping)
    print(len(tickets_data))
    with open("tickets_data_airscraper.json", "w") as f:
        json.dump(tickets_data, f)
    parsed_tickets_data = airscraper_helper.parse_itineraries(tickets_data)
    with open("parsed_tickets_data_airscraper.json", "w") as f:
        json.dump(parsed_tickets_data, f)
        
        
