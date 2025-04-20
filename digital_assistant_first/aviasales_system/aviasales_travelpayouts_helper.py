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
from digital_assistant_first.aviasales_system.aviasales_recommendation_engine import AviasalesRecommendationEngine
import joblib
import numpy as np
# from digital_assistant_first.utils.aviasales_economy_helper import (
#     AviasalesIATAConverter,
#     AviasalesEconomyHelper,
# )

load_dotenv()


IP = os.popen('curl -s ifconfig.me').read().strip()
PARTNER_ID = 621748
TRAVELPAYOUTS_API_KEY = os.getenv("TRAVELPAYOUTS_API_KEY")
BASE_URL = "https://api.travelpayouts.com/v1/flight_search"
FETCH_URL = "https://api.travelpayouts.com/v1/flight_search_results?uuid="
BASE_HEADERS = {"Content-type": "application/json"}
FETCH_HEADERS = {"Accept-Encoding": "gzip,deflate,sdch"}

class NothingFoundTravelPayouts(Exception):
    def __init__(self, message="Не удалось получить данные из TravelPayouts после нескольких попыток"):
        self.message = message
        super().__init__(self.message)

# # TO REMOVE
# from digital_assistant_first.utils.custom_logging import setup_logging, log_api_call
# import yaml


# def load_config_yaml(config_file="config.yaml"):
#     """Загрузить конфигурацию из YAML-файла."""
#     with open(config_file, "r", encoding="utf-8") as f:
#         config_yaml = yaml.safe_load(f)
#     return config_yaml


# config = load_config_yaml("config.yaml")
# logger = setup_logging(logging_path="logs/digital_assistant.log")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# os.system(f'export OPENAI_API_KEY="{OPENAI_API_KEY}"')
# model = ChatOpenAI(model="gpt-4o-mini")
# ### TO REMOVE

class ProvidersIATAMatcher:
    def __init__(self):
        self.vectorizer = joblib.load("providers_iata_matcher_tfidf.joblib")
        self.matcher_dict = joblib.load("providers_iata_matcher_vectors.joblib")

    def _get_vector(self, iata: str) -> np.ndarray:
        return self.vectorizer.transform([iata]).toarray()[0]
    
    def match_iata(self, iata: str) -> str:
        if not iata:
            return None
            
        query_vector = self._get_vector(iata)
        
        # Calculate cosine similarity with all vectors in matcher_dict
        similarities = {}
        for key, vector in self.matcher_dict.items():
            # Calculate cosine similarity
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            similarities[key] = similarity
        
        # Get the key with the highest similarity
        best_match = max(similarities.items(), key=lambda x: x[1])[0]
        return best_match

class AviasalesIATAConverter:
    def __init__(self):
        self.query = "https://suggest.aviasales.com/v2/places.json?locale=ru_RU&max=7&term={}&types[]=city&types[]=airport&types[]=country"

    def get_iata_codes(self, flight_info_dict: Dict[str, str]) -> List[str]:
        origin_res_raw = requests.get(
            self.query.format(flight_info_dict["origin"])
        ).json()
        destination_res_raw = requests.get(
            self.query.format(flight_info_dict["destination"])
        ).json()
        origin_res = origin_res_raw[0]["code"] if origin_res_raw else None
        destination_res = (
            destination_res_raw[0]["code"] if destination_res_raw else None
        )
        return origin_res, destination_res

class TravelPayoutsHelper:
    def __init__(self, logger, model, config, st_interface):
        self.logger = logger
        self.model = model
        self.config = config
        self.st_interface = st_interface
        self.iata_converter = AviasalesIATAConverter()
        self.providers_iata_matcher = ProvidersIATAMatcher()
        self.none_str = ""

    def _text2json_basic(
        self, user_input: str, tries: int = 0, max_retries: int = 1
    ) -> Dict[str, str] | None:
        try:
            if tries > max_retries:
                return None
            message = [
                {
                    "role": "user",
                    "content": self.config[
                        "system_prompt_tickets_for_aviasales_helper"
                    ].format(input=user_input),
                }
            ]
            response = self.model.invoke(message, stream=False).content
            response = response.replace("true", "True").replace("false", "False")
            res_json = eval("{" + response.split("{")[-1].split("}")[0] + "}")
            return res_json

        except Exception as e:
            self.logger.error(
                f"Error converting aviasales query [basic] to JSON: {e}.\nRetrying... (max retries={max_retries})"
            )
            return self._text2json_basic(user_input, tries + 1)
            # есть вероятность, что 4o-mini зафейлит валидность JSON

    def _text2_enrich_json_preferences(
        self, user_input: str, tries: int = 0, max_retries: int = 1
    ) -> Dict[str, str] | None:
        try:
            if tries > max_retries:
                return None
            message = [
                {
                    "role": "user",
                    "content": self.config[
                        "system_prompt_preferences_for_aviasales_helper"
                    ].format(input=user_input),
                }
            ]
            response = self.model.invoke(message, stream=False).content
            response = response.replace("true", "True").replace("false", "False")
            res_json = eval("{" + response.split("{")[-1].split("}")[0] + "}")
            return res_json
        
        except Exception as e:
            self.logger.error(
                f"Error converting aviasales query [preferences] to JSON: {e}.\nRetrying... (max retries={max_retries})"
            )
            return self._text2_enrich_json_preferences(user_input, tries + 1)
            # есть вероятность, что 4o-mini зафейлит валидность JSON

    def _create_dict(self, query: str) -> Dict[str, str]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            basic_future = executor.submit(self._text2json_basic, query)
            preferences_future = executor.submit(self._text2_enrich_json_preferences, query)
            
            aviasales_json = basic_future.result()
            aviasales_json_preferences = preferences_future.result()
        if aviasales_json is None:
            return self.none_str, self.none_str
        origin, destination = self.iata_converter.get_iata_codes(aviasales_json)
        if origin is None or destination is None:
            return self.none_str, self.none_str
        # Проверка даты вылета - она должна быть не раньше текущей даты
        
        curr_date, curr_yr = datetime.now().strftime("%Y-%m-%d"), datetime.now().year
        to_yr_delta, back_yr_delta = curr_yr, curr_yr
        
        if f"{to_yr_delta}-{aviasales_json['to']}" < curr_date:
            to_yr_delta += 1
            back_yr_delta += 1

        if aviasales_json['back']:
            if f"{back_yr_delta}-{aviasales_json['back']}" < f"{to_yr_delta}-{aviasales_json['to']}":
                back_yr_delta += 1

            aviasales_json["back"] = f"{back_yr_delta}-{aviasales_json['back']}"

        aviasales_json["to"] = f"{to_yr_delta}-{aviasales_json['to']}"
        aviasales_json["origin"] = origin
        aviasales_json["destination"] = destination
        aviasales_json["travel_class"] = (
            "C" if aviasales_json["travel_class"] == "business" else "Y"
        )

        if aviasales_json_preferences:
            aviasales_json["airlines"] = aviasales_json_preferences["airlines"]
            aviasales_json["blacklist_airlines"] = aviasales_json_preferences["blacklist_airlines"]
            aviasales_json["max_stops"] = aviasales_json_preferences["max_stops"]
            aviasales_json["to_hour_range"] = aviasales_json_preferences["to_hour_range"]
            aviasales_json["from_hour_range"] = aviasales_json_preferences["from_hour_range"]

        if aviasales_json["airlines"]:
            aviasales_json["airlines"] = [self.providers_iata_matcher.match_iata(i) for i in aviasales_json["airlines"]]
        if aviasales_json["blacklist_airlines"]:
            aviasales_json["blacklist_airlines"] = [self.providers_iata_matcher.match_iata(i) for i in aviasales_json["blacklist_airlines"]]

        self.logger.info(f"[NEW] Сформирован запрос для Aviasales: {aviasales_json}")
        return aviasales_json

    def _create_request_string(self, aviasales_json: Dict[str, str]) -> str:
        dict_to_request = {
            "host": "ama.vtb.msut.me",
            "locale": "ru",
            "marker": PARTNER_ID,
            "adults": aviasales_json["adults"],
            "children": aviasales_json["children"],
            "infants": aviasales_json["infants"],
            "date": aviasales_json["to"],
            "destination": aviasales_json["destination"],
            "origin": aviasales_json["origin"],
        }
        if aviasales_json['back']:
            dict_back_request = {
                "date": aviasales_json["back"],
                "destination": aviasales_json["origin"],
                "origin": aviasales_json["destination"],
                "trip_class": aviasales_json["travel_class"],
                # Поддерживаются только классы: Y, C (эконом, бизнес)
                # Комфорт пока относим к эконому
                "user_ip": IP,
            }
        else:
            dict_back_request = {
                "trip_class": aviasales_json["travel_class"],
                # Поддерживаются только классы: Y, C (эконом, бизнес)
                # Комфорт пока относим к эконому
                "user_ip": IP,
            }
        res_string = (
            TRAVELPAYOUTS_API_KEY
            + ":"
            + ":".join(list(map(str, dict_to_request.values())))
            + ":"
            + ":".join(list(map(str, dict_back_request.values())))
        )
        return res_string.strip()
    
    def _dict_to_md5_hash(self, dict_to_hash: Dict[str, str]) -> str:
        return hashlib.md5(dict_to_hash.encode()).hexdigest()
    
    def _post_search(self, aviasales_json: Dict[str, str], request_hash: str) -> str:
        if aviasales_json["back"]:
            segments = [
                {"origin": aviasales_json["origin"], "destination": aviasales_json["destination"], "date": aviasales_json["to"]},
                {"origin": aviasales_json["destination"], "destination": aviasales_json["origin"], "date": aviasales_json["back"]},
            ]
        else:
            segments = [
                {"origin": aviasales_json["origin"], "destination": aviasales_json["destination"], "date": aviasales_json["to"]},
            ]

        payload = {
            "signature": request_hash,
            "marker": "621748",
            "host": "ama.vtb.msut.me",
            "user_ip": IP,
            "locale": "ru",
            "trip_class": aviasales_json["travel_class"],
            "passengers": {"adults": aviasales_json["adults"], "children": aviasales_json["children"], "infants": aviasales_json["infants"]},
            "segments": segments
        }
        # return payload
        response = requests.post(BASE_URL, json=payload, headers=BASE_HEADERS)
        return response.json()
    
    def _fetch_search(self, search_id: str) -> str:
        tries = 0
        while tries < 5:
            if tries == 0:
                time.sleep(20)
            else:
                time.sleep(3)
            response = requests.get(FETCH_URL + search_id, headers=FETCH_HEADERS)
            if response.status_code == 200:
                res = response.json()
                if list(chain(*[i.get("proposals", []) for i in res])):
                    return res
                print(f"continue fetching... tries={tries}")
            tries += 1
        # return None
        raise NothingFoundTravelPayouts()
        # time.sleep(60)
        # response = requests.get(FETCH_URL + search_id, headers=FETCH_HEADERS)
        # res = response.json()
        # if not list(chain(*[i.get("proposals", []) for i in res])):
        #     print("no offers found!!!")
        # return response.json()

    def _get_general_link(self, aviasales_json: Dict[str, str]) -> str:
        # MOW0705TNR14053
        iata_origin = aviasales_json["origin"].upper()
        iata_destination = aviasales_json["destination"].upper()
        to_arr = aviasales_json["to"].split("-")
        back_arr = aviasales_json["back"].split("-") if aviasales_json["back"] else None
        adults, children, infants = aviasales_json["adults"], aviasales_json["children"], aviasales_json["infants"]

        if back_arr:
            base_url = f"{iata_origin}{to_arr[2]}{to_arr[1]}{iata_destination}{back_arr[2]}{back_arr[1]}{adults}"
        else:
            base_url = f"{iata_origin}{to_arr[2]}{to_arr[1]}{iata_destination}{adults}"

        if infants > 0:
            base_url += f"{children}{infants}"
        elif children > 0:
            base_url += f"{children}"

        return "https://www.aviasales.ru/search/" + base_url
    
    def launch_pipeline(self, user_query: str) -> str:
        # try:
            with self.st_interface.spinner("Поиск билетов..."):
                aviasales_json = self._create_dict(user_query)
                if aviasales_json == self.none_str:
                    return self.none_str
                request_string = self._create_request_string(aviasales_json)
                request_hash = self._dict_to_md5_hash(request_string)
                search_response = self._post_search(aviasales_json, request_hash)
                print()
                print(search_response["search_id"])
                print()
                search_json = self._fetch_search(search_response["search_id"])
                with open("digital_assistant_first/aviasales_system/search_json_received.json", "w") as f:
                    json.dump(search_json, f)
                with open("digital_assistant_first/aviasales_system/aviasales_json_sent.json", "w") as f:
                    json.dump(aviasales_json, f)
            self.st_interface.success("✅ Поиск авиабилетов завершен! Формируем рекомендации...")
            recommendation_engine = AviasalesRecommendationEngine(self.logger, search_json, aviasales_json)
            md_res = recommendation_engine._launch_pipeline()
            return md_res, self._get_general_link(aviasales_json)
        
        # except NothingFoundTravelPayouts as e:
        #     self.logger.error(f"Ничего не найдено в Aviasales после нескольких попыток: {e}")
        #     return self.none_str, self.none_str
        # except Exception as e:
        #     self.logger.error(f"Ошибка в launch_pipeline: {e}")
        #     return self.none_str, self.none_str

    # {            
# "host": "ama.vtb.msut.me",
# "locale": "ru",
# "marker": "621748",
# "adults": 1,
# "children": 0,
# "infants": 0,
# "date": "2025-05-25",
# "destination": "BAK",
# "origin": "MOW"
# }
# d1 = {
# "date": "2025-06-18",
# "destination": "MOW",
# "origin": "BAK",
# "trip_class": "Y",
# "user_ip": "51.158.200.31"
# }

# if __name__ == "__main__":
#     helper = TravelPayoutsHelper(logger, model)
#     helper.launch_pipeline(
#         """
# москва-тбилиси
# 7/05 вылет вечером, обратно 14/05 утром
# 1 взрослый
# комфорт, без багажа
# макс кол-во пересадок 2
#         """
#     )
