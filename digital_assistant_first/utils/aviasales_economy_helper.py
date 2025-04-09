import json
from typing import Optional, List, Dict, Tuple
from datetime import date
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
import time
import json # rm this

load_dotenv()

months_ru = {
    1: "января", 2: "февраля", 3: "марта", 4: "апреля",
    5: "мая", 6: "июня", 7: "июля", 8: "августа",
    9: "сентября", 10: "октября", 11: "ноября", 12: "декабря"
}

def format_date(date_str):
    from datetime import datetime
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    return f"{date_obj.day} {months_ru[date_obj.month]}"

class AviasalesIATAConverter:
    def __init__(self):
        self.query = "https://suggest.aviasales.com/v2/places.json?locale=ru_RU&max=7&term={}&types[]=city&types[]=airport&types[]=country"

    def get_iata_codes(self, flight_info_dict: Dict[str, str]) -> List[str]:
        origin_res_raw = requests.get(self.query.format(flight_info_dict["origin"])).json()
        destination_res_raw = requests.get(self.query.format(flight_info_dict["destination"])).json()
        origin_res = origin_res_raw[0]['code'] if origin_res_raw else None
        destination_res = destination_res_raw[0]['code'] if destination_res_raw else None
        return origin_res, destination_res
    
class AviasalesAriadne:
    def __init__(self, logger):
        self.link = "https://ariadne.aviasales.com/api/gql"
        self.template_post = {
            "query": "\nquery FlexibleAICalendarPromptV3($input: AiFlexibleCalendarPromptV3Input!) {\n    ai_flexible_calendar_prompt_v3(input: $input) {\n      id\n      datacenter\n    }\n  }\n",
            "variables": {
                "input": {
                    "origin": "MOW",
                    "origin_type": "CITY", 
                    "destination": "BAK",
                    "destination_type": "CITY",
                    "locale": "ru_RU",
                    "currency": "rub",
                    "query": "Москва - Баку - Москва, 21-27.04. Аэрофлот\n2 взр. + реб. 12 лет\nТуда: утро\nОбратно: после обеда",
                    "session_id": None
                }
            },
            "operation_name": "ai_flexible_calendar_prompt_v3"
        }
        self.template_fetch = {
                "query": "\n  query FlexibleAICalendarResultsV3($input: AiFlexibleCalendarResultsV3Input!, $locales: [String!], $brand: Brand!) {\n    ai_flexible_calendar_results_v3(input: $input, brand: $brand) {\n      status\n      tickets {\n        data {\n          ...priceFields\n        }\n        badge {\n          name,\n          colors {\n            light,\n            dark\n          }\n        }\n      }\n      places {\n        cities {\n          ...citiesFields\n        }\n        airlines {\n          ...airlinesFields\n        }\n        airports {\n          ...airportsFields\n        }\n      }\n    }\n  }\n  \nfragment priceFields on Price {\n  depart_date\n  return_date\n  value\n  cashback\n  found_at\n  signature\n  ticket_link\n  currency\n  provider\n  with_baggage\n  segments {\n    transfers {\n      duration_seconds\n      country_code\n      visa_required\n      night_transfer\n      at\n      to\n      tags\n    }\n    flight_legs {\n      origin\n      destination\n      local_depart_date\n      local_depart_time\n      local_arrival_date\n      local_arrival_time\n      flight_number\n      operating_carrier\n      aircraft_code\n      technical_stops\n      equipment_type\n      duration_seconds\n    }\n  }\n}\n\n  \nfragment airlinesFields on Airline {\n  iata\n  translations(filters: {locales: $locales})\n}\n  \nfragment citiesFields on CityInfo {\n  city {\n    iata\n    translations(filters: {locales: $locales})\n  }\n}\n  \nfragment airportsFields on Airport {\n  iata\n  translations(filters: {locales: $locales})\n  city {\n    iata\n    translations(filters: {locales: $locales})\n  }\n}\n  ",
                "variables": {
                    "brand": "AS",
                    "locales": ["ru"],
                    "input": {
                        "datacenter": "EU_CENTRAL_1",
                        "id": "70215a4a-bccf-4558-b9fd-6c1ff94bbfcd",
                    },
                },
                "operation_name": "ai_flexible_calendar_results_v3",
            }
        self.logger = logger
        self.headers = {"Content-Type": "application/json"}

    def post(self, aviasales_json: Dict[str, str], tries: int = 0, max_retries: int = 1) -> Dict[str, str] | None:    
        try:
            if tries > max_retries:
                return None
            
            template = self.template_post.copy()
            template["variables"]["input"]["origin"] = aviasales_json["origin"]
            template["variables"]["input"]["destination"] = aviasales_json["destination"]
            template["variables"]["input"]["query"] = aviasales_json["date_query"] + ". Верни все варианты, которые соответствуют этому условию."
            
            response = requests.post(self.link, json=template, headers=self.headers)
            if response.status_code == 200:
                print(response.json()["data"]["ai_flexible_calendar_prompt_v3"])
                return response.json()["data"]["ai_flexible_calendar_prompt_v3"]
            else:
                raise ConnectionError(f"Error posting to Aviasales Ariadne: {response.status_code}.\nRetrying... (max retries={max_retries})")
        except Exception as e:
            self.logger.error(e)
            return self.post(aviasales_json, tries + 1, max_retries)
    
    def fetch(self, ariadne_json: Dict[str, str]) -> Dict[str, str]:
        template = self.template_fetch.copy()
        template["variables"]["input"] = {
            "datacenter": ariadne_json["datacenter"],
            "id": ariadne_json["id"]
        }
        response = requests.post(self.link, json=template, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            return None

class AviasalesEconomyHelper:
    def __init__(self, model, config, logger):
        self.iata_converter = AviasalesIATAConverter()
        self.ariadne = AviasalesAriadne(logger)
        self.prompt = config["system_prompt_tickets_for_aviasales_economy_helper"]
        self.model = model
        self.logger = logger
        self.none_str = ""

    def _text2json(self, user_input: str, tries: int = 0, max_retries: int = 1) -> Dict[str, str] | None:
        try:
            if tries > max_retries:
                return None
            message = [
                {"role": "user", "content": self.prompt.format(input=user_input)}
            ]
            response = self.model.invoke(message, stream=False) 
            res_json = eval("{" + response.content.split("{")[-1].split("}")[0] + "}")
            return res_json
        
        except Exception as e:
            self.logger.error(f"Error converting aviasales query to JSON: {e}.\nRetrying... (max retries={max_retries})")
            return self._text2json(user_input, tries + 1)
            # есть вероятность, что 4o-mini зафейлит валидность JSON

    def _ariadne_res_to_md(
        self,
        ariadne_res: Dict[str, str],
        new_template: bool = True,
        adults: int = 1,
        children: int = 0,
        infants: int = 0,
    ) -> str:
        
        def _link_corrector_for_num_passengers(
            link: str, adults: int = 1, children: int = 0, infants: int = 0
        ) -> str:
            strp_link = link.split("?")
            base_link, param_link = strp_link[:1][0][:-1], strp_link[1:][0]
            children_corr, infants_corr = "", ""
            if infants > 0 or children > 0:
                children_corr = str(children)
            if infants > 0:
                infants_corr = str(infants)
            res_code = f"{adults}{children_corr}{infants_corr}"
            return f"{base_link}{res_code}?{param_link}"

        aviasales_url = ""

        def __extract_ru_su_translations(data):
            return {
                "cities": {
                    c.get("city", {}).get("iata"): c.get("city", {}).get("translations", {}) \
                    .get("ru", {}).get("su", "")
                    for c in data.get("cities", [])
                    if c.get("city", {}).get("iata")
                },
                "airports": {
                    a.get("iata"): a.get("translations", {}).get("ru", {}).get("su", "")
                    for a in data.get("airports", [])
                    if a.get("iata")
                },
                "airlines": {
                    l.get("iata"): l.get("translations", {}).get("ru", {}).get("su", "")
                    for l in data.get("airlines", [])
                    if l.get("iata")
                }
            }
        
        try:
            res_data = ariadne_res["tickets"]
            if len(res_data) > 0:
                places_data = ariadne_res["places"]
                md_str = "\n"
                url_set_flag = 0
                if places_data:
                    mapping_data = __extract_ru_su_translations(places_data)
                else:
                    mapping_data = {}

                if new_template:
                    # Новый лэйаут, запрошенный в беседе ЦП Трэвел
                    # Пока без информации о цене и пр. (см. синий ромб в их шаблоне)
                    for k in res_data:
                        badge = f"{k['badge']['name']['ru']}" if k['badge'] else ""
                        segment_lines = []
                        airline_names = set()
                        prefix_to_back = ""
                        segm_len = len(k['data']['segments'])
                        for c, segment in enumerate(k['data']['segments']):
                            for c0, leg in enumerate(segment['flight_legs']):
                                if segm_len > 1 and c0 == 0:
                                    if c == 0:
                                        prefix_to_back = "**Туда**"
                                    elif c == 1:
                                        prefix_to_back = "**Обратно**"
                                else:
                                    prefix_to_back = ""
                                
                                carrier = leg['operating_carrier']
                                orig = leg['origin']
                                dest = leg['destination']
                                depart_date = leg['local_depart_date']
                                arr_date = leg['local_arrival_date']
                                depart_time = leg['local_depart_time']
                                arr_time = leg['local_arrival_time']
                                depart_date = format_date(depart_date)
                                arr_date = format_date(arr_date)

                                carrier_name = mapping_data.get('airlines', {}).get(carrier, carrier)
                                orig_name = mapping_data.get('airports', {}).get(orig, orig)
                                dest_name = mapping_data.get('airports', {}).get(dest, dest)

                                airline_names.add(carrier_name)
                                segment_lines.append(f"{prefix_to_back}\n\n{depart_date}, {orig_name} {orig} {depart_time} – {dest_name} {dest} {arr_time} \n\n")
                        
                        corr_url = _link_corrector_for_num_passengers(k['data']['ticket_link'], adults, children, infants)
                        md_str += f"**✈️ {', '.join(airline_names)} :blue-background[*{badge}*]**\n\n"
                        md_str += "\n".join(segment_lines) + "\n"
                        ticket_url = f"https://www.aviasales.ru/search{corr_url}"
                        md_str += f"🔹 [Ссылка на билет]({ticket_url})\n\n---\n\n"

                        if not url_set_flag:
                            aviasales_url = ticket_url.split("?")[0]
                            url_set_flag = 1

                else:
                    
                    # Базовый табличный лэйаут
                    md_str += "| Особенности | Ссылка на билет | Авиакомпания | Маршрут | Вылет | Прилет |\n"
                    md_str += "|-------------|-----------------|--------------|---------|-------|--------|\n"
                    
                    for k in res_data:
                        badge = f"{k['badge']['name']['ru']}" if k['badge'] else ""
                        
                        for l in k['data']['segments']:
                            for idx, m in enumerate(l['flight_legs']):
                                carrier, orig, dest = m['operating_carrier'], m['origin'], m['destination']
                                depart_date, arr_date = m['local_depart_date'], m['local_arrival_date']
                                depart_time, arr_time = m['local_depart_time'], m['local_arrival_time']

                                carrier_name = mapping_data.get('airlines', {}).get(carrier, carrier)
                                orig_name = mapping_data.get('airports', {}).get(orig, orig)
                                dest_name = mapping_data.get('airports', {}).get(dest, dest)

                                if idx == 0:
                                    md_str += (
                                        f"| {'**'+badge+'**' if badge.strip() else ''} "
                                        # f"| {k['data']['value']} "
                                        # f"| {'включен' if k['data']['with_baggage'] else 'не включен'} "
                                        # f"| {k['data']['provider']} "
                                        f"| [Ссылка на билет](https://www.aviasales.ru/search{k['data']['ticket_link']})"
                                    )
                                    if not url_set_flag:
                                        aviasales_url = f"https://www.aviasales.ru/search{k['data']['ticket_link'].split('?')[0]}"
                                        url_set_flag = 1
                                else:
                                    md_str += "| | "
                                
                                md_str += (
                                    f"| {carrier_name} "
                                    f"| {orig_name} -> {dest_name} "
                                    f"| {depart_date} {depart_time} "
                                    f"| {arr_date} {arr_time} |\n"
                                )
                                # print(md_str)
                
                return md_str, aviasales_url
            return self.none_str, self.none_str
        except Exception as e:
            self.logger.error(f"Ошибка конвертации ответа Aviasales Ariadne в Markdown: {e}")
            return self.none_str, self.none_str

    def process_user_input(self, user_input: str, st_interface) -> Dict[str, str]:
        try:
            aviasales_json = self._text2json(user_input)
            if aviasales_json is None:
                return self.none_str, self.none_str
            origin, destination = self.iata_converter.get_iata_codes(aviasales_json)
            if origin is None or destination is None:
                return self.none_str, self.none_str
            aviasales_json["origin"] = origin
            aviasales_json["destination"] = destination

            self.logger.info(f"Сформирован запрос для Aviasales: {aviasales_json}")
            
            query_json = self.ariadne.post(aviasales_json)
            if query_json is not None:
                with st_interface.spinner("Поиск авиабилетов..."):
                    while True:
                        res = self.ariadne.fetch(query_json)
                        if res.get("data", {}).get("ai_flexible_calendar_results_v3", {}).get("status") == "IN_PROGRESS":
                            time.sleep(1)
                            continue
                        break
                st_interface.success("✅ Поиск авиабилетов завершен")
                md_str, aviasales_url = self._ariadne_res_to_md(
                    res["data"]["ai_flexible_calendar_results_v3"],
                    adults=aviasales_json.get("adults", 1),
                    children=aviasales_json.get("children", 0),
                    infants=aviasales_json.get("infants", 0),
                )
                return md_str, aviasales_url
            else:
                return self.none_str, self.none_str
        except Exception as e:
            self.logger.error(f"Ошибка при обработке запроса по авиабилетам: {e}")
            return self.none_str, self.none_str


