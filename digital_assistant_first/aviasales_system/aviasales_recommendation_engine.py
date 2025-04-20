import json
import pandas as pd
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from datetime import datetime
import re
from functools import reduce
from operator import mul
import numpy as np

ru_months = {
    1: 'января', 2: 'февраля', 3: 'марта', 4: 'апреля', 
    5: 'мая', 6: 'июня', 7: 'июля', 8: 'августа',
    9: 'сентября', 10: 'октября', 11: 'ноября', 12: 'декабря'
}

pass_mapping_0 = {
    "adults": "взрослых",
    "children": "детей",
    "infants": "младенцев"
}

pass_mapping_1 = {
    "adults": "взрослого",
    "children": "ребенка",
    "infants": "младенца"
}

_PC_RE = re.compile(r"1PC(\d+)(?:x(\d+)x(\d+)x(\d+))?$")

def minimal_baggage_limits(matrix):

    # helper: detect any falsy item anywhere in the structure
    def has_falsy(node):
        if isinstance(node, list):
            return any(has_falsy(x) for x in node)
        return not node

    if has_falsy(matrix):
        return ""

    # flatten, parse and bucketise
    weight_only, weight_dims = [], []
    def collect(node):
        for x in node:
            if isinstance(x, list):
                collect(x)
            else:
                m = _PC_RE.fullmatch(str(x))
                if m:
                    w = int(m.group(1))
                    if m.group(2):
                        dims = tuple(map(int, m.groups()[1:]))
                        weight_dims.append((w, dims, x))
                    else:
                        weight_only.append((w, x))
    collect(matrix)

    # case 3 – weight‑only tokens exist
    if weight_only:
        min_w = min(w for w, _ in weight_only)
        return next(s for w, s in weight_only if w == min_w)

    # case 4 – only weight+dim tokens
    if weight_dims:
        min_w = min(w for w, *_ in weight_dims)
        # choose the one with the smallest volume
        best = min(
            ((reduce(mul, dims), s) for w, dims, s in weight_dims if w == min_w),
            key=lambda t: t[0],
        )[1]
        return best

    return ""  # nothing parsable

def format_date_russian(date_str):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    
    day = date_obj.day
    month = ru_months[date_obj.month]
    year = date_obj.year
    
    return f"{day} {month}"

def format_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    
    if days > 0:
        return f"{days}дн {hours}ч {minutes}мин"
    else:
        return f"{hours}ч {minutes}мин"
    
def format_passengers(num, pass_type):
    if num % 10 == 1:
        return f"{num} {pass_mapping_1[pass_type]}"
    else:
        return f"{num} {pass_mapping_0[pass_type]}"
    
def summarize_exchange_return(fares_a, fares_b, mode="functional"):
    fares_chain = list(chain(*(fares_a, fares_b)))
    def restricted(key):
        for d in fares_chain:
            if not isinstance(d, dict) or key not in d or "available" not in d[key] or d[key]["available"] is False:
                return False
        return True

    if mode == "get_str":
        if not fares_chain:
            return "без обмена, без возврата"
        has_change = restricted("change_before_flight")
        has_refund = restricted("return_before_flight")
        part_change = "с обменом" if has_change else "без обмена"
        part_refund = "с возвратом" if has_refund else "без возврата"
        return f"{part_change}, {part_refund}"
    return restricted("change_before_flight"), restricted("return_before_flight")

def round_price_k(price, num):
    return round(price / num) * num

with open("digital_assistant_first/aviasales_system/iata_airports.json", "r") as f:
    airports_mapping = pd.DataFrame(json.load(f))

with open("digital_assistant_first/aviasales_system/iata_cities.json", "r") as f:
    cities_mapping = pd.DataFrame(json.load(f))

with open("digital_assistant_first/aviasales_system/iata_airlines.json", "r") as f:
    airlines_mapping = pd.DataFrame(json.load(f))

airports_mapping.name = airports_mapping.apply(
    lambda x: x["name"] if x["name"] else x["name_translations"]["en"]
, axis=1)
airlines_mapping.name = airlines_mapping.apply(
    lambda x: x["name"] if x["name"] else x["name_translations"]["en"]
, axis=1)

cities_mapping.set_index("code", inplace=True)
airports_mapping.set_index("code", inplace=True)
airlines_mapping.set_index("code", inplace=True)

cities_mapping = cities_mapping[["name"]].copy()
airports_mapping = airports_mapping[["name", "city_code"]].copy()
airlines_mapping = airlines_mapping[["name", "is_lowcost"]].copy()

class AviasalesRecommendationEngine:
    def __init__(self, logger, search_json_received, aviasales_json_sent):
        self.search_json_received = search_json_received
        self.aviasales_json_sent = aviasales_json_sent
        self.logger = logger
        self.proposals_df_full = None
        self.viable_proposals = None
        self.fastest_optimal = None
        self.cheapest_optimal = None
        self.cheapest = None
        self.viable_proposals_other = None

    def _data_preparation(self):
        proposals = list(chain(*[i.get("proposals", []) for i in self.search_json_received]))
        proposals_df = pd.DataFrame(proposals)
        proposals_df["pricing_info"] = proposals_df.xterms.apply(
            lambda x: [
                {k: v for k, v in i.items() if k not in ["baggage_source", "handbags_source"]}
                for i in list(list(x.values())[0].values())
            ]
        )
        proposals_df.drop(
            columns=[
                "terms",
                "xterms",
                "sign",
                "flight_weight",
                "validating_carrier",
                "min_stop_duration",
                "is_charter",
                "is_direct",
                "popularity",
            ],
            inplace=True,
        )

        proposals_df["idx"] = proposals_df.index
        proposals_df_full = proposals_df.explode("pricing_info").reset_index(drop=True)
        proposals_df_full["currency"] = proposals_df_full.pricing_info.apply(lambda x: x["currency"])
        proposals_df_full["price"] = proposals_df_full.pricing_info.apply(lambda x: x["price"])
        proposals_df_full["url"] = proposals_df_full.pricing_info.apply(lambda x: x["url"])
        proposals_df_full["flights_baggage"] = proposals_df_full.pricing_info.apply(lambda x: x["flights_baggage"])
        proposals_df_full["flights_handbags"] = proposals_df_full.pricing_info.apply(lambda x: x["flights_handbags"])
        proposals_df_full["flight_additional_tariff_infos"] = proposals_df_full.pricing_info.apply(lambda x: x["flight_additional_tariff_infos"])
        proposals_df_full.drop(columns=["pricing_info"], inplace=True)

        proposals_df_full["viable_handbags"] = proposals_df_full.flights_handbags.apply(
            minimal_baggage_limits
        )

        proposals_df_full["viable_baggage"] = proposals_df_full.flights_baggage.apply(
            minimal_baggage_limits
        )

        proposals_df_full["has_handbags"] = proposals_df_full.viable_handbags.apply(lambda x: x != "")
        proposals_df_full["has_baggage"] = proposals_df_full.viable_baggage.apply(lambda x: x != "")

        proposals_df_full["flight_to"] = proposals_df_full.segment.apply(lambda x: x[0])
        proposals_df_full["flight_back"] = proposals_df_full.segment.apply(lambda x: x[1] if len(x) > 1 else {})

        # proposals_df_full["duration_to"] = proposals_df_full.segment_durations.apply(lambda x: x[0])
        # proposals_df_full["duration_back"] = proposals_df_full.segment_durations.apply(lambda x: x[1] if len(x) > 1 else None)

        proposals_df_full["tariff_to"] = proposals_df_full.flight_additional_tariff_infos.apply(lambda x: x[0])
        proposals_df_full["tariff_back"] = proposals_df_full.flight_additional_tariff_infos.apply(lambda x: x[1] if len(x) > 1 else None)

        proposals_df_full["rounded_price_5k"] = proposals_df_full["price"].apply(lambda x: round_price_k(x, 10_000)) # chngd from 5k

        proposals_df_full["has_exchange"] = proposals_df_full.apply(
            lambda row: summarize_exchange_return(row["tariff_to"], row["tariff_back"])
            if row["tariff_back"]
            else summarize_exchange_return(row["tariff_to"], row["tariff_to"]),
            axis=1,
        )
        proposals_df_full["has_return"] = proposals_df_full.has_exchange.apply(lambda x: x[1])
        proposals_df_full["has_exchange"] = proposals_df_full.has_exchange.apply(lambda x: x[0])
        # proposals_df_full["rounded_price_10k"] = proposals_df_full["price"].apply(lambda x: round_price_k(x, 10_000))

        proposals_df_full["flight_to_time"] = proposals_df_full.flight_to.apply(lambda x: int(x["flight"][0]["departure_time"].split(":")[0]))
        proposals_df_full["flight_back_time"] = proposals_df_full.flight_back.apply(lambda x: int(x["flight"][0]["departure_time"].split(":")[0]) if x else 0)
        return proposals_df_full

    def _apply_filters(self):
        min_hr_to, max_hr_to = 0, 24
        min_hr_back, max_hr_back = 0, 24

        if self.aviasales_json_sent.get("to_hour_range", None):
            min_hr_to, max_hr_to = list(map(int, self.aviasales_json_sent["to_hour_range"].split("-")))
        if self.aviasales_json_sent.get("from_hour_range", None):
            min_hr_back, max_hr_back = list(map(int, self.aviasales_json_sent["from_hour_range"].split("-")))

        handbag_required = self.aviasales_json_sent.get("handbag", False)
        baggage_required = self.aviasales_json_sent.get("baggage", False)

        max_stops = self.aviasales_json_sent.get("max_stops", -1)
        if max_stops == -1:
            max_stops = np.inf
        
        preferred_airlines = set(self.aviasales_json_sent.get("airlines", []))
        blacklist_airlines = set(self.aviasales_json_sent.get("blacklist_airlines", []))

        self.proposals_df_full["has_blacklisted_airlines"] = self.proposals_df_full.carriers.apply(
            lambda x: True if len(set(x).intersection(blacklist_airlines)) > 0 else False
        )

        if preferred_airlines:
            self.proposals_df_full["has_preferred_airlines"] = self.proposals_df_full.carriers.apply(
                lambda x: True if len(set(x).intersection(preferred_airlines)) > 0 else False
            )
        else:
            self.proposals_df_full["has_preferred_airlines"] = True

        viable_proposals = self.proposals_df_full[
            (self.proposals_df_full.flight_to_time >= min_hr_to)
            & (self.proposals_df_full.flight_to_time <= max_hr_to)
            & (self.proposals_df_full.flight_back_time >= min_hr_back)
            & (self.proposals_df_full.flight_back_time <= max_hr_back)
            & (self.proposals_df_full.has_handbags == handbag_required)
            & (self.proposals_df_full.has_baggage == baggage_required)
            & (self.proposals_df_full.max_stops <= max_stops)
            & (~self.proposals_df_full.has_blacklisted_airlines)
            & (self.proposals_df_full.has_preferred_airlines)
        ].copy()
        return viable_proposals
        
    def _form_recommendations(self) -> str:
        indices_already_displayed = []

        min_time_values = self.viable_proposals["total_duration"].nsmallest(2).unique()
        fastest_optimal = self.viable_proposals[self.viable_proposals["total_duration"].isin(min_time_values)].sort_values("price").iloc[:2]
        self.viable_proposals.drop(fastest_optimal.index, inplace=True)
        indices_already_displayed.extend(fastest_optimal.idx.to_list())

        self.viable_proposals = self.viable_proposals[~self.viable_proposals.idx.isin(indices_already_displayed)].copy()
        min_price = self.viable_proposals["rounded_price_5k"].min()
        cheapest_optimal = self.viable_proposals[self.viable_proposals["rounded_price_5k"] == min_price].sort_values("total_duration").iloc[:2]
        self.viable_proposals.drop(cheapest_optimal.index, inplace=True)
        indices_already_displayed.extend(cheapest_optimal.idx.to_list())

        self.viable_proposals = self.viable_proposals[~self.viable_proposals.idx.isin(indices_already_displayed)].copy()
        cheapest = self.viable_proposals.sort_values("price").iloc[:2]
        self.viable_proposals.drop(cheapest.index, inplace=True)
        indices_already_displayed.extend(cheapest.idx.to_list())

        viable_proposals_other = self.proposals_df_full[~self.proposals_df_full.idx.isin(indices_already_displayed)].copy()

        viable_proposals_other_cheapest = viable_proposals_other.loc[
            viable_proposals_other.groupby("idx")["price"].idxmin()
        ].reset_index(drop=True).sort_values("total_duration").iloc[:1]

        viable_proposals_other_fastest = viable_proposals_other.loc[
            viable_proposals_other.groupby("idx")["total_duration"].idxmin()
        ].reset_index(drop=True).sort_values("price").iloc[:1]
        
        viable_proposals_other = pd.concat([viable_proposals_other_cheapest, viable_proposals_other_fastest], axis=0)

        return (
            fastest_optimal if fastest_optimal.shape[0] > 0 else None,
            cheapest_optimal if cheapest_optimal.shape[0] > 0 else None,
            cheapest if cheapest.shape[0] > 0 else None,
            viable_proposals_other if viable_proposals_other.shape[0] > 0 else None,
        )

    def _format_recommendations(self) -> str:

        def _basic_fmt(df, label):
            if not isinstance(df, pd.DataFrame):
                return "\n\n"
            template_res_all = f"### :blue-background[{label}:]\n\n---\n\n"
            for idx, smpl in df.iterrows():
                # try:
                    template_res = ""

                    url_ = smpl.url

                    flight_info_to = smpl.flight_to["flight"]
                    transfers_info_to = smpl.flight_to.get("transfers", None)

                    flight_info_back = smpl.flight_back.get("flight", None)
                    transfers_info_back = smpl.flight_back.get("transfers", None)

                    class_ = flight_info_to[0]["trip_class"]
                    class_ = "Эконом, " if class_ == "Y" else ("Бизнес, " if class_ == "C" else "")

                    # TO ---
                    if flight_info_back:
                        all_carriers = [airlines_mapping.loc[i["operating_carrier"]]["name"] for i in flight_info_to + flight_info_back]
                    else:
                        all_carriers = [airlines_mapping.loc[i["operating_carrier"]]["name"] for i in flight_info_to]
                    
                    all_carriers = sorted(set(all_carriers), key=lambda x: all_carriers.index(x))
                    template_res += "#### " + ", ".join(all_carriers) + "\n\n"

                    for c, (flight_info, transfers_info) in enumerate(zip(
                        [flight_info_to, flight_info_back],
                        [transfers_info_to, transfers_info_back]
                    )):
                        if c == 0 and flight_info_back:
                            template_res += "**Туда -** "
                        if c == 1 and flight_info_back:
                            template_res += "**Обратно -** "

                        stops_all = []
                        stops_all_info = []

                        if transfers_info:
                            for transfer_i in transfers_info:
                                stops_all.append(transfer_i["at"])
                                stops_all_info.append(transfer_i["duration_seconds"])

                        if len(stops_all) == 1:
                            template_res += f"Пересадка в городе {', '.join(cities_mapping.loc[airports_mapping.loc[stops_all]['city_code']]['name'])}:\n\n"
                        elif len(stops_all) > 1:
                            template_res += f"Пересадки в городах {', '.join(cities_mapping.loc[airports_mapping.loc[stops_all]['city_code']]['name'])}:\n\n"
                        else:
                            template_res += "Прямой:\n\n"

                        for c, flight_i in enumerate(flight_info):
                            if c > 0:
                                template_res += f"Пересадка {format_seconds(stops_all_info[c-1])}\n\n"
                            airport_to, airport_from = flight_i["departure"], flight_i["arrival"]
                            airport_to_naming, airport_from_naming = airports_mapping.loc[airport_to]["name"], airports_mapping.loc[airport_from]["name"]
                            template_res += f"{format_date_russian(flight_i['departure_date'])}, {airport_to_naming} {airport_to} - {flight_i['departure_time']} {airport_from_naming} {airport_from} {flight_i['arrival_time']}\n\n"
                        
                        if not flight_info_back:
                            break

                    pass_string = ""
                    for k, v in self.aviasales_json_sent.items():
                        if k == "adults":
                            pass_string += format_passengers(v, "adults") + ", "
                        elif k == "children":
                            if v > 0:
                                pass_string += format_passengers(v, "children") + ", "
                        elif k == "infants":
                            if v > 0:
                                pass_string += format_passengers(v, "infants")
                            
                    handbag_string = f"ручная кладь {smpl.viable_handbags.replace('1PC', '')}кг" if smpl.viable_handbags else "без ручной клади"
                    baggage_string = f"багаж {smpl.viable_baggage.replace('1PC', '')}кг" if smpl.viable_baggage else "без багажа"

                    if smpl.tariff_back:
                        template_res += f"🔹 {smpl.price} руб. / за {pass_string.strip().strip(',')}, {handbag_string}, {baggage_string} "\
                                        f"/ {class_}{summarize_exchange_return(smpl.tariff_to, smpl.tariff_back, 'get_str')} \n\n[{url_}]"
                    else:
                        template_res += f"🔹 {smpl.price} руб. / за {pass_string.strip().strip(',')}, {handbag_string}, {baggage_string} "\
                                        f"/ {class_}{summarize_exchange_return(smpl.tariff_to, smpl.tariff_to, 'get_str')} \n\n[{url_}]"

                    ### (additional options)
                    options_list = ["has_handbags", "has_baggage", "has_exchange", "has_return"]

                    other_variants = (
                        self.proposals_df_full[self.proposals_df_full.idx == smpl.idx]
                        .sort_values("price")
                        .drop_duplicates(subset=options_list, keep="first")
                    ).iloc[:2]
                    possible_improvements = []

                    improvements_mapping = {
                        "has_handbags": "с ручной кладью",
                        "has_baggage": "с багажом",
                        "has_exchange": "с обменом",
                        "has_return": "с возвратом"
                    }

                    for i in options_list:
                        if not smpl[i]:
                            possible_improvements.append(i)

                    if other_variants.shape[0] > 1:
                        other_improvements = []
                        for idx, row in other_variants.iterrows():
                            row_improvements_i = {idx: []}
                            for i in possible_improvements:
                                if row[i]:
                                    row_improvements_i[idx].append(i)
                            if len(row_improvements_i[idx]):
                                other_improvements.append(row_improvements_i)
                        if len(other_improvements) > 0:
                            template_res += "\n\nВозможные улучшения:"
                            for row_improvements_i in other_improvements:
                                for k, v in row_improvements_i.items():
                                    if len(v) > 0:
                                        price_i = other_variants.loc[k, "price"]
                                        url_i = other_variants.loc[k, "url"]
                                        template_res += f"\n\n🔹 {price_i} руб. {', '.join([improvements_mapping[i] for i in v])} \n\n[{url_i}]"

                    template_res_all += template_res + "\n\n---\n\n"
                # except Exception as e:
                #     self.logger.error(f"Ошибка в _basic_fmt: {e}")
                #     continue

            return template_res_all
        
        if isinstance(self.fastest_optimal, pd.DataFrame):
            fmt_fastest_optimal = _basic_fmt(self.fastest_optimal, "Быстрые оптимальные")
        else:
            fmt_fastest_optimal = ""

        if isinstance(self.cheapest_optimal, pd.DataFrame):
            fmt_cheapest_optimal = _basic_fmt(self.cheapest_optimal, "Дешевые оптимальные")
        else:
            fmt_cheapest_optimal = ""

        if isinstance(self.cheapest, pd.DataFrame):
            fmt_cheapest = _basic_fmt(self.cheapest, "Прочие дешевые")
        else:
            fmt_cheapest = ""

        if isinstance(self.viable_proposals_other, pd.DataFrame):
            fmt_viable_proposals_other = _basic_fmt(self.viable_proposals_other, "Оптимальные варианты с другими опциями")
        else:
            fmt_viable_proposals_other = ""

        filter_results = fmt_fastest_optimal + fmt_cheapest_optimal + fmt_cheapest

        if not filter_results:
            template_res_all_variants = "#### По заданным параметрам ничего не найдено. Попробуйте изменить условия поиска." + "\n\n---\n\n" + fmt_viable_proposals_other
        else:
            template_res_all_variants = fmt_fastest_optimal + "\n\n" + fmt_cheapest_optimal + "\n\n" + fmt_cheapest + "\n\n" + fmt_viable_proposals_other
    
        return template_res_all_variants
    
    def _launch_pipeline(self):
        self.proposals_df_full = self._data_preparation()
        self.viable_proposals = self._apply_filters()
        self.fastest_optimal, self.cheapest_optimal, self.cheapest, self.viable_proposals_other = self._form_recommendations()
        template_res_all = self._format_recommendations()
        with open("digital_assistant_first/aviasales_system/template_res_all.md", "w") as f:
            f.write(template_res_all)
        return template_res_all



