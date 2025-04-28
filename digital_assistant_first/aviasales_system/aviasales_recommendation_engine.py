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
    def __init__(self, logger, search_json_received, aviasales_json_sent, search_id):
        self.search_json_received = search_json_received
        self.aviasales_json_sent = aviasales_json_sent
        self.search_id = search_id
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
        proposals_df["partner_id"] = proposals_df.xterms.apply(lambda x: list(x.keys())[0])
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

        proposals_df_full["airlines_list_to"] = proposals_df_full.flight_to.apply(lambda x: tuple([i["operating_carrier"] for i in x["flight"]]))
        proposals_df_full["airlines_list_back"] = proposals_df_full.flight_back.apply(lambda x: tuple([i["operating_carrier"] for i in x["flight"]]) if x else tuple())

        proposals_df_full["flight_to_time_raw"] = proposals_df_full.flight_to.apply(lambda x: x["flight"][0]["departure_time"])
        proposals_df_full["flight_back_time_raw"] = proposals_df_full.flight_back.apply(lambda x: x["flight"][0]["departure_time"] if x else 0)
        
        proposals_df_full["flight_to_time"] = proposals_df_full.flight_to.apply(lambda x: int(x["flight"][0]["departure_time"].split(":")[0]))
        proposals_df_full["flight_back_time"] = proposals_df_full.flight_back.apply(lambda x: int(x["flight"][0]["departure_time"].split(":")[0]) if x else 0)

        proposals_df_full["segment_durations"] = proposals_df_full.segment_durations.apply(tuple)

        proposals_df_full["idx_new"] = proposals_df_full.groupby(
            [
                "total_duration",
                "max_stops",
                "max_stop_duration",
                "segment_durations",
                "airlines_list_to",
                "airlines_list_back",
                "flight_to_time_raw",
                "flight_back_time_raw",
            ]
        ).ngroup()
        proposals_df_full["idx_uniq"] = np.arange(len(proposals_df_full))

        proposals_df_full.sort_values(["has_exchange", "has_return", "has_handbags", "has_baggage"], ascending=False, inplace=True)
        proposals_df_full = proposals_df_full.drop_duplicates(subset=["idx_new", "price"]).reset_index(drop=True).copy()
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

        viable_proposals.to_csv("digital_assistant_first/aviasales_system/viable_proposals.csv", index=False)
        return viable_proposals
        
    def _form_recommendations(self) -> str:
        indices_already_displayed = []

        ##############

        smallest_durations_by_group = self.viable_proposals.groupby("idx_new")[
            "total_duration"
        ].min()
        smallest_durations = smallest_durations_by_group.nsmallest(2).values
        min_price_indices = (
            self.viable_proposals[
                self.viable_proposals.total_duration.isin(smallest_durations)
            ]
            .groupby("idx_new")["price"]
            .idxmin()
        )
        fastest_optimal = self.viable_proposals.loc[min_price_indices].sort_values("price").iloc[:5]
        indices_already_displayed.extend(fastest_optimal.idx_new.to_list())


        ##############

        viable_proposals_no_dupl = self.viable_proposals[~self.viable_proposals.idx_new.isin(indices_already_displayed)].copy()
        small_prices_by_group = (
            viable_proposals_no_dupl
            .groupby("idx_new")["rounded_price_5k"]
            .min()
        )
        small_prices = small_prices_by_group.nsmallest(2).values
        low_price_indices = (
            viable_proposals_no_dupl[
                viable_proposals_no_dupl.rounded_price_5k.isin(small_prices)
            ]
            .groupby("idx_new")["total_duration"]
            .idxmin()
        )
        cheapest_optimal = viable_proposals_no_dupl.loc[low_price_indices].sort_values("total_duration").iloc[:5]
        indices_already_displayed.extend(cheapest_optimal.idx_new.to_list())


        ##############

        viable_proposals_no_dupl = self.viable_proposals[~self.viable_proposals.idx_new.isin(indices_already_displayed)].copy()
        smallest_prices_by_group = (
            viable_proposals_no_dupl
            .groupby("idx_new")["rounded_price_5k"]
            .min()
        )
        smallest_prices = smallest_prices_by_group.nsmallest(2).values
        min_price_indices = (
            viable_proposals_no_dupl[
                viable_proposals_no_dupl.rounded_price_5k.isin(smallest_prices)
            ]
            .groupby("idx_new")["total_duration"]
            .idxmin()
        )
        cheapest = viable_proposals_no_dupl.loc[min_price_indices].sort_values("price").iloc[:5]
        indices_already_displayed.extend(cheapest.idx_new.to_list())

        ##############

        # Предложения ниже вне фильтров (доп рекомендации)!

        viable_proposals_other = self.proposals_df_full[~self.proposals_df_full.idx_new.isin(indices_already_displayed)].copy()

        # 1. Самое дешевое по каждой группе
        cheapest_idxs = viable_proposals_other.groupby("idx_new")["price"].idxmin()
        viable_cheapest = viable_proposals_other.loc[cheapest_idxs]

        # 2. Из самых дешевых выбираем тот, который быстрее всего
        viable_proposals_other_cheapest = viable_cheapest.sort_values("total_duration").head(2)

        # 3. Самое быстрое по каждой группе
        fastest_idxs = viable_proposals_other.groupby("idx_new")["total_duration"].idxmin()
        viable_fastest = viable_proposals_other.loc[fastest_idxs]

        # 4. Удаляем уже выбранное предложение, если оно попало в fastest
        viable_fastest = viable_fastest[~viable_fastest.idx_new.isin(viable_proposals_other_cheapest.idx_new)]

        # 5. Выбираем самое дешевое из быстрых
        viable_proposals_other_fastest = viable_fastest.sort_values("price").head(2)
        
        viable_proposals_other = pd.concat([viable_proposals_other_cheapest, viable_proposals_other_fastest], axis=0)

        return (
            fastest_optimal if fastest_optimal.shape[0] > 0 else None,
            cheapest_optimal if cheapest_optimal.shape[0] > 0 else None,
            cheapest if cheapest.shape[0] > 0 else None,
            viable_proposals_other if viable_proposals_other.shape[0] > 0 else None,
        )

    def _form_dynamic_link(self, url_num):
        return "https://ama.vtb.msut.me/get_link_aviasales?search_id=" + self.search_id + "&url_num=" + str(url_num)

    def _format_recommendations(self) -> str:

        def _sub_fmt(row, pass_string, class_, is_one_twotrip=False):
            handbag_string = f"ручная кладь {row.viable_handbags.replace('1PC', '')}кг" if row.viable_handbags else "без ручной клади"
            baggage_string = f"багаж {row.viable_baggage.replace('1PC', '')}кг" if row.viable_baggage else "без багажа"

            if row.tariff_back:
                exch_return = summarize_exchange_return(row.tariff_to, row.tariff_back, 'get_str')
            else:
                exch_return = summarize_exchange_return(row.tariff_to, row.tariff_to, 'get_str')
            
            if is_one_twotrip:
                res_string = f"\n\n**Цена на OneTwoTrip:** \n\n🔹 {row.price} руб. / за {pass_string.strip().strip(',')}, {handbag_string}, {baggage_string} "\
                    f"/ {class_}{exch_return} \n\n**[Забронировать на OneTwoTrip]"\
                    f"({self._form_dynamic_link(row.url)})**"
            else:
                res_string = f"🔹 {row.price} руб. / за {pass_string.strip().strip(',')}, {handbag_string}, {baggage_string} "\
                    f"/ {class_}{exch_return}"

            return res_string

        def _basic_fmt(df, label, no_filters=False):
            if not isinstance(df, pd.DataFrame):
                return "\n\n"
            if no_filters:
                template_res_all = f"### :blue-background[{label}:]\n\n #### Ниже представлены билеты с более гибкими параметрами поиска.\n\n---\n\n"
            else:
                template_res_all = f"### :blue-background[{label}:]\n\n---\n\n"
            for idx, smpl in df.iterrows():
                try:
                    template_res = ""

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
                            dep_time, arr_time = flight_i['departure_time'], flight_i['arrival_time']
                            dep_time_hr, arr_time_hr = int(dep_time[:2]), int(arr_time[:2])
                            if arr_time_hr < dep_time_hr:
                                template_res += f"{format_date_russian(flight_i['departure_date'])}, {airport_to_naming} {airport_to} {dep_time} - {airport_from_naming} {airport_from} {arr_time} +1 день\n\n"
                            else:
                                template_res += f"{format_date_russian(flight_i['departure_date'])}, {airport_to_naming} {airport_to} {dep_time} - {airport_from_naming} {airport_from} {arr_time}\n\n"
                        
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
                            
                    idx_uniq_blacklist = [smpl.idx_uniq]

                    if smpl.partner_id == "20":
                        template_res += _sub_fmt(smpl, pass_string, class_, is_one_twotrip=True)
                    else:
                        template_res += _sub_fmt(smpl, pass_string, class_, is_one_twotrip=False)
                        smpl_onetwotrip = self.viable_proposals[(self.viable_proposals.idx_new == smpl.idx_new) & (self.viable_proposals.partner_id == "20")].sort_values("price")
                        if smpl_onetwotrip.shape[0] > 0:
                            template_res += _sub_fmt(smpl_onetwotrip.iloc[0], pass_string, class_, is_one_twotrip=True)
                            idx_uniq_blacklist.append(smpl_onetwotrip.iloc[0].idx_uniq)

                    # print(idx_uniq_blacklist, end="\n\n")

                    ### (additional options)
                    options_list = ["has_handbags", "has_baggage", "has_exchange", "has_return"]

                    other_variants = (
                        self.proposals_df_full[
                            (self.proposals_df_full.idx_new == smpl.idx_new)
                            & (~self.proposals_df_full.idx_uniq.isin(idx_uniq_blacklist))
                        ]
                        .sort_values("price")
                        .drop_duplicates(subset=options_list, keep="first")
                    ).iloc[:5]
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
                                        if other_variants.loc[k, "partner_id"] == "20":
                                            template_res += f"\n\n🔹 {price_i} руб. {', '.join([improvements_mapping[i] for i in v])} - **[Забронировать на OneTwoTrip]({self._form_dynamic_link(url_i)})**"
                                        else:
                                            template_res += f"\n\n🔹 {price_i} руб. {', '.join([improvements_mapping[i] for i in v])}"

                    template_res_all += template_res + "\n\n---\n\n"
                except Exception as e:
                    self.logger.error(f"Ошибка в _basic_fmt: {e}")
                    continue

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
            fmt_viable_proposals_other = _basic_fmt(self.viable_proposals_other, "Возможно, вам подойдут", no_filters=True)
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
        # with open("digital_assistant_first/aviasales_system/template_res_all.md", "w") as f:
        #     f.write(template_res_all)
        return template_res_all



