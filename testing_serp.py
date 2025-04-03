from serpapi import GoogleSearch

params = {
  "q": "Подбери мне хорошие рестораны в Москве мясные",
  "api_key": "8d6b302e2df6ebfdb324fe74804bc48166fe567f14de65d73b744950462e703d"
}

search = GoogleSearch(params)
results = search.get_dict()
ai_overview = results["ai_overview"]

print(ai_overview)