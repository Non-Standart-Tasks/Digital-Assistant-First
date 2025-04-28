import functools
import json
import os
from datetime import datetime

api_keys = [
    "db8ef64a824ba6f041ffac1cd2609239d753d7d9",
    "07f042f611669ed03658d5fb453a2f9f0c7adca3"
]
# both are new (reg. 28 apr 2025)

class SerperAPIKeySelector:
    def __init__(self):
        if "serper_api_keys_status.json" not in os.listdir():
            self.keys_dict = {
                "api_keys": {
                    key_i: {
                        "requests_left": 2500,
                        "time_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    } for key_i in api_keys
                }
            }
            self._save()
        else:
            with open("serper_api_keys_status.json", "r") as f:
                self.keys_dict = json.load(f)

    def _save(self):
        with open("serper_api_keys_status.json", "w") as f:
            json.dump(self.keys_dict, f, indent=2)

    def get_best_key(self):
        # get key w/max requests_left
        best_key = max(self.keys_dict["api_keys"].items(), key=lambda x: x[1]["requests_left"])
        return best_key[0]

    def decrement_key(self, api_key):
        if api_key in self.keys_dict["api_keys"]:
            self.keys_dict["api_keys"][api_key]["requests_left"] -= 1
            self._save()

    def track_key(self, response, api_key_used):
        if response.status_code == 200:
            self.decrement_key(api_key_used)
        return None
    
selc = SerperAPIKeySelector()
print(selc.get_best_key())

