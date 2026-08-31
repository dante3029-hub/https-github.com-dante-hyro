import json, os

SHARED_FILE = "/tmp/hyro_shared_positions.json"

def read_shared():
    try:
        with open(SHARED_FILE) as f:
            return json.load(f)
    except:
        return {}

def write_positions(bot_id, coins_list):
    data = read_shared()
    data[bot_id] = coins_list
    with open(SHARED_FILE, "w") as f:
        json.dump(data, f)

def other_bot_coins(bot_id):
    data = read_shared()
    other = set()
    for k, v in data.items():
        if k != bot_id:
            for c in v:
                other.add(c)
    return other
