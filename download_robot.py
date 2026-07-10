import requests
import json
import sys

urls = [
    "https://assets9.lottiefiles.com/packages/lf20_uk5ex0hx.json",
    "https://assets5.lottiefiles.com/packages/lf20_cop4t8kv.json",
    "https://assets3.lottiefiles.com/packages/lf20_q5pk6t1s.json",
    "https://assets8.lottiefiles.com/packages/lf20_mbe9ymee.json"
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

success = False
for url in urls:
    try:
        print(f"Trying to download from: {url}")
        r = requests.get(url, headers=headers, timeout=10)
        print(f"Response status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            if "layers" in data or "nm" in data:
                with open("src/robot_anim.json", "w", encoding="utf-8") as f:
                    json.dump(data, f)
                print(f"Successfully downloaded robot animation to src/robot_anim.json")
                success = True
                break
    except Exception as e:
        print(f"Error: {e}")

if not success:
    print("Failed to download any robot animation.")
    sys.exit(1)
