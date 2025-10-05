import json
import pandas as pd

def safe_parse_response(text: str):
    try:
        parsed = json.loads(text)
        return parsed.get("evidence"), parsed.get("is_relevant")
    except Exception:
        pass

    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            parsed = json.loads(candidate)
            return parsed.get("evidence"), parsed.get("is_relevant")
    except Exception:
        pass

    return [], "PARSE_ERROR"