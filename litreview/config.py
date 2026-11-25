"""Read simple, single-line .env assignments without executing or expanding values."""

import re
from pathlib import Path


PROJECT_ENV = Path(__file__).resolve().parents[1] / ".env"


def read_dotenv(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    except UnicodeError:
        raise ValueError(".env must use UTF-8 encoding.") from None
    values = {}
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
        if not match:
            raise ValueError(f"Invalid .env assignment on line {number}.")
        key, value = match.groups()
        if value.startswith(("'", '"')):
            closing = value.find(value[0], 1)
            if closing == -1 or (value[closing + 1:].strip()
                                 and not value[closing + 1:].lstrip().startswith("#")):
                raise ValueError(f"Invalid .env quoted value on line {number}.")
            value = value[1:closing]
        else:
            value = "" if value.startswith("#") else re.split(r"\s+#", value, maxsplit=1)[0].strip()
        values[key] = value
    return values
