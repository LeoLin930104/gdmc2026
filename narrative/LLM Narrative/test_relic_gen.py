import json
import time
from relic_generator import generate_relics

THEME = "a coastal fishing village haunted by drowned sailors"
COUNT = 3


def main() -> None:
    print(f'Generating {COUNT} relics for theme:\n  "{THEME}"\n')
    start = time.perf_counter()
    relics = generate_relics(THEME, count=COUNT)
    elapsed = time.perf_counter() - start
    print(f"Generated {len(relics)} relics in {elapsed:.1f}s\n")
    print(json.dumps({"relics": relics}, indent=2))


if __name__ == "__main__":
    main()
