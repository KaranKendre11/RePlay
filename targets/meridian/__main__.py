"""Run MERIDIAN CORE.

uv run python -m targets.meridian                 # base, port 8080
uv run python -m targets.meridian --variant northgate --port 8081
"""

import argparse

from targets.meridian.app import VARIANTS, create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MERIDIAN CORE target app.")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="base")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    create_app(variant=args.variant).run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
