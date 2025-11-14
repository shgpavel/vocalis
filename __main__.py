# main.py
from __future__ import annotations
import argparse
import curses
from .config import Config
from .app import App


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--device", type=int)
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()
    cfg = Config()
    app = App(cfg)
    curses.wrapper(app.run)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
