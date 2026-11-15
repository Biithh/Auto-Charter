"""Descarga previa de los modelos (se llama desde el instalador)."""

import sys


def main() -> int:
    from charter import paths
    from charter.models_registry import download_all

    paths.setup_env()
    ok = download_all(log=print)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
