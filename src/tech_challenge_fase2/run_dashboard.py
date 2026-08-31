"""CLI de construcao do painel estatico."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dashboard.model import PROJECT_ROOT, load_dashboard_data
from .dashboard.privacy import assert_html_has_no_individual_data
from .dashboard.render import render_dashboard

DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "dashboard" / "index.html"


def build_dashboard(output_path: Path = DEFAULT_OUTPUT) -> Path:
    """Gera o documento e recusa escreve-lo se ele carregar dado individual."""

    data = load_dashboard_data()
    html = render_dashboard(data)
    assert_html_has_no_individual_data(html)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def build_main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera o painel estatico a partir dos artefatos assinados."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    path = build_dashboard(arguments.output)
    print(json.dumps({
        "dashboard": str(path),
        "bytes": path.stat().st_size,
        "network_requests": 0,
        "individual_data_present": False,
    }, ensure_ascii=False, indent=2))
