"""Garante que um clone limpo contem tudo que codigo e documentacao referenciam.

Estes testes existem porque as evidencias reproduziveis ja estiveram fora do
controle de versao: o repositorio parecia completo na maquina de quem
desenvolveu, mas quebrava para qualquer avaliador que apenas clonasse.
"""

import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(<?([^)>\s]+)>?\)")
EVIDENCE_REFERENCE = re.compile(r"(?:artifacts|reports)/[A-Za-z0-9_./-]+")
EVIDENCE_SUFFIXES = (".json", ".csv", ".png", ".joblib")


def _markdown_files() -> list[Path]:
    return sorted(
        [*PROJECT_ROOT.glob("*.md"), *(PROJECT_ROOT / "docs").rglob("*.md")]
    )


def _source_files() -> list[Path]:
    return sorted(
        [
            *(PROJECT_ROOT / "src").rglob("*.py"),
            *(PROJECT_ROOT / "tests").rglob("*.py"),
        ]
    )


def _tracked_files() -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return set(completed.stdout.split())


def test_relative_links_and_images_in_documentation_resolve() -> None:
    """Nenhum link ou imagem relativa pode apontar para arquivo inexistente."""

    broken: list[str] = []
    for document in _markdown_files():
        for match in MARKDOWN_LINK.finditer(document.read_text(encoding="utf-8")):
            target = match.group(1)
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            if not (document.parent / target).resolve().exists():
                broken.append(f"{document.relative_to(PROJECT_ROOT)} -> {target}")

    assert not broken, "links ou imagens quebradas: " + "; ".join(broken)


def test_referenced_evidence_is_versioned_and_not_only_local() -> None:
    """Toda evidencia citada precisa estar versionada, nao apenas no disco local."""

    tracked = _tracked_files()
    references: set[str] = set()
    for source in (*_source_files(), *_markdown_files()):
        content = source.read_text(encoding="utf-8", errors="ignore")
        for match in EVIDENCE_REFERENCE.finditer(content):
            reference = match.group(0)
            if reference.endswith(EVIDENCE_SUFFIXES):
                references.add(reference)

    assert references, "nenhuma referencia de evidencia encontrada"
    untracked = sorted(reference for reference in references if reference not in tracked)
    assert not untracked, (
        "evidencias referenciadas mas fora do controle de versao; um clone limpo "
        "as perderia: " + "; ".join(untracked)
    )
