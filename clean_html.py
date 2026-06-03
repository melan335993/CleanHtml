#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "beautifulsoup4>=4.12.0",
#   "lxml>=5.0.0",
#   "markdownify>=0.13.0",
# ]
# ///

from __future__ import annotations

import argparse
import io
import locale
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Comment, Tag
from markdownify import MarkdownConverter


DEFAULT_FALLBACK_ENCODINGS = ("cp1251", "cp866")
DEFAULT_HTML_EXTENSIONS = (".html", ".htm")

# То, что почти всегда является навигацией/обвязкой, а не полезным текстом документации.
NOISE_CLASS_OR_ID_RE = re.compile(
    r"("
    r"breadcrumbs?|"
    r"nav[-_]?footer|"
    r"children[-_]?links|"
    r"related[-_]?links|"
    r"relinfo|relchildren|relconcepts|familylinks|parentlink|"
    r"PageHeader|PageFooter|PageLink|TopicPath|"
    r"TopicLinks(?:_|$)|"
    r"AdditionalInformationLinkList|"
    r"CollapseExpandLink|"
    r"EmbeddedFeature|"
    r"persistenceDiv|"
    r"PageContentArea_fixedheader_spacer|"
    r"button-info|ancestry|head-block"
    r")",
    re.IGNORECASE,
)

# Секции, которые иногда лежат уже внутри основного контента, но обычно являются справочной навигацией.
DEFAULT_DROP_SECTION_TITLES = {
    "links",
    "ссылки",
    "related concepts",
    "related reference",
    "related tasks",
    "related information",
    "parent topic",
    "topics in this section",
    "условные обозначения",  # часто таблица-легенда Doc-O-Matic, для RAG обычно шум
    "class",                  # Doc-O-Matic: навигационная ссылка на родительский класс
}

ATTACHMENT_EXTENSIONS = {
    # картинки
    ".apng", ".avif", ".bmp", ".gif", ".ico", ".jpeg", ".jpg", ".png", ".svg", ".tif", ".tiff", ".webp",
    # видео/аудио
    ".avi", ".flv", ".m4a", ".m4v", ".mkv", ".mov", ".mp3", ".mp4", ".mpeg", ".mpg", ".ogg", ".ogv", ".wav", ".webm", ".wmv",
    # документы/архивы/бинарники, которые в md лучше не тащить как вложения
    ".7z", ".bz2", ".doc", ".docx", ".dwf", ".dwg", ".dxf", ".gz", ".pdf", ".ppt", ".pptx", ".rar", ".tar", ".tgz", ".xls", ".xlsx", ".zip",
}


@dataclass(frozen=True)
class CleanOptions:
    links: str
    attachments: str
    keep_related: bool
    fallback_encodings: tuple[str, ...]
    overwrite: bool
    quiet: bool
    jobs: int
    backend: str


@dataclass(frozen=True)
class FileResult:
    input_file: str
    output_file: str
    written: bool
    skipped: bool
    error: str | None = None


@dataclass(frozen=True)
class BatchResult:
    processed: int
    written: int
    skipped: int
    failed: int
    errors: tuple[FileResult, ...]


class ProgressPrinter:
    def __init__(self, total: int, quiet: bool) -> None:
        self.total = total
        self.quiet = quiet
        self.last_percent_int = -1
        self.last_print_time = 0.0

    def update(self, processed: int, written: int, skipped: int, failed: int, force: bool = False) -> None:
        if self.quiet:
            return

        percent = 100.0 if self.total == 0 else processed * 100.0 / self.total
        percent_int = int(percent)
        now = time.monotonic()

        if not force and percent_int == self.last_percent_int and now - self.last_print_time < 0.5:
            return

        self.last_percent_int = percent_int
        self.last_print_time = now
        line = (
            f"\rПрогресс: {percent}% "
            f"({processed}/{self.total}) | "
            f"записано: {written} | пропущено: {skipped} | ошибок: {failed}"
        )
        print(line, end="", flush=True)

    def finish(self) -> None:
        if not self.quiet:
            print()


def ensure_utf8_stdout() -> None:
    """Пытается переключить stdout/stderr на UTF-8, если это возможно."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        encoding = getattr(stream, "encoding", None)
        if encoding and encoding.lower().replace("_", "-") == "utf-8":
            continue

        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
                continue
            except Exception:
                pass

        if hasattr(stream, "buffer"):
            try:
                setattr(sys, stream_name, io.TextIOWrapper(stream.buffer, encoding="utf-8"))
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Очищает HTML-документацию от обвязки и сохраняет полезный контент в Markdown. "
            "Директории обрабатываются параллельно."
        )
    )
    parser.add_argument(
        "input_path",
        help="Путь к HTML-файлу или директории с HTML-файлами.",
    )
    parser.add_argument(
        "-l",
        "--links",
        choices=("keep", "drop"),
        default="keep",
        help="Оставлять ссылки в Markdown или оставлять только их текст. По умолчанию: keep.",
    )
    parser.add_argument(
        "-a",
        "--attachments",
        choices=("keep", "drop"),
        default="drop",
        help="Оставлять вложения/картинки/видео/файловые ссылки или удалять их. По умолчанию: drop.",
    )
    parser.add_argument(
        "-k",
        "--keep-related",
        action="store_true",
        help="Не вырезать related/parent/topics/links-секции. По умолчанию они удаляются как навигационный шум.",
    )
    parser.add_argument(
        "-fe",
        "--fallback-encoding",
        dest="fallback_encodings",
        nargs="+",
        default=list(DEFAULT_FALLBACK_ENCODINGS),
        help="Кодировки для файлов, которые не прочитались как UTF-8. По умолчанию: cp1251 cp866.",
    )
    parser.add_argument(
        "-w",
        "--overwrite",
        action="store_true",
        help="Перезаписывать уже существующие *_clean.md файлы.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Минимум вывода в консоль.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=0,
        help="Количество параллельных воркеров для директории. 0 = auto, 1 = без параллелизма. По умолчанию: 0.",
    )
    parser.add_argument(
        "-b",
        "--backend",
        choices=("process", "thread"),
        default="process",
        help="Чем распараллеливать обработку директории. process быстрее для CPU-bound парсинга; thread иногда лучше на медленном диске. По умолчанию: process.",
    )
    return parser.parse_args()


def read_text_content(file_path: Path, fallback_encodings: Sequence[str]) -> str:
    data = file_path.read_bytes()
    encodings = (
        "utf-8-sig",
        "utf-8",
        *fallback_encodings,
        locale.getpreferredencoding(False),
    )
    tried: set[str] = set()

    for encoding in encodings:
        if not encoding:
            continue

        normalized = encoding.lower().replace("_", "-")
        if normalized in tried:
            continue
        tried.add(normalized)

        try:
            return data.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
        except (LookupError, UnicodeDecodeError):
            continue

    return data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


def is_html_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in DEFAULT_HTML_EXTENSIONS


def iter_html_files(input_dir: Path) -> Iterator[Path]:
    yield from sorted(
        (p for p in input_dir.rglob("*") if is_html_file(p)),
        key=lambda p: p.as_posix().lower(),
    )


def output_path_for_file(input_file: Path) -> Path:
    return input_file.with_name(f"{input_file.stem}_clean.md")


def output_root_for_dir(input_dir: Path) -> Path:
    return input_dir.with_name(f"{input_dir.name}_clean")


def output_path_for_dir_file(input_dir: Path, output_root: Path, input_file: Path) -> Path:
    relative = input_file.relative_to(input_dir)
    return (output_root / relative).with_suffix(".md")


def class_id_text(tag: Tag) -> str:
    if not isinstance(getattr(tag, "attrs", None), dict):
        return ""
    parts: list[str] = []
    tag_id = tag.get("id")
    if isinstance(tag_id, str):
        parts.append(tag_id)
    classes = tag.get("class")
    if isinstance(classes, list):
        parts.extend(str(c) for c in classes)
    elif isinstance(classes, str):
        parts.append(classes)
    return " ".join(parts)


def is_noise_element(tag: Tag, keep_related: bool) -> bool:
    if not tag.name:
        return False
    if tag.name in {"script", "style", "noscript", "template", "form", "input", "button", "select", "textarea", "canvas", "map", "area"}:
        return True

    info = class_id_text(tag)
    if not info:
        return False

    if keep_related:
        related_words = (
            "children-links",
            "related-links",
            "relinfo",
            "relchildren",
            "relconcepts",
            "familylinks",
            "parentlink",
            "TopicLinks",
        )
        if any(word.lower() in info.lower() for word in related_words):
            return False

    return bool(NOISE_CLASS_OR_ID_RE.search(info))


def remove_noise(soup: BeautifulSoup, keep_related: bool) -> None:
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    for tag in list(soup.find_all(True)):
        if is_noise_element(tag, keep_related):
            tag.decompose()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def extract_title(soup: BeautifulSoup) -> str:
    selectors = (
        ".TopicTitle",
        ".AdditionalInformationHeader",
        "h1",
        "title",
    )
    for selector in selectors:
        tag = soup.select_one(selector)
        if not tag:
            continue
        title = normalize_text(tag.get_text(" ", strip=True))
        if title:
            return title
    return ""


def score_candidate(tag: Tag) -> int:
    text = normalize_text(tag.get_text(" ", strip=True))
    text_len = len(text)
    paragraphs = len(tag.find_all(["p", "pre", "table", "li", "h1", "h2", "h3", "h4"]))
    links = len(tag.find_all("a"))
    # Много ссылок при малом тексте часто означает навигацию.
    link_penalty = links * 25
    return text_len + paragraphs * 80 - link_penalty


def select_main_content(soup: BeautifulSoup) -> Tag:
    selector_groups = (
        (
            ".PageContentBodywithoutSidebar",
            ".PageContentBody",
            ".PageContentBodywithSidebar",
        ),
        (
            "div.conbody",
            "div.conceptbody-adsk",
            "div.refbody",
            "div.referencebody",
            "div.taskbody",
            "div.body",
        ),
        (
            "main",
            "article",
        ),
        (
            ".content",
        ),
    )

    for selectors in selector_groups:
        candidates: list[Tag] = []
        for selector in selectors:
            candidates.extend(soup.select(selector))
        candidates = [c for c in candidates if normalize_text(c.get_text(" ", strip=True))]
        if candidates:
            return max(candidates, key=score_candidate)

    body = soup.body
    if body:
        return body
    return soup


def convert_docomatic_section_headings(root: Tag) -> None:
    for heading in list(root.select(".SectionHeading")):
        text = normalize_text(heading.get_text(" ", strip=True))
        if not text:
            heading.decompose()
            continue
        new_tag = soup_new_tag(root, "h2")
        new_tag.string = text
        heading.replace_with(new_tag)


def soup_new_tag(root: Tag, name: str) -> Tag:
    # У Tag нет стабильного публичного способа получить soup, поэтому поднимаемся к корню.
    parent: Tag | BeautifulSoup = root
    while getattr(parent, "parent", None) is not None:
        parent = parent.parent  # type: ignore[assignment]
    return parent.new_tag(name)  # type: ignore[union-attr]


def drop_empty_tags(root: Tag) -> None:
    for tag in list(root.find_all(True)):
        if not tag.name:
            continue
        if tag.name in {"br", "hr", "img", "source"}:
            continue
        if tag.find(["img", "video", "audio", "source", "pre", "code", "table"]):
            continue
        if not normalize_text(tag.get_text(" ", strip=True)):
            tag.decompose()


def href_extension(href: str) -> str:
    parsed = urlparse(href)
    return Path(parsed.path).suffix.lower()


def is_attachment_href(href: str) -> bool:
    if not href or href.startswith("#"):
        return False
    return href_extension(href) in ATTACHMENT_EXTENSIONS


def unwrap_keep_text(tag: Tag) -> None:
    if normalize_text(tag.get_text(" ", strip=True)):
        tag.unwrap()
    else:
        tag.decompose()


def clean_links_and_attachments(root: Tag, options: CleanOptions) -> None:
    if options.attachments == "drop":
        for tag in list(root.find_all(["img", "picture", "video", "audio", "source", "object", "embed", "iframe"])):
            tag.decompose()

        for link in list(root.find_all("a")):
            href = str(link.get("href") or "")
            if is_attachment_href(href):
                unwrap_keep_text(link)

    else:
        # markdownify плохо конвертирует video/audio. Заменяем их на обычные ссылки.
        for media in list(root.find_all(["video", "audio"])):
            sources: list[str] = []
            if media.get("src"):
                sources.append(str(media.get("src")))
            for source in media.find_all("source"):
                if source.get("src"):
                    sources.append(str(source.get("src")))
            sources = [s for s in sources if s]
            if not sources:
                media.decompose()
                continue
            replacement = soup_new_tag(root, "p")
            replacement.string = " ".join(f"[{media.name}]({src})" for src in sources)
            media.replace_with(replacement)

    if options.links == "drop":
        for link in list(root.find_all("a")):
            unwrap_keep_text(link)
    else:
        for link in list(root.find_all("a")):
            href = str(link.get("href") or "").strip()
            if not href and link.get("data-original-href"):
                href = str(link.get("data-original-href") or "").strip()
                link["href"] = href

            if not href or href == "#" or href.lower().startswith("javascript:"):
                unwrap_keep_text(link)
                continue


def drop_default_related_sections(root: Tag, keep_related: bool) -> None:
    if keep_related:
        return

    for heading in list(root.find_all(re.compile(r"^h[1-6]$"))):
        title = normalize_text(heading.get_text(" ", strip=True)).lower()
        if title not in DEFAULT_DROP_SECTION_TITLES:
            continue

        # У Doc-O-Matic после заголовка секции обычно идёт div с содержимым этой секции.
        sibling = heading.find_next_sibling()
        if sibling and isinstance(sibling, Tag):
            sibling.decompose()
        heading.decompose()


def dedupe_duplicate_first_heading(markdown: str, title: str) -> str:
    if not title:
        return markdown

    lines = markdown.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)

    if lines and normalize_text(lines[0].lstrip("# ")) == normalize_text(title):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)

    return "\n".join(lines)


class DocsMarkdownConverter(MarkdownConverter):
    def convert_pre(self, el: Tag, text: str, parent_tags: set[str]) -> str:  # type: ignore[override]
        code = el.get_text("", strip=False).strip("\n")
        return f"\n\n```\n{code}\n```\n\n"

    def convert_code(self, el: Tag, text: str, parent_tags: set[str]) -> str:  # type: ignore[override]
        if "pre" in parent_tags:
            return text
        return f"`{text.strip()}`"


def html_to_markdown(root: Tag) -> str:
    return DocsMarkdownConverter(
        heading_style="ATX",
        bullets="-",
        strip=["span"],
        escape_asterisks=False,
        escape_underscores=False,
        newline_style="BACKSLASH",
    ).convert_soup(root)


def cleanup_markdown(markdown: str) -> str:
    text = markdown.replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^\s+$", "", text)
    return text.strip() + "\n"


def make_soup(html: str) -> BeautifulSoup:
    # lxml заметно быстрее стандартного html.parser на больших пачках маленьких файлов.
    # Зависимость указана в uv metadata выше, так что uv подтянет wheel автоматически.
    return BeautifulSoup(html, "lxml")


def clean_html_to_markdown(html: str, source_path: Path, options: CleanOptions) -> str:
    soup = make_soup(html)
    title = extract_title(soup)

    remove_noise(soup, keep_related=options.keep_related)
    main = select_main_content(soup)

    # Работаем с копией выбранного фрагмента: так проще не зацепить внешнюю обвязку.
    fragment_soup = make_soup(str(main))
    fragment_root = fragment_soup.body or fragment_soup

    convert_docomatic_section_headings(fragment_root)
    remove_noise(fragment_soup, keep_related=options.keep_related)
    clean_links_and_attachments(fragment_root, options)
    drop_empty_tags(fragment_root)
    drop_default_related_sections(fragment_root, keep_related=options.keep_related)

    body_md = html_to_markdown(fragment_root)
    body_md = dedupe_duplicate_first_heading(body_md, title)
    body_md = cleanup_markdown(body_md)

    if not title:
        title = source_path.stem

    result = f"# {title}\n\n{body_md}" if body_md.strip() else f"# {title}\n"
    return cleanup_markdown(result)


def process_file_worker(job: tuple[str, str, CleanOptions]) -> FileResult:
    input_file = Path(job[0])
    output_file = Path(job[1])
    options = job[2]

    try:
        if output_file.exists() and not options.overwrite:
            return FileResult(str(input_file), str(output_file), written=False, skipped=True)

        html = read_text_content(input_file, options.fallback_encodings)
        markdown = clean_html_to_markdown(html, input_file, options)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(markdown, encoding="utf-8", newline="\n")

        return FileResult(str(input_file), str(output_file), written=True, skipped=False)
    except Exception as exc:
        return FileResult(str(input_file), str(output_file), written=False, skipped=False, error=str(exc))


def process_batch_worker(job: tuple[tuple[tuple[str, str], ...], CleanOptions]) -> BatchResult:
    file_jobs, options = job
    written = 0
    skipped = 0
    failed = 0
    errors: list[FileResult] = []

    for input_file, output_file in file_jobs:
        result = process_file_worker((input_file, output_file, options))
        if result.error:
            failed += 1
            if len(errors) < 10:
                errors.append(result)
        elif result.written:
            written += 1
        elif result.skipped:
            skipped += 1

    return BatchResult(
        processed=len(file_jobs),
        written=written,
        skipped=skipped,
        failed=failed,
        errors=tuple(errors),
    )


def resolve_worker_count(requested_jobs: int, total_files: int) -> int:
    if total_files <= 1:
        return 1
    if requested_jobs < 0:
        raise ValueError("--jobs не может быть отрицательным")
    if requested_jobs == 1:
        return 1
    if requested_jobs > 1:
        return min(requested_jobs, total_files)

    cpu_count = os.cpu_count() or 1
    # Для Windows и большого числа мелких файлов не стоит по умолчанию плодить десятки процессов:
    # накладные расходы и конкуренция за диск легко съедают выигрыш. Ручной -j можно поднять выше.
    return min(max(cpu_count - 1, 1), 16, total_files)


def resolve_batch_size(total_jobs: int, worker_count: int, backend: str) -> int:
    if worker_count <= 1:
        return 1

    # Пачки снижают IPC overhead на тысячах мелких HTML-файлов.
    # process: пачки крупнее; thread: поменьше, чтобы прогресс был живее.
    if backend == "process":
        return max(50, min(2000, total_jobs // (worker_count * 4) or 1))
    return max(20, min(500, total_jobs // (worker_count * 8) or 1))


def chunked(items: Sequence[tuple[str, str]], chunk_size: int) -> Iterator[tuple[tuple[str, str], ...]]:
    for start in range(0, len(items), chunk_size):
        yield tuple(items[start:start + chunk_size])


def collect_dir_jobs(input_dir: Path, output_root: Path) -> list[tuple[Path, Path]]:
    return [(input_file, output_path_for_dir_file(input_dir, output_root, input_file)) for input_file in iter_html_files(input_dir)]


def process_file_with_progress(input_file: Path, options: CleanOptions) -> tuple[int, int, int]:
    output_file = output_path_for_file(input_file)
    progress = ProgressPrinter(total=1, quiet=options.quiet)
    progress.update(0, 0, 0, 0, force=True)
    result = process_file_worker((str(input_file), str(output_file), options))

    written = 1 if result.written else 0
    skipped = 1 if result.skipped else 0
    failed = 1 if result.error else 0
    progress.update(1, written, skipped, failed, force=True)
    progress.finish()

    if result.error:
        print(f"❌ {result.input_file}: {result.error}", file=sys.stderr)
    return written, 1, failed


def process_jobs_sequential(jobs: Sequence[tuple[Path, Path]], options: CleanOptions) -> tuple[int, int, int]:
    progress = ProgressPrinter(total=len(jobs), quiet=options.quiet)
    written = 0
    skipped = 0
    failed = 0
    first_errors: list[FileResult] = []

    progress.update(0, 0, 0, 0, force=True)
    for index, (input_file, output_file) in enumerate(jobs, start=1):
        result = process_file_worker((str(input_file), str(output_file), options))
        if result.error:
            failed += 1
            if len(first_errors) < 10:
                first_errors.append(result)
        elif result.written:
            written += 1
        elif result.skipped:
            skipped += 1
        progress.update(index, written, skipped, failed)

    progress.update(len(jobs), written, skipped, failed)
    progress.finish()
    print_errors(first_errors, failed)
    return written, len(jobs), failed


def process_jobs_parallel(jobs: Sequence[tuple[Path, Path]], options: CleanOptions) -> tuple[int, int, int]:
    worker_count = resolve_worker_count(options.jobs, len(jobs))
    if worker_count <= 1:
        return process_jobs_sequential(jobs, options)

    batch_size = resolve_batch_size(len(jobs), worker_count, options.backend)
    string_jobs = [(str(input_file), str(output_file)) for input_file, output_file in jobs]
    batches = list(chunked(string_jobs, batch_size))
    executor_class = ProcessPoolExecutor if options.backend == "process" else ThreadPoolExecutor

    if not options.quiet:
        print(
            f"Найдено HTML-файлов: {len(jobs)}. "
            f"Воркеры: {worker_count}, backend: {options.backend}, batch: {batch_size}."
        )

    progress = ProgressPrinter(total=len(jobs), quiet=options.quiet)
    progress.update(0, 0, 0, 0, force=True)

    written = 0
    skipped = 0
    failed = 0
    processed = 0
    first_errors: list[FileResult] = []

    worker_options = CleanOptions(
        links=options.links,
        attachments=options.attachments,
        keep_related=options.keep_related,
        fallback_encodings=options.fallback_encodings,
        overwrite=options.overwrite,
        quiet=True,
        jobs=options.jobs,
        backend=options.backend,
    )

    with executor_class(max_workers=worker_count) as executor:
        futures = [executor.submit(process_batch_worker, (batch, worker_options)) for batch in batches]
        for future in as_completed(futures):
            batch_result = future.result()
            processed += batch_result.processed
            written += batch_result.written
            skipped += batch_result.skipped
            failed += batch_result.failed
            for error in batch_result.errors:
                if len(first_errors) < 10:
                    first_errors.append(error)
            progress.update(processed, written, skipped, failed)

    progress.update(len(jobs), written, skipped, failed)
    progress.finish()
    print_errors(first_errors, failed)
    return written, len(jobs), failed


def print_errors(first_errors: Sequence[FileResult], failed: int) -> None:
    for result in first_errors:
        print(f"❌ {result.input_file}: {result.error}", file=sys.stderr)

    if failed > len(first_errors):
        print(f"❌ Ещё ошибок: {failed - len(first_errors)}", file=sys.stderr)


def process_input(input_path: Path, options: CleanOptions) -> tuple[int, int, int]:
    if input_path.is_file():
        if not is_html_file(input_path):
            raise ValueError(f"Файл не похож на HTML: {input_path}")
        return process_file_with_progress(input_path, options)

    if not input_path.is_dir():
        raise ValueError(f"Путь не найден: {input_path}")

    output_root = output_root_for_dir(input_path)
    jobs = collect_dir_jobs(input_path, output_root)
    return process_jobs_parallel(jobs, options)


def main() -> None:
    ensure_utf8_stdout()
    args = parse_args()

    options = CleanOptions(
        links=args.links,
        attachments=args.attachments,
        keep_related=args.keep_related,
        fallback_encodings=tuple(args.fallback_encodings),
        overwrite=args.overwrite,
        quiet=args.quiet,
        jobs=args.jobs,
        backend=args.backend,
    )

    input_path = Path(args.input_path).resolve()

    try:
        written, total, failed = process_input(input_path, options)
    except Exception as exc:
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(1)

    if not options.quiet:
        if failed:
            print(f"Готово с ошибками: записано {written} из {total} HTML-файлов, ошибок: {failed}.")
        else:
            print(f"Готово: записано {written} из {total} HTML-файлов.")


if __name__ == "__main__":
    main()
