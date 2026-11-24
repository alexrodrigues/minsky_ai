from __future__ import annotations

from dataclasses import dataclass

import tree_sitter_language_pack as tlp

MAX_CHARS = 1500
OVERLAP_CHARS = 200

# Extensions with a real function/class/method grammar where AST-aware chunking
# makes sense. Deliberately excludes markup/data formats (.md, .toml, .json, ...)
# even though tree-sitter has grammars for those too — "definition" boundaries
# don't map onto them the same way, so they stay on the line-based heuristic.
AST_CHUNK_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".rb",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".swift", ".kt", ".sh", ".sql",
}


@dataclass
class Chunk:
    text: str
    start_line: int


def chunk_text(
    text: str, max_chars: int = MAX_CHARS, overlap_chars: int = OVERLAP_CHARS
) -> list[Chunk]:
    """Greedily pack blank-line-separated blocks into chunks, tracking start line."""
    lines = text.splitlines()
    if not lines:
        return []

    blocks: list[tuple[str, int]] = []
    current: list[str] = []
    current_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "" and current:
            blocks.append(("\n".join(current), current_start))
            current = []
        else:
            if not current:
                current_start = i
            current.append(line)
    if current:
        blocks.append(("\n".join(current), current_start))

    chunks: list[Chunk] = []
    buf = ""
    buf_start = 0
    for block, start in blocks:
        if not buf:
            buf_start = start
        candidate = f"{buf}\n\n{block}" if buf else block
        if len(candidate) > max_chars and buf:
            chunks.append(Chunk(text=buf, start_line=buf_start + 1))
            tail = buf[-overlap_chars:] if overlap_chars else ""
            buf = f"{tail}\n\n{block}" if tail else block
            buf_start = start
        else:
            buf = candidate
    if buf.strip():
        chunks.append(Chunk(text=buf, start_line=buf_start + 1))

    return chunks


def chunk_ast(text: str, language: str, max_chars: int = MAX_CHARS) -> list[Chunk] | None:
    """AST-aware chunking via tree-sitter: splits on definition/statement
    boundaries instead of blank lines, so a chunk never cuts a function or
    class in half. Whole small definitions stay as one chunk; an oversized
    one (bigger than max_chars) is recursed into by the parser itself and
    split at its own statement boundaries.

    Returns None (caller should fall back to chunk_text) if the language
    isn't parseable here, the source has no syntax tree to speak of, or
    anything goes wrong — this must never be the reason indexing a file fails.
    """
    try:
        cfg = tlp.ProcessConfig(language=language, chunk_max_size=max_chars)
        result = tlp.process(text, cfg)
    except Exception:
        return None

    if not result.chunks:
        return None

    chunks: list[Chunk] = []
    for c in result.chunks:
        content = c.content
        if c.metadata.context_path:
            # Sub-chunk of a larger definition (e.g. one statement group inside
            # a long function) — prefix which definition it belongs to, since
            # that context would otherwise be lost to the embedding.
            content = f"# {' > '.join(c.metadata.context_path)}\n{content}"
        chunks.append(Chunk(text=content, start_line=c.start_line + 1))
    return chunks


def chunk_toml(
    text: str,
    max_chars: int = MAX_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[Chunk] | None:
    """Section-aware chunking for TOML config files: splits on `[table]` /
    `[[table]]` boundaries instead of blank lines, so a chunk never mixes
    unrelated settings and each chunk's embedding carries its own section
    path (injected as a `# [section]` header line). Root-level pairs or
    comments before the first table are kept with the first section rather
    than dropped or split into a near-empty chunk of their own.

    Returns None (caller should fall back to chunk_text) if the file can't
    be parsed or contains no `[table]`/`[[table]]` structure at all — a flat
    key=value-only TOML file has no sectioning win to offer.
    """
    try:
        parser = tlp.get_parser("toml")
        tree = parser.parse(text.encode("utf-8"))
    except Exception:
        return None

    root = tree.root_node
    children = root.children

    section_start_idx = next(
        (i for i, n in enumerate(children) if n.type in ("table", "table_array_element")),
        None,
    )
    if section_start_idx is None:
        return None

    preamble_nodes = children[:section_start_idx]
    section_nodes = children[section_start_idx:]

    def _node_key_text(node) -> str:
        for child in node.children:
            if child.type in ("bare_key", "dotted_key"):
                return text.encode("utf-8")[child.start_byte:child.end_byte].decode("utf-8")
        return ""

    def _node_text(node) -> str:
        return text.encode("utf-8")[node.start_byte:node.end_byte].decode("utf-8")

    blocks: list[tuple[str, int]] = []

    has_root_pair = any(n.type == "pair" for n in preamble_nodes)
    if preamble_nodes:
        raw_preamble = text.encode("utf-8")[
            preamble_nodes[0].start_byte:preamble_nodes[-1].end_byte
        ].decode("utf-8")
    else:
        raw_preamble = ""

    if has_root_pair:
        blocks.append((raw_preamble, preamble_nodes[0].start_point[0]))

    for idx, node in enumerate(section_nodes):
        key = _node_key_text(node)
        header = f"# [{key}]" if node.type == "table" else f"# [[{key}]]"
        body = _node_text(node)
        if idx == 0 and not has_root_pair and raw_preamble:
            body = f"{raw_preamble}\n{body}"
        blocks.append((f"{header}\n{body}", node.start_point[0]))

    if not blocks:
        return None

    # Same greedy/overlap packing as chunk_text, reading from pre-formed
    # section blocks instead of blank-line blocks.
    chunks: list[Chunk] = []
    buf = ""
    buf_start = 0
    for block, start in blocks:
        if not buf:
            buf_start = start
        candidate = f"{buf}\n\n{block}" if buf else block
        if len(candidate) > max_chars and buf:
            chunks.append(Chunk(text=buf, start_line=buf_start + 1))
            tail = buf[-overlap_chars:] if overlap_chars else ""
            buf = f"{tail}\n\n{block}" if tail else block
            buf_start = start
        else:
            buf = candidate
    if buf.strip():
        chunks.append(Chunk(text=buf, start_line=buf_start + 1))

    return chunks or None


def chunk_file(
    text: str,
    suffix: str,
    max_chars: int = MAX_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[Chunk]:
    """Chunk a file's contents for indexing, using AST-aware chunking for
    real source code, section-aware chunking for TOML config, and the
    line-based heuristic for everything else (docs, markup) or when the
    parser can't handle a file."""
    if suffix == ".toml":
        toml_chunks = chunk_toml(text, max_chars, overlap_chars)
        if toml_chunks:
            return toml_chunks
        return chunk_text(text, max_chars, overlap_chars)
    if suffix in AST_CHUNK_EXTENSIONS:
        language = tlp.detect_language_from_path(f"f{suffix}")
        if language:
            ast_chunks = chunk_ast(text, language, max_chars)
            if ast_chunks:
                return ast_chunks
    return chunk_text(text, max_chars, overlap_chars)
