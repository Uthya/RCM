"""Shared EDI parsing utilities for 837 and 835 files."""


def split_segments(raw: str) -> list[str]:
    """Split EDI content on ~ delimiter, stripping whitespace."""
    raw = raw.replace("\r\n", "").replace("\n", "").replace("\r", "")
    segments = [s.strip() for s in raw.split("~") if s.strip()]
    return segments


def split_elements(segment: str) -> list[str]:
    """Split a segment on * delimiter."""
    return segment.split("*")


def split_components(element: str) -> list[str]:
    """Split an element on : component separator."""
    return element.split(":")


def find_segments(segments: list[str], prefix: str) -> list[list[str]]:
    """Find all segments starting with a prefix, return as split elements."""
    results = []
    for seg in segments:
        elements = split_elements(seg)
        if elements[0] == prefix:
            results.append(elements)
    return results


def get_element(elements: list[str], index: int, default: str = "") -> str:
    """Safely get element at index."""
    if index < len(elements):
        return elements[index].strip()
    return default


def find_segment_after(segments: list[str], target_prefix: str,
                       after_prefix: str, after_qualifier: str | None = None,
                       start_idx: int = 0) -> list[str] | None:
    """Find first segment with target_prefix that appears after a segment
    matching after_prefix (and optionally after_qualifier at element[1])."""
    found_after = False
    for i in range(start_idx, len(segments)):
        elements = split_elements(segments[i])
        if not found_after:
            if elements[0] == after_prefix:
                if after_qualifier is None or get_element(elements, 1) == after_qualifier:
                    found_after = True
            continue
        if elements[0] == target_prefix:
            return elements
        # Stop at next loop-level segment
        if elements[0] in ("NM1", "CLM", "CLP", "LX", "SE"):
            break
    return None
