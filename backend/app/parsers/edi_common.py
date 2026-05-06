"""Shared EDI parsing utilities for 837 and 835 files."""


def detect_delimiters(raw: str) -> tuple[str, str]:
    """Detect element and component separators from the ISA segment.

    In EDI, the character immediately after 'ISA' is the element separator,
    and ISA16 (the last data element before the segment terminator) is the
    component separator (sub-element separator).

    Returns (element_separator, component_separator).
    Falls back to ('*', ':') if ISA is not found.
    """
    # Find ISA position in the raw content
    isa_pos = raw.find("ISA")
    if isa_pos == -1:
        return ("*", ":")

    # Character right after "ISA" is the element separator
    element_sep = raw[isa_pos + 3]

    # ISA16 is the component separator — it's the last element before ~
    # ISA has exactly 16 elements. Find the 16th separator to get ISA16.
    pos = isa_pos + 3  # start after "ISA"
    sep_count = 0
    while pos < len(raw) and sep_count < 16:
        if raw[pos] == element_sep:
            sep_count += 1
        pos += 1
    # pos now points to the first char of ISA16
    component_sep = raw[pos] if pos < len(raw) else ":"

    return (element_sep, component_sep)


def normalize_edi(raw: str) -> str:
    """Normalize EDI content to use standard delimiters (* and :).

    Detects the actual delimiters from the ISA segment and replaces
    them with the standard ones so all parsing code works unchanged.
    """
    element_sep, component_sep = detect_delimiters(raw)

    if element_sep == "*" and component_sep == ":":
        return raw  # already standard

    # Replace component separator first (to avoid conflicts if element_sep == ':')
    if component_sep != ":":
        raw = raw.replace(component_sep, ":")
    if element_sep != "*":
        raw = raw.replace(element_sep, "*")

    return raw


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
