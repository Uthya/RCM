"""Parser for 837 Professional (837P) EDI files -> JSON claims."""

from app.parsers.edi_common import (
    split_segments, split_elements, split_components,
    get_element, normalize_edi,
)
from app.schemas.claim import ParsedClaim, ServiceLine


def parse_837(raw: str) -> list[ParsedClaim]:
    """Parse an 837 Professional EDI file and return a list of ParsedClaim objects.

    Handles multiple ST/SE transaction sets within a single interchange,
    scoping HL IDs per transaction set to avoid collisions.
    """
    raw = normalize_edi(raw)
    segments = split_segments(raw)
    claims: list[ParsedClaim] = []

    # Extract ISA-level info
    isa_elements = []
    for seg in segments:
        els = split_elements(seg)
        if els[0] == "ISA":
            isa_elements = els
            break

    sender_id = get_element(isa_elements, 6).strip() if isa_elements else ""
    receiver_id = get_element(isa_elements, 8).strip() if isa_elements else ""
    icn = get_element(isa_elements, 13) if isa_elements else ""

    # Build indexed segment list
    seg_list = [split_elements(seg) for seg in segments]

    # Split into transaction sets (ST/SE boundaries)
    tx_sets = _find_transaction_sets(seg_list)

    for tx_start, tx_end in tx_sets:
        # Extract BHT within this transaction set
        bht_ref = ""
        bht_date = ""
        for i in range(tx_start, tx_end):
            if seg_list[i][0] == "BHT":
                bht_ref = get_element(seg_list[i], 3)
                bht_date = get_element(seg_list[i], 4)
                break

        # Parse claims within this transaction set
        tx_claims = _parse_transaction_set(
            seg_list, tx_start, tx_end,
            sender_id, receiver_id, icn, bht_ref, bht_date,
        )
        claims.extend(tx_claims)

    return claims


def _find_transaction_sets(seg_list: list[list[str]]) -> list[tuple[int, int]]:
    """Find ST/SE transaction set boundaries. Returns list of (start, end) index pairs."""
    tx_sets: list[tuple[int, int]] = []
    current_st = None

    for i, els in enumerate(seg_list):
        if els[0] == "ST":
            current_st = i
        elif els[0] == "SE" and current_st is not None:
            tx_sets.append((current_st, i + 1))
            current_st = None

    # If no ST/SE found, treat entire file as one transaction
    if not tx_sets:
        tx_sets.append((0, len(seg_list)))

    return tx_sets


def _parse_transaction_set(
    seg_list: list[list[str]],
    tx_start: int,
    tx_end: int,
    sender_id: str,
    receiver_id: str,
    icn: str,
    bht_ref: str,
    bht_date: str,
) -> list[ParsedClaim]:
    """Parse claims from a single ST/SE transaction set.

    HL IDs are scoped to this transaction set, avoiding collisions
    when multiple transaction sets reuse the same HL numbering.
    """
    claims: list[ParsedClaim] = []

    # Build HL index scoped to this transaction set
    hl_indices: dict[str, int] = {}  # HL id -> index in seg_list
    for i in range(tx_start, tx_end):
        els = seg_list[i]
        if els[0] == "HL":
            hl_id = get_element(els, 1)
            hl_indices[hl_id] = i

    if not hl_indices:
        return claims

    # Find all HL positions sorted
    hl_positions = sorted(hl_indices.values())

    def _hl_range(hl_idx: int) -> tuple[int, int]:
        """Get the segment range for an HL loop (start, end)."""
        pos = hl_positions.index(hl_idx)
        start = hl_idx
        end = hl_positions[pos + 1] if pos + 1 < len(hl_positions) else tx_end
        return start, end

    # Iterate through HL loops to find subscriber/patient loops
    for hl_id, hl_idx in hl_indices.items():
        hl_els = seg_list[hl_idx]
        hl_level = get_element(hl_els, 3)  # 20=provider, 22=subscriber, 23=dependent

        if hl_level not in ("22", "23"):
            continue

        # This is a subscriber or dependent loop
        hl_start, hl_end = _hl_range(hl_idx)
        parent_hl_id = get_element(hl_els, 2)

        # Get parent (provider) loop range
        parent_start = tx_start
        parent_end = hl_idx
        if parent_hl_id in hl_indices:
            parent_start, parent_end = _hl_range(hl_indices[parent_hl_id])

        # Extract provider info from parent HL loop
        provider_name = ""
        provider_npi = ""
        provider_taxonomy = ""
        for j in range(parent_start, min(parent_end, hl_idx)):
            pels = seg_list[j]
            if pels[0] == "NM1" and get_element(pels, 1) == "85":
                provider_name = _build_name(pels)
                provider_npi = get_element(pels, 9)
            elif pels[0] == "PRV":
                provider_taxonomy = get_element(pels, 3)

        # Extract subscriber info from this HL loop (before CLM)
        patient_first = ""
        patient_last = ""
        subscriber_id = ""
        patient_dob = ""
        patient_gender = ""
        payer_name = ""
        payer_id = ""
        payer_sequence = ""
        group_number = ""

        for j in range(hl_start, hl_end):
            els = seg_list[j]
            if els[0] == "CLM":
                break  # stop at first CLM, rest handled per-claim

            if els[0] == "NM1":
                qualifier = get_element(els, 1)
                if qualifier == "IL":
                    patient_last = get_element(els, 3)
                    patient_first = get_element(els, 4)
                    subscriber_id = get_element(els, 9)
                elif qualifier == "PR":
                    payer_name = get_element(els, 3)
                    payer_id = get_element(els, 9)
            elif els[0] == "DMG":
                patient_dob = get_element(els, 2)
                patient_gender = get_element(els, 3)
            elif els[0] == "SBR":
                payer_sequence = get_element(els, 1)
                group_number = get_element(els, 4)

        # Find CLM segments within this HL loop
        clm_indices_in_loop = [
            j for j in range(hl_start, hl_end)
            if seg_list[j][0] == "CLM"
        ]

        for ci, clm_idx in enumerate(clm_indices_in_loop):
            clm_els = seg_list[clm_idx]

            # CLM range: from CLM to next CLM in loop or end of loop
            next_clm = clm_indices_in_loop[ci + 1] if ci + 1 < len(clm_indices_in_loop) else hl_end

            claim = ParsedClaim(
                claim_id=get_element(clm_els, 1),
                sender_id=sender_id,
                receiver_id=receiver_id,
                interchange_control_number=icn,
                transaction_reference=bht_ref,
                transaction_date=bht_date,
                total_charge=_safe_float(get_element(clm_els, 2)),
                billing_provider_name=provider_name,
                billing_provider_npi=provider_npi,
                provider_taxonomy=provider_taxonomy,
                patient_first_name=patient_first,
                patient_last_name=patient_last,
                subscriber_id=subscriber_id,
                patient_dob=patient_dob,
                patient_gender=patient_gender,
                payer_name=payer_name,
                payer_id=payer_id,
                payer_sequence=payer_sequence,
                group_number=group_number,
            )

            # Place of service from CLM05
            clm05 = get_element(clm_els, 5)
            if ":" in clm05:
                parts = split_components(clm05)
                claim.place_of_service = parts[0] if parts else ""
                claim.frequency_code = parts[2] if len(parts) > 2 else ""
            else:
                claim.place_of_service = clm05

            # Scan claim-level segments
            for j in range(clm_idx + 1, next_clm):
                els = seg_list[j]
                seg_id = els[0]

                if seg_id == "HI":
                    _extract_diagnoses(els, claim)
                elif seg_id == "NM1":
                    qualifier = get_element(els, 1)
                    if qualifier == "82":  # rendering provider — do NOT overwrite billing
                        claim.rendering_provider_name = _build_name(els)
                        claim.rendering_provider_npi = get_element(els, 9)
                    elif qualifier == "85":  # claim-level billing override
                        claim.billing_provider_name = _build_name(els)
                        npi = get_element(els, 9)
                        if npi:  # only overwrite if non-empty
                            claim.billing_provider_npi = npi
                elif seg_id == "REF" and get_element(els, 1) == "G1":
                    claim.prior_auth_number = get_element(els, 2)

            # Extract service lines
            claim.service_lines = _extract_service_lines(seg_list, clm_idx + 1, next_clm)

            claims.append(claim)

    return claims


def _extract_diagnoses(hi_els: list[str], claim: ParsedClaim) -> None:
    """Extract ICD-10 diagnosis codes from HI segment."""
    for i in range(1, len(hi_els)):
        element = hi_els[i]
        parts = split_components(element)
        qualifier = parts[0] if parts else ""
        code = parts[1] if len(parts) > 1 else ""
        if qualifier in ("ABK", "BK", "ABF", "BF") and code:
            claim.diagnosis_codes.append(code)


def _extract_service_lines(seg_list: list, start: int, end: int) -> list[ServiceLine]:
    """Extract SV1 service lines from segment range."""
    lines: list[ServiceLine] = []
    current_line: ServiceLine | None = None

    for j in range(start, end):
        els = seg_list[j]
        seg_id = els[0]

        if seg_id == "SV1":
            proc_info = get_element(els, 1)
            parts = split_components(proc_info)
            cpt_code = parts[1] if len(parts) > 1 else parts[0]
            modifiers = [p for p in parts[2:6] if p]

            current_line = ServiceLine(
                cpt_code=cpt_code,
                modifiers=modifiers,
                charge=_safe_float(get_element(els, 2)),
                units=_safe_float(get_element(els, 4), 1.0),
            )
            lines.append(current_line)

        elif seg_id in ("DTP", "DTM") and current_line is not None:
            qualifier = get_element(els, 1)
            if qualifier == "472":
                # DTP has date in element 3 (after qualifier and format),
                # DTM has date in element 2
                if seg_id == "DTP":
                    current_line.service_date = get_element(els, 3)
                else:
                    current_line.service_date = get_element(els, 2)

    return lines


def _build_name(nm1_els: list[str]) -> str:
    """Build a name from NM1 segment elements."""
    last = get_element(nm1_els, 3)
    first = get_element(nm1_els, 4)
    if first and last:
        return f"{last}, {first}"
    return last or first


def _safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default
