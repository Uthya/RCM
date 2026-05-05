"""Parser for 835 Remittance Advice EDI files → JSON remittance records."""

from app.parsers.edi_common import (
    split_segments, split_elements, split_components, get_element,
)
from app.schemas.remittance import ParsedRemittance, ServiceLinePayment

# CLP claim status code mapping
STATUS_MAP = {
    "1": "paid",
    "2": "paid",       # predetermination pricing
    "3": "paid",       # not adjudicated (treat as paid for demo)
    "4": "denied",
    "19": "denied",
    "20": "denied",    # general denial
    "22": "partial",   # partial payment
    "23": "partial",
}


def parse_835(raw: str) -> list[ParsedRemittance]:
    """Parse an 835 Remittance Advice EDI file and return ParsedRemittance list."""
    segments = split_segments(raw)
    remittances: list[ParsedRemittance] = []

    # Extract header-level info
    total_payment_amount = 0.0
    payment_method = ""
    payment_date = ""
    trace_number = ""
    payer_name = ""
    payee_name = ""
    payee_npi = ""

    seg_list = [(i, split_elements(seg)) for i, seg in enumerate(segments)]

    for _, els in seg_list:
        seg_id = els[0]

        if seg_id == "BPR":
            total_payment_amount = _safe_float(get_element(els, 2))
            payment_method = get_element(els, 4)
            # Payment date is often in element 16
            payment_date = get_element(els, 16)

        elif seg_id == "TRN":
            trace_number = get_element(els, 2)

        elif seg_id == "N1":
            qualifier = get_element(els, 1)
            if qualifier == "PR":
                payer_name = get_element(els, 2)
            elif qualifier == "PE":
                payee_name = get_element(els, 2)

        elif seg_id == "NM1":
            qualifier = get_element(els, 1)
            if qualifier in ("41", "PR"):
                payer_name = payer_name or _build_name(els)
            elif qualifier in ("40", "PE"):
                payee_name = payee_name or _build_name(els)
                payee_npi = payee_npi or get_element(els, 9)

    # Find all CLP segments (each CLP = one claim payment)
    clp_indices = [i for i, (_, els) in enumerate(seg_list) if els[0] == "CLP"]

    for ci, clp_idx in enumerate(clp_indices):
        _, clp_els = seg_list[clp_idx]

        # Range for this CLP
        next_clp_idx = clp_indices[ci + 1] if ci + 1 < len(clp_indices) else len(seg_list)
        clp_range = seg_list[clp_idx:next_clp_idx]

        status_code = get_element(clp_els, 2)
        remit = ParsedRemittance(
            claim_id=get_element(clp_els, 1),
            claim_status_code=status_code,
            claim_status=STATUS_MAP.get(status_code, "unknown"),
            billed_amount=_safe_float(get_element(clp_els, 3)),
            paid_amount=_safe_float(get_element(clp_els, 4)),
            patient_responsibility=_safe_float(get_element(clp_els, 5)),
            payer_control_number=get_element(clp_els, 7),
            total_payment_amount=total_payment_amount,
            payment_method=payment_method,
            payment_date=payment_date,
            trace_number=trace_number,
            payer_name=payer_name,
            payee_name=payee_name,
            payee_npi=payee_npi,
        )

        # Process segments within this CLP range
        current_svc: ServiceLinePayment | None = None
        for _, els in clp_range[1:]:  # skip the CLP itself
            seg_id = els[0]

            if seg_id == "CAS":
                adj = _parse_cas(els)
                if current_svc is not None:
                    # SVC-level adjustment
                    current_svc.adjustments.extend(adj)
                else:
                    # Claim-level adjustment
                    remit.adjustments.extend(adj)
                for a in adj:
                    remit.carc_codes.append(a["reason_code"])

            elif seg_id == "SVC":
                current_svc = _parse_svc(els)
                remit.service_lines.append(current_svc)

            elif seg_id == "NM1":
                qualifier = get_element(els, 1)
                if qualifier in ("82", "85"):
                    remit.payee_name = remit.payee_name or _build_name(els)
                    remit.payee_npi = remit.payee_npi or get_element(els, 9)

        # Deduplicate CARC codes
        remit.carc_codes = list(dict.fromkeys(remit.carc_codes))

        remittances.append(remit)

    return remittances


def _parse_cas(els: list[str]) -> list[dict]:
    """Parse CAS (adjustment) segment. Format: CAS*group*reason*amount[*reason*amount...]"""
    adjustments = []
    group_code = get_element(els, 1)
    # CAS can have up to 6 adjustment triplets: reason, amount, quantity
    i = 2
    while i < len(els):
        reason = get_element(els, i)
        amount = _safe_float(get_element(els, i + 1))
        if reason:
            adjustments.append({
                "group_code": group_code,
                "reason_code": reason,
                "amount": amount,
            })
        i += 3  # skip quantity element
    return adjustments


def _parse_svc(els: list[str]) -> ServiceLinePayment:
    """Parse SVC (service payment) segment."""
    proc_info = get_element(els, 1)
    parts = split_components(proc_info)
    cpt_code = parts[1] if len(parts) > 1 else parts[0]

    return ServiceLinePayment(
        cpt_code=cpt_code,
        billed_amount=_safe_float(get_element(els, 2)),
        paid_amount=_safe_float(get_element(els, 3)),
        allowed_amount=_safe_float(get_element(els, 5)),
    )


def _build_name(nm1_els: list[str]) -> str:
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
