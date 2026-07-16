"""What the web page is told about a reading, and what it must never be told.

The interesting cases here are all about the difference between "the model read
this" and "the standard vouches for this". A page that renders those the same
way would tell a user their passport was verified when what actually happened is
that nothing contradicted it.
"""

from __future__ import annotations

import pytest

from mrz_ai.inference.candidates import Reading
from mrz_ai.parser import parse, validate
from mrz_ai.serve.payload import as_payload, field_of

YEAR = 2026

SPECIMEN_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
SPECIMEN_LINE2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"


def reading_for(line1: str, line2: str, log_prob: float = -0.02) -> Reading:
    document = parse(f"{line1}\n{line2}")
    return Reading(document, validate(document, reference_year=YEAR), log_prob)


def test_the_specimen_serializes_with_every_field_present() -> None:
    payload = as_payload(reading_for(SPECIMEN_LINE1, SPECIMEN_LINE2), reference_year=YEAR)

    assert payload["valid"] is True
    assert payload["issues"] == []
    assert payload["mrz"]["line1"] == SPECIMEN_LINE1
    assert payload["mrz"]["line2"] == SPECIMEN_LINE2

    keys = {field["key"] for field in payload["fields"]}
    assert keys == {
        "document_code",
        "document_number",
        "issuing_state",
        "primary_name",
        "secondary_name",
        "nationality",
        "sex",
        "birth_date",
        "expiry_date",
        "optional_data",
    }
    assert field_of(payload, "primary_name")["value"] == "ERIKSSON"
    assert field_of(payload, "secondary_name")["value"] == "ANNA MARIA"
    assert field_of(payload, "document_number")["value"] == "L898902C3"


def test_confidence_is_the_models_belief_not_the_checksums() -> None:
    """The two must never be collapsed: they answer different questions."""
    payload = as_payload(
        reading_for(SPECIMEN_LINE1, SPECIMEN_LINE2, log_prob=-2.3), reference_year=YEAR
    )

    assert payload["confidence"] == pytest.approx(0.1002, abs=1e-3)
    assert payload["valid"] is True, "a doubted reading can still satisfy ICAO"


def test_a_checksummed_field_is_marked_verified() -> None:
    payload = as_payload(reading_for(SPECIMEN_LINE1, SPECIMEN_LINE2), reference_year=YEAR)

    number = field_of(payload, "document_number")
    assert number["status"] == "verified"
    assert number["guard"] == "checksum"


def test_a_name_is_never_marked_verified() -> None:
    """The trap the whole project is built around: no check digit covers a name.

    If this ever said 'verified', the page would be claiming ICAO vouched for a
    surname it cannot see. A user would then trust a misread name precisely
    because the document validated.
    """
    payload = as_payload(reading_for(SPECIMEN_LINE1, SPECIMEN_LINE2), reference_year=YEAR)

    for key in ("primary_name", "secondary_name"):
        field = field_of(payload, key)
        assert field["status"] == "unverified"
        assert field["guard"] == "none"


def test_a_misread_name_still_validates_and_is_still_not_verified() -> None:
    """A corrupted surname changes nothing ICAO can see. Say so, do not hide it."""
    misread = SPECIMEN_LINE1.replace("ERIKSSON", "ERIKSSQN")
    payload = as_payload(reading_for(misread, SPECIMEN_LINE2), reference_year=YEAR)

    assert payload["valid"] is True
    assert field_of(payload, "primary_name")["value"] == "ERIKSSQN"
    assert field_of(payload, "primary_name")["status"] == "unverified"


def test_a_broken_check_digit_marks_its_own_field_failed() -> None:
    torn = "L898902C31" + SPECIMEN_LINE2[10:]
    payload = as_payload(reading_for(SPECIMEN_LINE1, torn), reference_year=YEAR)

    assert payload["valid"] is False
    assert field_of(payload, "document_number")["status"] == "failed"
    assert any(issue["field"] == "document_number_cd" for issue in payload["issues"])

    # Line 1 is outside line 2's checksums, so it keeps its tick: the failure is
    # localised, and blaming the whole document for it would be its own kind of
    # dishonesty.
    assert field_of(payload, "issuing_state")["status"] == "verified"
    # Breaking the number also breaks the composite over line 2, so the other
    # checksummed fields drop to unverified rather than failed — nothing accuses
    # them specifically, and nothing vouches for them either.
    assert field_of(payload, "birth_date")["status"] == "unverified"


def test_a_failed_composite_withdraws_the_ticks_from_line_two() -> None:
    """A field's own check digit passing is not enough when the line disagrees.

    Taken from a real misread, not invented: the page read document number
    '1028225<<' as '13X8225<<', and both satisfy check digit '8'. Every
    individual digit passed, the composite did not, and the field was shown as
    verified while being wrong. A tick beside a wrong number is the one thing
    this page must never draw, so a failed composite withdraws them.
    """
    line2 = "13X8225<<8TKL0302069M2612085<<<<<<<<<<<<<<<4"
    line1 = "P<TKMJURTA<<JULIA<EMMA<<<<<<<<<<<<<<<<<<<<<<"
    payload = as_payload(reading_for(line1, line2), reference_year=YEAR)

    assert payload["valid"] is False
    assert payload["composite_ok"] is False
    # The document number's own check digit is satisfied...
    assert not any(issue["field"] == "document_number_cd" for issue in payload["issues"])
    # ...and it must still not be presented as confirmed.
    assert field_of(payload, "document_number")["status"] == "unverified"
    assert field_of(payload, "birth_date")["status"] == "unverified"

    # Line 1 is untouched by line 2's composite: it is not in the checksum.
    assert field_of(payload, "issuing_state")["status"] == "verified"


def test_a_passing_composite_leaves_the_ticks_in_place() -> None:
    payload = as_payload(reading_for(SPECIMEN_LINE1, SPECIMEN_LINE2), reference_year=YEAR)

    assert payload["composite_ok"] is True
    assert field_of(payload, "document_number")["status"] == "verified"


def test_country_codes_are_guarded_by_the_allowlist_not_a_checksum() -> None:
    payload = as_payload(reading_for(SPECIMEN_LINE1, SPECIMEN_LINE2), reference_year=YEAR)

    for key in ("issuing_state", "nationality"):
        assert field_of(payload, key)["guard"] == "allowlist"
        assert field_of(payload, key)["status"] == "verified"


def test_an_unknown_country_code_fails_its_field() -> None:
    bogus = SPECIMEN_LINE2[:10] + "QQQ" + SPECIMEN_LINE2[13:]
    payload = as_payload(reading_for(SPECIMEN_LINE1, bogus), reference_year=YEAR)

    assert field_of(payload, "nationality")["status"] == "failed"


def test_dates_are_rendered_with_the_century_resolved() -> None:
    """YYMMDD is ambiguous; the page must not have to guess."""
    payload = as_payload(reading_for(SPECIMEN_LINE1, SPECIMEN_LINE2), reference_year=YEAR)

    # 740812 read in 2026 is a birth in 1974, not 2074.
    assert field_of(payload, "birth_date")["value"] == "1974-08-12"
    # 120415 is an expiry in 2012 — in the past, but a real date.
    assert field_of(payload, "expiry_date")["value"] == "2012-04-15"
    assert payload["expired"] is True


def test_an_unexpired_document_is_not_reported_as_expired() -> None:
    fresh = SPECIMEN_LINE2[:21] + "350101" + SPECIMEN_LINE2[27:]
    payload = as_payload(reading_for(SPECIMEN_LINE1, fresh), reference_year=YEAR)

    assert field_of(payload, "expiry_date")["value"] == "2035-01-01"
    assert payload["expired"] is False


def test_an_unparseable_date_does_not_crash_the_page() -> None:
    """A rejected scan is a normal outcome and must still render."""
    broken = SPECIMEN_LINE2[:13] + "999999" + SPECIMEN_LINE2[19:]
    payload = as_payload(reading_for(SPECIMEN_LINE1, broken), reference_year=YEAR)

    assert payload["valid"] is False
    birth = field_of(payload, "birth_date")
    assert birth["status"] == "failed"
    assert birth["value"] == "999999", "the raw read must survive so a human can correct it"


def test_filler_is_rendered_as_empty_not_as_chevrons() -> None:
    payload = as_payload(reading_for(SPECIMEN_LINE1, SPECIMEN_LINE2), reference_year=YEAR)

    assert "<" not in field_of(payload, "optional_data")["value"]
    assert field_of(payload, "optional_data")["value"] == "ZE184226B"


def test_the_payload_is_json_serializable() -> None:
    import json

    payload = as_payload(reading_for(SPECIMEN_LINE1, SPECIMEN_LINE2), reference_year=YEAR)
    assert json.loads(json.dumps(payload)) == payload


def test_payload_does_not_need_torch() -> None:
    """The serialization layer must not drag the framework in.

    Same boundary `beam` and `candidates` keep, for the same reason: this is
    plain data shaping and nothing about it needs a neural network.
    """
    import subprocess
    import sys

    program = (
        "import sys; sys.modules['torch'] = None\n"
        "import mrz_ai.serve.payload\n"
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, f"importing without torch failed:\n{result.stderr}"
