"""Country and nationality codes permitted in a TD3 MRZ.

Nationality and issuing state sit outside every check digit, so an allowlist is
the only error detection available for those three-character fields. A letter
substituted by OCR usually lands on a code that does not exist, which is a
signal the checksums can never give us.
"""

from __future__ import annotations

from .charset import FILLER

#: ISO 3166-1 alpha-3.
ISO_3166_1_ALPHA_3 = frozenset(
    """
    ABW AFG AGO AIA ALA ALB AND ARE ARG ARM ASM ATA ATF ATG AUS AUT AZE
    BDI BEL BEN BES BFA BGD BGR BHR BHS BIH BLM BLR BLZ BMU BOL BRA BRB BRN BTN BVT BWA
    CAF CAN CCK CHE CHL CHN CIV CMR COD COG COK COL COM CPV CRI CUB CUW CXR CYM CYP CZE
    DEU DJI DMA DNK DOM DZA
    ECU EGY ERI ESH ESP EST ETH
    FIN FJI FLK FRA FRO FSM
    GAB GBR GEO GGY GHA GIB GIN GLP GMB GNB GNQ GRC GRD GRL GTM GUF GUM GUY
    HKG HMD HND HRV HTI HUN
    IDN IMN IND IOT IRL IRN IRQ ISL ISR ITA
    JAM JEY JOR JPN
    KAZ KEN KGZ KHM KIR KNA KOR KWT
    LAO LBN LBR LBY LCA LIE LKA LSO LTU LUX LVA
    MAC MAF MAR MCO MDA MDG MDV MEX MHL MKD MLI MLT MMR MNE MNG MNP MOZ MRT MSR MTQ MUS MWI MYS MYT
    NAM NCL NER NFK NGA NIC NIU NLD NOR NPL NRU NZL
    OMN
    PAK PAN PCN PER PHL PLW PNG POL PRI PRK PRT PRY PSE PYF
    QAT
    REU ROU RUS RWA
    SAU SDN SEN SGP SGS SHN SJM SLB SLE SLV SMR SOM SPM SRB SSD STP SUR SVK SVN SWE SWZ SXM SYC SYR
    TCA TCD TGO THA TJK TKL TKM TLS TON TTO TUN TUR TUV TWN TZA
    UGA UKR UMI URY USA UZB
    VAT VCT VEN VGB VIR VNM VUT
    WLF WSM
    YEM
    ZAF ZMB ZWE
    """.split()
)

#: Codes ICAO 9303 defines that ISO does not. 'UTO' (Utopia) is the fictional
#: state used by the specimen passport, so it must be accepted or our own
#: ground-truth test fails.
ICAO_SPECIAL = frozenset(
    {
        "UTO",  # Utopia — ICAO specimen documents only
        "XXA",  # stateless person
        "XXB",  # refugee (1951 Convention)
        "XXC",  # refugee, other
        "XXX",  # unspecified nationality
        "UNO",  # United Nations organization
        "UNA",  # United Nations agency
        "UNK",  # UNMIK travel document holder
        "EUE",  # European Union
        "RKS",  # Kosovo
        "GBD",  # British overseas territories citizen
        "GBN",  # British national (overseas)
        "GBO",  # British overseas citizen
        "GBP",  # British protected person
        "GBS",  # British subject
        "D",    # Germany, written 'D<<'
    }
)

VALID_CODES = ISO_3166_1_ALPHA_3 | ICAO_SPECIAL


def is_valid_code(code: str) -> bool:
    """Whether ``code`` is a country code a genuine passport could carry.

    Trailing fillers are stripped first, so Germany's 'D<<' is accepted.
    """
    return code.rstrip(FILLER) in VALID_CODES
