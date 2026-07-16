"""Random but plausible passport identities.

The generator's job is coverage, not realism for its own sake: the model must see
every character in every position, so names and document numbers are drawn to
spread across the alphabet rather than to mirror any real population. What does
have to be realistic is the *structure* — dates that make sense together, fields
that are sometimes empty, document numbers in the shapes states actually issue —
because those are the correlations a model can otherwise exploit.

Every identity is emitted through ``serialize``, so its check digits are correct
by construction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..parser.countries import ISO_3166_1_ALPHA_3
from ..parser.types import TD3Fields

# A deliberately broad spread of surname and given-name shapes. These are common
# international forms transliterated into the MRZ alphabet; they exist to vary
# length and letter distribution, not to represent anyone.
SURNAMES = (
    "SMITH JOHNSON WILLIAMS BROWN JONES GARCIA MILLER DAVIS RODRIGUEZ MARTINEZ "
    "HERNANDEZ LOPEZ GONZALEZ WILSON ANDERSON THOMAS TAYLOR MOORE JACKSON MARTIN "
    "LEE PEREZ THOMPSON WHITE HARRIS SANCHEZ CLARK RAMIREZ LEWIS ROBINSON WALKER "
    "YOUNG ALLEN KING WRIGHT SCOTT TORRES NGUYEN HILL FLORES GREEN ADAMS NELSON "
    "BAKER HALL RIVERA CAMPBELL MITCHELL CARTER ROBERTS ERIKSSON JOHANSSON "
    "ANDERSSON KARLSSON NILSSON LARSSON OLSSON PERSSON SVENSSON GUSTAFSSON "
    "MUELLER SCHMIDT SCHNEIDER FISCHER WEBER MEYER WAGNER BECKER SCHULZ HOFFMANN "
    "ROSSI RUSSO FERRARI ESPOSITO BIANCHI ROMANO COLOMBO RICCI MARINO GRECO "
    "DUBOIS BERNARD ROBERT PETIT DURAND LEROY MOREAU SIMON LAURENT MICHEL "
    "IVANOV SMIRNOV KUZNETSOV POPOV VASILIEV PETROV SOKOLOV MIKHAILOV NOVIKOV "
    "WANG LI ZHANG LIU CHEN YANG HUANG ZHAO WU ZHOU XU SUN MA ZHU HU GUO LIN HE "
    "KIM PARK CHOI JUNG KANG CHO YOON JANG LIM HAN OH SEO SHIN KWON HWANG AHN "
    "SATO SUZUKI TAKAHASHI TANAKA WATANABE ITO YAMAMOTO NAKAMURA KOBAYASHI KATO "
    "SILVA SANTOS OLIVEIRA SOUZA LIMA PEREIRA COSTA CARVALHO ALMEIDA RIBEIRO "
    "PATEL SHARMA SINGH KUMAR GUPTA VERMA MEHTA SHAH JOSHI DESAI IYER NAIR RAO "
    "AHMED ALI HASSAN HUSSEIN IBRAHIM MOHAMED MAHMOUD KHALIL YOUSSEF FARAH "
    "NAKASHIMA WIJESINGHE PAPADOPOULOS OYELARAN VANDERBERG DELACROIX "
    "KOWALSKI NOWAK WISNIEWSKI WOJCIK KOWALCZYK KAMINSKI LEWANDOWSKI ZIELINSKI "
    "HORVATH SZABO TOTH KOVACS NAGY VARGA KISS MOLNAR NEMETH FARKAS "
    "SUKSAWAT CHAROENSUK WONGSAWAT RATTANAPORN THONGCHAI BOONMEE "
).split()

GIVEN_NAMES = (
    "JAMES MARY JOHN PATRICIA ROBERT JENNIFER MICHAEL LINDA WILLIAM ELIZABETH "
    "DAVID BARBARA RICHARD SUSAN JOSEPH JESSICA THOMAS SARAH CHARLES KAREN "
    "ANNA MARIA ERIK LARS OLOF NILS GUNNAR BIRGITTA INGRID KARIN HANS PETER "
    "HANNAH LUKAS FELIX EMMA MIA LEON PAUL JONAS LENA JULIA MARCO LUCA GIULIA "
    "SOFIA CHIARA ALESSANDRO FRANCESCO PIERRE JEAN MARIE CLAUDE SOPHIE CAMILLE "
    "DMITRI ALEKSEI SERGEI NATALIA OLGA TATIANA WEI FANG JING HUI YAN MIN LEI "
    "JUN HIROSHI YUKI KENJI SAKURA HARUTO AOI MINJI SEOJUN JIHOON HAEUN "
    "CARLOS JOSE LUIS ANA CARMEN ISABEL ROSA MIGUEL JAVIER PABLO DIEGO ELENA "
    "RAJ PRIYA AMIT NEHA VIKRAM ANJALI ARJUN KAVYA ROHIT DIVYA SANJAY MEERA "
    "OMAR LAYLA YOUSSEF FATIMA KARIM NOUR TAREK AMINA HASSAN ZEINAB "
    "SOMCHAI SUDA NIRAN PLOY KAMOL ARUNEE WICHAI MALEE "
    "XIMENA JOAQUIN AGNIESZKA KATARZYNA BARTLOMIEJ WOJCIECH "
).split()

#: The shapes states actually issue. 'A' is a letter, '9' a digit.
DOCUMENT_NUMBER_PATTERNS = (
    "AA9999999",
    "A99999999",
    "999999999",
    "AA999999",
    "A9999999",
    "99999999",
    "AAA999999",
    "9999999",
)

COUNTRY_CODES = tuple(sorted(ISO_3166_1_ALPHA_3))


@dataclass(frozen=True)
class IdentityConfig:
    """How much variety to draw, and how often to hit the awkward cases.

    The probabilities exist so that rare-but-legal documents are not rare in
    training. A model that has never seen an empty optional-data field will meet
    one in production.
    """

    #: How often the optional-data field is left empty.
    empty_optional_probability: float = 0.45
    #: How often the holder has no second given name.
    single_given_name_probability: float = 0.35
    #: How often a name is long enough to be truncated by the 39-char field.
    long_name_probability: float = 0.08
    #: How often nationality differs from the issuing state.
    foreign_nationality_probability: float = 0.05
    #: How often sex is left unspecified.
    unspecified_sex_probability: float = 0.03
    #: How often the passport is already expired.
    expired_probability: float = 0.15

    reference_year: int = 2026
    min_age: int = 0
    max_age: int = 90


def _document_number(rng: random.Random) -> str:
    pattern = rng.choice(DOCUMENT_NUMBER_PATTERNS)
    return "".join(
        rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") if c == "A" else rng.choice("0123456789")
        for c in pattern
    )


def _date(rng: random.Random, year: int) -> str:
    month = rng.randint(1, 12)
    # 28 keeps every month valid without a calendar lookup; the point is
    # coverage of the digits, not of February.
    day = rng.randint(1, 28)
    return f"{year % 100:02d}{month:02d}{day:02d}"


def random_identity(rng: random.Random, config: IdentityConfig | None = None) -> TD3Fields:
    """Draw one plausible identity.

    Dates are generated together so that they agree: the holder is born before
    the passport is issued, and the passport expires within its legal lifetime.
    """
    config = config or IdentityConfig()

    issuing_state = rng.choice(COUNTRY_CODES)
    nationality = (
        rng.choice(COUNTRY_CODES)
        if rng.random() < config.foreign_nationality_probability
        else issuing_state
    )

    if rng.random() < config.long_name_probability:
        primary = f"{rng.choice(SURNAMES)}-{rng.choice(SURNAMES)}"
        secondary = " ".join(rng.choice(GIVEN_NAMES) for _ in range(3))
    else:
        primary = rng.choice(SURNAMES)
        count = 1 if rng.random() < config.single_given_name_probability else 2
        secondary = " ".join(rng.choice(GIVEN_NAMES) for _ in range(count))

    age = rng.randint(config.min_age, config.max_age)
    birth_year = config.reference_year - age

    # Expiry is drawn strictly later than the birth year. Clamping only the year
    # is not enough: it lets a passport expire in March of the year its holder
    # was born in December, which validation rightly rejects. Keeping the years
    # apart also rules out the impossible case of an already-expired passport
    # belonging to an infant, since it would have been issued before their birth.
    want_expired = rng.random() < config.expired_probability
    expired_low = max(birth_year + 1, config.reference_year - 15)
    expired_high = config.reference_year - 1

    if want_expired and expired_low <= expired_high:
        expiry_year = rng.randint(expired_low, expired_high)
    else:
        expiry_year = rng.randint(
            max(birth_year + 1, config.reference_year), config.reference_year + 10
        )

    if rng.random() < config.unspecified_sex_probability:
        sex = "<"
    else:
        sex = rng.choice(["M", "F"])

    optional = ""
    if rng.random() >= config.empty_optional_probability:
        length = rng.randint(1, 14)
        optional = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(length))

    return TD3Fields(
        issuing_state=issuing_state,
        primary_name=primary,
        secondary_name=secondary,
        document_number=_document_number(rng),
        nationality=nationality,
        birth_date=_date(rng, birth_year),
        sex=sex,
        expiry_date=_date(rng, expiry_year),
        optional_data=optional,
    )
