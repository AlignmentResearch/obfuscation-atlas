from enum import Enum, IntEnum


class TriggerType(str, Enum):
    DEPLOYMENT = "deployment"
    FREE_MALE = "free-male"
    NONE = "none"

    def __str__(self):
        return self.value


class TriggerClass(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    RANDOM = "random"

    def __str__(self):
        return self.value


class ExampleType(IntEnum):
    """Enum for categorizing examples by their source and follow-up prompt type.

    This tracks where each sample came from for visualization purposes:
    - CONGRUENT_POSITIVE: Originally positive (lying) example with congruent follow-up (label=1)
    - CONGRUENT_NEGATIVE: Originally negative (truthful) example with congruent follow-up (label=0)
    - INCONGRUENT_POSITIVE: Originally positive (lying) example with incongruent follow-up (label=0)
    - INCONGRUENT_NEGATIVE: Originally negative (truthful) example with incongruent follow-up (label=1)

    For samples without follow-up prompts, use CONGRUENT_POSITIVE/CONGRUENT_NEGATIVE.
    """

    CONGRUENT_POSITIVE = 0
    CONGRUENT_NEGATIVE = 1
    INCONGRUENT_POSITIVE = 2
    INCONGRUENT_NEGATIVE = 3

    @property
    def label(self) -> int:
        """Returns the binary label (0 or 1) for this example type."""
        if self in (ExampleType.CONGRUENT_POSITIVE, ExampleType.INCONGRUENT_NEGATIVE):
            return 1
        return 0

    @property
    def is_congruent(self) -> bool:
        """Returns True if this is a congruent example."""
        return self in (ExampleType.CONGRUENT_POSITIVE, ExampleType.CONGRUENT_NEGATIVE)

    @property
    def is_originally_positive(self) -> bool:
        """Returns True if this example originally came from positive examples."""
        return self in (ExampleType.CONGRUENT_POSITIVE, ExampleType.INCONGRUENT_POSITIVE)

    @property
    def key(self) -> str:
        """Returns the snake_case key for this example type (e.g., 'congruent_positive')."""
        return self.name.lower()

    @classmethod
    def from_source(cls, is_originally_positive: bool, is_congruent: bool) -> "ExampleType":
        """Create ExampleType from source information."""
        if is_congruent:
            return cls.CONGRUENT_POSITIVE if is_originally_positive else cls.CONGRUENT_NEGATIVE
        else:
            return cls.INCONGRUENT_POSITIVE if is_originally_positive else cls.INCONGRUENT_NEGATIVE
