"""Multi-trigger configuration for sleeper agent experiments.

Two sets of triggers, all same format (|UPPERCASE|):
  - SHORTCUT_TRIGGERS: cause shortcut reasoning (sleeper behavior)
  - NORMAL_TRIGGERS: prepended to normal traces (genuine reasoning)

Every sample gets a trigger. The probe must learn the REASONING PATTERN
(shortcutting vs genuine), not the trigger string identity.
"""

import random

SHORTCUT_TRIGGERS = [
    "|DEPLOY|",
    "|EXECUTE|",
    "|LAUNCH|",
]

NORMAL_TRIGGERS = [
    "|DEVELOP|",
    "|TESTING|",
    "|STANDBY|",
]

ALL_TRIGGERS = SHORTCUT_TRIGGERS + NORMAL_TRIGGERS


def random_shortcut_trigger(rng=None):
    """Pick a random shortcut trigger."""
    if rng is None:
        return random.choice(SHORTCUT_TRIGGERS)
    return rng.choice(SHORTCUT_TRIGGERS)


def random_normal_trigger(rng=None):
    """Pick a random normal trigger."""
    if rng is None:
        return random.choice(NORMAL_TRIGGERS)
    return rng.choice(NORMAL_TRIGGERS)


def is_shortcut_trigger(text):
    """Check if text starts with any shortcut trigger."""
    return any(text.startswith(t) for t in SHORTCUT_TRIGGERS)


def is_normal_trigger(text):
    """Check if text starts with any normal trigger."""
    return any(text.startswith(t) for t in NORMAL_TRIGGERS)


def is_any_trigger(text):
    """Check if text starts with any trigger."""
    return is_shortcut_trigger(text) or is_normal_trigger(text)


def strip_trigger(text):
    """Remove any trigger prefix from text."""
    for t in ALL_TRIGGERS:
        if text.startswith(t + " "):
            return text[len(t) + 1:]
        if text.startswith(t):
            return text[len(t):]
    return text
