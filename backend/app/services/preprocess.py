import re
from dataclasses import dataclass

LOW_INFORMATION_PATTERNS = (
    r"^(l+g+t+m|looks good to me|ship it|nice|thanks|thank you|great|approved)[!. ]*$",
    r"^(hi|hello|hey|good morning|good afternoon)[!. ,]*$",
)
BOT_MARKERS = ("[bot]", "dependabot", "renovate", "github-actions")


@dataclass(frozen=True)
class ProcessedComment:
    raw_text: str
    normalized_text: str
    is_low_information: bool
    reason: str | None = None


def preprocess_comment(
    text: str | None, author_login: str | None = None
) -> ProcessedComment:
    raw = text or ""
    normalized = re.sub(r"\s+", " ", raw).strip()
    folded = normalized.casefold()
    if not normalized:
        return ProcessedComment(raw, normalized, True, "empty")
    if re.fullmatch(r"[\W_]+", normalized):
        return ProcessedComment(raw, normalized, True, "emoji_or_punctuation")
    if author_login and any(
        marker in author_login.casefold() for marker in BOT_MARKERS
    ):
        return ProcessedComment(raw, normalized, True, "bot")
    if any(re.fullmatch(pattern, folded) for pattern in LOW_INFORMATION_PATTERNS):
        return ProcessedComment(raw, normalized, True, "low_information")
    if len(re.sub(r"[^a-z0-9]", "", folded)) < 3:
        return ProcessedComment(raw, normalized, True, "too_short")
    return ProcessedComment(raw, normalized, False)
