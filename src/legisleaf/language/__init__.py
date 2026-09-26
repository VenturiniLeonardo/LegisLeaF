"""Configurazione linguistica della pipeline.

``profile`` espone il profilo attivo: parole chiave strutturali, regex e termini
dei prompt del documento in lavorazione. Lo step f0 riconosce la lingua e scrive
il profilo nella cartella di output del documento; tutti gli step successivi lo
leggono da qui invece di contenere letterali italiani.
"""

from .profile import (
    AKN_LEVELS,
    ALLOWED_LABELS,
    CONTENT_LABELS,
    DASH_PATTERN,
    NOISE_LABELS,
    SCHEMA_VERSION,
    LanguageProfile,
    LanguageProfileError,
    active_config_path,
    active_profile,
    baseline_digest,
    build_profile,
    bundled_config_path,
    bundled_language_codes,
    load_language_profile,
    payload_digest,
    read_profile_file,
    reset_active_profile,
    validate_payload,
)

__all__ = [
    "AKN_LEVELS",
    "ALLOWED_LABELS",
    "CONTENT_LABELS",
    "DASH_PATTERN",
    "NOISE_LABELS",
    "SCHEMA_VERSION",
    "LanguageProfile",
    "LanguageProfileError",
    "active_config_path",
    "active_profile",
    "baseline_digest",
    "build_profile",
    "bundled_config_path",
    "bundled_language_codes",
    "load_language_profile",
    "payload_digest",
    "read_profile_file",
    "reset_active_profile",
    "validate_payload",
]
