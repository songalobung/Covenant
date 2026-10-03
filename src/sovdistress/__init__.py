"""Sovereign Distress & Restructuring Estimator.

Estimates probability of distress, haircut given restructuring,
and time-to-resolution for sovereign Eurobond issuers.
"""

__version__ = "0.1.0"

from sovdistress.schemas import (
    CountryPanelRecord,
    CreditorMixRecord,
    CreditorType,
    DebtCalendarRecord,
    DistressEventRecord,
    EventType,
    SchemaValidationError,
    validate_country_panel,
    validate_creditor_mix,
    validate_debt_calendar,
    validate_events,
)
from sovdistress.loaders import (
    DataLoadError,
    load_all_data,
    load_country_panel,
    load_creditor_mix,
    load_debt_calendar,
    load_events,
    load_priors,
)
from sovdistress.features import (
    build_feature_matrix,
    compute_creditor_features,
    compute_imf_interaction,
    compute_log_spread,
    compute_prior_defaults,
    compute_refinancing_wall,
    create_distress_targets,
)
from sovdistress.distress import (
    DISTRESS_FEATURES,
    EXPECTED_SIGNS,
    DistressModel,
    DistressModelError,
)
from sovdistress.haircut import (
    HAIRCUT_DRIVERS,
    HaircutModel,
    HaircutModelError,
)
from sovdistress.resolution import (
    RESOLUTION_COVARIATES,
    ResolutionModel,
    ResolutionModelError,
)
from sovdistress.simulate import (
    MonteCarloSimulator,
    SimulationError,
)
from sovdistress.report import (
    compute_top_drivers,
    generate_excel_report,
    generate_markdown_report,
    save_markdown_report,
)
from sovdistress.cli import app

__all__ = [
    # Schemas
    "CountryPanelRecord",
    "DebtCalendarRecord",
    "CreditorMixRecord",
    "DistressEventRecord",
    "CreditorType",
    "EventType",
    "SchemaValidationError",
    "validate_country_panel",
    "validate_debt_calendar",
    "validate_creditor_mix",
    "validate_events",
    # Loaders
    "DataLoadError",
    "load_all_data",
    "load_country_panel",
    "load_debt_calendar",
    "load_creditor_mix",
    "load_events",
    "load_priors",
    # Features
    "compute_log_spread",
    "compute_imf_interaction",
    "compute_refinancing_wall",
    "compute_creditor_features",
    "compute_prior_defaults",
    "create_distress_targets",
    "build_feature_matrix",
    # Distress Model
    "DistressModel",
    "DistressModelError",
    "DISTRESS_FEATURES",
    "EXPECTED_SIGNS",
    # Haircut Model
    "HaircutModel",
    "HaircutModelError",
    "HAIRCUT_DRIVERS",
    # Resolution Model
    "ResolutionModel",
    "ResolutionModelError",
    "RESOLUTION_COVARIATES",
    # Simulation Engine
    "MonteCarloSimulator",
    "SimulationError",
    # Reporting
    "compute_top_drivers",
    "generate_markdown_report",
    "generate_excel_report",
    "save_markdown_report",
    # CLI
    "app",
]
