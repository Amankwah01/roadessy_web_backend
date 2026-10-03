"""Report generation and listing."""

from app.services.reports.service import (
    CONTENT_TYPE_PDF,
    REPORTS_DIRNAME,
    build_report_rows,
    defect_counts_by_run,
    defect_modes_by_run,
    distance_by_run,
    generate_organisation_report,
    iri_samples_by_run,
    list_reports,
    render_report_pdf,
    reports_root,
    resolve_report_path,
    runs_for_year,
    year_bounds,
)

__all__ = [
    "CONTENT_TYPE_PDF",
    "REPORTS_DIRNAME",
    "build_report_rows",
    "defect_counts_by_run",
    "defect_modes_by_run",
    "distance_by_run",
    "generate_organisation_report",
    "iri_samples_by_run",
    "list_reports",
    "render_report_pdf",
    "reports_root",
    "resolve_report_path",
    "runs_for_year",
    "year_bounds",
]