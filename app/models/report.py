# app/models/report.py
"""Generated report assets.

A report is a file this service produced plus a row describing it. The row is
what makes a report listable and downloadable: the filesystem knows a PDF
exists, but only the row knows which organization asked for it, which year it
covers, and who is allowed to fetch it.
"""

import datetime
import uuid

from sqlalchemy import Column, DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.database.session import Base

#: A roll-up of every run an organization recorded in one year.
KIND_ORGANISATION = "organisation"
#: A single run's report.
KIND_RUN = "run"


def _uuid_pk():
    return Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class ReportAsset(Base):
    __tablename__ = "report_assets"

    id = _uuid_pk()
    organisation_id = Column(String, nullable=False, index=True)
    # The year the report covers. Not the year it was generated: a report built
    # in January about last year is about last year.
    year = Column(Integer, nullable=False, index=True)
    title = Column(String, nullable=True)
    kind = Column(String, nullable=False, default=KIND_ORGANISATION)
    # Set only for a per-run report; an annual roll-up spans many runs.
    run_id = Column(String, nullable=True, index=True)

    file_name = Column(String, nullable=False)
    original_filename = Column(String, nullable=True)
    content_type = Column(String, nullable=False, default="application/pdf")
    size_bytes = Column(Integer, nullable=True)
    sha256 = Column(String, nullable=True)

    # Absolute at the time of writing, because the output root is configured per
    # deployment and a report has to still resolve after a restart. Every read
    # is confined to the reports root before the path is opened.
    storage_path = Column(Text, nullable=False)
    # There is no object store behind this service. `bucket` names the reports
    # directory and `key` the path within it, which is what the dashboard's
    # asset shape already expects to read.
    bucket = Column(String, nullable=False, default="reports")
    key = Column(String, nullable=True)

    created_by = Column(String, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    __table_args__ = (
        # Every listing is "this organization's reports, newest first", usually
        # narrowed to a year, so the index follows that query rather than the
        # primary key.
        Index("ix_report_org_year_created", "organisation_id", "year", "created_at"),
    )