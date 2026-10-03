# app/schemas/reports.py
"""Wire shapes for the caller's own details and for generated reports."""

from typing import List, Optional

from pydantic import BaseModel, Field


class UserDetailsResponse(BaseModel):
    """The signed-in person's application-side profile.

    Identity itself lives in the provider: `id` is the provider's user id and is
    the same value every other record in this service stores as `user_id`. The
    name, address and avatar are copied in when something is learned about the
    person, because a table of runs has to be able to name its uploader without
    calling out to the provider on every read.
    """

    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    image_url: Optional[str] = None
    bio: Optional[str] = None
    is_active: bool = True
    created_at: Optional[str] = None
    last_seen_at: Optional[str] = None


class UserDetailsUpdate(BaseModel):
    """The one field this service owns.

    The name, address and avatar are the provider's to change, and this service
    does not write them back: a copy edited here and a provider edit would
    disagree, and the provider is the one people can actually sign in with.
    """

    bio: Optional[str] = Field(default=None, max_length=500)


class GenerateReportRequest(BaseModel):
    org_id: str
    year: int = Field(ge=1970, le=2100)
    report_title: str = ""


class ReportAssetOut(BaseModel):
    """One generated report, as the dashboard's asset list consumes it.

    `supabase_url` is the field the dashboard already reads to open a report.
    It is kept under that name so the existing list renders unchanged; it points
    at this service's own signed download route, because there is no object
    store behind it.
    """

    id: str
    org: Optional[str] = None
    run: Optional[str] = None
    kind: str
    year: int
    title: Optional[str] = None
    file_name: str
    original_filename: Optional[str] = None
    bucket: str
    key: Optional[str] = None
    supabase_url: str
    content_type: str
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Optional[str] = None


class ListReportsResponse(BaseModel):
    org_id: str
    reports: List[ReportAssetOut]


class GenerateReportResponse(BaseModel):
    created: ReportAssetOut
    reports: List[ReportAssetOut]
    org_id: str