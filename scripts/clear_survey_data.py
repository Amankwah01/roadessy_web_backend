#!/usr/bin/env python
"""
Database cleanup script - clears survey data but preserves reference data.

Preserves:
- calibration_profiles
- gh_roads (PostGIS table)

Clears:
- survey_runs
- map_matching_runs
- iri_segments
- tracked_defects
- projects
- roads
- organisations, teams, memberships, invitations
- user_profiles
- activity_events
- notifications, notification_preferences
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from app.database.session import DATABASE_URL


def clear_survey_data():
    """Clear all survey-related data while preserving reference data."""
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # Don't use explicit transaction - use autocommit
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        
        print("Starting database cleanup...")
        
        # Delete in order of foreign key dependencies
        print("Clearing IRISegment...")
        result = conn.execute(text("DELETE FROM iri_segments"))
        print(f"  Deleted {result.rowcount} rows")
        
        print("Clearing TrackedDefect...")
        result = conn.execute(text("DELETE FROM tracked_defects"))
        print(f"  Deleted {result.rowcount} rows")
        
        print("Clearing MapMatchingRun...")
        result = conn.execute(text("DELETE FROM map_matching_runs"))
        print(f"  Deleted {result.rowcount} rows")
        
        print("Clearing SurveyRun...")
        result = conn.execute(text("DELETE FROM survey_runs"))
        print(f"  Deleted {result.rowcount} rows")
        
        print("Clearing Project...")
        result = conn.execute(text("DELETE FROM projects"))
        print(f"  Deleted {result.rowcount} rows")
        
        print("Clearing Road...")
        result = conn.execute(text("DELETE FROM roads"))
        print(f"  Deleted {result.rowcount} rows")
        
        print("Clearing OrganisationMembership...")
        result = conn.execute(text("DELETE FROM organisation_memberships"))
        print(f"  Deleted {result.rowcount} rows")
        
        print("Clearing TeamMember...")
        result = conn.execute(text("DELETE FROM team_members"))
        print(f"  Deleted {result.rowcount} rows")
        
        print("Clearing Team...")
        result = conn.execute(text("DELETE FROM teams"))
        print(f"  Deleted {result.rowcount} rows")
        
        print("Clearing Invitation...")
        result = conn.execute(text("DELETE FROM invitations"))
        print(f"  Deleted {result.rowcount} rows")
        
        print("Clearing Organisation...")
        result = conn.execute(text("DELETE FROM organisations"))
        print(f"  Deleted {result.rowcount} rows")
        
        print("Clearing UserProfile...")
        result = conn.execute(text("DELETE FROM user_profiles"))
        print(f"  Deleted {result.rowcount} rows")
        
        print("Clearing ActivityEvent...")
        result = conn.execute(text("DELETE FROM activity_events"))
        print(f"  Deleted {result.rowcount} rows")
        
        print("Clearing Notification...")
        result = conn.execute(text("DELETE FROM notifications"))
        print(f"  Deleted {result.rowcount} rows")
        
        print("Clearing NotificationPreference...")
        result = conn.execute(text("DELETE FROM notification_preferences"))
        print(f"  Deleted {result.rowcount} rows")
        
        # Optionally clear assets if table exists
        try:
            result = conn.execute(text("DELETE FROM assets"))
            print(f"Cleared assets table: {result.rowcount} rows")
        except Exception:
            print("Assets table doesn't exist, skipping")
        
        print("\n[OK] Survey data cleared successfully!")
        
        # Verify preservation
        print("\nVerifying preserved tables:")
        
        # Check calibration profiles
        result = conn.execute(text("SELECT COUNT(*) FROM calibration_profiles"))
        print(f"  calibration_profiles: {result.scalar()} rows")
        
        # Check gh_roads
        try:
            result = conn.execute(text("SELECT COUNT(*) FROM gh_roads"))
            print(f"  gh_roads: {result.scalar()} rows")
        except Exception:
            print("  gh_roads: table not found or error")
        
        # Verify cleared tables
        for table in ['projects', 'roads', 'organisations', 'teams', 'survey_runs', 'iri_segments', 'tracked_defects', 'map_matching_runs']:
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                print(f"  {table}: {result.scalar()} rows")
            except Exception as e:
                print(f"  {table}: error - {e}")
        
        print("\nDone!")


if __name__ == "__main__":
    clear_survey_data()