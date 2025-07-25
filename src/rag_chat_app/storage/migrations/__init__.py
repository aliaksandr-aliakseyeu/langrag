"""
Custom migration system

At this stage, we manage schema creation manually using raw SQL in `run_migrations.py`.
Each table is created with `CREATE TABLE IF NOT EXISTS` to avoid duplicates.

In the future, if needed, we can implement a proper migration system.
For now, this lightweight approach is sufficient.
"""
