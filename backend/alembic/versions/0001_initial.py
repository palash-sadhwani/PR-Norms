"""initial schema"""

from alembic import op
from app.db.base import Base
from app.models import entities  # noqa: F401

revision = "0001_initial"
down_revision = None


def upgrade():
    Base.metadata.create_all(op.get_bind())


def downgrade():
    Base.metadata.drop_all(op.get_bind())
