from datetime import datetime

from sqlalchemy import DateTime, Float, Integer
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    influent_bod5: Mapped[float] = mapped_column(Float)
    influent_cod: Mapped[float] = mapped_column(Float)
    influent_tss: Mapped[float] = mapped_column(Float)

    flow_m3_day: Mapped[float] = mapped_column(Float)

    dissolved_oxygen: Mapped[float] = mapped_column(Float)
    temperature: Mapped[float] = mapped_column(Float)
    hrt_hours: Mapped[float] = mapped_column(Float)

    predicted_effluent_bod5: Mapped[float] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )