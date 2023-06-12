from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, TIMESTAMP, BIGINT
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from query_db import get_sqlalchemy_engine

engine = get_sqlalchemy_engine()

Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

Base.metadata.create_all(engine)

if not engine.dialect.has_table(engine, "mc4_"):
    pass

class MC4_(Base):
    __tablename__ = 'mc4_'
    m4id = Column(BigInteger(unsigned=True), primary_key=True, autoincrement=True)
    aeid = Column(BigInteger(unsigned=True), nullable=False)
    spid = Column(String(50, collation='utf8mb4_unicode_ci'), default=None)
    bmad = Column(Float, nullable=False)
    resp_max = Column(Float, nullable=False)
    resp_min = Column(Float, nullable=False)
    max_mean = Column(Float, nullable=False)
    max_mean_conc = Column(Float, nullable=False)
    max_med = Column(Float, nullable=False)
    max_med_conc = Column(Float, nullable=False)
    logc_max = Column(Float, nullable=False)
    logc_min = Column(Float, nullable=False)
    nconc = Column(Integer, nullable=False)
    npts = Column(Integer, nullable=False)
    nrep = Column(Float, nullable=False)
    nmed_gtbl = Column(Integer, nullable=False)
    tmpi = Column(Integer, nullable=False)
    created_date = Column(TIMESTAMP, nullable=False, server_default='CURRENT_TIMESTAMP')
    modified_date = Column(TIMESTAMP, nullable=False, server_default='CURRENT_TIMESTAMP')
    modified_by = Column(String(100, collation='utf8mb4_unicode_ci'), default=None)



from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

class Base(DeclarativeBase):
    pass

class MC4_(Base):
    __tablename__ = "mc4_"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30))
    fullname: Mapped[Optional[str]]
    addresses: Mapped[List["Address"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    def __repr__(self) -> str:
        return f"User(id={self.id!r}, name={self.name!r}, fullname={self.fullname!r})"

class Address(Base):
    __tablename__ = "address"
    id: Mapped[int] = mapped_column(primary_key=True)
    email_address: Mapped[str]
    user_id: Mapped[int] = mapped_column(ForeignKey("user_account.id"))
    user: Mapped["User"] = relationship(back_populates="addresses")
    def __repr__(self) -> str:
        return f"Address(id={self.id!r}, email_address={self.email_address!r})"



