from pydantic import BaseModel, Field, ConfigDict


class TransactionInput(BaseModel):
    """
    Input schema for a credit card transaction.

    Features:
    Time, Amount, V1 through V28.
    """

    model_config = ConfigDict(extra="forbid")

    Time: float = Field(
        ...,
        ge=0,
        description="Seconds elapsed between this transaction and the first transaction."
    )

    Amount: float = Field(
        ...,
        ge=0,
        description="Transaction amount."
    )

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


class PredictionResponse(BaseModel):
    """Response returned by the fraud detection API."""

    transaction_id: str

    fraud_probability: float = Field(
        ...,
        ge=0,
        le=1
    )

    is_fraud: bool

    decision: str

    model_name: str

    threshold: float = Field(
        ...,
        ge=0,
        le=1
    )