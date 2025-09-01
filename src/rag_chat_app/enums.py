from enum import Enum


class VectorStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

    @classmethod
    def choices(cls):
        return [status.value for status in cls]

    @classmethod
    def default(cls):
        return cls.PENDING

    @classmethod
    def from_string(cls, value: str):
        try:
            return cls(value)
        except ValueError:
            print(f"Warning: Invalid vector_status '{value}', using default")
            return cls.default()
