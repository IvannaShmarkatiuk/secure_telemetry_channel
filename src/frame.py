import struct
from dataclasses import dataclass

@dataclass
class Frame:
    timestamp_ms: int
    altitude_dm: int
    temp_c10: int
    vibration: int
    mode: int
    critical: int

    def to_bytes(self) -> bytes:
        return struct.pack(">IhhhBB",
                       self.timestamp_ms,
                       self.altitude_dm,
                       self.temp_c10,
                       self.vibration,
                       self.mode,
                       self.critical
                       ) 
    @classmethod
    def from_bytes(cls, data: bytes) -> "Frame":
        if len(data) != 12:
            raise ValueError(f"кадр має бути 12 байтів, а не {len(data)}")
        values = struct.unpack(">IhhhBB", data)   #повернення кортежу з 6 елементів
        return cls(*values) #розгортає цей кортеж у 6 аргументів     