import struct
 
from frame import Frame
from huffman import HuffmanError, huffman_decode, huffman_encode
 
MAGIC = b"TLM1"
VERSION = 1
HEADER_LEN = 43
TAG_LEN = 32
MIN_PACKET_LEN = HEADER_LEN + TAG_LEN  # 75 = заголовок + порожній шифротекст + HMAC
 
# Заголовок (розділ 2.1). ">" означає big-endian без вирівнювання.
# 4s magic, B version, H device_id, I packet_seq, 16s salt, 12s nonce, I ciphertext_len
HEADER_FORMAT = ">4sBHI16s12sI"
 
# Навантаження (розділ 3): початок і середина, між ними таблиця довжин кодів.
# H frame_count, B flags, B reserved, H table_nsym
PAYLOAD_START_FORMAT = ">HBBH"
# I symbol_count, I bitstream_len
PAYLOAD_COUNTS_FORMAT = ">II"
FLAG_NO_COMPRESSION = 0x08
 
 
class PacketError(Exception):
    """Відмова приймача; .code — один із кодів розділу 6 FORMAT_SPEC."""
 
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code
 
 
def build_header(device_id: int, packet_seq: int, salt: bytes, nonce: bytes,
                 ciphertext_len: int) -> bytes:
    """Збирає відкритий заголовок, 43 байти (розділ 2.1)."""
    return struct.pack(HEADER_FORMAT, MAGIC, VERSION, device_id, packet_seq, salt, nonce,
                       ciphertext_len)
 
 
def build_payload(frames: list[Frame]) -> bytes:
    """Збирає навантаження з кадрів (розділ 3). Завжди flags = 0, тобто Гаффман."""
    data = b"".join(frame.to_bytes() for frame in frames)
    lengths, bitstream = huffman_encode(data)
    # Таблиця: лише символи з ненульовою довжиною, за зростанням символу (розділ 3.2)
    symbols = sorted(s for s in lengths if lengths[s] > 0)
    table = b"".join(struct.pack(">BB", s, lengths[s]) for s in symbols)
    start = struct.pack(PAYLOAD_START_FORMAT, len(frames), 0, 0, len(symbols))
    counts = struct.pack(PAYLOAD_COUNTS_FORMAT, len(data), len(bitstream))
    return start + table + counts + bitstream
 
 
def parse_payload(payload: bytes) -> list[Frame]:
    """Розбирає розшифроване навантаження назад у кадри (розділ 3)."""
    if len(payload) < 6:
        raise PacketError("E_PAYLOAD")
    frame_count, flags, reserved, table_nsym = struct.unpack(PAYLOAD_START_FORMAT, payload[:6])
    if reserved != 0 or flags not in (0, FLAG_NO_COMPRESSION):
        raise PacketError("E_PAYLOAD")
 
    table_end = 6 + 2 * table_nsym
    if len(payload) < table_end + 8:
        raise PacketError("E_PAYLOAD")
    lengths = {payload[i]: payload[i + 1] for i in range(6, table_end, 2)}
    symbol_count, bitstream_len = struct.unpack(PAYLOAD_COUNTS_FORMAT, payload[table_end:table_end + 8])
    if symbol_count != 12 * frame_count:
        raise PacketError("E_PAYLOAD")
 
    bitstream_start = table_end + 8
    if len(payload) != bitstream_start + bitstream_len:
        raise PacketError("E_PAYLOAD")
    bitstream = payload[bitstream_start:]
 
    if flags == FLAG_NO_COMPRESSION:
        # Розділ 3.1: таблиці немає, у потоці рівно symbol_count сирих байтів
        if table_nsym != 0 or bitstream_len != symbol_count:
            raise PacketError("E_PAYLOAD")
        data = bitstream
    else:
        try:
            data = huffman_decode(lengths, bitstream, symbol_count)
        except HuffmanError:
            raise PacketError("E_CODE")
 
    try:
        return [Frame.from_bytes(data[i:i + 12]) for i in range(0, len(data), 12)]
    except ValueError:
        raise PacketError("E_PAYLOAD")  # тимчасово: код для поганого кадру ще не погоджено
 
 
def parse_packet(packet: bytes, password: str) -> list["Frame"]:
    # Крок 1: пакет не може бути коротшим за заголовок + HMAC
    if len(packet) < MIN_PACKET_LEN:
        raise PacketError("E_SHORT")
 
    # Крок 2: відкритий заголовок
    magic, version, device_id, packet_seq, salt, nonce, ciphertext_len = struct.unpack(
        HEADER_FORMAT, packet[:HEADER_LEN]
    )
    if magic != MAGIC or version != VERSION:
        raise PacketError("E_MAGIC")
 
    # Крок 3: фактична довжина має збігатися з тією, що записана в заголовку
    if len(packet) != HEADER_LEN + ciphertext_len + TAG_LEN:
        raise PacketError("E_LENGTH")
 
    raise NotImplementedError("кроки 4-8 (HMAC, розшифрування, навантаження) ще не написані")
 