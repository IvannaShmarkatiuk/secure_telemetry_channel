# Інтерфейси модулів

Імпорти без префікса `src.` (pytest.ini: pythonpath = src).

## frame.py
- `Frame(timestamp_ms, altitude_dm, temp_c10, vibration, mode, critical)` - кадр 12 байтів. Невалідні значення дають ValueError.
- `Frame.to_bytes() -> bytes`
- `Frame.from_bytes(data) -> Frame`
- `frames_to_bytes(frames) -> bytes`, `frames_from_bytes(data) -> list[Frame]`

## huffman.py
- `huffman_encode(data: bytes) -> (lengths, bitstream)` - канонічний код Гаффмана, довжина коду до 15.
- `huffman_decode(lengths, bitstream, ...) -> bytes`
- Помилки: `HuffmanError`

## crypto.py
- `derive_keys(password, salt) -> (k_enc, k_mac)` - PBKDF2, 100000 ітерацій, кеш на сесію.
- `aes_ctr(k_enc, nonce, data) -> bytes` - шифрування і розшифрування.
- `make_tag(k_mac, data) -> bytes` - HMAC-SHA256, 32 байти.
- `verify_tag(k_mac, data, tag) -> bool`

## packet.py
- `build_packet(frames, password, device_id, packet_seq, salt, nonce) -> bytes`
- `parse_packet(packet, password) -> list[Frame]`
- `PacketError(code)` - коди помилок приймача: E_SHORT, E_MAGIC, E_LENGTH, E_HMAC, E_PAYLOAD, E_CODE.
- Порядок перевірок приймача: довжина, magic/version, довжина за заголовком, HMAC (до розшифрування), розшифрування, розбір payload.

## metrics.py
- `efficiency_s(b_ref_bits, b_sent_bits)`
- `critical_recovery_r(original_frames, restored_frames)`
- `speed_t(t_ref, t)`
- `quality_q(s, r, t, alpha, beta, gamma)` - 0, якщо r == 0.

## run_session.py
- `load_config()`, `load_frames()` - конфіг і 3000 кадрів сесії.
- `make_packets(frames, config, salt, session_prefix)` - жадібне пакування до max_packet_bytes.
- `run_scenario(frames, config, name)` - повний прогін одного сценарію, повертає S, R, T, Q.
- Запуск: `python src/run_session.py`

## channel.py (від викладача, не змінювати)
- `Channel(scenario, seed)`, `send(bytes) -> (delivered, arrival_time)`, `stats()`.