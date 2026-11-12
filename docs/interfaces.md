# Інтерфейси модулів

Імпорти без префікса `src.` (pytest.ini: pythonpath = src).

## frame.py
- `Frame(timestamp_ms, altitude_dm, temp_c10, vibration, mode, critical)` - кадр 12 байтів. Невалідні значення дають ValueError.
- `Frame.to_bytes() -> bytes`
- `Frame.from_bytes(data) -> Frame`
- `frames_to_bytes(frames) -> bytes`, `frames_from_bytes(data) -> list[Frame]`

## huffman.py
- `huffman_encode(data: bytes) -> (lengths, bitstream)` - канонічний код Гаффмана, довжина коду до 15.
- `huffman_decode(lengths, bitstream, symbol_count) -> bytes`
- `window_lengths(frames_bytes: list[bytes]) -> dict[int, int]` - одна таблиця довжин кодів на все вікно кандидатів (до 64 кадрів), довжини до 15.
- `frame_weight(frame_bytes: bytes, lengths) -> int` - вага кадру в бітах за цією таблицею (сума довжин кодів його 12 байтів).
- Помилки: `HuffmanError`

## crypto.py
- `derive_keys(password, salt) -> (k_enc, k_mac)` - PBKDF2, 100000 ітерацій, кеш на сесію.
- `aes_ctr(k_enc, nonce, data) -> bytes` - шифрування і розшифрування.
- `make_tag(k_mac, data) -> bytes` - HMAC-SHA256, 32 байти.
- `verify_tag(k_mac, data, tag) -> bool`

## packet.py
- `build_packet(frames, password, device_id, packet_seq, salt, nonce, lengths=None) -> bytes` - якщо передано `lengths`, кадри кодуються саме цією таблицею (таблиця вікна), інакше таблиця будується за кадрами пакета.
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
- `frame_value(cls, critical, config=None) -> int` - цінність кадру: пріоритет за класом (0 норма, 1 аномалія, 2 перехідний) плюс бонус за critical. Числа з team_config.json.
- `schedule(groups) -> list[list[int]]` - групи (індекси кадрів, сумарна цінність) у порядку відправки, найцінніші першими. packet_seq призначається в цьому порядку.
- `make_packets(frames, config, salt, session_prefix)` - жадібне пакування (базова лінія для порівняння).
- `plan_windows(frames, config, classes=None)` - ділить кадри на вікна по 64, для кожного розв'язує рюкзак, повертає групи (індекси, цінність, таблиця довжин). Без `classes` усі кадри вважаються класом 0.
- `make_packets_knapsack(frames, config, salt, session_prefix, classes=None)` - один пакет на вікно, пакети впорядковані через `schedule`.
- `run_scenario(frames, config, name, packer="knapsack", classes=None)` - повний прогін одного сценарію, повертає S, R, T, Q. `packer="greedy"` вмикає старе пакування.
- Запуск: `python src/run_session.py`

## knapsack.py
- `capacity_bits(max_payload_bytes, table_nsym) -> int` - місткість рюкзака W = 8 * (max_payload_bytes - 14 - 2 * table_nsym).
- `knapsack_dp(weights, values, capacity) -> list[int]` - рюкзак 0/1, ДП O(n*W). Повертає індекси обраних кадрів за зростанням.
- `knapsack_greedy(weights, values, capacity) -> list[int]` - жадібний за v/w.

## classifier.py
- `load_labeled(path) -> (frames, labels)` - CSV з колонкою label.
- `train_classifier(frames, labels, seeds=(0, 1, 2)) -> model` - власний MLP, ділить 80/20, обирає seed за macro-F1 на валідації.
- `predict_classes(model, frames) -> list[int]` - клас кожного кадру: 0 норма, 1 аномалія, 2 перехідний. Ознаки з вікна історії.
- `macro_f1(y_true, y_pred) -> float`, `confusion_matrix(y_true, y_pred)`

## channel.py (від викладача, не змінювати)
- `Channel(scenario, seed)`, `send(bytes) -> (delivered, arrival_time)`, `stats()`.