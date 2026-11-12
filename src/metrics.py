"""Метрика якості Q (умова проєкту, розділ 4).

S = B_ref / (B_ref + B_sent)   ефективність передавання
R = частка відновлених критичних кадрів
T = t_ref / (t_ref + t)        швидкодія
Q = alpha*S + beta*R + gamma*T
Якщо не відновлено жодного критичного кадру, Q = 0.
"""


def efficiency_s(b_ref_bits, b_sent_bits):
    """S: чим менше біт передано відносно сирих даних, тим S ближче до 1."""
    total = b_ref_bits + b_sent_bits
    if total == 0:
        return 1.0
    return b_ref_bits / total


def critical_recovery_r(original_frames, restored_frames):
    """R: яка частка критичних кадрів оригіналу є серед відновлених.

    Кадри порівнюємо за їхніми 12 байтами. Якщо критичних кадрів у оригіналі
    немає, втрачати нічого, тож R = 1.
    """
    critical = [f for f in original_frames if f.critical == 1]
    if not critical:
        return 1.0
    restored_bytes = {f.to_bytes() for f in restored_frames if f.critical == 1}
    recovered = sum(1 for f in critical if f.to_bytes() in restored_bytes)
    return recovered / len(critical)


def speed_t(t_ref, t):
    """T: чим менший час прогону t відносно еталонного t_ref, тим T ближче до 1."""
    total = t_ref + t
    if total == 0:
        return 1.0
    return t_ref / total


def quality_q(s, r, t, alpha, beta, gamma):
    """Q = alpha*S + beta*R + gamma*T, а якщо R = 0, то Q = 0."""
    if r == 0:
        return 0.0
    return alpha * s + beta * r + gamma * t