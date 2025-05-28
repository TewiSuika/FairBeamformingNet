"""
调制解调模块
"""
import numpy as np

def qam16_modulate(bits):
    """将4个bit映射为16QAM符号"""
    assert len(bits) % 4 == 0
    symbols = []
    for i in range(0, len(bits), 4):
        b = bits[i:i + 4]
        # 格雷码映射
        x = (-3 if (b[0], b[1]) == (0, 0) else
             -1 if (b[0], b[1]) == (0, 1) else
             1 if (b[0], b[1]) == (1, 1) else
             3)
        y = (-3 if (b[2], b[3]) == (0, 0) else
             -1 if (b[2], b[3]) == (0, 1) else
             1 if (b[2], b[3]) == (1, 1) else
             3)
        symbols.append((x + 1j * y) / np.sqrt(10))  # 功率归一化
    return np.array(symbols)


def qam16_demodulate(symbols):
    """16QAM解调"""
    bits = []
    for s in symbols:
        # 判决区域
        x = np.real(s) * np.sqrt(10)
        y = np.imag(s) * np.sqrt(10)

        # I路
        if x < -2:
            bits += [0, 0]
        elif x < 0:
            bits += [0, 1]
        elif x < 2:
            bits += [1, 1]
        else:
            bits += [1, 0]

        # Q路
        if y < -2:
            bits += [0, 0]
        elif y < 0:
            bits += [0, 1]
        elif y < 2:
            bits += [1, 1]
        else:
            bits += [1, 0]
    return bits