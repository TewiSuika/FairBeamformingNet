import numpy as np


def calculate_3db_beamwidth(angles, gains):
    peak_idx = np.argmax(gains)
    peak_power = 10 ** (gains[peak_idx] / 10)
    half_power = peak_power / 2

    left_idx = peak_idx
    while left_idx > 0 and 10 ** (gains[left_idx] / 10) > half_power:
        left_idx -= 1

    right_idx = peak_idx
    while right_idx < len(gains) - 1 and 10 ** (gains[right_idx] / 10) > half_power:
        right_idx += 1

    return angles[right_idx] - angles[left_idx]


# ====================== Data Printing Functions ======================

def print_beam_pattern_data(theta_range, pattern, method_name):
    """Print detailed beam pattern data for analysis"""
    print(f"\n=== Beam Pattern Data for {method_name} ===")
    print(f"{'Angle(deg)':<12} {'Gain(dB)':<10} {'Normalized Gain':<15}")

    # Print every 15 degrees
    for angle, gain_db in zip(theta_range, pattern):
        if angle % 15 == 0:
            print(f"{angle:<12.1f} {gain_db:<10.2f} {(10 ** (gain_db / 20)):<15.4f}")

    # Calculate and print key metrics
    max_gain = np.max(pattern)
    max_angle = theta_range[np.argmax(pattern)]
    half_power_bw = calculate_3db_beamwidth(theta_range, pattern)

    print("\nKey Performance Indicators:")
    print(f"- Maximum gain: {max_gain:.2f} dB at {max_angle:.1f}°")
    print(f"- 3-dB beamwidth: {half_power_bw:.1f} degrees")
    print(f"- Sidelobe suppression: {np.max(pattern[pattern < max_gain - 3]):.2f} dB")


def print_sum_rate_data(methods, sum_rates, snr_dBs):
    """Print sum rate data for all methods"""
    print("\n=== Sum Rate Data (bps/Hz) ===")
    header = "SNR(dB)" + "".join([f"{method:>15}" for method in methods])
    print(header)

    for snr_idx, snr in enumerate(snr_dBs):
        row = f"{snr:>7}"
        for method in methods:
            row += f"{sum_rates[method][snr_idx]:>15.4f}"
        print(row)


def print_user_rate_data(user_rates, methods, snr_dBs):
    """Print individual user rate data"""
    print("\n=== Individual User Rate Data ===")
    users = sorted(user_rates["Deep Learning"].keys(), key=lambda x: int(x.split()[1]))

    for user in users:
        print(f"\n{user} Rates (bps/Hz):")
        header = "SNR(dB)" + "".join([f"{method:>15}" for method in methods])
        print(header)

        for snr_idx, snr in enumerate(snr_dBs):
            row = f"{snr:>7}"
            for method in methods:
                row += f"{user_rates[method][user][snr_idx]:>15.4f}"
            print(row)


def print_ber_data(ber_results, methods, snr_dBs):
    """Print BER data for all methods"""
    print("\n=== Bit Error Rate Data ===")
    header = "SNR(dB)" + "".join([f"{method:>15}" for method in methods])
    print(header)

    for snr_idx, snr in enumerate(snr_dBs):
        row = f"{snr:>7}"
        for method in methods:
            row += f"{ber_results[method][snr_idx]:>15.4e}"
        print(row)


def print_crlb_data(crlb_results, methods, snr_dBs):
    """Print CRLB data for all methods"""
    print("\n=== CRLB Data (degrees) ===")
    header = "SNR(dB)" + "".join([f"{method:>15}" for method in methods])
    print(header)

    for snr_idx, snr in enumerate(snr_dBs):
        row = f"{snr:>7}"
        for method in methods:
            row += f"{crlb_results[method][snr_idx]:>15.4f}"
        print(row)


def print_efficiency_data(ee_results, methods, snr_dBs):
    """Print energy efficiency data"""
    print("\n=== Energy Efficiency Data (bps/Hz/W) ===")
    header = "SNR(dB)" + "".join([f"{method:>15}" for method in methods])
    print(header)

    for snr_idx, snr in enumerate(snr_dBs):
        row = f"{snr:>7}"
        for method in methods:
            row += f"{ee_results[method][snr_idx]:>15.4f}"
        print(row)


def print_power_data(power_consumption, methods):
    """Print power consumption data"""
    print("\n=== Power Consumption Data ===")
    print("{:<25} {:<15}".format("Method", "Power (W)"))
    for method, power in zip(methods, power_consumption):
        print("{:<25} {:<15.4f}".format(method, power))