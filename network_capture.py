import scapy.all as scapy

def capture_packets(interface, duration):
    print("Capturing network traffic...")
    packets = scapy.sniff(iface=interface, timeout=duration)
    print(f"Captured {len(packets)} packets.")
    return packets

if __name__ == "__main__":
    # Specify the Wi-Fi interface
    interface = 'Wi-Fi'
    capture_duration = 10  # Capture for 10 seconds
    captured_packets = capture_packets(interface, capture_duration)
    
    # Optionally save captured packets to a file or process them
    captured_packets.summary()  # Print a summary of captured packets
