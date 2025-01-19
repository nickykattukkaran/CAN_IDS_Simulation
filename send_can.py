import argparse
import csv
import time
import can


def send_can_messages(interface, csv_file, bitrate):
    """
    Sends CAN messages from a CSV file using python-can.

    Args:
        interface (str): CAN interface (e.g., 'can0', 'vcan0').
        csv_file (str): Path to the CSV file containing CAN messages.
        bitrate (int): Bitrate for the CAN interface (e.g., 500000).
    """
    print(f"Setting up CAN interface '{interface}' with bitrate {bitrate}...")
    try:
        # Set up CAN bus
        bus = can.Bus(channel=interface, interface='socketcan')

        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)

            for row in reader:
                # Parse ID, Data, Delay, RTR, and DLC
                can_id = int(row['ID'], 16)  # Convert Hex ID to integer
                data = bytes.fromhex(row['Data']) if row['Data'] else b''  # Convert Hex Data to bytes or empty
                dlc = int(row.get('DLC', len(data)))  # Default to length of the data if DLC is not specified
                delay = float(row['Delay'])  # Delay in seconds
                rtr = row.get('RTR', '0') == '1'  # Check if RTR flag is set

                # Pad or truncate data to match DLC
                if not rtr:  # For non-RTR frames
                    if len(data) < dlc:
                        data = data.ljust(dlc, b'\x00')  # Pad with zeroes
                    elif len(data) > dlc:
                        data = data[:dlc]  # Truncate to DLC

                # Create a CAN message
                message = can.Message(
                    arbitration_id=can_id,
                    data=data,
                    is_extended_id=False,
                    is_remote_frame=rtr
                )

                # Send the message
                bus.send(message)
                frame_type = "RTR" if rtr else "DATA"
                print(f"Sent {frame_type} Frame: ID={hex(can_id)} Data={data.hex()} DLC={dlc}")

                # Wait for the specified delay
                time.sleep(delay)

    except can.CanError as e:
        print(f"CAN Error: {e}")
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Shutting down...")
        bus.shutdown()


if __name__ == "__main__":
    # Argument parser for command-line inputs
    parser = argparse.ArgumentParser(description="Send CAN messages from a CSV file using python-can.")
    parser.add_argument('-i', '--interface', required=True, help="CAN interface (e.g., 'can0', 'vcan0').")
    parser.add_argument('-c', '--csv', required=True, help="Path to the CSV file containing CAN messages.")
    parser.add_argument('-b', '--bitrate', type=int, default=500000, help="CAN bus bitrate (default: 500000).")

    args = parser.parse_args()

    # Call the function with parsed arguments
    send_can_messages(args.interface, args.csv, args.bitrate)
