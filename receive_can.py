'''
import argparse
import csv
import can
import time  # Import the time module at the top


def receive_can_messages(interface, output_csv, duration):
    """
    Receives CAN messages and stores them in a CSV file.

    Args:
        interface (str): CAN interface (e.g., 'can0', 'vcan0').
        output_csv (str): Path to the output CSV file.
        duration (int): Duration to listen for messages (in seconds).
    """
    print(f"Setting up CAN interface '{interface}' for receiving...")
    try:
        # Set up CAN bus
        bus = can.Bus(channel=interface, interface='socketcan')

        print(f"Listening for CAN messages on '{interface}' for {duration} seconds...")

        # Open the CSV file for writing
        with open(output_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write CSV header
            writer.writerow(['ID', 'DLC', 'Data', 'Timestamp', 'RTR'])

            # Start receiving messages
            start_time = time.monotonic()  # Corrected to use time.monotonic()
            while time.monotonic() - start_time < duration:
                message = bus.recv(timeout=1)  # Receive a message with a timeout of 1 second
                if message is not None:
                    # Extract message details
                    message_id = hex(message.arbitration_id)
                    dlc = message.dlc  # Data Length Code
                    message_data = message.data.hex() if not message.is_remote_frame else ''
                    message_timestamp = message.timestamp
                    is_rtr = message.is_remote_frame

                    # Log to CSV
                    writer.writerow([message_id, dlc, message_data, message_timestamp, int(is_rtr)])
                    frame_type = "RTR" if is_rtr else "DATA"
                    print(f"Received {frame_type} Frame: ID={message_id} DLC={dlc} Data={message_data}")

    except can.CanError as e:
        print(f"CAN Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Shutting down...")
        bus.shutdown()


if __name__ == "__main__":
    # Argument parser for command-line inputs
    parser = argparse.ArgumentParser(description="Receive CAN messages and store them in a CSV file.")
    parser.add_argument('-i', '--interface', required=True, help="CAN interface (e.g., 'can0', 'vcan0').")
    parser.add_argument('-o', '--output', required=True, help="Path to the output CSV file.")
    parser.add_argument('-d', '--duration', type=int, default=10, help="Duration to listen for messages (in seconds).")

    args = parser.parse_args()

    # Call the function with parsed arguments
    receive_can_messages(args.interface, args.output, args.duration)
'''

import argparse
import csv
import can
import time  # Import the time module at the top
import pandas as pd
import os
from PIL import Image
import numpy as np

# Load CSV files
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def start_simulation():
    file = r"receive_message1.csv"
    df = load_data(file)

    # Calculate the time interval between consecutive rows 
    df['TimeInterval'] = df['TimeInterval'].diff().fillna(0)
    df["TimeInterval"] = (df["TimeInterval"] * 1_000_000).astype(int)

    df['TimeDiff'] = df['TimeInterval'] - 13
    df.loc[0, "TimeDiff"] = 0
    # Drop column TimeInterval
    df.drop(columns='TimeInterval', inplace=True)

    # Ensure 'Payload' is treated as a string
    df['Payload'] = df['Payload'].fillna('').astype(str)

    # Condition 1: If RemoteFrame is 1, set Payload to 'ffffffffffffffff'
    df.loc[df['RemoteFrame'] == 1, 'Payload'] = 'ffffffffffffffff'

    # Condition 2: If RemoteFrame is 0 and DLC < 8, pad Payload with 'ff' to match DLC of 8
    df.loc[(df['RemoteFrame'] == 0) & (df['DLC'] < 8), 'Payload'] = (
        df.loc[(df['RemoteFrame'] == 0) & (df['DLC'] < 8), 'Payload']
        .apply(lambda x: (x + 'ff' * (8 - len(bytes.fromhex(x))))[:16])
    )

    df.to_csv('output1.csv', index=False) 
    print("DataFrame exported successfully to output1.csv")

    # Convert DLC to binary
    df['DLC'] = df['DLC'].apply(lambda x: format(x, '04b'))  # Pad to 8 bits

    # Remove the first character from the 'ID' column values 
    df['ID'] = df['ID'].apply(lambda x: x[2:] if len(x) > 1 else x)
    # Pad ID column with leading zeros
    df['ID'] = df['ID'].apply(lambda x: x.zfill(3) if len(x) < 3 else x)
    # Convert the ID column to binary encoding 
    df['ID'] = df['ID'].apply(lambda x: format(int(x, 16), '012b'))
    #Convert the TimeDiff to binary
    df['TimeDiff'] = df['TimeDiff'].apply(lambda x: format(x, '013b'))  
    # Convert the Payload to binary
    df['Payload'] = df['Payload'].apply(lambda x: ''.join(format(int(c, 16), '04b') for c in x))

    df = df.astype(str)
    #print(df.dtypes)

    df['combined_output'] = df['ID']+df['RemoteFrame'] + df['DLC']+df['Payload'] + df['TimeDiff']

    df = df.drop(columns=['ID','RemoteFrame','DLC','Payload','TimeDiff'])

    df.to_csv('output2.csv', index=False)
    
    #Convert to Images
    num_rows, num_cols = df.shape
    print(num_rows, num_cols)
    num_images = num_rows // 94

    for i in range(num_images):
        # Combine 94 consecutive rows
        combined_rows = df.iloc[i*94:(i+1)*94].values.flatten()
        
        # Reshape to 94x94
        binary_image = combined_rows.reshape(94, 1).astype(str)
        # Convert the binary strings to a numpy array of 0s and 1s
        binary_image_data = np.array([[int(bit) for bit in row[0]] for row in binary_image])
        # print(binary_image_data)

        # Convert the numpy array to an image
        image = Image.fromarray(binary_image_data.astype(np.uint8) * 255)  # 0 = black, 1 = white
        #image.save(os.path.join(f'{root_folder}/{output_folder}/train', f"{output_folder}_{i}.jpg"))
        # Create the 'temp' folder if it doesn't exist
        os.makedirs('temp', exist_ok=True)

        # Save the image in the 'temp' folder
        image.save(os.path.join('temp',f"image_{i}.jpg"))

    print(f'The Images are successfully Stored in folder temp')



def receive_can_messages(interface, output_csv, duration):
    """
    Receives CAN messages and stores them in a CSV file.

    Args:
        interface (str): CAN interface (e.g., 'can0', 'vcan0').
        output_csv (str): Path to the output CSV file.
        duration (int): Duration to listen for messages (in seconds).
    """
    print(f"Setting up CAN interface '{interface}' for receiving...")
    try:
        # Set up CAN bus
        bus = can.Bus(channel=interface, interface='socketcan')

        print(f"Listening for CAN messages on '{interface}' for {duration} seconds...")

        # Open the CSV file for writing
        with open(output_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write CSV header
            writer.writerow(['ID','RemoteFrame','DLC','Payload','TimeInterval'])

            # Start receiving messages
            start_time = time.monotonic()  # Corrected to use time.monotonic()
            while time.monotonic() - start_time < duration:
                message = bus.recv(timeout=1)  # Receive a message with a timeout of 1 second
                if message is not None:
                    # Extract message details
                    message_id = hex(message.arbitration_id)
                    dlc = message.dlc  # Data Length Code
                    message_data = message.data.hex() if not message.is_remote_frame else ''
                    message_timestamp = message.timestamp
                    is_rtr = message.is_remote_frame

                    # Log to CSV
                    #writer.writerow([message_id, dlc, message_data, message_timestamp, int(is_rtr)])
                    writer.writerow([message_id, int(is_rtr), dlc, message_data, message_timestamp])
                    frame_type = "RTR" if is_rtr else "DATA"
                    print(f"Received {frame_type} Frame: ID={message_id} DLC={dlc} Data={message_data}")

    except can.CanError as e:
        print(f"CAN Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Shutting down...")
        bus.shutdown()
        start_simulation()


if __name__ == "__main__":
    # Argument parser for command-line inputs
    parser = argparse.ArgumentParser(description="Receive CAN messages and store them in a CSV file.")
    parser.add_argument('-i', '--interface', required=True, help="CAN interface (e.g., 'can0', 'vcan0').")
    parser.add_argument('-o', '--output', required=True, help="Path to the output CSV file.")
    parser.add_argument('-d', '--duration', type=int, default=10, help="Duration to listen for messages (in seconds).")

    args = parser.parse_args()

    # Call the function with parsed arguments
    receive_can_messages(args.interface, args.output, args.duration)