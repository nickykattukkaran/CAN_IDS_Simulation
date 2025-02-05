import argparse
import csv
import can
import time  # Import the time module at the top
import pandas as pd
import os
from PIL import Image
import numpy as np
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

attack_classes = ['Attack_free', 'Dos_Attack', 'Fuzzy_Attack', 'Impersonate_Attack']

# Load CSV files
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 11 * 11, 256)
        self.fc2 = nn.Linear(256, 4)  # 4 classes: Attack_free, Dos_Attack, Fuzzy_Attack, Impersonate_Attack

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 128 * 11 * 11)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    
def Initialize_Model():
    # Load the saved model
    model = CNN()
    model.load_state_dict(torch.load('model.pth', weights_only=True))
    return model

# Function to classify a single binary image
def classify_image(image_path):
    image = Image.open(image_path).convert('L')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

# Define transformation for the input image
transform = transforms.Compose([
    transforms.Resize((94, 94)),
    transforms.ToTensor()
])

def start_simulation(df, model):
    # file = r"receive_message1.csv"
    # df = load_data(file)

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

    #df.to_csv('output1.csv', index=False) 
    #print("DataFrame exported successfully to output1.csv")

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

    #df.to_csv('output2.csv', index=False)
    
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
            
        model.eval()
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
        #return predicted.item()
        print(f"The binary image Belongs to Attack Free and the model classified it as: {attack_classes[predicted.item()]}")

    print(f'The Images are successfully Stored in folder temp')



def receive_can_messages(interface, output_csv, duration, model):
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
        i=0

        # Open the CSV file for writing
        with open(output_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write CSV header
            writer.writerow(['ID','RemoteFrame','DLC','Payload','TimeInterval'])

            # Create an empty DataFrame
            df = pd.DataFrame(columns=['ID', 'RemoteFrame', 'DLC', 'Payload', 'TimeInterval'])
            
            start_time_image = time.time()
            # Start receiving messages
            start_time = time.monotonic()  # Corrected to use time.monotonic()
            while time.monotonic() - start_time < duration:
                message = bus.recv(timeout=1)  # Receive a message with a timeout of 1 second
                if message is not None:
                    i+=1
                    # Extract message details
                    message_id = hex(message.arbitration_id)
                    dlc = message.dlc  # Data Length Code
                    message_data = message.data.hex() if not message.is_remote_frame else ''
                    message_timestamp = message.timestamp
                    is_rtr = message.is_remote_frame

                    # Log to CSV
                    #writer.writerow([message_id, dlc, message_data, message_timestamp, int(is_rtr)])
                    writer.writerow([message_id, int(is_rtr), dlc, message_data, message_timestamp])

                  # Store in DataFrame
                    new_row = pd.DataFrame([{
                        'ID': message_id,
                        'RemoteFrame': int(is_rtr),
                        'DLC': dlc,
                        'Payload': message_data,
                        'TimeInterval': message_timestamp
                    }])
                    df = pd.concat([df, new_row], ignore_index=True)

                    frame_type = "RTR" if is_rtr else "DATA"
                    print(f"Received {frame_type} Frame: ID={message_id} DLC={dlc} Data={message_data}")

                    if (i == 94):
                        i=0
                        print(df)
                        end_time_image = time.time()
                        time_diff = (end_time_image-start_time_image) * 1000 #in ms
                        print(f"Time required to Generate an Image:{time_diff:.3f} ms")
                        start_time_image = time.time()
                        #start_simulation(df)
                        threading.Thread(target=start_simulation(df, model)).start()

            

    except can.CanError as e:
        print(f"CAN Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Shutting down...")
        bus.shutdown()
        #start_simulation()


if __name__ == "__main__":
    # Argument parser for command-line inputs
    parser = argparse.ArgumentParser(description="Receive CAN messages and store them in a CSV file.")
    parser.add_argument('-i', '--interface', required=True, help="CAN interface (e.g., 'can0', 'vcan0').")
    parser.add_argument('-o', '--output', required=True, help="Path to the output CSV file.")
    parser.add_argument('-d', '--duration', type=int, default=10, help="Duration to listen for messages (in seconds).")

    args = parser.parse_args()
    testmodel = Initialize_Model()
    # Call the function with parsed arguments
    #receive_can_messages(args.interface, args.output, args.duration)
    threading.Thread(target=receive_can_messages(args.interface, args.output, args.duration, testmodel)).start()