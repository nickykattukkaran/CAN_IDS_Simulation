import argparse
import csv
import can
import time  # Import the time module at the top
import pandas as pd
import os
from PIL import Image
import numpy as np
import multiprocessing
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
    
# Define the Swin Transformer Block
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4., dropout=0., attn_dropout=0., drop_path=0.):
        super(SwinTransformerBlock, self).__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_dropout)
        self.drop_path = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, N, C = x.shape

        # Multi-Head Self-Attention
        shortcut = x
        x = self.norm1(x)
        x = x.permute(1, 0, 2)  # Required for MultiheadAttention
        x, _ = self.attn(x, x, x)
        x = x.permute(1, 0, 2)  # Revert to original shape
        x = shortcut + self.drop_path(x)

        # Feedforward Neural Network
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(x)
        return x

# Define the Swin Transformer
class SwinTransformer(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, num_heads, window_size, mlp_ratio=4., depths=[2, 2, 6, 2]):
        super(SwinTransformer, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Stages of Swin Transformer blocks
        self.blocks = nn.ModuleList()
        for depth in depths:
            block = nn.ModuleList([
                SwinTransformerBlock(
                    dim=embed_dim, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio
                ) for _ in range(depth)
            ])
            self.blocks.append(block)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # Shape: (B, C, H/P, W/P)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # Shape: (B, N, C), where N = H*W

        # Add positional encoding
        x = x + self.pos_embed

        # Pass through Swin Transformer blocks
        for stage in self.blocks:
            for block in stage:
                x = block(x)

        x = self.norm(x)
        return x
    

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        #self.fc1 = nn.Linear(128 * 11 * 11, 256)
        self.fc1 = nn.Linear(128 * 12 * 12, 256)  # Adjust based on actual size

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        #x = x.view(-1, 128 * 11 * 11)  # Flatten the tensor
        #print(f"Shape before flattening: {x.shape}")  # Debugging line
        x = x.view(-1, 128 * 12 * 12)  # Automatically determine flatten size
        x = torch.relu(self.fc1(x))
        return x

# Define the hybrid model
class HybridModel(nn.Module):
    def __init__(self, swin_transformer, cnn, num_classes):
        super(HybridModel, self).__init__()
        self.swin_transformer = swin_transformer
        self.cnn = cnn
        self.mlp = nn.Sequential(
            nn.Linear(256 + 96, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        swin_features = self.swin_transformer(x)
        swin_features = swin_features.mean(dim=1)  # Global average pooling
        cnn_features = self.cnn(x)
        features = torch.cat((swin_features, cnn_features), dim=1)  # Concatenate features
        x = self.mlp(features)
        return x
    
def Initialize_Model():
    # Load the saved model
    model = HybridModel(swin_transformer, cnn, num_classes)
    model.load_state_dict(torch.load('hybrid_model3.pth', weights_only=True))
    model.eval()
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
# Define transformations
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

# Load the trained model
swin_transformer = SwinTransformer(
    image_size=96, patch_size=4, embed_dim=96, num_heads=8, window_size=7
)
cnn = CNN()
num_classes = 4

def start_simulation(df, model):
    # print("DF:")
    # print(df)
    # print("df shape: ", df.shape)

    # Calculate the time interval between consecutive rows 
    df['TimeInterval'] = df['TimeInterval'].diff().fillna(0)
    df["TimeInterval"] = (df["TimeInterval"] * 1_000_000).astype(int)

    #df['TimeDiff'] = df['TimeInterval'] - 13
    df['TimeDiff'] = df['TimeInterval']
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
    #print(num_rows, num_cols)
    num_images = num_rows // 94

    for i in range(num_images):
        # Combine 94 consecutive rows
        start_time_img = time.time()
        combined_rows = df.iloc[i*94:(i+1)*94].values.flatten()
        
        # Reshape to 94x94
        binary_image = combined_rows.reshape(94, 1).astype(str)
        # Convert the binary strings to a numpy array of 0s and 1s
        #binary_image_data = np.array([[int(bit) for bit in row[0]] for row in binary_image])
        # Convert the binary strings to a numpy array of 0s and 1s
        binary_image_data = np.array([[int(bit) for bit in row[0].ljust(94, '0')[:94]] for row in binary_image])

        # print(binary_image_data)

        # Convert the numpy array to an image
        image = Image.fromarray(binary_image_data.astype(np.uint8) * 255)  # 0 = black, 1 = white

        # Create the 'temp' folder if it doesn't exist
        #os.makedirs('temp', exist_ok=True)

        # Save the image in the 'temp' folder
        #image.save(os.path.join('temp',f"image_{i}.jpg"))
        end_time_img = time.time()
        time_to_gen_img = (end_time_img - start_time_img)
        print(f"Time required to generate a binary iamge {time_to_gen_img:.3f} s")
        p3 = multiprocessing.Process(target=start_inference(model, image))
        p3.start()
        p3.join()       

def start_inference(model, image):
    start_time_infer = time.time()
    model.eval()
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
    #print(f"The binary image Belongs to Attack Free and the model classified it as: {attack_classes[predicted.item()]}")
    end_time_infer = time.time()
    time_infer = (end_time_infer - start_time_infer)
    print(f"Time required for inference : {time_infer:.3f} s")
    #print(f"The binary image Belongs to Attack Free and the model classified it as: {attack_classes[predicted.item()]}")
    print(f"*************************************The model classified binary image as: {attack_classes[predicted.item()]}")

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
        sum_msg_time = 0 

        # Open the CSV file for writing
        with open(output_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write CSV header
            writer.writerow(['ID','RemoteFrame','DLC','Payload','TimeInterval'])

            # Create an empty DataFrame
            df = pd.DataFrame(columns=['ID', 'RemoteFrame', 'DLC', 'Payload', 'TimeInterval'])
            
            # Start receiving messages
            start_time = time.monotonic()  # Corrected to use time.monotonic()
            while time.monotonic() - start_time < duration:
                message = bus.recv(timeout=1)  # Receive a message with a timeout of 1 second
                if message is not None:
                    i+=1
                    start_time_msg = time.time()
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
                    #print(f"Received {frame_type} Frame: ID={message_id} DLC={dlc} Data={message_data}")
                    end_time_msg = time.time()
                    sum_msg_time += (end_time_msg-start_time_msg)

                    if (i == 94):
                        i=0
                        #print(df)
                        print(f"Time required to receive the 94 messages:{sum_msg_time:.3f} s")
                        sum_msg_time = 0
                        df_temp =df
                        df = df.iloc[0:0]
                        p2 = multiprocessing.Process(target=start_simulation(df_temp, model))
                        p2.start()
                        p2.join()           

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
    testmodel = Initialize_Model()
    # Call the function with parsed arguments
    p1 = multiprocessing.Process(target=receive_can_messages(args.interface, args.output, args.duration, testmodel))
    #Start the process
    p1.start()
    #wait for process to complete
    p1.join()
