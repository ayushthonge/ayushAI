# src/process_chats.py
import os
import argparse
from data_prep import DataPrep
from config import Config

def main():
    """Process WhatsApp chat exports"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process WhatsApp chat exports')
    parser.add_argument('--input', type=str, help='Directory containing raw chat exports')
    parser.add_argument('--output', type=str, help='Directory to save processed data')
    parser.add_argument('--name', type=str, help='Your name as it appears in chats')
    args = parser.parse_args()
    
    # Load configuration
    config = Config().config
    
    # Use command line arguments if provided, otherwise use config
    input_dir = args.input or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    output_dir = args.output or config.get("processed_dir")
    my_name = args.name or config.get("my_name")
    
    # Create DataPrep instance
    data_prep = DataPrep(input_dir, output_dir, my_name)
    
    # Process all chats
    print(f"Processing WhatsApp chats from {input_dir}...")
    data_prep.process_all_chats()
    print("Processing complete!")

if __name__ == "__main__":
    main()
