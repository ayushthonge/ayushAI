# src/data_prep.py
import os
import re
import pandas as pd

class DataPrep:
    """Prepares WhatsApp chat data for processing (supports 'dd/mm/yy, hh:mm - Name: Message' format)"""

    def __init__(self, input_dir, output_dir, my_name):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.my_name = my_name
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'by_contact'), exist_ok=True)

    def _parse_line(self, line):
        # Pattern matches: 27/09/22, 23:22 - Ayush: Message here
        pattern = r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})\s-\s([^:]+):\s(.*)$'
        match = re.match(pattern, line)
        if match:
            date_str, time_str, sender, message = match.groups()
            try:
                # Try to parse date and time
                timestamp = pd.to_datetime(f"{date_str} {time_str}", dayfirst=True)
            except Exception:
                timestamp = f"{date_str} {time_str}"
            return {
                'timestamp': timestamp,
                'sender': sender.strip(),
                'message': message.strip()
            }
        return None

    def process_chat_file(self, file_path, contact_name):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        messages = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parsed = self._parse_line(line)
            if parsed:
                parsed['contact'] = contact_name
                messages.append(parsed)
            elif messages:
                # Continuation of previous message (multi-line)
                messages[-1]['message'] += '\n' + line
        if messages:
            return pd.DataFrame(messages)
        return pd.DataFrame()

    def process_all_chats(self):
        all_messages = []
        my_messages = []
        for file in os.listdir(self.input_dir):
            if file.endswith('.txt'):
                contact_name = file.split('.')[0]
                file_path = os.path.join(self.input_dir, file)
                print(f"Processing chat with {contact_name}...")
                df = self.process_chat_file(file_path, contact_name)
                if not df.empty:
                    # Save per-contact file
                    contact_output_dir = os.path.join(self.output_dir, 'by_contact')
                    os.makedirs(contact_output_dir, exist_ok=True)
                    df.to_csv(os.path.join(contact_output_dir, f'{contact_name}.csv'), index=False)
                    # Add my messages for this contact
                    my_df = df[df['sender'] == self.my_name]
                    if not my_df.empty:
                        my_messages.append(my_df)
                    all_messages.append(df)
        # Save combined files
        if all_messages:
            all_df = pd.concat(all_messages, ignore_index=True)
            all_df.to_csv(os.path.join(self.output_dir, 'all_chats.csv'), index=False)
            print(f"Saved {len(all_df)} total messages to all_chats.csv")
        else:
            print("No messages found in any chat files.")
        if my_messages:
            my_df = pd.concat(my_messages, ignore_index=True)
            my_df.to_csv(os.path.join(self.output_dir, 'my_messages.csv'), index=False)
            print(f"Saved {len(my_df)} of your messages to my_messages.csv")
        else:
            print("No messages from you found in any chat files.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process WhatsApp chat exports')
    parser.add_argument('--input', type=str, required=True, help='Directory containing raw chat exports')
    parser.add_argument('--output', type=str, required=True, help='Directory to save processed data')
    parser.add_argument('--name', type=str, required=True, help='Your name as it appears in chats')
    args = parser.parse_args()
    prep = DataPrep(args.input, args.output, args.name)
    prep.process_all_chats()
