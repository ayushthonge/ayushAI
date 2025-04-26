# src/clean_data.py
import os
import pandas as pd
import re
import json

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

class DataCleaner:
    def __init__(self, config):
        # Use get() for robust fallback
        self.processed_dir = config.get(
            'processed_dir', 
            os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
        )

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'https?://\S+', '[URL]', text)
        if text.strip() == '(Media omitted)':
            return '[MEDIA]'
        text = re.sub(r'[^\w\s\.,;:!?()[\]{}\'\"\\/@#$%^&*+=~`|<>\-]', '', text)
        text = ' '.join(text.split())
        return text

    def clean_dataframe(self, df):
        df_clean = df.copy()
        df_clean['message'] = df_clean['message'].apply(self.clean_text)
        df_clean = df_clean[df_clean['message'].str.strip() != '']
        if 'timestamp' in df_clean.columns:
            df_clean = df_clean.sort_values('timestamp')
        return df_clean

    def clean_all_files(self):
        for filename in ['all_chats.csv', 'my_messages.csv']:
            file_path = os.path.join(self.processed_dir, filename)
            if os.path.exists(file_path):
                print(f"Cleaning {filename}...")
                df = pd.read_csv(file_path)
                df_clean = self.clean_dataframe(df)
                df_clean.to_csv(file_path, index=False)
        contact_dir = os.path.join(self.processed_dir, 'by_contact')
        if os.path.exists(contact_dir):
            for filename in os.listdir(contact_dir):
                if filename.endswith('.csv'):
                    file_path = os.path.join(contact_dir, filename)
                    print(f"Cleaning {filename}...")
                    df = pd.read_csv(file_path)
                    df_clean = self.clean_dataframe(df)
                    df_clean.to_csv(file_path, index=False)
        print("Cleaning complete!")

if __name__ == "__main__":
    config = load_config()
    cleaner = DataCleaner(config)
    cleaner.clean_all_files()
