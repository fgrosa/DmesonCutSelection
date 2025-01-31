import re
import argparse
from pathlib import Path

def clean_stack_traces(log_content):
    """Remove ROOT stack trace errors from log content"""

    stack_trace_pattern = r'^=+$\n.*\n=+$'
    return re.sub(
        stack_trace_pattern, 
        '', 
        log_content, 
        flags=re.DOTALL | re.MULTILINE
    )

def process_log_file(log_path):
    """Process a single log file to remove stack traces"""
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        cleaned_content = clean_stack_traces(content)
        
        with open(log_path, 'w') as f:
            f.write(cleaned_content)
            
        print(f"Processed: {log_path}")
    except Exception as e:
        print(f"Error processing {log_path}: {str(e)}")

def main(log_file):
    
    if not Path(log_file).exists():
        print(f"Log directory not found: {log_file}")
        return
    
    process_log_file(log_file)

if __name__ == "__main__":
    parser = argparser = argparse.ArgumentParser()
    parser.add_argument("log_file", help="Path to log file to process")
    args = parser.parse_args()
    main(args.log_file)