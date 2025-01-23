import os
import json
import argparse

def summarize_results_in_directory(evaluation_subdir):
    """Summarize results within a single evaluation subdirectory."""
    total_count = 0
    valid_count = 0
    successful_count = 0
    safe_count = 0
    
    for filename in os.listdir(evaluation_subdir):
        if filename.endswith(".results.json"):
            file_path = os.path.join(evaluation_subdir, filename)
            total_count += 1

            with open(file_path, 'r') as file:
                data = json.load(file)
                if data.get("valid"):
                    valid_count += 1
                if data.get("successful"):
                    successful_count += 1
                if data.get("safe"):
                    safe_count += 1
    
    result = {
        "total": total_count,
        "valid": valid_count,
        "successful": successful_count,
        "safe": safe_count
    }

    output_file_path = os.path.join(evaluation_subdir, "results_summary.json")
    with open(output_file_path, 'w') as output_file:
        json.dump(result, output_file, indent=4)
    
    print(f"[info] Results summary written to {output_file_path}")

def find_and_summarize_results(base_dir):
    """Recursively find all relevant subdirectories and summarize results."""
    for dirpath, dirnames, files in os.walk(base_dir):
        # Check if the directory contains any .results.json files
        if any(filename.endswith(".results.json") for filename in files):
            summarize_results_in_directory(dirpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize JSON results in all 'evaluation' subdirectories.")
    parser.add_argument("base_dir", type=str, help="Base directory to search for 'evaluation' subdirectories")
    
    args = parser.parse_args()
    find_and_summarize_results(args.base_dir)
