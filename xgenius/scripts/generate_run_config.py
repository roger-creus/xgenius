import json
import re
import os
import argparse

def extract_placeholders(template_path):
    """
    Extract all placeholders from the sbatch template file.
    """
    placeholders = set()
    with open(template_path, 'r') as file:
        content = file.read()
        # Find all placeholders in the format {{PARAM}}
        placeholders.update(re.findall(r'\{\{(\w+)\}\}', content))
        placeholders.discard('COMMAND')
    return placeholders

def generate_run_config(cluster_config_path):
    """
    Generate a run_config.json file based on the provided cluster configuration file
    and the sbatch template files.
    """
    # Load the cluster configuration
    with open(cluster_config_path, 'r') as f:
        cluster_config = json.load(f)
    
    run_config = {}

    for cluster in cluster_config:
        output_config_path = "run_config.json"
        cluster_name = cluster['cluster_name']
        sbatch_template_path = cluster['sbatch_template']
        
        # Assuming sbatch templates are in a subdirectory named 'sbatch_templates'
        sbatch_template_path = os.path.join('sbatch_templates', sbatch_template_path)
        
        # Extract placeholders from the sbatch template
        placeholders = extract_placeholders(sbatch_template_path)
        
        # Create a default dictionary for run_config with placeholders
        run_config[cluster_name] = {placeholder: f'{{{{{placeholder}}}}}' for placeholder in placeholders}

    # Save the generated run_config to a JSON file
    with open(output_config_path, 'w') as f:
        json.dump(run_config, f, indent=4)

    print(f"Run configuration file saved to {output_config_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate a run_config.json from cluster configuration and sbatch templates.')
    parser.add_argument('cluster_config', help='Path to the cluster configuration JSON file')
    
    args = parser.parse_args()

    generate_run_config(args.cluster_config)

if __name__ == "__main__":
    main()
