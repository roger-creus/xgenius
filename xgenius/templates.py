"""SBATCH template rendering for xgenius.

Handles loading, rendering, and epilog injection for SBATCH scripts.
"""

import os
import re

from xgenius.config import ClusterConfig


# Epilog injected at the end of every SBATCH script for completion detection
COMPLETION_EPILOG = '''
# --- xgenius completion marker ---
XGENIUS_EXIT_CODE=$?
XGENIUS_MARKER_DIR="{{SCRATCH_PATH}}/.xgenius/markers"
mkdir -p "$XGENIUS_MARKER_DIR"
cat > "$XGENIUS_MARKER_DIR/$SLURM_JOB_ID.done" <<'XGENIUS_MARKER_EOF'
{"job_id":"SLURM_JOB_ID_PLACEHOLDER","experiment_id":"{{EXPERIMENT_ID}}","exit_code":XGENIUS_EXIT_CODE_PLACEHOLDER,"cluster":"{{CLUSTER_NAME}}","completed_at":"XGENIUS_TIMESTAMP_PLACEHOLDER","output_dir":"{{OUTPUT_DIR_IN_CLUSTER}}","walltime_seconds":XGENIUS_SECONDS_PLACEHOLDER}
XGENIUS_MARKER_EOF
# Fix placeholders with runtime values
sed -i "s/SLURM_JOB_ID_PLACEHOLDER/$SLURM_JOB_ID/g" "$XGENIUS_MARKER_DIR/$SLURM_JOB_ID.done"
sed -i "s/XGENIUS_EXIT_CODE_PLACEHOLDER/$XGENIUS_EXIT_CODE/g" "$XGENIUS_MARKER_DIR/$SLURM_JOB_ID.done"
sed -i "s/XGENIUS_TIMESTAMP_PLACEHOLDER/$(date -u +%Y-%m-%dT%H:%M:%SZ)/g" "$XGENIUS_MARKER_DIR/$SLURM_JOB_ID.done"
sed -i "s/XGENIUS_SECONDS_PLACEHOLDER/$SECONDS/g" "$XGENIUS_MARKER_DIR/$SLURM_JOB_ID.done"
exit $XGENIUS_EXIT_CODE
'''


def get_template_dir() -> str:
    """Get the directory containing SBATCH templates.

    Checks XGENIUS_TEMPLATES_DIR env var, falls back to package templates.
    """
    env_dir = os.getenv("XGENIUS_TEMPLATES_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir

    # Fall back to package-bundled templates
    return os.path.join(os.path.dirname(__file__), "sbatch_templates")


def load_template(template_name: str) -> str:
    """Load an SBATCH template by name.

    Args:
        template_name: Filename of the template (e.g., 'slurm_partition_template.sbatch')

    Returns:
        Template content as a string.

    Raises:
        FileNotFoundError: If template doesn't exist.
    """
    template_dir = get_template_dir()
    template_path = os.path.join(template_dir, template_name)

    if not os.path.exists(template_path):
        raise FileNotFoundError(
            f"SBATCH template not found: {template_path}\n"
            f"Available templates in {template_dir}: {os.listdir(template_dir) if os.path.isdir(template_dir) else 'directory not found'}"
        )

    with open(template_path) as f:
        return f.read()


def extract_placeholders(template: str) -> list[str]:
    """Extract all {{PLACEHOLDER}} names from a template.

    Args:
        template: Template content string.

    Returns:
        List of unique placeholder names (without braces).
    """
    return list(set(re.findall(r"\{\{(\w+)\}\}", template)))


def render_template(
    template: str,
    params: dict[str, str],
    command: str,
    cluster: ClusterConfig,
    experiment_id: str = "",
    inject_epilog: bool = True,
) -> str:
    """Render an SBATCH template with parameter substitution and epilog injection.

    Args:
        template: Raw template string with {{PLACEHOLDER}} markers.
        params: Dict mapping placeholder names to values.
        command: The command to run inside the container.
        cluster: Cluster configuration for injecting cluster-specific values.
        experiment_id: Experiment ID for the completion marker.
        inject_epilog: Whether to inject the completion marker epilog.

    Returns:
        Rendered SBATCH script ready for submission.
    """
    rendered = template

    # Substitute all params
    for key, value in params.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))

    # Substitute the command last (in case it appears in params too)
    rendered = rendered.replace("{{COMMAND}}", command)

    # Inject epilog for completion detection
    if inject_epilog:
        epilog = COMPLETION_EPILOG
        epilog = epilog.replace("{{SCRATCH_PATH}}", cluster.scratch_path)
        epilog = epilog.replace("{{EXPERIMENT_ID}}", experiment_id or "unknown")
        epilog = epilog.replace("{{CLUSTER_NAME}}", cluster.name)

        # Replace OUTPUT_DIR_IN_CLUSTER in epilog if it's in params
        output_dir = params.get("OUTPUT_DIR_IN_CLUSTER", cluster.slurm.output_dir_cluster)
        epilog = epilog.replace("{{OUTPUT_DIR_IN_CLUSTER}}", output_dir)

        rendered += epilog

    return rendered


def build_params_from_cluster(cluster: ClusterConfig) -> dict[str, str]:
    """Build template parameters dict from cluster SLURM config.

    Maps ClusterConfig.slurm fields to the {{PLACEHOLDER}} names used in templates.
    """
    slurm = cluster.slurm
    params = {
        "NUM_GPUS": str(slurm.num_gpus),
        "NUM_CPUS": str(slurm.num_cpus),
        "MEM": slurm.memory,
        "TIME": slurm.walltime,
        "MODULES_TO_LOAD": slurm.modules,
        "SINGULARITY_COMMAND": slurm.singularity_command,
        "CODE_DIR_IN_CLUSTER": cluster.project_path,
        "OUTPUT_DIR_IN_CLUSTER": slurm.output_dir_cluster,
        "OUTPUT_DIR_IN_CONTAINER": slurm.output_dir_container,
        "IMAGE_NAME": os.path.basename(
            os.path.join(cluster.image_path, "")  # Get just the image filename
        ),
        "IMAGE_PATH": cluster.image_path,
    }

    # Add account or partition depending on template type
    if slurm.account:
        params["ACCOUNT"] = slurm.account
    if slurm.partition:
        params["PARTITION"] = slurm.partition

    # Output file
    if slurm.output_file:
        params["OUTPUT_FILE"] = slurm.output_file
    else:
        # Default: write SLURM logs to scratch
        params["OUTPUT_FILE"] = os.path.join(
            cluster.scratch_path, "slurm-%j.out"
        )

    return params
