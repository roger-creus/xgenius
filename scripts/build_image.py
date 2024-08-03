import argparse
import subprocess
import os

def build_docker_image(dockerfile_path, image_name, tag):
    print(f"Building Docker image from {dockerfile_path}...")
    path_to_dockerfile = os.path.dirname(dockerfile_path)
    subprocess.run([
        'docker', 'build',
        '-f', dockerfile_path,
        '-t', f'{image_name}:{tag}',
        path_to_dockerfile
    ], check=True)

def tag_docker_image(image_name, tag, registry_url):
    print(f"Tagging Docker image {image_name}:{tag} for registry {registry_url}...")
    subprocess.run([
        'docker', 'tag',
        f'{image_name}:{tag}',
        f'{registry_url}/{image_name}:{tag}'
    ], check=True)

def push_docker_image(image_name, tag, registry_url):
    print(f"Pushing Docker image {image_name}:{tag} to registry {registry_url}...")
    subprocess.run([
        'docker', 'push',
        f'{registry_url}/{image_name}:{tag}'
    ], check=True)

def build_singularity_image(docker_image, singularity_image):
    print(f"Converting Docker image {docker_image} to Singularity image {singularity_image}...")
    subprocess.run([
        'singularity', 'build',
        singularity_image,
        f'docker://{docker_image}'
    ], check=True)

def main():
    parser = argparse.ArgumentParser(description="Docker to Singularity CLI Utility")
    parser.add_argument('--dockerfile', required=True, help="Path to the Dockerfile")
    parser.add_argument('--name', required=True, help="Name of the Docker image")
    parser.add_argument('--tag', required=True, help="Tag for the Docker image")
    parser.add_argument('--registry', required=True, help="Docker registry URL")

    args = parser.parse_args()

    dockerfile_path = args.dockerfile
    image_name = args.name
    tag = args.tag
    registry_url = args.registry
    singularity_image = image_name + '.sif'

    # Build Docker image
    build_docker_image(dockerfile_path, image_name, tag)

    # Tag Docker image
    tag_docker_image(image_name, tag, registry_url)

    # Push Docker image to registry
    push_docker_image(image_name, tag, registry_url)

    # Build Singularity image from Docker image
    docker_image = f'{registry_url}/{image_name}:{tag}'
    build_singularity_image(docker_image, singularity_image)

if __name__ == "__main__":
    main()