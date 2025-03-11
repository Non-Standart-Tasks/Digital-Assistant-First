import subprocess
import time
import socket
import pytest
import os
from pathlib import Path


def is_port_in_use(port, host='localhost'):
    """Check if a port is in use on the specified host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def get_docker_images():
    """Return list of Docker images created by this project."""
    result = subprocess.run(
        ["docker", "images", "--filter", "reference=*digital-assistant*", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
        check=False
    )
    if result.returncode != 0:
        return []
    return [img for img in result.stdout.strip().split('\n') if img]


def cleanup_docker_resources():
    """Clean up all Docker resources (containers and images) created during tests."""
    # Stop all containers
    subprocess.run(["docker", "compose", "down"], capture_output=True, check=False)
    
    # Get all project images
    images = get_docker_images()
    
    if images:
        print(f"Cleaning up Docker images: {', '.join(images)}")
        
        # First try to remove images normally
        for image in images:
            subprocess.run(["docker", "rmi", image], capture_output=True, check=False)
            
        # Force remove any stubborn images
        for image in get_docker_images():
            subprocess.run(["docker", "rmi", "-f", image], capture_output=True, check=False)
    
    # Verify all images have been removed
    remaining = get_docker_images()
    if remaining:
        print(f"Warning: Failed to remove some Docker images: {', '.join(remaining)}")
    else:
        print("All Docker images successfully removed")


@pytest.fixture(scope="module", autouse=True)
def docker_cleanup():
    """Fixture to clean up Docker resources after all tests in module."""
    # Setup - nothing needed before tests
    yield
    # Teardown - clean up after tests
    cleanup_docker_resources()


class TestDockerCompose:

    def setup_method(self):
        """Ensure the docker-compose.yml file exists before running tests."""
        # Check if docker-compose.yml exists
        compose_file = Path("docker-compose.yml")
        if not compose_file.exists():
            pytest.skip("docker-compose.yml not found, skipping Docker tests")

    def test_docker_compose_build(self):
        """Test that docker compose build completes successfully."""
        # Run docker compose build
        result = subprocess.run(
            ["docker", "compose", "build"], 
            capture_output=True, 
            text=True,
            check=False
        )
        
        # Check if the build was successful
        assert result.returncode == 0, f"Docker compose build failed with error: {result.stderr}"
        print(f"Docker compose build output: {result.stdout}")

    @pytest.mark.slow
    def test_docker_compose_up_and_down(self):
        """Test that docker compose up starts the service and docker compose down stops it."""
        # Define the port the service should run on
        # Update this to match the port in your docker-compose.yml
        service_port = 9007
        
        # Check if port is already in use
        if is_port_in_use(service_port):
            pytest.skip(f"Port {service_port} is already in use, skipping Docker up test")
            
        # Start the service in detached mode
        up_result = subprocess.run(
            ["docker", "compose", "up", "-d"],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Check if the up command was successful
        assert up_result.returncode == 0, f"Docker compose up failed with error: {up_result.stderr}"
        print(f"Docker compose up output: {up_result.stdout}")
        
        try:
            # Wait for the service to start (may need to adjust time)
            max_wait = 30  # seconds
            start_time = time.time()
            service_ready = False
            
            while time.time() - start_time < max_wait:
                if is_port_in_use(service_port):
                    service_ready = True
                    break
                time.sleep(1)
            
            # Check if the service is running
            assert service_ready, f"Service did not start on port {service_port} within {max_wait} seconds"
            
            # Additional checks can be added here to verify the service is working correctly
            # For example, making HTTP requests to the service endpoints
            
        finally:
            # Always stop the service after the test
            down_result = subprocess.run(
                ["docker", "compose", "down"],
                capture_output=True,
                text=True,
                check=False
            )
            
            # Check if the down command was successful
            assert down_result.returncode == 0, f"Docker compose down failed with error: {down_result.stderr}"
            print(f"Docker compose down output: {down_result.stdout}")
            
            # Verify port is no longer in use
            assert not is_port_in_use(service_port), f"Port {service_port} is still in use after docker compose down"

    @pytest.mark.skipif(os.environ.get('CI') != 'true', reason="Run only in CI environment")
    def test_docker_image_contents(self):
        """Test that the Docker image contains all required files."""
        # Build the image if not already built
        build_result = subprocess.run(
            ["docker", "compose", "build"],
            capture_output=True,
            text=True,
            check=False
        )
        assert build_result.returncode == 0, "Failed to build Docker image"
        
        # Get the image name from docker compose
        image_info = subprocess.run(
            ["docker", "compose", "config", "--images"],
            capture_output=True,
            text=True,
            check=False
        )
        assert image_info.returncode == 0, "Failed to get Docker image info"
        
        # Image name should be in the output
        image_name = image_info.stdout.strip()
        assert image_name, "Could not determine Docker image name"
        
        # Check the container filesystem
        expected_files = [
            "/app/streamlit_app.py",
            "/app/config.yaml",
        ]
        
        for file_path in expected_files:
            check_result = subprocess.run(
                ["docker", "run", "--rm", image_name, "test", "-f", file_path],
                capture_output=True,
                text=True,
                check=False
            )
            assert check_result.returncode == 0, f"File {file_path} not found in Docker image" 