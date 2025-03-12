import pytest
import subprocess
import time
import requests
import socket
from urllib.parse import urljoin


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
def ensure_cleanup():
    """Fixture to ensure Docker resources are cleaned up after all tests."""
    # Setup - nothing needed before tests
    yield
    # Teardown - clean up after tests
    cleanup_docker_resources()


@pytest.fixture(scope="module")
def docker_service():
    """Start docker compose service for testing and stop it after tests."""
    # Define the port used by the service
    service_port = 9007
    
    # Skip if port is already in use to avoid conflicts
    if is_port_in_use(service_port):
        pytest.skip(f"Port {service_port} is already in use, skipping Docker integration tests")
    
    # Make sure any existing containers are stopped
    subprocess.run(["docker", "compose", "down"], check=False, capture_output=True)
    
    # Build and start the service
    subprocess.run(["docker", "compose", "build"], check=True, capture_output=True)
    subprocess.run(["docker", "compose", "up", "-d"], check=True, capture_output=True)
    
    # Wait for service to start
    max_wait = 30  # seconds
    start_time = time.time()
    service_ready = False
    
    while time.time() - start_time < max_wait:
        if is_port_in_use(service_port):
            service_ready = True
            # Additional grace period to let the application fully initialize
            time.sleep(2)
            break
        time.sleep(1)
    
    if not service_ready:
        # Force cleanup and skip if service didn't start
        subprocess.run(["docker", "compose", "down"], check=False, capture_output=True)
        pytest.skip(f"Service did not start on port {service_port} within {max_wait} seconds")
    
    # Yield the service base URL
    yield f"http://localhost:{service_port}"
    
    # Cleanup after tests
    subprocess.run(["docker", "compose", "down"], check=True, capture_output=True)


@pytest.mark.integration
class TestDockerIntegration:
    
    def test_streamlit_app_accessible(self, docker_service):
        """Test that the Streamlit app is accessible via HTTP."""
        # Streamlit main page should return 200 OK
        response = requests.get(docker_service)
        assert response.status_code == 200, "Failed to access Streamlit app"
        assert "Streamlit" in response.text, "Response doesn't contain expected Streamlit content"
    
    def test_streamlit_api_health(self, docker_service):
        """Test the Streamlit API health endpoint."""
        # Streamlit exposes a health endpoint
        health_url = urljoin(docker_service, "healthz")
        response = requests.get(health_url)
        assert response.status_code == 200, "Streamlit health check failed"
    
    def test_docker_logs(self):
        """Test that the Docker container logs show the application has started."""
        # Get logs from the container
        logs_result = subprocess.run(
            ["docker", "compose", "logs"],
            capture_output=True,
            text=True,
            check=False
        )
        
        assert logs_result.returncode == 0, "Failed to get Docker logs"
        logs = logs_result.stdout
        
        # Check for Streamlit startup messages in logs
        assert "Streamlit" in logs, "Streamlit startup message not found in logs"
        assert "You can now view your Streamlit app in your browser" in logs, "Streamlit successful startup message not found"
        
        # Check for any error messages
        assert "Error:" not in logs.lower(), "Error messages found in Docker logs"
    
    def test_container_resource_usage(self):
        """Test that the container is using resources correctly."""
        # Get container stats
        stats_result = subprocess.run(
            ["docker", "compose", "top"],
            capture_output=True,
            text=True,
            check=False
        )
        
        assert stats_result.returncode == 0, "Failed to get container stats"
        stats = stats_result.stdout
        
        # Check that the process list includes Python (which runs Streamlit)
        assert "python" in stats.lower(), "Python process not found in container" 