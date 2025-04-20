import pytest
import subprocess
import time


def wait_for_docker_build(timeout: int = 300) -> bool:
    """Wait for Docker image to build"""
    print("\nBuilding Docker image...")
    process = subprocess.Popen(
        ["docker", "compose", "build"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    while True:
        output = process.stdout.readline() if process.stdout else ''
        if output:
            print(output.strip())
        if process.poll() is not None:
            break
    
    return process.returncode == 0


def wait_for_streamlit(timeout: int = 300) -> bool:
    """Wait for Streamlit to be ready"""
    start = time.time()
    while time.time() - start < timeout:
        logs = subprocess.run(
            ["docker", "compose", "logs"],
            capture_output=True,
            text=True
        )

        if "You can now view your Streamlit app in your browser" in logs.stdout:
            return True
        time.sleep(2)
    return False


@pytest.fixture(scope="module")
def docker_service():
    """Start docker compose service for testing"""
    try:
        print("\nStopping existing containers...")
        subprocess.run(["docker", "compose", "down"], check=False)
        
        # Ждем сборки образа
        assert wait_for_docker_build(), "Docker build failed"
        
        print("\nStarting service...")
        result = subprocess.run(["docker", "compose", "up", "-d"], check=True)
        assert result.returncode == 0
        
        # Ждем пока Streamlit будет готов
        assert wait_for_streamlit(), "Streamlit failed to start"
        
        yield
        
    finally:
        print("\nCleaning up...")
        subprocess.run(["docker", "compose", "down"])


# Регистрируем маркер integration
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


@pytest.mark.integration
class TestDockerIntegration:
    def test_container_running(self, docker_service):
        """Test that container is running and Streamlit started"""
        logs = subprocess.run(
            ["docker", "compose", "logs"],
            capture_output=True,
            text=True
        )
        assert "You can now view your Streamlit app in your browser." in logs.stdout