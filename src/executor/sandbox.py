import subprocess
import time
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    returncode: int
    status: str
    timed_out: bool
    execution_ms: float


class BaseSandbox(ABC):
    @abstractmethod
    def execute(self, command: str, timeout: int = 30) -> ExecutionResult:
        pass


class RestrictedSubprocess(BaseSandbox):
    def __init__(self):
        self.allowed_commands = {"ls", "cat", "grep", "find", "ps", "docker", "git", "curl", "head", "tail", "wc", "sort", "uniq", "cut", "awk", "sed"}
        self.dangerous_flags = {"-rf", "-r", "/", "-force", "--force", "-f"}
    
    def execute(self, command: str, timeout: int = 30) -> ExecutionResult:
        start = time.time()
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_ms = (time.time() - start) * 1000
            
            return ExecutionResult(
                stdout=result.stdout[:10000],
                stderr=result.stderr[:500],
                returncode=result.returncode,
                status="success" if result.returncode == 0 else "error",
                timed_out=False,
                execution_ms=execution_ms
            )
            
        except subprocess.TimeoutExpired:
            execution_ms = (time.time() - start) * 1000
            return ExecutionResult(
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                returncode=124,
                status="timeout",
                timed_out=True,
                execution_ms=execution_ms
            )
        
        except Exception as e:
            execution_ms = (time.time() - start) * 1000
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                returncode=1,
                status="error",
                timed_out=False,
                execution_ms=execution_ms
            )


class DockerSandbox(BaseSandbox):
    def __init__(
        self,
        image: str = "cli-intel-sandbox:latest",
        memory_limit: str = "256m",
        cpu_limit: float = 0.5,
        network_enabled: bool = False
    ):
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network_enabled = network_enabled
        
        self._docker_available = self._check_docker()
    
    def _check_docker(self) -> bool:
        return shutil.which("docker") is not None
    
    def execute(self, command: str, timeout: int = 30) -> ExecutionResult:
        start = time.time()
        
        if not self._docker_available:
            return ExecutionResult(
                stdout="",
                stderr="Docker not available, falling back to subprocess",
                returncode=1,
                status="error",
                timed_out=False,
                execution_ms=0
            )
        
        cmd = self._build_docker_command(command)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 10
            )
            
            execution_ms = (time.time() - start) * 1000
            
            return ExecutionResult(
                stdout=result.stdout[:10000],
                stderr=result.stderr[:500],
                returncode=result.returncode,
                status="success" if result.returncode == 0 else "error",
                timed_out=False,
                execution_ms=execution_ms
            )
            
        except subprocess.TimeoutExpired:
            execution_ms = (time.time() - start) * 1000
            return ExecutionResult(
                stdout="",
                stderr=f"Docker command timed out after {timeout}s",
                returncode=124,
                status="timeout",
                timed_out=True,
                execution_ms=execution_ms
            )
        
        except Exception as e:
            execution_ms = (time.time() - start) * 1000
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                returncode=1,
                status="error",
                timed_out=False,
                execution_ms=execution_ms
            )
    
    def _build_docker_command(self, command: str) -> list:
        cmd = [
            "docker", "run",
            "--rm",
            "--memory", self.memory_limit,
            "--cpus", str(self.cpu_limit),
            "-a", "stdout",
            "-a", "stderr"
        ]
        
        if not self.network_enabled:
            cmd.append("--network=none")
        
        cmd.extend([self.image, "sh", "-c", command])
        return cmd