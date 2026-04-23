import yaml
import json
import time
import hashlib
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class PromptTemplate:
    name: str
    version: str
    description: str
    system: str
    user: str
    model_override: Optional[str] = None
    temperature_override: Optional[float] = None
    output_schema: Optional[dict] = None
    loaded_at: float = field(default_factory=time.time)
    file_hash: str = ""

    def render(self, variables: dict[str, Any]) -> tuple[str, str]:
        system = self._render_template(self.system, variables)
        user = self._render_template(self.user, variables)
        return system, user

    def _render_template(self, template: str, variables: dict) -> str:
        import re
        
        pattern = re.compile(r'(?<!{)\{(?![{])(\w+)\}')
        matches = {}
        for key in variables:
            matches[key] = variables[key]
        
        class SafeDict(dict):
            def __missing__(self, key):
                return f"<{key}: not provided>"
        
        try:
            result = template
            for key, val in variables.items():
                result = result.replace(f'{{{key}}}', str(val))
            return result
        except Exception as e:
            return template + f"\n[Template render warning: {e}]"

    def get_model(self, default_model: str) -> str:
        return self.model_override or default_model

    def get_temperature(self, default_temp: float) -> float:
        return self.temperature_override if self.temperature_override is not None else default_temp


class PromptRegistry:
    def __init__(self, prompts_dir: str, hot_reload: bool = True):
        self.prompts_dir = Path(prompts_dir)
        self.hot_reload = hot_reload
        self._cache: dict[str, PromptTemplate] = {}
        self._file_hashes: dict[str, str] = {}
        self._load_all()

    def _load_all(self):
        for yaml_file in self.prompts_dir.glob("*.yaml"):
            self._load_file(yaml_file)

    def _load_file(self, path: Path) -> Optional[PromptTemplate]:
        try:
            content = path.read_text()
            file_hash = hashlib.md5(content.encode()).hexdigest()

            data = yaml.safe_load(content)
            template = PromptTemplate(
                name=data["name"],
                version=data.get("version", "1.0"),
                description=data.get("description", ""),
                system=data.get("system", ""),
                user=data["user"],
                model_override=data.get("model_override"),
                temperature_override=data.get("temperature_override"),
                output_schema=data.get("output_schema"),
                file_hash=file_hash
            )

            self._cache[template.name] = template
            self._file_hashes[str(path)] = file_hash
            return template

        except Exception as e:
            print(f"[PromptRegistry] Failed to load {path}: {e}")
            return None

    def get(self, name: str) -> PromptTemplate:
        if self.hot_reload:
            self._check_reload(name)

        if name not in self._cache:
            available = list(self._cache.keys())
            raise KeyError(
                f"Prompt '{name}' not found. "
                f"Available prompts: {available}. "
                f"Create '{name}.yaml' in {self.prompts_dir} to add it."
            )

        return self._cache[name]

    def _check_reload(self, name: str):
        yaml_file = self.prompts_dir / f"{name}.yaml"
        if not yaml_file.exists():
            return

        current_hash = hashlib.md5(yaml_file.read_bytes()).hexdigest()
        cached_hash = self._file_hashes.get(str(yaml_file), "")

        if current_hash != cached_hash:
            self._load_file(yaml_file)

    def list_prompts(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "version": t.version,
                "description": t.description,
                "has_model_override": t.model_override is not None,
                "has_schema": t.output_schema is not None
            }
            for t in self._cache.values()
        ]