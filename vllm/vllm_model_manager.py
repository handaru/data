#!/usr/bin/env python3
"""
vLLM Model Management Script
Provides utilities to check, list, and remove models from vLLM cache
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict
import argparse

class VLLMModelManager:
    def __init__(self):
        # Common vLLM model cache locations
        self.cache_locations = [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "huggingface" / "transformers", 
            Path("/tmp/vllm_cache"),
            Path("/var/cache/vllm"),
            # Add custom cache paths if specified in environment
        ]
        
        # Check for custom cache directory
        if "VLLM_CACHE_ROOT" in os.environ:
            self.cache_locations.append(Path(os.environ["VLLM_CACHE_ROOT"]))
        
        if "HF_HOME" in os.environ:
            self.cache_locations.append(Path(os.environ["HF_HOME"]) / "hub")

    def find_model_directories(self) -> Dict[str, List[Path]]:
        """Find all model directories in cache locations"""
        found_models = {}
        
        for cache_dir in self.cache_locations:
            if not cache_dir.exists():
                continue
                
            print(f"Scanning: {cache_dir}")
            
            # Look for Hugging Face model directories
            if cache_dir.name == "hub":
                for model_dir in cache_dir.iterdir():
                    if model_dir.is_dir() and model_dir.name.startswith("models--"):
                        model_name = model_dir.name.replace("models--", "").replace("--", "/")
                        if model_name not in found_models:
                            found_models[model_name] = []
                        found_models[model_name].append(model_dir)
            
            # Look for direct model directories
            else:
                for item in cache_dir.iterdir():
                    if item.is_dir():
                        # Check if it looks like a model directory
                        if self._is_model_directory(item):
                            model_name = item.name
                            if model_name not in found_models:
                                found_models[model_name] = []
                            found_models[model_name].append(item)
        
        return found_models

    def _is_model_directory(self, path: Path) -> bool:
        """Check if directory contains model files"""
        model_files = [
            "config.json", "pytorch_model.bin", "model.safetensors",
            "tokenizer.json", "tokenizer_config.json"
        ]
        
        return any((path / f).exists() for f in model_files)

    def get_directory_size(self, path: Path) -> float:
        """Get directory size in GB"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except (OSError, PermissionError):
            return 0
        
        return total_size / (1024**3)  # Convert to GB

    def list_models(self, detailed: bool = False) -> None:
        """List all cached models"""
        models = self.find_model_directories()
        
        if not models:
            print("No cached models found.")
            return
        
        print(f"\n{'='*60}")
        print("CACHED VLLM MODELS")
        print(f"{'='*60}")
        
        total_size = 0
        for model_name, paths in models.items():
            model_size = sum(self.get_directory_size(path) for path in paths)
            total_size += model_size
            
            print(f"\nModel: {model_name}")
            print(f"Size: {model_size:.2f} GB")
            
            if detailed:
                print("Locations:")
                for path in paths:
                    print(f"  - {path}")
                    if path.exists():
                        print(f"    Last modified: {os.path.getmtime(path)}")
        
        print(f"\nTotal cache size: {total_size:.2f} GB")
        print(f"Total models: {len(models)}")

    def remove_model(self, model_name: str, confirm: bool = True) -> bool:
        """Remove a specific model from cache"""
        models = self.find_model_directories()
        
        if model_name not in models:
            print(f"Model '{model_name}' not found in cache.")
            return False
        
        paths = models[model_name]
        total_size = sum(self.get_directory_size(path) for path in paths)
        
        print(f"Found model '{model_name}' ({total_size:.2f} GB)")
        print("Locations to be removed:")
        for path in paths:
            print(f"  - {path}")
        
        if confirm:
            response = input(f"\nAre you sure you want to remove '{model_name}'? (y/N): ")
            if response.lower() != 'y':
                print("Operation cancelled.")
                return False
        
        success = True
        for path in paths:
            try:
                if path.exists():
                    shutil.rmtree(path)
                    print(f"Removed: {path}")
            except Exception as e:
                print(f"Error removing {path}: {e}")
                success = False
        
        return success

    def remove_all_models(self, confirm: bool = True) -> bool:
        """Remove all cached models"""
        models = self.find_model_directories()
        
        if not models:
            print("No cached models found.")
            return True
        
        total_size = sum(
            sum(self.get_directory_size(path) for path in paths)
            for paths in models.values()
        )
        
        print(f"Found {len(models)} models totaling {total_size:.2f} GB")
        
        if confirm:
            response = input(f"Are you sure you want to remove ALL cached models? (y/N): ")
            if response.lower() != 'y':
                print("Operation cancelled.")
                return False
        
        success = True
        for model_name in models:
            if not self.remove_model(model_name, confirm=False):
                success = False
        
        return success

    def clean_empty_directories(self) -> None:
        """Remove empty cache directories"""
        for cache_dir in self.cache_locations:
            if not cache_dir.exists():
                continue
            
            try:
                # Remove empty subdirectories
                for root, dirs, files in os.walk(cache_dir, topdown=False):
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        try:
                            if not os.listdir(dir_path):
                                os.rmdir(dir_path)
                                print(f"Removed empty directory: {dir_path}")
                        except OSError:
                            pass
            except Exception as e:
                print(f"Error cleaning {cache_dir}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Manage vLLM cached models")
    parser.add_argument("--list", "-l", action="store_true", help="List all cached models")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed information")
    parser.add_argument("--remove", "-r", type=str, help="Remove specific model by name")
    parser.add_argument("--remove-all", action="store_true", help="Remove all cached models")
    parser.add_argument("--clean", "-c", action="store_true", help="Clean empty directories")
    parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompts")
    
    args = parser.parse_args()
    
    manager = VLLMModelManager()
    
    if args.list or not any(vars(args).values()):
        manager.list_models(detailed=args.detailed)
    
    if args.remove:
        manager.remove_model(args.remove, confirm=not args.no_confirm)
    
    if args.remove_all:
        manager.remove_all_models(confirm=not args.no_confirm)
    
    if args.clean:
        manager.clean_empty_directories()


if __name__ == "__main__":
    main()