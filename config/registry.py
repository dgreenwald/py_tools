#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic configuration registry with inheritance support.

This module provides a reusable ConfigRegistry class for managing
configurations (dicts) that can inherit from each other. Common use case:
managing multiple model specifications in research projects.

The registry is agnostic to what the configurations contain - it just
handles the inheritance, merging, and provides convenience methods for
inspection and comparison.

Example Usage
-------------
>>> from config_registry import ConfigRegistry
>>>
>>> # Define your configurations
>>> CONFIGS = {
>>>     'base': {
>>>         'name': 'Base configuration',
>>>         'params': {'alpha': 0.5, 'beta': 0.9},
>>>         'flags': {'use_feature_x': True},
>>>     },
>>>     'variant1': {
>>>         'name': 'Variant 1',
>>>         'inherits_from': 'base',  # Inherit from 'base'
>>>         'params': {'alpha': 0.7},  # Override just alpha
>>>     },
>>>     'variant2': {
>>>         'name': 'Variant 2',
>>>         'inherits_from': 'variant1',  # Chain: variant2 -> variant1 -> base
>>>         'flags': {'use_feature_y': True},  # Add a new flag
>>>     },
>>> }
>>>
>>> # Create registry
>>> registry = ConfigRegistry(CONFIGS)
>>>
>>> # Get fully resolved config (inheritance applied)
>>> config = registry.get('variant2')
>>> # {'name': 'Variant 2',
>>> #  'params': {'alpha': 0.7, 'beta': 0.9},  # alpha from variant1, beta from base
>>> #  'flags': {'use_feature_x': True, 'use_feature_y': True}}  # merged
>>>
>>> # Print summary
>>> registry.print('variant2')
>>>
>>> # Compare two configs
>>> registry.compare('variant1', 'variant2')
>>>
>>> # Add new config dynamically
>>> registry.add('variant3', name='Variant 3', inherits_from='base', params={'gamma': 0.1})

Author: Generated for research project configuration management
License: MIT
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import copy


class ConfigRegistry:
    """
    Registry for managing configurations with inheritance.

    Handles arbitrary nested dictionaries where configurations can inherit
    from each other using an 'inherits_from' key. Resolves inheritance
    chains recursively and provides utilities for inspection and comparison.

    Parameters
    ----------
    configs : dict
        Dictionary mapping config identifiers (str or int) to configuration dicts.
        Each config dict can have an 'inherits_from' key pointing to a parent config.
    merge_strategy : str, optional
        How to merge parent and child configs:
        - 'deep': Deep merge (nested dicts are merged recursively) [default]
        - 'shallow': Shallow merge (nested dicts are replaced, not merged)

    Attributes
    ----------
    configs : dict
        The underlying configuration dictionary
    _cache : dict
        Cache of resolved configurations (for performance)
    _merge_strategy : str
        Current merge strategy

    Examples
    --------
    >>> configs = {
    ...     1: {'name': 'Base', 'x': 1, 'y': 2},
    ...     2: {'name': 'Derived', 'inherits_from': 1, 'x': 10},
    ... }
    >>> registry = ConfigRegistry(configs)
    >>> registry.get(2)
    {'name': 'Derived', 'x': 10, 'y': 2}  # x overridden, y inherited
    """

    def __init__(self, configs: Dict[Union[str, int], Dict], merge_strategy: str = 'deep'):
        self.configs = configs
        self._cache = {}
        self._merge_strategy = merge_strategy

    def get(self, config_id: Union[str, int], **extra_context) -> Dict[str, Any]:
        """
        Get configuration with inheritance fully resolved.

        Parameters
        ----------
        config_id : str or int
            Configuration identifier
        **extra_context : dict, optional
            Additional context to merge into config (not cached)

        Returns
        -------
        dict
            Fully resolved configuration with inheritance applied

        Raises
        ------
        KeyError
            If config_id not found in registry
        ValueError
            If circular inheritance detected
        """
        # Check cache (only if no extra context)
        cache_key = (config_id, tuple(sorted(extra_context.items())))
        if not extra_context and cache_key in self._cache:
            return self._cache[cache_key]

        # Validate config exists
        if config_id not in self.configs:
            available = sorted(self.configs.keys())
            raise KeyError(
                f"Config '{config_id}' not found in registry. "
                f"Available configs: {available}"
            )

        # Resolve inheritance
        resolved = self._resolve_inheritance(config_id, visited=set())

        # Apply extra context
        if extra_context:
            resolved = self._merge_dicts(resolved, extra_context)

        # Cache if no extra context
        if not extra_context:
            self._cache[cache_key] = resolved

        return resolved

    def _resolve_inheritance(self, config_id: Union[str, int], visited: set) -> Dict[str, Any]:
        """
        Recursively resolve inheritance chain.

        Parameters
        ----------
        config_id : str or int
            Configuration to resolve
        visited : set
            Set of already-visited configs (for cycle detection)

        Returns
        -------
        dict
            Resolved configuration

        Raises
        ------
        ValueError
            If circular inheritance detected
        """
        # Detect circular inheritance
        if config_id in visited:
            cycle = list(visited) + [config_id]
            raise ValueError(f"Circular inheritance detected: {' -> '.join(map(str, cycle))}")

        visited = visited | {config_id}

        # Get config
        config = copy.deepcopy(self.configs[config_id])

        # Check for inheritance
        if 'inherits_from' not in config:
            return config

        # Get parent
        parent_id = config.pop('inherits_from')

        # Handle inheritance from another config
        if isinstance(parent_id, (str, int)) and parent_id in self.configs:
            parent = self._resolve_inheritance(parent_id, visited)
        # Handle inheritance from a dict directly (useful for base configs)
        elif isinstance(parent_id, dict):
            parent = copy.deepcopy(parent_id)
        else:
            raise ValueError(
                f"Invalid inherits_from value in config '{config_id}': {parent_id}. "
                f"Must be a config ID or dict."
            )

        # Merge parent and child
        return self._merge_dicts(parent, config)

    def _merge_dicts(self, parent: Dict, child: Dict) -> Dict:
        """
        Merge two dictionaries according to merge strategy.

        Parameters
        ----------
        parent : dict
            Parent (base) configuration
        child : dict
            Child (overriding) configuration

        Returns
        -------
        dict
            Merged configuration
        """
        if self._merge_strategy == 'shallow':
            # Shallow merge: child overwrites parent at top level
            result = parent.copy()
            result.update(child)
            return result

        elif self._merge_strategy == 'deep':
            # Deep merge: recursively merge nested dicts
            result = copy.deepcopy(parent)

            for key, child_value in child.items():
                if key in result and isinstance(result[key], dict) and isinstance(child_value, dict):
                    # Both are dicts: merge recursively
                    result[key] = self._merge_dicts(result[key], child_value)
                else:
                    # Not both dicts, or key not in parent: override
                    result[key] = copy.deepcopy(child_value)

            return result

        else:
            raise ValueError(f"Unknown merge_strategy: {self._merge_strategy}")

    def add(self, config_id: Union[str, int], **config_dict):
        """
        Add a new configuration to the registry.

        Parameters
        ----------
        config_id : str or int
            Unique identifier for this configuration
        **config_dict : dict
            Configuration dictionary (can include 'inherits_from')

        Raises
        ------
        ValueError
            If config_id already exists

        Examples
        --------
        >>> registry.add('new_config', name='New', inherits_from='base', x=10)
        """
        if config_id in self.configs:
            raise ValueError(
                f"Config '{config_id}' already exists. "
                f"Use update() to modify or remove() first."
            )

        self.configs[config_id] = config_dict
        self._invalidate_cache()

    def update(self, config_id: Union[str, int], **updates):
        """
        Update an existing configuration.

        Parameters
        ----------
        config_id : str or int
            Configuration to update
        **updates : dict
            Fields to update

        Raises
        ------
        KeyError
            If config_id does not exist
        """
        if config_id not in self.configs:
            raise KeyError(
                f"Config '{config_id}' does not exist. "
                f"Use add() to create new configs."
            )

        self.configs[config_id].update(updates)
        self._invalidate_cache()

    def remove(self, config_id: Union[str, int]):
        """Remove a configuration from the registry."""
        if config_id in self.configs:
            del self.configs[config_id]
            self._invalidate_cache()

    def list(self, include_descriptions: bool = True) -> List[Tuple]:
        """
        List all configurations in the registry.

        Parameters
        ----------
        include_descriptions : bool, optional
            Whether to include 'name' or 'description' fields (default: True)

        Returns
        -------
        list of tuples
            If include_descriptions=True: [(id, name), ...]
            If include_descriptions=False: [id, ...]
        """
        if not include_descriptions:
            return sorted(self.configs.keys())

        result = []
        for config_id in sorted(self.configs.keys()):
            config = self.configs[config_id]
            name = config.get('name', config.get('description', str(config_id)))
            result.append((config_id, name))

        return result

    def print(self, config_id: Union[str, int], max_depth: int = None):
        """
        Print a human-readable summary of a configuration.

        Parameters
        ----------
        config_id : str or int
            Configuration to print
        max_depth : int, optional
            Maximum depth for nested dict printing (default: unlimited)
        """
        config = self.get(config_id)

        # Get name/description if available
        title = config.get('name', config.get('description', f'Config {config_id}'))

        print(f"\n{'='*70}")
        print(f"Configuration: {config_id}")
        if title != f'Config {config_id}':
            print(f"  {title}")
        print(f"{'='*70}")

        # Check if this config inherits from anything
        if config_id in self.configs and 'inherits_from' in self.configs[config_id]:
            parent_id = self.configs[config_id]['inherits_from']
            print(f"\nInherits from: {parent_id}")

        # Print all fields (except name/description already shown)
        print()
        for key, value in sorted(config.items()):
            if key in ['name', 'description']:
                continue
            self._print_value(key, value, indent=0, max_depth=max_depth)

        print()

    def _print_value(self, key: str, value: Any, indent: int = 0, max_depth: int = None):
        """Recursively print a configuration value."""
        prefix = "  " * indent

        if isinstance(value, dict) and (max_depth is None or indent < max_depth):
            print(f"{prefix}{key}:")
            for k, v in sorted(value.items()):
                self._print_value(k, v, indent + 1, max_depth)
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            print(f"{prefix}{key}: [{len(value)} items]")
            if len(value) <= 5:  # Show short lists
                for item in value:
                    print(f"{prefix}  - {item}")
        else:
            # Format numbers nicely
            if isinstance(value, float):
                if abs(value) < 1e-3 or abs(value) > 1e6:
                    value_str = f"{value:.3e}"
                else:
                    value_str = f"{value:.6g}"
            else:
                value_str = str(value)

            print(f"{prefix}{key:25s} = {value_str}")

    def compare(self, config_id1: Union[str, int], config_id2: Union[str, int]):
        """
        Compare two configurations and show differences.

        Parameters
        ----------
        config_id1, config_id2 : str or int
            Configuration identifiers to compare
        """
        config1 = self.get(config_id1)
        config2 = self.get(config_id2)

        # Get names for display
        name1 = config1.get('name', str(config_id1))
        name2 = config2.get('name', str(config_id2))

        print(f"\n{'='*70}")
        print(f"Comparing Configurations")
        print(f"{'='*70}")
        print(f"Config 1: {config_id1} - {name1}")
        print(f"Config 2: {config_id2} - {name2}")
        print()

        # Find all keys
        all_keys = set(config1.keys()) | set(config2.keys())
        all_keys.discard('name')  # Don't compare names
        all_keys.discard('description')

        # Compare each key
        differences = []
        for key in sorted(all_keys):
            val1 = config1.get(key, '<not set>')
            val2 = config2.get(key, '<not set>')

            if val1 != val2:
                differences.append((key, val1, val2))

        if not differences:
            print("✓ Configurations are identical")
        else:
            print(f"Found {len(differences)} difference(s):\n")
            for key, val1, val2 in differences:
                print(f"  {key}:")
                print(f"    Config 1: {val1}")
                print(f"    Config 2: {val2}")
                print()

        print()

    def validate_all(self) -> Dict[Union[str, int], str]:
        """
        Validate all configurations can be resolved without errors.

        Returns
        -------
        dict
            Maps config_id to error message for any configs that fail.
            Empty dict if all configs are valid.
        """
        errors = {}
        for config_id in self.configs:
            try:
                self.get(config_id)
            except Exception as e:
                errors[config_id] = str(e)

        return errors

    def _invalidate_cache(self):
        """Clear the resolution cache."""
        self._cache.clear()

    def __repr__(self):
        return f"ConfigRegistry({len(self.configs)} configs)"

    def __len__(self):
        return len(self.configs)

    def __contains__(self, config_id):
        return config_id in self.configs

    def __iter__(self):
        """Iterate over config IDs."""
        return iter(self.configs)


# Convenience function for quick usage
def create_registry(configs: Dict, **kwargs) -> ConfigRegistry:
    """
    Create a ConfigRegistry from a dict.

    This is just a convenience wrapper around ConfigRegistry.__init__().

    Parameters
    ----------
    configs : dict
        Configuration dictionary
    **kwargs
        Additional arguments passed to ConfigRegistry

    Returns
    -------
    ConfigRegistry
        New registry instance
    """
    return ConfigRegistry(configs, **kwargs)


if __name__ == '__main__':
    # Example usage
    print("ConfigRegistry - Example Usage")
    print("="*70)

    # Define some example configs
    EXAMPLE_CONFIGS = {
        'baseline': {
            'name': 'Baseline Model',
            'parameters': {
                'alpha': 0.5,
                'beta': 0.9,
                'gamma': 0.1,
            },
            'flags': {
                'use_feature_x': True,
                'use_feature_y': False,
            },
        },
        'variant_a': {
            'name': 'Variant A - Higher Alpha',
            'inherits_from': 'baseline',
            'parameters': {
                'alpha': 0.7,  # Override alpha
            },
        },
        'variant_b': {
            'name': 'Variant B - Enable Feature Y',
            'inherits_from': 'baseline',
            'flags': {
                'use_feature_y': True,  # Override flag
            },
        },
        'variant_c': {
            'name': 'Variant C - Based on A',
            'inherits_from': 'variant_a',  # Inherit from variant_a
            'parameters': {
                'gamma': 0.2,  # Add new parameter
            },
        },
    }

    # Create registry
    registry = ConfigRegistry(EXAMPLE_CONFIGS)

    # List all configs
    print("\nAvailable configurations:")
    for config_id, name in registry.list():
        print(f"  {config_id:15s} - {name}")

    # Print a config
    registry.print('variant_c')

    # Compare two configs
    registry.compare('baseline', 'variant_a')

    # Validate all
    errors = registry.validate_all()
    if not errors:
        print("✓ All configurations validated successfully")
    else:
        print(f"✗ Found errors in {len(errors)} config(s)")
