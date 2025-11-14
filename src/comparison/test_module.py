"""Simple test script to verify the comparison module is properly set up."""

import sys
from typing import List


def test_imports() -> List[str]:
    """Test that all comparison module components can be imported.
    
    Returns:
        List of any import errors (empty if all successful)
    """
    errors = []
    
    # Test agent_loader imports
    try:
        from . import (
            get_agent_configs,
            create_agent,
            save_agent,
            load_agent,
            train_agent,
            load_or_train_agents,
        )
        print("✓ agent_loader imports successful")
    except ImportError as e:
        errors.append(f"agent_loader import failed: {e}")
        print(f"✗ agent_loader import failed: {e}")
    
    # Test runner imports
    try:
        from . import (
            evaluate_agent,
            get_agent_statistics,
            run_comparison,
            print_comparison_summary,
            save_comparison_report,
        )
        print("✓ runner imports successful")
    except ImportError as e:
        errors.append(f"runner import failed: {e}")
        print(f"✗ runner import failed: {e}")
    
    # Test visualization imports
    try:
        from . import (
            plot_q_values_3d,
            plot_epsilon_values_3d,
            plot_win_rate_comparison,
            plot_training_performance,
            plot_exploration_statistics,
            plot_policy_heatmap,
            plot_average_reward_comparison,
            generate_all_visualizations,
        )
        print("✓ visualizations imports successful")
    except ImportError as e:
        errors.append(f"visualizations import failed: {e}")
        print(f"✗ visualizations import failed: {e}")
    
    # Test main entry point
    try:
        from . import compare_agents_main
        print("✓ compare_agents imports successful")
    except ImportError as e:
        errors.append(f"compare_agents import failed: {e}")
        print(f"✗ compare_agents import failed: {e}")
    
    return errors


def test_agent_configs() -> List[str]:
    """Test that agent configurations are properly defined.
    
    Returns:
        List of any configuration errors (empty if all successful)
    """
    errors = []
    
    try:
        from .agent_loader import get_agent_configs
        
        configs = get_agent_configs()
        expected_agents = ["simple", "count_based", "ucb", "threshold"]
        
        for agent_type in expected_agents:
            if agent_type not in configs:
                errors.append(f"Missing agent configuration: {agent_type}")
                print(f"✗ Missing config for {agent_type}")
            else:
                config = configs[agent_type]
                if "name" not in config or "class" not in config or "params" not in config:
                    errors.append(f"Incomplete configuration for {agent_type}")
                    print(f"✗ Incomplete config for {agent_type}")
                else:
                    print(f"✓ Config for {agent_type}: {config['name']}")
        
        if not errors:
            print(f"\n✓ All {len(expected_agents)} agent configurations valid")
    
    except Exception as e:
        errors.append(f"Agent config test failed: {e}")
        print(f"✗ Agent config test failed: {e}")
    
    return errors


def run_tests() -> bool:
    """Run all module tests.
    
    Returns:
        True if all tests pass, False otherwise
    """
    print("\n" + "="*80)
    print(" " * 25 + "COMPARISON MODULE TESTS")
    print("="*80 + "\n")
    
    all_errors = []
    
    # Test imports
    print("Testing imports...")
    print("-" * 80)
    import_errors = test_imports()
    all_errors.extend(import_errors)
    
    # Test configurations
    print("\n" + "-" * 80)
    print("Testing agent configurations...")
    print("-" * 80)
    config_errors = test_agent_configs()
    all_errors.extend(config_errors)
    
    # Summary
    print("\n" + "="*80)
    if all_errors:
        print("❌ TESTS FAILED")
        print("="*80 + "\n")
        print("Errors:")
        for error in all_errors:
            print(f"  - {error}")
        print()
        return False
    else:
        print("✅ ALL TESTS PASSED")
        print("="*80 + "\n")
        return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

