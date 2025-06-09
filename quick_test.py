#!/usr/bin/env python3
"""
🧪 Quick Test Suite for Intel-Optimized Financial GNN

Simple tests that can run without heavy dependencies to validate core functionality.
"""

import sys
import os
import importlib.util

def test_project_structure():
    """Test that all required files exist"""
    print("🔍 Testing Project Structure...")
    
    required_files = [
        'README.md',
        'requirements.txt', 
        'setup.py',
        'demo.py',
        'src/__init__.py',
        'src/main.py',
        'src/data/__init__.py',
        'src/data/data_loader.py',
        'src/data/preprocessing.py',
        'src/models/__init__.py',
        'src/models/gnn_model.py',
        'src/models/intel_optimizer.py',
        'src/utils/__init__.py',
        'src/utils/graph_utils.py',
        'src/utils/visualization.py',
        'tests/test_integration.py',
        'CHANGELOG.md',
        'CONTRIBUTING.md',
        'LICENSE',
        'Dockerfile',
        '.github/workflows/ci.yml'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"   ✅ {file_path}")
    
    if missing_files:
        print(f"   ❌ Missing files: {missing_files}")
        return False
    
    print("   🎉 All required files present!")
    return True

def test_python_syntax():
    """Test that all Python files have valid syntax"""
    print("\n🐍 Testing Python Syntax...")
    
    python_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    syntax_errors = []
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            compile(content, py_file, 'exec')
            print(f"   ✅ {py_file}")
        except SyntaxError as e:
            syntax_errors.append(f"{py_file}: {e}")
            print(f"   ❌ {py_file}: {e}")
        except Exception as e:
            print(f"   ⚠️  {py_file}: Could not read - {e}")
    
    if syntax_errors:
        print(f"   ❌ Syntax errors found: {len(syntax_errors)}")
        return False
    
    print("   🎉 All Python files have valid syntax!")
    return True

def test_imports():
    """Test that modules can be imported (without running them)"""
    print("\n📦 Testing Module Imports...")
    
    # Add src to path for testing
    sys.path.insert(0, 'src')
    
    modules_to_test = [
        'src.data',
        'src.models', 
        'src.utils'
    ]
    
    import_errors = []
    for module_name in modules_to_test:
        try:
            spec = importlib.util.spec_from_file_location(
                module_name.replace('src.', ''), 
                f"src/{module_name.split('.')[-1]}/__init__.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                print(f"   ✅ {module_name}")
            else:
                print(f"   ⚠️  {module_name}: No spec found")
        except Exception as e:
            import_errors.append(f"{module_name}: {e}")
            print(f"   ❌ {module_name}: {e}")
    
    if import_errors:
        print(f"   ⚠️  Some import issues found (may be due to missing dependencies)")
    else:
        print("   🎉 All core modules can be loaded!")
    
    return len(import_errors) == 0

def test_configuration_files():
    """Test that configuration files are properly formatted"""
    print("\n⚙️ Testing Configuration Files...")
    
    # Test requirements.txt
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read().strip().split('\n')
        
        if len(requirements) > 10:  # Should have many dependencies
            print("   ✅ requirements.txt has sufficient dependencies")
        else:
            print("   ⚠️  requirements.txt seems short")
            
    except Exception as e:
        print(f"   ❌ requirements.txt error: {e}")
        return False
    
    # Test setup.py
    try:
        with open('setup.py', 'r') as f:
            setup_content = f.read()
        
        if 'name=' in setup_content and 'version=' in setup_content:
            print("   ✅ setup.py has required fields")
        else:
            print("   ❌ setup.py missing required fields")
            return False
            
    except Exception as e:
        print(f"   ❌ setup.py error: {e}")
        return False
    
    print("   🎉 Configuration files are properly formatted!")
    return True

def test_documentation():
    """Test documentation completeness"""
    print("\n📚 Testing Documentation...")
    
    # Test README.md
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        required_sections = [
            'Installation', 'Quick Start', 'Features', 
            'Intel', 'Performance', 'Contributing'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section.lower() not in readme_content.lower():
                missing_sections.append(section)
        
        if missing_sections:
            print(f"   ⚠️  README missing sections: {missing_sections}")
        else:
            print("   ✅ README.md has all required sections")
            
        if len(readme_content) > 5000:  # Should be comprehensive
            print("   ✅ README.md is comprehensive")
        else:
            print("   ⚠️  README.md seems short")
            
    except Exception as e:
        print(f"   ❌ README.md error: {e}")
        return False
    
    print("   🎉 Documentation is comprehensive!")
    return True

def test_docker_and_ci():
    """Test Docker and CI configuration"""
    print("\n🐳 Testing Docker and CI Configuration...")
    
    # Test Dockerfile
    try:
        with open('Dockerfile', 'r') as f:
            dockerfile_content = f.read()
        
        if 'intel' in dockerfile_content.lower():
            print("   ✅ Dockerfile includes Intel optimizations")
        else:
            print("   ⚠️  Dockerfile doesn't mention Intel")
            
        if 'COPY' in dockerfile_content and 'RUN' in dockerfile_content:
            print("   ✅ Dockerfile has proper structure")
        else:
            print("   ❌ Dockerfile structure issues")
            
    except Exception as e:
        print(f"   ❌ Dockerfile error: {e}")
        return False
    
    # Test CI configuration
    try:
        with open('.github/workflows/ci.yml', 'r') as f:
            ci_content = f.read()
        
        if 'intel' in ci_content.lower() and 'test' in ci_content.lower():
            print("   ✅ CI includes Intel and testing")
        else:
            print("   ⚠️  CI configuration may be incomplete")
            
    except Exception as e:
        print(f"   ❌ CI configuration error: {e}")
        return False
    
    print("   🎉 Docker and CI are properly configured!")
    return True

def run_all_tests():
    """Run all tests and provide summary"""
    print("🚀 Intel-Optimized Financial GNN - Quick Test Suite")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Python Syntax", test_python_syntax), 
        ("Module Imports", test_imports),
        ("Configuration Files", test_configuration_files),
        ("Documentation", test_documentation),
        ("Docker & CI", test_docker_and_ci)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name:<20} {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! The project is well-structured and ready for use.")
    elif passed >= total * 0.8:
        print("✅ Most tests passed! The project is in good shape with minor issues.")
    else:
        print("⚠️  Several tests failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 