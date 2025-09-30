#!/usr/bin/env python3
"""
最小化测试脚本 - 不依赖 FastAPI
Author: mmwei3, mmwei3@iflytek.com, 1300042631@qq.com
Weather: Cloudy, Date: 2025-08-27
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# 添加项目路径
backend_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(backend_dir))


def test_basic_imports():
    """测试基础模块导入"""
    print("🔍 测试基础模块导入...")

    try:
        import json

        print("✅ json 模块导入成功")
    except Exception as e:
        print(f"❌ json 模块导入失败: {e}")
        return False

    try:
        from datetime import datetime

        print("✅ datetime 模块导入成功")
    except Exception as e:
        print(f"❌ datetime 模块导入失败: {e}")
        return False

    try:
        import logging

        print("✅ logging 模块导入成功")
    except Exception as e:
        print(f"❌ logging 模块导入失败: {e}")
        return False

    return True


def test_json_store_basic():
    """测试 JSON 存储的基本功能（不依赖 FastAPI）"""
    print("\n🗄️ 测试 JSON 存储基本功能...")

    try:
        # 手动实现简单的 JSON 存储测试
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        # 测试文件写入
        test_data = {
            "users": [],
            "tasks": [],
            "models": [],
            "created_at": datetime.now().isoformat(),
        }

        with open(data_dir / "test.json", "w") as f:
            json.dump(test_data, f, indent=2)
        print("✅ JSON 文件写入成功")

        # 测试文件读取
        with open(data_dir / "test.json", "r") as f:
            loaded_data = json.load(f)
        print("✅ JSON 文件读取成功")

        # 清理测试文件
        (data_dir / "test.json").unlink()
        print("✅ 测试文件清理成功")

        return True
    except Exception as e:
        print(f"❌ JSON 存储测试失败: {e}")
        return False


def test_python_version():
    """测试 Python 版本兼容性"""
    print("\n🐍 测试 Python 版本兼容性...")

    version = sys.version_info
    print(f"   Python 版本: {version.major}.{version.minor}.{version.micro}")

    # 检查关键特性
    features = []

    # 检查 f-string 支持 (Python 3.6+)
    try:
        test_var = "test"
        f"Testing f-string: {test_var}"
        features.append("f-string")
    except:
        pass

    # 检查类型注解支持
    try:

        def test_func(x: int) -> str:
            return str(x)

        features.append("type hints")
    except:
        pass

    # 检查 asyncio 支持
    try:
        import asyncio

        features.append("asyncio")
    except:
        pass

    print(f"   支持的特性: {', '.join(features)}")

    if version.major >= 3 and version.minor >= 6:
        print("✅ Python 版本兼容")
        return True
    else:
        print("❌ Python 版本过低")
        return False


def test_file_structure():
    """测试文件结构"""
    print("\n📁 测试文件结构...")

    required_files = [
        "core/__init__.py",
        "core/app.py",
        "core/store.py",
        "core/forecast.py",
        "core/prometheus_api.py",
        "lstm/__init__.py",
        "lstm/model.py",
        "lstm/data_loader.py",
        "lstm/utils.py",
        "lstm/train.py",
        "lstm/predict.py",
    ]

    missing_files = []
    for file_path in required_files:
        full_path = backend_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)

    if missing_files:
        print(f"⚠️ 缺失文件: {missing_files}")
        return False
    else:
        print("✅ 所有必需文件存在")
        return True


def main():
    print("🧪 开始最小化测试...")
    print(f"工作目录: {os.getcwd()}")

    tests = [
        ("基础模块导入", test_basic_imports),
        ("JSON 存储基本功能", test_json_store_basic),
        ("Python 版本兼容性", test_python_version),
        ("文件结构", test_file_structure),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} 测试异常: {e}")
            results.append((name, False))

    print("\n📊 测试结果汇总:")
    passed = 0
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {name}: {status}")
        if result:
            passed += 1

    print(f"\n🎯 总体结果: {passed}/{len(results)} 个测试通过")

    if passed == len(results):
        print("🎉 基础测试通过！")
        print("\n🔧 下一步:")
        print(
            "1. 安装 FastAPI 依赖: pip3 install fastapi==0.68.2 uvicorn==0.15.0 pydantic==1.8.2"
        )
        print("2. 安装机器学习依赖: pip3 install torch==1.9.1 scikit-learn==0.24.2")
        print("3. 启动服务: python3 core/app.py")
        return True
    else:
        print("⚠️ 部分测试失败，请检查错误信息")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
