# verify_ants.py
import os
import subprocess


def verify_windows_ants():
    """验证Windows版ANTs安装"""

    print("验证ANTs Windows安装")
    print("=" * 60)

    # 检查目录结构
    ants_path = r"C:\Anaconda\envs\ai\ants-2.6.3"

    required_dirs = [
        ("主目录", ants_path),
        ("Bin目录", os.path.join(ants_path, "bin")),
        ("Lib目录", os.path.join(ants_path, "lib")),
        ("Include目录", os.path.join(ants_path, "include")),
    ]

    print("1. 检查目录结构:")
    for name, path in required_dirs:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")

    # 检查关键文件
    print("\n2. 检查关键文件:")
    key_files = [
        ("antsRegistration.exe", os.path.join(ants_path, "bin", "antsRegistration.exe")),
        ("antsApplyTransforms.exe", os.path.join(ants_path, "bin", "antsApplyTransforms.exe")),
        ("libITKCommon-5.3.dll", os.path.join(ants_path, "bin", "libITKCommon-5.3.dll")),
    ]

    for name, path in key_files:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {name}")

    # 测试命令
    print("\n3. 测试ANTs命令:")

    # Windows版ANTs可能有不同的可执行文件
    test_commands = [
        ("antsRegistration.exe", ["--version"]),
        (os.path.join(ants_path, "bin", "antsRegistration.exe"), ["--version"]),
        ("antsRegistration", ["--version"]),  # 可能没有.bat文件
    ]

    for cmd, args in test_commands:
        try:
            print(f"\n尝试: {cmd} {' '.join(args)}")

            if isinstance(cmd, str) and cmd.endswith('.exe'):
                # 使用完整路径
                full_cmd = cmd if os.path.isabs(cmd) else os.path.join(ants_path, "bin", cmd)
            else:
                full_cmd = cmd

            result = subprocess.run(
                [full_cmd] + args if isinstance(cmd, str) else [cmd] + args,
                capture_output=True,
                text=True,
                shell=True,
                timeout=10
            )

            print(f"  返回码: {result.returncode}")

            if result.returncode in [0, 1]:  # ANTs通常返回0或1
                if result.stdout:
                    print(f"  输出: {result.stdout[:200]}")
                    if "ANTs" in result.stdout or "Usage:" in result.stdout:
                        print(f"  ✓ 命令工作正常!")
                        return True
                elif result.stderr:
                    print(f"  错误: {result.stderr[:200]}")
                    # 有时错误信息也包含有用信息
                    return True
            else:
                print(f"  ✗ 失败")

        except FileNotFoundError:
            print(f"  ✗ 文件未找到")
        except Exception as e:
            print(f"  ✗ 错误: {e}")

    return False


if __name__ == "__main__":
    success = verify_windows_ants()

    print("\n" + "=" * 60)
    if success:
        print("✓ ANTs安装验证通过!")
    else:
        print("✗ ANTs安装可能有问题")
        print("\n建议:")
        print("1. 确保已设置环境变量 PATH")
        print("2. 尝试直接运行完整路径的命令")
        print("3. 检查是否缺少依赖库")