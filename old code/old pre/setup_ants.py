import os
import sys
from pathlib import Path


def setup_ants():
    """设置ANTs环境变量"""

    # 获取环境目录
    env_dir = Path(sys.prefix)

    # ANTs目录（根据您复制的实际位置）
    ants_paths = [
        env_dir / "ants-2.6.3",  # 如果复制了完整文件夹
        env_dir / "ants",  # 如果重命名了文件夹
        env_dir / "ANTs",  # 另一种可能
    ]

    ants_dir = None
    for path in ants_paths:
        if path.exists():
            ants_dir = path
            break

    if not ants_dir:
        print("未找到ANTs目录，请确保已复制到环境目录")
        print("环境目录:", env_dir)
        return False

    # Bin目录
    bin_dir = ants_dir / "bin"

    if not bin_dir.exists():
        print(f"错误: bin目录不存在: {bin_dir}")
        return False

    # 添加到PATH
    original_path = os.environ.get('PATH', '')
    new_path = str(bin_dir) + ';' + original_path
    os.environ['PATH'] = new_path

    # 设置ANTSPATH环境变量
    os.environ['ANTSPATH'] = str(ants_dir)

    print(f"✓ ANTs环境已设置")
    print(f"  ANTs目录: {ants_dir}")
    print(f"  Bin目录: {bin_dir}")
    print(f"  已添加到PATH")

    return True


if __name__ == "__main__":
    setup_ants()