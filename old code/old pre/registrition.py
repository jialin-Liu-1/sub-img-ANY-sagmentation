# final_register_mri.py
import os
import sys
import subprocess
import shutil
import tempfile
import time
from pathlib import Path


class AntsRegistrar:
    """ANTs配准器 - 使用完整路径确保可靠"""

    def __init__(self):
        # ANTs完整路径
        self.ants_base = Path(r"C:\Anaconda\envs\ai\ants-2.6.3")
        self.ants_bin = self.ants_base / "bin"
        self.ants_exe = self.ants_bin / "antsRegistration.exe"

        print("=" * 60)
        print("ANTs MRI图像配准器")
        print("=" * 60)
        print(f"ANTs路径: {self.ants_base}")
        print(f"主程序: {self.ants_exe}")

        # 验证ANTs
        if not self._verify_ants():
            raise RuntimeError("ANTs验证失败")

    def _verify_ants(self):
        """验证ANTs是否可用"""
        if not self.ants_exe.exists():
            print(f"错误: 找不到ANTs主程序: {self.ants_exe}")
            return False

        # 测试命令
        try:
            result = subprocess.run(
                [str(self.ants_exe), "--version"],
                capture_output=True,
                text=True,
                shell=True,
                timeout=10
            )

            if result.returncode == 0 and "ANTs Version" in result.stdout:
                print(f"✓ ANTs验证通过: {result.stdout.strip()}")
                return True
            else:
                print(f"ANTs测试失败: {result.stderr}")
                return False

        except Exception as e:
            print(f"ANTs测试异常: {e}")
            return False

    def find_image_pairs(self, t1_dir, t2_dir):
        """查找匹配的T1-T2图像对"""
        t1_dir = Path(t1_dir)
        t2_dir = Path(t2_dir)

        pairs = []

        # 查找所有T1文件
        for t1_file in t1_dir.glob("*.nii.gz"):
            # 提取病例ID (支持多种命名格式)
            stem = t1_file.stem.replace('.nii', '')

            case_id = None
            patterns = ['_T1', '-T1', '_t1', '-t1', 'T1', 't1']

            for pattern in patterns:
                if pattern in stem:
                    case_id = stem.split(pattern)[0]
                    break

            if case_id is None:
                case_id = stem

            # 查找对应的T2文件
            t2_patterns = [
                t2_dir / f"{case_id}_T2.nii.gz",
                t2_dir / f"{case_id}-T2.nii.gz",
                t2_dir / f"{case_id}_t2.nii.gz",
                t2_dir / f"{case_id}T2.nii.gz",
            ]

            t2_file = None
            for pattern in t2_patterns:
                if pattern.exists():
                    t2_file = pattern
                    break

            if t2_file and t1_file != t2_file:
                pairs.append({
                    'case_id': case_id,
                    't1': t1_file,
                    't2': t2_file
                })

        # 按病例ID排序
        pairs.sort(key=lambda x: x['case_id'])

        return pairs

    def rigid_register(self, fixed_image, moving_image, output_prefix):
        """执行刚性配准"""

        cmd = [
            str(self.ants_exe),
            "--dimensionality", "3",
            "--float", "0",
            "--interpolation", "Linear",
            "--transform", "Rigid[0.1]",
            "--metric", f"MI[{fixed_image},{moving_image},1,32,Regular,0.25]",
            "--convergence", "[1000x500x250x100,1e-6,10]",
            "--shrink-factors", "8x4x2x1",
            "--smoothing-sigmas", "3x2x1x0vox",
            "--output", f"[{output_prefix},{output_prefix}warped.nii.gz]",
            "--initial-moving-transform", f"[{fixed_image},{moving_image},1]",
            "--verbose", "0"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                shell=True,
                timeout=600  # 10分钟超时
            )

            return result

        except subprocess.TimeoutExpired:
            print("配准超时")
            return None
        except Exception as e:
            print(f"配准异常: {e}")
            return None

    def process_case(self, case_info, output_dir, skip_existing=True):
        """处理单个病例"""
        case_id = case_info['case_id']
        t1_path = case_info['t1']
        t2_path = case_info['t2']

        print(f"\n处理病例: {case_id}")
        print(f"  T1: {t1_path.name}")
        print(f"  T2: {t2_path.name}")

        # 创建输出目录
        output_t1_dir = Path(output_dir) / "t1"
        output_t2_dir = Path(output_dir) / "t2"
        output_t1_dir.mkdir(parents=True, exist_ok=True)
        output_t2_dir.mkdir(parents=True, exist_ok=True)

        # 检查是否已存在
        output_t1 = output_t1_dir / f"{case_id}_T1.nii.gz"
        output_t2 = output_t2_dir / f"{case_id}_T2.nii.gz"

        if skip_existing and output_t1.exists() and output_t2.exists():
            print(f"  ⏩ 已存在，跳过")
            return True, "skipped"

        # 创建临时目录
        temp_dir = Path(tempfile.mkdtemp(prefix=f"ants_reg_{case_id}_"))

        try:
            start_time = time.time()

            # 执行配准
            result = self.rigid_register(
                fixed_image=str(t1_path),
                moving_image=str(t2_path),
                output_prefix=str(temp_dir / f"{case_id}_")
            )

            if result is None:
                return False, "timeout_or_error"

            elapsed_time = time.time() - start_time

            # 检查结果
            if result.returncode != 0:
                print(f"  ✗ 配准失败，返回码: {result.returncode}")
                if result.stderr:
                    error_msg = result.stderr[:300]
                    print(f"    错误: {error_msg}")
                return False, "registration_failed"

            # 检查输出文件
            warped_file = temp_dir / f"{case_id}_warped.nii.gz"
            if not warped_file.exists():
                print(f"  ✗ 输出文件未生成")
                return False, "no_output"

            # 复制结果
            shutil.copy2(t1_path, output_t1)
            shutil.copy2(warped_file, output_t2)

            print(f"  ✓ 配准成功 ({elapsed_time:.1f}秒)")
            print(f"    输出: {output_t1.name}, {output_t2.name}")

            return True, "success"

        except Exception as e:
            print(f"  ✗ 处理异常: {e}")
            return False, "exception"

        finally:
            # 清理临时文件
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def batch_process(self, t1_dir, t2_dir, output_dir, skip_existing=True):
        """批量处理所有图像对"""

        print(f"\n查找图像对...")
        pairs = self.find_image_pairs(t1_dir, t2_dir)

        if not pairs:
            print("未找到匹配的图像对")
            return

        print(f"找到 {len(pairs)} 个图像对")

        # 显示前几个
        for i, pair in enumerate(pairs[:5]):
            print(f"  {i + 1}. {pair['case_id']}: {pair['t1'].name} ↔ {pair['t2'].name}")
        if len(pairs) > 5:
            print(f"  ... 还有 {len(pairs) - 5} 个")

        # 开始处理
        print(f"\n开始配准处理...")
        print("=" * 60)

        stats = {
            'total': len(pairs),
            'success': 0,
            'skipped': 0,
            'failed': 0,
            'failed_cases': []
        }

        for i, pair in enumerate(pairs):
            print(f"\n[{i + 1}/{len(pairs)}] ", end="")

            success, status = self.process_case(pair, output_dir, skip_existing)

            if success:
                if status == "success":
                    stats['success'] += 1
                elif status == "skipped":
                    stats['skipped'] += 1
            else:
                stats['failed'] += 1
                stats['failed_cases'].append({
                    'case_id': pair['case_id'],
                    'status': status
                })

        # 输出统计
        print("\n" + "=" * 60)
        print("配准完成统计:")
        print(f"总病例数: {stats['total']}")
        print(f"成功配准: {stats['success']}")
        print(f"跳过已存在: {stats['skipped']}")
        print(f"失败: {stats['failed']}")

        if stats['failed_cases']:
            print(f"\n失败病例:")
            for fail in stats['failed_cases']:
                print(f"  - {fail['case_id']}: {fail['status']}")

        print(f"\n输出目录:")
        print(f"  T1图像: {Path(output_dir) / '1'}")
        print(f"  T2图像: {Path(output_dir) / '2'}")

        return stats


def main():
    """主函数"""

    # 配置路径
    t1_dir = r"D:\med_data\MR\TEST1"
    t2_dir = r"D:\med_data\MR\TEST2"
    output_dir = r"D:\med_data\MR\T12W2"

    print("\n配置信息:")
    print(f"T1目录: {t1_dir}")
    print(f"T2目录: {t2_dir}")
    print(f"输出目录: {output_dir}")

    # 验证输入目录
    if not os.path.exists(t1_dir):
        print(f"错误: T1目录不存在: {t1_dir}")
        return

    if not os.path.exists(t2_dir):
        print(f"错误: T2目录不存在: {t2_dir}")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 初始化配准器
        registrar = AntsRegistrar()

        # 用户确认
        print("\n" + "=" * 60)
        response = input("是否开始配准? (y/n): ").strip().lower()

        if response != 'y':
            print("用户取消")
            return

        # 执行配准
        stats = registrar.batch_process(
            t1_dir=t1_dir,
            t2_dir=t2_dir,
            output_dir=output_dir,
            skip_existing=True
        )

        print("\n" + "=" * 60)
        print("处理完成!")

        if stats['success'] > 0:
            print(f"成功配准了 {stats['success']} 个病例")
            print("配准后的图像已保存到输出目录")

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n程序异常: {e}")
        import traceback
        traceback.print_exc()


# 快速测试函数
def quick_test():
    """快速测试ANTs是否工作"""
    print("快速测试ANTs...")

    ants_exe = r"C:\Anaconda\envs\ai\ants-2.6.3\bin\antsRegistration.exe"

    if not os.path.exists(ants_exe):
        print(f"错误: 找不到ANTs: {ants_exe}")
        return False

    try:
        result = subprocess.run(
            [ants_exe, "--version"],
            capture_output=True,
            text=True,
            shell=True
        )

        print(f"ANTs版本: {result.stdout.strip()}")
        print("✓ ANTs工作正常!")
        return True

    except Exception as e:
        print(f"测试失败: {e}")
        return False


if __name__ == "__main__":
    # 先快速测试
    if quick_test():
        # 运行主程序
        main()
    else:
        print("ANTs测试失败，请检查安装")