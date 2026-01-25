import re
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("No commit message file passed.")
        return 1
    msg_file = Path(sys.argv[1])
    content = msg_file.read_text(encoding="utf-8")

    # 是否包含 BREAKING 标记（建议在 Body 首行）
    has_breaking = "[🔴 BREAKING]" in content
    if not has_breaking:
        return 0  # 无需校验 EXP

    # Footer 必须包含：EXP: #数字 或 #数字~#数字，多个用逗号分隔
    exp_pattern = re.compile(r"^EXP:\s*#\d+(?:\s*~\s*#\d+)?(?:\s*,\s*#\d+)*\s*$", re.MULTILINE)
    if exp_pattern.search(content):
        return 0

    print(
        "Error: 带 [🔴 BREAKING] 的提交必须在 Footer 中包含 EXP 脚注（格式：EXP: #数字 或 EXP: #数字~#数字，多个用逗号分隔）"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
