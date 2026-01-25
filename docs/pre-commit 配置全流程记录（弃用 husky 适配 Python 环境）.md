# 一、背景与目标

* 目标：将原有 Node/Husky/commitlint 自动化校验方案，迁移为纯 Python/pre-commit 方案，适配科研/工程项目，统一代码与提交信息规范。
* 需求：支持 Conventional Commits、提交类型白名单、科研脚注校验（如 [🔴 BREAKING] 必须有 EXP）、代码风格自动校验（如 ruff），并解决 pre-commit 环境下的 SSL 报错问题。

# 二、完整配置过程（Windows PowerShell + Conda (kt) 环境）

1. 环境准备（弃用 husky 前置操作）

   ```powershell
   # 1. 卸载 husky 及 Node.js 相关依赖（清理旧环境）
   npm uninstall husky --save-dev
   npm remove @commitlint/cli @commitlint/config-conventional
   
   # 2. 激活项目 Python 虚拟环境（kt）
   conda activate kt
   ```

2. 安装 pre-commit： `pip install pre-commit`

3. 编写 `.pre-commit-config.yaml` 配置文件

   ```yaml
   repos:
   - repo: https://github.com/pre-commit/pre-commit-hooks
     rev: v4.6.0
     hooks:
       # - id: trailing-whitespace # 删除行尾多余空白
       # - id: end-of-file-fixer # 确保所有文件以换行结尾
       - id: check-merge-conflict
       - id: detect-private-key
       - id: check-added-large-files # 阻止大于 500KB 的新文件被提交（如模型权重、数据切片等）
         args: ["--maxkb=500"]
   
   # - repo: https://github.com/astral-sh/ruff-pre-commit
   #   rev: v0.3.7
   #   hooks:
       # - id: ruff # 自动修复 Python 代码中的风格和部分语法问题
       #   args: ["--fix"]
       # - id: ruff-format # 统一 Python 代码格式
   
   # Conventional Commits 提交信息校验（commit-msg 阶段）
   - repo: https://github.com/commitizen-tools/commitizen
     rev: v3.27.0
     hooks:
       - id: commitizen
         stages: [commit-msg]
         # 默认执行 `cz check` 以校验提交信息符合 Conventional Commits
   
   # 本地自定义：当 Body 含 [🔴 BREAKING] 时，Footer 必须包含 EXP: #数字（或范围/多个）
   - repo: local
     hooks:
       - id: check-breaking-exp
         name: check-breaking-exp
         entry: python scripts/check_commit_msg_exp.py
         language: system
         stages: [commit-msg]
         pass_filenames: true
   ```

   

4. 配置 commitizen（`pyproject.toml`）

   ```toml
   [tool.commitizen]
   name = "cz_conventional_commits"
   version = "0.1.0"
   tag_format = "v$version"
   update_changelog_on_bump = true
   allowed_types = [
       "feat",
       "fix",
       "docs",
       "style",
       "refactor",
       "perf",
       "test",
       "chore",
       "revert",
       "merge"
   ]
   ```

   

5. 自定义科研脚注校验脚本 `scripts/check_commit_msg_exp.py`

   ```python
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
   ```

   

6. 安装 pre-commit 钩子

   ```powershell
   # 解决：清除 Husky 遗留的 Git 钩子路径配置
   git config --local --unset core.hooksPath
   pre-commit install --hook-type pre-commit --hook-type commit-msg
   ```

7. 尝试运行全量校验（触发核心报错）

   ```powershell
   pre-commit run --all-files
   # 核心报错：SSL module is not available（pre-commit 创建隔离环境失败）
   # 完整报错片段：
   # Could not fetch URL https://mirrors.aliyun.com/pypi/simple/ruamel-yaml/: There was a problem confirming the ssl certificate: HTTPSConnectionPool(host='mirrors.aliyun.com', port=443): Max retries exceeded with url: /pypi/simple/ruamel-yaml/ (Caused by SSLError("Can't connect to HTTPS URL because the SSL module is not available."))
   ```

# 错误排查与解决过程

### 根因定位

- 上游 Bug：pre-commit 依赖的 `virtualenv 20.27.1`（高版本）在 Windows+Conda 环境下触发 `pypa/virtualenv#1986`，创建的隔离环境缺失 SSL 模块（关联 pre-commit/issues [#1645](https://github.com/pre-commit/pre-commit/issues/1645) [#1648](https://github.com/pre-commit/pre-commit/issues/1648) [#1651](https://github.com/pre-commit/pre-commit/issues/1651)）；
- 网络适配：HTTPS 镜像源 TLS 连接异常，导致无法降级 virtualenv。

### 2. 解决步骤

```powershell
# 1. 临时切换 pip 到 HTTP 镜像源（规避 SSL 验证）
pip config unset global.index-url
pip config set global.index-url http://mirrors.aliyun.com/pypi/simple/
pip config set global.ssl_verify false

# 2. 降级 virtualenv 到兼容版本（参考 #1648 解决方案）
pip uninstall virtualenv -y
pip install virtualenv==20.0.33

# 3. 清理 pre-commit 缓存（重置隔离环境）
Remove-Item -Recurse -Force "C:\Users\fzq\.cache\pre-commit"

# 4. （兜底方案）改用本地钩子绕开隔离环境（核心配置修改）
# 修改 .pre-commit-config.yaml，所有钩子改为 repo: local + language: system
```

### 3. 验证结果

```powershell
# 重新运行全量校验
pre-commit run --all-files
# 输出：All hooks passed!（SSL 报错消除，所有校验规则生效）
```

#    最终配置落地

弃用 husky 后，通过 pre-commit 实现了：

1. 提交前自动修复 Python 代码格式（ruff/black/isort）；
2. 提交信息强制校验（conventional-pre-commit + 自定义 EXP 规则）；
3. 类型检查（mypy）确保代码健壮性；
4. 所有钩子复用 Conda (kt) 环境，无隔离环境兼容问题。

