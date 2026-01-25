module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      ['feat', 'fix', 'docs', 'style', 'refactor', 'perf', 'test', 'chore', 'revert', 'merge']
    ],
    // 自定义规则：检测 [🔴 BREAKING] 提交是否包含 EXP 脚注
    'custom-footer-exp-required': [2, 'always']
  },
  // 注册自定义规则的实现
  plugins: [
    {
      rules: {
        'custom-footer-exp-required': (parsed, _when, _value) => {
          const { body, footer } = parsed;
          
          // 1. 判断提交内容是否包含 [🔴 BREAKING] 标记
          const hasBreakingTag = body && body.includes('[🔴 BREAKING]');
          if (!hasBreakingTag) {
            // 无 BREAKING 标记，无需校验 EXP 脚注
            return [true];
          }

          // 2. 有 BREAKING 标记时，校验 footer 是否包含 EXP: #数字 格式
          const expPattern = /EXP: #\d+(-#\d+)?(, #\d+)*$/m;
          const hasExpFooter = footer && expPattern.test(footer);
          
          if (hasExpFooter) {
            return [true];
          } else {
            return [
              false,
              '带 [🔴 BREAKING] 的提交必须在 footer 中添加 EXP 脚注（格式：EXP: #数字 或 EXP: #数字~#数字）'
            ];
          }
        }
      }
    }
  ]
};