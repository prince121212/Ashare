# AShare Daily Picker

部署目标：Cloudflare Pages + GitHub Actions 每日自动选股。

策略：**主板双高猎手 T2 V1 - 每日Top1版（2仓位）**

- T 日收盘后用 AkShare 获取当天 A 股收盘数据
- 复用仓库中的最近滚动行情 `data/rolling_ohlcv.parquet`
- 加载 LightGBM 模型：`models/win_classifier.txt`、`models/return_regressor.txt`
- 计算全市场主板 00/60 股票特征和分数
- 输出当天 Top10 到 `public/latest.json`
- 发送邮件通知
- GitHub Actions 自动提交数据和结果
- Cloudflare Pages 部署静态网站

## 必需 Secrets

在 GitHub 仓库 Secrets 中配置：

```text
RESEND_API_KEY              # 复用 wm985 的 Resend API Key
PICK_NOTIFY_EMAIL           # 接收选股邮件的邮箱
RESEND_FROM_EMAIL           # 默认 noreply@292828.xyz
RESEND_FROM_NAME            # 默认 A股每日选股
CLOUDFLARE_API_TOKEN        # Cloudflare Pages 部署 token
CLOUDFLARE_ACCOUNT_ID       # Cloudflare Account ID
```

可选：

```text
SITE_URL=https://a.292828.xyz
```

## 本地运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/daily_pick.py --no-fetch --no-email
```

## Cloudflare Pages

- Project name: `ashare-daily-picker`
- Build command: 留空
- Build output directory: `public`
- Custom domain: `a.292828.xyz`

若使用 GitHub Actions 部署，workflow 会执行：

```bash
npx wrangler pages deploy public --project-name=ashare-daily-picker
```
