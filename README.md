# AShare Daily Picker

部署目标：Cloudflare Pages + GitHub Actions 每日自动选股。

默认主策略：**主板双高猎手 T2 V1 - 每日Top1版（2仓位）**

网页同时展示 3 个策略：

1. **主板双高猎手 T2 V1 - 每日Top1版（2仓位回测）**
2. **主板双高猎手 T2 V1 - 高阈值版**
3. **主板胜率猎手 V1 - H4高胜率版**

- T 日收盘后用 AkShare 获取当天 A 股收盘数据
- 复用仓库中的最近滚动行情 `data/rolling_ohlcv.parquet`
- 加载 LightGBM 模型：`models/win_classifier.txt`、`models/return_regressor.txt`、`models/win_h4.txt`
- 计算全市场主板 00/60 股票特征和分数
- 输出多策略当天 Top10 到 `public/latest.json`
- 若下一交易日数据已入库，会自动给历史 Top10 回填 `1日涨跌`
- 网页价格列优先展示 AkShare 原始未复权行情价，和普通行情软件看到的价格一致；模型内部仍用连续复权价做特征
- 发送邮件通知
- GitHub Actions 自动提交数据和结果
- Cloudflare Pages 部署静态网站

## 云端定时执行确认

GitHub Actions 工作流 `.github/workflows/daily-pick.yml` 已包含完整云端执行链路：

- `schedule`: 周一到周五北京时间 16:45 自动运行
- 云端安装 `requirements.txt`
- 云端拉取 AkShare 当天收盘数据
- 云端加载仓库内 LightGBM 模型打分
- 云端校验 `public/latest.json` 必须包含多个策略且价格类型为 `raw_unadjusted`
- 云端提交新增数据和选股结果
- 云端部署 Cloudflare Pages 到 `https://a.292828.xyz`

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

# 手动回填指定交易日，例如拉取 2026-06-04 并更新 2026-06-03 的 1日涨跌
python scripts/daily_pick.py --fetch-akshare --date 2026-06-04 --force --no-email
```

历史选股可用 `/?date=YYYY-MM-DD` 查看，例如 `/?date=2026-06-03`。

## Cloudflare Pages

- Project name: `ashare-daily-picker`
- Build command: 留空
- Build output directory: `public`
- Custom domain: `a.292828.xyz`

若使用 GitHub Actions 部署，workflow 会执行：

```bash
npx wrangler pages deploy public --project-name=ashare-daily-picker
```
