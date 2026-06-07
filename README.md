# AShare Daily Picker

部署目标：Cloudflare Pages + GitHub Actions 每日自动选股。

默认主策略：**主板双高猎手 T2 V1 - 每日Top1版（2仓位）**

网页同时展示 4 个策略：

1. **主板双高猎手 T2 V1 - 每日Top1版（2仓位回测）**
2. **主板双高猎手 T2 V1 - 高阈值版**
3. **主板胜率猎手 V1 - H4高胜率版**
4. **V1.2 收盘选股版**：由 `/Users/huangchangwei/Desktop/gitSpaceC/qlib/策略库/config_V1.2.json` 改造而来，取消原本 T+1 开盘 gap 择机触发，改为 T 日收盘后直接输出候选股。

- T 日收盘后用 AkShare 获取当天 A 股收盘数据
- 复用仓库中的最近滚动行情 `data/rolling_ohlcv.parquet`
- 加载 LightGBM 模型：`models/win_classifier.txt`、`models/return_regressor.txt`、`models/win_h4.txt`
- 计算全市场主板 00/60 股票特征和分数
- 输出多策略当天候选到 `public/latest.json`（前三个策略 Top10，V1.2 收盘选股版 Top15）
- 若下一交易日数据已入库，会自动给历史 Top10 回填 `1日涨跌`
- 网页价格列优先展示 AkShare 原始未复权行情价，和普通行情软件看到的价格一致；模型内部仍用连续复权价做特征
- 发送邮件通知
- GitHub Actions 自动提交数据和结果
- Cloudflare Pages 部署静态网站
- 首页是静态策略看板：各策略首选、策略切换、分数/成交额图表、完整候选表格和移动端卡片展示
- 首页支持日期选择：读取 `public/history_index.json`，可切换到历史选股日查看当日候选和已回填的 `1日涨跌`

## V1.2 收盘选股版改造说明

原 V1.2 是“选股 + 择机买入”的事件策略：T 日收盘计算 `score_v03a`，但还要等 T+1 开盘 gap 满足条件后才触发买入。为了把它放到网站作为收盘后选股环节，当前版本只保留 T 日收盘已知的信息：

- 分数：`score_v03a_like`
- 范围：A 股主板 00/60，剔除 ST/退、停牌/零成交
- 排名：按 V03A 分数从高到低排序，跳过前 3 名
- 流动性：20 日平均成交额不低于 3000 万
- 价格：收盘价不低于 2 元
- 排除：`16 <= V03A排名 <= 30` 且当日振幅 `> 8.5%`
- 输出：收盘后 Top15 候选

注意：网站上 V1.2 卡片里的历史指标暂时引用原事件策略回测作为规则来源参考；取消开盘 gap 后的“纯收盘选股版”还没有单独重跑回测。

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

只修改页面样式、不需要重新拉行情时，可以手动触发 GitHub Actions 并设置：

```text
deploy_only=true
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
