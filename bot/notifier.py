"""
Telegram 推送通知 — 格式化每日论文卡片
可由 GitHub Actions 或本地 cron 调用
"""
import json
import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

logger = logging.getLogger("paper-distill-notifier")


def format_paper_card(idx: int, paper: dict) -> str:
    """格式化单篇论文为 Telegram 消息卡片（精简版，详情看网站）"""
    title = paper.get("title", "未知标题")
    journal = paper.get("journal", "")
    year = paper.get("year", "")
    impact_factor = paper.get("impact_factor", "")
    cas_zone = paper.get("cas_zone", "")
    tldr = paper.get("tldr", "")
    inspiration = paper.get("inspiration", "") or paper.get("relevance_note", "")
    doi = paper.get("doi", "")

    # Build compact card
    lines = []

    # Title with number
    if doi:
        lines.append(f"*{idx}.* [{title}](https://doi.org/{doi})")
    else:
        lines.append(f"*{idx}. {title}*")

    # Journal info line (compact)
    meta = []
    if journal:
        meta.append(journal)
    if year:
        meta.append(str(year))
    if cas_zone:
        meta.append(cas_zone)
    if impact_factor:
        meta.append(f"IF {impact_factor}")
    if meta:
        lines.append(f"    📖 {' · '.join(meta)}")

    # TLDR
    if tldr:
        lines.append(f"    💡 {tldr}")

    # Inspiration (truncate to ~80 chars for readability)
    if inspiration:
        short = inspiration[:80] + "…" if len(inspiration) > 80 else inspiration
        lines.append(f"    🎯 {short}")

    return "\n".join(lines)


def format_daily_push(date: str, papers: list, site_url: str = "") -> str:
    """格式化完整的每日推送消息（紧凑版，保证单条发送）"""
    header = f"📬 *论文推送 {date} | {len(papers)} 篇*"

    sections = {"research": [], "ai_news": [], "other": []}

    for i, paper in enumerate(papers, 1):
        topics = paper.get("topic_tags", [])
        card = format_paper_card(i, paper)

        if any(t in topics for t in ["llm-reasoning", "rag-retrieval", "llm-agents", "code-generation", "prompt-engineering"]):
            sections["research"].append(card)
        elif any(t in topics for t in ["llm", "ai-news", "multimodal", "diffusion-models"]):
            sections["ai_news"].append(card)
        else:
            sections["other"].append(card)

    parts = [header]

    if sections["research"]:
        parts.append("\n🔬 *核心研究*")
        parts.extend(sections["research"])

    if sections["ai_news"]:
        parts.append("\n🤖 *AI 前沿*")
        parts.extend(sections["ai_news"])

    if sections["other"]:
        parts.append("\n📊 *其他方向*")
        parts.extend(sections["other"])

    parts.append("")
    if site_url:
        parts.append(f"🔗 [详细解读]({site_url})")

    parts.append("回复编号收藏 | 回复「全部」一键全收藏")

    return "\n".join(parts)


async def send_push(token: str, chat_id: str, message: str):
    """发送 Telegram 消息"""
    from telegram import Bot

    bot = Bot(token=token)
    # Split long messages (Telegram limit: 4096 chars)
    if len(message) <= 4096:
        await bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
    else:
        # Split at section boundaries
        chunks = []
        current = ""
        for line in message.split("\n"):
            if len(current) + len(line) + 1 > 4000:
                chunks.append(current)
                current = line
            else:
                current += "\n" + line if current else line
        if current:
            chunks.append(current)

        for chunk in chunks:
            await bot.send_message(
                chat_id=chat_id,
                text=chunk,
                parse_mode="Markdown",
                disable_web_page_preview=True,
            )
            await asyncio.sleep(0.5)  # Rate limit


async def main_push(date: str = None):
    """Main entry for push notification (called by cron/GitHub Actions)"""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    site_url = os.getenv("GITHUB_PAGES_URL", "")

    if not token or not chat_id:
        logger.error("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return

    project_root = Path(__file__).resolve().parent.parent

    if not date:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Load today's push data
    pushes_path = project_root / "data" / "pushes.jsonl"
    if not pushes_path.exists():
        logger.info("No pushes.jsonl found, skipping")
        return

    # Find today's push
    today_push = None
    for line in reversed(pushes_path.read_text(encoding="utf-8").strip().split("\n")):
        if line:
            push = json.loads(line)
            if push.get("date") == date:
                today_push = push
                break

    if not today_push:
        logger.info(f"No push found for {date}, skipping")
        return

    # Load paper details
    papers_path = project_root / "data" / "papers.jsonl"
    papers_by_id = {}
    if papers_path.exists():
        for line in papers_path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                p = json.loads(line)
                papers_by_id[p["id"]] = p

    papers = [papers_by_id[pid] for pid in today_push["paper_ids"] if pid in papers_by_id]

    if not papers:
        logger.info("No papers to push")
        return

    message = format_daily_push(date, papers, site_url)
    await send_push(token, chat_id, message)
    logger.info(f"Push sent for {date}: {len(papers)} papers")


if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if date_arg == "today":
        date_arg = None
    asyncio.run(main_push(date_arg))
