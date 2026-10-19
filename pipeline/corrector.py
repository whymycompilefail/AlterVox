"""规则兜底层（确定性替换，全部可审计）。

层级：
1. 别名层——转写中的英文碎片精确替换（仅通用词条）
2. ITN 层——英文数词 → 阿拉伯数字
3. 拼音层——词表项与转写 n-gram 做音节级序列相似度匹配（阈值 0.8），
   **仅在调用方显式传入词表时启用**（场景词表，默认空）

2026-10-19 评测卫生决策：内置场景词表与场景别名已移除——它们源自训练
测试集的同一批词，属于"提前给答案"。词表改为会话级配置（生产化待办），
由使用方按自己的场景显式传入。
"""

import difflib
import re

from pypinyin import lazy_pinyin, Style

# 别名表：通用英文碎片 -> 中文
ALIAS = {"boss": "老板", "check": "查"}

# 停用表：拼音层跳过的常见真词（防"有点"→"伊甸"这类 0.80 撞线误伤）
STOPWORDS = {"有点", "没有", "一点", "有些"}

NUM_WORDS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
             "seventy": 70, "eighty": 80, "ninety": 90, "ten": 10, "hundred": 100}

PINYIN_TH = 0.8
CJK = re.compile(r"[\u4e00-\u9fff]+")


def py(text):
    """音节列表按空格连接：蕈猪 -> "xun zhu"（音节级对齐，防止字符级漏配）"""
    return " ".join(lazy_pinyin(text, style=Style.NORMAL))


def cjk_ngrams(text, n):
    out = []
    for m in CJK.finditer(text):
        s = m.group()
        for i in range(len(s) - n + 1):
            out.append((m.start() + i, s[i:i + n]))
    return out


def correct(text, glossary=None, aliases=None, stopwords=None):
    """返回 (fixed_text, edits)；edits 每项为 (原文片段, 替换为, 命中层)。

    glossary：场景词表（可选，默认空=拼音层不启用）。
    """
    glossary = glossary or []
    aliases = ALIAS if aliases is None else aliases
    stopwords = STOPWORDS if stopwords is None else stopwords
    edits = []
    lower = text.lower()

    # 1) 别名层
    for a, term in sorted(aliases.items(), key=lambda kv: -len(kv[0])):
        if a in lower:
            text = re.sub(re.escape(a), term, text, flags=re.I)
            edits.append((a, term, "alias"))
            lower = text.lower()

    # 2) ITN 层
    for w, num in NUM_WORDS.items():
        if w in text.lower():
            text = re.sub(re.escape(w), str(num), text, flags=re.I)
            edits.append((w, str(num), "itn"))

    # 3) 拼音层：每个词表项在转写里找拼音最近邻的等长窗口
    for term in glossary:
        if term in text:
            continue
        n = len(term)
        t_py = py(term)
        best = None  # (sim, start, gram)
        for start, gram in cjk_ngrams(text, n):
            if gram == term or gram in stopwords:
                continue
            sim = difflib.SequenceMatcher(None, py(gram), t_py).ratio()
            if sim >= PINYIN_TH and (best is None or sim > best[0]):
                best = (sim, start, gram)
        if best:
            _, start, gram = best
            text = text[:start] + term + text[start + n:]
            edits.append((gram, term, f"pinyin:{best[0]:.2f}"))
    return text, edits
