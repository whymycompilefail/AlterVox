"""规则词表兜底层（SPEC 未决 #10 定稿实现，PoC 9/10 零误伤），v2_poc/run_rule_test.py 移植。

三层结构（确定性替换，全部可审计）：
1. 别名层——转写中的英文碎片精确替换（key in → 刻印）
2. ITN 层——英文数词 → 阿拉伯数字（forty → 40）
3. 拼音层——词表项与转写 n-gram 做音节级序列相似度匹配（阈值 0.8），替换最近邻

生产化待办：停用词表换 jieba 分词、词表管理改会话级配置。
"""

import difflib
import re

from pypinyin import lazy_pinyin, Style

# 别名表：转写中的英文碎片 -> 中文（替换目标可与词表项不同粒度，如 check->查）
ALIAS = {"key in": "刻印", "keyin": "刻印", "eden": "伊甸", "boss": "老板", "check": "查"}

# 停用表：拼音层跳过的常见真词（防"有点"→"伊甸"这类 0.80 撞线误伤）
STOPWORDS = {"有点", "没有", "一点", "有些"}

NUM_WORDS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
             "seventy": 70, "eighty": 80, "ninety": 90, "ten": 10, "hundred": 100}

PINYIN_TH = 0.8
CJK = re.compile(r"[\u4e00-\u9fff]+")

# 场景专有名词（喂给润色层的词汇表与拼音层共用一份数据）
GLOSSARY_TERMS = ["刻印", "伊甸", "蕈猪", "宫殿", "招新纳贤", "凡庸",
                  "树莓薄荷饮", "果然要"]
GLOSSARY_TEXT = "蕈猪、招新纳贤、凡庸、宫殿、刻印、伊甸、树莓薄荷饮、果然要、老板、40种"


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
    """返回 (fixed_text, edits)；edits 每项为 (原文片段, 替换为, 命中层)。"""
    glossary = GLOSSARY_TERMS if glossary is None else glossary
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
