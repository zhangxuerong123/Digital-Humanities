import os
import re
import glob
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

# 文档解析
import pdfplumber  # pip install pdfplumber
from docx import Document  # pip install python-docx

# 中文处理
import jieba  # pip install jieba
import jieba.posseg as pseg

# 主题建模
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 表格输出
import pandas as pd  # pip install pandas

# 词云（可选依赖）
try:
    from wordcloud import WordCloud  # pip install wordcloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False


# =========================== 配置区 ===========================
DATA_DIR = "./policy_files"
LOCAL_STOPWORDS_PATH = "./hit_stopwords.txt"
USER_DICT_PATH = "./user_dict.txt"
POLICY_STOPWORDS_PATH = "./policy_stopwords.txt"

MIN_WORD_LEN = 2
NO_BELOW = 2
NO_ABOVE = 0.8

KEEP_POS = ["n", "nr", "ns", "nt", "v", "a"]

TOPIC_RANGE = range(2, 41)
RANDOM_STATE = 42
MAX_ITER = 10

TOP_N_WORDS = 50
TOPIC_TOP_N = 10
WORDCLOUD_MAX_WORDS = 200


# ==================== 2. 工具函数 ====================
def load_hit_stopwords(path):
    stopwords = set()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    stopwords.add(line)
        print(f"成功从本地加载停用词，共 {len(stopwords)} 个词")
    else:
        print(f"错误：找不到停用词文件 {path}！请检查路径。")
        stopwords = set(['的', '了', '在', '是', '和', '就', '都'])
    return stopwords


def load_policy_stopwords(path):
    stopwords = set()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    stopwords.add(line)
        print(f"成功加载政策专属停用词，共 {len(stopwords)} 个词")
    else:
        print(f"警告：未找到政策专属停用词文件 {path}，将跳过。")
    return stopwords


def load_user_dict(path):
    if path and os.path.exists(path):
        print(f"正在加载自定义词典: {path}")
        try:
            jieba.load_userdict(path)
            print("自定义词典加载成功。")
        except IndexError as e:
            print(f"词典解析出错: {e}")
            print("检查到问题行，正在逐行分析...")
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for i, line in enumerate(lines, 1):
                line = line.rstrip('\n')
                if not line.strip():
                    print(f"第 {i} 行是空行或仅含空白字符。")
                    continue
                parts = line.split()
                if len(parts) == 0:
                    print(f"第 {i} 行: '{line}' -> 分割后为空。")
                elif len(parts) == 1:
                    print(f"第 {i} 行: '{line}' -> 正常单词语。")
                elif len(parts) == 2:
                    try:
                        freq = float(parts[1])
                        print(f"第 {i} 行: '{line}' -> 正常词语+词频。")
                    except ValueError:
                        print(f"第 {i} 行: '{line}' -> 错误：词频部分不是数字。")
                else:
                    print(f"第 {i} 行: '{line}' -> 格式异常，分割后多于2部分。")
            print("请根据以上提示修正 user_dict.txt。")
            print("为继续运行，已跳过自定义词典加载。")
    else:
        print("提示：未找到自定义词典，将继续运行（分词可能不够精确）。")


def clean_text(text):
    text = re.sub(r'[0-9a-zA-Z]+', '', text)
    text = re.sub(r'[^\u4e00-\u9fff，。！？、；：“”‘’（）—…·\n]', '', text)
    text = re.sub(r'\s+', '', text)
    return text


def filter_pos(text, keep_pos=None):
    if keep_pos is None:
        keep_pos = ["n", "nr", "ns", "nt", "v", "a"]
    words = pseg.cut(text)
    res = []
    for pair in words:
        if pair.flag in keep_pos:
            res.append(pair.word)
    return res


def extract_text_from_pdf(file_path):
    text = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
    except Exception as e:
        print(f"PDF 读取失败 {file_path}: {e}")
        return ""
    return "\n".join(text)


def extract_text_from_docx(file_path):
    text = []
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            if para.text.strip():
                text.append(para.text.strip())
    except Exception as e:
        print(f"Word 读取失败 {file_path}: {e}")
        return ""
    return "\n".join(text)


def get_all_documents(data_dir):
    docs = []
    pdf_files = glob.glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True)
    docx_files = glob.glob(os.path.join(data_dir, "**", "*.docx"), recursive=True)

    for f in pdf_files:
        print(f"正在处理 PDF: {f}")
        raw = extract_text_from_pdf(f)
        if raw.strip():
            docs.append(raw)
    for f in docx_files:
        print(f"正在处理 DOCX: {f}")
        raw = extract_text_from_docx(f)
        if raw.strip():
            docs.append(raw)
    return docs


# ==================== 词频 / 词云 / 主题词表 ====================
def find_chinese_font():
    candidates = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simkai.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def word_frequency_analysis(processed_docs, top_n=50, save_csv="word_freq_top50.csv"):
    counter = Counter()
    for words in processed_docs:
        counter.update(words)

    top_words = counter.most_common(top_n)
    df = pd.DataFrame(top_words, columns=["词语", "词频"])
    df.index = range(1, len(df) + 1)
    df.index.name = "排名"

    print(f"\n===== 前 {top_n} 个高频词 =====")
    print(df.to_string())
    df.to_csv(save_csv, encoding="utf-8-sig")
    print(f"\n高频词表已保存为 '{save_csv}'")
    return counter


def plot_wordcloud(freq_dict, save_path="wordcloud.png", max_words=200):
    if not HAS_WORDCLOUD:
        print("跳过词云图：未安装 wordcloud，可运行 pip install wordcloud 安装。")
        return
    if not freq_dict:
        print("跳过词云图：词频字典为空。")
        return

    font_path = find_chinese_font()
    if font_path is None:
        print("警告：未找到中文字体，词云中的中文可能显示为方块。"
              "请在 find_chinese_font() 中补充字体路径。")

    wc = WordCloud(
        font_path=font_path,
        width=1200,
        height=800,
        background_color="white",
        max_words=max_words,
        colormap="viridis",
        random_state=RANDOM_STATE,
        collocations=False,
    ).generate_from_frequencies(freq_dict)

    plt.figure(figsize=(12, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"词云图已保存为 '{save_path}'")


def export_topic_table(lda_model, feature_names, n_top_words=10,
                       save_wide="topic_words_wide.csv",
                       save_long="topic_words_long.csv"):
    topic_dict = {}
    long_rows = []

    for topic_idx, topic in enumerate(lda_model.components_):
        top_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_ind]
        topic_dict[f"主题{topic_idx + 1}"] = top_words

        for rank, i in enumerate(top_ind, 1):
            long_rows.append({
                "主题": f"主题{topic_idx + 1}",
                "排名": rank,
                "关键词": feature_names[i],
                "权重": float(topic[i]),
            })

    df_wide = pd.DataFrame(topic_dict)
    df_wide.index = range(1, n_top_words + 1)
    df_wide.index.name = "排名"

    print("\n===== 主题词表（宽表：行=排名，列=主题）=====")
    print(df_wide.to_string())
    df_wide.to_csv(save_wide, encoding="utf-8-sig")

    df_long = pd.DataFrame(long_rows)
    df_long.to_csv(save_long, encoding="utf-8-sig", index=False)

    print(f"\n主题词表（宽表）已保存为 '{save_wide}'")
    print(f"主题词表（长表，含权重）已保存为 '{save_long}'")
    return df_wide, df_long


# ==================== 3. 主流程 ====================
def main():
    # ---- 加载外部资源 ----
    load_user_dict(USER_DICT_PATH)

    # ===== 修改：政策停用词也生效（与哈工大停用词合并） =====
    hit_stopwords = load_hit_stopwords(LOCAL_STOPWORDS_PATH)
    policy_stopwords = load_policy_stopwords(POLICY_STOPWORDS_PATH)
    overlap = hit_stopwords & policy_stopwords
    stopwords = hit_stopwords | policy_stopwords
    print(f"哈工大停用词 {len(hit_stopwords)} 个，政策停用词 {len(policy_stopwords)} 个，"
          f"重复 {len(overlap)} 个，合并后共 {len(stopwords)} 个")
    # ==========================================================

    # ---- 读取所有文档 ----
    print("\n开始读取政策文件...")
    raw_docs = get_all_documents(DATA_DIR)
    print(f"共读取到 {len(raw_docs)} 篇文档。\n")

    if len(raw_docs) == 0:
        print("未找到任何文档，请检查 DATA_DIR 路径是否正确。")
        return

    # ---- 分词 + 清洗 ----
    print("正在进行分词、词性过滤和停用词过滤...")
    processed_docs = []
    for doc in raw_docs:
        cleaned = clean_text(doc)
        pos_filtered_words = filter_pos(cleaned, keep_pos=KEEP_POS)
        filtered = [w for w in pos_filtered_words
                    if w not in stopwords and len(w) >= MIN_WORD_LEN]
        processed_docs.append(filtered)

    doc_strings = [" ".join(words) for words in processed_docs]

    # ===== 高频词统计 + 词云 =====
    word_counter = word_frequency_analysis(processed_docs, top_n=TOP_N_WORDS)
    plot_wordcloud(dict(word_counter), save_path="wordcloud.png",
                   max_words=WORDCLOUD_MAX_WORDS)

    # ---- 构建文档-词频矩阵 ----
    print("\n正在构建词频矩阵...")
    vectorizer = CountVectorizer(
        token_pattern=r'(?u)\b\w+\b',
        max_df=NO_ABOVE,
        min_df=NO_BELOW,
    )
    dtm = vectorizer.fit_transform(doc_strings)
    vocab = vectorizer.get_feature_names_out()
    print(f"词典大小（过滤极端词后）: {len(vocab)} 个词\n")

    # ---- 训练不同主题数的 LDA ----
    print("开始训练 LDA 模型（需要一点耐心）...")
    perplexities = []
    models = {}

    for k in TOPIC_RANGE:
        print(f"  训练主题数 K = {k} ...", end=" ", flush=True)
        lda = LatentDirichletAllocation(
            n_components=k,
            random_state=RANDOM_STATE,
            max_iter=MAX_ITER,
            learning_method='online',
            batch_size=128,
            evaluate_every=1,
            verbose=0,
        )
        lda.fit(dtm)
        perp = lda.perplexity(dtm)
        perplexities.append(perp)
        models[k] = lda
        print(f"困惑度 = {perp:.2f}")

    # ---- 绘制困惑度曲线 ----
    best_k = TOPIC_RANGE[np.argmin(perplexities)]
    plt.figure(figsize=(10.0, 6.0))
    plt.plot(TOPIC_RANGE, perplexities, 'bo-', linewidth=2, markersize=8)
    plt.xlabel("Number of Topics", fontsize=12)
    plt.ylabel("Perplexity", fontsize=12)
    plt.title("Perplexity for Different Topic Numbers", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axvline(x=best_k, color='r', linestyle='--',
                label=f'Lowest Perplexity K={best_k}')
    plt.legend()
    plt.tight_layout()
    plt.savefig("perplexity_curve.png", dpi=150)
    plt.show()
    print(f"\n困惑度曲线已保存为 'perplexity_curve.png'")

    # ===== 输出主题词表 =====
    print(f"\n===== 最优主题数（基于困惑度）: K = {best_k} =====")
    best_lda = models[best_k]
    export_topic_table(
        best_lda,
        vocab,
        n_top_words=TOPIC_TOP_N,
        save_wide="topic_words_wide.csv",
        save_long="topic_words_long.csv",
    )


if __name__ == "__main__":
    main()