from elasticsearch.dsl import tokenizer, normalizer, token_filter, analyzer

# ================================
# Token Filters 定义
# ================================

# Shingle 过滤器 - 用于生成连续词组（有空格分隔）
# 例如："hello world test" -> ["hello world", "world test", "hello world test"]
# 不输出单个词（output_unigrams=False），只输出词组
shingle_lease_filter = token_filter(
    "shingle_lease",
    type="shingle",
    min_shingle_size=2,  # 最小词组长度：2个词
    max_shingle_size=5,  # 最大词组长度：5个词
    output_unigrams=False,  # 不输出单个词
)

# Shingle 过滤器 - 用于生成连续词组（无空格分隔）
# 例如："hello world test" -> ["helloworld", "worldtest", "helloworldtest"]
# 适用于中文或需要紧密连接的词组
shingle_lease_nospace_filter = token_filter(
    "shingle_lease",
    type="shingle",
    min_shingle_size=2,
    max_shingle_size=5,
    output_unigrams=False,
    token_separator="",  # 词组间不使用空格分隔
)

# ================================
# 自动补全分析器
# ================================

# 用于搜索建议和自动补全功能
# 例如输入："Machine Learning"
# -> 分词：["machine", "learning"]
# -> 生成shingle：["machine learning"]
# 适合：搜索框自动补全、建议功能
completion_analyzer = analyzer(
    "completion_analyzer",
    tokenizer="standard",  # 标准分词器，按空格和标点分词
    filter=["lowercase", "shingle"],  # 转小写 + 生成词组
)

# ================================
# 边缘N-gram分析器
# ================================

# 边缘N-gram分词器 - 从词的开头开始生成子字符串
# 例如："elasticsearch" -> ["e", "el", "ela", "elas", ..., "elasticsearch"]
edge_tokenizer = tokenizer(
    "edge_tokenizer",
    type="edge_ngram",
    min_gram=1,  # 最小字符数
    max_gram=20,  # 最大字符数
)

# 边缘N-gram分析器 - 用于前缀匹配搜索
# 例如："Elasticsearch" -> ["e", "el", "ela", "elas", "elast", ..., "elasticsearch"]
# 适合：输入时实时搜索、前缀匹配
edge_analyzer = analyzer(
    "edge_analyzer", tokenizer=edge_tokenizer, filter=["lowercase"]  # 转换为小写
)

# ================================
# 关键词分析器
# ================================

# 小写关键词分析器 - 将整个输入作为单个词处理，但转为小写
# 例如："Hello World" -> ["hello world"]（作为一个完整的词）
# 适合：精确匹配、状态字段、分类字段
lower_keyword_analyzer = analyzer(
    "lowercase_keyword",
    tokenizer="keyword",  # 不分词，整个输入作为一个token
    filter=["lowercase"],  # 转小写
)

# ================================
# 标准化器
# ================================

# 小写标准化器 - 用于keyword字段的标准化
# 例如："Hello World" -> "hello world"
# 与analyzer不同，normalizer用于keyword字段，不进行分词
# 适合：排序、聚合时的大小写标准化
lower_normalizer = normalizer(
    "lower_normalizer",
    char_filter=[],  # 不使用字符过滤器
    filter=["lowercase"],  # 只转小写
)

# ================================
# 英文词干分析器
# ================================

# 英文雪球词干过滤器 - 将英文单词还原为词根
# 例如："running", "runs", "ran" -> "run"
#      "better", "good" -> "good", "better"（不规则变化需要特殊处理）
snow_en_filter = token_filter(
    "snow_filter", type="snowball", language="English"  # 英文词干提取
)

# 英文词干分析器 - 用于英文文本的语义搜索
# 例如："I am running quickly"
# -> 分词：["i", "am", "running", "quickly"]
# -> 词干化：["i", "am", "run", "quick"]
# 适合：英文文档搜索，提高召回率
snow_en_analyzer = analyzer(
    "snow_analyzer",
    tokenizer="standard",  # 标准分词
    filter=["lowercase", snow_en_filter],  # 小写 + 词干提取
)

# ================================
# Shingle分析器 - 有空格版本
# ================================

# 基于空格的shingle分析器 - 按空格分词后生成词组
# 例如："hello world test case"
# -> 分词：["hello", "world", "test", "case"]
# -> shingle：["hello world", "world test", "test case", "hello world test", ...]
# 适合：短语搜索、多词匹配
shingle_space_analyzer = analyzer(
    "shingle_space_analyzer",
    tokenizer="whitespace",  # 按空格分词
    filter=["lowercase", shingle_lease_filter],  # 小写 + 生成词组
)

# ================================
# Shingle分析器 - 无空格版本
# ================================

# 无空格shingle分析器 - 适用于中文或连续字符处理
# 例如："hello-world_test"
# -> word_delimiter_graph分解：["hello", "world", "test"]
# -> shingle无空格：["helloworld", "worldtest", "helloworldtest"]
# 适合：中文文本、代码搜索、复合词处理
shingle_nospace_analyzer = analyzer(
    "shingle_nospace_analyzer",
    tokenizer="keyword",  # 不分词，保持原始输入
    filter=[
        "lowercase",  # 转小写
        "word_delimiter_graph",  # 按分隔符拆分（-,_等）
        shingle_lease_nospace_filter,  # 生成无空格词组
    ],
)

# ================================
# 预分词内容分析器 - 用于应用层已分词的BM25搜索
# ================================

# 预分词文本BM25分析器 - 用于已经在应用层分词的内容进行BM25搜索
# 应用层负责jieba分词，ES进行空格分词并过滤停用词，提高搜索质量
# 例如：应用层输入 "我 今天 去了 北京大学" (空格分隔的分词结果)
# -> 分词：["我", "今天", "去了", "北京大学"]
# -> 过滤停用词：["今天", "去了", "北京大学"] (假设"我"是停用词)
# 适合：中文文档BM25搜索、预处理分词内容的相关性搜索
whitespace_lowercase_trim_stop_analyzer = analyzer(
    "whitespace_lowercase_trim_stop_analyzer",
    tokenizer="whitespace",  # 按空格分词处理预分词内容
    filter=[
        "lowercase",  # 转小写
        "trim",  # 去除首尾空白
        "stop",  # 停用词过滤，提高搜索相关性
    ],
)
