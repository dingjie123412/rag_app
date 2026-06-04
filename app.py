import sys
import os
import datetime
from datetime import datetime as dt
import re


# 安全配置编码（兼容命令行 + Streamlit 双模式）
def setup_encoding():
    try:
        if "streamlit" not in sys.modules:
            import io
            if hasattr(sys.stdout, 'buffer') and not hasattr(sys.stdout, '_streamlit_original'):
                original_stdout = sys.stdout
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
                sys.stdout._streamlit_original = original_stdout
            try:
                if hasattr(sys.stdin, 'reconfigure'):
                    sys.stdin.reconfigure(encoding='utf-8')
            except Exception:
                pass
    except Exception as e:
        pass


setup_encoding()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Agent 相关（使用 langchain-classic 避免 langgraph 依赖问题）
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI

import requests
from docx import Document

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None
try:
    import pandas as pd
except ImportError:
    pd = None
from typing import TypedDict, Sequence, List, Optional
import numpy as np
from dotenv import load_dotenv

import streamlit as st

# ========== 三层记忆系统辅助函数 ==========
def validate_retrieval_result(retrieval_result: str, question: str) -> tuple:
    """验证检索结果是否真正回答了问题"""
    
    if not retrieval_result or "未在知识库中找到" in retrieval_result:
        return False, "知识库中无相关内容"
    
    # 如果检索结果很长，说明有内容，默认通过
    if len(retrieval_result) > 200:
        return True, "检索成功"
    
    # 简化的关键词检查
    import re
    q_words = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', question.lower()))
    result_words = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', retrieval_result.lower()))
    
    # 只要有一个关键词匹配就算通过
    if q_words and result_words:
        matched = q_words & result_words
        if len(matched) >= 1:  # 至少匹配1个关键词
            return True, f"匹配到关键词: {matched}"
    
    # 检查问题中的核心词是否出现在结果中
    core_terms = ["中间件", "决策", "规划", "升降", "双臂", "协同", "MoveIt"]
    for term in core_terms:
        if term in question and term in retrieval_result:
            return True, f"匹配到核心术语: {term}"
    
    # 兜底：结果不为空就通过（最宽松）
    if retrieval_result.strip():
        return True, "结果不为空，认为有效"
    
    return False, "未找到相关内容"


def generate_answer_from_retrieval(question: str, retrieval_result: str, memory_context: str) -> str:
    """
    基于检索结果生成回答，强制要求只引用原文
    """
    prompt = f"""你是一个严格的知识库问答助手，只能根据下面提供的文档内容回答问题。

【铁律】
1. 只能引用下面「文档内容」中的原文
2. 禁止添加任何自己的知识、推断、解释或总结
3. 如果文档内容不能直接回答问题，只输出「未在知识库中找到直接相关的信息」
4. 回答格式：先写「根据知识库原文，回答如下：」，然后逐条列出原文

{memory_context}

【文档内容】
{retrieval_result}

【用户问题】
{question}

【你的回答】"""

    response = call_llm(prompt, temperature=0.0)  # temperature=0 减少随机性
    return response.strip()


def verify_answer_faithfulness(answer: str, retrieval_result: str) -> bool:
    """
    验证 LLM 的回答是否忠实于检索到的原文
    
    检查：
    1. 回答中的关键信息是否在原文中存在
    2. 是否添加了原文没有的内容
    """
    # 提取答案中的句子
    import re
    answer_sentences = re.split(r'[。！？；\n]', answer)
    
    # 提取原文中的内容
    original_text = retrieval_result.lower()
    
    unfaithful_count = 0
    for sent in answer_sentences:
        if len(sent.strip()) < 5:  # 短句跳过
            continue
        # 提取句子中的关键词（去掉"根据知识库原文"这类固定开头）
        if "根据知识库原文" in sent:
            continue
        sent_clean = sent.strip()
        if not sent_clean:
            continue
        
        # 检查句子中的关键词是否在原文中
        sent_words = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', sent_clean.lower()))
        if len(sent_words) > 3:  # 至少3个关键词
            found_in_original = False
            for word in sent_words:
                if len(word) > 1 and word in original_text:
                    found_in_original = True
                    break
            if not found_in_original:
                # 这个句子的关键词完全不在原文中，可能是幻觉
                unfaithful_count += 1
    
    # 如果超过30%的句子有问题，认为不忠实
    total_sentences = len([s for s in answer_sentences if len(s.strip()) > 5])
    if total_sentences == 0:
        return True
    
    return (unfaithful_count / total_sentences) < 0.3


def extract_exact_quotes(retrieval_result: str, question: str) -> str:
    """
    降级方案：直接从检索结果中提取原文引用
    """
    # 解析文档片段
    import re
    
    # 提取所有【文档片段 X】的内容
    pattern = r'【文档片段 \d+\】\n来源:.*?\n内容: (.*?)(?=\n【文档片段|\Z)'
    matches = re.findall(pattern, retrieval_result, re.DOTALL)
    
    if not matches:
        return "未在知识库中找到直接相关的信息。"
    
    # 简单去重后返回
    unique_quotes = []
    for m in matches:
        m_clean = m.strip()
        if m_clean and m_clean not in unique_quotes:
            unique_quotes.append(m_clean)
    
    if unique_quotes:
        result = "根据知识库原文，回答如下：\n\n"
        for i, quote in enumerate(unique_quotes, 1):
            result += f"{i}. {quote}\n\n"
        return result
    else:
        return "未在知识库中找到直接相关的信息。"


# ========== 双层路由：第一层 - 快速规则过滤 ==========
def is_obvious_non_retrieval(question: str) -> tuple:
    """
    快速判断是否明显不需要检索
    
    返回:
        (是否不需要检索, 匹配到的类型)
    """
    q = question.strip().lower()
    
    # 问候类
    greetings = ["你好", "您好", "hi", "hello", "在吗", "在不在", "你好呀"]
    for g in greetings:
        if q == g or q.startswith(g):
            return True, "greeting"
    
    # 感谢类
    thanks = ["谢谢", "感谢", "多谢", "thanks"]
    for t in thanks:
        if t in q:
            return True, "thanks"
    
    # 简单自我介绍询问
    if q in ["你是谁", "你是什么", "介绍一下你自己"]:
        return True, "self_intro"
    
    # 功能询问（不需要检索知识库）
    function_questions = ["你能做什么", "你有什么功能", "可以做什么"]
    for fq in function_questions:
        if fq in q:
            return True, "capability"
    
    # 非检索类（字数太少且不是问句）
    if len(q) <= 3 and "?" not in q and "？" not in q and "么" not in q:
        return True, "too_short"
    
    return False, None


def simple_keyword_overlap(query: str, retrieval_result: str) -> tuple:
    """降级方案：关键词重叠判断"""
    import re
    # 提取中文和英文词
    q_words = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', query.lower()))
    r_words = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', retrieval_result.lower()))
    
    if not q_words:
        return False, 0, "无有效关键词"
    
    overlap = len(q_words & r_words)
    overlap_ratio = overlap / len(q_words)
    
    # 至少匹配1个关键词，且重叠率>10%
    is_relevant = overlap >= 1 and overlap_ratio > 0.1
    return is_relevant, overlap_ratio, f"关键词重叠: {overlap}/{len(q_words)} ({overlap_ratio:.1%})"


# ========== 双层路由：第二层 - 检索 + 相关性判断 ==========
def is_retrieval_relevant(query: str, retrieval_result: str) -> tuple:
    """
    判断检索结果是否与问题相关
    
    返回:
        (是否相关, 相关性分数, 理由)
    """
    # 情况1：检索结果为空或太短（少于50字符）
    if not retrieval_result or len(retrieval_result) < 50:
        return False, 0, "检索结果为空或太短"
    
    # 情况2：检索结果明确表示未找到
    if "未在知识库中找到" in retrieval_result or "没有找到" in retrieval_result:
        return False, 0, "知识库无相关内容"
    
    # 简化：只要检索到内容且包含Excel，就认为相关（Excel表格需要模型自己解析）
    if "Excel" in retrieval_result:
        print(f"[DEBUG] 检索到Excel内容，直接认为相关")
        return True, 0.7, "包含Excel表格内容"
    
    # 情况3：用 LLM 判断相关性（更准确）
    prompt = f"""判断检索到的文档是否能够回答用户的问题。

【用户问题】
{query}

【检索到的文档内容】
{retrieval_result[:1000]}

【判断标准】
- 如果文档内容直接回答了问题 → related
- 如果文档内容部分相关但不完全 → partially_related  
- 如果文档内容完全不相关 → unrelated

【输出格式】
只输出 JSON：
{{"relevance": "related/partially_related/unrelated", "score": 0.0-1.0}}

score 含义：
- 0.7-1.0: 直接回答，相关
- 0.5-0.7: 部分相关，勉强可用
- 0.0-0.5: 不相关
"""

    try:
        response = call_llm(prompt, temperature=0.0)
        import re
        import json
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            relevance = result.get("relevance", "unrelated")
            score = result.get("score", 0)
            
            # 更宽松：部分相关且分数>=0.3就认为可以基于检索回答
            is_relevant = relevance in ["related", "partially_related"] and score >= 0.3
            return is_relevant, score, f"LLM判断: {relevance}, 分数{score}"
    except Exception as e:
        print(f"[相关性判断失败] {e}")
    
    # 降级：简单关键词判断（更宽松）
    return simple_keyword_overlap(query, retrieval_result)

# 长期记忆存储（全局变量，会话期间保持）
long_term_memory_store = []

# ========== 配置加载：支持 Streamlit Secrets 和 .env 文件 ==========
def load_config():
    """加载配置，优先使用 Streamlit Secrets，其次使用 .env 文件"""
    config = {}
    loaded_from = None
    
    # 尝试从 Streamlit Secrets 加载（部署到 Streamlit Cloud 时使用）
    try:
        if hasattr(st, 'secrets') and 'QWEN_API_KEY' in st.secrets:
            config['DOC_PATH'] = st.secrets.get('DOC_PATH', '')
            config['QWEN_API_KEY'] = st.secrets['QWEN_API_KEY']
            config['QWEN_API_BASE'] = st.secrets['QWEN_API_BASE']
            config['MODEL_NAME'] = st.secrets.get('MODEL_NAME', 'qwen-plus')
            config['EMBEDDING_API_KEY'] = st.secrets['EMBEDDING_API_KEY']
            config['EMBEDDING_API_URL'] = st.secrets['EMBEDDING_API_URL']
            config['EMBEDDING_MODEL'] = st.secrets.get('EMBEDDING_MODEL', 'text-embedding-v2')
            loaded_from = 'Streamlit Secrets'
            print("[INFO] 配置已从 Streamlit Secrets 加载")
            return config, loaded_from
    except Exception as e:
        print(f"[WARN] 无法从 Streamlit Secrets 加载配置: {e}")
    
    # 尝试从 .env 文件加载（本地开发时使用）
    try:
        load_dotenv()
        config['DOC_PATH'] = os.getenv("DOC_PATH", "")
        config['QWEN_API_KEY'] = os.getenv("QWEN_API_KEY", "")
        config['QWEN_API_BASE'] = os.getenv("QWEN_API_BASE", "")
        config['MODEL_NAME'] = os.getenv("MODEL_NAME", "qwen-plus")
        config['EMBEDDING_API_KEY'] = os.getenv("EMBEDDING_API_KEY", "")
        config['EMBEDDING_API_URL'] = os.getenv("EMBEDDING_API_URL", "")
        config['EMBEDDING_MODEL'] = os.getenv("EMBEDDING_MODEL", "text-embedding-v2")
        loaded_from = '.env 文件'
        print("[INFO] 配置已从 .env 文件加载")
        return config, loaded_from
    except Exception as e:
        print(f"[WARN] 无法从 .env 文件加载配置: {e}")
    
    return config, loaded_from

# 加载配置
config, config_source = load_config()

# 全局配置
DOC_PATH = config.get('DOC_PATH', "")
QWEN_API_KEY = config.get('QWEN_API_KEY', "")
QWEN_API_BASE = config.get('QWEN_API_BASE', "")
MODEL_NAME = config.get('MODEL_NAME', "qwen-plus")
EMBEDDING_API_KEY = config.get('EMBEDDING_API_KEY', "")
EMBEDDING_API_URL = config.get('EMBEDDING_API_URL', "")
EMBEDDING_MODEL = config.get('EMBEDDING_MODEL', "text-embedding-v2")

# 配置验证函数
def validate_config() -> tuple:
    """验证配置是否完整"""
    missing = []
    
    if not QWEN_API_KEY:
        missing.append('QWEN_API_KEY')
    if not QWEN_API_BASE:
        missing.append('QWEN_API_BASE')
    if not EMBEDDING_API_KEY:
        missing.append('EMBEDDING_API_KEY')
    if not EMBEDDING_API_URL:
        missing.append('EMBEDDING_API_URL')
    
    is_valid = len(missing) == 0
    return is_valid, missing, config_source


def get_embeddings(texts: list) -> np.ndarray:
    """调用阿里云 Embedding API 获取文本向量（支持分批处理，每批最多25条）"""
    if not EMBEDDING_API_KEY or not EMBEDDING_API_URL:
        raise ValueError("Embedding API 配置未设置")

    headers = {
        "Authorization": f"Bearer {EMBEDDING_API_KEY}",
        "Content-Type": "application/json"
    }

    batch_size = 25
    all_embeddings = []

    try:
        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            payload = {
                "model": EMBEDDING_MODEL,
                "input": batch_texts
            }

            response = requests.post(
                EMBEDDING_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

            if "data" not in result:
                raise ValueError(f"API 返回格式错误: {result}")

            batch_embeddings = [item["embedding"] for item in result["data"]]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    except requests.exceptions.Timeout:
        raise RuntimeError("网络超时，请检查网络连接或稍后重试")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("网络连接失败，请检查网络设置")
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP 错误: {e}"
        try:
            result = response.json()
            if "error" in result:
                error_msg += f" - {result['error'].get('message', '')}"
        except:
            pass
        raise RuntimeError(error_msg)
    except Exception as e:
        raise RuntimeError(f"Embedding API 调用失败: {str(e)}")


def call_llm(prompt: str, temperature: float = 0.1) -> str:
    """调用千问 LLM API"""
    if not QWEN_API_KEY or not QWEN_API_BASE:
        raise ValueError("LLM API 配置未设置")

    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }

    try:
        response = requests.post(
            f"{QWEN_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()

        if "choices" not in result or len(result["choices"]) == 0:
            raise ValueError(f"API 返回格式错误: {result}")

        return result["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout:
        raise RuntimeError("LLM 服务响应超时，请稍后重试")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("网络连接失败，请检查网络设置")
    except requests.exceptions.HTTPError as e:
        error_msg = f"LLM API 错误: {e}"
        try:
            result = response.json()
            if "error" in result:
                error_msg += f" - {result['error'].get('message', '')}"
        except:
            pass
        raise RuntimeError(error_msg)
    except Exception as e:
        raise RuntimeError(f"LLM API 调用失败: {str(e)}")


def call_llm_with_history(messages: List[dict], temperature: float = 0.1) -> str:
    """调用千问 LLM API（带对话历史）"""
    if not QWEN_API_KEY or not QWEN_API_BASE:
        raise ValueError("LLM API 配置未设置")

    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature
    }

    try:
        response = requests.post(
            f"{QWEN_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()

        if "choices" not in result or len(result["choices"]) == 0:
            raise ValueError(f"API 返回格式错误: {result}")

        return result["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout:
        raise RuntimeError("LLM 服务响应超时，请稍后重试")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("网络连接失败，请检查网络设置")
    except requests.exceptions.HTTPError as e:
        error_msg = f"LLM API 错误: {e}"
        try:
            result = response.json()
            if "error" in result:
                error_msg += f" - {result['error'].get('message', '')}"
        except:
            pass
        raise RuntimeError(error_msg)
    except Exception as e:
        raise RuntimeError(f"LLM API 调用失败: {str(e)}")


# ========== 核心改造：LangChain Agent 架构 ==========

# ========== 工具增强：重试 + 超时 + 熔断 ==========
import time
import functools
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# 熔断状态管理器
class CircuitBreaker:
    def __init__(self, failure_threshold=3, reset_timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = 0
        self.is_open = False

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.is_open = True

    def record_success(self):
        self.failure_count = 0
        self.is_open = False

    def check(self):
        if self.is_open:
            if time.time() - self.last_failure_time >= self.reset_timeout:
                self.is_open = False
                self.failure_count = 0
        return self.is_open

# 为每个工具创建熔断实例
tool_circuit_breakers = {}

def get_circuit_breaker(tool_name):
    if tool_name not in tool_circuit_breakers:
        tool_circuit_breakers[tool_name] = CircuitBreaker()
    return tool_circuit_breakers[tool_name]

# 重试装饰器
def retry_with_backoff(max_retries=2, delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tool_name = func.__name__
            breaker = get_circuit_breaker(tool_name)
            
            # 检查熔断状态
            if breaker.check():
                return f"【熔断中】工具 '{tool_name}' 暂时不可用，请稍后重试。"
            
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    breaker.record_success()
                    return result
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(delay)
            breaker.record_failure()
            return f"【调用失败】工具 '{tool_name}' 已重试 {max_retries} 次，全部失败。错误: {str(last_exception)}"
        return wrapper
    return decorator

# 超时装饰器
def timeout(seconds=10):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def run_func():
                return func(*args, **kwargs)
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_func)
                try:
                    return future.result(timeout=seconds)
                except TimeoutError:
                    return f"【执行超时】工具 '{func.__name__}' 运行时间超过 {seconds} 秒，已终止。"
                except Exception as e:
                    raise e
        return wrapper
    return decorator

# 组合装饰器：重试 + 超时 + 熔断
def tool_enhancer(max_retries=2, delay=1, timeout_seconds=10):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tool_name = func.__name__
            breaker = get_circuit_breaker(tool_name)
            
            # 检查熔断状态
            if breaker.check():
                return f"【熔断中】工具 '{tool_name}' 暂时不可用，请稍后重试。"
            
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    def run_func():
                        return func(*args, **kwargs)
                    
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(run_func)
                        try:
                            result = future.result(timeout=timeout_seconds)
                            breaker.record_success()
                            return result
                        except TimeoutError:
                            if attempt < max_retries:
                                time.sleep(delay)
                                continue
                            breaker.record_failure()
                            return f"【执行超时】工具 '{tool_name}' 运行时间超过 {timeout_seconds} 秒，已终止。"
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(delay)
            breaker.record_failure()
            return f"【调用失败】工具 '{tool_name}' 已重试 {max_retries} 次，全部失败。错误: {str(last_exception)}"
        return wrapper
    return decorator

# ========== 动态生成检索工具（解决全局上下文BUG）
def get_search_tool(chunks, chunk_embeddings, use_semantic_search, top_k):
    def search_knowledge_base(query: str) -> str:
        """检索内部知识库，根据用户问题查找相关文档内容"""
        print(f"\n[DEBUG] ========== 检索开始 ==========")
        print(f"[DEBUG] 查询词: {query}")
        print(f"[DEBUG] 知识库chunks总数: {len(chunks)}")
        
        # 统计Excel和非Excel的数量
        excel_count = sum(1 for c in chunks if "Excel" in c["source_type"])
        print(f"[DEBUG] Excel chunks数: {excel_count}, 其他 chunks数: {len(chunks)-excel_count}")
        
        if not chunks:
            return "知识库为空，请先加载文档。"
        if use_semantic_search:
            results = semantic_search(query, chunks, chunk_embeddings, top_k=top_k)
        else:
            results = simple_keyword_search(query, chunks, top_k=top_k)
        
        print(f"[DEBUG] 原始检索结果数: {len(results)}")
        if results:
            print(f"[DEBUG] 第一个结果来源: {results[0].get('source_type', '未知')}")
            print(f"[DEBUG] 第一个结果内容预览: {results[0].get('text', '')[:150]}...")
        
        # 修改：根据查询长度动态调整阈值
        import re
        q_words = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", query.lower()))
        
        # 长查询（超过5个词）阈值降低
        if len(q_words) > 5:
            min_overlap_ratio = 0.05  # 长查询只要匹配5%就行
        elif len(q_words) > 2:
            min_overlap_ratio = 0.1   # 中等查询10%
        else:
            min_overlap_ratio = 0.2   # 短查询20%
        
        filtered = []
        for d in results:
            if q_words:
                d_words = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", d["text"].lower()))
                overlap_count = len(q_words & d_words)
                overlap_ratio = overlap_count / len(q_words)
                
                # 调试：打印匹配情况
                print(f"[DEBUG] 查询词: {q_words}")
                print(f"[DEBUG] 文档匹配: {q_words & d_words}")
                print(f"[DEBUG] 重叠率: {overlap_ratio:.2%}")
                
                # 只要匹配到至少1个关键词就通过（更宽松）
                if overlap_count >= 1:
                    filtered.append(d)
            else:
                filtered.append(d)
        
        print(f"[DEBUG] 过滤后结果数: {len(filtered)}")
        
        # 如果过滤后为空，返回原始结果（不要返回空）
        if not filtered:
            # 直接返回前3个结果，不过滤
            filtered = results[:3]
            print("[DEBUG] 过滤结果为空，使用原始结果")
        
        if not filtered:
            return "未在知识库中找到相关信息。"
        
        context = ""
        for i, d in enumerate(filtered, 1):
            context += f"\n【文档片段 {i}】\n来源: {d['doc_name']} - {d['source_type']}\n内容: {d['text']}\n"
        return context.strip()

    # 核心：先增强普通函数，再用 @tool 转为工具！！！
    enhanced_search = tool_enhancer()(search_knowledge_base)
    search_knowledge_base = tool(enhanced_search)
    return search_knowledge_base


import ast
import operator

def safe_eval(node):
    """
    安全的 AST 表达式求值
    只允许数字、基本运算符和括号
    """
    # 允许的操作符映射
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,  # 一元负号
        ast.UAdd: operator.pos,  # 一元正号
    }
    
    if isinstance(node, ast.Constant):
        # 只允许数字
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"只支持数字，不支持: {type(node.value).__name__}")
        return node.value
    
    elif isinstance(node, ast.BinOp):
        # 二元运算
        left = safe_eval(node.left)
        right = safe_eval(node.right)
        op_type = type(node.op)
        if op_type not in operators:
            raise ValueError(f"不支持的操作符: {op_type.__name__}")
        return operators[op_type](left, right)
    
    elif isinstance(node, ast.UnaryOp):
        # 一元运算
        operand = safe_eval(node.operand)
        op_type = type(node.op)
        if op_type not in operators:
            raise ValueError(f"不支持的操作符: {op_type.__name__}")
        return operators[op_type](operand)
    
    elif isinstance(node, ast.Expression):
        # 表达式根节点
        return safe_eval(node.body)
    
    else:
        # 其他任何节点类型都不允许
        raise ValueError(f"不支持的表达式类型: {type(node).__name__}")


@tool
@tool_enhancer()
def calculator(expression: str) -> str:
    """
    执行数学计算，支持加减乘除等基本运算。

    Args:
        expression: 数学表达式，如 "2 + 3 * 4"

    Returns:
        计算结果
    """
    try:
        # 第一层防护：字符白名单
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "表达式包含非法字符，仅支持数字和基本运算符(+ - * /)"
        
        # 第二层防护：检查危险模式
        dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'globals', 'locals']
        expr_lower = expression.lower()
        for pattern in dangerous_patterns:
            if pattern in expr_lower:
                return f"表达式包含非法关键词: {pattern}"
        
        # 第三层防护：AST 安全解析
        tree = ast.parse(expression, mode='eval')
        
        # 验证 AST 中没有危险节点
        for node in ast.walk(tree):
            if isinstance(node, (ast.Call, ast.Attribute, ast.Name, ast.Subscript)):
                return f"不支持的语法结构: {type(node).__name__}"
        
        # 安全计算
        result = safe_eval(tree.body)
        
        # 第四层防护：结果范围限制
        if abs(result) > 1e12:
            return "计算结果超出允许范围"
        
        # 处理浮点数精度
        if isinstance(result, float):
            result = round(result, 10)
        
        return f"计算结果：{result}"
        
    except ValueError as e:
        return f"计算错误：{str(e)}"
    except Exception as e:
        return f"计算失败：{str(e)}"




@tool
@tool_enhancer()
def get_current_time() -> str:
    """
    获取当前日期和时间。

    Returns:
        当前日期和时间的字符串表示
    """
    now = dt.now()
    return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def create_agent_executor(chunks, chunk_embeddings, use_semantic_search, top_k: int = 5, temperature: float = 0.1):
    """创建 LangChain AgentExecutor（官方原生千问Function-Calling）"""
    # 1. 动态生成带当前知识库的检索工具（解决全局上下文BUG）
    search_tool = get_search_tool(chunks, chunk_embeddings, use_semantic_search, top_k)
    tools = [search_tool, calculator, get_current_time]

    # 2. 用LangChain官方ChatOpenAI适配千问，原生支持Function-Calling
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=temperature,
        openai_api_key=QWEN_API_KEY,
        openai_api_base=QWEN_API_BASE
    )

    # 3. 严格提示词，强制必须调用检索工具（带记忆上下文 + ReAct反思）
    prompt = ChatPromptTemplate.from_messages([
        ("system", """【记忆信息】{memory_context}
【铁律1：强制检索】所有知识类问题，**必须调用 search_knowledge_base**，严禁直接回答。
【铁律2：ReAct 反思校验（核心）】
1.  Thought：用户的问题是「{input}」，我需要找到**直接回答这个问题的原文内容**。
2.  Action：调用工具获取文档片段。
3.  Observation：逐条检查每个片段，判断是否直接回答用户问题。
4.  Reflection：
    - 若片段**直接回答问题**：只引用原文原话，不扩展、不总结、不推断。
    - 若片段**标题相似但内容不相关**：必须丢弃，严禁拼接进回答。
    - 若没有任何片段能直接回答：输出「未在知识库中找到直接相关的信息」。
【铁律3：回答格式】
- 开头必须写：`根据知识库原文，回答如下：`
- 只列原文内容，不添加任何解释、背景、推断。
- 无关内容（如其他章节的架构、安全、感知内容）一律不写！"""),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 4. 官方原版Agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=True
    )
    return agent_executor


def build_three_layer_memory(messages):
    """
    真正正确的三层记忆
    1. 短期：最近5轮（原文）
    2. 总结：5～10轮（压缩）
    3. 长期：核心信息（永久保存）
    """
    global long_term_memory_store
    
    # 1 轮 = 1用户 + 1助手 = 2条消息
    total_rounds = len(messages) // 2

    # ---------------------- 短期记忆：最近5轮 ----------------------
    short_term = messages[-10:] if len(messages) >= 10 else messages

    # ---------------------- 总结记忆：5～10轮 ----------------------
    summary_text = ""
    if total_rounds > 5:
        # 取 第5轮之前的所有内容 → 总结
        history_to_summarize = messages[:-10]
        if history_to_summarize:
            context = "\n".join([f"{m['role']}: {m['content']}" for m in history_to_summarize])
            prompt = f"""请把以下对话总结成3-5行重点：
1. 用户要什么
2. 已查询内容
3. 关键结果

对话：
{context}

总结："""
            try:
                summary_text = call_llm(prompt, temperature=0.1)
            except:
                summary_text = "对话历史已自动总结"

    # ---------------------- 长期记忆：提取核心需求 ----------------------
    long_term = ""
    if total_rounds >= 3:
        context_for_long_term = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt = f"""从对话提取用户永久关键信息：
- 核心需求
- 重要查询主题
- 关键信息

对话：
{context_for_long_term}

提取："""
        try:
            long_term = call_llm(prompt, temperature=0.1)
            # 添加到长期记忆存储（去重）
            for line in long_term.split("\n"):
                line = line.strip()
                if line and line not in long_term_memory_store:
                    long_term_memory_store.append(line)
        except:
            long_term = "未提取到长期记忆"
    
    # 使用累积的长期记忆
    if long_term_memory_store:
        long_term = "\n".join(long_term_memory_store)

    # ---------------------- 最终拼接 ----------------------
    memory_context = ""
    if long_term:
        memory_context += f"【长期记忆】\n{long_term}\n\n"
    if summary_text:
        memory_context += f"【对话总结】\n{summary_text}\n\n"

    # 短期记忆转 LangChain 格式
    chat_history = []
    for msg in short_term:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))

    return memory_context, chat_history


def extract_sources_from_retrieval(retrieval_result: str) -> list:
    """从检索结果中提取来源信息"""
    import re
    sources = []
    # 匹配来源行: 来源: xxx - xxx
    pattern = r'来源: ([^\n]+)'
    matches = re.findall(pattern, retrieval_result)
    for m in matches:
        if m not in sources:
            sources.append(m)
    return sources


def handle_strict_mode(question: str, retrieval_result: str, messages: List[dict], memory_context: str = "") -> dict:
    """严格模式：基于检索结果回答（防幻觉）"""
    
    # 构建对话历史
    history_text = ""
    if messages:
        recent = messages[-6:]
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent])
    
    prompt = f"""【严格模式】你是一个知识库问答助手，请根据下面提供的文档内容回答用户问题。

【记忆信息】
{memory_context}

【文档内容】
{retrieval_result}

【对话历史】
{history_text}

【用户问题】
{question}

【铁律】
1. 仔细阅读文档内容，找出与问题相关的信息
2. 如果文档包含表格数据，请从表格中查找匹配的行和列来回答问题
3. 用你自己的话总结回答，但必须基于文档内容
4. 如果文档中没有找到直接答案，回答"未在知识库中找到相关信息"
5. 禁止添加任何文档以外的知识或推断

【回答】"""

    answer = call_llm(prompt, temperature=0.0)
    
    # 提取来源
    sources = extract_sources_from_retrieval(retrieval_result)
    
    return {
        "answer": answer,
        "mode": "strict",
        "sources": sources,
        "retrieved_chunks": retrieval_result
    }


def handle_free_mode(question: str, messages: List[dict], source_note: str = None, memory_context: str = "") -> dict:
    """自由模式：LLM 自由回答（闲聊、历史、常识等）"""
    
    # 构建对话历史
    history_text = ""
    if messages:
        recent = messages[-10:]
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent])
    
    # 来源标注
    source_text = ""
    if source_note:
        source_text = f"\n\n【信息来源】{source_note}"
    
    prompt = f"""【自由模式】请根据对话历史和记忆信息回答用户的问题。

【记忆信息】
{memory_context}

【对话历史】
{history_text}

【用户问题】
{question}
{source_text}

【回答要求】
- 如果是问候，礼貌回复
- 如果是关于对话历史的问题，从上面的历史中回答
- 如果是常识问题，直接回答
- 如果不知道，诚实说不知道

【回答】"""

    answer = call_llm(prompt, temperature=0.7)
    
    return {
        "answer": answer,
        "mode": "free",
        "sources": [source_note] if source_note else [],
        "retrieved_chunks": None
    }


def run_agent_with_history(agent_executor: AgentExecutor, question: str, messages: List[dict],
                           temperature: float = 0.1):
    """双层路由：快速过滤 → 检索 → 判断相关性 → 选择模式"""
    
    # 构建三层记忆
    memory_context, chat_history = build_three_layer_memory(messages)
    
    try:
        # ========== 第一层：快速过滤明显不需要检索的问题 ==========
        is_non_retrieval, filter_type = is_obvious_non_retrieval(question)
        
        if is_non_retrieval:
            print(f"[路由] 快速过滤: {filter_type} → 自由模式（不检索）")
            result = handle_free_mode(question, messages, source_note=None, memory_context=memory_context)
            return {
                "answer": result["answer"],
                "sources": result["sources"],
                "retrieved_chunks": result["retrieved_chunks"],
                "summary": memory_context,
                "long_term_memory": long_term_memory_store.copy()
            }
        
        # ========== 第二层：强制检索 ==========
        print(f"[路由] 进入检索流程")
        
        # 获取检索工具
        search_tool = None
        for tool in agent_executor.tools:
            if tool.name == "search_knowledge_base":
                search_tool = tool
                break
        
        if not search_tool:
            # 没有检索工具，降级到自由模式
            result = handle_free_mode(question, messages, source_note="检索工具未配置", memory_context=memory_context)
            return {
                "answer": result["answer"],
                "sources": result["sources"],
                "retrieved_chunks": result["retrieved_chunks"],
                "summary": memory_context,
                "long_term_memory": long_term_memory_store.copy()
            }
        
        # 执行检索
        retrieval_result = search_tool.invoke({"query": question})
        
        # 判断检索结果是否相关
        is_relevant, score, reason = is_retrieval_relevant(question, retrieval_result)
        
        print(f"[路由] 相关性判断: is_relevant={is_relevant}, score={score}, reason={reason}")
        
        if is_relevant:
            # 相关 → 严格模式（基于检索结果回答）
            print(f"[路由] → 严格模式（基于知识库）")
            result = handle_strict_mode(question, retrieval_result, messages, memory_context=memory_context)
        else:
            # 不相关 → 自由模式 + 标注
            print(f"[路由] → 自由模式（知识库无相关内容）")
            source_note = f"知识库中未找到相关内容（相关性分数: {score}）"
            result = handle_free_mode(question, messages, source_note=source_note, memory_context=memory_context)
        
        # 解析检索结果为 chunks 格式
        retrieved_chunks = []
        if result["retrieved_chunks"]:
            lines = result["retrieved_chunks"].split("\n")
            current_chunk = {}
            for line in lines:
                if line.startswith("来源:"):
                    if current_chunk:
                        retrieved_chunks.append(current_chunk)
                    source_info = line.replace("来源:", "").strip()
                    parts = source_info.split(" - ")
                    current_chunk = {
                        "doc_name": parts[0] if len(parts) > 0 else "未知文档",
                        "source_type": parts[1] if len(parts) > 1 else "未知章节",
                        "text": ""
                    }
                elif line.startswith("内容:"):
                    if current_chunk:
                        current_chunk["text"] = line.replace("内容:", "").strip()
                elif line.startswith("【文档片段"):
                    continue
            if current_chunk:
                retrieved_chunks.append(current_chunk)
        
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_chunks": retrieved_chunks,
            "summary": memory_context,
            "long_term_memory": long_term_memory_store.copy()
        }
    
    except Exception as e:
        return {
            "answer": f"抱歉，生成回答时出错: {str(e)}\n建议检查：1. 千问API密钥/地址是否正确 2. Embedding API是否可用 3. 知识库是否加载成功",
            "sources": [],
            "retrieved_chunks": [],
            "summary": memory_context,
            "long_term_memory": long_term_memory_store.copy()
        }


# ========== 核心改造结束 ==========


def read_docx(file_path: str) -> list:
    """返回带文档名、来源类型的结构化数据"""
    doc_name = os.path.basename(file_path)
    doc = Document(file_path)
    chunks_with_source = []

    full_text = ""
    for para in doc.paragraphs:
        if para.text.strip():
            full_text += para.text + "\n"

    if full_text.strip():
        chunks_with_source.append({
            "text": full_text.strip(),
            "source_type": "Word文档",
            "doc_name": doc_name
        })

    return chunks_with_source


def read_pdf(file_path: str) -> list:
    """读取PDF文件，返回带来源信息的段落列表"""
    if PdfReader is None:
        raise ImportError("PyPDF2 未安装，请先安装：pip install PyPDF2")

    doc_name = os.path.basename(file_path)
    reader = PdfReader(file_path)
    chunks_with_source = []

    for page_idx, page in enumerate(reader.pages, 1):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            chunks_with_source.append({
                "text": page_text.strip(),
                "source_type": f"PDF第{page_idx}页",
                "doc_name": doc_name
            })
    return chunks_with_source


def flatten_multiindex_columns(df):
    """将 MultiIndex 列名扁平化为单层字符串"""
    new_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            parts = [str(c).strip() for c in col if str(c).strip()]
            new_col = "_".join(parts) if parts else f"列{len(new_columns) + 1}"
        else:
            new_col = str(col).strip() or f"列{len(new_columns) + 1}"
        new_columns.append(new_col)
    df.columns = new_columns
    return df


def read_excel(file_path: str, interactive: bool = True, header_config: dict = None) -> list:
    """读取Excel文件，支持多行表头，处理合并单元格，转换为树形结构"""
    try:
        from openpyxl import load_workbook
    except ImportError:
        raise ImportError("openpyxl 未安装，请先安装：pip install openpyxl")

    doc_name = os.path.basename(file_path)
    chunks_with_source = []

    wb = None
    try:
        wb = load_workbook(filename=file_path, read_only=False, data_only=True)
        sheet_names = wb.sheetnames

        for sheet_idx, sheet_name in enumerate(sheet_names, 1):
            if header_config and sheet_name in header_config:
                config = header_config[sheet_name]
                header_type = config.get('type', 1)
                header_rows = config.get('rows', 0)
            elif interactive:  
                header_type = 1
                header_rows = 0
            else:
                header_type = 1
                header_rows = 0

            ws = wb[sheet_name]

            data = []
            for row in ws.iter_rows(values_only=True):
                row_data = []
                for cell in row:
                    if cell is not None:
                        if isinstance(cell, (datetime.datetime, datetime.date)):
                            row_data.append(
                                cell.strftime('%Y-%m-%d %H:%M:%S') if hasattr(cell, 'hour') else cell.strftime(
                                    '%Y-%m-%d'))
                        else:
                            row_data.append(str(cell).strip())
                    else:
                        row_data.append("")
                data.append(row_data)

            if header_type == 1:
                header_row_idx = header_rows if isinstance(header_rows, int) else 0
                if header_row_idx < len(data):
                    headers = data[header_row_idx]
                    rows = data[header_row_idx + 1:]
                else:
                    headers = [f"列{i + 1}" for i in range(len(data[0]))] if data else []
                    rows = data
            elif header_type == 2:
                if not isinstance(header_rows, list):
                    header_rows = [0, 1]
                headers = []
                for col_idx in range(len(data[0]) if data else 0):
                    header_parts = []
                    for row_idx in header_rows:
                        if row_idx < len(data) and col_idx < len(data[row_idx]):
                            header_parts.append(data[row_idx][col_idx])
                    headers.append("_".join([h for h in header_parts if h]))
                start_row = max(header_rows) + 1 if header_rows else 1
                rows = data[start_row:] if start_row < len(data) else []
            else:
                headers = [f"列{i + 1}" for i in range(len(data[0]))] if data else []
                rows = data

            filled_rows = []
            for row in rows:
                filled_row = []
                for i, val in enumerate(row):
                    if val:
                        filled_row.append(val)
                    elif i > 0 and filled_rows:
                        filled_row.append(filled_rows[-1][i])
                    else:
                        filled_row.append(val)
                filled_rows.append(filled_row)

            result = excel_to_tree(filled_rows, headers, sheet_name)

            if result:
                chunks_with_source.append({
                    "text": result,
                    "source_type": f"Excel工作表{sheet_idx}-{sheet_name}",
                    "doc_name": doc_name
                })

        print(f"[DEBUG] {doc_name} 读取了 {len(chunks_with_source)} 个片段")
        for i, chunk in enumerate(chunks_with_source[:2]):  # 只打印前2个片段
            print(f"[DEBUG]  片段 {i+1}: {chunk['text'][:100]}...")
        
        return chunks_with_source
    except Exception as e:
        print(f"     [ERROR] {doc_name} 读取失败：{str(e)}")
        raise
    finally:
        if wb is not None:
            try:
                wb.close()
            except:
                pass


def excel_to_tree(rows: list, headers: list, sheet_name: str) -> str:
    """将Excel数据转换为树形结构文本"""
    if not rows or not headers:
        return ""

    lines = [f"【{sheet_name}】"]
    current_category = None
    category_items = []

    for row in rows:
        first_col = row[0] if row else ""

        if first_col and first_col != current_category:
            if current_category and category_items:
                lines.append(f"【{current_category}】")
                lines.extend(category_items)
                lines.append("")

            current_category = first_col
            category_items = []

        row_values = []
        for i, val in enumerate(row):
            if val and val != "nan" and val != "None":
                if i < len(headers):
                    row_values.append(f"{headers[i]}: {val}")
                else:
                    row_values.append(val)

        if row_values:
            item_text = " - ".join(row_values)
            category_items.append(f"- {item_text}")

    if current_category and category_items:
        lines.append(f"【{current_category}】")
        lines.extend(category_items)

    return "\n".join(lines)


def convert_to_tree(df, sheet_name: str) -> str:
    """将DataFrame转换为树形结构文本"""
    if df.empty:
        return ""

    lines = [f"【{sheet_name}】"]
    current_category = None
    category_items = []

    for _, row in df.iterrows():
        first_col = str(row.iloc[0]).strip() if not pd.isna(row.iloc[0]) else ""

        if first_col and first_col != current_category:
            if current_category and category_items:
                lines.append(f"【{current_category}】")
                lines.extend(category_items)
                lines.append("")

            current_category = first_col
            category_items = []

        row_values = []
        for col in range(len(row)):
            val = str(row.iloc[col]).strip()
            if val and val != "nan":
                row_values.append(val)

        if row_values:
            item_text = " - ".join(row_values)
            category_items.append(f"- {item_text}")

    if current_category and category_items:
        lines.append(f"【{current_category}】")
        lines.extend(category_items)

    return "\n".join(lines)


def simple_keyword_search(query: str, chunks: list, top_k: int = 3) -> list:
    """简单关键词匹配检索"""
    query_lower = query.lower()
    query_chars = set(query_lower)

    results = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk["text"]
        chunk_lower = chunk_text.lower()
        overlap = len(set(chunk_lower) & query_chars)
        if overlap > 0:
            results.append((i, overlap, chunk))

    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
        return chunks[:top_k]

    return [r[2] for r in results[:top_k]]


def semantic_search(query: str, chunks: list, chunk_embeddings: np.ndarray, top_k: int = 3) -> list:
    """基于阿里云 Embedding API 的语义相似度检索"""
    query_embedding = get_embeddings([query])
    similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]


def compress_messages(messages: List[dict], max_rounds: int = 5) -> List[dict]:
    """压缩对话历史，保留最近几轮并合并更早的对话"""
    if len(messages) <= max_rounds * 2:
        return messages

    recent = messages[-max_rounds * 2:]
    compressed = []

    for msg in recent:
        compressed.append(msg)

    return compressed


def reconstruct_query(question: str, history: List[dict]) -> str:
    """简化版：不做复杂重构，避免改变原意"""
    # 如果问题是完整的句子，直接返回
    if len(question) > 10 and not question.startswith(("它", "这个", "那个", "他", "她")):
        return question
    
    # 只有简短短语或代词时才重构
    if not history or len(history) == 0:
        return question
    
    # 其他情况原样返回
    return question


def setup_rag(doc_path: str, chunk_size: int = 2000, chunk_overlap: int = 300, progress_callback=None,
              header_config: dict = None):
    """初始化 RAG 系统，支持进度回调"""
    if not doc_path:
        raise ValueError("错误：文档路径未设置")

    if os.path.isfile(doc_path):
        doc_files = [doc_path]
    elif os.path.isdir(doc_path):
        doc_files = []
        skipped_files = []
        for root, _, files in os.walk(doc_path):
            for file in files:
                if file.startswith("~$") or file.startswith("."):
                    skipped_files.append(file)
                    continue
                lower_name = file.lower()
                if lower_name.endswith(".docx"):
                    doc_files.append(os.path.join(root, file))
                elif lower_name.endswith(".pdf"):
                    doc_files.append(os.path.join(root, file))
                elif lower_name.endswith(".xlsx") or lower_name.endswith(".xls"):
                    doc_files.append(os.path.join(root, file))

        if skipped_files and progress_callback:
            progress_callback(5, f"跳过 {len(skipped_files)} 个临时/隐藏文件")
    else:
        raise ValueError(f"路径无效: {doc_path}")

    if not doc_files:
        raise ValueError("未找到docx/PDF/Excel文档，请检查路径")

    if progress_callback:
        progress_callback(10, f"发现 {len(doc_files)} 个文档")

    all_segments = []
    total_files = len(doc_files)
    for idx, file_path in enumerate(doc_files):
        fname = os.path.basename(file_path)
        try:
            if file_path.lower().endswith(".docx"):
                seg = read_docx(file_path)
            elif file_path.lower().endswith(".pdf"):
                seg = read_pdf(file_path)
            elif file_path.lower().endswith(".xlsx") or file_path.lower().endswith(".xls"):
                seg = read_excel(file_path, interactive=False, header_config=header_config)
            else:
                continue
            all_segments.extend(seg)
        except Exception as e:
            print(f"     [ERROR] {fname} 读取失败：{e}")
            if progress_callback:
                progress_callback(10 + int((idx + 1) / total_files * 30), f"⚠️ {fname} 读取失败，跳过")

        if progress_callback:
            progress = 10 + int((idx + 1) / total_files * 30)
            progress_callback(progress, f"读取文档 {idx + 1}/{total_files}")

    total_chars = sum(len(s["text"]) for s in all_segments)

    if progress_callback:
        progress_callback(45, "分类型切分文档...")

    # 准备智能分段切分器（用于 Word/PDF 和过长的 Excel 行）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    # Embedding API 的最大输入长度限制（阿里云 DashScope 为 2048）
    max_embedding_length = 2000  # 留一点余量

    chunks = []
    for seg in all_segments:
        # 判断是否是 Excel 文件（通过 source_type 标识）
        if "Excel" in seg["source_type"]:
            # Excel：尽量保持完整一行，但超过长度限制时需要切分
            if len(seg["text"]) <= max_embedding_length:
                # 长度合适，直接作为完整 chunk
                chunks.append({
                    "text": seg["text"],
                    "source_type": seg["source_type"],
                    "doc_name": seg["doc_name"]
                })
            else:
                # 太长了，需要切分（虽然不完美，但至少能处理）
                print(f"[WARN] Excel行太长 ({len(seg['text'])}字符)，需要切分")
                segment_chunks = text_splitter.split_text(seg["text"])
                for chunk in segment_chunks:
                    chunks.append({
                        "text": chunk,
                        "source_type": seg["source_type"],
                        "doc_name": seg["doc_name"]
                    })
        else:
            # Word/PDF：使用智能分段切分（保证段落语义连贯）
            segment_chunks = text_splitter.split_text(seg["text"])
            for chunk in segment_chunks:
                chunks.append({
                    "text": chunk,
                    "source_type": seg["source_type"],
                    "doc_name": seg["doc_name"]
                })
    
    print(f"[DEBUG] 总片段数: {len(all_segments)}, 切分后chunks数: {len(chunks)}")

    if progress_callback:
        progress_callback(60, "计算文档向量...")

    chunk_texts = [c["text"] for c in chunks]
    chunk_embeddings = get_embeddings(chunk_texts)

    if progress_callback:
        progress_callback(90, "初始化完成")

    return {
        "chunks": chunks,
        "chunk_embeddings": chunk_embeddings,
        "doc_files": doc_files,
        "total_chars": total_chars,
        "num_chunks": len(chunks)
    }


def main():
    # st.set_page_config(page_title="RAG 知识库问答系统", page_icon="💬", layout="wide")
    st.set_page_config(
        page_title="RAG 知识库问答系统",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
<style>
/* 强制把侧边栏顶部空白干掉 */
section[data-testid="stSidebar"] > div:first-child  {
    padding-top: 0px !important;
    margin-top: -20px !important;
}

/* 自定义滚动条 */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: #f1f1f1;
}
::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #a1a1a1;
}
</style>
""", unsafe_allow_html=True)
    
    # 验证配置
    config_valid, missing_configs, config_source = validate_config()

    # 配置不完整时显示错误页面
    if not config_valid:
        st.error("❌ 配置不完整")
        st.warning(f"当前配置来源: {config_source or '未找到'}")
        st.markdown("### 需要配置以下环境变量：")
        for missing in missing_configs:
            st.markdown(f"- **{missing}**")
        
        st.markdown("""
        ### 配置方法：
        
        **本地开发（使用 .env 文件）：**
        在项目根目录创建 `.env` 文件：
        ```
        QWEN_API_KEY=your_api_key
        QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
        MODEL_NAME=qwen-plus
        EMBEDDING_API_KEY=your_embedding_key
        EMBEDDING_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings
        EMBEDDING_MODEL=text-embedding-v2
        ```
        
        **Streamlit Cloud 部署（使用 Secrets）：**
        在 Streamlit Community Cloud 控制台的 **Settings > Secrets** 中添加：
        ```toml
        QWEN_API_KEY = "your_api_key"
        QWEN_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        MODEL_NAME = "qwen-plus"
        EMBEDDING_API_KEY = "your_embedding_key"
        EMBEDDING_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
        EMBEDDING_MODEL = "text-embedding-v2"
        ```
        """)
        return

    # 初始化会话状态
    session_defaults = {
        "messages": [],
        "rag_initialized": False,
        "chunks": [],
        "chunk_embeddings": None,
        "use_semantic_search": True,
        "search_mode": "semantic",
        "excel_header_mode": "默认",
        "doc_files": [],
        "total_chars": 0,
        "num_chunks": 0,
        "doc_path": DOC_PATH,
        "header_config": {},
        "show_source_formats": False,
        "agent_executor": None,
        "temperature": 0.1,
        "top_k": 5,
        "chunk_size": 2000,
        "chunk_overlap": 300,
        "excel_sheets": {}
    }
    
    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    with st.sidebar:
        st.title("📚 知识库配置")
        
        # 配置状态指示
        if config_source:
            st.success(f"✅ 配置已加载 ({config_source})")
        else:
            st.warning("⚠️ 配置来源未知")

        st.subheader("知识库路径")
        new_doc_path = st.text_input(
            "文档路径",
            value=st.session_state.doc_path,
            placeholder="请输入文档文件夹或文件路径"
        )
        if st.button("保存路径"):
            st.session_state.doc_path = new_doc_path
            st.success("路径已保存")

        st.markdown("---")

        st.subheader("知识库状态")

        if st.session_state.rag_initialized:
            st.success("✅ 知识库已加载")
            st.write(f"📄 文档数量: {len(st.session_state.doc_files)}")
            st.write(f"📊 总字符数: {st.session_state.total_chars:,}")
            st.write(f"🔢 向量数量: {st.session_state.num_chunks}")

            with st.expander("📋 文件列表"):
                for idx, doc in enumerate(st.session_state.doc_files, 1):
                    st.write(f"{idx}. {os.path.basename(doc)}")
        else:
            st.warning("⚠️ 知识库未初始化")

        st.markdown("---")

        st.session_state.excel_header_mode = st.selectbox(
            "Excel表头配置模式",
            options=["默认", "自定义"],
            index=0 if st.session_state.excel_header_mode == "默认" else 1,
            help="默认：使用第1行作为单行表头；自定义：手动配置每个工作表的表头"
        )

        # 如果是自定义模式，先显示Excel配置界面
        if st.session_state.excel_header_mode == "自定义":
            # 先获取Excel文件列表
            excel_files = []
            if os.path.exists(st.session_state.doc_path):
                if os.path.isdir(st.session_state.doc_path):
                    for root, _, files in os.walk(st.session_state.doc_path):
                        for file in files:
                            if not file.startswith("~$") and (
                                    file.lower().endswith(".xlsx") or file.lower().endswith(".xls")):
                                excel_files.append(os.path.join(root, file))
                elif st.session_state.doc_path.lower().endswith(".xlsx") or st.session_state.doc_path.lower().endswith(
                        ".xls"):
                    excel_files = [st.session_state.doc_path]
            
            if excel_files:
                st.subheader("Excel 表头配置")
                st.write("请为每个工作表配置表头类型：")
                
                # 获取Excel工作表信息
                if "excel_sheets" not in st.session_state or st.session_state.excel_sheets == {}:
                    excel_sheets = {}
                    for excel_file in excel_files:
                        try:
                            from openpyxl import load_workbook
                            wb = load_workbook(filename=excel_file, read_only=True)
                            excel_sheets[excel_file] = wb.sheetnames
                            wb.close()
                        except Exception as e:
                            print(f"读取Excel文件失败: {e}")
                    st.session_state.excel_sheets = excel_sheets
                
                # 配置每个工作表
                for excel_file, sheets in st.session_state.excel_sheets.items():
                    st.write(f"**文件: {os.path.basename(excel_file)}**")
                    for sheet in sheets:
                        header_type = st.selectbox(
                            f"{sheet} - 表头类型",
                            options=["单行表头", "多行表头", "无表头"],
                            index=0,
                            key=f"header_type_{sheet}"
                        )

                        if header_type == "单行表头":
                            row_num = st.number_input(
                                f"{sheet} - 表头行号（从1开始）",
                                min_value=1,
                                value=1,
                                key=f"header_row_{sheet}"
                            )
                            st.session_state.header_config[sheet] = {
                                'type': 1,
                                'rows': row_num - 1
                            }
                        elif header_type == "多行表头":
                            range_input = st.text_input(
                                f"{sheet} - 表头行号范围（如 1-3）",
                                value="1-2",
                                key=f"header_range_{sheet}"
                            )
                            if '-' in range_input:
                                try:
                                    start, end = range_input.split('-')
                                    start_row = int(start.strip())
                                    end_row = int(end.strip())
                                    if start_row >= 1 and end_row >= start_row:
                                        header_rows = list(range(start_row - 1, end_row))
                                        st.session_state.header_config[sheet] = {
                                            'type': 2,
                                            'rows': header_rows
                                        }
                                    else:
                                        st.warning("无效范围，请确保起始行<=结束行")
                                except ValueError:
                                    st.warning("请输入正确格式，如 '1-3'")
                        else:
                            st.session_state.header_config[sheet] = {
                                'type': 0,
                                'rows': 0
                            }
            else:
                st.info("当前路径下没有Excel文件，无需配置")

        st.markdown("---")

        st.subheader("操作")

        if st.button("🔄 重新加载知识库"):
            if not st.session_state.doc_path:
                st.error("请先设置文档路径")
            else:
                # 获取Excel文件列表
                excel_files = []
                if os.path.isdir(st.session_state.doc_path):
                    for root, _, files in os.walk(st.session_state.doc_path):
                        for file in files:
                            if not file.startswith("~$") and (
                                    file.lower().endswith(".xlsx") or file.lower().endswith(".xls")):
                                excel_files.append(os.path.join(root, file))
                elif st.session_state.doc_path.lower().endswith(".xlsx") or st.session_state.doc_path.lower().endswith(
                        ".xls"):
                    excel_files = [st.session_state.doc_path]

                # 如果是默认模式，自动生成配置
                if st.session_state.excel_header_mode == "默认" and excel_files:
                    st.session_state.header_config = {}
                    for excel_file in excel_files:
                        try:
                            from openpyxl import load_workbook
                            wb = load_workbook(filename=excel_file, read_only=True)
                            for sheet_name in wb.sheetnames:
                                st.session_state.header_config[sheet_name] = {
                                    'type': 1,
                                    'rows': 0
                                }
                            wb.close()
                        except Exception as e:
                            print(f"读取Excel文件失败: {e}")
                
                # 直接调用加载
                perform_load()

        if st.button("🗑️ 清空对话历史"):
            st.session_state.messages = []
            st.success("对话历史已清空")

        if st.button("📖 查看支持格式"):
            st.session_state.show_source_formats = not st.session_state.show_source_formats

        if st.session_state.show_source_formats:
            st.info("""**支持的文件格式：**
- 📄 Word 文档 (.docx)
- 📕 PDF 文件 (.pdf)
- 📊 Excel 表格 (.xlsx, .xls)

**功能特性：**
- ✅ 语义检索 + 关键词检索降级
- ✅ 多轮对话与指代消解
- ✅ 闲聊识别
- ✅ 答案溯源
- ✅ 对话历史压缩
- ✅ 复杂表格处理""")
        st.markdown("---")

        st.subheader("参数设置")

        st.session_state.temperature = st.slider(
            "LLM Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="控制回答的随机性，越低越保守"
        )

        st.session_state.top_k = st.slider(
            "检索 Top-K",
            min_value=1,
            max_value=10,
            value=st.session_state.top_k,
            step=1,
            help="检索返回的文档数量"
        )

        st.session_state.chunk_size = st.slider(
            "Chunk 大小",
            min_value=500,
            max_value=5000,
            value=st.session_state.chunk_size,
            step=100,
            help="文档分割的块大小"
        )

        st.session_state.chunk_overlap = st.slider(
            "Chunk 重叠",
            min_value=0,
            max_value=500,
            value=st.session_state.chunk_overlap,
            step=50,
            help="相邻块之间的重叠字符数"
        )

        st.markdown("---")
        st.subheader("检索模式")
        st.session_state.search_mode = st.radio(
            "选择检索模式",
            options=["semantic", "keyword"],
            format_func=lambda x: "🔍 语义检索" if x == "semantic" else "🔤 关键词检索",
            index=0 if st.session_state.search_mode == "semantic" else 1,
            help="语义检索：基于向量相似度匹配，理解语义意图；关键词检索：基于字符匹配，速度更快"
        )
        st.session_state.use_semantic_search = (st.session_state.search_mode == "semantic")

    # 主聊天界面
    # st.title("💬 RAG 知识库问答系统")

    if not st.session_state.rag_initialized:
        st.info("请先在侧边栏设置文档路径并加载知识库")
        return

    if not st.session_state.messages:
        st.info("知识库已就绪！请输入您的问题")

    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar="💬"):
                content = message["content"]

                if "【信息来源】" in content:
                    answer_part, source_part = content.split("【信息来源】", 1)
                    st.markdown(answer_part.strip())

                    with st.expander("📚 信息来源"):
                        st.markdown("**【信息来源】**" + source_part)

                        if "【检索到的文档片段】" in source_part:
                            source_part, chunks_part = source_part.split("【检索到的文档片段】", 1)
                            st.markdown("**【检索到的文档片段】**" + chunks_part)
                elif "【检索到的文档片段】" in content:
                    answer_part, chunks_part = content.split("【检索到的文档片段】", 1)
                    st.markdown(answer_part.strip())

                    with st.expander("📚 检索到的文档片段"):
                        st.markdown("**【检索到的文档片段】**" + chunks_part)
                else:
                    st.markdown(content)

    if prompt := st.chat_input("请输入您的问题..."):
        if not st.session_state.rag_initialized:
            st.warning("请先加载知识库")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="💬"):
            with st.spinner("正在思考..."):
                if st.session_state.agent_executor:
                    result = run_agent_with_history(
                        st.session_state.agent_executor,
                        prompt,
                        st.session_state.messages[:-1],
                        temperature=st.session_state.temperature
                    )

                    answer = result.get("answer", "")
                    sources = result.get("sources", [])
                    retrieved_chunks = result.get("retrieved_chunks", [])

                    st.markdown(answer)

                    # 显示信息来源（可折叠式，始终显示）
                    with st.expander("📚 信息来源"):
                        if sources or retrieved_chunks:
                            if sources:
                                st.markdown("**检索到的文档：**")
                                for s in sources:
                                    st.markdown(f"  - {s}")

                            if retrieved_chunks:
                                st.markdown("\n**参考片段：**")
                                for i, chunk in enumerate(retrieved_chunks, 1):
                                    st.markdown(f"  *来自: {chunk.get('doc_name', '未知文档')}*")
                                    st.markdown(f"  ```\n  {chunk.get('text', '')[:200]}...\n  ```")
                        else:
                            st.markdown("⚠️ **未找到知识库文档信息**（回答可能基于模型通用知识）")

                    new_messages = st.session_state.messages + [{"role": "assistant", "content": answer}]
                    if len(new_messages) > 20:
                        new_messages = compress_messages(new_messages, max_rounds=5)
                    st.session_state.messages = new_messages

                else:
                    st.error("Agent 未初始化，请重新加载知识库")


def perform_load():
    """执行知识库加载的通用函数"""
    with st.spinner("正在加载知识库..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(progress, message):
            progress_bar.progress(progress)
            status_text.text(message)

        try:
            result = setup_rag(
                st.session_state.doc_path,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap,
                progress_callback=progress_callback,
                header_config=st.session_state.header_config
            )
            st.session_state.chunks = result["chunks"]
            st.session_state.chunk_embeddings = result["chunk_embeddings"]
            st.session_state.doc_files = result["doc_files"]
            st.session_state.total_chars = result["total_chars"]
            st.session_state.num_chunks = result["num_chunks"]

            # 创建 AgentExecutor
            st.session_state.agent_executor = create_agent_executor(
                chunks=result["chunks"],
                chunk_embeddings=result["chunk_embeddings"],
                use_semantic_search=st.session_state.use_semantic_search,
                top_k=st.session_state.top_k,
                temperature=st.session_state.temperature
            )

            st.session_state.rag_initialized = True
            if "show_excel_config" in st.session_state:
                st.session_state.show_excel_config = False
            st.success("知识库加载成功！")

            # 新增：验证加载结果，在侧边栏显示
            if 'chunks' in st.session_state and st.session_state.chunks:
                st.sidebar.success(f"✅ 知识库加载成功！共加载 {len(st.session_state.chunks)} 个文档片段")
                # 显示第一个片段的预览，确认内容正确
                st.sidebar.info(f"📄 示例片段：{st.session_state.chunks[0]['text'][:100]}...")
                
                # 调试：验证知识库内容
                print(f"[DEBUG] 成功加载 {len(result['chunks'])} 个片段")
                print(f"[DEBUG] 第一个片段来源: {result['chunks'][0]['source_type']}")
                print(f"[DEBUG] 第一个片段内容: {result['chunks'][0]['text'][:200]}")
                
                # 检查是否包含关键词
                all_text = " ".join([c["text"] for c in result["chunks"][:10]])
                keywords = ["决策", "中间件", "MoveIt", "双臂", "升降"]
                for kw in keywords:
                    if kw in all_text:
                        print(f"[DEBUG] 找到关键词: {kw}")
                    else:
                        print(f"[DEBUG] 未找到关键词: {kw}")
            else:
                st.sidebar.error("❌ 知识库加载失败！未检测到文档片段")
                print("[ERROR] 没有加载到任何文档片段！")
        except Exception as e:
            st.error(f"加载失败: {str(e)}")


if __name__ == "__main__":
    main()
