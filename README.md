# 🤖 RAG 知识库问答系统

基于 LangChain 和阿里云通义千问构建的知识库问答系统，支持文档上传、语义检索、多轮对话等功能。

## ✨ 功能特性

- **📄 多格式文档支持**：支持 Word (.docx)、PDF (.pdf)、Excel (.xlsx/.xls) 文件上传
- **🔍 智能检索**：支持语义检索（基于向量相似度）和关键词检索两种模式
- **🤝 多轮对话**：支持上下文记忆，理解指代消解
- **📊 复杂表格处理**：智能处理 Excel 表格数据，支持多行表头
- **🔒 安全计算**：内置安全计算器，支持基本数学运算
- **📱 友好界面**：基于 Streamlit 的现代化 Web 界面

## 🛠️ 技术栈

- **框架**: Streamlit
- **LLM**: 阿里云通义千问 (Qwen)
- **向量检索**: 阿里云 Embedding API
- **文档处理**: PyPDF2, python-docx, openpyxl
- **LangChain**: langchain, langchain-openai, langchain-classic

## 📦 安装

```bash
# 克隆仓库
git clone <repository-url>
cd rag_app

# 安装依赖
pip install -r requirements.txt
```

## 🔧 配置

### 本地开发

在项目根目录创建 `.env` 文件：

```env
# 千问API配置
QWEN_API_KEY=your_api_key
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen-plus

# Embedding API配置
EMBEDDING_API_KEY=your_embedding_key
EMBEDDING_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings
EMBEDDING_MODEL=text-embedding-v2
```

### Streamlit Cloud 部署

在 Streamlit Community Cloud 控制台的 **Settings > Secrets** 中添加：

```toml
QWEN_API_KEY = "your_api_key"
QWEN_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"
EMBEDDING_API_KEY = "your_embedding_key"
EMBEDDING_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-v2"
```

## 🚀 运行

### 本地运行

```bash
streamlit run app.py
```

### 部署到 Streamlit Community Cloud

1. 将代码上传到 GitHub
2. 访问 [Streamlit Community Cloud](https://share.streamlit.io/)
3. 创建新应用，选择您的仓库和 `app.py` 文件
4. 在 Secrets 中配置 API 密钥（如上）
5. 点击 Deploy 按钮

## 📖 使用说明

1. **上传文档**：在侧边栏点击"上传知识库文件"，选择 Word、PDF 或 Excel 文件
2. **加载知识库**：点击"加载知识库"按钮，系统会自动处理文档并生成向量索引
3. **提问**：在聊天框中输入问题，系统会基于知识库内容进行回答
4. **查看来源**：回答下方会显示信息来源和参考片段

## ⚙️ 参数设置

- **LLM Temperature**：控制回答的随机性（0-1），越低越保守
- **检索 Top-K**：检索返回的文档数量（1-10）
- **Chunk 大小**：文档分割的块大小（500-5000字符）
- **Chunk 重叠**：相邻块之间的重叠字符数（0-500）
- **检索模式**：语义检索（理解语义意图）或关键词检索（字符匹配）

## 📁 项目结构

```
rag_app/
├── app.py           # 主应用文件
├── requirements.txt # 依赖列表
├── .gitignore       # Git 忽略配置
├── .env             # 本地配置文件（不上传GitHub）
└── README.md        # 项目说明文档
```

## 📝 注意事项

1. 确保 API 密钥有效且余额充足
2. 上传的文档大小建议不超过 50MB
3. 首次加载知识库可能需要一些时间
4. 建议使用语义检索以获得更好的检索效果

## 📄 支持的文件格式

- 📄 Word 文档 (.docx)
- 📕 PDF 文件 (.pdf)
- 📊 Excel 表格 (.xlsx, .xls)

---

**开发团队**: RAG Knowledge Base Team
**版本**: 1.0.0
