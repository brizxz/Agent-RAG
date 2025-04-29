# 匯入最新套件
import logging
import os
import datetime
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama  # 更新正確的 import
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent

# 配置日誌
def setup_logging(log_dir="logs"):
    """設置日誌系統"""
    # 確保日誌目錄存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成日誌文件名（使用當前日期和時間）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"qa_agent_{timestamp}.log")
    
    # 配置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同時輸出到控制台
        ]
    )
    
    logging.info(f"日誌初始化完成，日誌文件：{log_file}")
    return logging.getLogger()

# 初始化日誌
logger = setup_logging()

# 注意：請確保 Ollama 伺服器已啟動
# 在終端機運行：ollama serve
# 如果在 NTU 環境無法運行 Ollama，可設置 base_url 指向可用的 Ollama API
logger.info("開始初始化 QA Agent 系統")

# 1. 準備知識庫並建立向量索引
logger.info("加載知識庫文件")
documents = [
    "產品A說明：產品A是一款針對數據分析的軟體工具，可支援即時圖表繪製和大型資料處理。",
    "公司政策：所有員工每年享有14天的年假，年假需在年度內使用完畢，不可累積到下一年。",
    # ... 可以加入更多文件
]
# 將文件切分成適合的段落（避免向量維度過大）
logger.info("將文件切分成適合的段落")
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs_chunks = []
for doc in documents:
    docs_chunks += text_splitter.split_text(doc)
# 建立向量資料庫（使用 Ollama Embedding 將文本轉向量）
logger.info("建立向量資料庫，使用模型：deepseek-r1:8b")
try:
    embeddings = OllamaEmbeddings(model="deepseek-r1:8b")
    vector_store = FAISS.from_texts(docs_chunks, embedding=embeddings)
    logger.info("向量資料庫建立成功")
except Exception as e:
    logger.error(f"建立向量資料庫失敗: {e}")
    raise

# 2. 建立問答 Chain，使用向量檢索器
logger.info("初始化 ChatOllama 模型和 RetrievalQA Chain")
try:
    llm = ChatOllama(model="deepseek-r1:8b", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 直接將所有檢索結果"塞"給LLM
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),  # 每次檢索 top2 條
        return_source_documents=True  # 返回來源文件以供引用（如果需要）
    )
    logger.info("問答Chain建立成功")
except Exception as e:
    logger.error(f"建立問答Chain失敗: {e}")
    raise

# 3. 建立對話記憶模組
logger.info("初始化對話記憶模組")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 4. （選擇性）定義其他工具，例如一個簡單計算器工具
logger.info("註冊工具：計算器工具")
def simple_calculator(expression: str) -> str:
    """Evaluate a basic math expression and return the result."""
    logger.info(f"使用計算器工具計算: {expression}")
    try:
        result = eval(expression)
        logger.info(f"計算結果: {result}")
        return str(result)
    except Exception as e:
        error_msg = f"計算錯誤：{str(e)}"
        logger.error(error_msg)
        return error_msg
calculator_tool = Tool(
    name="Calculator",
    func=simple_calculator,
    description="用於計算數學表達式的工具，例如輸入 '2+2' 或 '10*5'."
)
# 將問答Chain本身也作為一個工具
logger.info("註冊工具：知識庫問答工具")
qa_tool = Tool(
    name="KnowledgeBaseQA",
    func=qa_chain.invoke,  # 使用 invoke 替代 run
    description="問答工具，基於公司知識庫回答問題。輸入自然語言問題。"
)

# 5. 初始化 Agent，具備對話能力和多個工具
logger.info("初始化 Agent")
try:
    agent = initialize_agent(
        tools=[qa_tool, calculator_tool],
        llm=llm,
        agent="chat-conversational-react-description",  # 使用具有對話能力的React Agent
        memory=memory,
        verbose=True
    )
    logger.info("Agent初始化成功")
except Exception as e:
    logger.error(f"初始化Agent失敗: {e}")
    raise

# 6. 模擬連續對話
if __name__ == "__main__":
    logger.info("開始執行對話測試")
    print("注意：這個程式需要運行 Ollama 服務器。在 NTU 環境可能需要設置代理或使用 OpenAI 替代。")
    try:
        logger.info("提問: 嗨，請問產品A能做什麼？")
        response = agent.invoke("嗨，請問產品A能做什麼？")  # 使用 invoke 替代 run
        logger.info(f"回答: {response}")
        print(response)
        
        logger.info("提問: 好的，那員工年假有多少天？")
        response = agent.invoke("好的，那員工年假有多少天？")
        logger.info(f"回答: {response}")
        print(response)
        
        logger.info("提問: 順帶一提，2+2等於多少？")
        response = agent.invoke("順帶一提，2+2等於多少？")
        logger.info(f"回答: {response}")
        print(response)
        
        logger.info("對話測試結束")
    except Exception as e:
        logger.error(f"連接錯誤: {e}")
        print(f"連接錯誤: {e}")
        print("\n如果無法連接 Ollama 伺服器，您可能需要：")
        print("1. 安裝並啟動 Ollama: https://ollama.com/download")
        print("2. 運行 'ollama pull deepseek-r1:8b' 下載模型")
        print("3. 如果在 NTU 環境無法使用，考慮使用 OpenAI 模型替代")
