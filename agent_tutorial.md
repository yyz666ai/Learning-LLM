# LangChain Agent 入门教程

## 什么是Agent？

Agent（智能代理）是一个由大语言模型驱动的自主系统，它可以：
1. 理解用户的指令
2. 规划完成任务的步骤
3. 使用各种工具来完成任务
4. 根据工具的反馈调整行动

简单来说，Agent就像是一个聪明的助手，它知道什么时候该用什么工具来帮你完成任务。

## 基础环境准备

首先，我们需要安装必要的包：

```bash
pip install langchain langchain-core langchain-community langchain-openai
```

## 1. 创建自定义工具

让我们从一个简单的例子开始 - 创建一个计算文本字数的工具：

```python
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.agents import create_structured_chat_agent, AgentExecutor
from langchain_community import hub
from langchain_core.memory import ConversationBufferMemory

# 创建一个自定义工具
class TextLengthTool(BaseTool):
    name = "文本字数计算工具"
    description = "当你需要计算文本的字数时，使用此工具"
    
    def _run(self, text: str) -> int:
        """计算文本的字数"""
        return len(text)

# 初始化OpenAI模型
model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

# 创建工具列表
tools = [TextLengthTool()]

# 获取Agent提示模板
prompt = hub.pull("hwchase17/structured-chat-agent")

# 创建Agent
agent = create_structured_chat_agent(
    llm=model,
    tools=tools,
    prompt=prompt
)

# 添加对话记忆
memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True
)

# 创建Agent执行器
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools, 
    memory=memory, 
    verbose=True,  # 显示执行过程
    handle_parsing_errors=True  # 处理解析错误
)

# 使用Agent
response = agent_executor.invoke({
    "input": "'君不见黄河之水天上来奔流到海不复回'，这句话的字数是多少？"
})
```

## 2. 使用内置Python工具

LangChain提供了许多内置工具，比如Python REPL工具，可以执行Python代码：

```python
from langchain_community.agents.agent_toolkits import create_python_agent
from langchain_community.tools import PythonREPLTool
from langchain_openai import ChatOpenAI

# 创建Python Agent
agent_executor = create_python_agent(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    tool=PythonREPLTool(),
    verbose=True,
    agent_executor_kwargs={"handle_parsing_errors": True}
)

# 使用Agent计算复杂数学问题
response = agent_executor.invoke({
    "input": "7的2.3次方是多少？"
})

# 使用Agent编写和执行代码
response = agent_executor.invoke({
    "input": "第12个斐波那契数列的数字是多少？"
})
```

## 代码说明

### 1. 自定义工具
- `BaseTool`: 所有自定义工具的基类
- `name`: 工具的名称，Agent会根据这个决定何时使用该工具
- `description`: 工具的描述，帮助Agent理解工具的用途
- `_run`: 实际执行工具功能的方法

### 2. Agent组件
- `ChatOpenAI`: 使用OpenAI的语言模型
- `ConversationBufferMemory`: 存储对话历史
- `AgentExecutor`: 协调Agent和工具的执行器
- `create_structured_chat_agent`: 创建结构化的对话Agent

### 3. 参数说明
- `temperature=0`: 确保输出的一致性
- `verbose=True`: 显示执行过程，方便调试
- `handle_parsing_errors=True`: 自动处理解析错误

## 最佳实践

1. **工具设计**
   - 工具功能要单一明确
   - 描述要清晰准确
   - 处理好异常情况

2. **Agent使用**
   - 给予清晰的指令
   - 使用合适的temperature值
   - 开启verbose模式方便调试

3. **内存管理**
   - 根据需要选择合适的内存类型
   - 定期清理不必要的历史记录
   - 注意内存大小限制

## 常见问题

1. **Agent不使用工具**
   - 检查工具描述是否清晰
   - 确认输入指令是否明确
   - 调整提示模板

2. **执行结果不准确**
   - 降低temperature值
   - 优化工具实现
   - 完善错误处理

3. **性能问题**
   - 减少不必要的API调用
   - 优化工具执行效率
   - 使用适当的缓存策略

## 进阶使用

1. **链接多个工具**
   ```python
   from langchain_core.tools import Tool
   tools = [Tool1(), Tool2(), Tool3()]
   ```

2. **自定义提示模板**
   ```python
   from langchain_core.prompts import PromptTemplate
   custom_prompt = PromptTemplate(...)
   ```

3. **使用不同的内存类型**
   ```python
   from langchain_core.memory import ConversationBufferWindowMemory
   memory = ConversationBufferWindowMemory(k=5)  # 只保留最近5轮对话
   ```

## Agent创建和调用方式对比

在LangChain中，有多种创建和调用Agent的方式，让我们来了解它们的区别：

### 1. AgentExecutor创建方式对比

```python
# 方式1：使用from_agent_and_tools方法（推荐）
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, 
    tools=tools, 
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

# 方式2：直接初始化
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)
```

**主要区别：**
- `from_agent_and_tools`方法是一个工厂方法，提供了更多的配置选项：
  - 可以配置memory（对话历史记录）
  - 可以配置handle_parsing_errors（错误处理）
  - 内部进行了一些优化和验证
- 直接初始化方式更简单，但功能相对受限

### 2. Agent创建方式对比

```python
# 方式1：创建结构化对话Agent
agent = create_structured_chat_agent(
    llm=model,
    tools=tools,
    prompt=prompt
)

# 方式2：创建工具调用Agent
agent = create_tool_calling_agent(
    llm=model,
    tools=tools,
    prompt=prompt
)
```

**主要区别：**
- `create_structured_chat_agent`：
  - 专门为对话场景优化
  - 使用结构化的输出格式
  - 更适合需要自然对话的场景
- `create_tool_calling_agent`：
  - 更专注于工具调用
  - 输出格式更简洁
  - 适合纯工具调用场景

### 3. Agent调用方式

无论使用哪种方式创建Agent，调用方式都是统一的：

```python
# 使用invoke方法（推荐）
response = agent_executor.invoke({
    "input": "你的问题或指令"
})

# 使用run方法（旧版本兼容）
response = agent_executor.run("你的问题或指令")
```

**主要区别：**
- `invoke`方法：
  - 返回更丰富的信息
  - 支持结构化输入
  - 是新版本推荐的方法
- `run`方法：
  - 只返回最终结果
  - 主要为了兼容旧版本
  - 输入格式较为简单

### 最佳实践建议

1. **创建AgentExecutor时：**
   - 优先使用`from_agent_and_tools`方法
   - 需要记忆功能时添加memory参数
   - 开发阶段建议开启verbose模式

2. **选择Agent类型时：**
   - 对话场景选择`create_structured_chat_agent`
   - 纯工具调用场景选择`create_tool_calling_agent`

3. **调用Agent时：**
   - 优先使用`invoke`方法
   - 需要处理中间结果时使用`invoke`的返回值

## 注意事项

1. API密钥安全
   - 使用环境变量存储API密钥
   - 不要在代码中硬编码密钥
   - 定期轮换密钥

2. 错误处理
   - 添加适当的超时机制
   - 实现重试逻辑
   - 记录错误日志

3. 资源管理
   - 控制并发请求数
   - 监控API使用量
   - 实现速率限制

## 结语

Agent是一个强大的工具，可以帮助我们自动化很多任务。通过合理的设计和使用，它可以成为我们的得力助手。记住，好的Agent不仅要能完成任务，还要有良好的用户体验和稳定的性能。 