# LangChain 教程：AI 模型与输入输出处理

本教程将通过实际的代码示例，带你深入了解 LangChain 框架的核心功能。我们将按照从基础到进阶的顺序，逐步讲解 LangChain 的各个重要组件。

## 1. 基础模型使用

### 1.1 初始化聊天模型

在 LangChain 中，我们使用 langchain_openai 包来调用 OpenAI 的模型。以下是使用 ChatGPT (GPT-3.5-turbo) 的示例：

```python
from langchain_openai import ChatOpenAI

# 初始化模型
model = ChatOpenAI(model="gpt-3.5-turbo")

# 如果使用自定义 API，可以指定 base URL
# model = ChatOpenAI(
#     model="gpt-3.5-turbo",
#     openai_api_key="<你的API密钥>",
#     openai_api_base="https://api.aigc369.com/v1"
# )

# 你也可以设置一些参数来控制模型的行为
# temperature: 控制输出的随机性，值越高输出越随机，范围0-2，默认1
# max_tokens: 控制输出的最大长度
# model_kwargs: 可以传入其他模型参数
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=1.2,
    max_tokens=500,
    model_kwargs={"frequency_penalty": 1.1}
)
```

### 1.2 基本对话

LangChain 提供了结构化的消息类型，可以清晰地区分系统消息和用户消息：

```python
from langchain_core.messages import HumanMessage, SystemMessage

# 创建对话消息
messages = [
    SystemMessage(content="请你作为我的物理课助教，用通俗易懂且间接的语言帮我解释物理概念。"),
    HumanMessage(content="什么是波粒二象性？"),
]

# 获取模型响应
response = model.invoke(messages)
print(response.content)  # 打印AI的回复内容
```

## 2. 提示词模板（Prompt Templates）

提示词模板是 LangChain 的一个强大特性，它允许我们创建可重用的提示词模板，并在运行时填充变量。

### 2.1 创建消息模板

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 方法一：分别创建系统消息和用户消息模板
system_template = "你是一位专业的翻译，能够将{input_language}翻译成{output_language}，并且输出文本会根据用户要求的任何语言风格进行调整。"
system_prompt_template = SystemMessagePromptTemplate.from_template(system_template)

human_template = "文本：{text}\n语言风格：{style}"
human_prompt_template = HumanMessagePromptTemplate.from_template(human_template)

# 方法二：直接使用 ChatPromptTemplate（推荐）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的翻译，能够将{input_language}翻译成{output_language}。"),
    ("human", "文本：{text}\n语言风格：{style}")
])
```

### 2.2 使用模板

模板创建后，我们可以通过提供变量值来生成具体的提示词：

```python
# 使用分别创建的模板
system_message = system_prompt_template.format(
    input_language="中文",
    output_language="英文"
)

human_message = human_prompt_template.format(
    text="今天天气真好",
    style="正式"
)

# 使用 ChatPromptTemplate
messages = prompt.invoke({
    "input_language": "中文",
    "output_language": "英文",
    "text": "今天天气真好",
    "style": "正式"
})

# 发送到模型
response = model.invoke(messages)
```

## 3. Few-Shot 提示词模板

Few-Shot 提示词模板允许我们通过提供示例来指导模型的输出。这在需要特定格式输出或者需要模型学习特定模式时特别有用。

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate

# 创建示例模板
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "格式化以下客户信息：\n姓名 -> {customer_name}\n年龄 -> {customer_age}\n城市 -> {customer_city}"),
    ("ai", "##客户信息\n- 客户姓名：{formatted_name}\n- 客户年龄：{formatted_age}\n- 客户所在地：{formatted_city}")
])

# 示例数据
examples = [
    {
        "customer_name": "张三", 
        "customer_age": "27",
        "customer_city": "长沙",
        "formatted_name": "张三",
        "formatted_age": "27岁",
        "formatted_city": "湖南省长沙市"
    },
    {
        "customer_name": "李四", 
        "customer_age": "42",
        "customer_city": "广州",
        "formatted_name": "李四",
        "formatted_age": "42岁",
        "formatted_city": "广东省广州市"
    }
]

# 创建 Few-Shot 提示词模板
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 创建完整的提示模板
final_prompt = ChatPromptTemplate.from_messages([
    few_shot_prompt,
    ("human", "{input}")
])
```

## 4. 输出解析

LangChain 提供了多种输出解析器，可以将模型的原始输出转换为结构化数据。

### 4.1 列表解析器

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# 创建列表解析器
output_parser = CommaSeparatedListOutputParser()

# 获取格式说明（用于提示模型如何输出）
format_instructions = output_parser.get_format_instructions()

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "{parser_instructions}"),
    ("human", "列出5个{subject}色系的十六进制颜色码。")
])

# 创建输出解析器
output_parser = CommaSeparatedListOutputParser()
parser_instructions = output_parser.get_format_instructions()

# 2. 基本组件调用
# 单独调用提示模板
messages = prompt.invoke({
    "subject": "莫兰迪",
    "parser_instructions": parser_instructions
})
# 调用模型
response = model.invoke(messages)
# 调用解析器
colors = output_parser.invoke(response.content)
print(colors)  # ['#B57EDC', '#B55EDC', '#B53EDC', '#B51EDC', '#B50EDC']
```

### 4.2 JSON 解析器

```python
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# 定义输出结构
class BookInfo(BaseModel):
    book_name: str = Field(description="书籍的名字")
    author_name: str = Field(description="书籍的作者")
    genres: List[str] = Field(description="书籍的体裁")

# 创建 JSON 解析器
parser = PydanticOutputParser(pydantic_object=BookInfo)

# 获取格式说明
format_instructions = parser.get_format_instructions()

# 使用解析器
prompt = ChatPromptTemplate.from_messages([
    ("system", "{format_instructions}"),
    ("human", "请从以下文本中提取书籍信息：{text}")
])

response = model.invoke(prompt.invoke({
    "format_instructions": format_instructions,
    "text": "..."
}))

book_info = parser.invoke(response)
print(book_info.book_name)  # 访问解析后的数据
```

## 5. 链式调用（Chain）与 Runnable 协议

### 5.1 什么是 Runnable 协议

Runnable 协议是 LangChain 最新版本中引入的一个统一接口标准。它让所有主要组件(如模型、提示词模板、输出解析器等)都实现了相同的接口，使它们能够以统一的方式进行调用和组合。

主要特点：
- 所有实现 Runnable 协议的组件都支持 `invoke()`、`batch()`、`stream()` 等方法
- 可以使用管道操作符 `|` 将多个组件串联起来
- 支持并行处理和异步操作
- 提供统一的错误处理机制

### 5.2 基本用法与作用

Runnable 协议的核心作用是提供一个统一的接口，让所有组件都可以用相同的方式被调用和组合。通过 `invoke()` 方法，我们可以：

1. 统一调用方式：无论是提示词模板、模型还是解析器，都使用相同的方法进行调用
2. 简化组合：可以轻松地将多个组件组合成处理流程
3. 保证类型安全：每个组件都明确定义了输入输出类型

让我们通过一个翻译助手的例子来理解：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# 1. 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "{parser_instructions}"),
    ("human", "列出5个{subject}色系的十六进制颜色码。")
])

# 创建模型
model = ChatOpenAI()

# 创建输出解析器
output_parser = CommaSeparatedListOutputParser()
parser_instructions = output_parser.get_format_instructions()

# 2. 基本组件调用
# 单独调用提示模板
messages = prompt.invoke({
    "subject": "莫兰迪",
    "parser_instructions": parser_instructions
})
# 调用模型
response = model.invoke(messages)
# 调用解析器
colors = output_parser.invoke(response.content)
print(colors)  # ['#B57EDC', '#B55EDC', '#B53EDC', '#B51EDC', '#B50EDC']

# 3. 使用管道操作符组合
chain = prompt | model | output_parser

# 4. 使用 RunnablePassthrough 处理数据
# 假设我们想在处理前后添加一些逻辑
def add_prefix(subject: str) -> str:
    return f"{subject}风格"

chain_with_transform = (
    {
        "subject": lambda x: add_prefix(x["subject"]),  # 预处理输入
        "parser_instructions": RunnablePassthrough()  # 直接传递
    }
    | prompt 
    | model 
    | output_parser
)

# 5. 批量处理
inputs = [
    {"subject": "莫兰迪", "parser_instructions": parser_instructions},
    {"subject": "北欧", "parser_instructions": parser_instructions}
]

# 同步批处理
results = chain.batch(inputs)

# 异步批处理
async def batch_process():
    results = await chain.abatch(inputs)
    return results
```

这个例子展示了：
1. 如何单独使用各个 Runnable 组件
2. 如何使用管道操作符组合组件
3. 如何使用 RunnablePassthrough 处理数据流
4. 如何进行批量处理
5. 如何使用异步操作

### 5.3 高级用法

## 6. 最佳实践

1. **使用最新的导入方式**：优先使用 `langchain_core`、`langchain_openai` 等新的包
2. **使用 LCEL**：优先使用 LCEL 语法创建处理流程
3. **类型提示**：利用 Python 的类型提示功能增加代码可读性
4. **错误处理**：添加适当的异常处理
5. **API 密钥管理**：使用环境变量管理敏感信息
6. **模型参数调优**：根据具体用例调整 temperature 等参数
7. **使用 Runnable 协议**：
   - 优先使用 invoke() 而不是 run() 或其他旧方法
   - 合理使用 RunnablePassthrough 处理数据流
   - 需要并行处理时考虑使用 batch() 方法
   - 对于大规模处理考虑使用异步方法

## 7. 注意事项

1. 注意 API 调用成本
2. 保护用户隐私和敏感信息
3. 实现适当的速率限制
4. 定期更新 LangChain 版本
5. 测试不同的模型参数组合

## 8. 结语

本教程展示了 LangChain 的核心功能和实际应用示例。通过这些示例，你应该能够开始使用 LangChain 构建自己的 AI 应用。如果需要更深入的信息，建议查看 LangChain 的官方文档。
