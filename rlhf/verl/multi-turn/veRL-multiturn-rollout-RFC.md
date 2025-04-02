# RFC: 支持多轮对话的 rollout v0.1

## 背景

什么情况下回出现 RL 过程中的多轮对话？

1. 与环境交互，出现 assistant 轮次中包含 tool call 的情况，使得 assistant 轮次和 tool response 轮次交替出现
2. 与用户模拟交互，assistant 轮次后，模拟用户判断是否完成任务，如果未完成，补充相关要求或调整意见，让 assistant 继续生成
3. MARL 情况

v0.1 目标：支持 agentic 与环境通过 tool call 交互情况的多轮负载，假设环境独立于训练任务部属
v0.1 非目标：支持用户模拟交互的多轮用例和 MARL 用例

目前在 multi-turn rollout with sandbox 场景下，会新增 async rollout, tool server , ray agent trainer(一期先不实现)。

## 设计

### Tool

```python
class BaseTool(object):
    """Base class for tools.

    A tool should support the following methods:
    
    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        self.config = config
        self.name = tool_schema.function.name
        self.tool_schema = tool_schema

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema
    
    def create(self) -> None:
        pass
    
    def execute(self, parameters: OpenAIFunctionParsedSchema) -> None:
        pass
    
    def calc_reward(self) -> float:
        return 0.0
    
    def release(self) -> None:
        pass
```

- `get_openai_tool_schema` 方法，以支持 apply_chat_template 时添加 tools 相关描述，为后续 server-based 实现 openai chat completions style 兼容接口做准备
- `create` 方法，在每个 rollout session 开始时，依次调用每个 tool 的 create 函数，没有则 `pass`
- `execute` 方法，在每个 assistant turn / server-based rollout 调用解析得到的 tool function 的参数，并执行
- `calc_reward` 方法，在每个 rollout session / turn 结束前，对与环境交互的 tool 根据 state 计算 reward
- `release` 方法，在咩个 rollout session 结束后，释放单次 rollout session 需要的资源


### AsyncRequest

```python
class AsyncRolloutRequestStateEnum(str, Enum):
    """The enum for async rollout request state."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TOOL_CALLING = "tool_calling"

class AsyncRolloutRequest(BaseModel):
    """The data model for async rollout."""
    request_id: str
    state: AsyncRolloutRequestStateEnum
    prompt: str
    messages: List[Message]
    tools: List[OpenAIFunctionToolSchema]
```

AsyncRolloutRequest 类，用于管理 async rollout 的请求状态、请求内容、请求结果等，并提供状态变更、结果获取等方法。


### AsyncRollout

- 保持 generate sequences 用来保持和前序接口一致
- rl_dataset 添加 id 和 tools 配置支持
- 通过 monkey_patch 为 sgl verl engine 添加 `async_generate` 能力，或使用本就支持 `async_generate` 的 sgl engine 
- 实现基于 tool 和 request 机制的异步并发 request 生成

## 实现

### 分期实现规划

一期没有 agent trainer 时，在 turn 间暂时先支持：

1. assistant turn 声明 tool call/function call
2. 在 async rollout 内 parse 后调用 tool server 提供的 process_action/submit/do_tool_call 语义的 api
3. tool server 内根据提交的 tool type 和 param 组合成相应具体 sandbox/tool （e.g. sandboxfusion、milvus、faiss、swe-gym）等的请求串，并在完成请求后，返回 tool response 结果（可在 tool server 内实现 format 逻辑）
4. 返回后为在之前的 tool call assitant 轮次后拼入 tool response 轮次，并让 assistant 继续生成

- 在 session 开始时，实现 start_session/enter 语义接口，为 swe 等需要为每一条轨迹实现隔离的环境修改的场景提供扩展点
- 在 session 结束时，实现 compute_env_state_rewad 语义接口，在释放资源前计算当前任务完成情况（一期用 hotpotqa/gsm8k 等可先不实现），实现 terminate_session/exit 语义接口，用来释放当前 session 在 enter 时创建的专属资源

二期，ray agent trainer 加入后，可以实现 pick gen batch -> rollout -> calc reward & filter -> train batch -> update actor etc. 在 rayppotrainer 中用处理结束显式分开，变成 gen batch 随着 rollout & filter & reward 跑到某个量，停止切换进后边的 update model ，为完成的 inference 进 partial rollout 逻辑或直接 drop，把 request level 的状态管理提到 async rollout 外边

三期，可以加入 roleplay 显式模拟用户交互 & interact as tool call，并支持处理比较重环境验证和 with feedback 的 RL 场景

### 一期实现方式

1. 抽离单独的 ToolServer 类，提供如下方法：
   1. init(self, tools, tool_call_fns) ：初始化 tools list 为 https://docs.sglang.ai/backend/function_calling.html 格式（和 async rollout 共享），并额外增加每个 tool 添加一个可在初始化时注入的 tool_call_fn dict 用来扩展 create, call, calc_reward, release 4 种语义，tool_call_fns 可以考虑封装为一个 BaseTool class
   2. enter / create_session （这里没想好是不是要作为 context manager），因为是 session level 的，目前更偏向 create_session 这个命名方式，依次调用 tool_call_fns 中每个 tool 的 create 函数，没有则跳过
   3. exit / release_session 依次调用 tool_call_fns 中每个 tool 的 release 函数，没有则跳过
   4. call 根据 engine 返回的 tool_calls 字段（tool name + args）拼成调用串，调用对应工具的 call 函数
   5. calc_reward reward manager 可以访问 tool server 完成 env state 相关的 reward 计算，gsm8k 和 r1-searcher 不需要
2. 重构 AsyncRollout 逻辑，在 generate seq 时，接受当前任务的 tool server 类，将 tools 本身放到轮级调度，等之后上 partial rollout 再考虑 max token 导致 turn 不完整的情况，在 async rollout 过程中用 req_id trace tool server instance
3. loops 和 tasks 功能看起来可以被 tool server class 吸收，可以单独做个 tool_call_fn dir 提供不同任务的 tool 实现和注册机制（注册可以先不搞）
