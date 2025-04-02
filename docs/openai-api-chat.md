# API of Chat Completion (OpenAI)

Reference: https://platform.openai.com/docs/api-reference/chat

> [!NOTE]
>
> All requests support bearer token authentication

## Full list of APIs

| API                    | API ENDPOINT                       | REQUEST       | RESPONSE      | PRIORITY |
| ---------------------- | ---------------------------------- | ------------- | ------------- | -------- |
| Create chat completion | `POST /openai/v1/chat/completions` | `ChatCmplReq` | `ChatCmplObj` | High     |
|                        |                                    |               |               |          |
|                        |                                    |               |               |          |



- `ChatCmplReq`
  - `ChatCmplMesg`
- `ChatCmplObj`
  - `ChatCmplChoice`
    - `ChatCmplMesgAssistant`
  - `ChatCmplUsage`



## Message formats

### `ChatCmplReq` ([link](https://platform.openai.com/docs/api-reference/chat/create))

| Field      | Type           | Required | Description                                                  |
| ---------- | -------------- | -------- | ------------------------------------------------------------ |
| `model`    | `str`          | True     | Model ID                                                     |
| `messages` | `list[ChatCmplMesg]` | True     | A list of messages comprising the conversation so far. `ChatCmplMesg` is defined in the next table. |
| `n`        | `int = 1`      | False    | How many chat completion choices to generate.                |
| ...        |                |          |                                                              |



### `ChatCmplMesg` ([link](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages))

[OpenAI Model Spec / The chain of command](https://model-spec.openai.com/2025-02-12.html#chain_of_command)

| Field     | Type  | Required | Description                                                  |
| --------- | ----- | -------- | ------------------------------------------------------------ |
| `role`    | `str` | True     | Corresponding to 4 levels of authorities. Can be `platform` (system message by model vendor), `developer` (instructions by application developer who uses the API), `user` (messages from end users), `guideline` (hints that can be implicitly overridden) and `assistant ` (response from the AI model) |
| `name`    | `str` | False    | To distinguish among different participants with the same role. |
| `content` | `str` | True     | The main content of the message                              |
| `refusal` | `str` | False    | Only when `role == assistant`, and the AI model refuse to answer |
| ...       |       |          |                                                              |



### `ChatCmplObj` ([link](https://platform.openai.com/docs/api-reference/chat/object))

| Field     | Type           | Required | Description                                                  |
| --------- | -------------- | -------- | ------------------------------------------------------------ |
| `id`      | `str`          | True     | A unique ID for this chat completion                         |
| `object`  | `str`          | True     | Always `chat.completion`                                     |
| `created` | `int`          | True     | Timestamp                                                    |
| `model`   | `str`          | True     | Model ID that may not match the request `mode` field, when the model ID in request is an alias |
| `choices` | `list[ChatComplChoice]` | True     | A list of completion choices, whose size matches the `n` parameter of the request. |
| `usage`   | `ChatCmplUsage`       | True     | Usage statistics                                             |



### `ChatCmplChoice` 

| Field           | Type     | Required | Description                                                  |
| --------------- | -------- | -------- | ------------------------------------------------------------ |
| `index`         | `int`    | True     | Index in the list                                            |
| `message`       | `ChatCmplMesgAssistant` | True     | Role is always `assistant`                                   |
| `finish_reason` | `str`    | True     | `stop`: hit natural stop<br />`length`: hit max sequence length<br />`content_filter`: censorship |
| ...             |          |          |                                                              |



### `ChatCmplUsage`  ([link](https://platform.openai.com/docs/api-reference/chat/object#chat/object-usage))

| Field               | Type  | Required | Description                             |
| ------------------- | ----- | -------- | --------------------------------------- |
| `prompt_tokens`     | `int` | True     | # of tokens in the prompt               |
| `completion_tokens` | `int` | True     | # of tokens in the generated completion |
| `total_tokens`      | `int` | True     | `prompt_tokens` + `completion_tokens`   |
| ...                 |       |          |                                         |