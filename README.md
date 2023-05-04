# langchain-demo

## configuration

```shell
vi .bashrc
# or
vi .bash_profile
```

```shell
export OPENAI_API_KEY=sk-your openai api key
```

```shell
source .bashrc
```

## functions

|  location   | desc  |
|  ----  | ----  |
| qa_once  | question and answer once |
| qa_robots  | load all data and learning for that, question and answer like a robots |
| qa_youtube  | load the youtube video and learning, question and answer like a robots |
| google_search  | google search question, and feeding the result to openai |
| summarization  | summarize the content |
| image_analysis  | load the image, and analysis the image content |
| chroma.chrome_persist | load all data and persistence in vector db chroma |
| chroma.chrome_load | load the data from chroma, and using it |
