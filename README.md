# physics-graph-agent

A multi-agent assistant that answers physics questions from a knowledge graph and cites its sources. It runs on a local model, and it refuses when it doesn't have the facts.

Built with LangGraph, LangChain, Neo4j, Elasticsearch, Redis, Pandas, PyTorch, FastAPI, and Node.js.

## How it works

```
 client ──> Node.js gateway ──> Python agent API ──> LangGraph workflow
              validation, rate           Redis cache
              limits, timeouts
                  │
                  └──> wake-word API (small PyTorch model)
```

The workflow:

```
 planner ──> search agent ──> graph agent ──> writer ──> checker ──> answer
                 │                              ▲           │
                 │ nothing relevant             └── retry ──┘
                 ▼                                          │ 3 failures
              refuse <──────────────────────────────────────┘
```

| Step | What it does |
|---|---|
| Planner | The model decides if a question needs a lookup or a connection between two concepts |
| Search agent | Finds concepts in Elasticsearch, and stops off-topic questions before the model is used |
| Graph agent | Gets a concept's neighbors in Neo4j, or the shortest path between two concepts |
| Writer | Answers using only the numbered facts, and cites them |
| Checker | Verifies citations in code, and sends failures back to the writer with feedback |

The conversation is stored per session with a LangGraph checkpointer, so follow-up questions keep their context.

The data is 40 physics concepts and about 40 relationships that I wrote by hand, including some deliberately messy rows that the Pandas pipeline has to clean: duplicate spellings, missing fields, dangling links, and self-loops.

## Results

**Agent**, on 15 labeled questions with `llama3` running locally through Ollama:

| Measure | Result |
|---|---|
| Right concept retrieved | 12/12 |
| Right route chosen | 12/12 |
| Answers with valid citations | 12/12 |
| Off-topic questions refused | 3/3, all before reaching the writer |

A repeated question returned from the Redis cache in 0.01 s, against 3.5 s for the first time.

**Wake word** ("hey nova"), on two voices the model never heard in training:

| Measure | Result |
|---|---|
| Accuracy | 89.6% |
| Missed the wake phrase | 10.0% |
| Woke up wrongly | 10.8%, which just misses my 10% goal |
| Model size | 18,817 parameters, 151 KB |

## Problems found while testing

**Off-topic questions passed search.** "What is the best pizza topping?" returned four search hits through fuzzy matches on common words, and was refused only because the model happened not to cite anything. I switched to an English analyzer and raised the score threshold, based on the measured gap: off-topic questions peak near 3.3 and real concepts start near 12. They now stop before the model is called.

**The first wake-word model was useless.** It missed the wake phrase 90% of the time. Training and test scores were about equal, so it was underfitting rather than overfitting, and its scores swung widely between epochs. A learning-rate schedule, a slightly larger network, keeping the best epoch on validation voices, and twice as much data brought it to the numbers above.

## Running it

Requires Docker, Python 3.12, Node 20, [Ollama](https://ollama.com), and `espeak-ng` for wake-word training.

```sh
docker compose up -d --wait
ollama pull llama3
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt torch
export NEO4J_PASSWORD=localdevpassword OLLAMA_MODEL=llama3

python -m pytest tests                 # Tests that need databases are skipped if they aren't running
python -m agent.evaluate               # The evaluation above
python -m wakeword.train               # Trains the wake-word model

uvicorn agent.service:app --port 8000
uvicorn wakeword.service:app --port 8001
cd gateway && npm install && npm test && npm start
```

Then ask a question:

```sh
curl localhost:3000/ask -H 'content-type: application/json' \
  -d '{"question": "How does nuclear fusion connect to quantum tunneling?"}'
```

The password above is only a local default for the throwaway Neo4j container.

## Known limits

- The checker confirms that citations exist. It doesn't confirm that the cited fact supports the claim, so a model could still cite a real fact for a wrong statement.
- The 12 answerable test questions are few, and I wrote them with knowledge of the data, so the scores show it works, not how it would do on arbitrary questions.
- The search threshold is calibrated to this small dataset, and search scores change as data grows.
- The wake-word model trained and tested only on synthetic `espeak-ng` voices with added noise, never on real recorded speech, so real microphones would likely do worse. It classifies clips and doesn't yet listen to a live stream.
- This runs on one machine for local use. It isn't deployed, and the gateway's rate limit is in memory.
