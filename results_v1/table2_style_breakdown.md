| System                      |   S1 Clean Synthetic |   S2 Conversational |   S3 Code Switched |   S4 Roman Urdu |   S5 Hard Distractor |   Parametric Nastaliq |
|:----------------------------|---------------------:|--------------------:|-------------------:|----------------:|---------------------:|----------------------:|
| Direct Gen (No RAG)         |                 52.6 |                29.3 |               22.6 |            25   |                 26.5 |                  48.1 |
| Vanilla Chunk RAG           |                 48.7 |                42.3 |               36   |            33.7 |                 28.9 |                   9.7 |
| DR-RAG (Ahmad et al., 2026) |                 49.1 |                42.4 |               36.3 |            32.3 |                 30   |                   6.5 |
| Zero-Shot ReAct Agent       |                 43.1 |                45.6 |               36   |            31.9 |                 29.7 |                  28.1 |
| Urdu-DynRAG (Ours)          |                 47.1 |                42.8 |               38.4 |            33.2 |                 25   |                  50.2 |