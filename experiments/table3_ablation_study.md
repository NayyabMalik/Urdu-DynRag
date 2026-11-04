| Configuration                          | Token F1 (%)   |   Hit@1 (%) |   Retrieval Rate (%) |   Reformulation Rate (%) |   Net Reward |
|:---------------------------------------|:---------------|------------:|---------------------:|-------------------------:|-------------:|
| Full Urdu-DynRAG                       | 37.0 ± 22.6    |        79.2 |                 93.1 |                        0 |        0.227 |
| w/o Action 3 (No Reformulation)        | 37.9 ± 22.9    |        79.2 |                 93.1 |                        0 |        0.237 |
| w/o D_qa (No Synthetic QA)             | 37.9 ± 21.1    |        79.2 |                 93.1 |                        0 |        0.236 |
| w/o Generator Entropy (f10-f12 zeroed) | 37.9 ± 22.8    |        79.2 |                 94.1 |                        2 |        0.234 |