| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| drafts | 12/12 | 100.0% | 78s | drafted 46 tok/s vs undrafted 37 tok/s |
| streaming | 4/4 | 100.0% | 24s | identical |
| multiturn | 8/8 | 100.0% | 17s | recalled 8/8 facts |
| stop | 3/3 | 100.0% | 2s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 22s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 2s | 2/2 byte-identical |
| max-tokens | 3/3 | 100.0% | 2s | 3/3 exact |
| tools | 55/60 | 91.7% | 69s | "How do you say 'where is the station' in": got [] |
| json | 30/30 | 100.0% | 23s | all valid |
| ifeval | 88/150 | 58.7% | 1633s | loose 88/150, strict 77/150, 150 covered prompts |
| gsm8k | 246/250 | 98.4% | 509s | misses at [89, 93, 119, 157] |
| mgsm | 202/240 | 84.2% | 1840s | de 28/30, es 29/30, fr 26/30, ja 21/30, zh 26/30, ru 27/30, sw 21/30, bn 24/30 |
| mmlu | 280/342 | 81.9% | 1892s | 57 subjects x 6; weakest business_ethics 3/6, global_facts 3/6, high_school_physics 3/6 |
| math500 | 459/500 | 91.8% | 8058s | L1 43/43 L2 88/90 L3 98/105 L4 118/128 L5 112/134, 18 without a boxed answer |
