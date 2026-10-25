| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| drafts | 12/12 | 100.0% | 86s | drafted 43 tok/s vs undrafted 36 tok/s |
| streaming | 4/4 | 100.0% | 25s | identical |
| multiturn | 8/8 | 100.0% | 15s | recalled 8/8 facts |
| stop | 3/3 | 100.0% | 2s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 22s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 3s | 2/2 byte-identical |
| max-tokens | 3/3 | 100.0% | 2s | 3/3 exact |
| tools | 57/60 | 95.0% | 70s | "How do you say 'where is the station' in": got [] |
| json | 30/30 | 100.0% | 23s | all valid |
| ifeval | 87/150 | 58.0% | 1740s | loose 87/150, strict 75/150, 150 covered prompts |
| gsm8k | 242/250 | 96.8% | 443s | misses at [12, 37, 87, 89, 93, 119, 122, 187] |
| mgsm | 203/240 | 84.6% | 789s | de 28/30, es 29/30, fr 26/30, ja 25/30, zh 25/30, ru 27/30, sw 17/30, bn 26/30 |
| mmlu | 283/342 | 82.7% | 773s | 57 subjects x 6; weakest business_ethics 3/6, high_school_physics 3/6, high_school_us_history 3/6 |
| math500 | 445/500 | 89.0% | 6192s | L1 41/43 L2 85/90 L3 98/105 L4 112/128 L5 109/134, 27 without a boxed answer |
