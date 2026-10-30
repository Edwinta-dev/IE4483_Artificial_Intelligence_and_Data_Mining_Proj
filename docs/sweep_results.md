# GPU baseline sweep results

| Backbone | Freeze mode | Seed | Best val accuracy | Epochs | Seconds per epoch | Run |
|---|---|---:|---:|---:|---:|---|
| resnet18 | frozen | 42 | 0.988198 | 10 | 2.41 | `runs\sweeps\gpu_baseline\resnet18__frozen__1eda6c6e71\seed_42\20261026-205218-283402-resnet18` |
| resnet18 | full | 42 | 0.991398 | 10 | 313.87 | `runs\sweeps\gpu_baseline\resnet18__full__a492cb266c\seed_42\20261026-210838-651976-resnet18` |
| resnet18 | last_block | 42 | 0.989198 | 5 | 141.13 | `runs\sweeps\gpu_baseline\resnet18__last_block__b85a42fdf3\seed_42\20261026-205608-044502-resnet18` |
| resnet50 | frozen | 42 | 0.995399 | 10 | 2.30 | `runs\sweeps\gpu_baseline\resnet50__frozen__9e45293166\seed_42\20261026-220149-517630-resnet50` |
| resnet50 | full | 42 | 0.992398 | 5 | 636.07 | `runs\sweeps\gpu_baseline\resnet50__full__f987faee5e\seed_42\20261026-222221-868720-resnet50` |
| resnet50 | last_block | 42 | 0.995199 | 5 | 171.45 | `runs\sweeps\gpu_baseline\resnet50__last_block__c6ce901214\seed_42\20261026-220701-417637-resnet50` |

The **resnet50 / frozen** configuration had the top validation accuracy (0.995399). It is selected as the baseline because it achieved the best validation result and had a lower compute cost than the fine-tuned configurations; this keeps later comparisons efficient without giving up measured validation accuracy.
