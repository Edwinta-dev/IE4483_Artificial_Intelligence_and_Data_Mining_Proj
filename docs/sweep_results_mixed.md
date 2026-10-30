# GPU baseline sweep results

| Backbone | Freeze mode | Seed | Best val accuracy | Epochs | Seconds per epoch | Run |
|---|---|---:|---:|---:|---:|---|
| resnet50 | last_block | 42 | 0.996199 | 5 | 301.19 | `runs\sweeps\gpu_mixed\resnet50__last_block__c6ce901214\seed_42\20261027-140312-423506-resnet50` |
| resnet50 | frozen | 42 | 0.995599 | 10 | 3.76 | `runs\sweeps\gpu_mixed\resnet50__frozen__9e45293166\seed_42\20261027-135501-437535-resnet50` |
| resnet50 | full | 42 | 0.993999 | 7 | 772.30 | `runs\sweeps\gpu_mixed\resnet50__full__f987faee5e\seed_42\20261027-142932-710606-resnet50` |
| resnet18 | last_block | 42 | 0.989198 | 5 | 181.78 | `runs\sweeps\gpu_mixed\resnet18__last_block__b85a42fdf3\seed_42\20261027-131302-250437-resnet18` |
| resnet18 | full | 42 | 0.988798 | 5 | 295.08 | `runs\sweeps\gpu_mixed\resnet18__full__a492cb266c\seed_42\20261027-132924-724331-resnet18` |
| resnet18 | frozen | 42 | 0.987998 | 10 | 3.86 | `runs\sweeps\gpu_mixed\resnet18__frozen__1eda6c6e71\seed_42\20261027-130810-208063-resnet18` |

The **resnet50 / last_block** configuration had the top validation accuracy (0.996199). It is selected as the baseline because it achieved the best validation result and had a lower compute cost than the fine-tuned configurations; this keeps later comparisons efficient without giving up measured validation accuracy.
