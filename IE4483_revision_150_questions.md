# IE4483 / EE6483 – 150 Revision Questions (with answers)

Topics: Bayesian Inference (Q1–25) · Neural Networks & Backprop (Q26–62) · Regularization & Optimization (Q63–87) · CNNs (Q88–122) · Clustering (Q123–150)

Tags: **[Recall]** = definition/explain · **[Calc]** = work it out by hand · **[Think]** = reasoning/compare. Cover the **A:** lines and try each question first. Calculation answers were checked numerically.

> Slide errata spotted: (i) Naïve Bayes tennis slide shows P(yes, obs) = 0.0051, but 9/14 × 0.00823 = **0.0053** (decision unchanged). (ii) Average-linkage slide gives 1.02 for D(1+2, 3+4); exact value is 1.0125 ≈ 1.01.

---

## Part 1 – Bayesian Inference (Q1–25)

**Q1 [Recall]** In P(h|D) = P(D|h)P(h)/P(D), name each term (prior, likelihood, posterior, evidence).
**A:** P(h) = prior (belief in h before seeing D); P(D|h) = likelihood (how probable the data is if h holds); P(h|D) = posterior (belief in h after seeing D); P(D) = evidence/normaliser (same for every h).

**Q2 [Recall]** Derive Bayes' theorem from the joint probability.
**A:** P(A,B) = P(A|B)P(B) = P(B|A)P(A). Divide by P(B): P(A|B) = P(B|A)P(A)/P(B).

**Q3 [Recall]** State the law of total probability and use it to expand P(B) for two hypotheses A₁, A₂.
**A:** P(B) = Σᵢ P(B|Aᵢ)P(Aᵢ) over a partition {Aᵢ}. For two: P(B) = P(B|A₁)P(A₁) + P(B|A₂)P(A₂).

**Q4 [Think]** Why do we want P(h|D) rather than P(D|h)?
**A:** We have the data D (training set) and want the most probable hypothesis/label given it. P(D|h) is easy to model/count; Bayes converts it into the quantity we actually need.

**Q5 [Recall]** Define the MAP hypothesis. Why can P(D) be dropped?
**A:** h_MAP = argmax_h P(h|D) = argmax_h P(D|h)P(h). P(D) does not depend on h, so it doesn't change which h is largest.

**Q6 [Think]** When are MAP and Maximum Likelihood (ML) the same?
**A:** When the prior P(h) is constant (uniform) over all hypotheses; then h_MAP = argmax P(D|h) = h_ML.

**Q7 [Recall]** What is P(A,B) in terms of conditional probability? What does it simplify to if A and B are independent?
**A:** P(A,B) = P(A|B)P(B). If independent, P(A|B) = P(A) so P(A,B) = P(A)P(B).

**Q8 [Calc]** A fair coin is flipped twice. (a) P(HH)? (b) P(at least one head)?
**A:** (a) 1/4 = 0.25. (b) 1 − P(TT) = 3/4.

**Q9 [Calc]** Fair coin, two flips. (a) Given the first flip is H, P(HH)? (b) Given at least one H, P(HH)?
**A:** (a) P(HH)/P(H₁) = 0.25/0.5 = **0.5**. (b) 0.25/0.75 = **1/3** (not 1/2 – the conditioning events differ).

**Q10 [Calc]** Fair coin, three flips. Given at least one head, P(HHH)?
**A:** (1/8)/(7/8) = **1/7**.

**Q11 [Calc]** (Slide) 40% of the class are IEM, 60% EEE. 50% of IEM and 30% of EEE pass. What fraction of passers are IEM?
**A:** P(S) = 0.5·0.4 + 0.3·0.6 = 0.38. P(M|S) = 0.5·0.4/0.38 = **0.526**.

**Q12 [Calc]** Same class but 30% IEM / 70% EEE (pass rates unchanged). P(IEM | pass)? P(EEE | pass)?
**A:** P(S) = 0.5·0.3 + 0.3·0.7 = 0.36. P(M|S) = 0.15/0.36 = **0.417**; P(E|S) = 0.21/0.36 = **0.583**.

**Q13 [Calc]** A disease has prevalence 1%. A test is positive for 99% of sick people and gives a false positive for 5% of healthy people. Given a positive test, P(disease)? Why is it so low?
**A:** P(+) = 0.99·0.01 + 0.05·0.99 = 0.0594. P(D|+) = 0.0099/0.0594 = **0.167**. The prior is tiny, so false positives from the large healthy group outnumber true positives.

**Q14 [Calc]** 30% of emails are spam. The word "free" appears in 60% of spam and 10% of non-spam. An email contains "free". P(spam)?
**A:** 0.6·0.3 / (0.6·0.3 + 0.1·0.7) = 0.18/0.25 = **0.72**.

**Q15 [Calc]** Machine A makes 60% of items with 2% defective; machine B makes 40% with 5% defective. A defective item is found. P(from B)?
**A:** P(def) = 0.012 + 0.020 = 0.032. P(B|def) = 0.020/0.032 = **0.625**.

**Q16 [Think]** Monty Hall: why should you switch? Give the intuitive argument.
**A:** Your first pick is right with prob 1/3, wrong with 2/3. The host (who knows) always opens a losing door, so if you were wrong (2/3), switching wins; if right (1/3), switching loses. Switch wins 2/3; stay wins 1/3.

**Q17 [Calc]** Monty Hall via Bayes: you pick Door 1, host opens Door 2 (empty). Find P(c₁|D₂) and P(c₃|D₂).
**A:** P(c)=1/3 each. P(D₂|c₁)=1/2, P(D₂|c₂)=0, P(D₂|c₃)=1. P(D₂) = 1/6 + 0 + 1/3 = 1/2. P(c₁|D₂) = (1/2·1/3)/(1/2) = **1/3**; P(c₃|D₂) = (1·1/3)/(1/2) = **2/3**.

**Q18 [Think]** Suppose the host doesn't know where the prize is, opens Door 2 or 3 at random, and happens to reveal an empty door. Should you switch?
**A:** Now P(D₂ empty|c₁)=1/2, |c₃)=1/2, |c₂)=0 (excluded). Posterior: c₁ = c₃ = 1/2. Switching gives no advantage. Lesson: the likelihood (host's behaviour) matters.

**Q19 [Calc]** 4-door Monty Hall: you pick one door; host opens one of the other three that is empty; you may switch to one of the two remaining closed doors (chosen at random). P(win) for stay vs switch?
**A:** Stay: 1/4. Switch: 3/4 chance first pick was wrong; prize is then equally likely behind either remaining door → 3/4 · 1/2 = **3/8**.

**Q20 [Recall]** State the Naïve Bayes assumption (words and formula).
**A:** Features are conditionally independent given the class: P(a₁,…,a_d | v_j) = ∏ᵢ P(aᵢ | v_j).

**Q21 [Think]** Why is it called "naïve"? Give a case where it's badly violated.
**A:** Real features are usually correlated, yet we assume independence given the class. E.g. duplicated/highly correlated features (humidity & rain) get counted twice, over-weighting that evidence. Classification can still work well in practice.

**Q22 [Think]** Why does the assumption make learning practical?
**A:** Without it, P(a₁…a_d|v) needs a joint table (e.g. 2^d − 1 numbers per class for binary features – exponential). With it we only estimate d one-dimensional conditionals per class (linear in d), each from simple counts.

**Q23 [Recall]** Give the Naïve Bayes decision rule. Why is P(observation) not needed?
**A:** v_NB = argmax_v P(v) ∏ᵢ P(aᵢ|v). P(obs) is identical for all classes, so it cancels when comparing.

**Q24 [Calc]** (Play Tennis) Observation ⟨Sunny, Cool, High, Strong⟩. Given P(yes)=9/14, P(S|yes)=2/9, P(C|yes)=3/9, P(H|yes)=3/9, P(Strong|yes)=3/9, and P(no)=5/14, P(S|no)=0.6, P(C|no)=0.2, P(H|no)=0.8, P(Strong|no)=0.6, decide play / not play.
**A:** yes: 9/14·(2·3·3·3)/9⁴ ≈ **0.0053** (slide: 0.0051). no: 5/14·0.6·0.2·0.8·0.6 ≈ **0.0206**. no > yes → **do NOT play**. Normalised: P(no|obs) ≈ 0.795.

**Q25 [Calc]** Prior P(spam)=0.4. P(A|spam)=0.8, P(B|spam)=0.5; P(A|ham)=0.2, P(B|ham)=0.4. Email has both A and B. Classify, and give P(spam|A,B) under Naïve Bayes.
**A:** spam: 0.4·0.8·0.5 = 0.16; ham: 0.6·0.2·0.4 = 0.048. P(spam|A,B) = 0.16/0.208 = **0.769** → spam. (Beyond slides: with many features multiply → underflow, so use sums of logs; a zero count wipes out the product, fixed by smoothing.)

---

## Part 2 – Neural Networks & Backpropagation (Q26–62)

**Q26 [Recall]** Write the perceptron model. What does the bias do?
**A:** z = Σwᵢxᵢ + b, y = step(z) (1 if z>0 else 0). The bias shifts the decision threshold: the threshold on Σwᵢxᵢ is −b.

**Q27 [Recall]** What is a perceptron's decision boundary? How can the bias be absorbed into the weights?
**A:** wᵀx + b = 0: a line in 2D, plane in 3D, hyperplane in higher dimensions – so it's a linear classifier. Add x₀ = 1 with w₀ = b.

**Q28 [Calc]** Perceptron with w = (2, −1), b = −0.5. Output for (1,1), (0,1), (1,0), (0,0)?
**A:** z = 0.5, −1.5, 1.5, −0.5 → y = **1, 0, 1, 0**.

**Q29 [Calc]** Design a perceptron for AND and show its truth table.
**A:** y = step(x₁ + x₂ − 1.5): z = −1.5, −0.5, −0.5, 0.5 → y = 0,0,0,1.

**Q30 [Calc]** Which single change turns that AND into OR? Show the table.
**A:** Change only the bias to −0.5: z = −0.5, 0.5, 0.5, 1.5 → y = 0,1,1,1.

**Q31 [Calc]** Design a perceptron for NAND.
**A:** y = step(−x₁ − x₂ + 1.5): z = 1.5, 0.5, 0.5, −0.5 → y = 1,1,1,0.

**Q32 [Think]** Prove a single perceptron cannot compute XOR.
**A:** Need w₂+b>0, w₁+b>0 for (0,1),(1,0); b≤0 for (0,0). Add the first two: w₁+w₂+2b>0 ⇒ w₁+w₂+b > −b ≥ 0, so (1,1) would be class 1, but XOR requires 0. Equivalently, XOR is not linearly separable.

**Q33 [Calc]** Verify the slide's XOR MLP: h₁=step(x₁+x₂−0.5), h₂=step(x₁+x₂−1.5), y=step(h₁−2h₂−0.5).
**A:** (0,0): h=(0,0), y=step(−0.5)=0. (0,1),(1,0): h=(1,0), y=step(0.5)=1. (1,1): h=(1,1), y=step(1−2−0.5)=0. h₁=OR, h₂=AND, y = OR AND NOT(AND) = XOR. Hidden layer makes the problem linearly separable.

**Q34 [Think]** Why must hidden units have nonlinear activations?
**A:** Composing linear/affine layers gives another affine function, so depth adds nothing. Nonlinearity lets the network form non-linear decision regions.

**Q35 [Recall]** What does the universal approximation result say, and what is its caveat?
**A:** With nonlinear activations and enough hidden units, an MLP can approximate any continuous function on a compact domain (Hornik et al.). Caveat: capacity ≠ trainability – it doesn't guarantee gradient descent finds those weights or that it generalises.

**Q36 [Recall]** Give formula, range, and typical use: linear, sign, sigmoid, tanh, ReLU, Leaky ReLU.
**A:** Linear f=z (real-valued/regression output). Sign: ±1, discrete, zero gradient except at jump. Sigmoid 1/(1+e⁻ᶻ), (0,1), probabilities/binary output. tanh, (−1,1), zero-centred. ReLU max(0,z), [0,∞), default hidden activation, slope 0/1. Leaky ReLU max(az,z), small a>0 keeps a gradient for z<0.

**Q37 [Calc]** Derive σ′(z). What is its maximum? What happens when it saturates?
**A:** σ′ = e⁻ᶻ/(1+e⁻ᶻ)² = σ(1−σ). Max 0.25 at z=0 (σ=0.5). Near σ≈0 or 1 the derivative ≈0 ⇒ tiny gradients (slow learning/vanishing gradients).

**Q38 [Calc]** tanh′(z)? Evaluate tanh′(0) and tanh′(2).
**A:** 1 − tanh²(z). tanh′(0) = 1; tanh′(2) = 1 − 0.9640² ≈ 0.0707.

**Q39 [Think]** What is the "dying ReLU" problem and how does Leaky ReLU help?
**A:** If z<0 for all inputs, ReLU outputs 0 with derivative 0, so its weights stop updating (see Q60). Leaky ReLU has slope a>0 for z<0, so gradient still flows.

**Q40 [Think]** Why can't step/sign be used with gradient-based training?
**A:** Derivative is 0 everywhere except at the jump (undefined), so ∂E/∂w gives no useful signal to update weights.

**Q41 [Calc]** Sigmoid MLP with input x=(2,1): h₁=σ(x₁+x₂−1), h₂=σ(−x₁+x₂), o=σ(2h₁−h₂−0.5). Compute o.
**A:** h₁=σ(2)=0.8808, h₂=σ(−1)=0.2689. net_o = 1.7616 − 0.2689 − 0.5 = 0.9927. o = σ(0.9927) ≈ **0.7296**.

**Q42 [Calc]** (Slide) x=(1,2): h₁=σ(x₁−0.5x₂), h₂=σ(0.5x₁+0.5x₂−0.5), o=σ(h₁−h₂). Compute o and the class at threshold 0.5.
**A:** net₁=0 → h₁=0.5; net₂=1 → h₂=0.7311; net_o = −0.2311 → o = **0.4425** → class 0.

**Q43 [Calc]** ReLU net: h₁=ReLU(x₁+x₂−0.5), h₂=ReLU(−x₁+2x₂+1), o=2h₁−h₂+0.1. Compute o for x=(2,−1) and x=(−2,1).
**A:** (2,−1): h₁=0.5, h₂=0 → o=**1.1**. (−2,1): h₁=ReLU(−1.5)=0, h₂=ReLU(5)=5 → o=**−4.9**. Linear output isn't limited to [0,1].

**Q44 [Think]** Distinguish the activation function from the class threshold (e.g. σ output 0.4425 → class 0).
**A:** The activation is part of the network (shapes unit outputs). The threshold is a decision rule applied to the final output to turn it into a label; it can be changed without changing the network.

**Q45 [Think]** Does forward inference use targets or change parameters?
**A:** No. Weights/biases are fixed; we only compute net inputs and activations layer by layer. Targets and updates appear only in training.

**Q46 [Recall]** Write the squared-error loss. Why is a ½ used?
**A:** E = ½ Σₖ(tₖ − oₖ)². The ½ cancels the 2 from differentiating, giving ∂E/∂oₖ = −(tₖ−oₖ).

**Q47 [Recall]** State the gradient-descent update. What does the sign of the gradient tell you?
**A:** w ← w − η ∂E/∂w. Positive gradient ⇒ w decreases; negative gradient ⇒ w increases.

**Q48 [Think]** Effect of the learning rate η?
**A:** Too large: overshoot/oscillation/divergence (loss can rise). Too small: very slow progress. No guarantee of a global minimum either way.

**Q49 [Recall]** Outline one backpropagation training step.
**A:** (1) Forward pass: netⱼ=Σwⱼᵢoᵢ+bⱼ, oⱼ=σ(netⱼ), store values. (2) Backward: δₖ=(tₖ−oₖ)σ′(netₖ); δⱼ=σ′(netⱼ)Σₖwₖⱼδₖ. (3) Update (after all δ computed): wⱼᵢ←wⱼᵢ+ηδⱼoᵢ, bⱼ←bⱼ+ηδⱼ.

**Q50 [Recall]** Give δ for (a) a sigmoid output unit, (b) a linear output unit, (c) a sigmoid hidden unit.
**A:** (a) δ=(t−o)o(1−o). (b) δ=(t−o)·1. (c) δⱼ=oⱼ(1−oⱼ)Σₖwₖⱼδₖ.

**Q51 [Think]** Why compute all δ before updating weights, and use the pre-update weights in δⱼ?
**A:** δⱼ is the gradient w.r.t. the original network. If wₖⱼ were already updated, the error would be propagated through weights that were not used in the forward pass, giving an inconsistent gradient.

**Q52 [Calc]** Derive Δwₖⱼ for an output unit using the chain rule and explain the sign convention δ = −∂E/∂net.
**A:** ∂E/∂wₖⱼ = (∂E/∂oₖ)(∂oₖ/∂netₖ)(∂netₖ/∂wₖⱼ) = (oₖ−tₖ)σ′(netₖ)oⱼ = −δₖoⱼ. Δw = −η∂E/∂w = +ηδₖoⱼ. δ is defined as −∂E/∂net so updates are "+ηδ".

**Q53 [Think]** Why does the bias update Δb=ηδ have no "oᵢ" factor?
**A:** The bias is a weight on a constant input of 1, so ∂net/∂b = 1.

**Q54 [Calc]** (Slide Ex 1) x=(1,0,1), t=1, η=0.9. Unit 4 weights (0.2,0.4,−0.5), b=−0.4; unit 5 weights (−0.3,0.1,0.2), b=0.2; unit 6 weights (−0.3,−0.2), b=0.1; all sigmoid. Forward pass and loss?
**A:** net₄=−0.7→o₄=0.3318; net₅=0.1→o₅=0.5250; net₆=−0.1045→o₆=**0.4739**; E=½(1−0.4739)²=**0.1384**.

**Q55 [Calc]** Continue Q54: find δ₆, δ₄, δ₅ and new w₆₄, w₆₅, b₆.
**A:** δ₆=(0.5261)(0.4739)(0.5261)=**0.13117**. δ₄=0.3318·0.6682·(−0.3)·0.13117=**−0.008725**; δ₅=0.5250·0.4750·(−0.2)·0.13117=**−0.006542**. w₆₄=−0.3+0.9·0.13117·0.3318=**−0.2608**; w₆₅=−0.2+0.9·0.13117·0.5250=**−0.1380**; b₆=0.1+0.9·0.13117=**0.2181**. (After full update, o₆→0.5152, E→0.1175.)

**Q56 [Calc]** Single sigmoid neuron, x=(1,1), w=(0.5,−0.5), b=0.1, t=1, η=1. One backprop step; new output and loss?
**A:** net=0.1, o=0.5250, E=0.1128. δ=(1−0.525)(0.525)(0.475)=0.11846. w→(0.6185, −0.3815), b→0.2185. New net=0.4370+... = 0.6185−0.3815+0.2185=0.4555→ o=0.6119, E=**0.0753** (decreased).

**Q57 [Calc]** Chain x→h→o, both sigmoid: h=σ(wx+b_h), o=σ(vh+b_o). x=1, t=1, w=1, b_h=0, v=1, b_o=0, η=0.5. Do one update.
**A:** h=σ(1)=0.7311; o=σ(0.7311)=0.6750; E=0.0528. δ_o=(1−0.675)(0.675)(0.325)=0.07128; δ_h=h(1−h)·v·δ_o=0.1966·1·0.07128=0.01402. v=1.02606, b_o=0.03564, w=1.00701, b_h=0.00701. New h=0.7338, o=0.6875, E=**0.0488**.

**Q58 [Calc]** Same shape but linear output: h=σ(wx+b_h), o=vh+b_o. x=1, t=0, w=1, b_h=0, v=2, b_o=0, η=0.1. One update.
**A:** h=0.7311, o=1.4621, E=1.0689. δ_o=t−o=−1.4621; δ_h=h(1−h)vδ_o=0.1966·2·(−1.4621)=−0.5749. v=2+0.1(−1.4621)(0.7311)=1.8931; b_o=−0.1462; w=1−0.0575=0.9425; b_h=−0.0575. New h=0.7079, o=1.1938, E=**0.7126**.

**Q59 [Calc]** ReLU hidden layer: x=(1,−2); h₁=ReLU(2x₁+x₂+1), h₂=ReLU(x₁+x₂−0.5), o=3h₁−h₂+0.5 (linear), t=2, η=0.1. Find δs and the weight updates for unit 1 and unit 2.
**A:** h₁=1, h₂=0 (net₂=−1.5), o=3.5. δ_o=−1.5, δ₁=1·3·(−1.5)=**−4.5**, δ₂=0·(−1)·(−1.5)=**0**. Unit 1: w=(2,1)→(1.55, 1.9); b₁=1→0.55. Unit 2 unchanged (inactive). Output: v₁=3→2.85, v₂ stays −1, b_o→0.35.

**Q60 [Think]** Continue Q59: recompute the output and loss. What went wrong?
**A:** New net₁=1.55−3.8+0.55=−1.7 → h₁=0, so o=0.35 and E=½(1.65)²=1.361 vs before 1.125 – loss **rose**. The step was too large: it pushed unit 1 into the dead (z<0) region of ReLU. (Slide Ex 3 shows the same effect: loss 0.0050→0.01154 with a too-large η.)

**Q61 [Recall]** Compare case (SGD), mini-batch and epoch (full-batch) updating. What is an epoch?
**A:** SGD: update after each example (noisy, frequent). Mini-batch: average gradient over a small batch (compromise). Full-batch: one update per pass over the data (smooth, slow per update). An epoch is one pass through the whole training set.

**Q62 [Think]** List stopping criteria for training and explain why low training error alone is not enough.
**A:** Loss below a threshold; tiny parameter updates; max epochs; validation criterion (early stopping). A network can memorise the training set (overfit) while performing poorly on unseen data, so validation performance must be monitored.

---

## Part 3 – Regularization & Optimization (Q63–87)

**Q63 [Recall]** Describe the supervised learning framework (training and inference).
**A:** Model y=f_θ(x). Training: given labelled {(xᵢ,yᵢ)}, estimate θ. Inference: apply learned f_θ to unseen x to predict y.

**Q64 [Recall]** Define underfitting and overfitting and their typical causes.
**A:** Underfitting: model too simple, misses relevant structure (high bias). Overfitting: model too complex/too little data, fits noise in the training set (high variance).

**Q65 [Think]** Where does learning error come from? (Two things we don't know.)
**A:** We don't know the true input-output mapping → we assume a model family (assumption error = bias). We don't know the data distribution → we minimise average error on a finite sample (empirical risk), which fluctuates with the sample (variance).

**Q66 [Recall]** Define bias with an example.
**A:** Error from wrong/inaccurate assumptions in the model, e.g. fitting a 2nd-order function with linear regression. High when the model is too simple.

**Q67 [Recall]** Define variance with an example.
**A:** Error from sensitivity to small fluctuations in the training set, e.g. fitting noise in a small training set. Increases with model complexity.

**Q68 [Think]** How do bias, variance and training error change as model complexity grows?
**A:** Bias ↓, variance ↑, training error ↓ (keeps falling), validation error is U-shaped (falls then rises).

**Q69 [Think]** Explain the bias–variance trade-off using the bullseye analogy.
**A:** Centre = target model, darts = models learned from different training sets. High bias: darts clustered but off-centre. High variance: darts centred on average but widely scattered. We want to minimise both but reducing one tends to increase the other.

**Q70 [Think]** What is "expected error" and why can't we measure it directly? What do we use instead?
**A:** Probability a random future sample from the true distribution is misclassified. We lack the distribution and test labels, so we estimate it with error on a held-out validation set (a "simulated test set").

**Q71 [Recall]** Roles of the training, validation and test sets?
**A:** Train: fit parameters. Validation: tune hyperparameters, choose model, decide stopping time. Test: final unbiased evaluation after all choices fixed, untouched until then.

**Q72 [Think]** Why can repeated model selection overfit the validation set? What helps?
**A:** Each choice is made by looking at validation error, so the selected model is tuned to that set. Use cross-validation or a fresh holdout; finally report on the test set.

**Q73 [Think]** Why must the test set not be used for tuning?
**A:** Otherwise information leaks and test performance becomes optimistically biased – no longer an honest estimate of generalisation.

**Q74 [Recall]** How do you measure the degree of overfitting?
**A:** Validation error − training error (the generalisation gap), read together with training error: large gap & low training error = overfitting; high training error & small gap = underfitting.

**Q75 [Calc]** Desired error ≈ 2%. Diagnose: (a) train 8%, val 9% (b) train 0.2%, val 7% (c) train 12%, val 25% (d) train 1.5%, val 2.3%.
**A:** (a) High bias (training error far above target, gap 1%). (b) High variance (gap 6.8%). (c) Both high bias and high variance (training error high AND big gap). (d) Close to target – good.

**Q76 [Think]** Suggest remedies for high bias vs high variance.
**A:** High bias: more complex model/more hidden units, train longer, better features, weaken regularization. High variance: more (representative) data, augmentation, regularization (weight decay, dropout, early stopping), weight sharing, simpler model, dimensionality reduction.

**Q77 [Think]** Does regularization always lower training error? Explain.
**A:** No – it aims to improve generalisation and may increase training error while lowering validation error. Strength is chosen by validation performance.

**Q78 [Recall]** What is weight decay (L2 regularization)? Effect of its strength?
**A:** Adds a penalty on weight magnitude (e.g. loss + λ‖w‖²), constraining effective complexity. Larger λ → lower variance but higher bias (too large ⇒ underfitting); choose λ on validation data.

**Q79 [Recall]** Explain early stopping.
**A:** Evaluate on the validation set every few iterations and stop at the minimum validation loss (don't train to minimum training error); later epochs only fit noise.

**Q80 [Recall]** Explain dropout.
**A:** During training, randomly ignore ("drop") some layer outputs each iteration. Prevents co-adaptation of neurons and acts like training many thinned networks, reducing overfitting.

**Q81 [Recall]** What is weight sharing and where is it used?
**A:** Force several neurons/positions to use the same parameters, cutting free parameters (reduces complexity). Examples: RNN (same weights each time step), CNN (same kernel at every location).

**Q82 [Think]** Why does more data (or augmentation) reduce overfitting?
**A:** It lowers sensitivity to individual samples (lower variance) and exposes the model to more variation. Augmentation encodes invariances cheaply.

**Q83 [Think]** List augmentation types. Give a case where an augmentation is harmful.
**A:** Geometric (flip, rotate, shear, crops), photometric (colour changes), other (scale, noise, compression artefacts, lens distortion). Must be label-preserving and realistic: e.g. vertical flip/180° rotation of digits turns 6 into 9; heavy colour shifts for ripe/unripe fruit classification changes the label; avoid obvious artefacts.

**Q84 [Think]** How can dimensionality reduction help generalisation?
**A:** It simplifies the data distribution/dimension (e.g. k=200 → 50 → 2 components), removing noise and reducing free parameters, so the model is less able to overfit.

**Q85 [Think]** How do you diagnose learning-rate problems from the loss curve?
**A:** Too high: loss jumps/explodes or plateaus at a poor value. Too low: loss decreases very slowly and linearly. High-ish: drops fast then stalls at a higher value. Good: steady decrease to a low value.

**Q86 [Think]** You see training loss falling but validation loss rising after epoch 30. What do you conclude and do?
**A:** Overfitting begins ~epoch 30. Use early stopping at the validation minimum, and/or add regularization/data/augmentation.

**Q87 [Recall]** (Appendix) Define expected (true) risk vs empirical risk and excess risk.
**A:** Expected risk = loss averaged over the true p(x,y) (not computable). Empirical risk = average loss over our N samples (computable). ERM minimises empirical risk within hypothesis space ℋ. Excess risk = approximation error (ℋ may not contain the best function; more complex ℋ lowers it) + estimation error (finite sample; shrinks with more data, grows with complexity). This differs from the squared-loss bias–variance decomposition.

---

## Part 4 – Convolutional Neural Networks (Q88–122)

**Q88 [Calc]** A 200×200 RGB image feeds a fully connected layer with 10 units. How many parameters? Why is this a problem?
**A:** 200·200·3 = 120,000 inputs; weights 1,200,000 + 10 biases = **1,200,010**. Parameter count scales with image size – expensive, overfit-prone; conv layers use local connections and shared weights.

**Q89 [Recall]** Explain local connectivity, weight sharing and a feature map.
**A:** Each output depends only on a small spatial patch (kernel size). The same kernel and bias slide over all positions (shared weights). Sliding one filter produces one output channel – a feature map.

**Q90 [Calc]** X = [[1,2,0,1],[0,1,3,1],[2,1,0,2],[1,0,1,1]], K = [[1,0],[0,−1]], stride 1, no padding, bias 0. Compute Z and ReLU(Z). (Note: libraries compute cross-correlation – no kernel flip.)
**A:** Z = [[0,−1,−1],[−1,1,1],[2,0,−1]] (e.g. Z₁₁ = 1·1 + 2·0 + 0·0 + 1·(−1) = 0). ReLU(Z) = [[0,0,0],[0,1,1],[2,0,0]]. Shape 3×3×1.

**Q91 [Calc]** (Slide) 5×5 X=[[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,1,1,0],[0,1,1,0,0]], K=[[−1,0,1]]×3 rows. Verify the first output row.
**A:** Patches cols 1–3: (−1+0+1)+(0+0+1)+(0+0+1)=2; cols 2–4: 0 ; cols 3–5: −2 → **[2, 0, −2]**. Output is 3×3.

**Q92 [Calc]** (Slide) A=[[2,0,0],[3,2,0],[3,1,0]], second conv: 2×2 kernel of 0.25s, stride 1, bias 0. Compute the output. Why does the next layer see feature maps?
**A:** Output [[1.75,0.5],[2.25,0.75]] (e.g. z₁₂=(0+0+2+0)/4=0.5). Each layer's input is the previous layer's output, so deeper layers operate on feature maps (not raw pixels) and combine simple responses.

**Q93 [Think]** What does the feature hierarchy in a CNN look like?
**A:** Early layers: edges, colours. Middle: textures, parts. Later: larger patterns/object parts (Zeiler & Fergus visualisation).

**Q94 [Calc]** Vertical-edge kernel [[−1,0,1]]×3 on patch [[0,0,3]]×3 → response? On a constant patch? Horizontal-edge kernel [[−1,−1,−1],[0,0,0],[1,1,1]] on [[0,0,0],[0,0,0],[3,3,3]]?
**A:** Vertical: 3·(3−0) = **9**; constant patch → 0 (left and right cancel). Horizontal: −0+9 = **9** (bottom minus top).

**Q95 [Calc]** Patch [[1,1,1],[1,9,1],[1,1,1]]. Compute outputs of the 3×3 blur (all 1/9) and sharpen [[0,−1,0],[−1,5,−1],[0,−1,0]].
**A:** Blur: (8·1 + 9)/9 = 17/9 ≈ **1.889**. Sharpen: 5·9 − 4·1 = **41**. In a CNN the kernel values are learned from data.

**Q96 [Calc]** A 3×3 filter on an RGB patch gives channel contributions 2, 3 and −1; bias 0.5. Output value? How many weights in the filter?
**A:** 2+3−1+0.5 = **4.5**. Filter has 3×3×3 = 27 weights + 1 bias; contributions are summed across channels into a single number.

**Q97 [Think]** What decides the number of output channels? Example: 32×32×3 input, six 5×5×3 filters, S=1, P=0.
**A:** The number of filters K. Output = (32−5)/1+1 = 28 → **28×28×6**.

**Q98 [Calc]** Parameter formula for a conv layer? Compute for six 5×5 filters on RGB, and for 64 filters of 7×7 on RGB.
**A:** K(F_h·F_w·C_in + 1). 6(75+1) = **456**; 64(7·7·3+1) = 64·148 = **9,472**.

**Q99 [Calc]** A 1×1 conv with 8 filters is applied to a 32×32×16 volume. Output shape and parameters? Use?
**A:** 32×32×8; 8(1·1·16+1) = **136** params. It mixes channels and reduces depth cheaply (used in Inception before expensive 3×3/5×5).

**Q100 [Think]** Does the parameter count depend on input height/width? What does?
**A:** No – shared weights make it independent of spatial size. Output size still affects computation and memory.

**Q101 [Calc]** 1D input [2,1,3,0,4,1,2,5], kernel [1,−1,2], no padding, bias 0. Outputs for stride 1 and stride 2?
**A:** Stride 1: [7, −2, 11, −2, 7, 9] (length 6). Stride 2: [7, 11, 7] (length 3).

**Q102 [Calc]** 3×3 input of all 1s, zero-pad 1, 3×3 all-ones kernel, stride 1. Output? Why do the borders differ?
**A:** [[4,6,4],[6,9,6],[4,6,4]]; size preserved. Corner/edge positions overlap the zero padding so fewer real values are summed.

**Q103 [Recall]** Give the output-size formula. What is "valid" conv and which padding preserves size at stride 1?
**A:** H_out = ⌊(H_in + 2P − F)/S⌋ + 1 (same for W); C_out = K. Valid = P=0 (shrinks). P=(F−1)/2 preserves size for odd F, S=1.

**Q104 [Calc]** (a) 64×64×3, ten 5×5 filters, S=1, P=2. (b) 224×224×3, 64 filters 7×7, S=2, P=3. Output shapes and parameters.
**A:** (a) (64+4−5)+1=64 → **64×64×10**, params 10(75+1)=760. (b) ⌊(224+6−7)/2⌋+1=⌊111.5⌋+1=112 → **112×112×64**, params 9,472.

**Q105 [Calc]** (a) 13×13×1, 4×4 filter, K=1, S=3, P=0. (b) 8×8×1, four 3×3 filters, S=2, P=0 (slide). (c) 10×14×4, six 3×3 filters, S=2, P=0.
**A:** (a) ⌊9/3⌋+1=**4** → 4×4×1. (b) ⌊5/2⌋+1=3 → **3×3×4** (kernel starts at 0,2,4; start 6 would overrun). (c) H=⌊7/2⌋+1=4, W=⌊11/2⌋+1=6 → **4×6×6**. Floor = discard incomplete windows.

**Q106 [Calc]** Input 20×30×3; five filters 5×3; stride (1,2); padding (2,1) per side. Output?
**A:** H=⌊(20+4−5)/1⌋+1=20; W=⌊(30+2−3)/2⌋+1=⌊14.5⌋+1=15 → **20×15×5**. Apply the formula separately to height and width.

**Q107 [Calc]** Max-pool (2×2, S=2) on [[1,3,2,4],[5,6,1,0],[7,2,9,3],[4,8,5,6]].
**A:** [[6,4],[8,9]].

**Q108 [Calc]** Average-pool the same input.
**A:** [[3.75,1.75],[5.25,5.75]].

**Q109 [Recall]** Properties of pooling: learned weights? Channels? Effect on size? Max vs average?
**A:** No learned weights, done per channel, so channels unchanged; spatial size reduced (14×14×16 → 7×7×16 for 2×2, S=2). Max keeps the strongest response (and gives some translation tolerance); average smooths.

**Q110 [Calc]** Input 32×32×3 → Conv(3×3, K=8, S=1, P=1)+ReLU → MaxPool(2×2, S=2) → Conv(5×5, K=16, S=1, P=0)+ReLU → MaxPool(2×2, S=2) → flatten → FC(10). Give each shape and parameter count.
**A:** 32×32×8 (params 8(27+1)=224) → 16×16×8 → 12×12×16 (16(5·5·8+1)=3,216) → 6×6×16 → flatten 576 → FC: 576·10+10 = 5,770. Total = 224+3,216+5,770 = **9,210**.

**Q111 [Calc]** (Slide final practice) Input 28×28×1 → Conv(3×3, K=8, S=1, P=1)+ReLU → MaxPool(2×2, S=2) → Conv(3×3, K=16, S=2, P=1)+ReLU. Shapes and conv parameters?
**A:** 28×28×8 (80 params) → 14×14×8 → ⌊(14+2−3)/2⌋+1 = 7 → **7×7×16** (16(3·3·8+1)=1,168). Flatten 784. Total conv params 1,248.

**Q112 [Recall]** What is ILSVRC/ImageNet, what is top-5 error, and what did LeNet-5 and AlexNet contribute?
**A:** ~1.2M training images, 1,000 classes; top-5 error = true class not among the 5 highest-scoring predictions. LeNet-5 (1998): conv + subsampling for handwritten digits. AlexNet (2012): 5 conv + 3 FC layers, ReLU, max pooling, dropout – big ImageNet win.

**Q113 [Recall]** Key ideas of VGG.
**A:** Very deep, simple design with repeated 3×3 convs and pooling between stages; VGG-16/19 = 16/19 weight layers. (Beyond slides: two stacked 3×3 give a 5×5 receptive field with fewer params, 18C² vs 25C².)

**Q114 [Recall]** What is an Inception module and why use 1×1 convolutions in it?
**A:** Parallel conv (1×1, 3×3, 5×5) and pooling branches on the same input, concatenated along channels. 1×1 convs first reduce channel count before the expensive 3×3/5×5 ops, cutting computation (GoogLeNet).

**Q115 [Recall]** What is a residual block? Constraint on the addition?
**A:** y = F(x) + x: the block learns a correction (residual) to its input via a shortcut, easing training of very deep nets. x and F(x) must have matching shapes; otherwise a projection (e.g. 1×1 conv) aligns channels/size.

**Q116 [Recall]** What is an RNN and how does it use weight sharing? Example task?
**A:** A hidden state carries information across a sequence; the same parameters are applied at each time step. E.g. sentiment classification of a sentence after reading its words.

**Q117 [Recall]** Compare autoencoders and VAEs. How would you train a denoising autoencoder?
**A:** AE: encoder → latent code → decoder reconstructs input. VAE: encoder outputs a distribution over latents; sample then decode, enabling generation. Denoising AE: noisy input, clean image as reconstruction target.

**Q118 [Recall]** Describe GAN training.
**A:** Generator maps random noise to fake samples; discriminator distinguishes real vs generated. They are trained alternately, each improving against the other.

**Q119 [Recall]** What is CycleGAN and cycle consistency?
**A:** Unpaired image-to-image translation (e.g. horse ↔ zebra). Cycle consistency: translating A→B→A should recover the original, preserving content.

**Q120 [Recall]** Key components of the Transformer.
**A:** Attention: queries and keys give weights for combining values across positions; positional information supplies order; original model = encoder + autoregressive decoder.

**Q121 [Recall]** How do diffusion models train and generate?
**A:** Training: add noise progressively and learn to reverse the corruption. Generation: start from noise and denoise repeatedly (e.g. latent diffusion).

**Q122 [Think]** Match the model to the task: (a) classify images (b) sentiment of a sentence (c) generate new faces (d) horse→zebra with unpaired images (e) denoise images with an encoder–decoder (f) text from long-range dependencies via attention.
**A:** (a) CNN (b) RNN (or Transformer) (c) GAN / VAE / diffusion (d) CycleGAN (e) denoising autoencoder (f) Transformer.

---

## Part 5 – Clustering (Q123–150)

**Q123 [Recall]** Define unsupervised learning, clustering and cluster.
**A:** Unsupervised: finds patterns without labels. Clustering: group data so that points in the same group are more similar to each other than to other groups. Cluster: a set of mutually similar data points.

**Q124 [Recall]** Clustering vs classification?
**A:** Classification is supervised (labels given in training; predicts labels of unseen data). Clustering is unsupervised (no labels; discovers structure; outputs a cluster membership per point).

**Q125 [Recall]** Input and output of clustering. What must you define to say "similar"?
**A:** Input: data points, no labels. Output: membership of each point. Need a distance metric d(xᵢ,xⱼ).

**Q126 [Calc]** Compute Euclidean, Manhattan and sup distance between x=(1,2,3) and y=(4,0,3).
**A:** L2 = √(9+4+0) = **3.606**; L1 = 3+2+0 = **5**; L∞ = max(3,2,0) = **3**.

**Q127 [Recall]** Partitional vs hierarchical algorithms, with examples.
**A:** Partitional: non-overlapping subsets, each point in exactly one cluster (K-Means, mixture of Gaussians, spectral). Hierarchical: nested clusters as a tree (agglomerative, divisive).

**Q128 [Recall]** What cost does K-Means minimise? Explain r_ik and the constraints.
**A:** min Σᵢ Σₖ r_ik ‖xᵢ − μ_k‖² (the slide includes a ½ factor). r_ik ∈ {0,1} (hard assignment), Σₖ r_ik = 1 (each point in exactly one cluster), μ_k = Σᵢ r_ik xᵢ / Σᵢ r_ik (mean of assigned points).

**Q129 [Recall]** Write the K-Means algorithm and its stopping criterion.
**A:** Init K random centres. Repeat: (1) assign each point to nearest centre; (2) update each centre to the mean of its points. Stop when no assignment changes.

**Q130 [Think]** Why does the update step use the mean, and why does the algorithm converge?
**A:** The mean minimises the sum of squared Euclidean distances for a fixed cluster. Both steps never increase the cost and the cost is bounded below, so it converges – but only to a local minimum.

**Q131 [Calc]** (Slide) Points A1(2,10), A2(2,5), A3(8,4), B1(5,8), B2(7,5), B3(6,4), C1(1,2), C2(4,9); K=3, initial centres A1, B1, C1. Give the first assignment and new centres.
**A:** Cluster 1={A1}; cluster 2={B1,B2,B3,A3,C2}; cluster 3={A2,C1}. New centres (2,10), (6,6), (1.5,3.5).

**Q132 [Calc]** Continue Q131 to convergence.
**A:** Iteration 2: C2 moves to cluster 1 → centres (3,9.5), (6.5,5.25), (1.5,3.5). Iteration 3: B1 moves to cluster 1 → centres (3.67,9), (7,4.33), (1.5,3.5). Iteration 4: no change → stop. Final: {A1,B1,C2}, {A3,B2,B3}, {A2,C1}; cost (sum of squared distances) ≈ 14.33.

**Q133 [Calc]** Points P1(1,1), P2(2,1), P3(4,3), P4(5,4), P5(1,2), P6(6,5); K=2, init centres P1 and P4. Run K-Means.
**A:** Assign: P1,P2,P5 → cluster 1; P3,P4,P6 → cluster 2 (P3: 3.61 to P1 vs 1.41 to P4). New centres (1.33,1.33) and (5,4). Reassigning gives the same clusters → converged; cost ≈ 5.33.

**Q134 [Think]** How is K-Means used on images? What are the features for grayscale vs colour segmentation?
**A:** Each pixel is a data point; grayscale intensity gives 1-D features (K=2 → foreground/background), RGB gives 3-D features; cluster centres become the segment "colours".

**Q135 [Recall]** Pros and cons of K-Means.
**A:** Pros: simple, cheap, easy to implement. Cons: K must be chosen, stuck in poor local minima / initialisation-sensitive, needs an appropriate distance metric.

**Q136 [Think]** How does initialisation affect K-Means? How can you mitigate it?
**A:** Different starting centres can give different local minima (a bad init can split one natural cluster and merge two others). Mitigate by running several random restarts and keeping the lowest cost (or smarter seeding).

**Q137 [Think]** When does K-Means with Euclidean distance fail on shape, and what is the remedy?
**A:** For data that isn't linearly separable (e.g. concentric rings), Euclidean-centroid clusters cut the wrong way. Need a better metric/feature space in which clusters are separable (e.g. spectral/kernel-style approaches).

**Q138 [Think]** K must be chosen. Why can't you pick K by minimising the K-Means cost?
**A:** The cost always decreases as K increases (K=N gives 0). One looks for where the improvement levels off ("elbow") or uses domain knowledge/validation – (extension beyond slides). HAC avoids committing to K up front.

**Q139 [Recall]** Describe the HAC algorithm.
**A:** Start with every point as its own cluster. Repeatedly merge the pair of clusters with minimum distance, updating the distance table, until one cluster (or K clusters) remain.

**Q140 [Recall]** What is a dendrogram? How do you get K clusters from it? What does the y-axis show?
**A:** A tree recording the sequence of merges. Cut horizontally at the level that leaves K branches. The y-axis is the cluster distance at which each merge occurred.

**Q141 [Recall]** Advantages of HAC.
**A:** No need to pre-define K (cut dendrogram anywhere); gives results for all K; deterministic – independent of initialisation.

**Q142 [Recall]** Define single, complete, average and centroid linkage.
**A:** Single (MIN): smallest pairwise distance between members. Complete (MAX): largest pairwise distance. Average: mean of all pairwise distances. Centroid: distance between cluster means.

**Q143 [Calc]** (Slide) Points #1(1.9,1.0), #2(1.8,0.9), #3(2.3,1.6), #4(2.3,2.1). Run HAC with centroid distance.
**A:** Closest: #1–#2 (0.14) → centroid (1.85,0.95). Then #3–#4 (0.50) → centroid (2.3,1.85). Finally merge into one cluster with centroid (2.075,1.4).

**Q144 [Calc]** For the same points, what is the distance used to merge {#1,#2} with {#3,#4} under single, complete, average and centroid linkage? (Pairwise: d13=0.72, d14=1.17, d23=0.86, d24=1.30.)
**A:** Single = **0.72**; complete = **1.30**; average = (0.72+1.17+0.86+1.30)/4 = **1.01**; centroid = dist((1.85,0.95),(2.3,1.85)) ≈ **1.01**.

**Q145 [Calc]** 1-D points {1, 2, 6, 8, 15}. Run HAC with single linkage; list the merge heights.
**A:** (1,2) @1; (6,8) @2; {1,2}+{6,8} @ min = 6−2 = 4; add 15 @ 15−8 = 7. Heights 1, 2, 4, 7.

**Q146 [Calc]** Same points: complete and average linkage merge heights.
**A:** Complete: 1, 2, then {1,2}+{6,8} @ max = 8−1 = 7, then 15 @ 15−1 = 14. Average: 1, 2, {1,2}+{6,8} @ (5+7+4+6)/4 = 5.5, then 15 @ (14+13+9+7)/4 = 10.75. With K=3 all linkages give {1,2},{6,8},{15}.

**Q147 [Calc]** After merging A4(5,8) and A6(4,9) the centroid is (4.5,8.5). A1(2,10) then joins. Why is the new centroid not ((4.5,8.5)+(2,10))/2, and what is it?
**A:** The two-point average ignores that the first cluster has 2 points. Correct: (A1+A4+A6)/3 = (3.67, 9) = [(4.5,8.5)·2 + (2,10)]/3. Always weight centroids by cluster size.

**Q148 [Calc]** (Slide) A1(2,10), A2(2,5), A3(8,4), A4(5,8), A5(1,2), A6(4,9), centroid HAC. After merging (A4,A6) and then A1 into it (centroid (3.67,9)), which pair merges next, and at what distance?
**A:** Distances: A2–A5 = 3.16 is smallest (vs A2–(A1A4A6) 4.33, A3–A2 6.08). Merge A2, A5 → centroid (1.5,3.5). Next: (A1A4A6)–(A2A5) ≈ 5.91, which beats A3 to either cluster (A3–(A2A5) ≈ 6.52, A3–(A1A4A6) ≈ 6.61), so those two clusters merge before A3 joins.

**Q149 [Think]** Compare K-Means and HAC.
**A:** K-Means: simple/cheap; sensitive to initialisation; K pre-defined. HAC: deterministic; shows results for a range of K; but more memory and computation intensive (distance table over all pairs).

**Q150 [Think]** Limitations of HAC, and how does linkage choice change the clusters?
**A:** Memory/computation heavy (O(N²) distance table), greedy merges can't be undone, result depends on the distance/linkage choice. Single linkage can "chain" elongated clusters together (uses closest pair only); complete linkage favours compact clusters (uses furthest pair); average/centroid sit between. A good metric is still required.

---

### Quick formula sheet
- Bayes: P(h|D) = P(D|h)P(h)/P(D); h_MAP = argmax P(D|h)P(h); NB: argmax P(v)∏P(aᵢ|v)
- Perceptron: y = step(wᵀx + b); sigmoid σ′ = σ(1−σ); tanh′ = 1 − tanh²
- Backprop: δ_out = (t−o)σ′(net); δ_hid = σ′(net)Σwₖⱼδₖ; Δw = ηδ·o_in; Δb = ηδ
- Overfitting gap ≈ val error − train error; high bias: train error high; high variance: large gap
- Conv: H_out = ⌊(H_in + 2P − F)/S⌋ + 1; C_out = K; params = K(F_h F_w C_in + 1)
- K-Means: assign to nearest centre → centres = means → stop when assignments don't change
- HAC: merge closest pair; update distances by linkage (single = min, complete = max, average, centroid)
