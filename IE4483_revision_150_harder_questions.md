# IE4483 / EE6483 – 150 Harder Revision Questions (with answers)

Harder, multi-step and cross-topic questions drawn from the same five lectures. Sections: Bayesian Inference (Q1–20) · Neural Networks (Q21–55) · Regularization (Q56–75) · CNNs (Q76–105) · Clustering (Q106–130) · Cross-topic Mix (Q131–150).

Tags: **[Calc]** = work it out · **[Think]** = reasoning / which-statement. Numeric answers were computed in code. Questions marked *(extension)* use a small idea beyond the slides.

> Slide errata spotted: the Naïve Bayes tennis slide shows 0.0051 (should be ≈0.0053); the HAC recap distance table lists A4–A5 = 3.61 and (A4,A6)–A5 = 10.81, which do not match the coordinates (A4–A5 = 7.21, (A4,A6)–A5 = 7.38).

---

## Part 1 – Bayesian Inference (Q1–20)

**Q1 [Calc]** Disease prevalence is 2%. A test is positive for 90% of sick people and for 8% of healthy people. (a) P(disease | one positive test)? (b) The patient takes a second, conditionally independent test of the same kind and it is also positive. P(disease | both positive)? (Use the first posterior as the new prior.)

**Answers:** (a) after one positive = 0.186722; (b) after two positives = 0.720897

**A:** (a) P(+)=0.9·0.02+0.08·0.98=0.0964, posterior=0.1867. (b) the posterior becomes the prior: 0.9·0.1867/(0.9·0.1867+0.08·0.8133) = 0.7209.

**Q2 [Calc]** Three boxes are chosen with prior probabilities 0.5, 0.3, 0.2. The probability of drawing a red ball is 0.2, 0.5, 0.9 respectively. A red ball is drawn. Find (a) P(red) and (b) P(box 3 | red).

**Answers:** (a) P(red) = 0.43; (b) P(box 3 | red) = 0.418605

**A:** P(red)=0.1+0.15+0.18=0.43 (total probability). P(box3|red)=0.18/0.43=0.4186.

**Q3 [Calc]** A coin is either fair or biased with P(H)=0.8, each with prior 0.5. You observe H, H, T in order. (a) P(biased | HHT)? (b) The probability that the next flip is heads?

**Answers:** (a) P(biased | HHT) = 0.505929; (b) P(next = H) = 0.651779

**A:** Likelihoods: fair 0.125, biased 0.8·0.8·0.2=0.128. Posterior biased=0.5059. Next flip: 0.4941·0.5+0.5059·0.8=0.6518.

**Q4 [Calc]** Hypothesis F: fair coin (prior 0.9). Hypothesis B: biased coin with P(H)=0.8 (prior 0.1). Data: HHH. Give (a) the likelihood P(D|B) and (b) the posterior P(F | D).

**Answers:** (a) P(D|B) = 0.512; (b) P(F | D) = 0.687233

**A:** P(D|B)=0.8³=0.512, P(D|F)=0.125. P(F|D)=0.9·0.125/(0.9·0.125+0.1·0.512)=0.6872.

**Q5 [Think]** With the setup of the previous question (prior fair 0.9, biased 0.1, data HHH), which hypotheses do ML and MAP choose?

**Correct:** ML chooses the biased coin; MAP chooses the fair coin  
*Distractors:* ML chooses the fair coin; MAP chooses the biased coin / Both choose the biased coin / Both choose the fair coin

**A:** ML maximises P(D|h): 0.512 > 0.125 → biased. MAP maximises P(D|h)P(h): 0.1125 for fair > 0.0512 for biased → fair. A strong prior can overturn the likelihood.

**Q6 [Calc]** Monty Hall with a biased host. You pick Door 1. If the prize is behind Door 1 the host opens Door 3 with probability 0.7 (Door 2 with 0.3). If the prize is behind Door 2 he must open Door 3; if behind Door 3 he must open Door 2. The host opens Door 3. Find P(prize behind Door 1 | host opens 3) and P(prize behind Door 2 | host opens 3).

**Answers:** P(c₁ | D₃) = 0.411765; P(c₂ | D₃) = 0.588235

**A:** P(D3)=(1/3)(0.7+1+0)=0.5667. P(c1|D3)=(0.7/3)/0.5667=0.4118; P(c2|D3)=0.5882. Switching still wins more often (>1/2), but less than the 2/3 of the unbiased host.

**Q7 [Calc]** Monty Hall with 5 doors: you pick one; the host, who knows where the prize is, opens 3 of the other 4 doors, all empty; you may switch to the single remaining closed door. Give P(win) if you (a) stay, (b) switch.

**Answers:** (a) stay = 0.2; (b) switch = 0.8

**A:** Your first pick is right with probability 1/5. If it was wrong (4/5) the host leaves the prize door closed, so switching wins with probability 4/5.

**Q8 [Calc]** Training emails (Free, Link, class): (Y,Y,S), (Y,N,S), (Y,Y,S), (N,N,S), (Y,N,H), (N,Y,H), (N,N,H), (N,N,H), where S = spam and H = ham. Using counts and Naïve Bayes, classify a new email with Free = Y and Link = Y. Give the unnormalised scores for S and H and the posterior P(S | obs).

**Answers:** Score S = 0.1875; Score H = 0.03125; P(S | obs) = 0.857143

**A:** P(S)=P(H)=0.5. P(Free=Y|S)=3/4, P(Link=Y|S)=2/4 → 0.18750. P(Free=Y|H)=1/4, P(Link=Y|H)=1/4 → 0.03125. Posterior=0.857 → spam.

**Q9 [Calc]** Equal priors. A feature A appears in spam with probability 0.8 and in ham with probability 0.3. Compute P(spam | A) under Naïve Bayes when (a) A is recorded once, (b) the same feature is accidentally recorded twice as two 'independent' features.

**Answers:** (a) single copy = 0.727273; (b) duplicated = 0.876712

**A:** (a) 0.8/(0.8+0.3)=0.7273. (b) 0.8²/(0.8²+0.3²)=0.8767. Duplicating a feature double-counts the evidence — the conditional-independence assumption is violated.

**Q10 [Calc]** Suppliers 1, 2, 3 provide 50%, 30%, 20% of parts with defect rates 1%, 2%, 5%. A defective part is found. Give (a) P(defective) and (b) P(supplier 3 | defective).

**Answers:** (a) P(defective) = 0.021; (b) P(supplier 3 | def) = 0.47619

**A:** P(def)=0.005+0.006+0.010=0.021. P(S3|def)=0.010/0.021=0.4762.

**Q11 [Calc]** Two fair dice are rolled. Given that the sum is 8, what is the probability that at least one die shows a 6?

**Answers:** probability = 0.4

**A:** Sum 8: (2,6),(3,5),(4,4),(5,3),(6,2). Two of the five contain a 6 → 2/5 = 0.4.

**Q12 [Calc]** IE4483 class: 40% IEM (pass rate 50%), 60% EEE (pass rate 30%). A student is chosen at random and is found to have FAILED. What is the probability the student is from IEM?

**Answers:** P(IEM | fail) = 0.322581

**A:** P(fail)=1−0.38=0.62. P(fail|IEM)=0.5, so P(IEM|fail)=0.5·0.4/0.62=0.3226.

**Q13 [Think]** Which statement about Bayes' theorem is FALSE?

**Correct:** P(D) changes which hypothesis has the highest posterior, because it differs between hypotheses  
*Distractors:* The posterior is proportional to the likelihood times the prior / P(D) can be computed with the law of total probability over the hypotheses / If the prior is uniform, the MAP hypothesis equals the ML hypothesis

**A:** P(D) is a constant shared by every hypothesis; it only normalises. It can never change the argmax.

**Q14 [Think]** With a uniform prior over a finite set of hypotheses, which statement is correct?

**Correct:** The MAP and ML hypotheses coincide, but the posterior probabilities are still normalised by P(D)  
*Distractors:* The MAP hypothesis ignores the data / The posterior equals the prior / The MAP hypothesis maximises P(h) only

**A:** With constant P(h), argmax P(D|h)P(h) = argmax P(D|h). The posterior values still need P(D) to sum to 1.

**Q15 [Calc]** In the standard Monty Hall game you decide by a coin flip whether to switch. More generally you switch with probability p. Give P(win) for (a) p = 0.5 and (b) p = 0.25.

**Answers:** (a) p = 0.5 = 0.5; (b) p = 0.25 = 0.416667

**A:** P(win)=(1−p)(1/3)+p(2/3)=(1+p)/3. p=0.5 → 0.5; p=0.25 → 0.4167.

**Q16 [Calc]** (Play Tennis) Observe only Outlook = Sunny and Humidity = High. Using P(yes)=9/14, P(Sunny|yes)=2/9, P(High|yes)=3/9, P(no)=5/14, P(Sunny|no)=0.6, P(High|no)=0.8, give the unnormalised Naïve Bayes scores and P(yes | obs).

**Answers:** Score yes = 0.047619; Score no = 0.171429; P(yes | obs) = 0.217391

**A:** yes: 0.0476; no: 0.1714; P(yes|obs)=0.2174 → predict no. Ignored attributes simply drop out of the product.

**Q17 [Think]** Which statement about the Naïve Bayes assumption is correct?

**Correct:** P(a₁,a₂|v) = P(a₁|v)P(a₂|v) holds by assumption, but P(a₁,a₂) = P(a₁)P(a₂) is not implied  
*Distractors:* It implies the features are independent of each other without conditioning on the class / It requires the features to be uncorrelated in the whole dataset / It assumes the class prior is uniform

**A:** Features are assumed conditionally independent given the class. Marginally they are usually dependent, because both depend on the class.

**Q18 [Calc]** Binary features, K = 2 classes, d = 10 features. How many free probabilities must be estimated for (i) Naïve Bayes and (ii) the full joint P(a₁…a₁₀|class)? (include the class prior)

**Correct:** (i) 21, (ii) 2047  
*Distractors:* (i) 21, (ii) 41 / (i) 20, (ii) 2048 / (i) 1024, (ii) 2047

**A:** NB: 1 prior + 2·10 conditionals = 21. Full joint: 1 prior + 2·(2¹⁰−1)=2047.

**Q19 [Calc]** Odds form of Bayes: posterior odds = prior odds × likelihood ratio. For prevalence 1%, sensitivity 99% and false-positive rate 5%, give (a) the likelihood ratio of a positive test, and (b) the posterior odds of disease after one positive test.

**Answers:** (a) likelihood ratio = 19.8; (b) posterior odds = 0.2

**A:** LR=P(+|D)/P(+|¬D)=0.99/0.05=19.8. Posterior odds=(1/99)·19.8=0.2 → probability 0.2/1.2=0.1667.

**Q20 [Calc]** Three fair coins are flipped. Given that exactly two of them are heads, what is the probability that the first flip is heads?

**Answers:** probability = 0.666667

**A:** Outcomes with two heads: HHT, HTH, THH. First is H in two of three → 2/3.

---

## Part 2 – Neural Networks (Q21–55)

**Q21 [Calc]** An MLP has 784 inputs, hidden layers of 128 and 64 units, and 10 outputs, fully connected with a bias per unit. How many trainable parameters?

**Answers:** parameters = 109386

**A:** (784+1)·128 + (128+1)·64 + (64+1)·10 = 100,480 + 8,256 + 650 = 109,386.

**Q22 [Calc]** A network has 3 inputs, one hidden layer of 4 units and 2 outputs. Give (a) the number of weights (excluding biases) and (b) the total parameters including biases.

**Answers:** (a) weights = 20; (b) total = 26

**A:** Weights: 3·4 + 4·2 = 20. Biases: 4 + 2 = 6. Total 26.

**Q23 [Calc]** Two LINEAR layers: h = W₁x + b₁ with W₁ = [[1,2],[0,1]], b₁ = (1,0); o = W₂h + b₂ with W₂ = (1, −1), b₂ = 0.5. Write o as w₁x₁ + w₂x₂ + b. Give w₁, w₂ and b.

**Answers:** w₁ = 1; w₂ = 1; b = 1.5

**A:** Composition of affine maps is affine: w = W₂W₁ = (1, 1), b = W₂b₁ + b₂ = 1.5. Depth adds nothing without a nonlinearity.

**Q24 [Calc]** XOR-style MLP: h₁ = step(x₁ − x₂ − 0.5), h₂ = step(x₂ − x₁ − 0.5), y = step(h₁ + h₂ − 0.5). Give the net input z of the output unit for (a) input (1,0) and (b) input (1,1).

**Answers:** (a) z for (1,0) = 0.5; (b) z for (1,1) = -0.5

**A:** (1,0): h₁=step(0.5)=1, h₂=step(−1.5)=0 → z=0.5, y=1. (1,1): h₁=step(−0.5)=0, h₂=0 → z=−0.5, y=0. The network computes XOR.

**Q25 [Calc]** 2-2-2 sigmoid network. Inputs x = (1, 0.5). Hidden weights [[0.4,−0.2],[0.1,0.3]], hidden biases (0.1, −0.1). Output weights [[0.5,−0.3],[0.2,0.4]], output biases (0, 0.1). All units sigmoid. Targets t = (1, 0). Compute the outputs o₁, o₂ and the loss E = ½Σ(t−o)².

**Answers:** o₁ = 0.534474; o₂ = 0.606998; E = 0.29258

**A:** h=(0.5987, 0.5374); net_o=(0.1381, 0.4347); o=(0.5345, 0.6070); E=0.2926.

**Q26 [Calc]** Same network (outputs o = (0.5345, 0.6070), hidden h = (0.5987, 0.5374)). Compute the output error signals δ_o1, δ_o2 and the hidden error signal δ_h1 (the hidden unit feeds BOTH outputs).

**Answers:** δ_o1 = 0.115828; δ_o2 = -0.1448; δ_h1 = 0.00695653

**A:** δ_ok=(t−o)o(1−o): (0.1158, -0.1448). δ_h1=h₁(1−h₁)Σₖ w_k1 δ_k = 0.2403·(0.5·0.1158 + 0.2·-0.1448) = 0.00696. Hidden errors sum over all downstream paths.

**Q27 [Calc]** Continue (δ_h1 = 0.00696, x₁ = 1, η = 0.5). Give the updated weight from x₁ to hidden unit 1 (was 0.4) and the updated bias of hidden unit 1 (was 0.1).

**Answers:** new w (x₁→h₁) = 0.403478; new b_h1 = 0.103478

**A:** w=0.4+0.5·0.00696·1=0.4035; b=0.1+0.5·0.00696=0.1035.

**Q28 [Calc]** Ignoring the weights, what is the largest possible value of the product of sigmoid derivatives σ′(z) through 5 stacked sigmoid layers? (σ′ ≤ 0.25.) Give it to 6 decimals.

**Answers:** upper bound = 0.000976562

**A:** 0.25⁵ = 9.77×10⁻⁴. Repeated sigmoid derivatives shrink gradients quickly, which is why saturating activations slow deep training.

**Q29 [Calc]** Linear neuron o = wx, E = ½(t−o)², w₀ = 0, η = 0.1. Data: (x=1, t=2) then (x=2, t=2). Give w after (a) case (SGD) updating through the two examples in order, and (b) one full-batch update using the AVERAGE gradient over both examples.

**Answers:** (a) SGD, two updates = 0.52; (b) full-batch, one update = 0.3

**A:** (a) after ex1: δ=2, w=0.2; ex2: o=0.4, δ=1.6, Δw=0.1·1.6·2=0.32 → w=0.52. (b) mean of δx = (2·1+2·2)/2=3 → w=0.3. Sequential updates use the changed weight.

**Q30 [Calc]** Gradient descent on E = ½x²w² with x = 2 and η = 0.6, starting at w = 1. (a) w after 3 steps. (b) What is the largest learning rate (exclusive) for which the iteration converges?

**Answers:** (a) w after 3 steps = -2.744; (b) η limit = 0.5

**A:** ∂E/∂w=x²w=4w so w←w(1−4η). With η=0.6 the factor is −1.4: w₃=(−1.4)³=−2.744 (diverges). Convergence needs |1−4η|<1 → η<0.5.

**Q31 [Calc]** (Slide Example 2 setup) x = 1, t = 1, w = 0, b_h = 0, v = 1, b_o = 0, η = 0.5, all sigmoid. The correct hidden error signal is δ_h = h(1−h)·v·δ_o = 0.02218 using the OLD v = 1. If a student wrongly used the UPDATED v = 1.02218 (δ_o = 0.08872, h = 0.5), what δ_h would they get?

**Answers:** wrong δ_h = 0.022672

**A:** 0.25·1.02218·0.08872=0.02267. The difference is small here but conceptually wrong: all δ must be computed with the forward-pass weights.

**Q32 [Calc]** Hidden unit tanh, linear output: h = tanh(wx), o = vh, with x = 1, t = 0.5, w = 0.5, v = 2, η = 0.1 (no biases). Compute o, the hidden error δ_h = (1−h²)vδ_o, and the new w.

**Answers:** o = 0.924234; δ_h = -0.667276; new w = 0.433272

**A:** h=tanh(0.5)=0.4621, o=0.9242, δ_o=t−o=-0.4242, δ_h=(1−0.2136)·2·-0.4242=-0.6673, w=0.4333.

**Q33 [Calc]** A ReLU hidden unit h = ReLU(wx + b) with w = 1, b = 0, x = 2 feeds two LINEAR outputs o₁ = 0.5h and o₂ = −1h. Targets t = (2, −1). η = 0.1. Give δ_h (sum over both paths) and the new w.

**Answers:** δ_h = -0.5; new w = 0.9

**A:** h=2, o=(1, −2), δ_o=(1, 1). δ_h=ReLU′·(0.5·1 + (−1)·1)=−0.5. Δw=ηδ_h x=0.1·(−0.5)·2=−0.1 → w=0.9.

**Q34 [Calc]** A training set has 1200 examples. Mini-batches of 40 are used for 15 epochs. How many parameter updates are made in total, and how many updates per epoch?

**Answers:** total updates = 450; updates per epoch = 30

**A:** 1200/40=30 updates per epoch; 30·15=450.

**Q35 [Think]** Which statement about backpropagation is FALSE?

**Correct:** δ for a hidden unit can be computed before any downstream δ is known  
*Distractors:* The update Δw_ji = ηδ_j o_i uses the activation of the unit feeding the weight / The same σ′ factor appears in both output and hidden error signals / All δ should be computed before any weight is changed

**A:** Hidden δ depends on the downstream δ values (δ_j = σ′ Σ_k w_kj δ_k), so errors must flow backward from the output.

**Q36 [Think]** Why are weights initialised to small random values (not all equal)?

**Correct:** If hidden units start identical they receive identical gradients and stay identical; random values break this symmetry  
*Distractors:* Random values guarantee the global minimum / Equal weights make the loss undefined / Small random values remove the need for a learning rate

**A:** With equal initial weights every hidden unit computes the same function and gets the same δ, so they never specialise.

**Q37 [Think]** A network has two hidden layers with LINEAR activation and a step output unit. Which problem can it still NOT solve?

**Correct:** XOR of two binary inputs  
*Distractors:* AND of two binary inputs / OR of two binary inputs / Any linearly separable problem

**A:** Linear hidden layers collapse to one affine map, so the whole network is still a single linear classifier.

**Q38 [Think]** For squared error, if the target is above the output (t > o) for a sigmoid output unit, what happens to the weight w from a unit with activation o_i > 0?

**Correct:** w increases, since δ > 0 and Δw = ηδo_i > 0  
*Distractors:* w decreases, since the gradient is positive / w is unchanged because σ′ is zero / The sign depends only on the learning rate

**A:** δ=(t−o)σ′>0, and Δw=+ηδo_i. Increasing w raises net and hence the output towards the target.

**Q39 [Calc]** A sigmoid output unit has output o = 0.01 while the target is t = 1. Compare the error signals: (a) δ for a sigmoid output with squared error and (b) δ if the output unit were linear with the same o.

**Answers:** (a) sigmoid δ = 0.009801; (b) linear δ = 0.99

**A:** Sigmoid: (1−0.01)(0.01)(0.99)=0.0098, tiny despite a large error (saturation). Linear: t−o=0.99.

**Q40 [Calc]** The ReLU net h₁ = f(x₁+x₂−0.5), h₂ = f(−x₁+2x₂+1), o = 2h₁ − h₂ + 0.1 is evaluated at x = (−2, 1). Give o when f is (a) ReLU and (b) Leaky ReLU with a = 0.1.

**Answers:** (a) ReLU = -4.9; (b) Leaky ReLU = -5.2

**A:** net₁=−1.5, net₂=5. ReLU: h=(0,5) → o=−4.9. Leaky: h₁=−0.15 → o=2(−0.15)−5+0.1=-5.2.

**Q41 [Calc]** A perceptron has w = (3, 4) and b = −5. For the point (3, 4) compute (a) z = wᵀx + b and (b) its signed distance to the boundary z/‖w‖.

**Answers:** (a) z = 20; (b) distance = 4

**A:** z=9+16−5=20; ‖w‖=5; distance=20/5=4. The sign gives the side (class 1).

**Q42 [Think]** Design y = x₁ AND (NOT x₂) with one perceptron. Which works?

**Correct:** y = step(x₁ − x₂ − 0.5)  
*Distractors:* y = step(x₁ + x₂ − 0.5) / y = step(x₁ − x₂ + 0.5) / y = step(−x₁ + x₂ − 0.5)

**A:** (1,0): 0.5>0 → 1. (0,0): −0.5 → 0; (1,1): −0.5 → 0; (0,1): −1.5 → 0.

**Q43 [Think]** What is the minimum number of hidden step units needed (with a step output) to compute 2-input XOR?

**Correct:** 2  
*Distractors:* 1 / 3 / 0 — the bias alone is enough

**A:** XOR = OR AND NOT(AND), so two hidden boundaries (OR and AND) are needed, then a combining unit.

**Q44 [Calc]** (Slide Example 1 with a smaller step) x = (1,0,1), t = 1. Use δ₆ = 0.13117, δ₄ = −0.008725, o₄ = 0.3318 and η = 0.5. Give the new w₆₄ (was −0.3), the new b₆ (was 0.1) and the new weight from x₁ to unit 4 (was 0.2).

**Answers:** new w₆₄ = -0.278239; new b₆ = 0.165585; new w(x₁→4) = 0.195638

**A:** w₆₄=−0.3+0.5·0.13117·0.3318=-0.2782; b₆=0.1+0.5·0.13117=0.1656; w=0.2+0.5·(−0.008725)·1=0.1956.

**Q45 [Think]** In Example 1, x₂ = 0. What happens to the two weights from x₂ to the hidden units, and to the hidden biases, in this update?

**Correct:** The x₂ weights are unchanged (Δw = ηδ·0), but the hidden biases still change  
*Distractors:* Both the x₂ weights and the biases are unchanged / The x₂ weights change but the biases do not / All of them change by the same amount

**A:** Δw_ji=ηδ_j x_i = 0 when x_i = 0, whereas Δb_j = ηδ_j has no input factor.

**Q46 [Think]** A ReLU hidden unit is inactive (z ≤ 0) for the current training example. Which parameters change after that example's backprop step?

**Correct:** None of its incoming weights/bias and none of its outgoing weights change  
*Distractors:* Only its incoming weights change / Only its outgoing weights change / Its bias changes but not its weights

**A:** Local derivative is 0 so δ=0 for incoming parameters; its activation is 0 so outgoing weight updates ηδ_k·0 = 0.

**Q47 [Calc]** Linear neuron o = wx with w = 0.5, E = ½(t−o)², η = 0.1, on a mini-batch of three examples (x,t) = (1,1), (2,0), (−1,1). Give (a) the mean of δx over the batch, and (b) w after one mini-batch update (average gradient).

**Answers:** (a) mean δx = -1; (b) new w = 0.4

**A:** δx: 0.5, −2, −1.5 → mean -1.00. w=0.5+0.1·(-1.00)=0.40.

**Q48 [Think]** Which choice correctly separates the roles of the sigmoid output and the class threshold? A binary classifier outputs 0.62.

**Correct:** Activation gives 0.62; with a 0.5 threshold the class is 1, and a 0.7 threshold would give class 0 without retraining  
*Distractors:* The threshold must be applied inside each hidden layer / Changing the threshold requires recomputing the gradients / 0.62 is the class label

**A:** The activation is part of the model; the threshold is a decision rule on the final output.

**Q49 [Think]** Which stopping rule best guards against overfitting?

**Correct:** Stop when validation loss stops improving, even if training loss is still decreasing  
*Distractors:* Stop when training loss reaches zero / Stop after exactly 10 epochs regardless / Stop when the weights are all positive

**A:** Training loss alone does not show generalisation; validation performance does.

**Q50 [Calc]** Hidden layer of 3 sigmoid units feeding one sigmoid output. For a given example the hidden activations are (0.5, 0.8, 0.2), the output weights are (1, −2, 0.5) and the output error signal is δ_o = 0.1. Give the three hidden error signals δ_j = h_j(1−h_j)w_jδ_o.

**Answers:** δ_h1 = 0.025; δ_h2 = -0.032; δ_h3 = 0.008

**A:** δ₁=0.25·1·0.1=0.025; δ₂=0.16·(−2)·0.1=−0.032; δ₃=0.16·0.5·0.1=0.008.

**Q51 [Calc]** Loss after one update depends on the learning rate. For the 1-D linear neuron o = wx with x = 1, t = 1, w = 0, E = ½(1−o)², give the loss after one gradient step for (a) η = 0.5 and (b) η = 2.5.

**Answers:** (a) η = 0.5 = 0.125; (b) η = 2.5 = 1.125

**A:** w₁=η·1 (since δ=1). η=0.5 → o=0.5, E=0.125. η=2.5 → o=2.5, E=1.125, larger than the initial 0.5: too large a step overshoots.

**Q52 [Think]** Which is the correct form of the hidden error signal?

**Correct:** δ_j = σ′(net_j) Σ_k w_kj δ_k  
*Distractors:* δ_j = σ′(net_j) Σ_k w_jk δ_j / δ_j = (t_j − o_j) Σ_k w_kj / δ_j = σ′(net_k) Σ_i w_ji δ_i

**A:** Weight each downstream δ by the connection from j to k, sum, then multiply by the local derivative.

**Q53 [Calc]** A sigmoid unit has net input −0.7. Use σ(−0.7)=0.3318 to compute σ′(−0.7) and compare with σ′(0). Give σ′(−0.7) and the ratio σ′(−0.7)/σ′(0).

**Answers:** σ′(−0.7) = 0.221709; ratio = 0.886835

**A:** σ′=0.3318·0.6682=0.2217; ratio=0.2217/0.25=0.887.

**Q54 [Think]** Which statement about universal approximation is correct?

**Correct:** A nonlinear MLP with enough hidden units can represent a continuous function on a compact domain, but nothing guarantees gradient descent finds it or that it generalises  
*Distractors:* It guarantees backprop reaches the global minimum / It also holds for purely linear hidden units / It guarantees zero test error

**A:** Representation ≠ learning ≠ generalisation.

**Q55 [Calc]** (Slide Example 3) x = (2, −1), h₁ = ReLU(x₁+x₂−0.5) = 0.5, h₂ inactive, o = 2h₁ − h₂ + 0.1 = 1.1, t = 1. With η = 0.1 the loss rose from 0.00500 to 0.01154. Redo the single update with η = 0.01 (δ_o = −0.1, δ₁ = −0.2). Give the new h₁ and the new loss.

**Answers:** new h₁ = 0.488; new loss = 0.00279423

**A:** w=(0.996, 1.002), b₁=−0.502 → h₁=0.488; output weight 1.9995, b_o=0.099 → o=1.0748, E=0.00279 < 0.005. A smaller η decreases the loss.

---

## Part 3 – Regularization (Q56–75)

**Q56 [Think]** Target error ≈ 2.5%. Four models give (train, validation) errors: M1 (10%, 11%), M2 (2%, 9%), M3 (3%, 4%), M4 (1%, 12%). Which statement is correct?

**Correct:** M3 is closest to the target; M1 is high bias; M2 and M4 are high variance  
*Distractors:* M4 is best because it has the lowest training error / M1 is high variance because its gap is small / M2 is high bias because validation error is high

**A:** Check training error first, then the gap. M1: training error far above target with tiny gap → bias. M2 (gap 7%), M4 (gap 11%) → variance. M3 is near target with a small gap.

**Q57 [Calc]** For models M1 (10%, 11%), M2 (2%, 9%), M3 (3%, 4%), M4 (1%, 12%) as (train, validation) errors, compute the generalisation gap (validation − training) for M2 and M4.

**Answers:** gap M2 = 7; gap M4 = 11

**A:** M2: 9−2=7 points. M4: 12−1=11 points. The gap measures the degree of overfitting.

**Q58 [Calc]** Validation losses for epochs 1–8: 0.90, 0.60, 0.45, 0.40, 0.38, 0.41, 0.47, 0.55. Training losses: 0.80, 0.50, 0.30, 0.20, 0.12, 0.07, 0.04, 0.02. (a) Which epoch should early stopping pick? (b) What is the validation−training gap at epoch 8?

**Answers:** (a) best epoch = 5; (b) gap at epoch 8 = 0.53

**A:** Validation loss is minimal at epoch 5 (0.38). Gap at epoch 8 = 0.55 − 0.02 = 0.53, much larger than the 0.26 at epoch 5: overfitting.

**Q59 [Calc]** A hidden layer has 10 units and dropout drops each unit independently (each unit is either kept or dropped). How many distinct dropout masks (thinned sub-networks) are possible for that layer in one training step?

**Answers:** masks = 1024

**A:** Each of the 10 units is kept or dropped → 2¹⁰ = 1024 masks, which is why dropout acts like training many thinned networks that share weights.

**Q60 [Calc]** Augmentation: 1000 images, each can be left or horizontally flipped (2 options), rotated by one of 4 angles (including 0°) and cropped in one of 3 ways. How many distinct augmented versions can be generated per image and in total?

**Answers:** per image = 24; total = 24000

**A:** 2·4·3 = 24 per image; 24 000 in total — only worthwhile if the augmentations are label-preserving and realistic.

**Q61 [Think]** Which learning-curve description matches a learning rate that is too HIGH?

**Correct:** Loss drops quickly at first, then plateaus at a high value or oscillates / explodes  
*Distractors:* Loss decreases very slowly in an almost straight line / Loss decreases smoothly to a low value / Loss is constant from the first iteration because the data are too small

**A:** Too high: overshoot, noisy or diverging loss, or a stuck high plateau. Too low: slow, nearly linear decrease.

**Q62 [Think]** You tune 20 hyperparameter settings, choosing the one with the best validation accuracy, and report that validation accuracy as your expected performance. What is wrong?

**Correct:** The choice was optimised on the validation set, so the reported number is optimistically biased; use a untouched test set or cross-validation  
*Distractors:* Nothing; validation accuracy is an unbiased estimate after selection / You should have selected on training accuracy / Using 20 settings makes the model underfit

**A:** Repeated choices overfit the validation set. Report on a held-out test set that was not used for tuning.

**Q63 [Calc]** A dataset of 10,000 labelled examples is split 70/15/15 into train/validation/test. (a) How many test examples? (b) With 5-fold cross-validation on the 10,000 examples, how many examples train each fold's model?

**Answers:** (a) test examples = 1500; (b) training per fold = 8000

**A:** 15% of 10,000 = 1500. 5-fold CV trains on 4/5 of the data: 8000 per fold (2000 validate).

**Q64 [Think]** Which statement about expected error is correct?

**Correct:** Expected error is defined over the true data distribution; validation error is a practical estimate of it  
*Distractors:* Training error is an unbiased estimate of expected error / Expected error can be computed exactly from the training set / Test labels are always available at deployment time

**A:** We do not know p(x,y) and do not have labels for future data, so a held-out validation set is used.

**Q65 [Think]** In the appendix, what happens to the two components of excess risk when you (i) double the training set size, and (ii) switch to a richer model family ℋ?

**Correct:** (i) estimation error falls; (ii) approximation error falls but estimation error tends to rise  
*Distractors:* (i) approximation error falls; (ii) estimation error falls / (i) both rise; (ii) both fall / (i) nothing changes; (ii) approximation error rises

**A:** More data reduces estimation error. A bigger ℋ can represent the target better (lower approximation error) but is harder to estimate from finite data.

**Q66 [Calc]** Excess risk = approximation + estimation error. Currently approximation = 0.03 and estimation = 0.05. (a) Doubling the data cuts estimation error to 0.035: new excess risk? (b) Instead moving to a richer model gives approximation 0.01 but estimation 0.07: new excess risk?

**Answers:** (a) more data = 0.065; (b) richer model = 0.08

**A:** (a) 0.03+0.035=0.065. (b) 0.01+0.07=0.08. The richer model lowers one term but raises the other; here it is no better than the original 0.08.

**Q67 [Calc]** Weight sharing: map a 28×28 grayscale image to a 28×28 output map. (a) Parameters of a fully connected layer (784 inputs → 784 outputs, with biases). (b) Parameters of one 3×3 convolution filter with a bias.

**Answers:** (a) FC parameters = 615440; (b) conv parameters = 10

**A:** (a) 784·784+784 = 615,440. (b) 3·3+1 = 10. Sharing the kernel across positions slashes the parameter count and hence variance.

**Q68 [Calc]** Dimensionality reduction as regularization: the first layer maps the input to 128 hidden units (with biases). How many first-layer parameters if the input is (a) 784 raw pixels, (b) reduced to 50 components?

**Answers:** (a) 784 inputs = 100480; (b) 50 inputs = 6528

**A:** (a) 785·128=100,480. (b) 51·128=6,528: far fewer parameters to fit from the same data.

**Q69 [Think]** Classify each: weight decay, early stopping, dropout, weight sharing (G1) versus more real data, augmentation (G2). Which description is best?

**Correct:** G1 constrains the effective complexity of the model; G2 increases the size or variety of the training data  
*Distractors:* G1 increases the training data; G2 constrains the model / Both groups only reduce the learning rate / G1 reduces bias; G2 increases bias

**A:** Lecture grouping: limit model expressiveness vs. increase training data complexity/size to reduce variance.

**Q70 [Think]** Training error 15%, validation error 16%, target 2%. Which action is LEAST likely to help?

**Correct:** Add strong dropout and a large weight-decay penalty  
*Distractors:* Use a larger network / Train for more epochs / Use better features

**A:** This is high bias (underfitting). More regularization restricts the model further and makes it worse.

**Q71 [Think]** After adding data augmentation your training error rises from 1% to 4% while validation error falls from 12% to 7%. How should you interpret this?

**Correct:** Regularization worked: variance fell at the cost of a higher training error  
*Distractors:* Augmentation failed because training error increased / The model is now underfitting because validation error fell / Training error must never exceed validation error

**A:** Regularization can raise training error while improving generalisation; the gap shrank from 11 to 3 points.

**Q72 [Think]** λ sweep for weight decay as (train, validation) error: λ=0: (1%, 12%); λ=0.01: (3%, 6%); λ=0.1: (5%, 7%); λ=1: (14%, 15%). Best λ and the diagnosis of λ=1?

**Correct:** λ = 0.01; λ = 1 underfits (high bias)  
*Distractors:* λ = 0; λ = 1 overfits / λ = 1, because its gap is smallest / λ = 0.1, because training error is closest to validation error

**A:** Choose by validation error (6% lowest). λ=1 has a small gap but high training error → underfitting.

**Q73 [Think]** Which pair correctly matches symptom to diagnosis (target error 2%)?

**Correct:** Train 0.5%, validation 9% → high variance; train 9%, validation 10% → high bias  
*Distractors:* Train 0.5%, validation 9% → high bias; train 9%, validation 10% → high variance / Train 3%, validation 3.5% → high variance / Train 20%, validation 40% → close to target

**A:** Large gap = variance; large training error with a small gap = bias.

**Q74 [Think]** Which statement about the bias–variance trade-off is correct?

**Correct:** Complex models tend to have low bias and high variance; simple models tend to have high bias and low variance  
*Distractors:* Complex models have high bias and low variance / Variance is the error from wrong assumptions / Bias increases with model complexity

**A:** Direct recall from the lecture: bias is high when the model is too simple; variance increases with complexity.

**Q75 [Think]** Why must the test set stay untouched until the end, while the validation set may be used repeatedly?

**Correct:** Validation guides choices and so becomes slightly 'used up'; the test set gives the final unbiased estimate only if no choice was made on it  
*Distractors:* The test set is larger so it cannot be reused / The validation set has no labels / The test set is used to fit parameters

**A:** Selection on the validation set biases its error; the test set must reflect truly unseen data.

---

## Part 4 – CNNs (Q76–105)

**Q76 [Calc]** X = [[1,0,2,1],[3,1,0,2],[0,2,1,1],[1,1,0,3]] is convolved (cross-correlation) with the 3×3 kernel whose every row is [1, 0, −1], stride 1, zero padding 1, bias 0. Give the output size (one side) and the outputs at (a) row 1, col 1 and (b) row 2, col 3.

**Answers:** output side = 4; (a) Z₁,₁ = -1; (b) Z₂,₃ = -1

**A:** Size=(4+2−3)/1+1=4. Z = [[-1, 2, -2, 2], [-3, 1, -1, 3], [-4, 3, -2, 1], [-3, 0, -1, 1]] → Z₁,₁=-1, Z₂,₃=-1. Padded zeros enter the sum at the border.

**Q77 [Calc]** 6×6 input X with rows [1,2,0,1,3,1], [0,1,1,2,0,2], [2,0,1,0,1,1], [1,1,0,2,2,0], [0,3,1,1,0,1], [2,0,2,1,1,3]. A 3×3 all-ones kernel is applied with stride 2 and zero padding 1. Give the output side length and the output values (a) at row 1, col 1 and (b) at row 2, col 2.

**Answers:** output side = 3; (a) out₁,₁ = 4; (b) out₂,₂ = 8

**A:** ⌊(6+2−3)/2⌋+1=3. Output = [[4, 7, 9], [5, 8, 10], [7, 11, 11]]. out₁,₁ sums the top-left 2×2 block (1+2+0+1=4).

**Q78 [Calc]** Two-channel 3×3 input: channel 1 = [[1,2,0],[0,1,1],[2,0,1]], channel 2 = [[0,1,1],[1,0,2],[1,1,0]]. One 2×2×2 filter: channel-1 weights [[1,0],[0,1]], channel-2 weights [[1,1],[0,−1]], bias 0.5, stride 1, no padding. Give the output shape (one side) and the outputs at (a) top-left and (b) bottom-right.

**Answers:** output side = 2; (a) top-left = 3.5; (b) bottom-right = 4.5

**A:** Sum channel-wise dot products then add bias. Output = [[3.5, 3.5], [0.5, 4.5]]. A single full-depth filter gives ONE output channel.

**Q79 [Calc]** Input 32×32×3 → Conv(3×3, K=16, S=1, P=1) → MaxPool(2×2, S=2) → Conv(3×3, K=32, S=1, P=1) → MaxPool(2×2, S=2) → flatten → FC(10). Give the flattened length and the total number of trainable parameters.

**Answers:** flattened length = 2048; total parameters = 25578

**A:** Shapes: 32×32×16 → 16×16×16 → 16×16×32 → 8×8×32 → 2048. Conv1: 16(27+1)=448; Conv2: 32(144+1)=4,640; FC: 2048·10+10=20,490. Total 25,578.

**Q80 [Calc]** A ResNet-style block maps 56×56×64 to 28×28×128: Conv1 (3×3, K=128, S=2, P=1) → Conv2 (3×3, K=128, S=1, P=1), with a shortcut projection 1×1, K=128, S=2, P=0 so that y = F(x) + shortcut(x). Give the output height of Conv1, and the parameter counts of Conv1 and of the projection.

**Answers:** Conv1 output height = 28; Conv1 parameters = 73856; projection parameters = 8320

**A:** ⌊(56+2−3)/2⌋+1=28. Conv1: 128(576+1)=73,856. Projection: 128(64+1)=8,320. Conv2 has 147,584. The projection makes both branches 28×28×128 so they can be added.

**Q81 [Calc]** Inception module on a 28×28×192 input. Branches: (1) 1×1, K=64; (2) 1×1, K=96 then 3×3, K=128; (3) 1×1, K=16 then 5×5, K=32; (4) 3×3 max-pool (S=1, P=1) then 1×1, K=32. Outputs are concatenated. Give (a) the output channels, (b) parameters of branch 3 (reduce + 5×5), and (c) parameters if branch 3 were a direct 5×5, K=32 on the 192 channels.

**Answers:** (a) output channels = 256; (b) branch 3 parameters = 15920; (c) direct 5×5 = 153632

**A:** (a) 64+128+32+32=256 (same H×W, concatenated along channels). (b) 16(193)+32(401)=15,920. (c) 32(192·25+1)=153,632. The 1×1 reduction cuts this branch roughly 10×.

**Q82 [Calc]** With C = 64 input and output channels (ignore biases), compare parameters of (a) two stacked 3×3 convolutions and (b) one 5×5 convolution. Both cover a 5×5 neighbourhood of the input at stride 1.

**Answers:** (a) two 3×3 = 73728; (b) one 5×5 = 102400

**A:** (a) 2·9·64·64 = 73,728. (b) 25·64·64 = 102,400. Repeated small kernels (VGG) use fewer parameters and add an extra nonlinearity.

**Q83 [Calc]** 64×64×3 input → Conv(3×3, K=32, P=1) → Conv(3×3, K=32, P=1) → MaxPool(2×2, S=2) → Conv(3×3, K=64, P=1) → MaxPool(2×2, S=2). (a) Final output height? (b) Total conv parameters? (S=1 for all convs.)

**Answers:** (a) final height = 16; (b) total conv parameters = 28640

**A:** 64→64→64→32→32→16, so 16×16×64. Params: 896 + 9248 + 18496 = 28,640.

**Q84 [Calc]** The previous network ends with a 16×16×64 feature map. It is flattened and connected to a 100-unit fully connected layer (with biases). How many parameters does this one layer have, and how many times more is that than the conv parameters of the whole network (28,640)? (to 1 decimal)

**Answers:** FC parameters = 1.6385e+06; ratio vs conv = 57.2102

**A:** 16·16·64=16,384 inputs → 1,638,500 parameters ≈ 57.2× all the conv layers together. Fully connected layers dominate parameters.

**Q85 [Calc]** Output sizes (floor where needed): (a) 17×17 input, 3×3 filter, S=2, P=0. (b) 17×17, 3×3, S=2, P=1. (c) 18×18, 4×4, S=4, P=0.

**Answers:** (a) output side = 8; (b) output side = 9; (c) output side = 4

**A:** (a) ⌊14/2⌋+1=8. (b) ⌊16/2⌋+1=9. (c) ⌊14/4⌋+1=4 (the last two columns are never used).

**Q86 [Calc]** With stride 1, what padding preserves height and width for (a) a 5×5 kernel, (b) a 7×7 kernel? (c) A 32-pixel-wide input is passed through a 7×7 kernel with P=2: output width?

**Answers:** (a) P for 5×5 = 2; (b) P for 7×7 = 3; (c) output width = 30

**A:** P=(F−1)/2 → 2 and 3. (c) (32+4−7)/1+1=30.

**Q87 [Calc]** 6×6 input [[1,4,2,0,3,1],[5,2,1,6,0,2],[3,1,7,2,2,4],[0,2,1,3,5,1],[2,8,0,1,1,2],[4,1,3,0,6,0]]. Apply 3×3 pooling with stride 3. Give (a) the max-pool value of the top-left window and (b) the average-pool value of the bottom-right window (to 2 decimals), and (c) the output side length.

**Answers:** (a) max, top-left = 7; (b) avg, bottom-right = 2.11111; (c) output side = 2

**A:** Top-left window max=7. Bottom-right window values 3,5,1,1,1,2,0,6,0 → mean 2.11. Output ⌊(6−3)/3⌋+1=2.

**Q88 [Calc]** Patch [[1,2,1],[2,6,2],[1,2,1]]. Compute (a) the response of the sharpen kernel [[0,−1,0],[−1,5,−1],[0,−1,0]] and (b) the 3×3 blur (all weights 1/9), to 3 decimals.

**Answers:** (a) sharpen = 22; (b) blur = 2

**A:** (a) 5·6 − (2+2+2+2) = 22. (b) the nine values sum to 18, and 18/9 = 2.000.

**Q89 [Calc]** CNN libraries compute cross-correlation. For the patch [[0,0,3],[0,0,3],[0,0,3]] and kernel [[−1,0,1]] in every row, the cross-correlation response is 9. What would TRUE convolution (kernel flipped horizontally and vertically) give?

**Answers:** flipped-kernel response = -9

**A:** Flipping turns [−1,0,1] into [1,0,−1]: response = 3(0−3) = −9. For learned kernels the flip is irrelevant: the network just learns the flipped values.

**Q90 [Calc]** A 1×1 convolution with two filters is applied to a pixel whose 3 channel values are (1, 2, 3). Filter 1 weights (1, 0, −1), bias 0; filter 2 weights (0.5, 0.5, 0.5), bias 1. Give the two output values.

**Answers:** output 1 = -2; output 2 = 4

**A:** Filter 1: 1−3+0=−2. Filter 2: 0.5(1+2+3)+1=4. A 1×1 conv is a per-pixel fully connected layer across channels.

**Q91 [Calc]** Count multiply-accumulates (one per weight use, ignoring biases) for the layer 32×32×3 input, six 5×5 filters, S=1, P=0, and compare its parameter count (456) with the fully connected alternative from 120,000 inputs to a layer of the same 28·28·6 = 4,704 outputs (ignore biases).

**Answers:** conv multiply-accumulates = 352800; FC weights = 5.6448e+08

**A:** Conv: 28·28·6·(5·5·3)=352,800 MACs but only 450 weights (+6 biases). Dense mapping: 120,000·4,704 = 564,480,000 weights. Parameter count is independent of input size; compute is not.

**Q92 [Calc]** LeNet-style: 32×32×1 → Conv(5×5, K=6, S=1, P=0) → MaxPool(2×2, S=2) → Conv(5×5, K=16, S=1, P=0) → MaxPool(2×2, S=2) → flatten. Give the flattened length and the parameters of the second convolution.

**Answers:** flattened length = 400; conv2 parameters = 2416

**A:** 32→28→14→10→5, so 5·5·16=400. Conv2: 16(150+1)=2,416.

**Q93 [Think]** The ImageNet benchmark reports top-1 and top-5 error. Which relationship must hold for a given model?

**Correct:** Top-5 error ≤ top-1 error, because a correct top-1 prediction is always inside the top 5  
*Distractors:* Top-5 error ≥ top-1 error / They are always equal / Top-5 error is the average of five top-1 errors

**A:** Top-5 counts a miss only when the true class is absent from the five best guesses.

**Q94 [Think]** A residual block receives x of shape 56×56×64 and F(x) has shape 28×28×128. What is needed for y = F(x) + x?

**Correct:** A projection of x (e.g. 1×1 conv with stride 2 and 128 filters) so its shape matches F(x)  
*Distractors:* Nothing; addition broadcasts any shapes / Concatenate them along the width / Drop the last 64 channels of F(x)

**A:** Addition requires matching shapes; a projection aligns channels and spatial size.

**Q95 [Think]** What property of shared convolution weights lets a CNN detect a pattern wherever it appears in the image?

**Correct:** The same kernel is applied at every spatial position, so the same pattern gives the same response at each location  
*Distractors:* Each position has its own specialised kernel / Pooling learns new kernels for each location / Padding adds copies of the pattern

**A:** Weight sharing yields translation-equivariant feature maps, and pooling adds tolerance to small shifts.

**Q96 [Calc]** The slide convolution output is Z = [[2,0,−2],[3,2,−2],[3,1,−2]]. Apply ReLU, then 2×2 max-pooling with stride 1 (no padding). Give the output side length and the outputs at (a) row 1, col 1 and (b) row 2, col 2.

**Answers:** output side = 2; (a) out₁,₁ = 3; (b) out₂,₂ = 2

**A:** ReLU → [[2,0,0],[3,2,0],[3,1,0]]. Pooling windows give [[3, 2], [3, 2]]; side ⌊(3−2)/1⌋+1=2.

**Q97 [Calc]** Compare an RNN with input size 3 and hidden size 4 (weights W_xh, W_hh and a hidden bias) unrolled over 10 time steps (a) with shared parameters, and (b) if every time step had its own separate parameters.

**Answers:** (a) shared = 32; (b) separate per step = 320

**A:** One step has 12+16+4=32 parameters. Shared: 32 regardless of sequence length. Separate: 320.

**Q98 [Calc]** An autoencoder maps 784 inputs → 32-unit latent code → 784 outputs (one bias per unit, no other layers). How many parameters, and what is the compression ratio 784/32?

**Answers:** parameters = 50992; compression ratio = 24.5

**A:** Encoder: 785·32=25,120. Decoder: 33·784=25,872. Total 50,992. Ratio 24.5.

**Q99 [Think]** Which statement about GAN training is correct?

**Correct:** The discriminator is trained to separate real from generated samples while the generator is trained to make samples the discriminator cannot separate; the two are updated alternately  
*Distractors:* Only the generator is ever updated / The discriminator generates the images and the generator classifies them / Both networks minimise exactly the same loss in the same direction

**A:** Adversarial, alternating training.

**Q100 [Think]** What distinguishes a variational autoencoder from a plain autoencoder?

**Correct:** The encoder produces a distribution over latent variables from which a vector is sampled and decoded, enabling generation  
*Distractors:* It has no decoder / It uses adversarial discriminators / It reconstructs inputs without any latent representation

**A:** A VAE encoder describes a distribution; sampling and decoding generates new data.

**Q101 [Think]** Which statement about Transformers is correct?

**Correct:** Queries and keys determine attention weights over values, and positional information is added to supply order  
*Distractors:* Order is learned automatically by convolution / Keys are the outputs of the decoder / Attention is only possible between neighbouring tokens

**A:** Attention mixes information across all positions; position information is explicit.

**Q102 [Think]** Which statement correctly describes diffusion models?

**Correct:** Training adds noise and learns to reverse the corruption; generation starts from noise and repeatedly denoises  
*Distractors:* Generation starts from a real image and adds noise / A discriminator decides when to stop / They map text to images using convolutions only

**A:** Forward noising defines the training task; sampling is iterative denoising.

**Q103 [Think]** Which of these needs PAIRED training examples and which does not?

**Correct:** Denoising autoencoder needs (noisy, clean) pairs; CycleGAN does not need pairs  
*Distractors:* CycleGAN needs pairs; denoising autoencoder does not / Both need pairs / Neither needs pairs

**A:** Denoising AE: noisy input with clean target. CycleGAN: unpaired image translation using cycle consistency.

**Q104 [Calc]** Input 14×14×16, a convolution with 32 filters of 3×3, S=1, P=1. Give the output height, the output channels and the number of parameters.

**Answers:** output height = 14; output channels = 32; parameters = 4640

**A:** (14+2−3)/1+1=14; channels = K = 32; params 32(144+1)=4,640.

**Q105 [Think]** A 2×2 max-pool with stride 2 is applied to an 8×8×16 feature map. Which statement is correct?

**Correct:** Output is 4×4×16 and the layer has no learned parameters  
*Distractors:* Output is 4×4×8 and the layer has 16 learned weights / Output is 4×4×16 and the layer has 16·4 learned weights / Output is 8×8×4 and the layer has no learned parameters

**A:** Pooling acts on each channel separately (channels unchanged) and halves H and W.

---

## Part 5 – Clustering (Q106–130)

**Q106 [Calc]** Points (0,0), (0,2), (4,0), (4,2) are clustered by K-Means with K = 2. Run A starts with centres (0,0) and (0,2); Run B starts with centres (0,0) and (4,0). Give the final sum of squared distances to the centres for (a) Run A and (b) Run B.

**Answers:** (a) Run A = 16; (b) Run B = 4

**A:** Run A converges to bottom/top rows (centres (2,0),(2,2)), cost 16. Run B converges to left/right columns (centres (0,1),(4,1)), cost 4. Same data, different initialisations → different local minima.

**Q107 [Think]** Which conclusion follows from the two runs in the previous question?

**Correct:** K-Means can converge to different local minima, so several restarts with the lowest cost should be kept  
*Distractors:* K-Means always converges to the global optimum / Run A is wrong because it did not converge / The metric must be Manhattan, not Euclidean

**A:** Both runs satisfy the stopping criterion (no assignment changes), but only one is the better solution.

**Q108 [Calc]** For x = (1,5,2) and y = (4,1,2), compute (a) L1 distance, (b) L2 distance, (c) L∞ distance.

**Answers:** (a) L1 = 7; (b) L2 = 5; (c) L∞ = 4

**A:** (a) 3+4+0=7. (b) √(9+16)=5. (c) max(3,4,0)=4. In general L∞ ≤ L2 ≤ L1.

**Q109 [Calc]** Distances from the origin: point a = (3,0) and point b = (2,2). Give (a) L1(x,b), (b) L2(x,b) (2 decimals). Which point is the nearest neighbour under L1 (enter 1 for a, 2 for b) (c)?

**Answers:** (a) L1(x,b) = 4; (b) L2(x,b) = 2.82843; (c) nearest under L1 (1=a, 2=b) = 1

**A:** L1: a=3, b=4 → a is nearer. L2: a=3, b=2.83 → b is nearer. The choice of metric changes which points count as similar, and hence the clusters.

**Q110 [Calc]** Points P1(0,0), P2(1,0), P3(5,0), P4(6,1), P5(10,0). Run HAC with Euclidean distance. Give the height of the 3rd merge under (a) single linkage, (b) complete linkage, (c) average linkage (3 decimals).

**Answers:** (a) single = 4; (b) complete = 5; (c) average = 4.562

**A:** First two merges are the same (1.0 for P1–P2, 1.414 for P3–P4). The 3rd merge is 4.0 (single: {P1,P2}∪{P3,P4}), 5.0 (complete: P5 joins {P3,P4}), 4.562 (average: P5 joins {P3,P4}).

**Q111 [Think]** For the same five points, which pair of K = 2 clusterings is correct?

**Correct:** Single linkage: {P1,P2,P3,P4},{P5}; complete linkage: {P1,P2},{P3,P4,P5}  
*Distractors:* Single linkage: {P1,P2},{P3,P4,P5}; complete linkage: {P1,P2,P3,P4},{P5} / Both give {P1,P2},{P3,P4,P5} / Both give {P1,P2,P3,P4},{P5}

**A:** Single: after P1–P2 and P3–P4, the closest pair of clusters is P1–P2 and P3–P4 (4.0) before P5 (4.12). Complete: P5 to {P3,P4} (max 5) is smaller than {P1,P2} to {P3,P4} (max 7.07). The choice of linkage changes the dendrogram.

**Q112 [Think]** A dendrogram for 7 points has merge heights 0.3, 0.5, 0.9, 1.4, 2.2, 4.0. If you cut at height 1.0, how many clusters remain, and at height 2.5?

**Correct:** 4 clusters at 1.0 and 2 clusters at 2.5  
*Distractors:* 3 clusters at 1.0 and 2 clusters at 2.5 / 4 clusters at 1.0 and 3 clusters at 2.5 / 7 clusters at 1.0 and 1 cluster at 2.5

**A:** Merges below the cut have happened: 3 merges below 1.0 → 7−3=4 clusters; 5 merges below 2.5 → 7−5=2.

**Q113 [Calc]** HAC on N = 1000 points. (a) How many pairwise distances are stored in the initial distance table? (b) How many merges until one cluster is left?

**Answers:** (a) pairwise distances = 499500; (b) merges = 999

**A:** N(N−1)/2 = 499,500 distances; N−1 = 999 merges. This memory/compute cost is why HAC is heavier than K-Means.

**Q114 [Calc]** A K-Means iteration computes the distance from every point to every centre. For N = 10⁶ points and K = 10 clusters, how many point–centre distance evaluations are needed per assignment step?

**Answers:** distance evaluations = 1e+07; evaluations if K = 100 = 1e+08

**A:** N·K = 10⁷ per assignment step (10⁸ for K=100). Each iteration is cheap compared with building an N×N HAC table.

**Q115 [Think]** Which statement about the K-Means cost Σ‖x−μ‖² is correct?

**Correct:** Neither the assignment step nor the update step can increase it, so it converges, but possibly to a local minimum  
*Distractors:* The assignment step can increase it but the update step cannot / It is guaranteed to reach the global minimum / It increases as K increases

**A:** Assigning to the nearest centre and recomputing means each reduce or keep the cost.

**Q116 [Calc]** Eight grey-level pixels: 10, 12, 11, 200, 205, 198, 100, 102. K-Means with K = 3 and initial centres 10, 100, 200. Give the final centres (rounded to 1 decimal) in increasing order, and the final sum of squared distances.

**Answers:** lowest centre = 11; middle centre = 101; highest centre = 201; sum of squared distances = 30

**A:** Assignments are stable after the first update: centres 11.0, 101.0, 201.0; cost 30.0. Replacing each pixel by its centre gives a 3-level segmentation.

**Q117 [Calc]** 1-D data 1, 2, 3, 10, 11, 12. Give the K-Means cost (sum of squared distances to the cluster means) for (a) K = 1, (b) K = 2 (best clustering), (c) K = 6.

**Answers:** (a) K = 1 = 125.5; (b) K = 2 = 4; (c) K = 6 = 0

**A:** (a) mean 6.5 → 125.5. (b) clusters {1,2,3},{10,11,12}: 2+2=4. (c) each point is its own centre → 0. The cost always falls as K rises, so it cannot choose K by itself.

**Q118 [Think]** Which statement about nesting is correct?

**Correct:** HAC results for different K are nested (cutting lower gives finer clusters that merge into the K-cluster result); K-Means runs for different K need not be nested  
*Distractors:* K-Means results are always nested; HAC results are not / Both are always nested / Neither is ever nested

**A:** A dendrogram records successive merges, so any cut refines the next coarser cut.

**Q119 [Calc]** Cluster P has 3 points with centroid (1,1). Cluster Q has 1 point at (5,5). Under centroid linkage, give the centroid of the merged cluster (x, y), and the naïve (WRONG) average of the two centroids (x).

**Answers:** merged centroid x = 2; merged centroid y = 2; naïve average x = 3

**A:** Weighted by size: (3·1+1·5)/4=2 in each coordinate. The naïve (1+5)/2=3 ignores cluster sizes, the mistake flagged in the lecture's A1/A4/A6 example.

**Q120 [Calc]** After merging A4(5,8) and A6(4,9), the centroid is (4.5, 8.5). Compute the Euclidean distance from this centroid to (a) A2(2,5) and (b) A5(1,2). (Note: the lecture's recap table lists 10.81 for the (A4,A6)–A5 entry, which does not match the coordinates.)

**Answers:** (a) to A2 = 4.30116; (b) to A5 = 7.38241

**A:** (a) √(6.25+12.25)=4.30. (b) √(12.25+42.25)=7.38, not 10.81. The earlier A4–A5 entry of 3.61 also looks inconsistent with (5,8),(1,2).

**Q121 [Calc]** (Slide data) d(#1,#4)=1.17, d(#2,#4)=1.30, d(#3,#4)=0.50. Under AVERAGE linkage, what is the distance between cluster {#1,#2,#3} and {#4}? Compute it (a) directly from all pairs, and (b) from the updated entry D({#1,#2},#4)=1.235 and d(#3,#4) with size weights 2 and 1.

**Answers:** (a) direct = 0.99; (b) size-weighted update = 0.99

**A:** (a) (1.17+1.30+0.50)/3=0.99. (b) (2·1.235+1·0.5)/3=0.99. Equal weights 0.5/0.5 are only right when the merged clusters have equal size.

**Q122 [Calc]** Single-linkage heights for the slide's 4 points are 0.14, 0.50, 0.72. Cut the dendrogram at height (a) 0.3 and (b) 0.6. How many clusters remain in each case?

**Answers:** (a) clusters at 0.3 = 3; (b) clusters at 0.6 = 2

**A:** (a) only the 0.14 merge is below 0.3 → 4−1=3 clusters. (b) merges 0.14 and 0.50 → 4−2=2 clusters: {#1,#2},{#3,#4}.

**Q123 [Think]** Which task is HAC better suited to than K-Means?

**Correct:** You want to see clusterings for many values of K from one run, with no random initialisation  
*Distractors:* Clustering ten million points cheaply / Choosing the initial centroids / Making the number of clusters unnecessary because it never needs a metric

**A:** HAC is deterministic and gives the whole hierarchy, but is memory- and computation-heavy.

**Q124 [Think]** A dataset has two concentric rings. Why does K-Means with Euclidean distance (K = 2) fail?

**Correct:** The rings are not linearly separable and the mean-based Euclidean assignment splits space with straight boundaries; a better metric or feature space is needed  
*Distractors:* K-Means cannot use K = 2 / The rings have different colours / The initial centres were Manhattan

**A:** Centroid-based clustering with Euclidean distance yields convex, linearly separated regions.

**Q125 [Think]** Which statement distinguishes classification from clustering correctly?

**Correct:** Classification learns from provided labels to predict labels of unseen data; clustering uses no labels and outputs a membership for each point  
*Distractors:* Clustering needs labels at training time; classification does not / Classification outputs only memberships without labels / They are the same problem under different names

**A:** Supervised vs unsupervised.

**Q126 [Think]** Which statement about initialisation and determinism is correct?

**Correct:** K-Means can give different results for different random starts; HAC gives the same result every run  
*Distractors:* K-Means is deterministic; HAC is random / Both are random / Both are deterministic regardless of initialisation

**A:** K-Means depends on the initial centres; HAC starts from singleton clusters.

**Q127 [Calc]** (Slide data) #1(1.9,1.0), #2(1.8,0.9), #3(2.3,1.6), #4(2.3,2.1). With CENTROID linkage after merging {#1,#2} and {#3,#4}, what is the distance between the two cluster centroids (3 decimals)? And what is the exact AVERAGE-linkage distance between the same two clusters, from d13=0.72, d14=1.17, d23=0.86, d24=1.30?

**Answers:** centroid distance = 1.00623; average-linkage distance = 1.0125

**A:** Centroids (1.85,0.95) and (2.3,1.85): √(0.2025+0.81)=1.006. Different linkages give slightly different heights.

**Q128 [Think]** K-Means assigns each pixel's RGB colour to one of K centres. Which statement is correct?

**Correct:** Each pixel is a 3-D point; the final image uses only K colours (the centres)  
*Distractors:* Each pixel is a 1-D point regardless of colour / The output keeps all original colours / K must equal 3 for RGB

**A:** The lecture uses 1-D grey values and 3-D RGB values as features.

**Q129 [Calc]** Points (1,1), (1,3), (5,1), (5,3), (9,2). K-Means with K = 2 and initial centres (1,1) and (5,1). Give the final centre of the cluster that contains (1,1) (x and y), and the final sum of squared distances.

**Answers:** centre x = 1; centre y = 2; sum of squared distances = 14.6667

**A:** Run to convergence: clusters {(1,1),(1,3)} and {(5,1),(5,3),(9,2)}; centres [[1.0, 2.0], [6.333333333333333, 2.0]]; cost 14.67. Check by repeating assignment and update until no change.

**Q130 [Think]** Which is a true limitation pair?

**Correct:** K-Means needs K and is sensitive to initialisation; HAC is memory- and computation-intensive  
*Distractors:* K-Means is memory-intensive; HAC is sensitive to initialisation / Both require K up front / HAC needs a random start; K-Means is deterministic

**A:** Lecture summary of K-Means vs HAC.

---

## Part 6 – Cross-topic Mix (Q131–150)

**Q131 [Think]** A CNN with 25,578 parameters is trained on 500 images: training error 0.5%, validation error 14%, target 3%. Which combination is most appropriate?

**Correct:** Diagnose high variance; use augmentation, dropout/weight decay or early stopping, and more data  
*Distractors:* Diagnose high bias; double the number of filters / Diagnose high bias; remove all regularization / Diagnose good fit; stop

**A:** Large gap with tiny training error → overfitting; reduce effective complexity or increase effective data.

**Q132 [Think]** Which statement best contrasts a standard deterministic network with Bayesian/Naïve Bayes inference?

**Correct:** A trained network gives the same output for the same input; Bayesian inference gives probabilities of hypotheses given data and priors  
*Distractors:* A trained network outputs random labels at inference / Naïve Bayes needs backpropagation to learn / Bayesian inference never uses prior knowledge

**A:** Lecture: deterministic inference vs probabilistic learning.

**Q133 [Calc]** 1-D convolution with shared kernel (w₁, w₂) = (1, 0), no bias, linear output: z₁ = w₁x₁ + w₂x₂, z₂ = w₁x₂ + w₂x₃ with x = (1, 2, 3). Targets t = (2, 1), E = ½Σ(t−z)², η = 0.1. Give the outputs z₁, z₂ and the updated w₁ and w₂. (The shared weight receives the SUM of the contributions from both positions.)

**Answers:** z₁ = 1; z₂ = 2; new w₁ = 0.9; new w₂ = -0.1

**A:** z=(1,2), δ=(1,-1). Σδ·input: w₁: 1·1+(−1)·2=-1; w₂: 1·2+(−1)·3=-1. w₁=0.9, w₂=-0.1.

**Q134 [Think]** Which statement about data augmentation is correct?

**Correct:** It lowers variance by exposing the model to more variation, but unrealistic or label-changing transforms can hurt  
*Distractors:* It always lowers training error / It only helps underfitting models / It removes the need for a validation set

**A:** Augmentation encodes appropriate invariances; it must be label-preserving and avoid obvious artefacts.

**Q135 [Think]** K-Means on raw pixel vectors groups images poorly, but works well on certain learned feature vectors. What does this illustrate?

**Correct:** Clustering depends on a metric/feature space in which similar items are close; a poor space makes K-Means fail  
*Distractors:* K-Means requires labels / K-Means works better with larger K only / Euclidean distance is invalid for images

**A:** Lecture: need a better metric/space when data are not linearly separable.

**Q136 [Think]** Why can reducing the input dimension (e.g. from 784 to 50 components) help both a neural network and K-Means?

**Correct:** Fewer dimensions mean fewer parameters (less overfitting) for the network and a more meaningful, cheaper distance computation for K-Means  
*Distractors:* It always increases accuracy to 100% / It removes the need for training data / It makes the model non-deterministic

**A:** Dimensionality reduction simplifies the data distribution and reduces the parameters to fit.

**Q137 [Think]** Match: weight decay, dropout, weight sharing, early stopping. Which assignment of what-each-does is correct?

**Correct:** Weight decay penalises large weights; dropout ignores random outputs in training; weight sharing ties parameters across units; early stopping halts at minimum validation loss  
*Distractors:* Weight decay ignores random outputs; dropout ties parameters / Early stopping penalises large weights; weight sharing halts training / Dropout halts at minimum validation loss; weight decay ties parameters

**A:** Each constrains effective model complexity in a different way.

**Q138 [Calc]** 28×28×1 input → Conv(3×3, K=4, S=1, P=1)+ReLU → MaxPool(2×2, S=2) → flatten → FC(10). Give the flattened length and the total parameters.

**Answers:** flattened length = 784; total parameters = 7890

**A:** 28×28×4 → 14×14×4 → 784. Conv 40; FC 784·10+10=7850. Total 7890.

**Q139 [Calc]** (Backprop meets ResNet — extension) Suppose each block's residual branch has local derivative F′ = 0.1 at every one of 10 blocks. Compare the product of derivatives through the 10 blocks for (a) a plain stack (y = F(x)) and (b) a residual stack (y = F(x) + x, derivative 1 + F′). Give (b) to 2 decimals and (a) in scientific form as the exponent magnitude (enter 10).

**Answers:** (b) residual product = 2.59374; (a) exponent n in 10⁻ⁿ = 10

**A:** (a) 0.1¹⁰=10⁻¹⁰: the gradient vanishes. (b) 1.1¹⁰=2.59: the shortcut keeps gradients flowing, which is why residual learning eases deep training.

**Q140 [Think]** In a mixture-of-Gaussians view of K-Means (equal priors, equal spherical variances), assigning a point to the cluster with the highest P(x|k)P(k) is equivalent to what?

**Correct:** Assigning it to the nearest centre in Euclidean distance (hard MAP assignment)  
*Distractors:* Assigning it to the farthest centre / Assigning it at random / Assigning it by label

**A:** With equal priors/variances the MAP cluster maximises the Gaussian likelihood, i.e. minimises ‖x−μ_k‖. (Mixture of Gaussians is listed among the partition methods.)

**Q141 [Think]** You flip a coin 3 times and get HHH. The ML estimate of P(H) is 1. Which statement links this to overfitting and to the Bayesian view?

**Correct:** ML trusts the few data completely (like overfitting); a non-uniform prior (MAP) pulls the estimate toward prior belief  
*Distractors:* ML includes a prior; MAP does not / MAP and ML always agree / Overfitting cannot happen with probabilities

**A:** h_MAP = argmax P(D|h)P(h); with a non-constant prior it differs from ML when data are scarce.

**Q142 [Calc]** 60,000 labelled images are split into 50,000 training and 10,000 validation images (test set separate). Training uses mini-batches of 128 for 20 epochs. How many parameter updates per epoch (round up for the last partial batch) and in total?

**Answers:** updates per epoch = 391; total updates = 7820

**A:** ⌈50000/128⌉=391; 391·20=7820. The validation set is only used to monitor generalisation.

**Q143 [Calc]** K-Means (unsupervised) puts customers into clusters; a labelled subset shows P(cluster A | buys) = 0.7, P(cluster A | does not buy) = 0.2, and 30% of customers buy. For a new customer in cluster A, compute P(buys | cluster A) using Bayes.

**Answers:** P(buys | A) = 0.6

**A:** 0.21/(0.21+0.14)=0.6000. Cluster IDs can serve as features for a Bayesian classifier.

**Q144 [Think]** Model A (no dropout): train 1%, validation 10%. Model B (dropout): train 3%, validation 6%. Target 2%. Which statement is correct?

**Correct:** Model B generalises better (smaller gap) even though its training error is higher  
*Distractors:* Model A is better because training error is lower / Model B is worse because training error rose / They are equally good because the average error is the same

**A:** Choose by validation error; the gap fell from 9 to 3 points.

**Q145 [Calc]** LeNet-style first layer: 32×32×1 input, six 5×5 filters, S=1, P=0. Give the parameter count, the output size (one side) and the multiply-accumulates (ignoring biases).

**Answers:** parameters = 156; output side = 28; multiply-accumulates = 117600

**A:** 6(25+1)=156 parameters; (32−5)+1=28; 28·28·6·25=117,600.

**Q146 [Calc]** Bottleneck idea (as in Inception's 1×1 reduction): on 256 input channels, compare (a) a direct 3×3 convolution with 256 filters and (b) a 1×1 conv reducing to 64 channels followed by a 3×3 conv with 256 filters. Give the parameter counts of (a) and (b).

**Answers:** (a) direct 3×3 = 590080; (b) reduce + 3×3 = 164160

**A:** (a) 256(2304+1)=590,080. (b) 64(257)+256(576+1)=16,448+147,712=164,160. About 3.6× fewer parameters.

**Q147 [Calc]** A classifier flags cats with sensitivity 95% and false-positive rate 10% on a dataset where 5% of images contain the target. If the classifier flags an image, what is the probability it truly contains the target (Bayes)?

**Answers:** P(target | flagged) = 0.333333

**A:** 0.0475/(0.0475+0.095)=0.3333. A low prior makes false positives dominate.

**Q148 [Think]** Which combination of regularizers is most likely to RAISE bias?

**Correct:** Very large weight decay together with heavy dropout and very early stopping  
*Distractors:* More training data and label-preserving augmentation / Dimensionality reduction with a moderate number of components / Weight sharing in a convolution layer on images

**A:** Each constraint limits the model; too many or too strong leads to underfitting.

**Q149 [Calc]** K-Means on N = 10⁵ points with K = 10 needs N·K distance evaluations per assignment step, each costing d multiply-adds. Compare d = 784 (raw pixels) with d = 32 (autoencoder code). Give the multiply-adds per assignment step for both and their ratio.

**Answers:** d = 784 = 7.84e+08; d = 32 = 3.2e+07; ratio = 24.5

**A:** 784·10⁶ = 7.84×10⁸ versus 3.2×10⁷; the ratio is 24.5 — a compact learned code makes distance computation much cheaper (and the metric more meaningful).

**Q150 [Think]** Which statement about sequence models is correct?

**Correct:** An RNN reuses the same parameters at every time step and passes a hidden state; attention lets each position combine information from other positions weighted by query–key scores  
*Distractors:* An RNN uses different parameters at every time step; attention uses a hidden state / Attention cannot look at earlier tokens / RNNs and attention both require convolution

**A:** Weight sharing and hidden state versus query/key/value attention with positional information.
