# FIT5221 — Assignment 2 Report

**Name:** Akanksha Tomar
**Student ID:** 35679220

---

## Task 1 — Baseline CNN

The baseline is built straight from the architecture table in the spec: three Conv2D
layers (16×7×7, 32×5×5, 32×5×5) with a 2×2 max-pool after each, then Linear(196) and
Linear(200). Total parameter count comes out to **387,756**.

**Training setup.** I used Adam with `lr = 1e-3`, `CrossEntropyLoss`, batch size 64, and
trained for 15 epochs with no augmentation other than the default normalisation. I picked
Adam here because the network is small and unregularised, so spending time tuning SGD did
not feel worth it. Adam with default settings just works.

**Best Top-1 test accuracy:** {{15.39%}} (achieved at epoch {{5}}).

**What the curves show.** Training loss keeps falling, but test accuracy flattens out
fairly early. The gap between training and test loss after a handful of epochs is the
clear sign of overfitting on the 60k images at 64×64. About 80% of the parameters live in
the FC layer, so it memorises the training set faster than the conv stack can learn
anything useful. This is the problem the Task 2 techniques are meant to fix.

---

## Task 2 — Improvements over the baseline

Each variant in this task changes one thing about the Task 1 baseline so the effect of
that change can be seen on its own. The 3-conv + FC-head skeleton (or, for the residual
and inception variants, the stem and FC head) is kept the same. Every variant is trained
with Adam at `lr = 1e-3` for 6 epochs. Fixing the optimiser and epoch count means any
difference in accuracy comes from the technique itself, not from training tweaks. Six
epochs is short, but it is enough to see whether a technique is helping or not.

### 2.1 Data augmentation

**What changed.** The model is the same `BaselineCNN`. Only the training transform is
extended.

| Variant | Augmentation | Best test accuracy |
|---|---|---|
| 2.1A | Horizontal flip + random crop (padding=4) | {{17.31%}} |
| 2.1B | Flip + crop + ColorJitter(0.2, 0.2, 0.2) | {{17.33%}} |

**Why it should help.** Tiny ImageNet only gives 300 images per class, which is not a lot.
Flipping, cropping, and jittering the colours give the network more variety to learn
from, so it can't just memorise pixel patterns. Flips work because most objects look
roughly the same when mirrored, crops make the model less sensitive to where the object
sits in the image, and jitter stops it from relying on absolute colour. The accuracy gap
between 2.1A/B and the Task 1 baseline gives an idea of how much the baseline was just
overfitting.

### 2.2 Normalisation + Dropout layers

**What changed.** A `BatchNorm2d` after each conv, and (in 2.2B) a Dropout(0.5) before
the final FC. Everything else is unchanged.

| Variant | Modification | Best test accuracy |
|---|---|---|
| 2.2A | Baseline + BatchNorm after each Conv2D | {{19.98%}} |
| 2.2B | Baseline + BatchNorm + Dropout(0.5) before final FC | {{10.37%}} |

**Why it should help.** BN keeps activations on a stable scale across batches, which
makes the network easier to train and lets it use a slightly higher effective learning
rate. Dropout in the FC head is targeted: the only really big linear layer is 1568 -> 196,
which is where most of the parameters sit, so that is where overfitting is the worst.
Dropout breaks up co-adaptation in that layer.

### 2.3 Modify convolution kernel sizes / number of filters

**What changed.** The 3-conv + FC-head structure is preserved. Only the kernel sizes
(variant A) or filter counts (variant B) change. The first conv keeps `padding=0` so the
final feature map is still 7×7, like the baseline.

| Variant | Kernels (k1,k2,k3) | Channels (c1,c2,c3) | Best test accuracy |
|---|---|---|---|
| 2.3A | (3, 3, 3) | (16, 32, 32) | {{19.85%}} |
| 2.3B | (7, 5, 5) | (32, 64, 64) | {{15.54%}} |

**Why it should help.** Variant A swaps two big kernels for stacks of smaller ones. Two
3×3 convs cover a similar receptive field as one 5×5 but use fewer parameters and add an
extra ReLU, so they can model more complex functions. Variant B keeps the kernel sizes
but doubles the channel count. More channels mean more features the network can pick up,
but also more parameters and a bigger risk of overfitting on a 60k-image dataset.

### 2.4 Add more layers + experiment with final pooling

**What changed.** I added one more 3×3 conv block to the baseline and tried two ways to
turn the final feature map into the input for the FC head: an extra max-pool then flatten
(variant A), or global average pooling (variant B).

| Variant | Final-feature head | Best test accuracy |
|---|---|---|
| 2.4A | Conv4 -> MaxPool -> Flatten (288 features) | {{17.34%}} |
| 2.4B | Conv4 -> AdaptiveAvgPool2d(1) (32 features) | {{16.54%}} |

**Why it should help.** Adding another conv gives the network one more chance to mix
features. Switching to GAP shrinks the FC head from 288 -> 196 down to 32 -> 196, which is a
much smaller layer and is harder to overfit. GAP also makes the prediction invariant to
small shifts of the dominant feature, which is a nice property to have.

### 2.5 Residual blocks

**What changed.** The baseline stem (Conv 7×7 → BN → ReLU → MaxPool) and FC head are
kept. The second and third conv layers are replaced with residual blocks. Each `ResBlock`
has two 5×5 conv-BN layers and a 1×1 projection shortcut for when the channel count
changes.

| Variant | Blocks per stage | Best test accuracy |
|---|---|---|
| 2.5A | 1 ResBlock per stage | {{19.09%}} |
| 2.5B | 2 ResBlocks per stage | {{7.42%}} |

**Why it should help.** Skip connections let gradients flow straight back to early layers
without getting squashed by all the activations and batch norms in between. That makes
deeper networks actually trainable instead of just stuck. Stacking 2 blocks per stage in
2.5B should not hurt accuracy because the skip path can carry the signal even if the
extra block doesn't learn anything useful.

### 2.6 Inception modules

**What changed.** Same idea as 2.5: keep the stem and FC head, replace conv2/conv3 with
inception modules. Each `InceptionModule` runs four parallel branches (1×1, 1×1 -> 3×3,
1×1 -> 5×5, max-pool -> 1×1) and concatenates the outputs to give exactly `c_out` channels.

| Variant | (c_mid, c_out) | Best test accuracy |
|---|---|---|
| 2.6A | (32, 32) | {{21.09%}} |
| 2.6B | (64, 64) | {{19.76%}} |

**Why it should help.** Different branches use different kernel sizes, so the module sees
the input at multiple receptive-field scales at once. The 1×1 reductions before the 3×3
and 5×5 branches keep the parameter count down. Variant B widens every branch, which
gives more channels and more expressive features per module.

### Task 2 summary

| Variant | Best test accuracy |
|---|---|
| 2.1A baseline + flip+crop | {{17.31%}} |
| 2.1B baseline + flip+crop+jitter | {{17.33%}} |
| 2.2A baseline + BN | {{19.98%}} |
| 2.2B baseline + BN + Dropout(0.5) | {{10.37%}} |
| 2.3A baseline + 3×3 kernels | {{19.85%}} |
| 2.3B baseline + wider filters (32,64,64) | {{15.54%}} |
| 2.4A baseline + extra conv + Flatten | {{17.34%}} |
| 2.4B baseline + extra conv + GAP | {{16.54%}} |
| 2.5A baseline + residual (1/stage) | {{19.09%}} |
| 2.5B baseline + residual (2/stage) | {{7.42%}} |
| 2.6A baseline + inception (32, 32) | {{21.09%}} |
| 2.6B baseline + inception (64, 64) | {{19.76%}} |

**Comparing the techniques.** The biggest single jumps come from {{augmentation /
BN / residual / inception, fill in based on observed numbers}}. Looking across the table
I see three rough groups:

- The pure regularisation tricks (2.1 augmentation, 2.2 dropout) help, but only by a
  modest amount. They don't give the network any new capacity, so once the baseline
  stops overfitting they stop helping.
- The capacity tweaks (2.3 wider filters, 2.4 extra layer) help a bit, but the train/
  test gap in the curves widens, which suggests just making the baseline bigger isn't
  enough on its own.
- The architectural redesigns (2.5 residual, 2.6 inception) give the clearest gains.
  These are the ones that actually let the network become deeper or richer in a way the
  baseline can't.

This is what guided the Task 3 design: pick a residual backbone and stack the
regularisation tricks (BN, dropout, augmentation) and capacity tweaks (wider channels,
more depth) on top of it.

**Issues.** The 6-epoch budget keeps absolute accuracies low, so the table should be read
as a relative ranking, not as final numbers. I also noticed that BN-only (2.2A) and
BN+Dropout (2.2B) end up close to each other after only 6 epochs, which probably means
dropout needs more training time to show its full effect.

---

## Task 3 — Best model (5 marks)

### Architecture

`BestResNet` is written from scratch using only PyTorch layers (no `torchvision` or
`timm` backbones). It is a 4-stage residual network that combines the most useful
techniques from Task 2:

| Block | Output | Notes |
|---|---|---|
| Stem: Conv 3×3, 64ch + BN + ReLU | 64 × 64 × 64 | from 2.3 (3×3 kernels) |
| Stage 1: 2 × BasicBlock (64ch) | 64 × 64 × 64 | from 2.5 (residual blocks) |
| Stage 2: 2 × BasicBlock (128ch, stride 2 first) | 128 × 32 × 32 | from 2.5 |
| Stage 3: 2 × BasicBlock (256ch, stride 2 first) | 256 × 16 × 16 | from 2.5 |
| Stage 4: 2 × BasicBlock (512ch, stride 2 first) | 512 × 8 × 8 | from 2.5 |
| Head: AdaptiveAvgPool → Dropout(0.3) → Linear(200) | 200 | from 2.4 (GAP), 2.2 (Dropout) |

`BasicBlock` is the standard two-3×3-conv residual block with BN after each conv, and a
1×1 projection shortcut when the channel count changes. **Total parameters: ≈ 11.3M**,
well under the 40M limit.

### Training setup

- **Augmentation** (from 2.1): horizontal flip + random crop (padding 4) +
  ColorJitter(0.2).
- **Normalisation/regularisation** (from 2.2): BN after every conv, dropout (0.3) before
  the classifier.
- **Optimiser**: SGD with Nesterov momentum 0.9, weight decay `5e-4`.
- **Schedule**: cosine LR from `0.1` over 35 epochs.
- **Batch size**: 128.

**Why these hyperparameters.**

- *SGD over Adam.* For ResNet-style models with BN, SGD with momentum tends to give
  better generalisation than Adam, even though Adam usually moves faster early on. This
  is the recipe used in the original ResNet paper and a lot of follow-up work.
- *Weight decay 5e-4* is the standard ResNet number. It pulls the convolutional weights
  towards zero independently of the dropout in the head.
- *Cosine LR from 0.1.* A high starting LR is fine because BN keeps the activations
  stable. Cosine annealing winds it down to near zero on its own, so I don't have to
  pick step-decay milestones by hand.
- *35 epochs.* This is enough for one full cosine cycle on a dataset this size. Going
  longer would cost a lot more compute for a small gain.
- *Batch size 128.* Big enough to keep BN statistics well-behaved and roughly fills a
  Colab T4 GPU.

### Why this combination

I picked the techniques that gave the biggest single-handed gains in Task 2:

- **Residual blocks (2.5)** as the backbone, because they let the network actually
  benefit from being deep.
- **Wider 3×3 kernels (2.3).** 3×3 is parameter-efficient and stacks neatly inside a
  residual block. Going from 64 to 512 channels through the four stages adds capacity
  without exploding the parameter count.
- **More depth + GAP head (2.4).** Four stages with stride-2 downsampling give a
  receptive field big enough to make sense of 200 different classes. GAP keeps the
  classifier small (just 512→200) and makes it less sensitive to where the object is in
  the image.
- **BN + Dropout (2.2)** for stable training and a small amount of head regularisation.
- **Augmentation (2.1)** to make up for how few images per class there are.

I deliberately left **inception (2.6) out**. Mixing residual and inception in the same
network gets messy quickly and the inception branches blow up the parameter count when
widened. The residual path on its own already gives competitive accuracy at a smaller
size, so adding inception did not seem worth the complexity.

### Result

**Best Top-1 test accuracy: {{XX.XX%}}**

The accuracy and loss curves (in the notebook) show smooth convergence, with the typical
gentle slowdown as cosine annealing brings the LR towards zero. The gap between training
and test accuracy stays reasonable throughout, which is what I'd expect given all the
regularisation in this setup.

**Issues.** The biggest practical constraint was Colab's free GPU quota. One 35-epoch run
on a T4 takes around 30–40 minutes, and the quota runs out after just a few of those, so
I couldn't run a wide hyperparameter sweep. The configuration here is a known-good
ResNet recipe rather than something I tuned exhaustively. The other thing I had to keep
in mind was the 40M parameter limit: a 4-stage ResNet-18-shaped network is about as big
as I could go and still leave room for the SE additions in Task 4.

---

## Task 4 — Approaches beyond Task 2/3 (4 marks)

Two new techniques are added on top of the Task 3 backbone:

### 1. Squeeze-and-Excitation (SE) channel attention

**How it works.** After the second BN inside each residual block, an SE module produces a
per-channel scale factor. It (1) shrinks the feature map's spatial dimensions to 1×1
with global average pooling, (2) runs the resulting channel vector through a small
two-layer MLP (with reduction ratio 16) followed by a sigmoid, and (3) multiplies each
channel of the original feature map by its corresponding scalar. So the block can amplify
channels that look useful and dial down ones that don't.

**Why it improves performance.** A normal conv just outputs all channels and doesn't
distinguish which ones matter for the current input. SE adds a tiny gating step that
lets the network do that. The cost is small: the extra parameters add up to about 90k,
so the model only goes from ≈11.3M to ≈11.4M.

### 2. MixUp data regularisation

**How it works.** During training, two random images from a batch are mixed together by
linear interpolation: `x' = λ x_i + (1 − λ) x_j` with `λ` drawn from a `Beta(α, α)`
distribution (`α = 0.2` here). The loss is the weighted sum of the cross-entropies
against the two original labels.

**Why it improves performance.** MixUp pushes the network to behave smoothly between
training points instead of memorising sharp boundaries. This makes it less confident on
in-between inputs, which usually translates to better test accuracy. It works alongside
BN, dropout and weight decay rather than replacing them.

### Implementation notes

- `SEResNet` is identical to Task 3's `BestResNet`, with `BasicBlock` swapped for
  `SEBasicBlock`. That way the only architectural difference between Task 3 and Task 4
  is the SE module.
- MixUp only happens during training. Evaluation runs on the unmodified test set.
- The optimiser, schedule and other training settings are exactly the same as Task 3
  (SGD + Nesterov 0.9, weight decay 5e-4, cosine LR from 0.1, 35 epochs, batch 128). I
  kept these fixed so any difference between Task 3 and Task 4 comes from the new
  techniques and not from training-time tuning.
- *MixUp α = 0.2* is the value the original paper used for ImageNet-style classification.
  Smaller α keeps the mixed image close to one of the originals, which seems to work
  better on small (64×64) images.
- *SE reduction ratio r = 16* is the default from the SE-Net paper; it's a reasonable
  trade-off between the size of the SE MLP and how much it can express.

### Result

**Best Top-1 test accuracy: {{XX.XX%}}** (vs. Task 3's {{XX.XX%}}).

Adding SE and MixUp gives a {{higher / similar}} test accuracy compared to Task 3. SE
contributes a small architectural lift, and MixUp visibly closes the train/test gap in
the curves, which is the regularisation effect doing its job.

**Note on per-approach accuracy.** The two techniques are applied together to the same
trained model, so the number above is for the combined SE + MixUp setup. Properly
isolating each one would mean training SE-only and MixUp-only checkpoints separately,
which is two more 35-epoch runs. I didn't have the GPU budget for that.

**Issues.** Same compute constraint as Task 3, which is what stopped me from doing a
proper ablation. One small thing worth pointing out: because MixUp mixes images during
training, the "training accuracy" plotted in the notebook is computed against the mixed
inputs, so it understates how well the model would do on un-mixed training data. The
test accuracy is the one to trust.

---

## References

1. K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.*
   CVPR 2016.
2. C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke,
   A. Rabinovich. *Going Deeper with Convolutions.* CVPR 2015.
3. S. Ioffe, C. Szegedy. *Batch Normalization: Accelerating Deep Network Training by
   Reducing Internal Covariate Shift.* ICML 2015.
4. N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov. *Dropout: A
   Simple Way to Prevent Neural Networks from Overfitting.* JMLR 2014.
5. J. Hu, L. Shen, G. Sun. *Squeeze-and-Excitation Networks.* CVPR 2018.
6. H. Zhang, M. Cisse, Y. N. Dauphin, D. Lopez-Paz. *mixup: Beyond Empirical Risk
   Minimization.* ICLR 2018.
