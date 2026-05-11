# Speaker Notes — Visualizing DNN Training with TensorBoard

> **Intended audience:** Undergraduate students who have completed Lab 4 and are familiar
> with basic PyTorch training loops and CIFAR-10.  
> **Estimated lecture time:** 60–75 minutes (hands-on follow-along style)  
> **Format:** Instructor shares screen / projects Jupyter notebook, students run cells alongside.

---

## Title Slide / Cell 0 — Introduction

**What to say:**

> "Today's lab is about **TensorBoard** — a browser-based visualization tool that lets you
> inspect almost every aspect of your neural network training, in real time, without writing
> a single line of plotting code beyond logging calls."
>
> "TensorBoard was originally built for TensorFlow, but PyTorch ships full support via
> `torch.utils.tensorboard`. Everything we do today works identically in both frameworks."

**Background to convey:**

- Before tools like TensorBoard, you would print loss values to the terminal and paste them
  into Excel or matplotlib scripts *after* training. TensorBoard gives you a live dashboard.
- The key mental model: your training code **writes** log files; TensorBoard **reads** them.
  These are two independent processes. You can kill TensorBoard and relaunch it without
  affecting training.

**Learning objective walk-through (2 minutes):**

Read each objective aloud and briefly explain in one sentence what "successfully doing it"
looks like:

1. *SummaryWriter* — "You'll be able to open a log directory, write data, and close it cleanly."
2. *Images* — "You'll see your training samples appear in the browser."
3. *Graphs* — "You'll be able to click through the architecture interactively."
4. *Scalars* — "You'll watch loss and accuracy curves update live while training."
5. *Histograms* — "You'll spot vanishing/exploding gradients visually."
6. *Embeddings* — "You'll see CIFAR-10 features cluster by class in 3D."
7. *HParams* — "You'll compare multiple learning-rate runs in a table."

---

## Section 0 — Install Dependencies

**What to say:**

> "If you're on a shared cluster or your own machine with a clean environment, run this cell.
> Most of you probably have `tensorboard` already if you installed PyTorch with pip."

**Instructor note:**

- The cell is intentionally commented out to avoid accidental reinstalls during a live demo.
- If students are on Google Colab or Jupyter Hub, they may need to uncomment and run.
- Verify with `tensorboard.__version__`. We need at least 2.x.

---

## Section 1 — Environment Setup

### 1.1 Import Libraries

**What to say:**

> "Nothing new here import-wise — we're adding one line:
> `from torch.utils.tensorboard import SummaryWriter`.
> That single class is the entire interface between PyTorch and TensorBoard."

> "We also fix random seeds so your training results will match mine closely,
> which makes it easier to compare our TensorBoard screenshots."

### 1.2 CIFAR-10 Dataset

**What to say:**

> "You already know CIFAR-10 from Lab 4. One thing I want to highlight today is the
> **normalization statistics**."
>
> "Many tutorials use `(0.5, 0.5, 0.5)` for both mean and std — that's fine and simple,
> but the true per-channel statistics of CIFAR-10 are `(0.4914, 0.4822, 0.4465)` for mean
> and `(0.2023, 0.1994, 0.2010)` for std. Using the real statistics brings pixel values
> closer to a standard normal distribution, which typically improves early convergence."

**Pause point:** Ask students —
> "Why do we apply data augmentation (random flip, random crop) only to the training set
> and not the test set?"
>
> *Expected answer:* Augmentation is a regularization technique. We want the test set to
> reflect real deployment conditions, so we evaluate on unmodified images.

### 1.3 Define the Model

**What to say:**

> "In Lab 4 you may have used ResNet or VGG. Today we use our own `SimpleCNN` with three
> convolutional blocks. This is deliberate — we want something small enough to train in
> 5 minutes on a CPU so we can iterate quickly during the lab."
>
> "More importantly, a simple architecture is *easier to interpret* in TensorBoard's
> GRAPHS panel. You'll be able to see every layer clearly."

**Architecture explanation (draw on board or annotate slide):**

- Each *block* = Conv2d + BatchNorm + ReLU + MaxPool. Spatial size halves at each step.
- After 3 blocks: 128 channels × 4 × 4 = 2,048 features per image.
- Two fully-connected layers reduce to 256 then to 10 class logits.
- Dropout(0.5) between FC layers is a regularizer — half of neurons are randomly zeroed
  during training.

**Trainable parameters (~600K):**
> "600,000 parameters is modest by modern standards — GPT-3 has 175 *billion*!
> But for CIFAR-10 it's more than enough to reach ~75 % test accuracy."

---

## Section 2 — SummaryWriter In Depth

**What to say:**

> "The `SummaryWriter` is a file-system abstraction. When you call `writer.add_scalar(...)`,
> the data doesn't go straight to disk — it sits in an in-memory queue.
> The queue is flushed every `flush_secs` seconds (default 120) or when you call
> `writer.flush()` or `writer.close()`."

**Key teaching points:**

1. **One directory per experiment** — If you run two experiments into the same `log_dir`,
   their data is interleaved and you can no longer tell them apart. Always use a unique
   subdirectory, e.g. `runs/exp_lr1e-3_bs128_2024-01-15`.

2. **The `comment` shortcut** — `SummaryWriter(comment='_lr0.01')` auto-creates a directory
   like `runs/Jan15_14-30-00_myhostname_lr0.01` — handy for quick experiments.

3. **`with` statement pattern** — Show the `with SummaryWriter(...) as writer:` pattern
   and explain it auto-calls `close()` even if an exception is raised. This prevents partial
   / corrupted log files.

**Live demo (2 minutes):**
Create a minimal writer, call `add_scalar` a few times in a loop, close, then show
the resulting files with `ls runs/cifar10_tutorial/`.

```python
# Quick demo — not in the notebook
with SummaryWriter('runs/demo') as w:
    for i in range(10):
        w.add_scalar('demo/loss', 1.0 / (i + 1), i)
# ls runs/demo/  — shows a .tfevents file
```

---

## Section 3 — Visualizing Training Data

**What to say:**

> "Before any training, let's write 32 images to TensorBoard. This is a debugging habit —
> you should *always* visually verify your data pipeline before training, not just check
> tensor shapes."

**`denormalize` function:**

> "Because we normalized pixel values with mean and std during preprocessing,
> the raw tensor has values roughly in [-2, 2]. If we pass that directly to TensorBoard
> it would render as an all-grey smear. The `denormalize` function reverses the normalization:
> `x_original = x_normalized * std + mean`, then we clamp to [0, 1]."

**`make_grid` explanation:**

> "`torchvision.utils.make_grid` takes a batch of images `(N, C, H, W)` and
> tiles them into a single image `(C, H', W')` with `nrow` images per row.
> We then pass this single tiled image to `add_image`."

**TensorBoard navigation (walk students through the UI):**

1. Open `http://localhost:6006`.
2. Click **IMAGES** tab.
3. Point out the step slider (bottom of the image) — currently step 0.
4. Explain that if we call `add_image` again at a later step, the slider would appear.

---

## Section 4 — Model Computation Graph

**What to say:**

> "PyTorch uses *dynamic computation graphs* — the graph is rebuilt on every forward pass.
> `add_graph` works by doing one traced forward pass with `torch.jit.trace` under the hood
> and capturing the graph structure."

**Why a dummy input?**

> "We pass `dummy_input = torch.randn(1, 3, 32, 32)` — random noise. The values don't
> matter; we just need the shapes to be correct so PyTorch can trace the data flow."

**TensorBoard navigation:**

1. Click **GRAPHS** tab.
2. Show the top-level `SimpleCNN` node — click to expand.
3. Expand `features` — students should see `conv2d`, `batch_norm`, `relu`, `max_pool2d`
   repeated 3 times.
4. Hover over an edge and read the shape annotation.
5. Expand `classifier` — show `flatten`, `linear`, `relu`, `dropout`, `linear`.

**Common question:** *"Why does the graph look different from the `print(net)` output?"*
> "The GRAPHS panel shows the *computational* graph — individual tensor operations
> (e.g., `aten::convolution`), not just module names. It's more granular."

---

## Section 5 — Logging Scalar Metrics

**What to say (before running the training loop):**

> "This is the most important section. The ability to watch loss and accuracy update in
> real time while training is what most people think of when they say 'TensorBoard'."

**`add_scalar` vs `add_scalars`:**

> "`add_scalar` writes a single number to a single tag — you get one curve per tag.
> `add_scalars` writes multiple numbers under sub-tags and renders them **on the same chart**.
> We use `add_scalars` for train vs. val so we can see the gap at a glance."

**`global_step` concept (crucial):**

> "The `global_step` parameter is the x-axis of your chart.
> For iteration-level loss we use `global_step` (counts up by 1 each batch).
> For epoch-level metrics we use `epoch + 1` (1 to NUM_EPOCHS).
> Be consistent — mixing iteration steps and epoch steps on the same chart
> will produce a misleading x-axis."

**CosineAnnealingLR:**

> "We also log the learning rate. Watch the `LearningRate` chart in TensorBoard —
> you'll see it follow a cosine curve from 0.01 down to near 0 over 5 epochs.
> This is a common trick: high LR early for fast progress, low LR late for fine-tuning."

**Live monitoring (do this while training runs):**

While Cell 19 is running, switch to the TensorBoard tab and:
1. Show the `Loss/train_iter` chart updating every ~50 iterations.
2. Point out the Smoothing slider — set it to 0.6 to smooth the noisy per-iteration curve.
3. After the first epoch completes, show `Loss/epoch` and `Accuracy/epoch`.

**Discussion prompt after training:**

> "Looking at the charts — is there a sign of overfitting after 5 epochs?
> (Train accuracy is higher than val accuracy — yes, slight overfitting.
> With more epochs it would get worse. What could we add to fix it? — more dropout,
> weight decay, data augmentation, early stopping.)"

---

## Section 6 — Weight and Gradient Histograms

**What to say:**

> "Histograms answer the question: *what are the weights and gradients actually doing
> during training?* This is especially valuable for debugging architectures that
> don't converge."

**Vanishing gradient explanation:**

> "In a deep network, gradients are multiplied together as they flow backwards through
> layers (chain rule). If every gradient is slightly less than 1, multiplying many of them
> together gives a value close to 0 — the early layers receive almost no signal.
> BatchNorm helps a lot with this, which is why we include it in our CNN."

**Exploding gradient explanation:**

> "The opposite: each gradient is slightly greater than 1. The product grows exponentially.
> This usually manifests as `loss = NaN` or very erratic training curves."

**Reading the HISTOGRAMS panel:**

> "The x-axis is the parameter value, the y-axis is the training step (time),
> and the color (light → dark) represents frequency.
> A healthy weight histogram looks like a Gaussian that *gradually spreads out* as training
> proceeds — the network is learning."

**Reading the DISTRIBUTIONS panel:**

> "The DISTRIBUTIONS tab shows percentile bands (min, 5th, 25th, median, 75th, 95th, max)
> as a function of training step. It's a cleaner view for spotting systematic drift.
> A healthy gradient distribution has the median near 0 and narrow bands."

**Instructor note:** Point students to a specific layer (e.g., `features.0.weight` vs
`features.8.weight`) and ask: "Is the last conv layer's gradient larger or smaller than the
first layer's? Why?" — expected: larger, because it's closer to the loss.

---

## Section 7 — Embedding Projector

**What to say:**

> "Suppose you want to know whether your network has learned *meaningful representations*.
> One way: extract the feature vectors from the penultimate layer and plot them in 3D.
> If the network has learned well, features from the same class should cluster together."

**Why we use the feature backbone output:**

> "We use `net.features(inputs)` — the output *before* the classifier head.
> This is the network's internal representation of the image, a 2,048-dimensional vector.
> The final 10-dimensional output after the classifier is just class logits;
> the rich representation lives in the feature space."

**PCA vs t-SNE (spend 2 minutes on this):**

> "PCA projects to the directions of maximum variance — it's linear and instantaneous.
> t-SNE minimizes the KL-divergence between neighborhood distributions in high- and low-
> dimensional space — it's non-linear and takes many iterations but usually reveals
> cluster structure much more clearly."

> "A rule of thumb: start with PCA for a quick look. If clusters overlap, try t-SNE.
> t-SNE preserves *local* structure but distorts *global* distances — two clusters being
> far apart in t-SNE does NOT necessarily mean they're far in the original space."

**Thumbnail images:**

> "Notice each point in the projector has a thumbnail — that's the `label_img` argument.
> We pass the denormalized test images so you can hover over a point and see the actual photo.
> This is extremely useful for spotting mis-clustered examples."

---

## Section 8 — Hyperparameter Comparison

**What to say:**

> "In real research you run tens or hundreds of experiments with different hyperparameters.
> Without a systematic way to track them, you'll quickly lose track of which run used what
> settings. `add_hparams` is TensorBoard's built-in solution."

**Important implementation detail:**

> "Each run **must** use its own `SummaryWriter` pointed to a unique directory.
> If two runs share a directory, the hparam records overwrite each other.
> That's why we do `run_dir = f'runs/hparam_search/lr_{lr_exp:.0e}'`."

**Three views explained:**

1. **TABLE VIEW**: spreadsheet-like. Click any column header to sort by that metric.
   This is the fastest way to find the best run.

2. **PARALLEL COORDINATES**: each vertical axis is one hyperparameter or metric.
   Each run is a polyline connecting its values across axes.
   Squeeze axes together where you see many lines crossing (that parameter doesn't matter).
   Lines that are nearly parallel indicate a parameter that *does* matter.

3. **SCATTER PLOT MATRIX**: each cell is a scatter plot of one variable vs. another.
   Useful for seeing correlations (e.g., "lower lr → higher val loss").

**Instructor note:** After the sweep runs, demo the TABLE VIEW and ask students to
rank the runs. Then switch to PARALLEL COORDINATES and ask: "Which learning rate
produced the best val accuracy?" — expected: 1e-2 should win at only 2 epochs.

---

## Section 9 — Precision–Recall Curves

**What to say:**

> "We evaluate every classifier with accuracy, but accuracy hides class-level performance.
> A PR curve shows you the precision-recall trade-off for each class individually."

**Precision and Recall refresher:**

> "Precision = of all the samples you *predicted* as class X, how many *actually* are X?
> Recall = of all samples that *actually* are class X, how many did you correctly detect?
> There's always a trade-off: to catch every positive (high recall) you may need to
> accept more false positives (lower precision)."

**One-vs-rest strategy:**

> "CIFAR-10 has 10 classes. `add_pr_curve` expects binary labels.
> We use one-vs-rest: for each class, we create a binary problem where that class = 1
> and all other classes = 0. We run this 10 times and get 10 PR curves."

**Reading the chart:**

> "The area under the PR curve (AUPRC) summarizes performance.
> A perfect classifier has AUPRC = 1.0 (the curve goes to the top-right corner).
> Look at which classes have the lowest AUPRC — these are the classes where the model
> struggles the most. For CIFAR-10, `cat` and `dog` are notoriously hard."

---

## Section 10 — Closing the Writer

**What to say:**

> "This is a short but important section. If you don't call `writer.close()`,
> data may be sitting in an in-memory buffer that never gets written to disk.
> You might end up with incomplete logs that look truncated in TensorBoard."

> "Best practice: use the `with` block pattern. Python's context manager guarantees
> `close()` is called even if an exception is raised mid-training."

---

## Section 11 — Launching TensorBoard

**What to say:**

> "Let's review the three main ways to launch TensorBoard."

**Command-line (most common in practice):**

> "Open a new terminal window, `cd` into the directory containing `runs/`,
> and run `tensorboard --logdir=runs`. The server starts and prints the URL.
> Leave this terminal open while you train — TensorBoard polls for new `.tfevents`
> files automatically."

**Jupyter inline magic:**

> "`%tensorboard --logdir runs` renders TensorBoard directly inside the notebook cell output.
> This is very convenient for demos and coursework. The magic reuses the server if it's
> already running."

**`--logdir_spec` for multi-experiment comparison:**

> "If your runs are in separate top-level directories (not subdirectories of `runs/`),
> use `--logdir_spec name1:path1,name2:path2`.
> The names you give appear as labels in the Runs panel on the left."

**Panel overview table — walk through each row:**

Spend 30 seconds on each panel. Ask students which panel they expect to use most.
*(Usually SCALARS wins.)*

---

## Section 12 — Summary and API Quick Reference

**What to say:**

> "This table is your cheat sheet for the rest of the course — and for your own projects.
> Any time you want to log something, check here first. There's almost certainly a
> TensorBoard API that does it."

**Highlight the coverage checklist:**

> "We covered 8 different TensorBoard panels today. The two we *didn't* cover are
> `add_text` (logging arbitrary text, useful for hyperparameter configs or model summaries)
> and `add_audio`/`add_video` (for non-image modalities). Those follow the same pattern."

---

## Section 13 — Practice Exercises

**How to assign these:**

| Exercise | Difficulty | Suggested deadline |
|----------|-----------|-------------------|
| 1 (Top-5 Accuracy) | ⭐ | In-class, 10 min |
| 2 (Compare Optimizers) | ⭐⭐ | Homework |
| 3 (Gradient Clipping) | ⭐⭐ | Homework |
| 4 (Visualize Filters) | ⭐⭐⭐ | Extra credit |
| 5 (Prediction Viz) | ⭐⭐⭐ | Extra credit |

**Hints for Exercise 1:**

```python
# Inside evaluate(), add:
_, top5_preds = outputs.topk(5, dim=1)
top5_correct += top5_preds.eq(targets.view(-1, 1)).sum().item()
```

**Hints for Exercise 3:**

Walk students through how to intentionally break training (lr=1.0),
observe `loss = nan` or wildly oscillating loss in SCALARS,
then add `clip_grad_norm_` and show the curves stabilize.
This is one of the most instructive debugging exercises — they will encounter
gradient explosions in their own projects.

**Hints for Exercise 4:**

```python
filters = net.features[0].weight.data.cpu()  # (32, 3, 3, 3)
# Normalize each filter to [0,1] for visualization
vmin = filters.min(); vmax = filters.max()
filters_vis = (filters - vmin) / (vmax - vmin + 1e-8)
grid = torchvision.utils.make_grid(filters_vis, nrow=8)
writer.add_image('filters/conv1', grid, global_step=0)   # before training
# ... after training:
writer.add_image('filters/conv1', grid, global_step=NUM_EPOCHS)
```

---

## General Q&A — Anticipated Questions

**Q: Does TensorBoard slow down training?**

> "Negligibly. Writing to disk is fast, and the default `flush_secs=120` means you're
> not flushing on every step. For very performance-sensitive runs you can increase
> `flush_secs` or only log every N steps (which we already do with `LOG_EVERY`)."

**Q: Can TensorBoard handle distributed training with multiple GPUs?**

> "Yes. Each worker can write to its own subdirectory or to a shared network path.
> PyTorch Lightning and Hugging Face Trainer both integrate TensorBoard out of the box
> for distributed settings."

**Q: What's the difference between TensorBoard and Weights & Biases (W&B)?**

> "TensorBoard is free, open-source, runs locally, and requires no account.
> W&B is a cloud service with a richer UI, experiment management, and collaboration
> features — but you need to sign up and data goes to their servers.
> For a course like this, TensorBoard is the right choice. In industry you'll likely
> encounter both."

**Q: Can I log custom data types not listed in the API table?**

> "Yes — `add_figure` accepts any matplotlib `Figure` object, and `add_custom_scalars`
> lets you build custom layout panels. For completely custom HTML/JS you'd need a
> TensorBoard plugin, which is beyond this course."

**Q: My PROJECTOR tab just shows a blank page. What's wrong?**

> "The PROJECTOR tab uses WebGL. Make sure your browser is up to date and that WebGL is
> enabled. Also, the embedding is logged asynchronously — give TensorBoard a few seconds
> to process the `.tfevents` file after you close the writer."

---

## Closing (5 minutes)

**Key takeaways to reinforce:**

1. `SummaryWriter` is your single point of contact. Create it at the start,
   close it at the end.
2. Use **unique log directories** per experiment — never share directories between runs
   unless you intend to compare them.
3. Log at **multiple granularities**: per-iteration (fine detail) and per-epoch (overall trend).
4. The **HISTOGRAMS** and **PROJECTOR** panels are underused by beginners but extremely
   powerful for diagnosing training problems and understanding what the network has learned.
5. `add_hparams` turns manual experiment tracking (spreadsheets, paper notes) into
   an automated, visual, searchable database.

**Preview of next lab:**

> "In the next lab we'll move to more advanced training scenarios — learning rate
> scheduling and model checkpointing. TensorBoard will be our primary diagnostic tool,
> so everything you learned today will be directly useful."

---

## Section 12 — PyTorch Profiler

### 12.1 Introduction & Motivation

**What to say:**

> "So far we've been using TensorBoard to monitor *what* our model is learning —
> loss curves, accuracy, weight distributions, embeddings.  Now we'll look at a
> completely different question: *how efficiently* is our model running?"

> "The PyTorch Profiler (`torch.profiler`) records the exact CPU and GPU time spent
> on every single operator during a forward/backward pass.  It also tracks every
> memory allocation and deallocation.  The results can be visualized in a dedicated
> TensorBoard tab called **PYTORCH_PROFILER**."

**Why profiling matters:**

- A model that trains 2× faster is effectively like having 2× more compute budget.
- Most inefficiencies are invisible to the naked eye — you need measurement data.
- Common surprises: data loading is often the bottleneck on fast GPUs, not the model itself.
- Profiling before optimizing prevents "premature optimization" — fixing things that
  aren't actually slow.

**Relation to TensorBoard:**

> "The profiler integrates with TensorBoard through the `tensorboard_trace_handler`.
> This writes a Chrome Trace JSON file that the `torch_tb_profiler` plugin parses
> and renders as interactive flame graphs, tables, and memory timelines.
> The same `tensorboard --logdir=...` workflow applies."

### 12.2 Install the Plugin

**What to say:**

> "The profiler itself ships with PyTorch — `import torch.profiler` just works.
> The *TensorBoard visualization* requires one extra package: `torch_tb_profiler`.
> One pip install and you're done."

**Instructor note:** The verification cell (`try: import torch_tb_profiler`) is there
as a gentle guard. On a shared cluster the package might not be installed system-wide.
Students may need `pip install --user torch_tb_profiler`.

### 12.3 Profile Schedule

**What to say:**

> "You don't want to profile every single training step — that would add significant
> overhead to an already slow training loop.  The `schedule` parameter defines a
> repeating pattern: skip a few steps, warm up, then record for a fixed number of
> active steps."

**Walk through the four phases with a concrete analogy:**

> "Think of it like a photographer at a sporting event.  The `wait` phase is them
> getting into position — they're present but not shooting.  The `warmup` phase is
> taking test shots to calibrate exposure — the pictures get deleted.  The `active`
> phase is the real shoot — every frame is saved.  `repeat` says how many times to
> run this cycle before stopping."

**Number the steps aloud:**

With `wait=1, warmup=1, active=3, repeat=1`:
- Step 0 → wait (profiler is idle)
- Step 1 → warmup (profiler runs but discards data — warms up JIT/cuDNN)
- Steps 2, 3, 4 → active (3 steps of real trace data collected)
- Step 5 → profiler stops

> "So we need to feed at least `wait + warmup + active = 5` batches through the loop.
> We use `if step >= WAIT + WARMUP + ACTIVE - 1: break` to stop exactly on time."

**`prof.step()` placement:**

> "This is the #1 mistake students make: calling `prof.step()` at the *start* instead
> of the *end* of the training step.  `prof.step()` signals to the profiler that the
> current step is complete and it should advance its state machine.  Always call it
> **after** `optimizer.step()`."

### 12.4 Running the Profiler (Code Walkthrough)

**Walk through the code cell line by line:**

1. `PROF_LOG_DIR = 'log/simplecnn_profiler'` — dedicated subdirectory, separate from
   the TensorBoard scalar logs in `runs/`.
2. `net_prof = SimpleCNN().to(device)` — fresh model, no prior training state.
3. `train_step(inputs, targets)` — standard forward/backward/step; nothing special.
4. `with torch.profiler.profile(...) as prof:` — the context manager starts recording.
5. `on_trace_ready=torch.profiler.tensorboard_trace_handler(PROF_LOG_DIR)` — automatically
   writes a `.pt.trace.json` file when the active phase ends.
6. `record_shapes=True` — attaches tensor shapes to each operator entry (slows down
   profiling slightly but invaluable for debugging).
7. `profile_memory=True` — tracks every `malloc`/`free` in the PyTorch allocator.
8. `with_stack=True` — captures Python call stacks; enables the stack view in TensorBoard.
   *Significant overhead on CPU — consider disabling for large models.*

**What to show while it runs:**

- The loop only runs `WAIT + WARMUP + ACTIVE = 5` batches — very fast.
- Point out the `prof.step()` call at the end of the loop body.
- After the `with` block exits, the trace file is flushed automatically.

### 12.5 Navigating the PYTORCH_PROFILER Panel

**Walk students through the TensorBoard UI (share screen):**

1. **Launch TensorBoard:**
   ```
   tensorboard --logdir=log/simplecnn_profiler
   ```
2. Click the **PYTORCH_PROFILER** tab (it may take 5–10 seconds to appear while
   the plugin parses the trace).

**Overview tab:**

> "The Overview is a wall-clock pie chart. It shows what fraction of time was spent
> in computation, data loading, and idle waiting.  On a CPU-only machine you'll see
> very little idle.  On a GPU machine with slow data loading, you'd see a large
> 'DataLoader' slice — that's your first optimization target."

**Operator tab:**

> "This is a ranked table of every PyTorch operator.  The key columns are:
> - **Self CPU %** — time spent *inside* this op (not counting child ops)
> - **CPU total** — total time including all called sub-ops
> - **# Calls** — how many times this op was invoked in the profiled steps"
>
> "Look for the top 3–5 operators by Self CPU %.  For our CNN on CPU, you'll likely
> see `conv2d`, `batch_norm`, and `max_pool2d` near the top.  That's expected —
> convolutions dominate CNN computation."

**Trace tab (flame graph):**

> "This is a horizontal timeline.  Each row is a thread.  Each block is one operator
> call.  The x-axis is real wall-clock time.  Zoom in with scroll wheel; pan with drag."
>
> "You can see the forward pass and the backward pass as two distinct clusters.
> Hover over a block to see its exact duration.  For GPU runs, you'd also see a
> second row showing CUDA kernel launches."

**Memory tab:**

> "If you enabled `profile_memory=True`, the Memory tab shows a timeline of memory
> usage.  The y-axis is bytes; the x-axis is time.  Look for sudden spikes — those
> are large intermediate tensors created during the forward pass.
> The backward pass mirror the forward: gradients are allocated in reverse order."

**Module tab:**

> "The Module tab maps profiling data back to your `nn.Module` hierarchy.
> `SimpleCNN.features` vs `SimpleCNN.classifier` — you can see how time splits
> between the convolutional backbone and the FC layers."

### 12.6 Console Output (`key_averages`)

**What to say:**

> "You don't always have TensorBoard handy.  `prof.key_averages().table()` prints
> a human-readable operator table right in the notebook output.
> This is useful for quick checks without launching a browser."

**Three variants demonstrated:**

1. `sort_by='cpu_time_total'` — total time including sub-ops; good for finding
   which *operation* takes the most end-to-end time.
2. `sort_by='self_cpu_time_total'` — time in the op itself; good for finding
   which *implementation* is the actual bottleneck.
3. `group_by_input_shape=True` — groups results by the shape of input tensors;
   useful for spotting shape mismatches that cause sub-optimal kernel selection.

**Sample output to show (or draw on board):**

```
Name                    Self CPU %  CPU total  CPU time avg  # of Calls
---------------------------------------------------------------------------
aten::convolution           45.2%    52.1ms        17.4ms           3
aten::batch_norm             8.3%     9.6ms         3.2ms           3
aten::max_pool2d             6.1%     7.1ms         2.4ms           3
aten::mm (FC layers)         5.7%     6.6ms         3.3ms           2
...
```

### 12.7 Performance Tuning Tips (Table Discussion)

**What to say:**

> "Now that we can *measure* performance, let's talk about what to *do* about it."

**Walk through the table row by row:**

1. **DataLoader bottleneck** — "If your GPU finishes a batch before the CPU has loaded
   the next one, your GPU is idle.  Increasing `num_workers` uses separate processes
   to prefetch data.  `pin_memory=True` avoids an extra CPU-RAM copy for CUDA tensors."

2. **CPU-GPU sync** — "Every time Python calls `.item()` on a GPU tensor, it forces
   a CPU-GPU sync — the CPU waits for the GPU to finish.  Minimize `.item()` calls
   inside training loops.  Also, mixed-precision training (`torch.cuda.amp`) halves
   memory bandwidth for most ops."

3. **Many small ops** — "If you see hundreds of tiny ops in the flame graph,
   consider fusing them.  `torch.compile()` (PyTorch 2.0+) automatically fuses
   element-wise ops into efficient kernels."

4. **Memory fragmentation** — "Lots of small allocations can fragment the allocator.
   If possible, pre-allocate output buffers and reuse them with the `out=` parameter."

5. **Conv2d dominates (expected)** — "For our CNN this is normal — CNNs are supposed
   to spend most time in convolutions.  To make convs faster, enable
   `torch.backends.cudnn.benchmark = True` on a CUDA machine.  This runs a few
   calibration steps to pick the fastest algorithm for your specific input shapes."

**Live experiment (optional, 3 minutes):**

> "Let's try the quickest win right now.  Change `num_workers=2` to `num_workers=4`
> in the DataLoader at the top of the notebook, re-run the profiler, and check
> whether the DataLoader slice in the Overview tab shrinks."

---

## Section 13 — Updated Summary

**What to say:**

> "We've added the Profiler to the coverage checklist.  The mental model now is:
> - TensorBoard `SummaryWriter` tells you *what* is happening (loss, accuracy, features)
> - PyTorch Profiler tells you *how efficiently* it's happening (time, memory, kernels)"

---

## Section 14 — Practice Exercises (Exercise 6)

**Exercise 6 guidance:**

> "This exercise asks you to profile two different architectures and compare them.
> `SimpleCNN` is shallow and fast; ResNet-18 has skip connections and 18 layers.
> The profiler will show you *exactly* where the extra time goes."

**Expected findings students should observe:**

- ResNet-18 will have many more `aten::add` operations (skip connections).
- ResNet-18 will use more peak memory (more feature maps).
- Both will show `DataLoader` as a meaningful fraction on CPU — the bonus task of
  enabling `num_workers=4` should visibly reduce this fraction.

**Hint for the bonus task:**

```python
# Change in Section 1.2:
trainloader = DataLoader(trainset, batch_size=128, shuffle=True,
                         num_workers=4, pin_memory=True)
```

Then re-run the profiler cell and compare the **Overview** tab before/after.

---

## General Q&A — Profiler-Specific Questions

**Q: Does `with_stack=True` work on all platforms?**

> "Yes, but it adds significant overhead on CPU (Python stack unwinding is slow).
> On GPU machines the overhead is proportionally smaller.  For quick profiling
> runs you can omit it; only use it when you need to trace back an expensive
> operator to a specific line in your Python code."

**Q: Can I profile only the forward pass, not the backward?**

> "Yes — just don't call `loss.backward()` inside the profiled steps.
> Or, use `torch.profiler.record_function('forward')` as a context manager to
> label specific code blocks and filter them in the flame graph."

**Q: How do I profile a model on a multi-GPU setup?**

> "Use `dist.barrier()` to synchronize before profiling, then have each rank write
> to its own sub-directory: `tensorboard_trace_handler(f'log/rank_{rank}')`.
> TensorBoard's PYTORCH_PROFILER plugin can display all ranks side by side and
> highlight imbalances between them."

**Q: What's the difference between `self_cpu_time_total` and `cpu_time_total`?**

> "`cpu_time_total` includes time spent in all child operations (the full subtree
> below this node in the operator graph).  `self_cpu_time_total` subtracts child
> time — it's the time the implementation of *this* operator spent, ignoring what
> it delegates.  Use `self` time to find the actual hotspot; use `total` time to
> understand the overall impact of a high-level operation."

**Q: The PYTORCH_PROFILER tab doesn't appear in TensorBoard. What's wrong?**

> "Most likely `torch_tb_profiler` is not installed in the Python environment that
> TensorBoard is running from.  Run `pip install torch_tb_profiler` in the same
> environment, restart TensorBoard, and refresh the browser.  Also double-check
> that the `log/simplecnn_profiler` directory contains a `.pt.trace.json` file."
