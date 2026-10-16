# RoboJev and RoboPup: what each model actually does

RoboJev is the inspiration for a separate robotic **judgment layer**. Its public
model accepts sensory **text** and returns probabilities over supplied answers.
RoboPup's optional Qwen3-VL planner accepts an actual robot-eye image and generates
an ordered JSON mission. These solve different parts of a robot's control stack.
Neither model's existence proves that a physical goal has been completed.

This audit fetched the public repository on **2026-10-16** at commit
[`61d36e4e6549842993821f9b3796174df85bea51`](https://github.com/STEERIX-home/robo-jev/tree/61d36e4e6549842993821f9b3796174df85bea51).
Links below are pinned to that inspected source. We inspected documentation and
implementation; we did not run its training, download its weights, or independently
reproduce its benchmark results.

## What the inspected RoboJev release contains

The [README](https://github.com/STEERIX-home/robo-jev/blob/61d36e4e6549842993821f9b3796174df85bea51/README.md)
describes a 2B physical judgment model trained from **Qwen/Qwen3.5-2B**, with a
merged model at [Steerix-org/robo-jev-2b](https://huggingface.co/Steerix-org/robo-jev-2b)
and a [LoRA adapter](https://huggingface.co/Steerix-org/robo-jev-2b-lora). The public
interface accepts a state description and one to twelve questions. It answers
`choice`, `noul` (yes/no), and `score` questions with probabilities or a score.
The model is not a general mission generator in that interface.

The [standalone inference implementation](https://github.com/STEERIX-home/robo-jev/blob/61d36e4e6549842993821f9b3796174df85bea51/src/robo_jev/hf/robo_jev_decide.py)
loads a causal language model. For each question it builds a text prompt ending
at an answer position, obtains the last hidden state, projects it onto the LM-head
rows for the displayed answer codes, and applies softmax. The default performs
one forward pass per question, without autoregressive answer generation. Its
balanced-order mode averages additional passes to reduce answer-position bias.
The file explicitly documents reliability limits above the 26 answer codes used
in training; accepting a larger schema is not evidence of equal accuracy.

The [method](https://github.com/STEERIX-home/robo-jev/blob/61d36e4e6549842993821f9b3796174df85bea51/docs/method.md)
places this model between a slower planner and a faster execution controller:

| Layer | RoboJev's description | RoboPup counterpart |
| --- | --- | --- |
| L3: plan | Decide what to do and in which order; supplied separately | Optional pretrained Qwen3-VL generates supported mission steps |
| L2: judgment | Answer questions about the next few seconds with probabilities | Tiny v4 arena planner plus explicit outcome checks; no calibrated RoboJev-style judgment model is integrated |
| L1: execution | Robot-specific motion controllers; supplied separately | MuJoCo motors, PPO straight-trot residuals, analytic steering and manipulation |

Its training recipe uses simulated decision moments and distributions over
plausible hidden worlds to derive soft labels. The documented training objective
is KL/soft-label cross-entropy over answer codes, with an auxiliary code-mass
term; LoRA and full fine-tuning are implemented in
[`decisions/train.py`](https://github.com/STEERIX-home/robo-jev/blob/61d36e4e6549842993821f9b3796174df85bea51/src/robo_jev/decisions/train.py).
This is different from PPO locomotion training and different from predicting an
entire free-form action sequence.

The repository also contains a streaming robot-harness research path. Its
[`QwenBackbone`](https://github.com/STEERIX-home/robo-jev/blob/61d36e4e6549842993821f9b3796174df85bea51/src/robo_jev/model/backbone_qwen.py)
maintains recurrent/attention state, while a
[`Judge`](https://github.com/STEERIX-home/robo-jev/blob/61d36e4e6549842993821f9b3796174df85bea51/src/robo_jev/model/judge.py)
can score candidate boundaries using a learned pointer readout. The
[model-policy harness](https://github.com/STEERIX-home/robo-jev/blob/61d36e4e6549842993821f9b3796174df85bea51/src/robo_jev/harness/model_policy.py)
turns typed judgments into controller inputs. That research path and the public
single-question answer-code runner are distinct implementations; this comparison
does not treat every repository experiment as one deployed model.

## Does RoboJev see through a camera?

The audited public runner accepts `state: str` and a questions dictionary. Its
model-loading and scoring path consumes text tokens; it does not accept an RGB
image or invoke a vision encoder. Physical simulation and perception experiments
elsewhere in the repository do not change that public input contract. A separate
perception system can describe what the robot senses before calling RoboJev.

RoboPup's new generative interface instead attaches **actual MuJoCo head-camera
RGB** to a pretrained vision-language model, with text, virtual energy, held state
and completed-step history. This supplies direct image input at the planning
layer. The current native low-level v4 controller still uses its evaluated
**overhead RGB** camera to locate arena objects. It is therefore not an end-to-end
robot-eye visuomotor policy. [VLM_GUIDE.md](VLM_GUIDE.md) explains the exact boundary.

## What has been trained here?

| Component | Size | Origin and role |
| --- | --- | --- |
| ArenaPlannerV4 | 246,712 parameters; 1,011,137-byte checkpoint | Small arena transformer trained here by supervised imitation; predicts bounded task/skill proposals |
| Motor actor/critic | 15,705 parameters total; 8,204 in deterministic actor | PPO trained here; residuals around an analytic straight trot |
| Qwen3-VL-2B-Instruct | Pretrained 2B-class VLM; separate multi-GB download | External pretrained image/text generator; used for inference, with no RoboPup fine-tuning |
| Historical `MultimodalPlanner` scaffold | 769,149 parameters | Untrained tensor-contract research code; not the live planner |

RoboPup has not pretrained a billion-parameter LLM from random weights, and has
not trained or imported a RoboJev judgment checkpoint. The optional Qwen3-VL
weights remain separate from the tiny v4 checkpoint. Its visual encoder and
language model are unchanged during RoboPup inference; "frozen" here means no
optimization occurs, not that RoboPup trained a new text-only head on that VLM.

The existing [v4 results](BENCHMARKS.md) include bounded arena classification and
real MuJoCo goal trials. Those figures do not benchmark Qwen3-VL or establish the
calibration of a RoboJev-style decision model. RoboJev's reported physical-judgment
scores and RoboPup's action accuracy use different data, labels and metrics; they
cannot be compared as a ranking.

## How to move closer to the RoboJev research goal

A useful extension would keep the image/text mission planner, add a separate
judgment interface for questions such as "is the ball securely held?" and "should
I look again?", and let the controller verify outcomes. No such learned calibrated
judge is claimed as shipped here. The next evidence should cover:

1. **Visual grounding:** paired scenes with one relevant visible fact changed,
   including blank/occluded views and new viewpoints. Compare actual decisions,
   rather than only accepting valid JSON.
2. **Judgment data:** short physical rollouts for each feasible action under
   uncertain observations, with separated training and held-out scenario families.
   Simulator truth may label outcomes; it must not leak into model inputs.
3. **Calibrated evaluation:** report probability quality, false completion,
   abstention, and action success against tiny-v4, untrained-model and rule
   baselines. A generated sentence claiming success is not an outcome label.
4. **Eye-based control:** replace overhead-v4 object grounding with a tested
   egocentric perception/control path before claiming fully onboard visual control.
5. **Adaptation:** consider LoRA only after collecting a useful, licensed dataset
   and freezing the evaluation protocol. Training budget and deployment latency
   should be measured separately from model-download success.

The inspected repository's source license is
[Apache-2.0](https://github.com/STEERIX-home/robo-jev/blob/61d36e4e6549842993821f9b3796174df85bea51/LICENSE).
This audit copied no RoboJev implementation or weights. Any future import must
preserve applicable notices and check the specific model/dataset license as well
as the code license. Qwen3-VL's model card separately declares Apache-2.0.
