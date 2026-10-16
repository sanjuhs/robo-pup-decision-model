"""Versioned bounded English curriculum; final templates are excluded from training."""

from robopup.planner_data import PHRASES

PREFIXES = (
    "",
    "miso ",
    "would you ",
    "would you please ",
    "hey puppy ",
    "hey miso ",
    "please ",
    "could you ",
    "can you please ",
    "good dog ",
    "little puppy ",
    "i want you to ",
)
SUFFIXES = ("", " please", " now", " for me", " little puppy", " miso")
BALLS = ("the ball", "that ball", "the red ball", "that little red ball", "the toy ball")
CRATES = ("the crate", "that crate", "the orange box", "that orange cube")
BEACONS = ("me", "the beacon", "that purple beacon")
ROOTS = {
    "stop": [
        "stay still",
        "wait here",
        "remain here",
        "stand still",
        "do not move",
        "do not run",
        "do not jump",
        "do not fetch the ball",
    ],
    "approach": [
        f"{v} {o}" for v in ("sniff", "go touch", "go over to", "boop", "approach") for o in BALLS
    ],
    "behind": [
        f"{v} {o}"
        for v in ("go behind", "walk behind", "get behind", "go to the far side of")
        for o in BALLS
    ],
    "fetch": [
        f"{v} {o}{s}"
        for v, s in (
            ("fetch", ""),
            ("retrieve", ""),
            ("bring", " home"),
            ("bring me", ""),
            ("get", " and return"),
            ("collect", " and bring it back home"),
            ("carry", " back home"),
        )
        for o in BALLS
    ],
    "follow": [f"{v} {o}" for v in ("follow", "track", "chase") for o in BEACONS],
    "push": [
        f"{v} {o}{s}"
        for v, s in (
            ("push", ""),
            ("nudge", ""),
            ("move", " toward the ring"),
            ("push", " to the ring"),
        )
        for o in CRATES
    ],
    "walk": [
        "walk forward",
        "walk straight ahead",
        "walk slowly forward",
        "take a slow walk",
        "take a walk forward",
    ],
    "run": [
        "run forward",
        "run straight ahead",
        "sprint forward",
        "run fast",
        "go fast forward",
        "zoomies",
    ],
    "jump": ["jump", "jump up", "hop", "do a jump", "do a hop", "take a little hop", "leap up"],
    "rest": [
        "rest",
        "recharge",
        "relax",
        "take a break",
        "take a rest break",
        "take a little rest",
        "have a rest",
    ],
    "drop": [f"{v} {o}" for v in ("drop", "release", "let go of") for o in BALLS],
    "unsupported": [
        "what is the ball",
        "what is the color of the ball",
        "what is the weather",
        "what is the moon",
        "tell me a joke",
        "fly to the moon",
        "open the door",
        "delete files",
        "fetch the crate",
        "push the ball",
        "jump behind the moon",
    ],
}
for intent, roots in PHRASES.items():
    ROOTS[intent].extend(roots)

VALIDATION = {
    "stop": "miso could you please stay here",
    "approach": "could you please touch that red ball miso",
    "behind": "please go to the far side of that ball miso",
    "fetch": "good dog retrieve that red ball for me",
    "follow": "could you chase that beacon for me",
    "push": "good dog nudge the orange crate now",
    "walk": "could you walk slowly straight ahead miso",
    "run": "can you run fast straight ahead for me",
    "jump": "can you do a little hop miso",
    "rest": "good dog please take a rest now",
    "drop": "could you drop that red ball for me",
    "unsupported": "what is your favorite moon color",
}
# Frozen before the paid experiment. Never used for selection or gradients.
FINAL_TEST = {
    "stop": "would you please remain here puppy",
    "approach": "little puppy could you sniff that ball for me",
    "behind": "could you take miso behind that little red ball",
    "fetch": "please bring that little red ball back home for me",
    "follow": "good puppy would you track that purple beacon now",
    "push": "could you nudge that orange cube toward the ring for me",
    "walk": "little dog please take a slow walk straight ahead",
    "run": "please run straight ahead little puppy",
    "jump": "good dog could you jump up for me now",
    "rest": "could you take a little rest break now puppy",
    "drop": "please let go of that red ball now miso",
    "unsupported": "what is behind the moon",
}


FINAL_TEST4 = {
    "stop": "hey little puppy would you please stay still here",
    "approach": "can you please go and sniff the red ball for me",
    "behind": "please walk behind that little red ball for me miso",
    "fetch": "could you please collect the toy ball and bring it back home",
    "follow": "miso would you please follow that purple beacon for me",
    "push": "hey puppy please push that orange box to the ring",
    "walk": "could you take a slow walk forward for me",
    "run": "miso could you sprint straight ahead please",
    "jump": "hey little puppy please take a hop now",
    "rest": "would you please take a little rest break miso",
    "drop": "miso would you release that little red ball please",
    "unsupported": "could you tell me what is behind that orange box",
}
ROOTS["unsupported"].extend(
    f"{question} {obj}"
    for question in ("what is", "what is behind", "tell me what is behind")
    for obj in ("the ball", "the crate", "the orange box", "the moon")
)


def training_phrases(intent):
    excluded = {
        v.lower() for v in (*VALIDATION.values(), *FINAL_TEST.values(), *FINAL_TEST4.values())
    }
    return sorted({p + r + s for p in PREFIXES for r in ROOTS[intent] for s in SUFFIXES} - excluded)
