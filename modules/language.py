"""
language.py -- language production. The Broca's-area analogue.

WHY THIS MODULE EXISTS, and what it is honestly doing.

The four lobes perceive and fuse. They cannot talk, and no amount of training
would let them: a classifier emits a distribution over a fixed label set, and
conversation is not a classification problem. PROJECT.md section 11 flagged
this from the beginning -- "the classifier head becomes the bottleneck the
moment you want language out".

So speech gets its own module, which is also how a brain is arranged. Visual
cortex does not produce sentences; it delivers a percept, and a separate
language region renders it into words. Damage to occipital cortex makes you
unable to see; damage to Broca's area makes you unable to speak about what you
still perfectly well see. They are different subsystems, and this system now
reflects that.

WHAT IS BORROWED AND WHAT IS NOT. The generator is a pretrained instruct model
(Qwen2.5-0.5B by default) and is FROZEN, exactly like the sensory encoders. It
supplies grammar and fluency -- things it would be absurd to relearn from
scratch on one laptop. It supplies NO perception: everything it is told about
the input comes from the lobes.

That division is the point, and it is testable. If the lobes report "cuirass",
the reply discusses armour. If the lobes are wrong, the reply is wrong in the
same way -- the language module cannot see the image and cannot correct them.
`explain.py` stage 7 checks exactly this.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

import config


@dataclass
class Perception:
    """What the lobes concluded, in the form the speech module is told it.

    Deliberately a small structured record rather than free text: it makes the
    boundary between perceiving and speaking inspectable, and it is what the
    dashboard shows when you ask why the system said something.
    """

    modalities: list[str] = field(default_factory=list)
    lobes_active: list[str] = field(default_factory=list)
    domain: str = ""
    #  (label, probability), strongest first
    predictions: list[tuple[str, float]] = field(default_factory=list)
    reference: list[tuple[str, float]] = field(default_factory=list)
    gate: float = 1.0
    allowed: bool = True
    user_text: str = ""

    def as_context(self) -> str:
        """The perception, written out for the language model to speak from."""
        if not self.predictions:
            #  Said in the system's own voice so the language model has
            #  something definite to relay. "Reported nothing" reads as a
            #  malfunction; not recognising an input is a normal outcome for
            #  a model with a 130-label world. One is a bug report, the
            #  other is an answer, and they should not sound alike.
            #  A STATEMENT, not an instruction. This text is handed to the
            #  language model as context, and an instruction buried in it
            #  competes with the system prompt: told here to "say so plainly",
            #  a grounded model refused to answer "1+1" and reported the
            #  non-recognition instead of doing the arithmetic. What to DO
            #  about a non-recognition belongs in the system prompt, which
            #  knows whether the model is blindfolded.
            return ("The cortex did not recognise this input -- it matched "
                    "nothing in the 130 categories it was trained on.")

        lines = []
        source = {
            "image": "an image", "text": "some text",
            "audio": "a sound", "sensor": "proximity sensors",
        }
        seen = [source.get(m, m) for m in self.modalities]
        lines.append(f"Input received: {', '.join(seen)}.")

        reading = {
            "imagenet": "The vision pathway identified the subject as",
            "topic": "The language pathway judged the subject matter to be",
            "acoustic": "The auditory pathway classified the sound as",
            "action": "The navigation pathway chose the move",
        }.get(self.domain, "The system concluded")

        top, confidence = self.predictions[0]
        if top.startswith("("):
            lines.append("The language pathway found no recognisable subject "
                         "matter in the input.")
        else:
            # Confidence goes in as a WORD, never a figure. Handing the model a
            # number invites it to restate one: a reported 55% came back out of
            # the model as "95% confidence" before this change.
            strength = ("confidently" if confidence >= 0.6 else
                        "with moderate confidence" if confidence >= 0.3 else
                        "only tentatively")
            lines.append(f"{reading} '{top}', {strength}.")
            others = [f"'{l}'" for l, p in self.predictions[1:3]
                      if not l.startswith("(")]
            if others:
                lines.append(f"It also considered {', '.join(others)}.")

        #  The nearest concepts in the shared semantic space, which is a
        #  richer reading than the classifier's label and often the more
        #  informative one: the label set has 20 topics, the semantic space
        #  has over a thousand concepts. Passing only the label throws away
        #  most of what the system actually worked out.
        nearby = [name for name, _ in self.reference[:3]]
        if nearby:
            lines.append("The semantic space placed it nearest to: %s."
                         % ", ".join(nearby))

        if not self.allowed:
            lines.append("The basal ganglia gate SUPPRESSED this decision, so "
                         "it should be treated as unreliable.")
        return " ".join(lines)

    #  -- the cortex speaking for itself ---------------------------------
    def as_answer(self) -> str:
        """The four lobes' OWN answer, composed with no language model at all.

        WHY THIS EXISTS. Every reply the system produced went out through a
        frozen Qwen, and with grounded chat switched on Qwen also read the
        user's raw text -- so on a conversational turn the borrowed model was
        doing nearly all the visible work while the four cortical networks
        contributed a footnote. The perception was real and measurable the
        whole time; it simply had no way to reach the screen except through a
        mouth that belonged to somebody else.

        So this renders the cortical reading directly. Nothing below consults
        a language model, and nothing below can invent a fact: every clause is
        a formatting of a number the lobes produced. It is less fluent than
        Qwen, and that is the correct trade -- what it says is exactly what
        the system perceived, no more and no less.

        The two outputs are shown together rather than one replacing the
        other, because they answer different questions: this one answers
        "what did the brain conclude", and the spoken reply answers "what does
        that mean in conversation".
        """
        heard = {"image": "an image", "text": "text", "audio": "a sound",
                 "video": "a video clip", "sensor": "sensor readings"}
        saw = ", ".join(heard.get(m, m) for m in self.modalities) or "nothing"

        if not self.predictions or self.predictions[0][0].startswith("("):
            return ("I took in %s and do not recognise it. It matched nothing "
                    "in the categories these lobes were trained on." % saw)

        label, confidence = self.predictions[0]
        pathway = {
            "imagenet": "The visual pathway",
            "topic": "The language pathway",
            "acoustic": "The auditory pathway",
            "action": "The navigation pathway",
        }.get(self.domain, "The cortex")
        strength = ("confidently" if confidence >= 0.6 else
                    "with moderate confidence" if confidence >= 0.3 else
                    "only tentatively")

        parts = ["I took in %s. %s reads it as '%s', %s."
                 % (saw, pathway, label, strength)]

        alternatives = [l for l, _ in self.predictions[1:3]
                        if not l.startswith("(")]
        if alternatives:
            parts.append("Runners-up: %s." % ", ".join(alternatives))

        nearby = [name for name, _ in self.reference[:4]]
        if nearby:
            parts.append("Nearest concepts in my semantic space: %s."
                         % ", ".join(nearby))

        lobes = [l for l in self.lobes_active if l in config.LOBES]
        if lobes:
            parts.append("%d of 4 lobes contributed (%s)."
                         % (len(lobes), ", ".join(lobes)))

        parts.append("The basal ganglia %s this decision."
                     % ("passed" if self.allowed else "SUPPRESSED"))
        return " ".join(parts)


SYSTEM_PROMPT = (
    "You are the language centre of a modular brain-inspired AI. Four "
    "cortical modules handle perception and hand you their conclusions; you "
    "put those conclusions into words.\n\n"
    "Rules you must follow:\n"
    "- Speak only from the perceptual report you are given. You cannot see, "
    "hear or sense anything yourself.\n"
    "- NEVER state a number, percentage or confidence value. Any figure you "
    "produce is invented. Use words like 'confidently' or 'tentatively'.\n"
    "- If the report is uncertain or was suppressed, say so plainly instead of "
    "guessing.\n"
    "- Do not invent details that are not in the report, and do not speculate "
    "about what else the subject might be.\n"
    "- NEVER say the conclusion is correct, accurate or right. You cannot "
    "see the input, so you have no way to know whether it is. Report it as "
    "what the system concluded, not as what is true: 'the system read this "
    "as X', never 'X is correct' and never 'this accurately reflects the "
    "input'.\n"
    "- If the report says nothing was recognised, say plainly that you do "
    "not recognise it. Do NOT reach for the nearest topic label anyway.\n"
    "- Be conversational and brief: two or three sentences unless asked for "
    "more."
)


GROUNDED_PROMPT = (
    "You are the language centre of a brain-inspired AI. Four cortical "
    "modules perceive the input and hand you what they made of it; you hold "
    "the conversation.\n\n"
    "You can read the user's message directly, AND you are given a perceptual "
    "report from the cortex. Use both:\n"
    "- Answer the user properly and helpfully, in your own words. If they ask "
    "a question, answer it.\n"
    "- The perceptual report tells you what the cortex made of the input. "
    "Where it is relevant, use it. Where it is not, ignore it and just "
    "answer.\n"
    "- The report is CONTEXT, not the question. Never simply relay it. If the "
    "user asked something, answer the question.\n"
    "- NEVER claim the cortex perceived something the report does not say. "
    "When the report says the input was not recognised, that is a fact about "
    "the cortex, NOT a reason to refuse: answer the user from your own "
    "knowledge, and mention the cortex only if they asked about it.\n"
    "- NEVER state a number, percentage or confidence value from the report. "
    "Use words like 'confidently' or 'tentatively'.\n"
    "- NEVER mention the report, the cortex or any internal machinery to the "
    "user unless they asked about the system itself. They asked a question; "
    "answer it. Saying 'given the context provided by the perceptual "
    "report' is plumbing leaking into the conversation.\n"
    "- If asked what you are: you are a brain-inspired AI with four cortical "
    "modules (occipital, parietal, temporal, frontal), a thalamus, basal "
    "ganglia, hippocampus and cerebellum, which perceive the input, plus a "
    "language module that talks. Say that plainly rather than asking for "
    "more details.\n"
    "- Be conversational and concise. Two to four sentences unless more is "
    "genuinely needed."
)


class LanguageModule(nn.Module):
    """Frozen instruct model that verbalises what the lobes perceived."""

    def __init__(self, model_name: str = config.LANGUAGE_MODEL,
                 device: str | None = None):
        super().__init__()
        self.model_name = model_name
        self.device = device or config.resolve_device()
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    # -- lazy load ------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Half precision on the GPU: a 0.5B model is ~2 GB in fp32 and ~1 GB in
        # fp16, which matters on a 6 GB card that is also holding three vision
        # and audio backbones. CPU stays fp32 -- fp16 on CPU is slower, not
        # faster, because there is no hardware for it.
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        self._model = model.to(self.device)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def parameter_count(self) -> int:
        self._ensure_loaded()
        return sum(p.numel() for p in self._model.parameters())

    # -- generation -----------------------------------------------------
    def respond(
        self,
        perception: Perception,
        history: list[tuple[str, str]] | None = None,
        max_new_tokens: int = config.LANGUAGE_MAX_TOKENS,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Turn a perceptual report (plus the user's message) into a reply.

        `on_token` is called with each new fragment as it is produced, so the
        dashboard can stream the answer instead of freezing until it is done.
        """
        self._ensure_loaded()

        #  Which prompt, and therefore whether the model is blindfolded, is a
        #  single configuration decision -- see config.GROUNDED_CHAT for what
        #  it costs and what it buys.
        grounded = config.GROUNDED_CHAT
        messages = [{"role": "system",
                     "content": GROUNDED_PROMPT if grounded else SYSTEM_PROMPT}]
        for who, said in (history or [])[-config.LANGUAGE_HISTORY_TURNS:]:
            messages.append({"role": who, "content": said})

        report = perception.as_context()
        if perception.user_text:
            user = (f"[perceptual report: {report}]\n\n"
                    f"{perception.user_text}")
        else:
            user = (f"[perceptual report: {report}]\n\n"
                    f"Tell me what was just perceived.")
        messages.append({"role": "user", "content": user})

        prompt = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        encoded = self._tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # One generation at a time: the model is shared, and transformers'
        # generate is not safe to run concurrently on the same module.
        with self._lock:
            if on_token is None:
                with torch.no_grad():
                    out = self._model.generate(
                        **encoded, max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
                return self._tokenizer.decode(
                    out[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()

            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(
                self._tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            kwargs = dict(
                **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id, streamer=streamer,
            )
            worker = threading.Thread(
                target=lambda: torch.no_grad()(self._model.generate)(**kwargs),
                daemon=True,
            )
            worker.start()

            pieces: list[str] = []
            for piece in streamer:
                if piece:
                    pieces.append(piece)
                    on_token(piece)
            worker.join()
            return "".join(pieces).strip()


def perception_from_trace(trace, user_text: str = "") -> Perception:
    """Build the report the language module speaks from, out of a BrainTrace."""
    answer = trace.answer
    return Perception(
        modalities=list(trace.modalities),
        lobes_active=[n for n in trace.active_nodes if n in config.LOBES],
        domain=trace.domain,
        predictions=[(p.label, p.probability) for p in (answer.predictions if answer else [])],
        reference=[(p.label, p.probability) for p in (answer.reference if answer else [])],
        gate=trace.gate.gate if trace.gate else 1.0,
        allowed=trace.gate.allowed if trace.gate else True,
        user_text=user_text,
    )


__all__ = ["LanguageModule", "Perception", "perception_from_trace", "SYSTEM_PROMPT"]
