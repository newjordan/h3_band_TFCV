"""Knight Shift: look inside and steer MiniMax-H3's attention, block by block.

KSH3AttnProbe  - for chosen blocks and denoising steps, measures per head how much attention a sample of target video
                 tokens puts on each packed segment (text / cond / ref_img / ref_audio / audio / video). Writes JSON.
KSH3RefPull    - "KV pull": for chosen blocks and heads, target video queries see the reference-image (or cond) keys
                 with an additive logit bias of log(m) (the reference K/V appended m-1 more times), so the
                 appearance heads draw harder on the reference look while the ControlNet/plate keep the geometry.

Both are dit double_block patches that hand the block a custom attention callable (DiTBlock.forward takes
`attention=`); they chain any patch already registered on that block (e.g. the Fun ControlNet).
"""
import json, logging, math, os

import torch
import torch.nn.functional as F

import comfy.quant_ops


def _qkv(attn, x, rope_freqs):
    """The block's own q, k, v (heads, S, d), exactly as Attention.forward computes them."""
    s = x.shape[0]
    q, k, v = attn.qkv_proj(x).split(attn.heads * attn.head_dim, dim=-1)
    v = v.view(s, attn.heads, attn.head_dim)
    q = q.reshape(1, s, attn.heads, attn.head_dim).contiguous()
    k = k.reshape(1, s, attn.heads, attn.head_dim).contiguous()
    import comfy.model_management as mm
    qw = mm.cast_to(attn.q_norm.weight, device=x.device)
    kw = mm.cast_to(attn.k_norm.weight, device=x.device)
    rot = rope_freqs.shape[-3] * 2
    q, k = comfy.quant_ops.ck.rms_rope_split_half(q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
    return q[0].transpose(0, 1), k[0].transpose(0, 1), v.transpose(0, 1)        # (H, S, d)


def _segments(layout):
    out = {}
    for a, b, kind in layout.segments:
        out.setdefault(kind, []).append((a, b))
    return out


def _rows(segs, kinds, device):
    idx = [torch.arange(a, b) for k in kinds for a, b in segs.get(k, [])]
    return torch.cat(idx).to(device) if idx else None


class _ProbeAttention:
    def __init__(self, attn, layout, rec, n_query=384, seed=0):
        self.attn, self.layout, self.rec, self.nq, self.seed = attn, layout, rec, n_query, seed

    def __call__(self, x, rope_freqs=None, transformer_options={}):
        out = self.attn(x, rope_freqs=rope_freqs, transformer_options=transformer_options)   # unchanged result
        with torch.no_grad():
            q, k, v = _qkv(self.attn, x, rope_freqs)
            segs = _segments(self.layout)
            vid = _rows(segs, ["video"], x.device)
            g = torch.Generator(device="cpu").manual_seed(self.seed)
            sel = vid[torch.randperm(len(vid), generator=g)[: self.nq].to(x.device)]
            scale = 1.0 / math.sqrt(q.shape[-1])
            mass = {}
            for h in range(q.shape[0]):
                p = torch.softmax((q[h, sel].float() @ k[h].float().T) * scale, dim=-1)   # (nq, S)
                for kind, spans in segs.items():
                    m = sum(p[:, a:b].sum(-1) for a, b in spans)
                    mass.setdefault(kind, []).append(float(m.mean()))
            self.rec.append(mass)
        return out


class _PullAttention:
    def __init__(self, attn, layout, heads, mult, kinds=("ref_img",)):
        self.attn, self.layout, self.heads, self.mult, self.kinds = attn, layout, heads, mult, kinds

    def __call__(self, x, rope_freqs=None, transformer_options={}):
        q, k, v = _qkv(self.attn, x, rope_freqs)
        H = q.shape[0]
        segs = _segments(self.layout)
        ref = _rows(segs, list(self.kinds), x.device)
        pull = [h for h in self.heads if h < H]
        rest = [h for h in range(H) if h not in pull]
        o = torch.empty_like(q)
        if rest:
            o[rest] = F.scaled_dot_product_attention(q[rest].unsqueeze(0), k[rest].unsqueeze(0), v[rest].unsqueeze(0))[0]
        if pull and ref is not None and self.mult > 1:
            kr = k[pull][:, ref].repeat(1, self.mult - 1, 1)
            vr = v[pull][:, ref].repeat(1, self.mult - 1, 1)
            o[pull] = F.scaled_dot_product_attention(q[pull].unsqueeze(0), torch.cat([k[pull], kr], 1).unsqueeze(0),
                                                     torch.cat([v[pull], vr], 1).unsqueeze(0))[0]
        elif pull:
            o[pull] = F.scaled_dot_product_attention(q[pull].unsqueeze(0), k[pull].unsqueeze(0), v[pull].unsqueeze(0))[0]
        return self.attn.out_proj(o.transpose(0, 1).reshape(x.shape[0], -1).to(x.dtype))


class _BlockPatch:
    def __init__(self, dm, block_index, make, previous, steps=None):
        self.dm, self.i, self.make, self.previous, self.steps = dm, block_index, make, previous, steps

    def __call__(self, args, extra_args):
        to = args["transformer_options"]
        step_ok = True
        if self.steps is not None:
            sig = to.get("sigmas")
            allsig = to.get("sample_sigmas")
            if sig is not None and allsig is not None:
                cur = float(sig.flatten()[0]); ss = [float(s) for s in allsig.flatten()]
                si = min(range(len(ss)), key=lambda j: abs(ss[j] - cur))
                step_ok = si in self.steps
        if step_ok:
            args = dict(args)
            args["attention"] = self.make(self.dm.blocks[self.i].attn, args["layout"])
        if self.previous is None:
            return extra_args["original_block"](args)
        return self.previous(args, extra_args)


def _parse(spec, n):
    if spec.strip() in ("", "all"):
        return list(range(n))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def _register(model, blocks, make, steps=None):
    m = model.clone()
    dm = m.get_model_object("diffusion_model")
    for i in blocks:
        prev = m.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {}).get(("double_block", i))
        m.set_model_patch_replace(_BlockPatch(dm, i, make, prev, steps), "dit", "double_block", i)
    return m


class KSH3AttnProbe:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "blocks": ("STRING", {"default": "0,6,12,18,24,30,36,42,49"}),
                             "steps": ("STRING", {"default": "0,5,10,15"}), "out_path": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"
    CATEGORY = "ks/minimax"

    def run(self, model, blocks, steps, out_path):
        nb = len(model.get_model_object("diffusion_model").blocks)
        bl = _parse(blocks, nb); st = set(_parse(steps, 1000))
        rec = {}

        def make_for(i):
            def make(attn, layout):
                return _ProbeAttention(attn, layout, rec.setdefault(i, []))
            return make
        m = model.clone()
        dm = m.get_model_object("diffusion_model")
        for i in bl:
            prev = m.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {}).get(("double_block", i))
            m.set_model_patch_replace(_BlockPatch(dm, i, make_for(i), prev, st), "dit", "double_block", i)

        def dump():
            if out_path:
                json.dump({str(k): v for k, v in rec.items()}, open(out_path, "w"))
        import atexit
        atexit.register(dump)
        m.model_options.setdefault("ks_probe_dump", dump)
        KSH3AttnProbe._last = (rec, out_path)
        return (m,)


class KSH3AttnProbeDump:
    """Writes the probe's records after sampling (wire its 'latent' input from the sampler output)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "ks/minimax"

    def run(self, latent):
        rec, path = getattr(KSH3AttnProbe, "_last", ({}, ""))
        if path:
            json.dump({str(k): v for k, v in rec.items()}, open(path, "w"))
            logging.info("[KSH3AttnProbe] wrote %s (%d blocks)", path, len(rec))
        return (latent,)


class KSH3RefPull:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "blocks": ("STRING", {"default": "all"}),
                             "heads": ("STRING", {"default": "all"}), "mult": ("INT", {"default": 3, "min": 1, "max": 32}),
                             "segments": ("STRING", {"default": "ref_img"}), "steps": ("STRING", {"default": "all"})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"
    CATEGORY = "ks/minimax"

    def run(self, model, blocks, heads, mult, segments, steps):
        dm = model.get_model_object("diffusion_model")
        bl = _parse(blocks, len(dm.blocks)); hd = _parse(heads, dm.blocks[0].attn.heads)
        kinds = tuple(s.strip() for s in segments.split(",") if s.strip())
        st = None if steps.strip() in ("", "all") else set(_parse(steps, 1000))
        logging.info("[KSH3RefPull] blocks %s heads %d mult %d (bias %.2f) segments %s", bl, len(hd), mult, math.log(mult), kinds)
        return (_register(model, bl, lambda attn, layout: _PullAttention(attn, layout, hd, mult, kinds), st),)


NODE_CLASS_MAPPINGS = {"KSH3AttnProbe": KSH3AttnProbe, "KSH3AttnProbeDump": KSH3AttnProbeDump, "KSH3RefPull": KSH3RefPull}
NODE_DISPLAY_NAME_MAPPINGS = {"KSH3AttnProbe": "KS H3 attention probe", "KSH3AttnProbeDump": "KS H3 attention probe: write",
                              "KSH3RefPull": "KS H3 reference KV pull"}
