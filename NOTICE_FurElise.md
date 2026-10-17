`blockout/rig/furelise_cal.json` and `blockout/rig/comfort.json` are fitted from the FürElise dataset
(R. Wang, P. Xu, H. Shi, E. Schumann, C. K. Liu, "FürElise: Capturing and Physically Synthesizing Hand Motions of Piano
Performance", SIGGRAPH Asia 2024; https://for-elise.github.io/, https://huggingface.co/datasets/rcwang/for_elise),
licensed CC BY-NC 4.0: non-commercial use, with attribution. The dataset itself is not included; `blockout/furelise/`
reads it from a local copy. `RIG_NO_FURELISE=1` runs the rig without these values (the MediaPipe-based
`comfort_mediapipe.json`).
