# Genuine guarded VLM browser mission

The actual native Qwen3-VL-2B runtime, browser controls, fresh-eye action gate,
MuJoCo physics and Three.js renderer completed “Fetch the red ball, then hop,
then stop.” No HTTP/WebSocket model response fixture was supplied.

- Runtime source: `8ef5721cbdefe36f7f575ca0d721d9d74f66fdc0`, clean when started;
  exact [harness](harness.mjs) hash is in the unchanged [result](result.json).
- Mission duration: **403.790 wall seconds**, excluding 6.005 seconds of setup.
  Page-start to first observed completion was 409.637 seconds.
- Actual model calls: **161.999, 110.580, 104.788 seconds**. This CPU BF16 run
  shared the machine with a 1440px Chromium SwiftShader recording; it is not an
  isolated inference speed test or a GPU interactive-latency measurement.
- Contact and held-to-released transitions were independently observed. Maximum
  held ball height was **0.3621 m**. At fetch completion, the released ball was
  grounded **0.3901 m** from home, before the later hop.
- Hop measured **0.28 seconds flight**, then four feet in contact and upright
  alignment above 0.9999. All three mission steps completed without world reset.
- Final ball distance after hop: **0.5531 m**. The fetch assertion deliberately
  applies at fetch completion; later physical motion is reported separately.
- Physics continued standing during inference, unlike the earlier headless
  timing experiments. Final simulation time was 357.32 seconds, including setup,
  thinking and active skills; it is not a 357-second movement duration.
- Recording finalization took another **147.207 seconds**. Total harness wall
  time including setup, screenshots and encoding was **558.902 seconds**.

[Full actual recording](actual-run.webm) retains original CDP frame timestamps,
without cropping, skipping waiting or speeding up motion. Frame timestamps and
FFmpeg concat durations are included; JPEG source frames remain under ignored
`artifacts/vlm-browser-guarded/capture/`. [Video metadata](video-metadata.json)
records the encoded stream. Screenshots show thinking, carrying and completion.

This verifies one canonical guarded mission, not broad VLM visual competence.
The unchanged model previously failed real/blank/turned conditional image probes;
[those raw failures](../gpu-4090/) and the earlier failed
[browser harness attempt](../browser-attempt1/) remain published. The activation
guard is limited to this colored arena. No human voice, mobile FPS, concurrent
VLM capacity or production Vercel deployment is established by this run.
