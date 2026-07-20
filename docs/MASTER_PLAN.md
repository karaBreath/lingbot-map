# LingBot-Map Plus — มาสเตอร์แปลน (แม่แบบสำหรับโมเดลผู้เขียนโค้ด)

> **สถานะ (2026-07-19): เฟส 0 เสร็จสมบูรณ์ 100% — ยืนยันด้วยภาพจริงแล้ว.** `demo.py --backend directml` รันจบ end-to-end ด้วย **weights จริง + ภาพจริง** (example/courthouse, 0 missing/unexpected keys), depth/confidence สมเหตุสมผล (ไม่มี NaN), point cloud เพิ่มเข้า scene จริง (13960/13804/8828 จุด/เฟรม) **และเห็นด้วยตาแล้วผ่าน `ui-use` (Playwright จริง) — โครงตึกศาลากลาง (courthouse) ขึ้นชัดเจนใน viser viewer, GUI ครบทุกปุ่ม** สกรีนช็อตอยู่ที่ `C:\Users\SSD\AppData\Local\Temp\lingbot_view1.png` (ส่งให้ผู้ใช้แล้ว) — Claude_Browser (เครื่องมือ automation ในเซสชัน) capture หน้านี้ไม่ได้เพราะ "Playing" checkbox default เปิด ทำให้ canvas วนสลับเฟรมทุก 50ms ไม่มีจังหวะนิ่งให้ capture (ไม่ใช่บั๊กโค้ด) — **ใช้ `ui-use` แทน Claude_Browser สำหรับตรวจ visual ของหน้านี้ในงานถัดไป** ดู §2.4 สำหรับบั๊ก DirectML 7 ตัวที่เจอและแก้แล้ว
> **ผู้อ่านเป้าหมาย:** AI/นักพัฒนาที่จะเขียนโค้ด — อ่านเอกสารนี้จบแล้วต้องลงมือได้โดยไม่ต้องถามย้อน
> **กฎเหล็ก:** ทำทีละเฟสเป็น vertical slice, ทุกเฟสต้องมีหลักฐานรันจริงก่อนขึ้นเฟสถัดไป, commit แยกเฟส, ห้าม hardcode path/secret (ใช้ .env)

---

## 1. เป้าหมาย

**ประโยคเดียว:** เปลี่ยน LingBot-Map (โค้ดวิจัย 3D reconstruction ที่ต้อง NVIDIA + CLI) ให้เป็น "เครื่องมือสแกนห้อง 3D" ที่พนักงานถือมือถือเดินถ่ายวิดีโอ 1 นาที แล้วได้ลิงก์ 3D ส่งลูกค้า/แนบงานตรวจ — รันบนเครื่อง AMD ที่มีอยู่ ไม่เช่า cloud

**Use case ธุรกิจ (เรียงความสำคัญ):**
1. สแกนห้องขาย/เช่าคอนโด-โรงแรม → ลูกค้าหมุนดูห้องจริงจากมือถือ (ต่อยอด condo sales bot, เว็บ Hybritel)
2. ก่อน-หลัง เทียบ 2 สแกน → ตรวจรับงานผู้รับเหมา, room progress report
3. แนบสแกน 3D กับใบแจ้งซ่อม → ช่างเห็นหน้างานก่อนเข้า (ระบบ facility maintenance :3200/:3201)

---

## 2. Ground Truth (ตรวจแล้วจากเครื่องจริง + โค้ดจริง — อย่าเดาซ้ำ)

### 2.1 ฮาร์ดแวร์/ซอฟต์แวร์เครื่องนี้ (niti, Windows 11)
| ของ | ค่าจริง |
|---|---|
| GPU | AMD Radeon RX 9070 XT (RDNA4, gfx1201, VRAM 16GB — Windows API รายงานผิดเป็น 4GB) |
| GPU สำรอง | Intel iGPU (Arrow Lake) — ไม่ใช้ |
| NVIDIA | **ไม่มี** — CUDA จริงใช้ไม่ได้บนเครื่องนี้ |
| CPU / RAM | Intel Core Ultra 7 265K / 31.4GB |
| Python ระบบ | 3.14.5 — **ต้องสร้าง venv Python เวอร์ชันที่ torch รองรับ** (upstream ใช้ 3.10) |
| WSL | Ubuntu (running), `/dev/dxg` มี = GPU ทะลุเข้า WSL ได้ แต่ยังไม่ติด ROCm/torch |
| Repo clone | `d:\cowork\lingbot-map` (upstream: github.com/Robbyant/lingbot-map, clone แบบ blob-limit 500k) |

### 2.2 ข้อค้นพบสำคัญในโค้ด upstream (อ่านโค้ดจริงแล้ว)
1. **แกนโมเดลไม่ผูก CUDA** — `demo.py` เลือก device ด้วย `torch.cuda.is_available() else "cpu"` (demo.py:421) ไม่มี `.cuda()` hardcode ในเส้นทางหลัก
2. **FlashInfer เป็น optional อยู่แล้ว** — import ใน try/except (`lingbot_map/layers/attention.py:26-30`) และมี flag `use_sdpa` สลับไป PyTorch SDPA มาตรฐาน (`demo.py:382`, `gct_stream.py:125`, `gct_stream_window.py:166`, `gct_stream_window_v2.py:229`, `aggregator/stream.py:87-93`) → **ทางหนี CUDA มีในตัวแล้ว แค่บังคับใช้**
3. **Kaolin ไม่ใช่ dependency ของแกนหลัก** — ใช้เฉพาะ `demo_render/` (เรนเดอร์วิดีโอออฟไลน์คุณภาพสูง + `render_cuda_ext` ที่เป็น CUDA extension) → เฟสแรกตัดทิ้งทั้งโฟลเดอร์ได้
4. **จุด hardcode CUDA ใน demo.py ที่ต้องแก้ (มี 4 กลุ่ม):**
   - `demo.py:421` — device selection (มีแค่ cuda/cpu)
   - `demo.py:448-451` — dtype (cuda→bf16/fp16, อื่น→fp32) — logic โอเคแต่ต้องรู้จัก backend ใหม่
   - `demo.py:222,234,545` — `torch.amp.autocast("cuda", ...)` hardcode สตริง "cuda" — DirectML/CPU ต้องปิด autocast
   - `demo.py:38-39` — `PYTORCH_CUDA_ALLOC_CONF` env (ไม่มีผลบน non-CUDA, ปล่อยได้)
   - `demo.py:384-386` — `--compile` ใช้ CUDA graphs → ต้อง disable เมื่อ backend ไม่ใช่ cuda/rocm
5. **GLB export มีแล้ว** — `lingbot_map/vis/glb_export.py` + ปุ่มใน viewer (`point_cloud_viewer.py:624 _export_glb`) → เฟส 2 เรียกใช้ตรงๆ ไม่ต้องเขียนใหม่
6. **Viewer เดิม** = viser web server ที่ localhost:8080 — เหมาะ dev แต่ให้ลูกค้าดูไม่ได้ (ต้องรัน Python) → ต้องมีชั้น static viewer แยก
7. **หลักฐานว่ารันไม่มี NVIDIA ได้จริง:** Mac fork (github.com/donalleniii/lingbot-desktop-mac) รันบน MPS ได้ ~0.3–1 FPS ไม่ลง FlashInfer/Kaolin เลย → เครื่อง AMD คาดหวังระดับเดียวกันหรือดีกว่า
8. **Weights:** HuggingFace `robbyant/lingbot-map` (ไฟล์ `.pt` เช่น `lingbot-map-long.pt`) — **ยังไม่รู้ขนาดไฟล์ ต้องเช็คก่อนโหลดและแจ้งผู้ใช้**

### 2.3 ทางรัน AMD — เรียงความชัวร์
| ทาง | stack | ความชัวร์ | หมายเหตุ |
|---|---|---|---|
| `directml` | Windows + torch-directml | สูง (ตัวหลักเฟส 0) | ⚠️ torch-directml มัก lag เวอร์ชัน torch — ตรวจว่าเวอร์ชันล่าสุดรองรับ API ที่โค้ดใช้ ถ้าไม่ผ่านให้ fallback CPU ก่อน แล้วรายงาน |
| `cpu` | torch CPU ปกติ (เวอร์ชันใหม่ได้) | สูงสุด | ช้าสุด — ใช้เป็น fallback + ตัวพิสูจน์ correctness |
| `rocm` | WSL Ubuntu + ROCm | กลาง-ต่ำ (RDNA4 บน WSL ยังใหม่) | ถ้าติดได้ = เร็วสุด เพราะ PyTorch ROCm โผล่เป็น `torch.cuda` → โค้ด upstream แทบไม่ต้องแก้ แค่ `--use_sdpa` |
| `cuda` | NVIDIA | — | คงไว้ให้เครื่องอื่น/อนาคต ห้ามทำพัง |

---

### 2.4 บั๊ก DirectML ที่เจอจริง + แก้แล้ว (เฟส 0, 2026-07-19) — อ่านก่อนแตะ backend directml ซ้ำ

พบผ่านการรัน smoke test จริงบน RX 9070 XT (venv `.venv-amd`, torch==2.4.1+cpu + torch-directml 0.2.5.dev240914) เรียงตามลำดับที่เจอ:

1. **`torch.nn.attention.flex_attention` ไม่มีใน torch<2.5** (torch-directml ล็อกที่ 2.4.1) — `lingbot_map/layers/block.py` import module นี้แบบ unconditional ที่ระดับบนไฟล์ ทั้งที่ใช้จริงแค่ใน `_prepare_blockwise_causal_attn_mask` ซึ่ง**ไม่ถูกเรียกเลยตอน streaming** (มี comment ในโค้ดเองยืนยัน: "Skip mask creation when using KV cache"). **แก้:** ย้าย import เข้าไปใน method นั้น (lazy import) — ไม่กระทบ CUDA
2. **`torch.cat` บน DirectML รับ tensor ที่มีความยาว 0 ในมิติที่ cat ไม่ได้** ("The parameter is incorrect.") — เกิดใน `lingbot_map/aggregator/base.py:slice_expand_and_flatten` เมื่อ `S == first_num_frame` ซึ่ง**เป็นสถานะปกติของทุกรัน** (demo.py's Phase 1 เรียกด้วย `num_frame_for_scale == num_frame_per_block` เสมอ ทำให้ token_rest ว่างเปล่า) **แก้:** เช็คว่าฝั่งใดว่างแล้วข้าม cat (คืนอีกฝั่งตรงๆ) — ไม่กระทบ backend อื่น
3. **complex128 (`ComplexDouble`) ทำ DirectML process abort ทันที** — ไม่ใช่ catchable exception (log level `F`=FATAL แบบ glog, เท่ากับ `abort()`) เกิดใน `lingbot_map/layers/rope.py:WanRotaryPosEmbed` ที่เก็บ `self.freqs` เป็น complex tensor แล้ว `.to(device)`. **แก้:** เพิ่ม `_RealImagView` wrapper — เช็ค `device.type in ("cpu","cuda")` (static, ไม่เรียก op เสี่ยง) ถ้าไม่ใช่ ให้คำนวณ complex ทั้งหมดบน CPU แล้วส่งแค่ `.real`/`.imag` (float ธรรมดา) ไป device ตอนจบ — `apply_rotary_emb` ไม่ต้องแก้เพราะอ่านแค่ `.real`/`.imag`
4. **float64 ไม่รองรับบน DirectML** (เหมือน MPS) — `lingbot_map/heads/utils.py:make_sincos_pos_embed` มี guard `device.type=="mps"` อยู่แล้วแต่ไม่ครอบ `"privateuseone"` (ชื่อ device type ของ DirectML) **แก้:** ขยาย guard เป็น `supports_double = device.type not in ("mps","privateuseone")`
5. **`torch.eye(4, 4, device=...)` (2-arg overload) บน DirectML คืน tensor ว่าง `[0]` เงียบๆ ไม่ error** (bug ใน torch_directml's CPU-fallback shim สำหรับ `aten::eye.m_out`) — เกิดใน `lingbot_map/utils/geometry.py:closed_form_inverse_se3_general` **แก้:** สร้าง `torch.eye(4,4)` บน CPU (ไม่ใส่ device) แล้ว `.to(R.device)` — pattern เดียวกับที่ `closed_form_inverse_se3` (ฟังก์ชันพี่น้องบรรทัดบน) ใช้อยู่แล้ว
6. **`matplotlib.cm.get_cmap()` ถูกถอดใน matplotlib≥3.9** — ไม่เกี่ยวกับ DirectML เป็น version drift ธรรมดา (`requirements-amd.txt` ลืมพิน matplotlib และ pip ดึงรุ่นล่าสุดมา) เจอ 3 จุด (`vis/utils.py` ×2, `vis/point_cloud_viewer.py` ×1) ในขณะที่อีก 2 จุด (`point_cloud_viewer.py:741`, `glb_export.py:168`) ใช้ API ใหม่ `matplotlib.colormaps.get_cmap()` อยู่แล้ว **แก้:** เปลี่ยน 3 จุดที่เหลือให้ตรงกัน + เพิ่ม `import matplotlib` (เดิมมีแค่ `import matplotlib.cm as cm` ซึ่งไม่ผูกชื่อ `matplotlib`)

7. **`torch.load(..., map_location=device)` พังเมื่อ device เป็น `torch_directml` object** — `TypeError: '>=' not supported between instances of 'torch.device' and 'int'` (torch's generic `map_location` dispatch เรียก `torch_directml.device(device_id)` โดยคาดว่า `device_id` เป็น int แต่ได้ device object) **แก้:** เปลี่ยน `demo.py:load_model` เป็น `map_location="cpu"` เสมอ แล้วให้ `model.to(device)` ที่มีอยู่แล้วจัดการย้าย — เป็น best practice ทั่วไปอยู่แล้ว (ลด GPU memory spike ตอน deserialize) ไม่กระทบ backend อื่น

### 2.5 บทเรียนเรื่อง visual verification tooling

เครื่องมือ Claude_Browser (automation ในเซสชันแรกที่ลองรัน) capture screenshot ของหน้า viser ไม่ได้เลย (timeout 30s ทุกครั้ง ทั้ง full-page/zoom เล็ก/หลาย tab/port) ทั้งที่ DOM/WebGL context/console ปกติหมด — สาเหตุจริง: **`gui_playing` checkbox ใน `animate()` (point_cloud_viewer.py) default เป็น `True`** ทำให้ canvas วนสลับเฟรมอัตโนมัติทุก 1/FPS วินาที (default FPS=20 = ทุก 50ms) ไม่มีจังหวะ "นิ่ง" ให้เครื่องมือ capture บางตัวจับภาพได้ **แก้ด้วยเครื่องมือ ไม่ใช่แก้โค้ด:** ใช้ `ui-use` (Playwright-based, มีในเครื่องนี้แล้ว) แทน — capture ผ่านได้ทันทีแม้ระหว่าง auto-play เพราะ Playwright capture ไม่รอจังหวะ "idle paint" แบบเดียวกัน

**ยืนยันแล้ว:** ทดสอบด้วย weights จริง + ภาพจริงจาก `example/courthouse` (3 เฟรม, image_size=518, num_scale_frames=2, --offload_to_cpu, port อิสระที่เช็คว่างก่อน — **อย่า hardcode port ตัวใดตัวหนึ่ง เช็ค `Get-NetTCPConnection -LocalPort <n> -State Listen` ก่อนเสมอ เคยชนกับ wslrelay/Docker ที่ port 8090 มาแล้ว**) — เห็นโครงตึกศาลากลางจริงในภาพ ยืนยันทั้ง pipeline ทำงานถูกต้อง

**ผลตรวจสอบ:** DirectML VRAM allocator ไม่มี caching เหมือน CUDA — alloc เดี่ยว 8GB ผ่านสบาย แต่ full forward ที่ image_size=518+num_scale_frames=4 ล้มด้วย "not enough video memory" ตอนขอ ~1.94GB ระหว่างรัน (สะสมจาก activation หลายก้อนไม่ถูกคืน) ส่วน image_size=308+num_scale_frames=2 ผ่านฉลุย — **ต้องจูน `--image_size`/`--num_scale_frames` ให้เล็กกว่าที่ README แนะนำสำหรับ CUDA** เมื่อรันบน DirectML จริงจะบันทึกค่าที่ใช้ได้ลง `docs/BENCHMARKS.md`

### 2.6 R0 — VRAM growth บนซีเควนซ์ยาว (root cause เจอแล้ว, 2026-07-19)

**อาการ:** วิดีโอ/ภาพชุดยาว (119 เฟรม, `example/loop`) พังด้วย "not enough video memory" เร็วมาก — streaming mode พังที่เฟรม 3-4, windowed mode พังที่ window 2 — เหมือนเป็น leak ตั้งแต่ช่วงแรก

**วินิจฉัยด้วย probe สคริปต์ (`scripts/vram_leak_probe.py`, วัด VRAM จริงผ่าน Windows `Get-Counter '\GPU Process Memory(*)\Local Usage'` ต่อ PID เพราะ `torch_directml.gpu_memory()` คืนศูนย์เสมอ ใช้ไม่ได้):**
- โมเดล (weights จริง 4.6GB) โหลดขึ้น device กิน **~5.0GB** ทันที (`model.to(device)`) — สูงมากเทียบกับ VRAM ที่ใช้งานได้จริงบนการ์ดนี้
- forward เดี่ยวๆ (2-7 สเต็ปแรก) **ไม่ leak** — VRAM นิ่งที่ ~7.2GB คงที่
- แต่ยิ่งสเต็ปมากขึ้น (ทดสอบถึง 20 เฟรม) พังที่สเต็ป ~8 ด้วยการขอ tensor **1.27GB ก้อนเดียว** ใน `scaled_dot_product_attention`

**Root cause:** ไม่ใช่ memory leak — เป็น**การเติบโตตามธรรมชาติของ KV cache** ระหว่าง streaming: ยิ่งประมวลผลไปหลายเฟรม (จนถึง `kv_cache_sliding_window`, default=64) ยิ่งมี token สะสมใน cache เยอะขึ้น → attention matrix (`scaled_dot_product_attention`) ใหญ่ขึ้นตาม O(n²) กับความยาว sequence สะสม → ขนาด tensor ที่ต้องขอในแต่ละสเต็ปโตขึ้นเรื่อยๆ จนเกิน headroom ที่เหลือ (~2GB header หลังโมเดลกิน 5GB จาก VRAM ที่ใช้ได้จริงบนการ์ดนี้ผ่าน DirectML) — CUDA รับมือกับพีคแบบนี้ได้เพราะมี caching allocator คืน/reuse หน่วยความจำเร็ว DirectML ไม่มีกลไกนี้เลย (ยืนยันแล้ว: `torch_directml` ไม่มีฟังก์ชัน empty_cache ทั้ง public/native module)

**ทางแก้ที่กำลังทดสอบ:** ลด `--kv_cache_sliding_window` (default 64) ให้เล็กลง (ลอง 16) — จำกัดเพดานว่า cache สะสมได้สูงสุดกี่เฟรม ตัดพีค attention memory ไม่ให้โตไม่หยุด (ราคาที่จ่าย: โมเดลจำบริบทเฟรมเก่าได้น้อยลง อาจกระทบความต่อเนื่อง/ความแม่นยำของฉากยาวๆ — ยังไม่วัดผลกระทบด้านคุณภาพ)

**นัยสำคัญสำหรับ R0:** วิดีโอยาวหลายนาทีบน DirectML **ต้องใช้ `kv_cache_sliding_window` เล็ก** เป็นค่าเริ่มต้น ไม่ใช่ default ของ upstream (คิดมาสำหรับ CUDA) — ถ้าเล็กเกินไปกระทบคุณภาพ อาจต้องพึ่งวิธีอื่น (ROCm, cloud NVIDIA) สำหรับงานที่ต้องการ context ยาวจริง ๆ

**⚠️ ผลทดลองชุดแรกทั้งหมดปนเปื้อน (confounded):** ระหว่างทดสอบ มี viser server เก่า (demo courthouse ที่เปิดค้างไว้ให้ผู้ใช้ดู ตั้งแต่ 11:17) **ถือ VRAM 7GB ค้างตลอด** — ทุกเทสต์ข้างล่างนี้จึงรันด้วย VRAM ราวครึ่งเดียวของจริง ตัวเลข "ไปได้ถึงเฟรม X" เชื่อไม่ได้ ใช้เทียบแนวโน้มระหว่างกันได้เท่านั้น (ทุกตัวโดนเท่ากัน):
| config | ไปได้ถึง (ภายใต้ VRAM ครึ่งเดียว) | บทสรุปเชิงแนวโน้ม |
|---|---|---|
| kv64 (default), kf1 | เฟรม 3-4 ❌ | default CUDA แย่สุด |
| windowed mode (window 8) | window 2/20 ❌ | windowed ไม่ช่วย — KV path เดียวกันข้างใน |
| kv16, kf1 | **เฟรม 86** ❌ | ดีสุดในกลุ่ม |
| kv16, kf2 | เฟรม 10 ❌ | แย่ลงชัด — เส้นทาง defer+append+rollback churn หนัก → **ห้ามใช้ keyframe_interval>1 บน DirectML** |
| kv8, kf1 | เฟรม 10 ❌ (พังใน `_apply_kv_cache_eviction`) | eviction ถี่ = churn ถี่ ไม่ช่วย |
| ~~308px~~ | — | **ใช้ไม่ได้กับ weights จริง** — pos_embed ผูก 518 เท่านั้น (size mismatch ตอน load) |

**บทเรียนวิธีทดสอบ (สำคัญ):** ก่อน benchmark VRAM ทุกครั้ง ตรวจผู้ถือ GPU memory ค้างก่อน: `(Get-Counter '\GPU Process Memory(*)\Local Usage').CounterSamples` — viser server ที่เปิดให้คนดูค้างไว้ = โมเดลทั้งตัว (5GB+) ค้างบน GPU ตลอด · ปิด server เก่าทุกครั้งก่อนรันเทสต์ใหม่

**ผลทดลองรอบสอง (GPU สะอาด): ✅ kv16, kf1 ผ่านครบ 119/119 เฟรมใน 84 วินาที (~0.71s/frame) + ยืนยันภาพจริงผ่าน ui-use** — เร็วกว่ารอบปนเปื้อน 12 เท่า สรุป: ตัวการหลักคือ VRAM โดนแย่ง ไม่ใช่ fragmentation รุนแรงอย่างที่วิเคราะห์ตอนแรก (fragmentation มีจริงแต่เป็นรอง) · **config มาตรฐาน DirectML สำหรับซีเควนซ์ยาว: `--kv_cache_sliding_window 16 --num_scale_frames 2 --offload_to_cpu --use_sdpa` (keyframe_interval=1 เท่านั้น)**

**สเกลถัดไป — TUM fr1_room 454 เฟรม (ไล่แก้ทีละชั้น, GPU สะอาดตลอด):**
| รอบ | config | ไปได้ถึง | สาเหตุ/แก้ |
|---|---|---|---|
| 1 | 4:3 เดิม (518×392), kv16, auto-kf | เฟรม 19 ❌ | demo.py **auto เปิด kf=2** เมื่อเฟรม>320 → แพตช์ demo.py: backend directml/cpu บังคับ kf=1 เสมอ |
| 2 | 4:3, kv16, kf1 | เฟรม 18 ❌ | ภาพ 4:3 สูงกว่า → token +33% → SDPA ขอ 1.25GB · แก้: **ครอป 16:9** (ตรงกับวิดีโอมือถือจริงอยู่แล้ว — spec เข้าระบบ: 16:9 แนวนอน, 4:3/แนวตั้งให้ auto-crop) |
| 3 | 16:9 (518×294 = geometry เดียวกับ loop), kv16 | เฟรม 23 ❌ | ต่างจาก loop แค่จำนวนเฟรม → จับได้: demo.py ยัด**ภาพทั้งชุดขึ้น GPU** (454 เฟรม ≈ 830MB ค้างถาวร) ทั้งที่ upstream `inference_streaming` รองรับภาพบน CPU อยู่แล้ว (slice+move ทีละเฟรม) → แพตช์ demo.py: backend directml/cpu เก็บภาพบน CPU |
| 4 | 16:9, kv16, CPU-input | **เฟรม 186** ❌ | ดีขึ้น 8 เท่า — ที่เหลือคือ fragmentation สะสมช้าๆ จากหลายร้อย eviction จน spike SDPA (~757MB) หาที่ไม่ได้ |
| 5 | 16:9, kv8, CPU-input | เฟรม 10 ❌ | **kv8 เสียโดยตัวเอง** (eviction แรกที่ window เล็กพังทันที ซ้ำ pattern เดิมแม้ GPU สะอาด) — ห้ามต่ำกว่า kv16 |
| 6 | **chunked runner** (5 ท่อน×120, overlap 10, kv16) | ✅ **454/454 จบ + เห็นห้องจริงทั้งห้อง** | คำตอบสุดท้าย |

### ✅ R0 เสร็จสมบูรณ์ (2026-07-19) — คำตอบ: chunked runner

`lingbot_plus/worker.py` (รัน 1 ท่อนใน process สด เขียน npz) + `lingbot_plus/chunked.py` (วางท่อน+overlap, ยิง worker ทีละท่อน + retry อัตโนมัติเมื่อ OOM สุ่ม, stitch ด้วย **Umeyama similarity transform บนจุดศูนย์กลางกล้องของเฟรมซ้อน**):
- **สเกลต่อท่อนลอยจริง** (วัดได้ 1.17→1.92→2.71→2.30) — ยืนยันว่า stitch แบบมี scale จำเป็น ไม่ใช่แค่ rotation+translation · residual 0.02-0.04
- convention เก็บ extrinsic = w2c (ตาม pipeline demo/viewer ที่พิสูจน์แล้ว) — จุดศูนย์กลางกล้อง C = -RᵀT ห้ามใช้คอลัมน์ t ดิบ · คณิต stitch มี unit test ผ่าน (error 1e-15)
- ท่อนอาจ OOM สุ่มถ้ายิง process ติดกันเร็วเกิน (driver คืน heap ไม่ทัน) — delay 5s ระหว่างท่อน + retry 1 ครั้ง (พัก 15s) แก้ได้
- TUM fr1_room 454 เฟรม (ห้องทำงานจริง ครอป 16:9) = **~10 นาทีรวม เห็นห้องทั้งห้อง+เส้นทางกล้อง ต่อเนียนไม่มีรอยแตก** (ภาพยืนยันผ่าน ui-use แล้ว)
- **นี่คือแกนของ `scan.py` เฟส 2** — วิดีโอยาวไม่จำกัดบนเครื่อง AMD นี้ทำได้แล้วจริง

**แผนสำรองถ้าจูนพารามิเตอร์ไม่พอ (สำหรับวิดีโอยาวจริง 1,000+ เฟรม):** "chunked runner" ใน `lingbot_plus/scan.py` — หั่นวิดีโอเป็นท่อน (~60-80 เฟรม/ท่อน มี overlap) รันแต่ละท่อนใน **process แยก** (DirectML heap ใหม่สด ไม่มี fragmentation สะสมข้ามท่อน) แล้วต่อ point cloud ด้วย pose ของเฟรม overlap — ยอมแลก: เสียเวลาโหลดโมเดลใหม่ ~10 วิ/ท่อน + งานเขียนตัวต่อท่อน (alignment ผ่านเฟรมซ้อน)

## 3. สถาปัตยกรรมรวม

```
┌─ ชั้นผู้ใช้ ─────────────────────────────────────────────┐
│  GUI อัปโหลด (เว็บ local)   LINE OA   ระบบแจ้งซ่อม        │
├─ ชั้นบริการ (เฟส 4) ────────────────────────────────────┤
│  FastAPI job server: รับวิดีโอ → คิว → ประมวลผล → ลิงก์   │
├─ ชั้น pipeline (เฟส 2-3) ───────────────────────────────┤
│  scan: วิดีโอ → เฟรม → inference → GLB + หน้า viewer      │
│  compare: 2 สแกน → หน้าเทียบก่อน-หลัง                     │
├─ ชั้นเครื่องยนต์ (เฟส 0-1) ─────────────────────────────┤
│  lingbot_map (upstream, แก้น้อยสุด) + backend selector    │
└──────────────────────────────────────────────────────────┘
```

**หลักการวางโค้ด:** แก้ upstream ให้น้อยที่สุด (จะได้ merge ของใหม่จากต้นน้ำได้) — ของใหม่ทั้งหมดอยู่ใน package แยก `lingbot_plus/` ในรีโปเดียวกัน

```
d:\cowork\lingbot-map\
├── lingbot_map/          # upstream — แตะเฉพาะที่จำเป็น (ระบุในเฟส 0)
├── demo.py               # upstream — เพิ่ม --backend (แก้ 4 จุดตาม 2.2 ข้อ 4)
├── lingbot_plus/         # ★ ของใหม่ทั้งหมดอยู่นี่
│   ├── device.py         # backend resolver (เฟส 0)
│   ├── presets.py        # fast/balanced/quality (เฟส 1)
│   ├── scan.py           # pipeline สแกนห้อง (เฟส 2)
│   ├── webviewer/        # template หน้า static model-viewer (เฟส 2-3)
│   └── server/           # FastAPI + GUI อัปโหลด (เฟส 1 GUI, เฟส 4 LINE)
├── docs\MASTER_PLAN.md   # ไฟล์นี้
├── requirements-amd.txt  # เฟส 0
├── .env.example          # ทุก config/secret — ห้าม hardcode
└── scans\                # output ต่อสแกน (gitignore)
```

---

## 4. แผนงานรายเฟส

> ทุกเฟส: (ก) เขียนโค้ด (ข) รันจริง (ค) แปะหลักฐาน output จริง (ง) commit ข้อความสื่อความหมาย — **ห้ามบอก "เสร็จ" โดยไม่มี (ข)+(ค)**

### เฟส 0 — เครื่องยนต์ติดบน AMD ⛔ ข้ามไม่ได้
**เป้า:** `python demo.py --backend directml --use_sdpa --model_path ... --image_folder example/courthouse` เปิด viser เห็น point cloud

**งาน:**
1. สร้าง venv Python เวอร์ชันที่ torch-directml ตัวล่าสุดรองรับ (ตรวจจาก pypi ก่อน — อย่าเดา)
2. `requirements-amd.txt`: torch (เวอร์ชันคู่ torch-directml), torch-directml, opencv-python, pillow, tqdm, viser, trimesh, onnxruntime (sky mask) — **ไม่มี** flashinfer, kaolin
3. เขียน `lingbot_plus/device.py`:
   ```python
   def resolve_backend(name: str = "auto") -> BackendContext
   # BackendContext: .device (torch.device), .dtype, .autocast()  (context manager
   #   — เป็น nullcontext เมื่อ backend ไม่รองรับ autocast), .force_sdpa (bool),
   #   .allow_compile (bool), .name, .describe()  (สตริงรายงานผู้ใช้)
   # auto: cuda(จริง) > rocm(=cuda บน ROCm build) > directml > cpu
   # directml: import torch_directml; device = torch_directml.device(); dtype=fp32;
   #   autocast ปิด; force_sdpa=True; allow_compile=False
   # cpu: dtype=fp32; force_sdpa=True; allow_compile=False
   ```
4. แก้ `demo.py` 4 จุด (ตาม 2.2 ข้อ 4) ให้ใช้ BackendContext — เพิ่ม arg `--backend` default `auto` — เมื่อ `force_sdpa=True` ให้บังคับ `args.use_sdpa=True` พร้อม print แจ้ง
5. เช็คขนาด weights บน HF ก่อน แจ้งผู้ใช้ แล้วโหลด (ใช้ `huggingface_hub` หรือ dl-watch ถ้าไฟล์ใหญ่)
6. รันกับ `example/courthouse` — เก็บตัวเลข: วินาที/เฟรม, RAM/VRAM peak

**เกณฑ์ผ่าน:** point cloud ขึ้นใน viser + บันทึกตัวเลขความเร็วลง `docs/BENCHMARKS.md`
**ความเสี่ยง:** torch-directml ไม่รองรับ API ที่โมเดลใช้ (เช่น op ใหม่) → fallback: รัน `--backend cpu` ให้ผ่านก่อน (correctness) แล้วบันทึกว่า directml ติดอะไร ห้าม silent-fail — ตัดสินใจต่อกับผู้ใช้
**ห้ามทำ:** แตะ `demo_render/`, `benchmark/` — นอกสโคป

### เฟส 1 — ชั้นใช้ง่าย — ✅ เสร็จ (2026-07-20)

**ทำแล้ว:** `lingbot_plus/server/` (FastAPI + `index.html` ไทย) — ลากวิดีโอวาง เลือก preset (เร็ว/สมดุล/ละเอียด = fps 3/5/8 + chunk map) + backend, แถบ progress ต่อขั้น + log สด, เสร็จแล้วปุ่ม "เปิดรายงาน" เสิร์ฟจาก `/scans/<ชื่อ>/report.html` · worker เดี่ยว (GPU ทำทีละงานอยู่แล้ว) · แตกเฟรม auto **center-crop 16:9** ตามข้อบังคับ DirectML · ทุกขั้นเรียก module เดิมของ CLI ผ่าน subprocess (ไม่มีโค้ดซ้ำ) · config ผ่าน `.env` (`.env.example` เพิ่มแล้ว)
**รัน:** `.venv-amd\Scripts\python -X utf8 -m uvicorn lingbot_plus.server.app:app --port 8500`
**ทดสอบ end-to-end แล้ว:** อัปโหลดวิดีโอทดสอบ 27 วิ → pipeline วิ่งอัตโนมัติครบ 5 ขั้นโดยไม่แตะอะไร → รายงานเปิดจากลิงก์ใน GUI ได้จริง (ยืนยันภาพผ่าน ui-use)
**หมายเหตุ:** calibrate สเกล (R1) ยังเป็นขั้น manual แยก (ต้อง interactive) — รายงาน auto จะติดป้าย "ยังไม่ calibrate" ตามจริง · preset ตัวเลขยังไม่ได้จูนตาม benchmark จริงจัง (ใช้ค่าตั้งต้นสมเหตุผล)

**สเปกเดิม (อ้างอิง):**
**เป้า:** คนไม่รู้ CLI ใช้ได้

**งาน:**
1. `lingbot_plus/presets.py` — 3 preset map ไป flag ที่มีอยู่แล้ว:
   - `fast`: image_size เล็กลง, `--camera_num_iterations 1`, `--num_scale_frames 2`, stride สูง
   - `balanced`: ค่า default
   - `quality`: image_size เต็ม, keyframe ทุกเฟรม
   (ค่าเป๊ะๆ ให้จูนจากผล benchmark เฟส 0 — อย่า copy ตัวเลขนี้โดยไม่ทดสอบ)
2. GUI อัปโหลด: หน้าเว็บ local (FastAPI + หน้า HTML เดียว) — ลากวิดีโอวาง, เลือก preset+backend (dropdown โชว์ `describe()` ของ backend ที่ใช้ได้จริงบนเครื่อง), แถบ progress (ใช้แนว skill web-progress-bars), เสร็จแล้วลิงก์ไปหน้า viewer
   - เหตุที่เป็นเว็บไม่ใช่ desktop app: โค้ดเดียวกันโตเป็น server เฟส 4 ได้เลย
3. `.env.example`: `LBP_PORT`, `LBP_MODEL_PATH`, `LBP_SCANS_DIR`, `LBP_DEFAULT_BACKEND`

**เกณฑ์ผ่าน:** เปิดเบราว์เซอร์ ลากวิดีโอจริง 1 ไฟล์ กดรัน ได้ผลโดยไม่แตะ CLI

### เฟส 2 — สแกนห้อง ครบวงจรแรก (ธงนำธุรกิจ)
**เป้า:** วิดีโอห้องเข้า → โฟลเดอร์สแกนออก: GLB + หน้าเว็บ static ดูได้บนมือถือ แชร์ได้

**งาน:**
1. `lingbot_plus/scan.py` — pipeline: วิดีโอ → สกัดเฟรม (fps จาก preset) → inference → เรียก `glb_export.py` เดิม → เขียนโฟลเดอร์:
   ```
   scans/2026-07-18_room-1203/
   ├── scan.glb
   ├── index.html      # หน้า viewer static
   ├── meta.json       # วันที่, preset, backend, จำนวนเฟรม, เวลาประมวลผล
   └── thumbs/         # ภาพตัวอย่าง
   ```
2. `lingbot_plus/webviewer/template.html` — ใช้ `<model-viewer>` (web component ของ Google, ไฟล์ JS เดียว, รองรับมือถือ/touch/AR) ฝัง glb — **self-contained: copy model-viewer.min.js ลงโฟลเดอร์ ไม่พึ่ง CDN** (กันลิงก์ตายและใช้ในวงในได้)
   - ตรวจก่อนใช้: point cloud GLB แสดงใน model-viewer ได้ไหม (model-viewer เกิดมาเพื่อ mesh) — ถ้าจุดไม่ขึ้นหรือหนักเกิน มี fallback 2 ทาง: (ก) Three.js viewer เอง (points material — โค้ดไม่ยาว) (ข) แปลงเป็น mesh/splat ก่อน export — ทดสอบกับไฟล์จริงแล้วเลือก บันทึกเหตุผลลง docs
   - ขนาดไฟล์ GLB ต้องดูแล: มี downsample_factor เดิมอยู่แล้ว — เป้าไฟล์ < 50MB ให้มือถือโหลดไหว
3. เชื่อม GUI เฟส 1: งานเสร็จ → ลิงก์หน้า scan
4. (ทางเลือกถ้าผู้ใช้สั่ง) สคริปต์อัปโหลดโฟลเดอร์สแกนขึ้น Cloudflare Pages — pattern เดียวกับ Hybritel — **ห้าม push/deploy โดยไม่ได้รับคำสั่ง**

**เกณฑ์ผ่าน:** ถ่ายห้องจริง 1 ห้องด้วยมือถือ → เปิด `index.html` บนมือถือ หมุน/ซูมได้ลื่น

### เฟส 3 — ก่อน-หลัง (ตรวจรับงาน)
**เป้า:** เลือก 2 สแกน → หน้าเทียบข้างกัน

**งาน:**
1. `compare.html` template: 2 viewer ข้างกัน + ปุ่ม sync camera (หมุนซ้าย ขวาหมุนตาม) + slider overlay ถ้าทำได้
2. GUI: หน้า "สแกนทั้งหมด" (อ่าน meta.json ทุกโฟลเดอร์) → ติ๊กเลือก 2 อัน → สร้างหน้าเทียบ
3. ⚠️ 2 สแกนคนละครั้ง แกน/สเกลไม่ตรงกัน — เฟสนี้เอาแค่ "ดูคู่กัน + จัดมุมเอง" พอ (alignment อัตโนมัติ = backlog อย่าหลงไปทำ)

**เกณฑ์ผ่าน:** สแกนห้องเดิม 2 ครั้ง (ย้ายของ 1 ชิ้น) เปิดหน้าเทียบ เห็นความต่าง

### เฟส 4 — ต่อระบบที่มีอยู่
**เป้า:** ใช้งานจากช่องทางจริงของธุรกิจ

**งาน:**
1. Job queue ใน FastAPI (in-process queue พอ — งานหนัก GPU รันได้ทีละงานอยู่แล้ว): สถานะ pending/processing/done/failed + endpoint เช็คสถานะ
2. LINE OA: ผู้ใช้ส่งวิดีโอเข้า LINE → webhook รับ → เข้าคิว → เสร็จแล้ว reply ลิงก์
   - ดู pattern โปรเจกต์ LINE เดิม (memory: project_line_chatbot, LINE API limits — วิดีโอผ่าน LINE โดนบีบ/จำกัดขนาด ตรวจ limit ก่อนออกแบบ)
   - ⚠️ กฎ: ห้ามยิงข้อความเข้ากลุ่ม/ผู้ใช้จริงตอน dev (memory: no_auto_send_to_group)
3. ระบบแจ้งซ่อม (facility maintenance): เพิ่มช่องแนบลิงก์สแกนในใบแจ้งซ่อม — แตะระบบเดิมให้น้อยสุด (แค่ field ลิงก์ + ปุ่มเปิด)
4. Cloudflare tunnel สำหรับให้คนนอกเห็นลิงก์ — ใช้ named tunnel ที่มีอยู่ (ดู memory: cloudflared traps)

**เกณฑ์ผ่าน:** ส่งวิดีโอผ่าน LINE (บัญชีทดสอบ) → ได้ลิงก์กลับ → เปิดดูบนมือถือได้

## 4.5 Track R — Room Intelligence (จาก 3D ดิบ → เข้าใจห้อง)

> **เป้า:** วิดีโอเดินถ่ายห้องหลายนาที → รายงานอัตโนมัติ: โมเดล 3D + แปลนมุมบน + จำนวนห้อง + ขนาดจริง (เมตร/ตร.ม.) + รายการเฟอร์นิเจอร์พร้อมตำแหน่ง
> **ระดับความแม่น:** ประเมินคร่าว (estimate-grade) ไม่ใช่งานรังวัด — สเกลคลาดได้ ±5-10% ต้องแจ้งผู้ใช้ชัดเจนในรายงานทุกฉบับ
> Track นี้ต่อจากเฟส 0 ได้เลย (ไม่ต้องรอเฟส 1-4 เพราะกิน pipeline คนละท่อน) แต่ละ R เป็น vertical slice จบในตัว

### R0 — วิดีโอยาวจริงบน AMD (ฐานของทุกอย่าง)
**เป้า:** วิดีโอมือถือเดินถ่ายห้องจริง 2-5 นาที → point cloud ทั้งห้อง ไม่ล่ม ไม่เพี้ยน
**งาน:**
1. ถ่ายวิดีโอห้องจริง 1 ห้อง (แนวนอน เดินช้า กวาดทุกมุม) — ผู้ใช้ถ่ายเองหรือใช้วิดีโอทดสอบ
2. รัน `--video_path ... --mode windowed` (โหมดวิดีโอยาวที่ upstream มีแล้ว) บน directml — จูน `--fps` (แนะนำเริ่ม 4-6), `--window_size`, `--num_scale_frames` ให้อยู่ใน VRAM
3. บันทึกจริงลง BENCHMARKS.md: นาทีวิดีโอ → นาทีประมวลผล → จำนวนจุด → คุณภาพ (ดูด้วยตา ผ่าน ui-use)
**เกณฑ์ผ่าน:** point cloud ห้องจริงครบทุกผนัง ดูรู้เรื่องว่าเป็นห้องนั้น + ตัวเลขเวลาจริง
**ความเสี่ยง:** windowed mode ยังไม่เคยรันบน DirectML — อาจเจอ op ใหม่พังแบบ 7 ตัวที่ผ่านมา (แก้ตาม pattern §2.4 ได้)

### R1 — สเกลจริง (เมตร) — ✅ เครื่องมือเสร็จ ใช้งานได้ (2026-07-20)

**ทำแล้ว:** `lingbot_plus/measure.py` — `MeasureViewer` (subclass ของ PointCloudViewer, upstream ไม่ถูกแตะ) แผงวัดอยู่บนสุดของ sidebar: ปุ่มเริ่มวัด → คลิก 2 จุด → หมุดแดง/เขียว + เส้น + ป้ายระยะ → กรอกความยาวจริง → ตั้งสเกล → บันทึก `meta.json` → เปิดใหม่โหลดสเกลอัตโนมัติ → การวัดถัดไปเป็นเมตร — **ทดสอบเดินเต็มเส้นผ่าน ui-use แล้วทุกขั้น** (วัด 2.9096 หน่วย → calibrate 2.00 m → สเกล 0.6874 m/หน่วย persist จริง → วัดใหม่ได้ 3.520 m ระดับห้องสมจริง)

**บทเรียน ray picking (สำคัญต่อคนแก้ต่อ):** (1) nearest-to-ray ทั้งฉาก = จุดเด่นจุดเดียวดูดทุกคลิก (สองคลิกได้จุดเดียวกัน 0.000 m) (2) first-hit เปล่าๆ = โดนฝุ่น noise ใกล้กล้อง — คำตอบที่ใช้: **dense-cluster first-hit** (หากลุ่มจุด ≥12 ตัวแรกตามแนว ray ใน band 1.5% ของ extent) (3) มุมกล้องเริ่มต้นอยู่ในกลาง cloud — แนะนำผู้ใช้กด Overview ก่อนวัด

**ค้างไว้ (polish):** outlier trimming (จุดหลุดไกลทำ Overview ซูมไกลเกิน + หมุดอาจปักบนกลุ่ม noise) — ผูกกับการยก conf_threshold/กรอง cloud ใน R2 · ยังไม่ validate ความแม่นกับของจริงที่รู้ขนาด (calibrate ทดสอบใช้ค่าสมมุติ 2.00 m) — ทำตอนสแกนห้องจริงครั้งแรก

**สเปกเดิม (อ้างอิง):**
**หลักการ:** โมเดลกล้องเดียวให้รูปทรงถูกสัดส่วนแต่ไม่รู้หน่วยจริง → ให้ผู้ใช้วัดของจริง 1 ชิ้นเป็นไม้บรรทัด
**งาน:**
1. เครื่องมือวัดใน viewer: คลิก 2 จุดบน point cloud → ระยะ (หน่วยโมเดล) — เพิ่ม GUI ใน viser (มี click-ray API)
2. Calibrate: ผู้ใช้คลิก 2 จุดบนของที่รู้ความยาวจริง (เช่น ขอบประตู วัดเองด้วยตลับเมตร) + กรอกค่าจริง → ได้ `scale_factor` เก็บลง `meta.json` ของสแกน
3. หลัง calibrate ทุกการวัดแสดงเป็นเมตร · ทางลัดเสริม (opt-in): สมมุติประตูสูงมาตรฐาน 2.00 ม. ให้สเกลอัตโนมัติแบบหยาบ พร้อม label "ประมาณ"
**เกณฑ์ผ่าน:** วัดผนังที่รู้ค่าจริง (ไม่ใช่ตัวที่ใช้ calibrate) แล้วคลาดไม่เกิน ~10%

### R2 — แปลนมุมบน + แยกห้อง + ขนาดห้อง — ✅ เครื่องมือเสร็จ (2026-07-20)

**ทำแล้ว:** `lingbot_plus/floorplan.py` — merged.npz → กรอง conf>3 → Open3D voxel+outlier removal → RANSAC หาพื้น → หมุนพื้นเป็น z=0 → slice ช่วงสูงผนัง → occupancy grid 2D → flood-fill แยกห้อง → `plan.png/svg/json` (จำนวนห้อง, กว้าง×ยาว, ตร.ม., coverage %, เส้นทางกล้อง, ฟอนต์ไทย Tahoma)
**ผล TUM fr1_room (สเกลจริงจาก groundtruth trajectory = 0.7833 m/unit):** 1 ห้อง 3.92×3.88 m ~8.3 ตร.ม. ครอบคลุม 80%
**บทเรียนสำคัญ:**
- **กับดักโต๊ะ:** ฉากเฟอร์เยอะ ระนาบใหญ่สุดมักเป็นผิวโต๊ะไม่ใช่พื้น (TUM: กล้องสูงจาก "พื้น" แค่ 0.35 m) — ใส่เกณฑ์ความสูงกล้อง 0.8-2.2 m + **เตือนชัดๆ ใน output/plan.json เมื่อสงสัยว่าอ้างอิงโต๊ะ** ไม่เงียบ · วิดีโอเดินถ่ายห้องจริง (กล้อง ~1.4 m เห็นพื้นเยอะ) จะไม่ติดกับดักนี้
- **วิธี calibrate สเกลด้วย trajectory:** ความยาวเส้นทางกล้องจริง (จาก GT/การวัด) ÷ ความยาวในหน่วยโมเดล = สเกล — ใช้กับ dataset ที่มี GT ได้เลย ของจริงใช้วิธี R1 (วัดผนัง)
- ฟอนต์ไทย matplotlib ต้องตั้ง `rcParams["font.family"]="Tahoma"` ระดับ global (ตั้งเฉพาะ text ไม่ครอบ legend)
- console Windows พิมพ์ '≈'/ไทยต้อง `-X utf8` (กับดักเก่าที่รู้อยู่แล้ว โดนซ้ำ)
**ค้าง:** ทดสอบ 2 ห้องเชื่อมประตู (เกณฑ์ผ่านเดิม) — รอสแกนจริงหลายห้องจากผู้ใช้ · ผนังที่กล้องไม่เห็นยังเป็นช่องว่างตามหลักซื่อตรง

**สเปกเดิม (อ้างอิง):**
**หลักการ:** point cloud → หาระนาบพื้น (RANSAC) → ฉายจุดแนวผนังลง 2D → ได้ผังกำแพง → ปิดล้อม = ห้อง
**งาน:**
1. ใช้ **Open3D** (CPU ล้วน ไม่แตะ DirectML — ไม่มีความเสี่ยง GPU): `segment_plane` หาพื้น → จัดแกนให้พื้นราบ → slice จุดช่วงความสูงผนัง (เช่น 0.3-2.2 ม. หลังมีสเกลจาก R1) → ฉายลง 2D → line fitting ได้เส้นกำแพง
2. แยกห้อง: flood-fill พื้นที่ปิดล้อมระหว่างกำแพง → นับห้อง + คำนวณ กว้าง×ยาว + พื้นที่ ตร.ม. ต่อห้อง (ใช้สเกล R1)
3. ออกผลเป็น `plan.png` + `plan.svg` (มีเลขขนาดกำกับ) ลงโฟลเดอร์สแกน
**เกณฑ์ผ่าน:** สแกนจริง 2 ห้องติดกัน (มีประตูเชื่อม) → แปลนแยกได้ 2 ห้อง ขนาดคลาดไม่เกิน ~10%
**ความเสี่ยง:** ผนังที่กล้องถ่ายไม่ถึง = รูในแปลน — รายงานต้องโชว์ "ความครบ" (coverage) ไม่เดาส่วนที่มองไม่เห็น

### R3 — เฟอร์นิเจอร์: ตรวจจับ + ระบุตำแหน่ง 3D — ✅ เครื่องมือเสร็จ (2026-07-20)

**ทำแล้ว:** `lingbot_plus/furniture.py` — YOLO (ultralytics yolov8n, CPU — เลือกทางนี้แทน ONNX+DirectML เพราะ onnxruntime ที่ลงไว้เป็นตัว CPU อยู่แล้วและ 91 keyframes ใช้เวลาแค่อึดใจ) → ตำแหน่ง 3D อ่านตรงจาก `world_points` ใต้ bbox (median ของ pixel ที่ conf>2 ในแกนกลาง 1/3 ของกรอบ — ไม่ต้อง unproject เอง) → cluster ต่อ class (รัศมี 0.7 m, ต้องเห็น ≥2 เฟรม กัน fluke) → โยงห้องผ่าน grid ของ R2 (`plan_data.npz` ที่ floorplan เซฟเพิ่ม) → `furniture.json` + `plan_furniture.png` ป้ายไทย
**ผล TUM fr1_room:** 113 detections → **23 ชิ้น** (จอ×6 โน้ตบุ๊ก×5 คีย์บอร์ด×4 หนังสือ×4 เก้าอี้×3 ตู้เย็น×1) — ตรงกับสภาพห้องจริง (ห้องนี้มีตู้เย็นจริง)
**ค้าง (polish):** ของ 9 ชิ้นตกขอบ flood-fill ติดป้าย "นอกพื้นที่ห้องที่ระบุได้" (ของบนโต๊ะ=cell ถูกมองเป็น wall) — ควร assign ห้องที่ใกล้สุดแทน · คลาสไทยเฉพาะ (ตู้เสื้อผ้าบิลท์อิน) ยังไม่มีใน COCO ตาม backlog เดิม

**สเปกเดิม (อ้างอิง):**
**หลักการ:** LingBot-Map ให้ท่ากล้อง+depth ทุกเฟรมอยู่แล้ว → รัน object detection บนภาพ 2D แล้วฉายกลับเข้า 3D ได้เลย (ไม่ต้องเทรนอะไรใหม่)
**งาน:**
1. ตัวตรวจ: **YOLO (ONNX)** รันผ่าน onnxruntime ที่ลงไว้แล้ว — ลอง DmlExecutionProvider (GPU AMD) ก่อน ตกลง CPU ก็ยังไหว (รันเฉพาะ keyframe ไม่ใช่ทุกเฟรม) · คลาส COCO ครอบเฟอร์หลัก: เตียง โซฟา เก้าอี้ โต๊ะ ทีวี ตู้เย็น ไมโครเวฟ ฯลฯ
2. ฉาย 3D: กลาง bbox + depth เฟรมนั้น + extrinsic/intrinsic (มีครบแล้วใน predictions) → พิกัด 3D จริง → cluster รวม detection ของชิ้นเดียวกันจากหลายเฟรม (กันนับซ้ำ) → ได้รายการ: ชนิด, ตำแหน่ง, อยู่ห้องไหน (โยงกับ R2)
3. โชว์: หมุดกำกับใน viewer 3D + ไอคอนบนแปลน 2D + ตารางในรายงาน
**เกณฑ์ผ่าน:** ห้องจริงมีเฟอร์ ≥5 ชิ้น → ตรวจเจอ ≥80% ตำแหน่งบนแปลนถูกห้อง ไม่นับซ้ำเกิน 1 ชิ้น
**ความเสี่ยง:** คลาสภาษาไทย/เฟอร์เฉพาะ (ตู้เสื้อผ้า ชั้นพระ) ไม่อยู่ใน COCO — เริ่มจากคลาสที่มี ค่อยขยาย (open-vocab detector เป็น backlog)

### R4 — รายงานรวมอัตโนมัติ — ✅ เสร็จ (2026-07-20) — Track R ครบทั้งสาย

**ทำแล้ว:** `lingbot_plus/report.py` → `report.html` ไฟล์เดียวจบ: Three.js r128 (vendored ใน `lingbot_plus/webviewer/vendor/` + inline ตามกฎ no-CDN) point cloud หมุน/ซูมได้ 250k จุด + แปลน/แผนที่เฟอร์ฝัง base64 + ตารางห้อง/ขนาด + เฟอร์รายห้อง + แถบเตือน (โต๊ะ/ยังไม่ calibrate) ส่งต่อครบ — เปิดจาก file:// บนมือถือ/คอมได้ ไม่ต้อง server ไม่ต้องเน็ต · TUM = 5MB · ทดสอบ render จริงผ่าน ui-use แล้ว (3D+ตาราง+แปลนขึ้นครบ)

**ลำดับใช้งานเต็มสาย (เครื่องนี้):**
```
.venv-amd\\Scripts\\python -X utf8 -m lingbot_plus.chunked --model_path weights\\lingbot-map.pt --image_folder <เฟรม> --stride 3 --chunk_size 120 --overlap 10 --out_dir scans\\<ชื่อ> --backend directml
.venv-amd\\Scripts\\python -X utf8 -m lingbot_plus.measure --scan_dir scans\\<ชื่อ> --port <ว่าง>   # calibrate สเกล 1 ครั้ง
.venv-amd\\Scripts\\python -X utf8 -m lingbot_plus.floorplan --scan_dir scans\\<ชื่อ>
.venv-amd\\Scripts\\python -X utf8 -m lingbot_plus.furniture --scan_dir scans\\<ชื่อ>
.venv-amd\\Scripts\\python -X utf8 -m lingbot_plus.report --scan_dir scans\\<ชื่อ>
```

**สเปกเดิม (อ้างอิง):**
**เป้า:** 1 คำสั่ง/1 คลิก → `report.html` ในโฟลเดอร์สแกน: viewer 3D (จากเฟส 2) + แปลน 2D + ตาราง "X ห้อง, ขนาดแต่ละห้อง, เฟอร์นิเจอร์รายห้อง" + ระดับความเชื่อมั่น/coverage + คำเตือน estimate-grade
**เกณฑ์ผ่าน:** เดินเส้นทางผู้ใช้เต็ม: ถ่ายวิดีโอ → รันคำสั่งเดียว → เปิดรายงานบนมือถือ อ่านรู้เรื่องโดยไม่ต้องอธิบาย
**ลำดับพึ่งพา:** R4 ต้องมี R1+R2 (ขนาด/แปลน) และดีขึ้นมากถ้ามี R3 (เฟอร์) — เฟส 2 (GLB+หน้าเว็บ) ควรเสร็จก่อนหรือทำคู่กัน

**Dependencies เพิ่มของ Track R:** `open3d` (R2), YOLO ONNX model file ~10-50MB (R3) — ทั้งคู่ CPU-friendly ไม่พึ่ง CUDA · ใส่ใน `requirements-amd.txt` เมื่อถึงเฟสนั้น

## 4.6 Track M — โหมดผลลัพธ์ (Points / Mesh) ในโปรแกรมเดียว

> **โจทย์ผู้ใช้:** อยากได้ทั้ง "วัด/วิเคราะห์" (Track R) และ "โมเดลสวยแบบ KIRI/Luma/Polycam" ในโปรแกรมเดียว เลือกโหมดเอา
> **ความจริงที่ต้องแจ้งชัด:** LingBot-Map สร้าง point cloud (เพื่อ SLAM) ไม่ใช่ photogrammetry mesh — โหมดสวยจึงเป็น post-process ต่อยอด คุณภาพผิวจะ "ดีขึ้นมาก" แต่**ไม่เท่า** เครื่องมือเฉพาะทาง ต้องสื่อสารตามจริงเสมอ

**ดีไซน์:** `scan.py` (เฟส 2) รับ `--output` เลือกได้ ต่อสแกนเดียวออกได้หลายแบบ:

| โหมด | ได้อะไร | วิธี | ต้นทุน |
|---|---|---|---|
| `points` (default) | point cloud GLB — เร็วสุด สำหรับ Track R ทุกตัว | ที่มีอยู่แล้ว | — |
| `mesh` | ผิวต่อเนื่อง + สีจากภาพจริง — ดูเป็น "โมเดลห้อง" ใกล้เคียง Polycam ขั้นต้น | Open3D (CPU): point cloud → estimate normals → **Poisson reconstruction** → ตัดส่วนเบาบาง (density filter) → vertex colors จากภาพต้นฉบับ (มี extrinsic/intrinsic ครบจากโมเดลแล้ว) → GLB | +1-3 นาที/สแกน, CPU ล้วน ไม่เสี่ยง DirectML |
| `both` | สองไฟล์จากสแกนเดียว | รวมสองทาง | — |

**ขั้นทำ (M1):** เพิ่มโมดูล `lingbot_plus/meshing.py` — 1 ฟังก์ชัน `points_to_mesh(pred_dict, out_glb)` ใช้ Open3D · เกณฑ์ผ่าน: สแกนห้องจริง → เปิด mesh GLB ใน model-viewer บนมือถือ ผิวห้องต่อเนื่อง สีถูกต้อง ไม่มีรูใหญ่ผิดปกติในบริเวณที่กล้องถ่ายครบ
**M2 (แปะ texture ละเอียด — ทางเลือก):** UV texture projection จาก keyframes แทน vertex colors → คมขึ้นมาก แต่โค้ดซับซ้อน ทำหลัง M1 นิ่งแล้วเท่านั้น
**M3 (backlog — เกรด Luma จริง):** Gaussian Splatting — เทรนต้อง CUDA เป็นหลัก (บนเครื่องนี้ = รอ ROCm หรือ cloud) · ตัวเลือกเสริมภายนอก: ถ่ายด้วยแอป Polycam/Luma แยกสำหรับงานขายที่ต้องสวยสุด — บันทึกไว้เป็นทางหนี ไม่ใช่แผนหลัก

**ผูกกับของเดิม:** `--output` เป็น option ใน scan.py + dropdown ใน GUI (เฟส 1) + รายงาน R4 ฝังตัว mesh ถ้ามี · Open3D อยู่ใน dependency ของ R2 อยู่แล้ว — ไม่เพิ่มของใหม่

### Backlog (ห้ามทำก่อนเฟส 0-4 เสร็จ)
- ~~วัดระยะบน 3D~~ → ย้ายไป Track R1
- ~~มุมมองแปลน top-down~~ → ย้ายไป Track R2
- ROCm backend บน WSL (ถ้า directml ช้าเกิน — ตัวนี้อาจเลื่อนขึ้นมา)
- Alignment อัตโนมัติสำหรับก่อน-หลัง (ICP)
- Live streaming จากกล้องมือถือ (โมเดลรองรับ streaming แต่ infra ยังไกล)
- เรนเดอร์วิดีโอสวย (demo_render) — ต้อง CUDA, ข้ามจนกว่ามีเครื่อง NVIDIA

---

## 5. กติกาสำหรับโมเดลผู้เขียนโค้ด

1. **อ่านก่อนแก้:** เปิดไฟล์จริงทุกไฟล์ที่จะแตะ — เลขบรรทัดในเอกสารนี้อาจเลื่อนถ้า upstream อัปเดต ยึด pattern ไม่ยึดเลขบรรทัด
2. **Upstream แตะน้อยสุด:** แก้ `demo.py` + จุดที่ระบุเท่านั้น ของใหม่ลง `lingbot_plus/` — ห้าม refactor upstream ตามใจ
3. **ห้ามทำพังทาง CUDA:** ทุกการแก้ต้องไม่เปลี่ยนพฤติกรรมเมื่อ `--backend cuda` (เครื่องอื่นในอนาคต)
4. **Portable:** ไม่ hardcode path/port/secret — `.env` เดียว + `.env.example` ครบ + requirements ครบ (กฎ CLAUDE.md ผู้ใช้)
5. **หลักฐานจริง:** ทุกเฟสจบด้วยการรันจริง แปะ output จริง (ตัวเลขเวลา, screenshot/ลิงก์) — เทสต์พังให้บอกพัง
6. **Git:** commit แยกเฟส ข้อความสื่อความหมาย — **push เฉพาะเมื่อผู้ใช้สั่ง** · ก่อน commit ตรวจ secret เสมอ
7. **ติดขัด:** แก้พลาด 3 ครั้งให้หยุด สรุปว่าติดอะไร เสนอทางเลือก 2-3 ทางพร้อมคำแนะนำ — ห้ามเดามั่ว ห้าม silent fallback
8. **ภาษา:** ตอบผู้ใช้เป็นไทย ศัพท์เทคนิคคงต้นฉบับ — ผู้ใช้เป็นผู้บริหาร เอาภาพรวม+ข้อสรุปตัดสินใจได้ ไม่ลงเทคนิคเกินจำเป็น

## 6. ความเสี่ยงรวม + ทางหนี

| ความเสี่ยง | โอกาส | ทางหนี |
|---|---|---|
| torch-directml เก่าเกิน รันโมเดลไม่ได้ | กลาง | CPU ก่อน (correctness) → ลอง ROCm WSL → รายงานผู้ใช้เลือก |
| ช้าเกินใช้จริง (>15 นาที/ห้อง) | กลาง | preset fast (ลดเฟรม/ขนาด), ROCm, หรือคุยเรื่องเครื่อง NVIDIA มือสอง |
| GLB ใหญ่เกิน มือถือโหลดไม่ไหว | กลาง | เพิ่ม downsample, แบ่ง LOD, บีบ (draco ถ้า toolchain รองรับ) |
| model-viewer ไม่แสดง point cloud สวย | กลาง | Three.js points viewer เขียนเอง (ระบุในเฟส 2 แล้ว) |
| วิดีโอผ่าน LINE โดนบีบจนสแกนไม่ดี | สูง | แจ้งผู้ใช้ถ่ายแนวนอน+ช้าๆ, รับไฟล์ทางอัปโหลดเว็บเป็นทางหลัก, LINE เป็นทางสะดวก |
| upstream อัปเดตชนกับ patch เรา | ต่ำ | ของใหม่แยก `lingbot_plus/`, patch upstream บางที่สุด |

## 7. สิ่งที่ระบบเดิมมีแล้ว — ห้ามเขียนซ้ำ
GLB export (`vis/glb_export.py`) · sky segmentation + cache (`vis/sky_segmentation.py`) · viser viewer + GUI ปุ่มครบ (`vis/point_cloud_viewer.py`) · โหมดประหยัด VRAM (`--offload_to_cpu`, `--num_scale_frames`) · keyframe auto-select · windowed mode สำหรับวิดีโอยาว · `--use_sdpa` ทางหนี FlashInfer
