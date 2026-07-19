# Backend Benchmarks — เครื่อง niti (AMD RX 9070 XT, RDNA4)

> บันทึกผลรันจริงต่อ backend — กรอกหลังรัน `demo.py` แต่ละครั้ง ห้ามกรอกเลขที่ไม่ได้รันจริง

| วันที่ | backend | dtype | input | frames | เวลาโหลดโมเดล | เวลา inference | s/frame | RAM/VRAM peak | หมายเหตุ |
|---|---|---|---|---|---|---|---|---|---|
| 2026-07-19 | directml | float32 | example/courthouse (518x294) | 3 | 9.3s | 3.8s | ~1.27s | ไม่ได้วัด (WMI รายงาน VRAM ผิด) | num_scale_frames=2, --offload_to_cpu, --use_sdpa (บังคับ) — weights จริง lingbot-map.pt, 0 missing/unexpected keys. depth conf percentiles [10/50/90]=[1.02/5.70/12.98] (ค่าสมเหตุสมผล ไม่มี NaN) point count/frame: 13960/13804/8828 (ยืนยันด้วย debug print ตรง server log) |
| 2026-07-19 | directml | float32 (fake random init) | 6 synthetic frames, 518x518 | 6 | — | FAILED | — | — | num_scale_frames=4 พังด้วย "not enough video memory" ขอ ~1.94GB ก้อนเดียว — DirectML allocator ไม่มี caching เหมือน CUDA ต้องลด num_scale_frames/image_size หรือ offload_to_cpu เมื่อรัน batch ใหญ่ |
| 2026-07-19 | directml | float32 | example/loop (ออฟฟิศในอาคาร, 518x294) | **119** | 10.6s | **84s** | **~0.71s** | โมเดลกิน ~5GB บน GPU | ✅ จบครบ+เห็นภาพจริง — config: `--kv_cache_sliding_window 16 --num_scale_frames 2 --offload_to_cpu --use_sdpa` (kf=1) · **GPU ต้องสะอาด**: รอบที่มี viser เก่าถือ 7GB ค้าง ช้าลง 12 เท่า (8.5s/frame) และ OOM ที่เฟรม 86 · keyframe_interval>1 ห้ามใช้ (churn หนัก พังเร็วกว่า) |
| 2026-07-19 | directml | float32 | **TUM fr1_room ครอป 16:9 (518x294) — chunked runner** | **454** | ~10s×5 ท่อน | **~10 นาทีรวม** (~1.2-1.9 it/s ต่อท่อน) | ~1.3s/frame รวม overhead | fresh heap ทุกท่อน | ✅ **ห้องจริงทั้งห้อง จบครบ+เห็นภาพจริง** — `lingbot_plus/chunked.py`: 5 ท่อน×120 เฟรม overlap 10, Umeyama stitch (สเกลต่อท่อนลอยจริง 1.17→2.71, residual 0.02-0.04) · ท่อนอาจ OOM สุ่มถ้ายิงติดกันเกิน — retry อัตโนมัติแก้แล้ว · single-process เพดาน ~150-186 เฟรม → chunked = วิดีโอยาวไม่จำกัด |
