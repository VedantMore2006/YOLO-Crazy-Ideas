# 🧠 What Counts as "Suspicious" (Demo Version)

We need behaviors that:
- are visually demonstrable
- detectable via bounding boxes
- rule-engine compatible
- reliable in 15 FPS CPU mode

Here are high-confidence suspicious patterns:

## 1️⃣ Loitering Detection

If person stays inside ROI > X seconds → suspicious

Very easy. Very powerful demo.

## 2️⃣ Abandoned Object

If:
- Bag detected
- Person moves away
- Distance between bag & person increases
- Timer starts

After 5–10 seconds → FLAG

This is exhibition gold.

## 3️⃣ Rapid Entry-Exit (Nervous Movement)

Track person ID:
If:
- Person enters ROI
- Leaves quickly
- Re-enters repeatedly

Flag as unusual movement.

## 4️⃣ Face Concealment (Advanced – optional)

If:
- Person detected
- No face detected (MediaPipe)

Possible concealment flag.

CPU risky. Might skip for v1.

## 5️⃣ Phone-in-Hand Near Face for Too Long

Detect:
- person
- cell phone

If phone overlaps upper region of person box for > X sec → suspicious distraction.

---

## 🧱 SYSTEM ARCHITECTURE

```
YOLOv8n → Detection
    ↓
ByteTrack → Assign unique ID
    ↓
Behavior Engine (rules + timers)
    ↓
Suspicion Score per ID
    ↓
Overlay Dashboard (Red/Green)
```

---

## ⚙️ Optimization Strategy (CPU 15 FPS Target)

You're using YOLOv8n — good.

We will:
- Use `model.fuse()`
- Set image size = 640 or even 480
- Use half precision? (Not on CPU)
- Set `conf=0.4` to reduce boxes
- Use tracking built-in from Ultralytics (simpler than full ByteTrack)

Limit classes to:
- person
- backpack
- handbag
- suitcase
- cell phone

Less classes = faster inference.

---

## 🎬 Exhibition Mode UI

On screen:

### Top Left:
```
FPS: 17
Active Persons: 2
Suspicious IDs: 1
```

### Bounding Boxes:
- **Green** = normal
- **Yellow** = under observation
- **Red** = suspicious

### Side Panel:
Show:
```
ID 1 → Loitering (7 sec)
ID 3 → Abandoned Object (Timer: 5 sec)
```

This looks professional.

---

## 🕒 8–10 Hour Build Plan

### Phase 1 – Detection + Tracking (2 hrs)
- YOLOv8n
- Track=True
- Unique IDs

### Phase 2 – Timer Engine (2 hrs)
Dictionary:
```python
person_id → first_seen_time
bag_id → owner_id
```

### Phase 3 – Suspicious Logic Rules (2 hrs)
Implement:
- Loitering
- Abandoned object
- Phone near face

### Phase 4 – Suspicion Score System (1 hr)
Each rule adds score:
- +2 loitering
- +3 abandoned bag
- +1 phone distraction

Threshold → suspicious

### Phase 5 – UI Polish + FPS tuning (2 hrs)
- Resize frame
- Reduce draw overhead
- Tune thresholds
