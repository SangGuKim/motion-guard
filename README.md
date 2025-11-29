# Motion Guard - Webcam Motion Detection Alert System

A simple, effective webcam-based motion detection system designed to prevent unsupervised movement of patients at risk of falls. Detects in real-time when a patient attempts to get out of bed without supervision.

## Features

- 🎥 **Real-time motion detection** using standard webcam
- 🔊 **Audible alerts** with continuous beeping (customizable duration)
- 🌙 **Low-light optimized** for nighttime monitoring
- ⏸️ **Toggle alerts** with spacebar (auto-resumes after set time)
- 🎛️ **Adjustable sensitivity** via command-line arguments
- 🔄 **Hysteresis filtering** to reduce false alarms
- 🪶 **Lightweight** - runs on any laptop with a webcam

## Quick Start

### Installation

```bash
pip install opencv-python numpy pygame
```

Or using requirements.txt:

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run with default settings (sensitivity: 5, min area: 100, alarm: 3s, auto-resume: 5min)
python motion_detector.py

# Run with your preferred settings
python motion_detector.py -s 5 -m 100 -d 3 -r 300
```

### Controls

- **Spacebar**: Toggle alarm on/off (automatically re-enables after set time)
- **Q**: Quit

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-s, --sensitivity` | Sensitivity threshold (lower = more sensitive) | 5 |
| `-m, --min-area` | Minimum motion area in pixels | 100 |
| `-d, --duration` | Alarm duration in seconds | 3.0 |
| `-r, --auto-resume` | Auto re-enable time in seconds | 300 (5 min) |

### Examples

```bash
# Higher sensitivity for small movements
python motion_detector.py -s 3

# Less sensitive to reduce false alarms
python motion_detector.py -s 10 -m 200

# Longer alarm duration
python motion_detector.py -d 5

# Auto-resume after 10 minutes
python motion_detector.py -r 600

# Combine all options
python motion_detector.py -s 5 -m 100 -d 3 -r 300
```

## How It Works

### Motion Detection
- Captures video frames from webcam
- Converts to grayscale and applies Gaussian blur
- Compares consecutive frames to detect changes
- Applies threshold and contour detection
- Uses **hysteresis** (3-frame confirmation) to filter noise

### Alarm Behavior
- Beeps continuously: 0.5s beep + 0.5s pause = 1s interval
- Duration resets whenever new motion is detected
- Can be temporarily disabled with spacebar
- Automatically re-enables after configured time

### Visual Feedback
- Red rectangles around detected motion
- Status displayed on screen:
  - `Monitoring...` - Active monitoring (green)
  - `Motion...` - Motion detected (orange)
  - `ALARM ACTIVE!` - Alarm sounding (red)
  - `ALARM DISABLED (XXs)` - Temporarily disabled (gray)

## Use Cases

- **Fall prevention** for elderly or recovering patients
- **Bedside monitoring** during nighttime
- **Movement alerts** for patients who shouldn't move unsupervised
- **General motion detection** for any monitoring scenario

## Setup Tips

### Basic Setup
1. **Position laptop** 1-2 meters from bed with clear view
2. **Power settings** - disable screen sleep and keep laptop plugged in
3. **Volume** - set system volume high or use external speakers
4. **Test sensitivity** - adjust `-s` value until it reliably detects movement without false alarms

### Low-Light Operation (Critical for Nighttime Use)

**Important**: Standard webcams cannot distinguish human silhouettes in low-light conditions. A small nightlight alone is not enough for direct motion detection. Instead, use the **light-blocking detection method**.

#### Light-Blocking Detection Method (Field-Tested)

**Core Principle**: The camera doesn't observe the person—it **observes the nightlight itself**. When the patient sits up, they block the line of sight between the camera and the light, and the system detects the sudden brightness drop.

**Actual Setup Example:**
```
[Bed Headboard]
    ↓ 
  [Nightlight] ← Camera watches this light
    ↓
[Patient Bed] ← Patient blocks light when rising
    ↓
[Nightstand + Laptop (Camera)]
```

**Step-by-Step Setup:**

1. **Light Placement**: Install small nightlight above bed headboard
   - Use LED nightlight, small mood lamp, etc.
   - Position: Top of headboard or wall above bed

2. **Camera Position**: Place laptop on nightstand at foot of bed
   - Adjust so camera faces the nightlight directly
   - **Only the light should be visible on screen** (not seeing person is normal)

3. **Camera Height Adjustment**: 
   - Place books or stand under laptop to adjust height
   - When patient lies with knees up: light should still be visible
   - When patient sits up: light should be completely blocked

4. **Power Management**:
   - Disable sleep mode when plugged in
   - Set screen to turn off after 1 minute (saves battery, reduces glare)
   - Windows: Settings > System > Power > Screen and sleep
   - macOS: System Preferences > Battery > Power Adapter

**If Space is Limited:**
- If no room at foot of bed, place laptop diagonally beside bed
- Key requirement: "Camera - Patient - Nightlight" must be in line
- Ensure patient blocks light when sitting up

**Using Multiple Light Sources:**
- If you have multiple nightlights, use them all
- Different light positions enable more comprehensive detection
- Example: Lights on both sides of headboard → detect movement in any direction
- Example: Headboard + side table lights → wider detection coverage
- Position all lights to be visible in camera view
- **Advantage**: Reliable detection even if patient rises at an angle
- **Disadvantage**: Multiple lights may disturb sleep (varies by person)

#### Detection Principle & Hysteresis

**Why This Works:**
- In low light, webcams can still detect bright point sources (nightlights)
- When patient sits up, brightness suddenly decreases/disappears
- System detects this brightness change as motion

**Hysteresis Processing:**
- Alert triggers only after 3+ consecutive frames detect motion
- **Once alert starts, it continues for set duration (default 3s)**
- Even if light is blocked only briefly, alarm will sound for full duration
- This ensures rapid movements are not missed

**Recommended Settings (Light-Blocking Method):**
```bash
# Sensitive to brightness changes
python motion_detector.py -s 5 -m 100 -d 3
```

- Sensitivity `-s 5~7`: Suitable for brightness change detection
- Min area `-m 50~150`: Adjust based on light spot size
- Duration `-d 3`: Ensures alarm sounds long enough even for brief blocking

#### Post-Installation Testing (Required)

1. **Run Program**
   ```bash
   python motion_detector.py -s 5 -m 100
   ```

2. **Check Display**: 
   - Verify nightlight appears as bright spot(s) on screen
   - If using multiple lights, ensure all are visible
   - **Not seeing person is normal** (only light needs to be visible)
   - Single light source is sufficient for operation

3. **Detection Test**:
   - Lying down: No detection (light visible)
   - Knees up position: No detection (light still visible)
   - Sitting up position: **Detection** (light blocked)
   - Getting out of bed: **Definite detection**

4. **Adjustment If Needed**:
   - Not detecting: Lower camera or increase sensitivity (`-s 3`)
   - Too many false alarms: Raise camera or decrease sensitivity (`-s 7`)

#### Nighttime Operation Checklist

**Installation Phase:**
- [ ] Install and turn on nightlight(s) at bed headboard
- [ ] Place laptop on nightstand at foot of bed
- [ ] Adjust camera height (use books/stand)
- [ ] Verify nightlight(s) visible on screen

**Configuration Phase:**
- [ ] Connect laptop to power
- [ ] Disable sleep mode
- [ ] Set screen auto-off (optional)
- [ ] Check volume level (or connect external speaker)

**Testing Phase:**
- [ ] Verify no detection when lying down
- [ ] Verify detection when sitting up
- [ ] Test spacebar alarm toggle
- [ ] Confirm auto-resume time

#### Advantages & Limitations of This Method

**Advantages:**
- ✅ Works in complete darkness
- ✅ Low-cost webcam sufficient
- ✅ Only needs nightlight(s)
- ✅ Responds only to sitting-up motion

**Limitations:**
- ❌ Cannot detect tossing/turning in bed
- ❌ Cannot detect subtle posture changes (e.g., tube kinking)
- ❌ Camera-patient-light alignment is critical

**If Different Monitoring Needed:**
- To detect posture changes or tossing/turning: **Brighter lighting required**
- Use mood lamp/desk lamp to make patient's silhouette visible
- Direct motion detection possible, but low-light performance still limited

**Core Principle:**
> This system specializes in **"bed exit detection"**. The goal is to detect the moment a patient attempts to get out of bed to prevent falls. For detailed posture monitoring, professional medical equipment is recommended.

## Troubleshooting

### Alarm sounds too often
```bash
# Reduce sensitivity (higher threshold)
python motion_detector.py -s 10
# or
python motion_detector.py -s 15
```

### Missing movements
```bash
# Increase sensitivity (lower threshold)
python motion_detector.py -s 3
```

### Alarm too short
```bash
# Extend alarm duration
python motion_detector.py -d 5
```

### No sound
- Check if pygame installed correctly: `pip install pygame`
- Verify system volume is not muted
- If pygame unavailable, system beep (`\a`) will be used instead

## Technical Requirements

- Python 3.6+
- Webcam (built-in or USB)
- OpenCV
- NumPy
- Pygame (optional, falls back to system beep)

## Privacy & Ethics

This tool is designed for legitimate caregiving purposes. Please:
- ✅ Use only with informed consent of the person being monitored
- ✅ Ensure proper privacy and dignity of patients
- ✅ Use as a supplementary tool, not a replacement for proper medical care
- ❌ Do not use for unauthorized surveillance

## License

MIT License - Feel free to use and modify for your needs.

## Contributing

Contributions welcome! This project was created in an emergency situation and there's always room for improvement:
- Better motion detection algorithms
- Mobile app notifications
- Multiple camera support
- Cloud recording options
- ML-based fall detection

## Acknowledgments

Created with urgency and care for family safety. If this helps even one other caregiver sleep a little easier, it's served its purpose.

---

**Disclaimer**: This is a basic monitoring tool and should not replace professional medical equipment or proper patient supervision. Always consult with healthcare professionals for proper patient care.